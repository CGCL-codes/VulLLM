import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, inputs_embeds=None, input_ids=None, position_idx=None, attn_mask=None, labels=None):
        bs, l = input_ids.size()

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        if input_ids is not None:
            inputs_embeds = self.encoder.roberta.embeddings.word_embeddings(input_ids)
        else:
            inputs_embeds = inputs_embeds

        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeds)
        inputs_embeds = inputs_embeds * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        outputs = self.encoder.roberta(inputs_embeds=inputs_embeds, attention_mask=attn_mask, position_ids=position_idx,
                                       token_type_ids=position_idx.eq(-1).long())[0]
        logits = self.classifier(outputs)
        # shape: [batch_size, num_classes]
        prob = F.softmax(logits, dim=-1)
        input_size = input_ids.size() if input_ids is not None else inputs_embeds.size()
        assert prob.size(0) == input_size[0], (prob.size(), input_size)
        assert prob.size(1) == 2, prob.size()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)
        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs_ids = batch[0].to("cuda")
            position_idx = batch[1].to("cuda")
            attn_mask = batch[2].to("cuda")
            label = batch[3].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(None, inputs_ids, position_idx, attn_mask, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)

        probs = [[prob[0], 1 - prob[0]] for prob in logits]
        pred_labels = [0 if label else 1 for label in logits[:, 0] > 0.5]

        return probs, pred_labels
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import SequentialSampler, DataLoader

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x

    
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()

        self.convs = Conv1d(embedding_dim, n_filters, filter_sizes)
        self.fc = Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch size, sent len, emb dim]
        embedded = x.permute(0, 2, 1)  # [B, L, D] -> [B, D, L]
        conved = self.convs(embedded)

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))    # [B, n_filters * len(filter_sizes)]
        return self.fc(cat)

class CNNClassificationSeq(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.d_size = self.args.d_size
        self.dense = nn.Linear(2*self.d_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.W_w = nn.Parameter(torch.Tensor(2*config.hidden_size, 2*config.hidden_size))
        self.u_w = nn.Parameter(torch.Tensor(2*config.hidden_size, 1))
        self.linear = nn.Linear(self.args.filter_size*config.hidden_size, self.d_size)
        self.rnn = nn.LSTM(config.hidden_size, config.hidden_size, 3, bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)

        # CNN
        self.window_size = self.args.cnn_size
        self.filter_size = []
        for i in range(self.args.filter_size):
            i = i+1
            self.filter_size.append(i)

        self.cnn = TextCNN(config.hidden_size, self.window_size, self.filter_size, self.d_size, 0.2)
        
        self.linear_mlp = nn.Linear(6*config.hidden_size, self.d_size)
        self.linear_multi = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, features, **kwargs):
        # ------------- cnn -------------------------
        x = torch.unsqueeze(features, dim=1) # [B, L*D] -> [B, 1, D*L]
        x = x.reshape(x.shape[0], -1, 768)     # [B, L, D]
        outputs = self.cnn(x)                  # [B, D]
        #features = self.linear_mlp(features)       # [B, L*D] -> [B, D]
        features = self.linear(features)
        x = torch.cat((outputs, features), dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        # ------------- cnn ----------------------

        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):   
    def __init__(self, encoder, config, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.args = args
        self.linear = nn.Linear(3, 1)        # 3->5
        self.query = 0
        self.cnnclassifier = CNNClassificationSeq(config, self.args)

    def forward(self, seq_ids=None, input_ids=None, labels=None):       
        batch_size = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        token_len = seq_ids.shape[-1]

        seq_inputs = seq_ids.reshape(-1, token_len)                                 # [4, 3, 400] -> [4*3, 400]
        seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0]    # [4*3, 400] -> [4*3, 400, 768]
        seq_embeds = seq_embeds[:, 0, :]                                           # [4*3, 400, 768] -> [4*3, 768]
        outputs_seq = seq_embeds.reshape(batch_size, -1)                           # [4*3, 768] -> [4, 3*768]

        logits_path = self.cnnclassifier(outputs_seq)

        prob_path = torch.sigmoid(logits_path)
        prob = prob_path
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0]+1e-10)*labels+torch.log((1-prob)[:, 0]+1e-10)*(1-labels)
            loss = -loss.mean()
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
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            seq_inputs = batch[2].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(seq_inputs, inputs, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)

        probs = [[1 - prob[0], prob[0]] for prob in logits]
        pred_labels = [1 if label else 0 for label in logits[:, 0] > 0.5]

        return probs, pred_labels
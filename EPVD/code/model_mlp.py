# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.autograd as autograd


class MlpClassificationSeq(nn.Module):
    """classification tasks by the MLP."""
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.d_size = self.args.d_size
        
        self.linear = nn.Linear(self.args.filter_size*config.hidden_size, self.d_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(self.d_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        # ------------- mlp -------------------------
        x = self.linear(features)
        x = self.dropout(x)
        x = self.dense(x)
        # ------------- mlp ----------------------
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.linear = nn.Linear(3, 1)        # 3->5
        
        self.mlp_classifier = MlpClassificationSeq(config, self.args)

    def forward(self, seq_ids=None, input_ids=None, labels=None):
        batch_size = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        token_len = seq_ids.shape[-1]

        seq_inputs = seq_ids.reshape(-1, token_len)                                 # [4, 3, 400] -> [4*3, 400]
        seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0]    # [4*3, 400] -> [4*3, 400, 768]
        seq_embeds = seq_embeds[:, 0, :]                                           # [4*3, 400, 768] -> [4*3, 768]
        outputs_seq = seq_embeds.reshape(batch_size, -1)                           # [4*3, 768] -> [4, 3*768]

        logits_mlp = self.mlp_classifier(outputs_seq)

        prob = torch.sigmoid(logits_mlp)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0]+1e-10)*labels+torch.log((1-prob)[:, 0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

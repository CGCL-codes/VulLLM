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


class LSTMClassificationSeq(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.linear_mlp = nn.Linear(3*config.hidden_size, config.hidden_size)
        self.rnn = nn.LSTM(config.hidden_size, config.hidden_size, 3, bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)

    def forward(self, features, **kwargs):
        x = torch.unsqueeze(features, dim=1) # [B, L*768] -> [B, 1, L*768]
        x = x.reshape(x.shape[0], -1, 768)
        outputs, hidden = self.rnn(x)   # [10, 3, 2*768] []
        
        x = outputs[:, -1, :]   # [B, L, 2*D] -> [B, 2*D]
        x = self.linear(x)  # [B, 2*D] -> [B, D]
        x_ori = self.linear_mlp(features)
        x = torch.cat((x, x_ori), dim=-1)
        
        x = self.dropout(x)
        x = self.dense(x)
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

        self.lstmclassifier = LSTMClassificationSeq(config, self.args)

    def forward(self, seq_ids=None, input_ids=None, labels=None):
        batch_size = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        token_len = seq_ids.shape[-1]

        seq_inputs = seq_ids.reshape(-1, token_len)                                 # [4, 3, 400] -> [4*3, 400]
        seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0]    # [4*3, 400] -> [4*3, 400, 768]
        seq_embeds = seq_embeds[:, 0, :]                                           # [4*3, 400, 768] -> [4*3, 768]
        outputs_seq = seq_embeds.reshape(batch_size, -1)                           # [4*3, 768] -> [4, 3*768]

        logits_path = self.lstmclassifier(outputs_seq)
        prob_path = torch.sigmoid(logits_path)
        prob = prob_path
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0]+1e-10)*labels+torch.log((1-prob)[:, 0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

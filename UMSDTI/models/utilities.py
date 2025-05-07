import pdb

from sklearn.metrics import mean_squared_error
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLayerPooling(nn.Module):
    def __init__(self, layers=12):
        super(WeightedLayerPooling, self).__init__()
        self.layers = layers
        self.layer_weights = nn.Parameter(
            torch.tensor([1] * layers, dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_hidden_states = torch.stack(all_hidden_states, dim=0)
        all_layer_embedding = all_hidden_states[-self.layers:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class Multisample_Dropout(nn.Module):
    def __init__(self):
        super(Multisample_Dropout, self).__init__()
        self.dropout = nn.Dropout(.1)
        self.dropouts = nn.ModuleList([nn.Dropout((i + 1) * .1) for i in range(5)])

    def forward(self, x, module):
        x = self.dropout(x)
        return torch.mean(torch.stack([module(dropout(x)) for dropout in self.dropouts], dim=0), dim=0)


class Bi_RNN(nn.Module):
    def __init__(self, size, hidden_size, layers=1):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(size, hidden_size, num_layers=layers, bidirectional=True, bias=False, batch_first=True)

    def forward(self, x):
        x, hidden = self.rnn(x)
        return torch.cat((x[:, -1, :self.hidden_size], x[:, 0, self.hidden_size:]), dim=-1)


class Bi_RNN_FOUT(nn.Module):
    def __init__(self, size, hidden_size, layers=1):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(size, hidden_size, num_layers=layers, bidirectional=True, bias=False, batch_first=True)

    def forward(self, x):
        x, hidden = self.rnn(x)
        return x


def get_groups(param_groups, group_name):
    groups = sorted(set([param_g[group_name] for param_g in param_groups]))
    groups = ["{:2e}".format(group) for group in groups]
    return groups


def process_outputs(out):
    n_layers = int(len(out) / 2)
    out = torch.stack(out[-n_layers:], dim=0)
    out_mean = torch.mean(out, dim=0)
    out_max, _ = torch.max(out, dim=0)
    out_std = torch.std(out, dim=0)
    last_hidden_states = torch.cat((out_mean, out_max, out_std), dim=-1)
    return last_hidden_states


class Attention(nn.Module):
    def __init__(self, in_dim, hidden_dim, pool_dim=1):
        super().__init__()
        self.pool_dim = pool_dim
        self.attention = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, 1),
                                       nn.Softmax(dim=self.pool_dim))

    def forward(self, x):
        weights = self.attention(x)
        context = torch.sum((weights * x), dim=self.pool_dim)
        return context


class MeanPooling(nn.Module):
    def __init__(self, clamp_min=1e-9):
        super(MeanPooling, self).__init__()
        self.clamp_min = clamp_min

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=self.clamp_min)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class LSTMPooling(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers=12, is_lstm=True):
        super(LSTMPooling, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.bidirectional = False

        self.is_lstm = is_lstm

        if self.is_lstm:
            self.lstm = nn.LSTM(self.hidden_size,
                                self.hidden_size,
                                bidirectional=self.bidirectional,
                                batch_first=True)
        else:
            self.lstm = nn.GRU(self.hidden_size,
                               self.hidden_size,
                               bidirectional=self.bidirectional,
                               batch_first=True)

        self.pooling = MeanPooling(.0)

    def forward(self, all_hidden_states, mask):

        hidden_states = torch.stack([
            self.pooling(layer_i, mask) for layer_i in all_hidden_states[-self.num_hidden_layers:]
        ], dim=1)
        out, _ = self.lstm(hidden_states)
        out = out[:, -1, :]
        return out

if __name__ == '__main__':
    e = torch.rand((32, 45, 256))
    mask = torch.ones((32, 45))
    x = [e, e, e]

    model = LSTMPooling(hidden_size=256, num_hidden_layers=3)
    y = model(x, mask)
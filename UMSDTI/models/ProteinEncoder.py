# -*- coding: utf-8 -*-
import pdb

import math
from torch import Tensor
import torch.nn.init as init
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from transformers import EsmModel, EsmTokenizer


class SEAttention(nn.Module):
    def __init__(self, input_dim, reduction_ratio=16):
        super(SEAttention, self).__init__()
        self.squeeze = nn.Linear(input_dim, input_dim // reduction_ratio)
        self.excitation = nn.Linear(input_dim // reduction_ratio, input_dim)
        self.attention_polling = AttentionPooling(input_dim)

    def forward(self, x, mask):
        # batch_size, seq_len, input_dim = x.size()
        # squeeze_output = torch.mean(x, dim=1)  # Global average pooling
        squeeze_output = self.attention_polling(x, mask)
        squeeze_output = F.relu(self.squeeze(squeeze_output))
        excitation_output = torch.sigmoid(self.excitation(squeeze_output)).unsqueeze(1)
        return x * excitation_output


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask=None):
        w = self.attention(last_hidden_state).float()  # (B, T, 1)

        if attention_mask is not None:
            w[attention_mask == 0] = float('-inf')

        w = torch.softmax(w, dim=1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, conv_nums, kernel_size, se_ratio=16):
        super(ProteinEncoder, self).__init__()
        self.dim = hidden_dim
        self.conv = conv_nums
        self.kernel = kernel_size

        self.protein_embedding = nn.Embedding(vocab_size, self.dim, padding_idx=0)

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=self.kernel[0], padding='same'),
            nn.BatchNorm1d(num_features=self.dim),
            nn.LeakyReLU()
        )

        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(
                in_channels=self.dim, out_channels=self.conv, kernel_size=self.kernel[0], padding='same'
            ),
            nn.BatchNorm1d(num_features=self.conv),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.kernel[1], padding='same'
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=self.conv * 2),
            nn.Conv1d(
                in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.kernel[2], padding='same',
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=self.conv * 4),
            nn.Conv1d(
                in_channels=self.conv * 4, out_channels=self.conv * 2, kernel_size=self.kernel[3], padding='same',
                groups=4
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=self.conv * 2),
            nn.Conv1d(
                in_channels=self.conv * 2, out_channels=self.dim, kernel_size=self.kernel[4], padding='same',
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=self.dim)
        )
        self.enhance = SEAttention(self.dim, reduction_ratio=se_ratio)
        self.fc = nn.Linear(self.dim, self.dim)

    def forward(self, protein, protein_mask):
        protein_embed = self.protein_embedding(protein)
        protein_embed = self.Protein_CNNs(protein_embed.permute(0, 2, 1)).permute(0, 2, 1) + \
                        self.conv1d(protein_embed.permute(0, 2, 1)).permute(0, 2, 1)
        protein_embed = self.enhance(protein_embed, protein_mask)
        protein_embed = self.fc(protein_embed)

        return protein_embed


def swiglu(x, dim):
    x = torch.chunk(x, 2, dim=dim)
    return F.silu(x[0]) * x[1]


class Encoder(nn.Module):
    """protein feature extraction."""

    def __init__(self, args, vocab_size, hid_dim, n_layers, kernel_size, dropout):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers

        self.protein_embedding = nn.Embedding(vocab_size, self.hid_dim, padding_idx=0)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(args.device)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in
             range(self.n_layers)])  # convolutional layers
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(2 * hid_dim) for _ in range(self.n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hid_dim, self.hid_dim)
        # self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein, mask=None):
        conv_input = self.protein_embedding(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))  # [bs, dim * 2, seq_len]
            conved = self.bns[i](conved)
            # conved = swiglu(conved, dim=1)
            conved = F.glu(conved, dim=1)  # [bs, dim, seq_len]
            # apply ressidual connection / high way
            conved = (conved + conv_input) * self.scale
            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.fc(conved)
        return conved


class ContactPredictor(nn.Module):
    """get protein contact map"""
    def __init__(self, args):
        super().__init__()
        self.model = EsmModel.from_pretrained(args.esm_model_path)
    
    def forward(self, token_ids, attention_mask):
        return self.model.predict_contacts(token_ids, attention_mask)

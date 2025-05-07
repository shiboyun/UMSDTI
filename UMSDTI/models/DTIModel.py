# -*- coding: utf-8 -*-
import pdb

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import EsmConfig
from torch_geometric.data import Data, Batch

from models.MPNN import MPN
from models.ProteinEncoder import ProteinEncoder, AttentionPooling, Encoder, ContactPredictor
from models.Trm import ProbAttention, ProbMultiHeadAttention
from models.SmilesCNN import SmilesEncoder
from models.MHA import MultiHeadAttention
from models.GNN import GNNEncoder


class CrossSeNetAttention(nn.Module):
    def __init__(self, input_dim, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.Linear(input_dim, input_dim // reduction_ratio)
        self.excitation = nn.Linear(input_dim // reduction_ratio, input_dim)
        self.attention_polling = AttentionPooling(input_dim)

    def forward(self, src, tgt, tgt_mask):
        # batch_size, seq_len, input_dim = x.size()
        # squeeze_output = torch.mean(x, dim=1)  # Global average pooling
        squeeze_output = self.attention_polling(tgt, tgt_mask)
        squeeze_output = F.relu(self.squeeze(squeeze_output))
        excitation_output = torch.sigmoid(self.excitation(squeeze_output)).unsqueeze(1)
        return src * excitation_output

class LocalAttention(nn.Module):
    def __init__(self, input_dim, window_size=5):
        super(LocalAttention, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size

        # 定义局部注意力权重的学习参数
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        attention_scores = torch.zeros(seq_len, seq_len).to(x.device)

        for t in range(seq_len):
            start = max(0, t - self.window_size)
            end = min(seq_len, t + self.window_size)

            query = self.query_projection(x[:, t, :]).unsqueeze(1)  # (batch_size, 1, input_dim)
            keys = self.key_projection(x[:, start:end, :])  # (batch_size, window_size*2+1, input_dim)

            scores = torch.matmul(query, keys.permute(0, 2, 1))  # (batch_size, 1, window_size*2+1)
            attention_scores[t, start:end] = scores.squeeze(1)

        # 如果存在mask，则将mask应用到注意力分数上
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=1)
        output = torch.matmul(attention_weights.unsqueeze(1), x).squeeze(1)

        return output


def swiglu(x, dim):
    x = torch.chunk(x, 2, dim=dim)
    return F.silu(x[0]) * x[1]

class ModelPlus(nn.Module):
    def __init__(self, args, protein_node_embedding, vocab_size=None):
        super().__init__()
        self.config = EsmConfig.from_pretrained(args.esm_model_path)
        self.threshold = args.threshold
        self.device = args.device
        self.batch_size = args.Batch_size
        self.emb_dim = args.hidden_size
        """ protein """
        self.protein_encoder = Encoder(
            args=args, vocab_size=vocab_size, hid_dim=args.protein_hidden_size, n_layers=3, kernel_size=7, dropout=0.1
        )
        self.protein_node_embedding = protein_node_embedding
        for param in self.protein_node_embedding.parameters():
            param.requires_grad = False
        self.node_transformer = nn.Linear(self.config.hidden_size, args.hidden_size)

        self.gcn = GNNEncoder(in_channels=args.hidden_size, hidden_channels=args.hidden_size, out_channels=args.hidden_size)
        self.protein_mlp = nn.Linear(args.hidden_size*2, args.hidden_size)

        """" drug """
        self.drug_encoder = MPN(args)
        self.smiles_encoder = SmilesEncoder(args, 70, hid_dim=args.hidden_size, n_layers=3, kernel_size=5, dropout=args.dropout)

        self.atom_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.bond_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.mol_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)

        """ interaction """
        self.atom_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.bond_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.pro_node_seq_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.mol_protein_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.protein_mol_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)

        self.mol_layerNorm = nn.LayerNorm(args.hidden_size)
        self.protein_layerNorm = nn.LayerNorm(args.hidden_size)

        self.drug_protein_graph_pooling = AttentionPooling(args.hidden_size)
        self.protein_attention_pooling = AttentionPooling(args.hidden_size)
        self.mol_attention_pooling = AttentionPooling(args.hidden_size)

        self.interaction_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 4, args.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, 2)
        )

    def build_graph_with_node_features(self, input_ids_list, protein_graph):
        node_features_list = []
        for seq in input_ids_list:
            outputs = self.node_transformer(self.protein_node_embedding(seq.to(self.device))).squeeze(0)
            node_features_list.append(outputs)
        protein_graph.x = torch.cat(node_features_list, dim=0).to(self.device)
        return protein_graph

    def forward(self, mol_graph, smiles, smiles_mask, protein, protein_mask, seqs, protein_graph):
        """ drug """
        # 每一层的node编码和bond编码
        atom_embed, atom_mask, bond_embed, bond_mask = self.drug_encoder(mol_graph)  # Graph
        smiles_embed = self.smiles_encoder(smiles)  # Smiles

        atom_cross_embed = self.atom_smiles_attention(
            atom_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )[0]
        bond_cross_embed = self.bond_smiles_attention(
            bond_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )[0]

        atom_embed = self.atom_mlp(torch.concat([atom_embed, atom_cross_embed], dim=-1))
        bond_embed = self.bond_mlp(torch.concat([bond_embed, bond_cross_embed], dim=-1))
        mol_embed = self.mol_layerNorm(self.mol_mlp(torch.concat([atom_embed, bond_embed], dim=-1)))

        """protein"""       
        protein_graph = self.build_graph_with_node_features(seqs,protein_graph)

        pro_node_features, pro_node_mask = self.gcn(protein_graph)
        # seq features
        pro_seq_embed = self.protein_encoder(protein, protein_mask)  # sequence

        protein_embed = self.protein_layerNorm(self.protein_mlp(torch.concat([pro_node_features, pro_seq_embed[:,:pro_node_features.shape[1]]], dim=-1)))

        """ interaction """
        mol_cross_embed = self.mol_protein_attention(
            mol_embed, protein_embed, protein_embed, key_padding_mask=(pro_node_mask == 0)
        )[0]
        protein_cross_embed = self.protein_mol_attention(
            protein_embed, mol_embed, mol_embed, key_padding_mask=(atom_mask == 0)
        )[0]

        mol_embed = self.mol_attention_pooling(mol_cross_embed, atom_mask)
        protein_embed = self.protein_attention_pooling(protein_cross_embed, pro_node_mask)
        pair_embed = torch.cat([mol_embed, protein_embed], dim=-1)
        output = self.interaction_layer(pair_embed)

        return output
    
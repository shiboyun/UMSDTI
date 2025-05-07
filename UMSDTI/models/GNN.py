import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import GraphConv, GINConv
from torch.nn.utils.rnn import pad_sequence
from transformers import EsmConfig
import pdb 

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(GNNEncoder, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, batch):
        # batch.x:[num_nodes, in_channels]
        # batch.edge_index:[2, num_edges]
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_attr.squeeze(1)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        
        batch_vecs = []
        batch_mask = []

        for i in range(batch.num_graphs):
            node_mask = (batch.batch == i)
            node_repr = x[node_mask]

            if node_repr.size(0) == 0:
                batch_vecs.append(self.cached_zero_vector.unsqueeze(0))
                batch_mask.append(torch.tensor([0.], device=x.device))
            else:
                batch_vecs.append(node_repr)  # List of [num_nodes_i, out_channels]
                batch_mask.append(torch.ones(node_repr.size(0), device=x.device))

        padded_vecs = pad_sequence(batch_vecs, batch_first=True, padding_value=0)  
        padded_mask = pad_sequence(batch_mask, batch_first=True, padding_value=0)  

        return padded_vecs, padded_mask



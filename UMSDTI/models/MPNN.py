import pdb
from argparse import Namespace
from typing import List, Union, Tuple, Any

import torch
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
import math
import torch.nn.functional as F
from torch import Tensor

class GraphConvolution(nn.Module):
    """Simple GCN layer"""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Uniform weight and bias.
        :return:
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        :param input: input of model.
        :param adj: adjacency matrix.
        :return: output.
        """
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=256):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        # if not hasattr(self, '_flattened'):
        #     self.gru.flatten_parameters()
        #     setattr(self, '_flattened', True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(
            -1.0 / math.sqrt(self.hidden_size), 1.0 / math.sqrt(self.hidden_size)
        )

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)  # [bs, max_node_num, dim]
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        return message


class MPN(nn.Module):
    def __init__(self, args: Namespace):
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim(atom_messages=True)
        self.encoder_atom = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim, atom_messages=True)
        self.bond_fdim = get_bond_fdim(atom_messages=False)
        self.encoder_bond = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim, atom_messages=False)

    def forward(self, batch: Union[List[str], BatchMolGraph], features_batch: List[np.ndarray] = None):
        atom_vecs, atom_mask = self.encoder_atom.forward(batch)
        bond_vecs, bond_mask = self.encoder_bond.forward(batch)

        return atom_vecs, atom_mask, bond_vecs, bond_mask


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int, atom_messages: bool):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
       """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.undirected = args.undirected
        self.device = args.device
        self.atom_messages = atom_messages

        # Dropout
        self.dropout = nn.Dropout(args.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            # w_h_input_size = self.hidden_size + self.bond_fdim
            w_h_input_size = self.hidden_size
        else:
            w_h_input_size = self.hidden_size

        for depth in range(self.depth - 1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        # self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self, mol_graph: BatchMolGraph):
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(
            atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(
            self.device), b2a.to(self.device), b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # (num_atoms / num_bonds) x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                # nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                # nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                # message = nei_message.sum(dim=1) * nei_message.max(dim=1)[0]  # num_atoms x hidden + bond_fdim
                message = nei_a_message.sum(dim=1) * nei_a_message.max(dim=1)[0]
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1) * nei_a_message.max(dim=1)[0]  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                # x2y: 列表内容为y，将此列表作为索引返回得到的内容为x
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            # message = self.W_h(message)
            message = self._modules[f'W_h_{depth}'](message)
            message = self.act_func(input + message)  # (num_atoms/num_bonds) x hidden_size
            message = self.dropout(message)  # (num_atoms/num_bonds) x hidden_size

        # atom hidden
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1) * nei_a_message.max(dim=1)[0]  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)  # num_atoms x hidden

        mol_vecs, mol_mask = self.readout(atom_hiddens, a_scope)

        return mol_vecs, mol_mask  # num_molecules x hidden

    def readout(self, hiddens, scope):
        mol_vecs = []
        mol_length = []
        for i, (start, size) in enumerate(scope):
            if size == 0:
                mol_vecs.append(self.cached_zero_vector)
                mol_length.append(torch.Tensor([0]).float())
            else:
                cur_hiddens = hiddens.narrow(0, start, size)
                mol_vecs.append(cur_hiddens)
                mol_length.append(torch.Tensor([1] * cur_hiddens.shape[0]).float())

        mol_vecs = torch.nn.utils.rnn.pad_sequence(mol_vecs, batch_first=True, padding_value=0)
        mol_mask = torch.nn.utils.rnn.pad_sequence(mol_length, batch_first=True, padding_value=0).to(
            mol_vecs.device)

        return mol_vecs, mol_mask

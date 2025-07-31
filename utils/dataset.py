import pdb
import torch
import numpy as np
from chemprop.features import mol2graph
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from transformers import EsmTokenizer
from torch_geometric.data import Data
import heapq
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_softmax


CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARISOSMISET = {
     "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
     "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
     "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
     "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
     "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
     "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
     "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
     "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64
}

class CustomDataSet(Dataset):
    def __init__(self, args, data):
        self.data = data
        self.protein_max_length = args.protein_max_length

    def __getitem__(self, idx):
        return self.encode(self.data[idx])

    def encode(self, item):
        smiles, protein, label = item
        protein = protein[: self.protein_max_length]
        protein_mask = [1] * len(protein)
        label = int(float(label))

        return smiles, protein, protein_mask, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        batch_smiles = []
        batch_protein, batch_protein_mask = [], []
        batch_labels = []

        for smiles, protein, protein_mask, label in batch:
            batch_smiles.append(smiles)
            batch_protein.append([CHARPROTSET[x] for x in protein])
            # batch_protein.append(protein)
            batch_protein_mask.append(protein_mask)
            batch_labels.append(label)

        batch_mol_graph = mol2graph(batch_smiles)  # 转换为分子图

        """ amino acid sequence """
        batch_protein_str = torch.tensor(sequence_padding(batch_protein), dtype=torch.long)
        batch_protein_mask = torch.tensor(sequence_padding(batch_protein_mask), dtype=torch.long)

        """ label """
        batch_label = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "mol_graph": batch_mol_graph,
            "protein": batch_protein_str,
            "protein_mask": batch_protein_mask,
            "label": batch_label
        }

    @staticmethod
    def collate_fn_ngram(batch):
        batch_smiles, batch_smiles_ids = [], []
        batch_protein, batch_protein_mask = [], []
        batch_labels = []

        for smiles, protein, protein_mask, label in batch:
            batch_smiles.append(smiles)
            batch_smiles_ids.append([CHARISOSMISET[x] for x in smiles])
            # batch_protein.append([CHARPROTSET[x] for x in protein])
            batch_protein.append(protein)
            batch_protein_mask.append(protein_mask)
            batch_labels.append(label)

        """ Smiles """
        batch_mol_graph = mol2graph(batch_smiles)  # 转换为分子图
        batch_smiles_mask = []
        for item in batch_smiles_ids:
            batch_smiles_mask.append([1] * len(item))
        batch_smiles_ids = torch.tensor(sequence_padding(batch_smiles_ids), dtype=torch.long)
        batch_smiles_mask = torch.tensor(sequence_padding(batch_smiles_mask), dtype=torch.long)

        """ amino acid sequence """
        batch_protein_str = torch.tensor(sequence_padding(batch_protein), dtype=torch.long)
        batch_protein_mask = torch.tensor(sequence_padding(batch_protein_mask), dtype=torch.long)

        """ label """
        batch_label = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "mol_graph": batch_mol_graph,
            "smiles": batch_smiles_ids,
            "smiles_mask": batch_smiles_mask,
            "protein": batch_protein_str,
            "protein_mask": batch_protein_mask,
            "label": batch_label
        }


class ContactMapCollator:
    def __init__(self, args):
         self.tokenizer = EsmTokenizer.from_pretrained(args.esm_model_path)
         self.args = args 

    def __call__(self, batch):
        ouputs = self.tokenizer(batch, truncation=True, padding=True, return_tensors='pt', max_length=self.args.protein_max_length+2)
        input_ids, attention_mask = ouputs['input_ids'], ouputs['attention_mask']

        return {
            "input_ids": input_ids, "attention_mask": attention_mask
        }

def sequence_padding(sequences):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in sequences]
    return padded_sequences

class ContactMapDataset(Dataset):
    def __init__(self, args, data):
        self.data = data
        self.protein_max_length = args.protein_max_length
        self.tokenizer = EsmTokenizer.from_pretrained(args.esm_model_path)
        self.threshold = args.threshold

    def __getitem__(self, idx):
        return self.encode(self.data[idx])

    def encode(self, item):
        smiles, sequence, ngram_words, contact_map, label = item

        sequence_tokenized = self.tokenizer(
            sequence,
            max_length=self.protein_max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            return_tensors='pt'
        )
        sequence_ids = sequence_tokenized['input_ids']
        sequence_mask = sequence_tokenized['attention_mask']
        seq_len = sequence_ids.size(1)

        ngram_words = ngram_words[:seq_len]
        ngram_words_mask = [1] * len(ngram_words)
        label = int(float(label))
        contact_map = contact_map[:seq_len, :seq_len]
        contact_map_no_diag = contact_map.copy()
        np.fill_diagonal(contact_map_no_diag, -np.inf)

        triu_indices = np.triu_indices(seq_len, k=1)
        flat = contact_map_no_diag[triu_indices]
        k = max(1, int(self.threshold * len(flat)))
        topk_indices = np.argpartition(flat, -k)[-k:]
        contact_edges = np.vstack(triu_indices)[:, topk_indices].T
        contact_weights = contact_map[contact_edges[:, 0], contact_edges[:, 1]]
        edges_rev = contact_edges[:, [1, 0]]
        edges = np.concatenate([contact_edges, edges_rev], axis=0)
        weights = np.concatenate([contact_weights, contact_weights], axis=0)
        self_loops = np.arange(seq_len).reshape(-1, 1).repeat(2, axis=1)
        self_weights = np.ones(seq_len, dtype=np.float32)

        all_edges = np.concatenate([edges, self_loops], axis=0)
        all_weights = np.concatenate([weights, self_weights], axis=0)
        

        edge_index = torch.tensor(all_edges.T, dtype=torch.long)
        edge_weight = torch.tensor(all_weights, dtype=torch.float)

        row, col = edge_index
        is_self_loop = (row == col)
        non_self_mask = ~is_self_loop

        non_self_edge_index = edge_index[:, non_self_mask]
        non_self_edge_weight = edge_weight[non_self_mask]
        non_self_row = non_self_edge_index[0]
        normalized_non_self_weight = scatter_softmax(non_self_edge_weight, non_self_row)
        edge_attr = torch.zeros_like(edge_weight)
        edge_attr[non_self_mask] = normalized_non_self_weight
        edge_attr[is_self_loop] = 1.0

        edge_attr = edge_attr.unsqueeze(1)

        protein_graph = Data(
            input_ids=sequence_ids.squeeze(0),
            seq_mask=sequence_mask.squeeze(0),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=seq_len
        )

        return smiles, sequence_ids, sequence_mask, ngram_words, ngram_words_mask, protein_graph, label

    def __len__(self):
        return len(self.data)


class MainCollator:
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        batch_smiles, batch_smiles_ids = [], []
        batch_seq, batch_seq_mask = [], []
        batch_words, batch_words_mask = [], []
        batch_protein_graphs = []
        batch_labels = []

        for smiles, sequence_ids, sequence_mask, ngram_words, ngram_words_mask, protein_graph, label in batch:
            batch_smiles.append(smiles)
            batch_smiles_ids.append([CHARISOSMISET[x] for x in smiles])

            batch_seq.append(sequence_ids)
            batch_seq_mask.append(sequence_mask)

            batch_words.append(ngram_words)
            batch_words_mask.append(ngram_words_mask)

            batch_protein_graphs.append(protein_graph)
            batch_labels.append(label)

        batch_mol_graph = mol2graph(batch_smiles)

        batch_smiles_mask = [[1] * len(s) for s in batch_smiles_ids]
        batch_smiles_ids = torch.tensor(sequence_padding(batch_smiles_ids), dtype=torch.long)
        batch_smiles_mask = torch.tensor(sequence_padding(batch_smiles_mask), dtype=torch.long)

        batch_words = torch.tensor(sequence_padding(batch_words), dtype=torch.long)
        batch_words_mask = torch.tensor(sequence_padding(batch_words_mask), dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        batch_protein_graph = Batch.from_data_list(batch_protein_graphs)

        return {
            "mol_graph": batch_mol_graph,
            "smiles": batch_smiles_ids,
            "smiles_mask": batch_smiles_mask,
            "seqs": batch_seq,
            "protein_graph": batch_protein_graph,
            "ngram_words": batch_words,
            "ngram_words_mask": batch_words_mask,
            "label": batch_labels
        }

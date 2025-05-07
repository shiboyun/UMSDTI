# -*- coding: utf-8 -*-
import os
import pdb
from collections import defaultdict
import pandas as pd
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from prefetch_generator import BackgroundGenerator
from transformers import EsmTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc

from utils.dataset import ContactMapCollator, MainCollator, ContactMapDataset
from utils.ema import EMA
from utils.tools import get_optimizer_and_scheduler, setup_seed
from models.ProteinEncoder import ContactPredictor
from models.DTIModel import ModelPlus
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def getConfig():
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default="DrugBank")

    """ train args """
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--Batch_size', type=int, default=16)
    parser.add_argument('--Epoch', type=int, default=100)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--ema_decay', type=float, default=0.997)
    parser.add_argument('--early_stop_nums', type=int, default=20)
    parser.add_argument('--esm_model_path', type=str, default="/home/sby/DTI/PretrainedModels/esm2_t12_35M_UR50D") ##path to pretrain model
    parser.add_argument('--threshold', type=float, default=0.1)

    parser.add_argument('--drug_encoder_learning_rate', type=float, default=2e-4)
    parser.add_argument('--protein_encoder_learning_rate', type=float, default=2e-4)
    parser.add_argument('--interaction_learning_rate', type=float, default=1e-4)

    """ Protein Model Parameters """
    parser.add_argument('--protein_max_length', type=int, default=2048, help='氨基酸序列长度')
    parser.add_argument('--protein_hidden_size', type=int, default=128)  # 256 128
    parser.add_argument('--protein_conv', type=int, default=40)
    parser.add_argument('--n_gram', type=int, default=3)

    """ Drug Model Parameters """
    parser.add_argument('--hidden_size', type=int, default=128, help='Dimensionality of hidden layers in MPN')  # 256 128
    parser.add_argument('--bias', action='store_true', default=False, help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=4, help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ELU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=True,
                        help='Undirected edges (always sum the two relevant bond vectors)')

    return parser.parse_args()


def evalueate(model, dataset_load, LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []

    with torch.no_grad():
        for batch in dataset_load:
            '''data preparation '''
            inputs = {
                "mol_graph": batch["mol_graph"],
                "smiles": batch["smiles"].to(args.device),
                "smiles_mask": batch["smiles_mask"].to(args.device),
                "protein": batch["ngram_words"].to(args.device),
                "protein_mask": batch["ngram_words_mask"].to(args.device),
                "seqs": batch["seqs"],
                "protein_graph": batch["protein_graph"].to(args.device),
            }
            labels = batch["label"].to(args.device)
            predicted_scores = model(**inputs)

            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()

            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())

    try:
        Precision = precision_score(Y, P, zero_division=1)
        Reacll = recall_score(Y, P)
        AUC = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        PRC = auc(fpr, tpr)
        Accuracy = accuracy_score(Y, P)
        test_loss = np.average(test_losses)
    except:
        pdb.set_trace()

    return Y, P, test_loss, Accuracy, Precision, Reacll, AUC, PRC


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]] for i in range(len(sequence) - ngram + 1)]
    return words


def get_contact_map(args, model, tokenizer, collator, sequence):
    total_contact_maps = []
    dataloader = DataLoader(sequence, batch_size=1, shuffle=False, collate_fn=collator)
    for batch in tqdm(dataloader, total=len(dataloader), desc="get contact map"):
        inputs = {
            "input_ids": batch['input_ids'].to(args.device), "attention_mask": batch['attention_mask'].to(args.device)
        }
        with torch.no_grad():
            batch_contact_maps = model(inputs['input_ids'], inputs['attention_mask']).cpu().numpy()
        
        total_contact_maps.extend(batch_contact_maps)
    torch.cuda.empty_cache()
    gc.collect()
    return total_contact_maps


if __name__ == "__main__":
    """init hyperparameters"""
    args = getConfig()

    """select seed"""
    setup_seed(args)

    """ init logging """
    os.makedirs(f'./log/{args.dataset}', exist_ok=True)
    logging.basicConfig(
        filename=f'./log/{args.dataset}/{args.dataset}_seed{args.seed}_dim128_threshold{args.threshold}.log', level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', filemode='w'
    )
    logging.info(args.__dict__)

    """ load data """
    data_list_df = pd.read_csv(f"./data/{args.dataset}/independent-train.csv", header=None)
    data_list_df.columns = ['protein', 'drug', 'label']
    data_list = [[a, b, c] for a, b, c in zip(
        data_list_df['drug'].values.tolist(), data_list_df['protein'].values.tolist(),
        data_list_df['label'].values.tolist()
    )]

    test_data_list_df = pd.read_csv(f"./data/{args.dataset}/independent-test.csv", header=None)
    test_data_list_df.columns = ['protein', 'drug', 'label']
    test_data_list = [[a, b, c] for a, b, c in zip(
        test_data_list_df['drug'].values.tolist(), test_data_list_df['protein'].values.tolist(),
        test_data_list_df['label'].values.tolist()
    )]

    data_list = shuffle_dataset(data_list, args.seed)
    train_data_list, valid_data_list = split_dataset(data_list, 0.8)

    """initialize contact predict model """
    contact_predictor = ContactPredictor(args)
    contact_predictor.to(args.device)
    collator = ContactMapCollator(args)
    contact_tokenizer = EsmTokenizer.from_pretrained(args.esm_model_path)

    """ initialize n-gram word_dict """
    word_dict = defaultdict(lambda: len(word_dict))
    # train_list, valid_list, test_list = [], [], []

    def data_format(args, contact_predictor, contact_tokenizer, collator, data_list):
        total_smiles, total_sequences, total_words, total_labels = [], [], [], []
        for line in data_list:
            smiles, sequence, label = line
            total_smiles.append(smiles)
            total_sequences.append(sequence)
            total_words.append(split_sequence(sequence, args.n_gram))
            total_labels.append(label)

        unique_sequences = list(set(total_sequences))

        unique_contact_maps = get_contact_map(args, contact_predictor, contact_tokenizer, collator, unique_sequences)

        seq_to_contact = {seq: contact_map for seq, contact_map in zip(unique_sequences, unique_contact_maps)}
        total_contact_maps = [seq_to_contact[seq] for seq in total_sequences]

        return list(zip(total_smiles, total_sequences, total_words, total_contact_maps, total_labels))

    train_data_list = data_format(args, contact_predictor, contact_tokenizer, collator, train_data_list)
    valid_data_list = data_format(args, contact_predictor, contact_tokenizer, collator, valid_data_list)
    test_data_list = data_format(args, contact_predictor, contact_tokenizer, collator, test_data_list)
    
    logging.info(f"n_words: {len(word_dict)}")

    """ load train/test data """
    train_dataset = ContactMapDataset(args, train_data_list)
    valid_dataset = ContactMapDataset(args, valid_data_list)
    test_dataset = ContactMapDataset(args, test_data_list)

    collator = MainCollator(args)

    train_dataset_load = DataLoader(
        train_dataset, batch_size=args.Batch_size, shuffle=True, num_workers=8, collate_fn=collator
    )
    valid_dataset_load = DataLoader(
        valid_dataset, batch_size=args.Batch_size, shuffle=False, num_workers=8, collate_fn=collator
    )
    test_dataset_load = DataLoader(
        test_dataset, batch_size=args.Batch_size, shuffle=False, num_workers=8, collate_fn=collator
    )

    """create model"""
    model = ModelPlus(args,contact_predictor.model.embeddings.word_embeddings, len(word_dict))
    # model = Model(args, 30)

    """weight initialize"""
    num_total_steps = len(train_dataset_load) * int(args.Epoch)
    iters = len(train_dataset_load)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, num_total_steps)
    model = model.to(args.device)

    """load trained model"""
    Loss = nn.CrossEntropyLoss(weight=None)

    """Start training."""
    best_auc, best_aupr = 0.0, 0.0

    ema = EMA(model, decay=args.ema_decay)
    ema.register()

    step = 0
    early_stop = 0
    for epoch in range(1, args.Epoch + 1):
        torch.cuda.empty_cache()
        gc.collect()

        trian_pbar = tqdm(
            BackgroundGenerator(train_dataset_load),
            total=len(train_dataset_load), desc=f"[Epoch: {epoch}]")
        """train"""
        train_losses_in_epoch = []
        model.train()
        for i, batch in enumerate(trian_pbar):
            step += 1

            optimizer.zero_grad()
            '''data preparation '''
            inputs = {
                "mol_graph": batch["mol_graph"],
                "smiles": batch["smiles"].to(args.device),
                "smiles_mask": batch["smiles_mask"].to(args.device),
                "protein": batch["ngram_words"].to(args.device),
                "protein_mask": batch["ngram_words_mask"].to(args.device),
                "seqs": batch["seqs"],
                "protein_graph": batch["protein_graph"].to(args.device),
            }
            trian_labels = batch["label"].to(args.device)
            predicted_interaction = model(**inputs)
            train_loss = Loss(predicted_interaction, trian_labels)

            train_losses_in_epoch.append(train_loss.item())  # 将每个batch的loss加入列表

            train_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()
            # scheduler.step(epoch + i / iters)
            scheduler.step()
            train_loss.detach()

            ema.update()

            trian_pbar.set_postfix(loss=np.average(train_losses_in_epoch))
            # trian_pbar.set_postfix(loss=train_loss.item())

        """ Test """

        ema.apply_shadow()

        _, _, _, _, valid_Precision, valid_Reacll, valid_AUC, valid_PRC = evalueate(model, valid_dataset_load, Loss)
        _, _, _, _, test_Precision, test_Reacll, test_AUC, test_PRC = evalueate(model, test_dataset_load, Loss)

        if best_auc <= valid_AUC or best_aupr <= valid_PRC:
            best_auc = valid_AUC
            best_aupr = valid_PRC
            torch.save(model.state_dict(), f"./checkpoints/{args.dataset}.bin")
            early_stop = 0
        else:
            early_stop += 1

        logging.info(
            f'Epoch: {epoch} ' +
            f'valid_AUC: {valid_AUC:.5f} ' +
            f'valid_PRC: {valid_PRC:.5f} ' +
            f'valid_Precision: {valid_Precision:.5f} ' +
            f'valid_Reacll: {valid_Reacll:.5f} ' +
            f'test_AUC: {test_AUC:.5f} ' +
            f'test_PRC: {test_PRC:.5f} ' +
            f'test_Precision: {test_Precision:.5f} ' +
            f'test_Reacll: {test_Reacll:.5f} ' +
            f'best_AUC: {best_auc: .5f} ' +
            f'best_AUPR: {best_aupr: .5f} '
        )
        
        ema.restore()
            
        if early_stop >= args.early_stop_nums:
            break

    """ Test """
    model = ModelPlus(args, contact_predictor.model.embeddings.word_embeddings, len(word_dict))
    model.load_state_dict(torch.load(f"./checkpoints/{args.dataset}.bin"))
    model.to(args.device)

    _, _, _, _, Precision, Reacll, AUC, PRC = evalueate(model, test_dataset_load, Loss)
    
    logging.info(
        f'test_AUC: {AUC:.5f} ' +
        f'test_PRC: {PRC:.5f} ' +
        f'test_Precision: {Precision:.5f} ' +
        f'test_Reacll: {Reacll:.5f} '
    )
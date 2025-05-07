import logging
import random
import os
import pdb
import torch
import random
import numpy as np
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, \
    get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR


# def setup_device(args):
#     args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def get_optimizer_and_scheduler(args, model, num_total_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    drug_encoder_optimizer = []
    protein_encoder_optimizer = []
    interaction_optimizer = []

    for name, param in model.named_parameters():
        if 'drug_encoder' in name or 'smiles_encoder' in name:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            drug_encoder_optimizer.append((name, param))
        elif 'protein_encoder' in name:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            protein_encoder_optimizer.append((name, param))
        else:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            interaction_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in drug_encoder_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.drug_encoder_learning_rate},
        {
            'params': [p for n, p in drug_encoder_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': args.drug_encoder_learning_rate},
        {
            'params': [p for n, p in protein_encoder_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.protein_encoder_learning_rate},
        {
            'params': [p for n, p in protein_encoder_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': args.protein_encoder_learning_rate},
        {
            'params': [p for n, p in interaction_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.interaction_learning_rate},
        {
            'params': [p for n, p in interaction_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': args.interaction_learning_rate}

    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_ratio * num_total_steps, num_training_steps=num_total_steps * 1.05
    )
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-5)

    return optimizer, scheduler


def get_optimizer_and_scheduler_cosine(args, model, num_total_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    drug_encoder_optimizer = []
    protein_encoder_optimizer = []
    interaction_optimizer = []

    for name, param in model.named_parameters():
        if 'drug_encoder' in name or 'smiles_encoder' in name:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            drug_encoder_optimizer.append((name, param))
        elif 'protein_encoder' in name:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            protein_encoder_optimizer.append((name, param))
        else:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            interaction_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in drug_encoder_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.drug_encoder_learning_rate},
        {
            'params': [p for n, p in drug_encoder_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': args.drug_encoder_learning_rate},
        {
            'params': [p for n, p in protein_encoder_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.protein_encoder_learning_rate},
        {
            'params': [p for n, p in protein_encoder_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': args.protein_encoder_learning_rate},
        {
            'params': [p for n, p in interaction_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.interaction_learning_rate},
        {
            'params': [p for n, p in interaction_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': args.interaction_learning_rate}

    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_ratio * num_total_steps, num_training_steps=num_total_steps,
        num_cycles=0.5
    )
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-5)

    return optimizer, scheduler
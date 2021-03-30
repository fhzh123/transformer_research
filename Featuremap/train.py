# Import modules
import os
import time
import pickle
from tqdm import tqdm

# Import PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import custom modules
from dataset import CustomDataset, PadCollate
from model.optimizer import Ralamb, WarmupLinearSchedule

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    print('Data Load & Setting!')
    with open(os.path.join(args.save_path, 'processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        train_indices = data_['train_indices']
        valid_indices = data_['valid_indices']
        train_title_indices = data_['train_title_indices']
        valid_title_indices = data_['valid_title_indices']
        train_total_indices = data_['train_total_indices']
        valid_total_indices = data_['valid_total_indices']
        train_label = data_['train_label']
        valid_label = data_['valid_label']
        word2id = data_['word2id']
        id2word = data_['id2word']
        vocab_num = len(word2id.keys())
        del data_

    dataset_dict = {
        'train': CustomDataset(train_total_indices, train_indices, 
                               train_title_indices, max_len=args.max_len),
        'train': CustomDataset(train_total_indices, train_indices, 
                               train_title_indices, max_len=args.max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)
    }
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    print("Instantiating models...")
    model = Transformer_model(vocab_num=vocab_num, pad_idx=args.pad_idx, bos_idx=args.bos_idx, 
                              eos_idx=args.eos_idx, max_len=args.max_len, d_model=args.d_model, 
                              d_embedding=args.d_embedding, n_head=args.n_head, 
                              dim_feedforward=args.dim_feedforward, n_layers=args.n_layers, 
                              dropout=args.dropout, baseline=args.baseline, device=device)
    optimizer = Ralamb(params=filter(lambda p: p.requires_grad, model.parameters()),
                       lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=len(dataloader_dict['train'])*3, 
                                     t_total=len(dataloader_dict['train'])*args.num_epoch)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    model.to(device)

    #===================================#
    #=========Model Train Start=========#
    #===================================#
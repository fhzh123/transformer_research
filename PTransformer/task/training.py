import os
import pickle
from time import time
# Import PyTorch
import torch
from torch.utils.data import DataLoader
# Import custom modules
from model.transformer import Transformer
from optimizer.optimizer import Ralamb
from optimizer.scheduler import WarmupLinearSchedule

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    print('Data Load & Setting!')
    with open(os.path.join(args.preprocess_path, 'unlabeled_processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        train_src_indices = data_['train_src_indices']
        valid_src_indices = data_['valid_src_indices']
        train_trg_indices = data_['train_trg_indices']
        valid_trg_indices = data_['valid_trg_indices']
        src_word2id = data_['src_word2id']
        trg_word2id = data_['trg_word2id']
        del data_

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(train_src_indices, train_trg_indices, 
                               min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
        'valid': CustomDataset(valid_src_indices, valid_trg_indices,
                               min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(), drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    print("Instantiating models...")
    model = Transformer(src_vocab_num=args.src_vocab_size, trg_vocab_num=args.trg_vocab_size,
                        pad_idx=args.pad_id, bos_idx=args.bos_id, eos_idx=args.eos_id,
                        src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                        d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
                        d_k=args.d_k, d_v=args.d_v, dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout, embedding_dropout=args.embedding_dropout,
                        n_common_layers=args.n_common_layers, n_encoder_layers=args.n_encoder_layers,
                        n_decoder_layers=args.n_decoder_layers,
                        trg_emb_prj_weight_sharing=args.trg_emb_prj_weight_sharing,
                        emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing, parallel=args.parallel,
                        device=device)
    model = model.train()
    model = model.to(device)

    # 2) Optimizer & Learning rate scheduler setting
    optimizer = Ralamb(transformer.parameters(), lr=opt.max_lr, weight_decay=opt.weight_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=len(dataloader_dict['train'])*args.n_warmup_epochs, 
                    t_total=len(dataloader_dict['train'])*args.num_epochs)

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#
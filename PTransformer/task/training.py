import os
import pickle
from time import time
# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
# Import Huggingface
from transformers import AdamW
# Import custom modules
from dataset import CustomDataset, PadCollate
from model.transformer import Transformer
from optimizer.optimizer import Ralamb
from optimizer.utils import shceduler_select
from utils import cal_loss

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    print('Data Load & Setting!')
    with open(os.path.join(args.preprocess_path, 'processed.pkl'), 'rb') as f:
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
    # 2) Optimizer setting
    optimizer = AdamW(model.parameters(), lr=args.max_lr, eps=1e-8)
    # optimizer = Ralamb(filter(lambda p: p.requires_grad, model.parameters()), lr=args.max_lr, weight_decay=args.w_decay)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)

    # 3) Model resume
    start_epoch = 0
    # if args.resume:
    #     checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint.pth.tar'))
    #     start_epoch = checkpoint['epoch'] + 1
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

    print('Train start!')

    for epoch in range(start_epoch, args.num_epochs):
        start_time_e = time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                print('Validation start...')
                val_loss = 0
                val_acc = 0
                model.eval()
            for i, (src, trg) in enumerate(dataloader_dict[phase]):

                # Input, output setting
                src = src.to(device, non_blocking=True)
                trg = trg.to(device, non_blocking=True)

                non_pad = trg != args.pad_id
                trg_sequences_target = trg[non_pad].contiguous().view(-1)
                
                if phase == 'train':
                    with torch.set_grad_enabled(True):
                        seq_logit = model(src, trg, non_pad_position=non_pad)

                        # Optimizer setting
                        optimizer.zero_grad()

                        # Loss calculate
                        # loss = cal_loss(seq_logit, trg_sequences_target, args.pad_id, smoothing=True)
                        loss = F.cross_entropy(seq_logit, trg_sequences_target)
                        loss.backward()
                        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                        optimizer.step()
                        if args.scheduler in ['constant', 'warmup']:
                            scheduler.step()
                        if args.scheduler == 'reduce_train':
                            scheduler.step(loss)

                        # Print loss value only training
                        if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                            acc = (seq_logit.max(dim=1)[1] == trg_sequences_target).sum() / len(trg_sequences_target)
                            print("[Epoch:%d][%d/%d] train_loss:%3.3f  | train_acc:%3.3f | learning_rate:%3.6f | spend_time:%3.2fmin"
                                    % (epoch+1, i, len(dataloader_dict['train']), 
                                    loss.item(), acc, optimizer.param_groups[0]['lr'], 
                                    (time() - start_time_e) / 60))
                            freq = 0
                        freq += 1

                if phase == 'valid':
                    with torch.no_grad():
                        seq_logit = model(src, trg, non_pad_position=non_pad)
                        # loss = cal_loss(seq_logit, trg_sequences_target, args.pad_id, smoothing=False)
                        loss = F.cross_entropy(seq_logit, trg_sequences_target)
                    val_loss += loss.item()
                    val_acc += (seq_logit.max(dim=1)[1] == trg_sequences_target).sum() / len(trg_sequences_target)
                    if args.scheduler == 'reduce_valid':
                        scheduler.step(val_loss)
                    if args.scheduler == 'lambda':
                        scheduler.step()

            if phase == 'valid':
                val_loss /= len(dataloader_dict[phase])
                val_acc /= len(dataloader_dict[phase])
                print(f'Validation Loss: {val_loss}')
                print(f'Validation Accuracy: {val_acc}')
                if val_acc > best_val_acc:
                    print('Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, f'transformer_testing.pth.tar')
                    best_val_acc = val_acc
                    best_epoch = epoch

    # 3) Print results
    print(f'Best Epoch: {best_epoch+1}')
    print(f'Best Accuracy: {round(best_val_acc.item(), 2)}')
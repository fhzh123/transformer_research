# Import modules
import os
import time
import pickle
from tqdm import tqdm

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# Import custom modules
from model.model import Transformer
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
                            train_title_indices, train_label,
                            max_len=args.max_len),
        'valid': CustomDataset(valid_total_indices, valid_indices, 
                            valid_title_indices, valid_label,
                            max_len=args.max_len),
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

    # 1) Model initiating
    print("Instantiating models...")
    model = Transformer(vocab_num=vocab_num, pad_idx=args.pad_idx, bos_idx=args.bos_idx, 
                        eos_idx=args.eos_idx, max_len=args.max_len, d_model=args.d_model, 
                        d_embedding=args.d_embedding, n_head=args.n_head, 
                        dim_feedforward=args.dim_feedforward, n_layers=args.n_layers, 
                        dropout=args.dropout, device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = Ralamb(params=filter(lambda p: p.requires_grad, model.parameters()),
    #                 lr=args.lr, weight_decay=args.w_decay)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=len(dataloader_dict['train'])*3, 
    #                                 t_total=len(dataloader_dict['train'])*args.num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    model = model.train()
    model = model.to(device)

    # 2) Model resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    # 1) Pre-setting
    best_val_loss = None

    # 2) Training start
    for e in range(start_epoch, args.num_epochs):
        start_time_e = time.time()

        for i, (total, comment, title, label) in enumerate(tqdm(dataloader_dict['train'])):
            # Source, Target  setting
            total = total.to(device)
            comment = comment.to(device)
            label = label.to(device)
            
            # Optimizer setting
            optimizer.zero_grad()

            # Model / Calculate loss
            output = model(total)
            output_cls_token = output[:,0]
            loss = criterion(output_cls_token, label)
            predicted = output_cls_token.max(dim=1)[1]
            acc = sum(predicted == label).item() / len(label)

            # If phase train, then backward loss and step optimizer and scheduler
            loss.backward()
            # clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            # scheduler.step()

            # Print loss value only training
            if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                total_loss = loss.item()
                print("[Epoch:%d][%d/%d] train_loss:%5.3f | train_acc:%5.3f | learning_rate:%3.8f | spend_time:%5.2fmin"
                        % (e+1, i, len(dataloader_dict['train']), 
                        total_loss, acc,
                        optimizer.param_groups[0]['lr'], 
                        (time.time() - start_time_e) / 60))
                freq = 0
            freq += 1

            # # Finishing iteration
            # if phase == 'valid':
            #     val_loss /= len(dataloader_dict['valid'])
            #     val_acc /= len(dataloader_dict['valid'])
            #     print("[Epoch:%d] val_loss:%5.3f | val_acc:%5.3f | spend_time:%5.2fmin"
            #             % (e+1, val_loss, val_acc,
            #             (time.time() - start_time_e) / 60))
            #     if not best_val_loss or val_loss < best_val_loss:
            #         print("[!] saving model...")
            #         if not os.path.exists(args.save_path):
            #             os.mkdir(args.save_path)
            #         torch.save(model.state_dict(), 
            #                     os.path.join(args.save_path, f'model_{val_loss}.pt'))
            #         best_val_loss = val_loss
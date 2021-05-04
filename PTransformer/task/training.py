# Import modules
import os
import gc
import pickle
import logging
from time import time
from tqdm import tqdm
# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.transformer import Transformer
from model.dataset import CustomDataset
from optimizer.utils import shceduler_select, optimizer_select
from utils import label_smoothing_loss, TqdmLoggingHandler, write_log

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()
    with open(os.path.join(args.preprocess_path, 'processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        train_src_indices = data_['train_src_indices']
        valid_src_indices = data_['valid_src_indices']
        train_trg_indices = data_['train_trg_indices']
        valid_trg_indices = data_['valid_trg_indices']
        src_word2id = data_['src_word2id']
        trg_word2id = data_['trg_word2id']
        src_vocab_num = len(src_word2id)
        trg_vocab_num = len(trg_word2id)
        del data_
    gc.enable()
    write_log(logger, "Finished loading data!")

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(train_src_indices, train_trg_indices, 
                            min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
        'valid': CustomDataset(valid_src_indices, valid_trg_indices,
                            min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    model = Transformer(src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                        pad_idx=args.pad_id, bos_idx=args.bos_id, eos_idx=args.eos_id,
                        d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
                        dim_feedforward=args.dim_feedforward,
                        num_common_layer=args.num_common_layer, num_encoder_layer=args.num_encoder_layer,
                        num_decoder_layer=args.num_decoder_layer,
                        src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                        dropout=args.dropout, embedding_dropout=args.embedding_dropout,
                        trg_emb_prj_weight_sharing=args.trg_emb_prj_weight_sharing,
                        emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing, 
                        parallel=args.parallel)
    model.train()
    model = model.to(device)
    tgt_mask = model.generate_square_subsequent_mask(args.trg_max_len - 1, device)

    # 2) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

    write_log(logger, 'Traing start!')

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time_e = time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                write_log(logger, 'Validation start...')
                val_loss = 0
                val_acc = 0
                model.eval()
            for i, (src, trg) in enumerate(tqdm(dataloader_dict[phase], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

                # Optimizer setting
                optimizer.zero_grad(set_to_none=True)

                # Input, output setting
                src = src.to(device, non_blocking=True)
                trg = trg.to(device, non_blocking=True)

                trg_sequences_target = trg[:, 1:]
                non_pad = trg_sequences_target != args.pad_id
                trg_sequences_target = trg_sequences_target[non_pad].contiguous().view(-1)
                
                # Train
                if phase == 'train':

                    # Loss calculate
                    with autocast():
                        predicted = model(src, trg[:, :-1], tgt_mask, non_pad_position=non_pad)
                        predicted = predicted.view(-1, predicted.size(-1))
                        loss = label_smoothing_loss(predicted, trg_sequences_target, args.pad_id)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    if args.scheduler in ['constant', 'warmup']:
                        scheduler.step()
                    if args.scheduler == 'reduce_train':
                        scheduler.step(loss)

                    # Print loss value only training
                    if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                        acc = (predicted.max(dim=1)[1] == trg_sequences_target).sum() / len(trg_sequences_target)
                        iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.3f | train_acc:%03.2f%% | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                            (epoch, i, len(dataloader_dict['train']), 
                            loss.item(), acc*100, optimizer.param_groups[0]['lr'], 
                            (time() - start_time_e) / 60)
                        write_log(logger, iter_log)
                        freq = 0
                    freq += 1

                # Validation
                if phase == 'valid':
                    with torch.no_grad():
                        predicted = model(src, trg[:, :-1], tgt_mask, non_pad_position=non_pad)
                        loss = F.cross_entropy(predicted, trg_sequences_target)
                    val_loss += loss.item()
                    val_acc += (predicted.max(dim=1)[1] == trg_sequences_target).sum() / len(trg_sequences_target)
                    if args.scheduler == 'reduce_valid':
                        scheduler.step(val_loss)
                    if args.scheduler == 'lambda':
                        scheduler.step()

            if phase == 'valid':
                val_loss /= len(dataloader_dict[phase])
                val_acc /= len(dataloader_dict[phase])
                write_log(logger, 'Validation Loss: %3.3f' % val_loss)
                write_log(logger, 'Validation Accuracy: %3.2f%%' % (val_acc * 100))
                if val_acc > best_val_acc:
                    write_log(logger, 'Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict()
                    }, f'checkpoint.pth.tar')
                    best_val_acc = val_acc
                    best_epoch = epoch
                else:
                    else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc.item()*100, 2)})%% is better...'
                    write_log(logger, else_log)

    # 3) Print results
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_val_acc.item(), 2)}')
# Import modules
import os
import time
import pickle
import random
from tqdm import tqdm
# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
# Import custom modules
from optimizer.learning_rate import WarmupLinearSchedule
from task.pretrain.model import kcBERT_pretraining
from task.pretrain.dataset import CustomDataset, PadCollate
# Import Huggingface
from transformers import AdamW

def pretraining(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    print('Data Load & Setting!')
    with open(os.path.join('./preprocessing/', 'unlabeled_processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        unlabel_title = data_['unlabel_title']
        unlabel_comments = data_['unlabel_comments']
        del data_

    # 2) Train, Test split
    split_ix = int(len(unlabel_title) * (1-args.split_ratio))
    ix = list(range(len(unlabel_title)))
    random.shuffle(ix)
    unlabel_title = [unlabel_title[i] for i in ix]
    unlabel_comments = [unlabel_comments[i] for i in ix]

    train_unlabel_title = unlabel_title[:split_ix]
    train_unlabel_comments = unlabel_comments[:split_ix]
    test_unlabel_title = unlabel_title[split_ix:]
    test_unlabel_comments = unlabel_comments[split_ix:]

    # 3) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(train_unlabel_title, train_unlabel_comments),
        'valid': CustomDataset(test_unlabel_title, test_unlabel_comments)
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
    #===========Model setting===========#
    #===================================#

    # 1) Model initiating
    print("Instantiating models...")
    model = kcBERT_pretraining(vocab_size=30000)
    model = model.train()
    model = model.to(device)
    print('Done!')

    # 2) Optimizer setting
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=len(dataloader_dict['train']), 
                                    t_total=len(dataloader_dict['train'])*args.num_epochs)

    # 2) Model resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load('./checkpoint_testing.pth.tar', map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

    print('Train start!')

    for epoch in range(start_epoch, args.num_epochs):
        start_time_e = time.time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                print('Validation start...')
                val_mlm_loss = 0
                val_nsp_loss = 0
                model.eval()
            for i, (masking_text, segment, text, nsp_label) in enumerate(tqdm(dataloader_dict[phase])):

                # Input, output setting
                masking_text = masking_text.to(device)
                masking_position = masking_text==4
                segment = segment.to(device)
                mlm_label = text[masking_position].to(device)
                nsp_label = nsp_label.to(device)

                if phase == 'train':
                    with torch.set_grad_enabled(True):
                        mlm_logit, nsp_logit = model(src_input_sentence=masking_text, src_segment=segment)

                        # Optimizer setting
                        optimizer.zero_grad()

                        # 
                        mlm_loss = F.cross_entropy(mlm_logit[masking_position], mlm_label)
                        nsp_loss = F.cross_entropy(nsp_logit, nsp_label)
                        total_loss = mlm_loss + nsp_loss
                        total_loss.backward()
                        clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                        scheduler.step()

                        # Print loss value only training
                        if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                            print("[Epoch:%d][%d/%d] train_mlm_loss:%3.3f  | train_nsp_loss:%3.3f | learning_rate:%3.6f | spend_time:%3.2fmin"
                                    % (epoch+1, i, len(dataloader_dict['train']), 
                                    mlm_loss.item(), nsp_loss.item(), optimizer.param_groups[0]['lr'], 
                                    (time.time() - start_time_e) / 60))
                            freq = 0
                        freq += 1

                if phase == 'valid':
                    with torch.no_grad():
                        pred = model(src_input_sentence=masking_text, src_segment=segment)
                    loss = F.cross_entropy(pred, label.contiguous())
                    val_mlm_loss += mlm_loss.item()
                    val_nsp_loss += nsp_loss.item()
                    val_mlm_acc += (mlm_logit.max(dim=1)[1] == label).sum() / len(label)
                    val_nsp_acc += (nsp_logit.max(dim=1)[1] == label).sum() / len(label)

            if phase == 'valid':
                val_mlm_loss /= len(dataloader_dict[phase])
                val_nsp_loss /= len(dataloader_dict[phase])
                val_mlm_acc /= len(dataloader_dict[phase])
                val_nsp_acc /= len(dataloader_dict[phase])
                print(f'Validation MLM Loss: {val_mlm_loss}')
                print(f'Validation MLM Accuracy: {val_nsp_loss}')
                print(f'Validation MLM Loss: {val_mlm_acc}')
                print(f'Validation MLM Accuracy: {val_nsp_acc}')
                val_acc = (val_mlm_acc + val_nsp_acc)/2
                if val_acc > best_val_acc:
                    print('Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, 'checkpoint_testing5.pth.tar')
                    best_val_acc = val_acc
                    best_epoch = epoch

    # 3)
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_val_acc.item(), 2)}')
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
from task.reconstruction.model import kcBERT_reconstruct
from task.reconstruction.dataset import CustomDataset, PadCollate

# Import Huggingface
from transformers import AdamW

def reconstruction(args):
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
    model = kcBERT_reconstruct(vocab_size=30000)
    model = model.train()
    model = model.to(device)

    # 2) Optimizer setting
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

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
                val_loss = 0
                val_acc = 0
                model.eval()
            for i, text in enumerate(tqdm(dataloader_dict[phase])):
                # Optimizer setting
                optimizer.zero_grad()

                # Input, output setting
                text = text.to(device)
                label = text.view(-1)

                if phase == 'train':
                    with torch.set_grad_enabled(True):
                        pred = model(src_input_sentence=text)
                        loss = F.cross_entropy(pred, label.contiguous())
                        # loss = F.cross_entropy(pred, text.contiguous(), ignore_index=0)

                        # Loss backpropagation
                        loss.backward()
                        clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()

                        # Print loss value only training
                        if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                            acc = (pred.max(dim=1)[1] == label).sum() / len(label)
                            print("[Epoch:%d][%d/%d] train_loss:%5.3f  | train_acc:%5.3f | learning_rate:%3.6f | spend_time:%3.2fmin"
                                    % (epoch+1, i, len(dataloader_dict['train']), 
                                    loss.item(), acc, optimizer.param_groups[0]['lr'], 
                                    (time.time() - start_time_e) / 60))
                            freq = 0
                        freq += 1

                if phase == 'valid':
                    with torch.no_grad():
                        pred = model(src_input_sentence=text)
                    loss = F.cross_entropy(pred, label.contiguous())
                    val_loss += loss.item()
                    val_acc += (pred.max(dim=1)[1] == label).sum() / len(label)

            if phase == 'valid':
                val_loss /= len(dataloader_dict[phase])
                val_acc /= len(dataloader_dict[phase])
                print(f'Validation Loss: {val_loss}')
                print(f'Validation F1: {val_acc}')
                if val_acc > best_val_acc:
                    print('Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, 'checkpoint_testing4.pth.tar')
                    best_val_acc = val_acc
                    best_epoch = epoch

    # 3)
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_val_acc, 2)}')
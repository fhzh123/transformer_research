# Import modules
import os
import time
import pickle
from tqdm import tqdm
from sklearn.metrics import f1_score
# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
# Import custom modules
from task.classification.model import kcBERT_custom
from task.classification.dataset import CustomDataset, PadCollate

# Import Huggingface
from transformers import AdamW

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    print('Data Load & Setting!')
    with open(os.path.join(args.preprocess_path, 'processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        train_comment_indices = data_['train_comment_indices']
        valid_comment_indices = data_['valid_comment_indices']
        train_title_indices = data_['train_title_indices']
        valid_title_indices = data_['valid_title_indices']
        train_label = data_['train_label']
        valid_label = data_['valid_label']
        del data_

    dataset_dict = {
        'train': CustomDataset(train_comment_indices, train_title_indices, 
                               train_label, phase='train'),
        'valid': CustomDataset(valid_comment_indices, valid_title_indices, 
                               valid_label, phase='valid'),
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
    model = kcBERT_custom(args, num_labels=3, noise_augment=args.noise_augment, 
                          mix_augment=args.mix_augment, augment_ratio=args.augment_ratio,
                          reconstruction_feature_use=args.reconstruction_feature_use,
                          device=device)
    model = model.train()
    model = model.to(device)

    # Optimizer setting
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

    best_val_f1 = 0

    print('Train start!')

    for epoch in range(start_epoch, args.num_epochs):
        start_time_e = time.time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                print('Validation start...')
                val_loss = 0
                val_f1 = 0
                model.eval()
            for i, (comment, title, label) in enumerate(tqdm(dataloader_dict[phase])):
                # Optimizer setting
                optimizer.zero_grad()

                # Input, output setting
                comment = comment.to(device)
                label = label.to(device)

                if phase == 'train':
                    with torch.set_grad_enabled(True):
                        pred, new_label = model(src_input_sentence=comment, src_label=label)
                        pred = pred.log_softmax(dim=-1)
                        loss = torch.mean(torch.sum(-new_label.to(device) * pred, dim=-1))

                        # Loss backpropagation
                        loss.backward()
                        clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()

                        # Print loss value only training
                        if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                            total_loss = loss.item()
                            print("[Epoch:%d][%d/%d] train_loss:%5.3f  | learning_rate:%3.6f | spend_time:%3.2fmin"
                                    % (epoch+1, i, len(dataloader_dict['train']), 
                                    total_loss, optimizer.param_groups[0]['lr'], 
                                    (time.time() - start_time_e) / 60))
                            freq = 0
                        freq += 1

                if phase == 'valid':
                    with torch.no_grad():
                        pred = model(src_input_sentence=comment)
                    loss = F.cross_entropy(pred, label)
                    val_loss += loss.item()
                    val_f1 += f1_score(pred.max(dim=1)[1].tolist(), label.tolist(), average='macro')

            if phase == 'valid':
                val_loss /= len(dataloader_dict[phase])
                val_f1 /= len(dataloader_dict[phase])
                print(f'Validation Loss: {val_loss}')
                print(f'Validation F1: {val_f1}')
                if val_f1 > best_val_f1:
                    print('Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, 'checkpoint_testing3.pth.tar')
                    best_val_f1 = val_f1
                    best_epoch = epoch

    # 3)
    print(f'Best Epoch: {best_epoch}')
    print(f'Best F1-Score: {round(best_val_f1, 2)}')
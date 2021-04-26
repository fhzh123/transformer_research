# Import modules
import os
import time
import pickle
# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
# Import custom modules
from dataset import CustomDataset, PadCollate
from utils import optimizer_select, shceduler_select
# Import Huggingface
from transformers import BertForSequenceClassification, AdamW

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
        test_comment_indices = data_['test_comment_indices']
        train_label = data_['train_label']
        test_label = data_['test_label']
        del data_
    
    if args.augmentation_data_training:
        with open(os.path.join(args.preprocess_path, 'augmented_processed.pkl'), 'rb') as f:
            data_ = pickle.load(f)
            train_comment_indices = data_['augmented_comment_indices']
            train_label = data_['augmented_label']

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(train_comment_indices, train_label, 
                               min_len=args.min_len, max_len=args.max_len),
        'test': CustomDataset(test_comment_indices, test_label, 
                               min_len=args.min_len, max_len=args.max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'test': DataLoader(dataset_dict['test'], collate_fn=PadCollate(), drop_last=False,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)
    }
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    # 1) Model initiating
    print("Instantiating models...")
    model = BertForSequenceClassification.from_pretrained('bert-large-cased')
    model = model.train()
    for para in model.bert.parameters():
        para.reguires_grad = False
    model = model.to(device)

    # Optimizer setting
    # optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)

    # 2) Model resume
    start_epoch = 0
    # if args.resume:
    #     checkpoint = torch.load('./checkpoint_testing.pth.tar', map_location='cpu')
    #     start_epoch = checkpoint['epoch'] + 1
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_test_acc = 0

    print('Train start!')

    for epoch in range(start_epoch, args.num_epochs):
        start_time_e = time.time()
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            if phase == 'test':
                print('Test start...')
                test_loss = 0
                test_acc = 0
                model.eval()
            for i, batch in enumerate(dataloader_dict[phase]):
                # Optimizer setting
                optimizer.zero_grad()

                # Input, output setting
                src_seq = batch[0].to(device)
                label = batch[1].to(device)

                if phase == 'train':
                    with torch.set_grad_enabled(True):
                        out = model(src_seq, attention_mask=src_seq!=0, labels=label)
                        acc = sum(out.logits.max(dim=1)[1] == label) / len(label)

                        # Loss backpropagation
                        out.loss.backward()
                        clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                        if args.scheduler in ['warmup', 'reduce_train']:
                            scheduler.step()

                        # Print loss value only training
                        if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                            print("[Epoch:%d][%d/%d] train_loss:%3.3f  | train_acc:%3.3f | learning_rate:%3.6f | spend_time:%3.3fmin"
                                    % (epoch+1, i, len(dataloader_dict['train']), 
                                    out.loss.item(), acc.item(), optimizer.param_groups[0]['lr'], 
                                    (time.time() - start_time_e) / 60))
                            freq = 0
                        freq += 1

                if phase == 'test':
                    with torch.no_grad():
                        out = model(src_seq, attention_mask=src_seq!=0, labels=label)
                    acc = sum(out.logits.max(dim=1)[1] == label) / len(label)
                    test_loss += out.loss.item()
                    test_acc += acc.item()
                    if args.scheduler in ['reduce_valid', 'lambda']:
                        scheduler.step()

            if phase == 'test':
                test_loss /= len(dataloader_dict[phase])
                test_acc /= len(dataloader_dict[phase])
                print(f'Test Loss: {test_loss:3.3f}')
                print(f'Test Accuracy: {test_acc*100:2.2f}%')
                if test_acc > best_test_acc:
                    print('Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, 'checkpoint_testing3.pth.tar')
                    best_test_acc = test_acc
                    best_epoch = epoch

    # 3)
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_test_acc, 2)}')
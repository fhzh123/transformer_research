# Import modules
import os
import time
import pickle
import random
# Import PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
# Import custom modules
from model.Vision_Transformer import Vision_Transformer
from optimizer.utils import shceduler_select
# Import Huggingface
from transformers import AdamW

def train_epoch(args, model, dataloader, optimizer, scheduler):

    # Train setting
    start_time_e = time.time()
    model = model.train()

    for i, (img, label) in enumerate(dataloader):

        # Optimizer setting
        optimizer.zero_grad()

        # Input, output setting
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # Model
        logit = model(img)
        first_token = logit[:,0,:]

        # Loss calculate
        loss = F.cross_entropy(first_token, label)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if args.scheduler in ['constant', 'warmup']:
            scheduler.step()
        if args.scheduler == 'reduce_train':
            scheduler.step(mlm_loss)

        # Print loss value only training
        acc = (((first_token.argmax(dim=1) == label).sum()) / label.size(0)) * 100
        if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
            print("[Epoch:%d][%d/%d] train_loss:%2.3f  | train_acc:%02.2f | learning_rate:%3.6f | spend_time:%3.2fmin"
                    % (epoch+1, i, len(dataloader_dict['train']), 
                    loss.item(), acc.item(), optimizer.param_groups[0]['lr'], 
                    (time.time() - start_time_e) / 60))
            freq = 0
        freq += 1

def valid_epoch(args, model, dataloader):

    # Validation setting
    model = model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(dataloader):

            # Input, output setting
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # Model
            logit = model(img)
            first_token = logit[:,0,:]

            # Loss calculate
            loss = F.cross_entropy(first_token, label)

            # Print loss value only training
            acc = (((first_token.argmax(dim=1) == label).sum()) / label.size(0)) * 100
            val_loss += loss.item() / len(dataloader)
            val_acc += acc.item() / len(dataloader)

    return val_loss, val_acc

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Dataloader setting
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_dict = {
        'train': torchvision.datasets.CIFAR10(root='./dataset/cifar10', 
            train=True, download=True, transform=transform),
        'valid': torchvision.datasets.CIFAR10(root='./dataset/cifar10', 
            train=False, download=True, transform=transform)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    # 1) Model initiating
    print("Instantiating models...")
    model = Vision_Transformer(n_class=10, img_size=32, patch_size=16)
    model.train()
    model = model.to(device)
    print('Done!')

    # 2) Optimizer setting
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)

    # 2) Model resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.model_path, 'checkpoint.pth.tar'), map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.train()
        model = model.to(device)
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

    print('Train start!')

    for epoch in range(start_epoch, args.num_epochs):

        train_epoch(args, model, dataloader['train'], optimizer, scheduler)
        val_loss, val_acc = valid_epoch(args, model, dataloader['valid'])

    # 3)
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_val_acc.item(), 2)}')
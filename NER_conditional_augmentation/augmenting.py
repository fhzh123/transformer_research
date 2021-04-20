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
# Import custom modules
from dataset import CustomDataset, PadCollate
from model import Custom_ConditionalBERT

def augmenting(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)

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

    #

    model = Custom_ConditionalBERT(mask_id_token=103)
    model = model.to(device)
    model = model.eval()

    #===================================#
    #===========Augmentation============#
    #===================================#

    with torch.no_grad():
        for batch in dataloader_dict['train']:
            src_seq = batch[0].to(device)
            label = batch[1].to(device)

            mlm_logit, ner_masking_tensor = model(src_seq)

            i = 0 
            for n in ner_masking_tensor:
                if (n==104).sum().item() == 0:
                    continue
                else:

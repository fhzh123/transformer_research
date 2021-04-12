# Import modules
import os
import time
import pickle
import pandas as pd
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

def testing(args):
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

    # 1) Data open
    print('Data Load & Setting!')
    with open(os.path.join(args.preprocess_path, 'test_processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        test_comment_indices = data_['test_comment_indices']
        test_title_indices = data_['test_title_indices']
        del data_

    test = pd.read_csv(os.path.join(args.data_path, 'test.hate.no_label.csv'))

    # test_dataset = CustomDataset(test_comment_indices, test_title_indices, phase='test')
    # test_dataloader = DataLoader(test_dataset, collate_fn=PadCollate(), drop_last=True,
    #                              batch_size=args.batch_size, shuffle=False, pin_memory=True,
    #                              num_workers=args.num_workers)
    dataset_dict = {
        'valid': CustomDataset(valid_comment_indices, valid_title_indices, 
                               valid_label, phase='valid'),
        'test': CustomDataset(test_comment_indices, test_title_indices, phase='test'),
    }
    dataloader_dict = {
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(), drop_last=False,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'test': DataLoader(dataset_dict['test'], collate_fn=PadCollate(isTest=True), drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }

    # 1) Model initiating
    print("Instantiating models...")
    model = kcBERT_custom(num_labels=3, augment_ratio=args.augment_ratio, device=device)
    # model = model.train()
    # model = model.to(device)

    # 2) Model load
    checkpoint = torch.load('./checkpoint_testing.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    model = model.to(device)
    del checkpoint

    for phase in ['valid', 'test']:
        if phase == 'valid':
            print('Validation checking...')
            val_loss = 0
            val_f1 = 0
        if phase == 'test':
            print('Submission predicting...')
            test_predict = list()
        for i, batch_ in enumerate(tqdm(dataloader_dict[phase])):
            if phase == 'valid':
                comment = batch_[0].to(device)
                title = batch_[1].to(device)
                label = batch_[2].to(device)

            if phase == 'test':
                comment = batch_[0].to(device)
                title = batch_[1].to(device)

            with torch.no_grad():
                pred = model(src_input_sentence=comment)
                if phase == 'valid':
                    loss = F.cross_entropy(pred, label)
                    val_loss += loss.item()
                    val_f1 += f1_score(pred.max(dim=1)[1].tolist(), label.tolist(), average='macro')
                if phase == 'test':
                    test_predict.extend(pred.max(dim=1)[1].tolist())

        if phase == 'valid':
            val_loss /= len(dataloader_dict[phase])
            val_f1 /= len(dataloader_dict[phase])
            print(f'Validation Loss: {val_loss}')
            print(f'Validation F1: {val_f1}')

        if phase == 'test':
            test['label'] = test_predict
            # test.replace({'label': {0: 'none', 1: 'offensive', 2:'hate'}}, inplace=True)
            test.to_csv(os.path.join(args.preprocess_path, 'submission.csv'), index=False, encoding='utf-8-sig')
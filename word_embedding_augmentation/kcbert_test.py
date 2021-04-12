# Import modules
import os
import re
import emoji
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from soynlp.normalizer import repeat_normalize
# Import PyTorch
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
# Import Huggingface
from transformers import AdamW
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

def main(args):

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer & model load
    bert_config = BertConfig.from_pretrained('beomi/kcbert-base', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')
    bert = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', config=bert_config)
    bert = bert.to(device)

    # Data setting
    data_dict = {
        'train': preprocess_dataframe(pd.read_csv(os.path.join(args.data_path, 'train.hate.csv')), tokenizer),
        'valid': preprocess_dataframe(pd.read_csv(os.path.join(args.data_path, 'dev.hate.csv')), tokenizer),
    }
    dataset_dict = {
        'train': TensorDataset(
            torch.tensor(data_dict['train']['document'].to_list(), dtype=torch.long),
            torch.tensor(data_dict['train']['label'].to_list(), dtype=torch.long),
        ),
        'valid': TensorDataset(
            torch.tensor(data_dict['valid']['document'].to_list(), dtype=torch.long),
            torch.tensor(data_dict['valid']['label'].to_list(), dtype=torch.long),
        )
    }
    dataloader_dict = {
        'train': DataLoader(
            dataset_dict['train'], batch_size=8, shuffle=True, num_workers=4
        ),
        'valid': DataLoader(
            dataset_dict['valid'], batch_size=8, shuffle=True, num_workers=4
        )
    }

    optimizer = optim.AdamW(bert.parameters(), lr=5e-5)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.0},
        {'params': [p for n, p in bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(bert.parameters(), lr=5e-5, eps=1e-8)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    start_epoch = 0
    best_val_f1 = None

    for para in bert.bert.embeddings.parameters():
        para.requires_grad = False
    
    # for para in bert.classifier.parameters():
    #     para.requires_grad = True

    print('Train start!')

    for epoch in range(start_epoch, 10):
        for phase in ['train', 'valid']:
            if phase == 'train':
                bert.train()
            if phase == 'valid':
                print('Validation start...')
                val_loss = 0
                val_f1 = 0
                bert.eval()
            for i, (comment, label) in enumerate(tqdm(dataloader_dict[phase])):
                # Optimizer setting
                optimizer.zero_grad()

                # Input, output setting
                comment = comment.to(device)
                src_key_padding_mask = (comment != 0)
                label = label.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # pred = bert(input_ids=comment, attention_mask=src_key_padding_mask, labels=label)
                    pred = bert(input_ids=comment, attention_mask=src_key_padding_mask)
                    # loss = criterion(pred.logits, label)
                    loss = F.cross_entropy(pred.logits, label)
                    # loss = pred[0]
                    loss_p = loss.item()
                    if phase == 'train':
                        loss.backward()
                        clip_grad_norm_(bert.parameters(), 5)
                        optimizer.step()

                        if (i+1) % 100 == 0:
                            print(f'| epoch: {epoch} | lr: {optimizer.param_groups[0]["lr"]} | loss: {loss_p:.4f}')
                    if phase == 'valid':
                        val_loss += loss.item()
                        val_f1 += f1_score(pred.logits.max(dim=1)[1].tolist(), label.tolist(), average='macro')

            if phase == 'valid':
                val_loss /= len(dataloader_dict[phase])
                val_f1 /= len(dataloader_dict[phase])
                print(f'Validation Loss: {val_loss}')
                print(f'Validation F1: {val_f1}')
                if not best_val_f1 or val_f1 > best_val_f1:
                    print('Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': bert.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, 'checkpoint.pth.tar')
                    best_val_f1 = val_f1
                    best_epoch = epoch

    print(best_epoch)
    print(best_val_f1)

def preprocess_dataframe(df, tokenizer):
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    def clean(x):
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x

    df['label'] = [0 if x=='none' else 1 if x=='hate' else 2 for x in df['label']]
    df.rename(columns = {'comments' : 'document'}, inplace = True)
    df['document'] = df['document'].map(lambda x: tokenizer.encode(
        clean(str(x)),
        padding='max_length',
        max_length=200,
        truncation=True,
    ))
    return df

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Path setting
    parser.add_argument('--data_path', default='/HDD/dataset/korean-hate-speech-detection/', type=str,
                        help='Original data path')
    args = parser.parse_args()
    main(args)
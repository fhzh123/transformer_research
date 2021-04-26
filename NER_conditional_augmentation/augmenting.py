# Import modules
import os
import time
import pickle
import pandas as pd
from tqdm import tqdm
# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
# Import HuggingFace
from transformers import BertTokenizer
# Import custom modules
from dataset import CustomDataset, PadCollate
from model import Custom_ConditionalBERT
from utils import encoding_text

def augmenting(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    print('Data Load & Setting!')
    with open(os.path.join(args.preprocess_path, 'processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        train_comment_indices = data_['train_comment_indices']
        train_label = data_['train_label']
        del data_

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(train_comment_indices, train_label, 
                               min_len=args.min_len, max_len=args.max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #

    model = Custom_ConditionalBERT(mask_id_token=103, device=device)
    model = model.to(device)
    model = model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    #===================================#
    #===========Augmentation============#
    #===================================#

    augmented_dataset = pd.DataFrame()
    augmented_count = 0
    original_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader_dict['train']):
            src_seq = batch[0].to(device)
            label = batch[1].tolist()

            mlm_logit, ner_masking_tensor = model(src_seq)

            # Pre-setting
            i = 0
            old_masking_token_count = 0
            label_pop_list = list()
            augmented_tensor = torch.LongTensor([]).to(device)
            top_3_predicted = mlm_logit[ner_masking_tensor==103].topk(3, 1)[1]

            # Augmentation
            for n_i, n in enumerate(ner_masking_tensor):
                if (n==103).sum().item() == 0:
                    # label = torch.cat([label[0:n_i], label[n_i+1:]])
                    label_pop_list.append(n_i)
                    continue
                else:
                    for k in range(args.augment_top_k):
                        n_augmented = n.clone().detach()
                        masking_token_count = (n_augmented==103).sum().item()
                        for ix in (n_augmented == 103).nonzero(as_tuple=True)[0]:
                            n_augmented[ix] = top_3_predicted[i][k]
                            i += 1
                            if i == masking_token_count + old_masking_token_count:
                                i = old_masking_token_count
                        augmented_tensor = torch.cat((augmented_tensor, n_augmented.unsqueeze(0)), dim=0)
                    i += masking_token_count
                    old_masking_token_count += masking_token_count

            # Counting
            augmented_count += augmented_tensor.size(0)
            original_count += len(label_pop_list)

            # Process non NER masking sequence
            if len(label_pop_list) != 0:
                for i, original_ix in enumerate(label_pop_list):
                    if i == 0:
                        original_seq = src_seq[original_ix].unsqueeze(0)
                    else:
                        original_seq = torch.cat((original_seq, src_seq[original_ix].unsqueeze(0)), dim=0)

                # Concat
                augmented_text = tokenizer.batch_decode(augmented_tensor, skip_special_tokens=True)
                augmented_text = augmented_text + tokenizer.batch_decode(original_seq, skip_special_tokens=True)
                original_label = [value for i, value in enumerate(label) if i in label_pop_list]
                label = [i for j, i in enumerate(label) if j not in label_pop_list]
                augmented_label = [item for item in label for i in range(args.augment_top_k)]
                augmented_label = augmented_label + original_label

            # If NER_mask in none in sequence
            else:
                augmented_text = tokenizer.batch_decode(augmented_tensor, skip_special_tokens=True)
                label = [i for j, i in enumerate(label) if j not in label_pop_list]
                augmented_label = [item for item in label for i in range(args.augment_top_k)]

            new_dat = pd.DataFrame({
                'comment': augmented_text,
                'sentiment': augmented_label
            })
            augmented_dataset = pd.concat([augmented_dataset, new_dat], axis=0)

    print(f'Augmented data size: {augmented_count}')
    print(f'Non NER_Masking data size: {original_count}')
    print(f'Total data size: {augmented_dataset.shape[0]}')
    augmented_dataset.to_csv(os.path.join(args.preprocess_path, 'augmented_train.csv'), index=False)

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Cleansing...')

    # 1) Cleansing
    augmented_dataset['comment'] = encoding_text(augmented_dataset['comment'], 
                                                 tokenizer, args.max_len)

    # 2) Training pikcle saving
    with open(os.path.join(args.preprocess_path, 'augmented_processed.pkl'), 'wb') as f:
        pickle.dump({
            'augmented_comment_indices': augmented_dataset['comment'].tolist(),
            'augmented_label': augmented_dataset['sentiment'].tolist(),
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')
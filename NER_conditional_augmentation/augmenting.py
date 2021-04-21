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

def augmenting(args):
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

    model = Custom_ConditionalBERT(mask_id_token=103, device=device)
    model = model.to(device)
    model = model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    #===================================#
    #===========Augmentation============#
    #===================================#

    augmented_dataset = pd.DataFrame()

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
                    for k in range(3):
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

            augmented_text = tokenizer.batch_decode(augmented_tensor, skip_special_tokens=True)
            label = [i for j, i in enumerate(label) if j not in label_pop_list]
            augmented_label = [item for item in label for i in range(3)]
            new_dat = pd.DataFrame({
                'comment': augmented_text,
                'sentiment': augmented_label
            })
            augmented_dataset = pd.concat([augmented_dataset, new_dat], axis=0)

    augmented_dataset.to_csv(os.path.join(args.preprocess_path, 'augmented_train.csv'), index=False)

    #===================================#
    #==============Saving===============#
    #===================================#

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
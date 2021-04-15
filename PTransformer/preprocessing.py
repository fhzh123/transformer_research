# Import Modules
import os
import json
import time
import pickle
import argparse
import numpy as np
import sentencepiece as spm
from glob import glob
from tqdm import tqdm
from collections import Counter

def preprocessing(args):

    #===================================#
    #============Data Load==============#
    #===================================#

    print('Data Load...')
    # 1) Train data load
    with open(os.path.join(args.data_path, 'train.de'), 'r') as f:
        train_src_sequences = f.readlines()
    with open(os.path.join(args.data_path, 'train.de'), 'r') as f:
        train_trg_sequences = f.readlines()
    # 1) Valid data load
    with open(os.path.join(args.data_path, 'train.de'), 'r') as f:
        train_src_sequences = f.readlines()
    with open(os.path.join(args.data_path, 'train.de'), 'r') as f:
        train_trg_sequences = f.readlines()

    #===================================#
    #=======Hanja Pre-processing========#
    #===================================#

    print('Hanja sentence parsing...')
    start_time = time.time()

    if args.src_baseline:
        ind_set, hj_word2id = sentencepiece_training('hanja', split_src_record, args)
        train_hanja_indices = ind_set[0]
        valid_hanja_indices = ind_set[1]
        test_hanja_indices = ind_set[2]
    else:
        with open(os.path.join(args.save_path, 'hj_word2id.pkl'), 'rb') as f:
            hj_word2id = pickle.load(f)['hanja_word2id']

        # 1) Hanja sentence parsing setting
        train_hanja_indices = list()
        valid_hanja_indices = list()
        test_hanja_indices = list()

        # 2) Parsing sentence

        word_counter = Counter()
        hanja_word2id = ['<pad>', '<s>', '</s>', '<unk>']
        # Hanja Counter
        for sentence in split_src_record['train']:
            for word in sentence:
                word_counter.update(word)

        hanja_word2id.extend([w for w, freq in word_counter.items() if w in hj_word2id.keys() and freq >= 10])
        hj_word2id = {w: i for i, w in enumerate(hanja_word2id)}

        # 3-1) Train & valid & test data parsing (From utils.py)

        print('Train data start...')
        train_hanja_indices = hj_encode_to_ids(split_src_record['train'], hj_word2id, args)

        print('Valid data start...')
        valid_hanja_indices = hj_encode_to_ids(split_src_record['valid'], hj_word2id, args)

        print('Test data start...')
        test_hanja_indices = hj_encode_to_ids(split_src_record['test'], hj_word2id, args)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #=======Korean Pre-processing=======#
    #===================================#

    ind_set, kr_word2id = sentencepiece_training('korean', split_trg_record, args)
    
    train_korean_indices = ind_set[0]
    valid_korean_indices = ind_set[1]
    test_korean_indices = ind_set[2]

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Parsed sentence save setting...')
    start_time = time.time()

    with open(os.path.join(args.save_path, 'nmt_processed.pkl'), 'wb') as f:
        pickle.dump({
            'hj_train_indices': train_hanja_indices,
            'hj_valid_indices': valid_hanja_indices,
            'hj_test_indices': test_hanja_indices,
            'kr_train_indices': train_korean_indices,
            'kr_valid_indices': valid_korean_indices,
            'kr_test_indices': test_korean_indices,
            'king_train_indices': split_king_record['train'],
            'king_valid_indices': split_king_record['valid'],
            'king_test_indices': split_king_record['test'],
            'hj_word2id': hj_word2id,
            'kr_word2id': kr_word2id,
            'hj_id2word': {v: k for k, v in hj_word2id.items()},
            'kr_id2word': {v: k for k, v in kr_word2id.items()}
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')
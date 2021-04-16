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
        train_src_sequences = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'train.en'), 'r') as f:
        train_trg_sequences = [x.replace('\n', '') for x in f.readlines()]
    # 2) Valid data load
    with open(os.path.join(args.data_path, 'val.de'), 'r') as f:
        valid_src_sequences = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'val.en'), 'r') as f:
        valid_trg_sequences = [x.replace('\n', '') for x in f.readlines()]

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    print('SentencePiece Training')
    start_time = time.time()

    # 1) Source lanugage
    # Make text to train vocab
    with open(f'{args.preprocess_path}/src.txt', 'w') as f:
        for text in train_src_sequences:
            f.write(f'{text}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.preprocess_path}/text.txt --model_prefix={args.preprocess_path}/m_src_{args.src_vocab_size} '
        f'--vocab_size={args.src_vocab_size} --character_coverage=0.9995 --split_by_whitespace=true '
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--model_type={args.sentencepiece_model}')

    src_vocab = list()
    with open(f'{args.preprocess_path}/m_hanja_{args.vocab_size}.vocab') as f:
        for line in f:
            src_vocab.append(line[:-1].split('\t')[0])

    src_word2id = {w: i for i, w in enumerate(src_vocab)}

    spm_src = spm.SentencePieceProcessor()
    spm_src.Load(f'{args.preprocess_path}/m_src_{args.vocab_size}.model')

    train_src_indices = [spm_src.EncodeAsIds(['<eos>'] + text + ['<eos>']) for text in train_src_sequences]
    valid_src_indices = [spm_src.EncodeAsIds(['<eos>'] + text + ['<eos>']) for text in valid_src_sequences]

    # 2) Target lanugage
    # Make text to train vocab
    with open(f'{args.preprocess_path}/trg.txt', 'w') as f:
        for text in train_trg_sequences:
            f.write(f'{text}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.preprocess_path}/text.txt --model_prefix={args.preprocess_path}/m_trg_{args.trg_vocab_size} '
        f'--vocab_size={args.trg_vocab_size} --character_coverage=0.9995 --split_by_whitespace=true '
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--model_type={args.sentencepiece_model}')

    trg_vocab = list()
    with open(f'{args.preprocess_path}/m_hanja_{args.vocab_size}.vocab') as f:
        for line in f:
            trg_vocab.append(line[:-1].split('\t')[0])

    trg_word2id = {w: i for i, w in enumerate(trg_vocab)}

    spm_trg = spm.SentencePieceProcessor()
    spm_trg.Load(f'{args.preprocess_path}/m_trg_{args.vocab_size}.model')

    train_trg_indices = [spm_trg.EncodeAsIds(['<eos>'] + text + ['<eos>']) for text in train_trg_sequences]
    valid_trg_indices = [spm_trg.EncodeAsIds(['<eos>'] + text + ['<eos>']) for text in valid_trg_sequences]

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Parsed sentence save setting...')
    start_time = time.time()

    with open(os.path.join(args.preprocess_path, 'processed.pkl'), 'wb') as f:
        pickle.dump({
            'train_src_indices': train_src_indices,
            'valid_src_indices': valid_src_indices,
            'train_trg_indices': train_trg_indices,
            'valid_trg_indices': valid_trg_indices,
            'src_word2id': src_word2id,
            'trg_word2id': trg_word2id
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')
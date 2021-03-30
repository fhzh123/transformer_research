import os
import time
import pickle
import pandas as pd
import sentencepiece as spm

def preprocessing(args):

    start_time = time.time()

    #===================================#
    #=============Data Load=============#
    #===================================#

    # 1) Comment data open
    train = pd.read_csv(os.path.join(args.data_path, 'train.hate.csv'))
    valid = pd.read_csv(os.path.join(args.data_path, 'dev.hate.csv'))
    test = pd.read_csv(os.path.join(args.data_path, 'test.hate.no_label.csv'))

    with open(os.path.join(args.data_path, 'train.news_title.txt'), 'r') as f:
        train_title = f.readlines()
        train_title = [x.replace('\n','') for x in train_title]
    with open(os.path.join(args.data_path, 'dev.news_title.txt'), 'r') as f:
        valid_title = f.readlines()
        valid_title = [x.replace('\n','') for x in valid_title]
    with open(os.path.join(args.data_path, 'test.news_title.txt'), 'r') as f:
        test_title = f.readlines()
        test_title = [x.replace('\n','') for x in test_title]

    # 2) Path setting
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    #===================================#
    #===========SentencePiece===========#
    #===================================#

    # 1) Make text to train vocab
    total_train_text = train['comments'].tolist() + train_title

    with open(f'{args.save_path}/input.txt', 'w') as f:
        for text in total_train_text:
            f.write(f'{text}\n')

    # 2) SentencePiece model training
    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.save_path}/input.txt --model_prefix={args.save_path}/m_model --model_type=bpe '
        f'--vocab_size={args.vocab_size} --character_coverage=0.995 --split_by_whitespace=true '
        f'--pad_id={args.pad_idx} --unk_id={args.unk_idx} --bos_id={args.bos_idx} --eos_id={args.eos_idx} '
        f'--user_defined_symbols=[SEP],[P]')

    # 3) Vocabulary setting
    vocab = list()
    with open(f'{args.save_path}/m_model.vocab') as f:
        for line in f:
            vocab.append(line[:-1].split('\t')[0])
    word2id = {w: i for i, w in enumerate(vocab)}

    # 4) SentencePiece model load
    spm_ = spm.SentencePieceProcessor()
    spm_.Load(f"{args.save_path}/m_model.model")

    # 5) Parsing by SentencePiece model

    # 5-1) Comment parsing
    train_indices = [[args.bos_idx] + spm_.EncodeAsIds(text) + [args.eos_idx] for text in train['comments']]
    valid_indices = [[args.bos_idx] + spm_.EncodeAsIds(text) + [args.eos_idx] for text in valid['comments']]
    test_indices = [[args.bos_idx] + spm_.EncodeAsIds(text) + [args.eos_idx] for text in test['comments']]

    # 5-2) Title parsing
    train_title_indices = [[args.bos_idx] + spm_.EncodeAsIds(text) + [args.eos_idx] for text in train_title]
    valid_title_indices = [[args.bos_idx] + spm_.EncodeAsIds(text) + [args.eos_idx] for text in valid_title]
    test_title_indices = [[args.bos_idx] + spm_.EncodeAsIds(text) + [args.eos_idx] for text in test_title]

    # 5-3) Total parsing
    train_total_indices = [[args.bos_idx] + text1[1:] + [word2id['[SEP]']] + text2[:-1] + [args.eos_idx] \
                           for text1, text2 in zip(train_indices, train_title_indices)]
    valid_total_indices = [[args.bos_idx] + text1[1:] + [word2id['[SEP]']] + text2[:-1] + [args.eos_idx] \
                           for text1, text2 in zip(valid_indices, valid_title_indices)]
    test_total_indices = [[args.bos_idx] + text1[1:] + [word2id['[SEP]']] + text2[:-1] + [args.eos_idx] \
                          for text1, text2 in zip(test_indices, test_title_indices)]

    #===================================#
    #==============Saving===============#
    #===================================#

    # 1) Print status
    print('Parsed sentence save setting...')

    max_train_len = max([len(x) for x in train_indices])
    max_valid_len = max([len(x) for x in valid_indices])
    max_test_len = max([len(x) for x in test_indices])

    max_train_title_len = max([len(x) for x in train_title_indices])
    max_valid_title_len = max([len(x) for x in valid_title_indices])
    max_test_title_len = max([len(x) for x in test_title_indices])

    max_train_total_len = max([len(x) for x in train_total_indices])
    max_valid_total_len = max([len(x) for x in valid_total_indices])
    max_test_total_len = max([len(x) for x in test_total_indices])
    
    print(f'Train data max length => title: {max_train_len} | comment: {max_train_title_len} | total: {max_train_total_len}')
    print(f'Valid data max length => title: {max_valid_len} | comment: {max_valid_title_len} | total: {max_valid_total_len}')
    print(f'Test data max length => title: {max_test_len} | comment: {max_test_title_len} | total: {max_test_total_len}')

    # 2) Saving

    with open(os.path.join(args.save_path, 'processed.pkl'), 'wb') as f:
        pickle.dump({
            'train_indices': train_indices,
            'valid_indices': valid_indices,
            'test_indices': test_indices,
            'train_title_indices': train_title_indices,
            'valid_title_indices': valid_title_indices,
            'test_title_indices': test_title_indices,
            'train_total_indices': train_total_indices,
            'valid_total_indices': valid_total_indices,
            'test_total_indices': test_total_indices,
            'train_label': train['label'],
            'valid_label': valid['label'],
            'word2id': word2id,
            'id2word': {v: k for k, v in word2id.items()}
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')
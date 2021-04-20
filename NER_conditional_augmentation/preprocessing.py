import os
import re
import time
import emoji
import pickle
import pandas as pd
import sentencepiece as spm
# Import Huggingface
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

def preprocessing(args):

    start_time = time.time()

    print('Start preprocessing!')

    #===================================#
    #=============Data Load=============#
    #===================================#

    # 1) Comment data open
    train = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(args.data_path, 'test.csv'))

    # 2) Path setting
    if not os.path.exists(args.preprocess_path):
        os.mkdir(args.preprocess_path)

    #===================================#
    #=============Tokenizer=============#
    #===================================#

    print('Tokenizer setting...')

    # 1) Tokenizer open
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    #===================================#
    #=============Cleansing=============#
    #===================================#

    print('Cleansing...')

    # 1) Regular expression compile
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(r'<[^>]+>')

    # 2) Definition clean
    def clean(x):
        x = pattern.sub(' ', x)
        x = x.strip()
        return x
    
    def encoding_text(list_x, tokenizer):
        encoded_text_list = list_x.map(lambda x: tokenizer.encode(
            clean(str(x)),
            max_length=args.max_len,
            truncation=True
        ))
        return encoded_text_list

    # 3) Preprocess comments
    train['comment'] = encoding_text(train['comment'], tokenizer)
    test['comment'] = encoding_text(test['comment'], tokenizer)

    #===================================#
    #==========Label processing=========#
    #===================================#

    print('Label processing...')

    train.replace({'sentiment': {'positive': 0, 'negative': 1}}, inplace=True)
    test.replace({'sentiment': {'positive': 0, 'negative': 1}}, inplace=True)

    #===================================#
    #==============Saving===============#
    #===================================#

    # 1) Print status
    print('Parsed sentence save setting...')

    max_train_len = max([len(x) for x in train['comment']])
    max_test_len = max([len(x) for x in test['comment']])
    mean_train_len = sum([len(x) for x in train['comment']]) / len(train['comment'])
    mean_test_len = sum([len(x) for x in test['comment']]) / len(test['comment'])

    print(f'Train data max length => comment: {max_train_len}')
    print(f'Train data mean length => comment: {mean_train_len}')
    print(f'Test data max length => comment: {max_test_len}')
    print(f'Test data mean length => comment: {mean_test_len}')

    # 2) Training pikcle saving
    with open(os.path.join(args.preprocess_path, 'processed.pkl'), 'wb') as f:
        pickle.dump({
            'train_comment_indices': train['comments'].tolist(),
            'test_comment_indices': test['comments'].tolist(),
            'train_label': train['sentiment'].tolist(),
            'test_label': test['sentiment'].tolist()
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')
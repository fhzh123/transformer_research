import os
import re
import time
import emoji
import pickle
import pandas as pd
import sentencepiece as spm
from soynlp.normalizer import repeat_normalize
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
    train = pd.read_csv(os.path.join(args.data_path, 'train.hate.csv'))
    valid = pd.read_csv(os.path.join(args.data_path, 'dev.hate.csv'))
    test = pd.read_csv(os.path.join(args.data_path, 'test.hate.no_label.csv'))

    # 2) Title data open
    with open(os.path.join(args.data_path, 'train.news_title.txt'), 'r') as f:
        train_title = [x.replace('\n','') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'dev.news_title.txt'), 'r') as f:
        valid_title = [x.replace('\n','') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'test.news_title.txt'), 'r') as f:
        test_title = [x.replace('\n','') for x in f.readlines()]

    # 3) Unlabeld data open
    if args.unlabeled_data_processing:
        unlabel_title = pd.read_csv(os.path.join(args.data_path, 'unlabeled_comments.news_title.txt'), 
                                    names=['title'])
        unlabel_comments = pd.read_csv(os.path.join(args.data_path, 'unlabeled_comments.txt'), 
                                    names=['comments'])

    # 4) Path setting
    if not os.path.exists(args.preprocess_path):
        os.mkdir(args.preprocess_path)

    #===================================#
    #=============Tokenizer=============#
    #===================================#

    print('Tokenizer setting...')

    # 1) Tokenizer open
    if args.custom_training_tokenizer:
        tokenizer = BertWordPieceTokenizer(lowercase=False)
        with open(os.path.join(args.preprocessing_path, 'unlabeld.txt'),'w') as f:
            for i in range(len(unlabel_title)):
                f.write(unlabel_title.tolist()[i])
                f.write('\n')
                f.write(unlabel_comments.tolist()[i])
                f.write('\n')
        tokenizer.train([os.path.join(args.preprocessing_path, 'unlabeld.txt')], 
                        vocab_size=args.vocab_size, limit_alphabet=args.limit_alphabet)
        tokenizer.save_model(args.preprocessing_path)
    else:
        tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')

    #===================================#
    #=============Cleansing=============#
    #===================================#

    print('Cleansing...')

    # 1) Regular expression compile
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    # 2) Definition clean
    def clean(x):
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x
    
    def encoding_text(list_x, tokenizer):
        encoded_text_list = list_x.map(lambda x: tokenizer.encode(
            clean(str(x)),
            max_length=args.max_len,
            truncation=True
        ))
        return encoded_text_list

    # 3) Preprocess comments
    train['comments'] = encoding_text(train['comments'], tokenizer)
    valid['comments'] = encoding_text(valid['comments'], tokenizer)
    test['comments'] = encoding_text(test['comments'], tokenizer)

    # 4) Title parsing
    train['title'] = encoding_text(pd.Series(train_title), tokenizer)
    valid['title'] = encoding_text(pd.Series(valid_title), tokenizer)
    test['title'] = encoding_text(pd.Series(test_title), tokenizer)

    # 5) Unlabel data parsing
    if args.unlabeled_data_processing:
        unlabel_title = encoding_text(unlabel_title['title'], tokenizer)
        unlabel_comments = encoding_text(unlabel_comments['comments'], tokenizer)

    #===================================#
    #==========Label processing=========#
    #===================================#

    print('Label processing...')

    train.replace({'label': {'none': 0, 'offensive': 1, 'hate': 2}}, inplace=True)
    valid.replace({'label': {'none': 0, 'offensive': 1, 'hate': 2}}, inplace=True)

    #===================================#
    #==============Saving===============#
    #===================================#

    # 1) Print status
    print('Parsed sentence save setting...')

    max_train_len = max([len(x) for x in train['comments']])
    max_valid_len = max([len(x) for x in valid['comments']])
    max_test_len = max([len(x) for x in test['comments']])

    max_train_title_len = max([len(x) for x in train['title']])
    max_valid_title_len = max([len(x) for x in valid['title']])
    max_test_title_len = max([len(x) for x in test['title']])

    if args.unlabeled_data_processing:
        max_unlabel_title_len = max([len(x) for x in unlabel_title])
        max_unlabel_comments_len = max([len(x) for x in unlabel_comments])
    
    print(f'Train data max length => title: {max_train_len} | comment: {max_train_title_len}', end=' | ')
    print(f'total: {max_train_len + max_train_title_len}')
    print(f'Valid data max length => title: {max_valid_len} | comment: {max_valid_title_len}', end=' | ')
    print(f'total: {max_valid_len + max_valid_title_len}')
    print(f'Test data max length => title: {max_test_len} | comment: {max_test_title_len}', end=' | ')
    print(f'total: {max_test_len + max_test_title_len}')
    if args.unlabeled_data_processing:
        print(f'Unlabel data max length => title: {max_unlabel_title_len} | comment: {max_unlabel_comments_len}', end=' | ')
        print(f'total: {max_unlabel_title_len + max_unlabel_comments_len}')

    # 2) Training pikcle saving
    with open(os.path.join(args.preprocess_path, 'processed.pkl'), 'wb') as f:
        pickle.dump({
            'train_comment_indices': train['comments'].tolist(),
            'valid_comment_indices': valid['comments'].tolist(),
            'train_title_indices': train['title'].tolist(),
            'valid_title_indices': valid['title'].tolist(),
            'train_label': train['label'].tolist(),
            'valid_label': valid['label'].tolist()
        }, f)

    # 3) Test pickle saving
    with open(os.path.join(args.preprocess_path, 'test_processed.pkl'), 'wb') as f:
        pickle.dump({
            'test_comment_indices': test['comments'].tolist(),
            'test_title_indices': test['title'].tolist(),
        }, f)

    # 4) Unlabeled pickle saving
    if args.unlabeled_data_processing:
        with open(os.path.join(args.preprocess_path, 'unlabeled_processed.pkl'), 'wb') as f:
            pickle.dump({
                'unlabel_title': unlabel_title,
                'unlabel_comments': unlabel_comments,
            }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')
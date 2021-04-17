

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    print('Data Load & Setting!')
    with open(os.path.join(args.save_path, 'ner_processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        test_indices = data_['test_indices']
        test_title_indices = data_['test_title_indices']
        test_total_indices = data_['test_total_indices']
        word2id = data_['word2id']
        id2word = data_['id2word']
        del data_

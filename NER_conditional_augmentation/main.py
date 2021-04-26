# Import modules
import time
import argparse

# Import custom modules
from preprocessing import preprocessing
from augmenting import augmenting
from train import training
# from test import testing

def main(args):
    # Time setting
    total_start_time = time.time()

    preprocessing
    if args.preprocessing:
        preprocessing(args)

    # Augmentation by NER_Masking
    if args.augmenting:
        augmenting(args)

    # training
    if args.training:
        training(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--augmenting', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path setting
    parser.add_argument('--data_path', default='/HDD/kyohoon/imdb_classification', type=str,
                        help='Original data path')
    parser.add_argument('--preprocess_path', default='./preprocessing', type=str,
                        help='Preprocessed data  file path')
    parser.add_argument('--save_path', default='/HDD/kyohoon/model_checkpoint/hate_speech/', type=str,
                        help='Model checkpoint file path')
    # Preprocessing setting
    parser.add_argument('--vocab_size', default=30000, type=int, 
                        help='Vocabulary size; Default is 30000')
    parser.add_argument('--min_len', default=4, type=int, 
                        help='Minimum Length of Source Sentence; Default is 4')
    parser.add_argument('--max_len', default=150, type=int, 
                        help='Maximum Length of Source Sentence; Default is 150')
    # Augmentation setting
    parser.add_argument('--augment_top_k', default=3, type=int,
                        help='Augmented size of NER_Masking; Default is 3')
    # Training setting
    parser.add_argument('--augmentation_data_training', action='store_true')
    parser.add_argument('--num_epochs', default=10, type=int, 
                        help='Epoch count; Default is 10')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Num CPU Workers; Default is 4')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size; Default is 16')
    parser.add_argument('--dropout', default=0.5, type=float, 
                        help='Dropout ratio; Default is 0.5')
    parser.add_argument('--embedding_dropout', default=0.3, type=float, 
                        help='Embedding dropout ratio; Default is 0.3')
    parser.add_argument('--lr', default=5e-5, type=float, 
                        help='Learning rate; Default is 5e-5')
    parser.add_argument('--w_decay', default=5e-6, type=float, 
                        help='Weight decay ratio; Default is 5e-6')
    parser.add_argument('--grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')
    # Optimizer setting
    optim_list = ['AdamW', 'Adam', 'SGD']
    scheduler_list = ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'AdamW', 'Adam', 'SGD'; Default is AdamW")
    parser.add_argument('--scheduler', default='constant', type=str, choices=scheduler_list,
                        help="Choose scheduler setting in 'constant', 'warmup', 'reduce'; Default is constant")
    parser.add_argument('--momentum', default=0.9, type=float,
                        help="SGD's momentum; Default is 0.9")
    parser.add_argument('--n_warmup_epochs', default=2, type=int, 
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")
    # Print frequency
    parser.add_argument('--print_freq', default=100, type=int, help='Print training process frequency; Default is 100')
    args = parser.parse_args()

    main(args)
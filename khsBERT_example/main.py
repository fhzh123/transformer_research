# Import modules
import time
import argparse

# Import custom modules
from preprocessing import preprocessing
from task.reconstruction.reconstruction import reconstruction
from task.pretrain.pretrain import pretraining
from task.classification.train import training
from test import testing

def main(args):
    # Time setting
    total_start_time = time.time()

    # preprocessing
    # if args.preprocessing:
    #     preprocessing(args)

    # reconstruction
    if args.reconstruction:
        reconstruction(args)

    if args.pretraining:
        pretraining(args)

    # training
    if args.training:
        training(args)

    # testing
    # if args.testing:
    #     testing(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--pretraining', action='store_true')
    parser.add_argument('--reconstruction', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path setting
    parser.add_argument('--data_path', default='/HDD/dataset/korean-hate-speech-detection/', type=str,
                        help='Original data path')
    parser.add_argument('--preprocess_path', default='./preprocessing', type=str,
                        help='Preprocessed data  file path')
    parser.add_argument('--save_path', default='/HDD/kyohoon/model_checkpoint/hate_speech/', type=str,
                        help='Model checkpoint file path')
    # Preprocessing setting
    parser.add_argument('--vocab_size', default=30000, type=int, help='Vocabulary size; Default is 30000')
    parser.add_argument('--max_len', default=150, type=int, help='Max Length of Source Sentence; Default is 150')
    # Training setting
    parser.add_argument('--num_epochs', default=10, type=int, help='Epoch count; Default is 10')
    parser.add_argument('--num_workers', default=8, type=int, help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size; Default is 16')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout ratio; Default is 0.5')
    parser.add_argument('--embedding_dropout', default=0.3, type=float, help='Embedding dropout ratio; Default is 0.3')
    parser.add_argument('--lr', default=5e-5, type=float, help='Learning rate; Default is 5e-5')
    parser.add_argument('--w_decay', default=5e-6, type=float, help='Weight decay ratio; Default is 5e-6')
    parser.add_argument('--grad_norm', default=5, type=int, help='Graddient clipping norm; Default is 5')
    # Custom setting
    parser.add_argument('--augment_ratio', default=0.2, type=float, help='Augmented ration; Default is 0.2')
    parser.add_argument('--custom_training_tokenizer', action='store_true')
    parser.add_argument('--unlabeled_data_processing', action='store_true')
    parser.add_argument('--noise_augment', action='store_true')
    parser.add_argument('--mix_augment', action='store_true')
    parser.add_argument('--split_ratio', default=0.2, type=float)
    parser.add_argument('--reconstruction_feature_use', action='store_true')
    # Print frequency
    parser.add_argument('--print_freq', default=100, type=int, help='Print training process frequency; Default is 100')
    args = parser.parse_args()

    main(args)
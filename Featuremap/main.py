# Import modules
import time
import argparse

# Import custom modules
from preprocessing import preprocessing
from train import training

def main(args):
    # Time setting
    total_start_time = time.time()

    # preprocessing
    if args.preprocessing:
        preprocessing(args)

    # training
    if args.training:
        training(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path setting
    parser.add_argument('--data_path', default='/HDD/dataset/korean-hate-speech-detection/', type=str,
                        help='Original data path')
    parser.add_argument('--save_path', default='./preprocessing', type=str,
                        help='Preprocessed data & Model checkpoint file path')
    # Preprocessing setting
    parser.add_argument('--vocab_size', default=24000, type=int, help='Vocabulary size; Default is 24000')
    parser.add_argument('--pad_idx', default=0, type=int, help='pad index')
    parser.add_argument('--bos_idx', default=1, type=int, help='index of bos token')
    parser.add_argument('--eos_idx', default=2, type=int, help='index of eos token')
    parser.add_argument('--unk_idx', default=3, type=int, help='index of unk token')
    parser.add_argument('--max_len', default=150, type=int, help='Max Length of Source Sentence; Default is 150')
    # Model setting
    parser.add_argument('--d_model', default=768, type=int, help='Model dimension; Default is 768')
    parser.add_argument('--d_embedding', default=256, type=int, help='Embedding dimension; Default is 256')
    parser.add_argument('--n_head', default=12, type=int, help='Mutlihead count; Default is 12')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='Feedforward layer dimension; Default is 2048')
    parser.add_argument('--n_layers', default=16, type=int, help='Layer count; Default is 16')
    # Training setting
    parser.add_argument('--num_epochs', default=30, type=int, help='Epoch count; Default is 30')
    parser.add_argument('--num_workers', default=8, type=int, help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size; Default is 16')
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout ratio; Default is 0.3')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate; Default is 1e-3')
    parser.add_argument('--w_decay', default=5e-4, type=float, help='Weight decay ratio; Default is 5e-4')
    parser.add_argument('--grad_norm', default=5, type=int, help='Graddient clipping norm; Default is 5')
    # Print frequency
    parser.add_argument('--print_freq', default=100, type=int, help='Print training process frequency; Default is 100')
    args = parser.parse_args()

    main(args)
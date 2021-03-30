# Import modules
import time
import argparse

# Import custom modules
from preprocessing import preprocessing

def main(args):
    # Time setting
    total_start_time = time.time()

    # preprocessing
    if args.preprocessing:
        preprocessing(args)

    # # training
    # if args.training:
    #     training(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
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
    args = parser.parse_args()

    main(args)
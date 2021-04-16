import os
import argparse
from time import time

# Import custom modules
from task.preprocessing import preprocessing
from task.training import training
from utils import str2bool

def main(args):

    # Time setting
    total_start_time = time()

    if args.preprocessing:
        preprocessing(args)

    if args.training:
        training(args)


    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    # Path setting
    parser.add_argument('--data_path', default='/HDD/dataset/korean-hate-speech-detection/', type=str,
                        help='Original data path')
    parser.add_argument('--preprocess_path', default='./preprocessing', type=str,
                        help='Preprocessed data  file path')
    parser.add_argument('--save_path', default='/HDD/kyohoon/model_checkpoint/hate_speech/', type=str,
                        help='Model checkpoint file path')
    # Preprocessing setting
    parser.add_argument('--sentencepiece_model', default='unigram', choices=['unigram', 'bpe', 'word', 'char'],
                        help="Google's SentencePiece model type; Default is unigram")
    parser.add_argument('--src_vocab_size', default=30000, type=int, 
                        help='Source language vocabulary size; Default is 30000')
    parser.add_argument('--trg_vocab_size', default=30000, type=int, 
                        help='Target language vocabulary size; Default is 30000')
    parser.add_argument('--pad_id', default=0, type=int,
                        help='Padding token index; Default is 0')
    parser.add_argument('--unk_id', default=0, type=int,
                        help='Unknown token index; Default is 3')
    parser.add_argument('--bos_id', default=0, type=int,
                        help='Padding token index; Default is 0')
    parser.add_argument('--eos_id', default=0, type=int,
                        help='Padding token index; Default is 0')
    # Training setting
    parser.add_argument('--min_len', default=4, type=int,
                        help='Minimum length of sequences; Default is 4')
    parser.add_argument('--src_max_len', default=0, type=int,
                        help='Minimum length of source sequences; Default is 300')
    parser.add_argument('--trg_max_len', default=0, type=int,
                        help='Minimum length of target sequences; Default is 300')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size; Default is 16')
    parser.add_argument('--num_epochs', default=10, type=int, 
                        help='Epoch count; Default is 10')
    # Model setting
    parser.add_argument('--parallel', default=True, type=str2bool,
                        help='PTransformer option; Default is True')
    parser.add_argument('--d_model', default=768, type=int, 
                        help='Transformer model dimension; Default is 768')
    parser.add_argument('--d_embedding', default=256, type=int, 
                        help='Transformer embedding word token dimension; Default is 256')
    parser.add_argument('--n_head', default=12, type=int, 
                        help="Multihead Attention's head count; Default is 12")
    parser.add_argument('--d_k', default=64, type=int, 
                        help="Transformer's key dimension; Default is 64")
    parser.add_argument('--d_v', default=64, type=int, 
                        help="Transformer's value dimension; Default is 64")
    parser.add_argument('--dim_feedforward', default=2048, type=int, 
                        help="Feedforward network's dimension; Default is 2048")
    parser.add_argument('--dropout', default=0.3, type=float, 
                        help="Dropout ration; Default is 0.3")
    parser.add_argument('--embedding_dropout', default=0.1, type=float, 
                        help="Embedding dropout ration; Default is 0.1")
    parser.add_argument('--n_common_layers', default=6, type=int, 
                        help="In PTransformer, parallel layer count; Default is 6")
    parser.add_argument('--n_encoder_layers', default=6, type=int, 
                        help="Number of encoder layers; Default is 6")
    parser.add_argument('--n_decoder_layers', default=6, type=int, 
                        help="Number of decoder layers; Default is 6")
    parser.add_argument('--trg_emb_prj_weight_sharing', default=False, type=str2bool, 
                        help="Share weight between target embedding & last dense layer; Default is False")
    parser.add_argument('--emb_src_trg_weight_sharing', default=True, type=str2bool, 
                        help="Share weight between source and target embedding; Default is True")

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
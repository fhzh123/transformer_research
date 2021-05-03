# Import modules
import os
import gc
import time
import pickle
import logging
import sentencepiece as spm
from tqdm import tqdm
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu

# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.transformer import Transformer
from model.dataset import CustomDataset
from utils import label_smoothing_loss, TqdmLoggingHandler, write_log

def testing(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()
    with open(os.path.join(args.preprocess_path, 'test_processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        test_src_indices = data_['test_src_indices']
        test_trg_indices = data_['test_trg_indices']
        src_word2id = data_['src_word2id']
        trg_word2id = data_['trg_word2id']
        trg_id2word = {v: k for k, v in trg_word2id.items()}
        src_vocab_num = len(src_word2id)
        trg_vocab_num = len(trg_word2id)
        del data_
    gc.enable()
    write_log(logger, "Finished loading data!")

    # 2) Dataloader setting
    test_dataset = CustomDataset(test_src_indices, test_trg_indices,
                                 min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len)
    test_dataloader = DataLoader(test_dataset, drop_last=False, batch_size=args.batch_size, shuffle=False,
                                 pin_memory=True, num_workers=args.num_workers)
    write_log(logger, f"Total number of trainingsets  iterations - {len(test_dataset)}, {len(test_dataloader)}")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    model = Transformer(src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                        pad_idx=args.pad_id, bos_idx=args.bos_id, eos_idx=args.eos_id,
                        d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
                        dim_feedforward=args.dim_feedforward,
                        num_common_layer=args.num_common_layer, num_encoder_layer=args.num_encoder_layer,
                        num_decoder_layer=args.num_decoder_layer,
                        src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                        dropout=args.dropout, embedding_dropout=args.embedding_dropout,
                        trg_emb_prj_weight_sharing=args.trg_emb_prj_weight_sharing,
                        emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing, 
                        parallel=args.parallel)
    tgt_mask = model.generate_square_subsequent_mask(args.trg_max_len - 1, device)

    # loda model
    model = model.to(device)
    model.load_state_dict(torch.load('checkpoint.pth.tar', map_location=device)['model'])
    model.eval()

    # load sentencepiece model
    write_log(logger, "Load SentencePiece model")
    spm_trg = spm.SentencePieceProcessor()
    spm_trg.Load(f'{args.preprocess_path}/m_trg_{args.trg_vocab_size}.model')

    # Pre-setting
    predicted_list = list()
    label_list = list()
    reference_token = list()
    candidate_token = list()

    k = args.beam_size
    every_batch = torch.arange(0, k*args.batch_size, k, device=device)
    start_time_e = time.time()

    # Decoding text file setting
    f1 = open('./prediction_text.txt', 'w+')
    f2 = open('./label_text.txt', 'w+')
    f1.close()
    f2.close()

    # Beam search
    with torch.no_grad():
        for ix_batch, (src_sequences, trg_sequences) in enumerate(tqdm(test_dataloader)):

            # Device setting
            src_sequences = src_sequences.to(device)
            label_list.extend(trg_sequences.tolist())
            encoder_out_dict = defaultdict(list)

            # Encoding
            encoder_out = model.src_embedding(src_sequences).transpose(0, 1) # (src_seq, batch_size, d_model)
            src_key_padding_mask = (src_sequences == model.pad_idx) # (batch_size, src_seq)
            if args.parallel:
                for i in range(len(model.encoders)):
                    encoder_out_dict[i] = model.encoders[i](encoder_out, src_key_padding_mask=src_key_padding_mask) # (src_seq, batch_size, d_model)
            else:
                for i in range(len(model.encoders)):
                    encoder_out = model.encoders[i](encoder_out, 
                                    src_key_padding_mask=src_key_padding_mask) # (src_seq, batch_size, d_model)

            # Expanding
            src_key_padding_mask = src_key_padding_mask.view(args.batch_size, 1, -1).repeat(1, k, 1).view(-1, src_key_padding_mask.size(1)) # (batch_size * k, src_seq)
            if args.parallel:
                for i in encoder_out_dict:
                    encoder_out_dict[i] = encoder_out_dict[i].view(-1, args.batch_size, 1, args.d_model).repeat(1, 1, k, 1).view(encoder_out.size(0), -1, args.d_model) # (src_seq, batch_size * k, d_model)
            else:
                encoder_out = encoder_out.view(-1, args.batch_size, 1, args.d_model).repeat(1, 1, k, 1).view(encoder_out.size(0), -1, args.d_model) # (src_seq, batch_size * k, d_model)

            # Scores save vector & decoding list setting
            scores_save = torch.zeros(k * args.batch_size, 1).to(device) # (batch_size * k, 1)
            top_k_scores = torch.zeros(k * args.batch_size, 1).to(device) # (batch_size * k, 1)
            complete_seqs = defaultdict(list)
            complete_ind = set()

            # Decoding start token setting
            seqs = torch.tensor([[model.bos_idx]], dtype=torch.long, device=device) 
            seqs = seqs.repeat(k*args.batch_size, 1).contiguous() # (batch_size * k, 1)

            for step in range(model.trg_max_len):
                # Decoder setting
                tgt_mask = model.generate_square_subsequent_mask(seqs.size(1), device) # (out_seq)
                tgt_mask = tgt_mask.to(device, non_blocking=True)
                tgt_key_padding_mask = (seqs == model.pad_idx) # (batch_size * k, out_seq)

                # Decoding sentence
                decoder_out = model.trg_embedding(seqs).transpose(0, 1) # (out_seq, batch_size * k, d_model)
                if args.parallel:
                    for i in range(len(model.decoders)):
                        decoder_out = model.decoders[i](decoder_out, encoder_out_dict[i], tgt_mask=tgt_mask, 
                                        memory_key_padding_mask=src_key_padding_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask) # (out_seq, batch_size * k, d_model)
                else:
                    for i in range(len(model.decoders)):
                        decoder_out = model.decoders[i](decoder_out, encoder_out, tgt_mask=tgt_mask, 
                                        memory_key_padding_mask=src_key_padding_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask) # (out_seq, batch_size * k, d_model)

                # Score calculate
                scores = F.gelu(model.trg_output_linear(decoder_out[-1])) # (batch_size * k, d_embedding)
                scores = model.trg_output_linear2(model.trg_output_norm(scores)) # (batch_size * k, vocab_num)
                scores = F.log_softmax(scores, dim=1) # (batch_size * k, vocab_num)

                # Repetition Penalty
                if step >= 1 and args.repetition_penalty != 0:
                    next_ix = next_word_inds.view(-1)
                    for ix_ in range(len(next_ix)):
                        if scores[ix_][next_ix[ix_]] < 0:
                            scores[ix_][next_ix[ix_]] *= args.repetition_penalty
                        else:
                            scores[ix_][next_ix[ix_]] /= args.repetition_penalty

                # Add score
                scores = top_k_scores.expand_as(scores) + scores  # (batch_size * k, vocab_num)
                if step == 0:
                    scores = scores[::k] # (batch_size, vocab_num)
                    scores[:, model.eos_idx] = float('-inf') # set eos token probability zero in first step
                    top_k_scores, top_k_words = scores.topk(k, 1, True, True)  # (batch_size, k) , (batch_size, k)
                else:
                    top_k_scores, top_k_words = scores.view(args.batch_size, -1).topk(k, 1, True, True)

                # Previous and Next word extract
                prev_word_inds = top_k_words // trg_vocab_num # (batch_size * k, out_seq)
                next_word_inds = top_k_words % trg_vocab_num # (batch_size * k, out_seq)
                top_k_scores = top_k_scores.view(args.batch_size*k, -1) # (batch_size * k, out_seq)
                top_k_words = top_k_words.view(args.batch_size*k, -1) # (batch_size * k, out_seq)
                seqs = seqs[prev_word_inds.view(-1) + every_batch.unsqueeze(1).repeat(1, k).view(-1)] # (batch_size * k, out_seq)
                seqs = torch.cat([seqs, next_word_inds.view(k*args.batch_size, -1)], dim=1) # (batch_size * k, out_seq + 1)

                # Find and Save Complete Sequences Score
                if model.eos_idx in next_word_inds:
                    eos_ind = torch.where(next_word_inds.view(-1) == model.eos_idx)
                    eos_ind = eos_ind[0].tolist()
                    complete_ind_add = set(eos_ind) - complete_ind
                    complete_ind_add = list(complete_ind_add)
                    complete_ind.update(eos_ind)
                    if len(complete_ind_add) > 0:
                        scores_save[complete_ind_add] = top_k_scores[complete_ind_add]
                        for ix in complete_ind_add:
                            complete_seqs[ix] = seqs[ix].tolist()

            # If eos token doesn't exist in sequence
            if 0 in scores_save:
                score_save_pos = torch.where(scores_save == 0)
                for ix in score_save_pos[0].tolist():
                    complete_seqs[ix] = seqs[ix].tolist()
                scores_save[score_save_pos] = top_k_scores[score_save_pos]

            # Beam Length Normalization
            lp = torch.tensor([len(complete_seqs[i]) for i in range(args.batch_size*k)], device=device)
            lp = (((lp + k) ** args.beam_alpha) / ((k + 1) ** args.beam_alpha)).unsqueeze(1)
            scores_save = scores_save / lp

            # Predicted and Label processing
            _, ind = scores_save.view(args.batch_size, k, -1).max(1)
            ind_expand = ind.view(-1) + every_batch
            predicted_list.extend([complete_seqs[i] for i in ind_expand.tolist()])

            # Decoding & BLEU calculate
            for i, comp_seqs in enumerate([complete_seqs[i] for i in ind_expand.tolist()]):
                # Decoding
                pred = spm_trg.DecodeIds(comp_seqs)
                pred_token = [trg_id2word[ix] for ix in comp_seqs]
                real = spm_trg.DecodeIds(trg_sequences.tolist()[i])
                real_token = [trg_id2word[ix] for ix in trg_sequences.tolist()[i]]
                # File writing
                with open('./prediction_text.txt', 'a') as f:
                    f.write(pred + '\n')
                with open('./label_text.txt', 'a') as f:
                    f.write(real + '\n')
                # Append for BLEU calculate
                reference_token.append([real_token])
                candidate_token.append(pred_token)

            # BLEU calculate
            corpus_bleu_score = corpus_bleu(reference_token, candidate_token)

            if ix_batch == 0 or freq == args.print_freq:
                batch_log = '[%d/%d] Corpus BLEU: %3.3f | Spend time:%3.3fmin' % (ix_batch, len(test_dataloader), corpus_bleu_score, (time.time() - start_time_e) / 60)
                write_log(logger, batch_log)
                freq = 0
            freq += 1

    final_bleu_score = corpus_bleu_score / len(test_dataloader)
    with open(f'./results_beam_{args.beam_size}_{args.beam_alpha}_{args.repetition_penalty}_test.pkl', 'wb') as f:
        pickle.dump({'pred': predicted_list, 'real': label_list}, f)
    final_log = f'Total BLEU Score: {final_bleu_score}'
    write_log(logger, final_log)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# Import custom modules
from .embedding.transformer_embedding import TransformerEmbedding
from .modules.layers import EncoderLayer, DecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab_num, trg_vocab_num, pad_idx=0, bos_idx=1, eos_idx=2, 
                 src_max_len=300, trg_max_len=300, d_model=512, d_embedding=256, 
                 n_head=8, d_k=64, d_v=64, dim_feedforward=2048, dropout=0.1, 
                 n_common_layers=6, n_encoder_layers=6, n_decoder_layers=6, device=None):
    
    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64, 
            n_encoder_layers=6, n_decoder_layers=6, n_common_layers=8, 
            n_layers=6, dropout=0.1, n_position=200, share_qk=False, swish_activation=False,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=False):

        super(Transformer, self).__init__()

        # Hyper-parameter setting
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.src_vocab_num = src_vocab_num
        self.trg_vocab_num = trg_vocab_num

        self.n_common_layers = n_common_layers
        self.n_encoder_nonparallel = n_encoder_layers - n_common_layers

        # Source embedding part
        self.src_embedding = TransformerEmbedding(src_vocab_num, d_model, d_embedding, 
                                pad_idx=self.pad_idx, max_len=self.src_max_len, mask_id=mask_idx,
                                embedding_dropout=embedding_dropout, dropout=dropout)
        self.encoder_norms = nn.ModuleList([
            nn.LayerNorm(d_model, eps=1e-6) for _ in range(self.n_common_layers)])

        self.src_output_linear = nn.Linear(d_model, d_embedding, bias=False)
        self.src_output_norm = nn.LayerNorm(d_embedding)
        self.src_output_linear2 = nn.Linear(d_embedding, src_vocab_num, bias=False)
        
        # Target embedding part
        self.trg_embedding = TransformerEmbedding(trg_vocab_num, d_model, d_embedding,
                                pad_idx=self.pad_idx, max_len=self.trg_max_len, 
                                embedding_dropout=embedding_dropout, dropout=dropout, king_num=None)
        self.trg_output_linear = nn.Linear(d_model, d_embedding, bias=False)
        self.trg_output_norm = nn.LayerNorm(d_embedding)
        self.trg_output_linear2 = nn.Linear(d_embedding, trg_vocab_num, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, dim_feedforward, n_head, d_k, d_v, 
                         dropout=dropout) for _ in range(n_encoder_layers)])

    def forward(self, src_seq, trg_seq):
        src_mask = (src_seq != self.pad_idx).unsqueeze(-2)
        trg_mask = (trg_seq != self.pad_idx).unsqueeze(-2)
        # trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        
        # Embedding
        enc_output = self.TransformerEmbedding(src_seq)
        dec_output = self.TransformerEmbedding(trg_seq)

        # [Non-parallel] Encoder
        for enc_layer in self.encoder_layers[:self.n_encoder_nonparallel+1]:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)

        # [Parallel] Encoder -> Decoder
        for enc_layer, enc_norm, dec_layer in zip(
                self.encoder_layers[self.n_encoder_nonparallel+1:],
                self.encoder_norms[:-1],
                self.decoder_layers[:self.n_common_layers-1]):
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_norm(enc_output), slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            
        # [Non-parallel] Decoder
        enc_output = self.encoder_norms[-1](enc_output)
        for dec_layer in self.decoder_layers[self.n_common_layers-1:]:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
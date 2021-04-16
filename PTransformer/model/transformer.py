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
                 n_head=8, d_k=64, d_v=64, dim_feedforward=2048, 
                 dropout=0.1, embedding_dropout=0.1,
                 n_common_layers=6, n_encoder_layers=6, n_decoder_layers=6, 
                 trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=False,
                 parallel=True, device=None):
        super(Transformer, self).__init__()

        # Hyper-parameter setting
        self.parallel = parallel
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.src_vocab_num = src_vocab_num
        self.trg_vocab_num = trg_vocab_num

        self.n_common_layers = n_common_layers
        self.n_encoder_nonparallel = n_encoder_layers - n_common_layers

        # Source embedding part
        self.src_embedding = TransformerEmbedding(src_vocab_num, d_model, d_embedding, 
                                pad_idx=self.pad_idx, max_len=src_max_len,
                                embedding_dropout=embedding_dropout)
        self.encoder_norms = nn.ModuleList([
            nn.LayerNorm(d_model, eps=1e-6) for _ in range(self.n_common_layers)])
        
        # Target embedding part
        self.trg_embedding = TransformerEmbedding(trg_vocab_num, d_model, d_embedding,
                                pad_idx=self.pad_idx, max_len=trg_max_len, 
                                embedding_dropout=embedding_dropout)
        self.trg_output_linear = nn.Linear(d_model, d_embedding, bias=False)
        self.trg_output_norm = nn.LayerNorm(d_embedding)
        self.trg_output_linear2 = nn.Linear(d_embedding, trg_vocab_num, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, dim_feedforward, n_head, d_k, d_v, 
                         dropout=dropout) for _ in range(n_encoder_layers)])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, dim_feedforward, n_head, d_k, d_v, 
                         dropout=dropout) for _ in range(n_decoder_layers)])

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_output_linear2.weight = self.trg_embedding.token.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.src_embedding.token.weight = self.trg_embedding.token.weight

    def forward(self, src_seq, trg_seq):
        src_mask = (src_seq != self.pad_idx).unsqueeze(-2)
        trg_mask = (trg_seq != self.pad_idx).unsqueeze(-2)
        # trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        
        # Embedding
        enc_output = self.src_embedding(src_seq)
        dec_output = self.trg_embedding(trg_seq)

        # P-Transformer
        if self.parallel:
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
        # Normal Transformer
        else:
            for enc_layer in self.encoder_layers:
                enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            for dec_layer in self.decoder_layers:
                dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                    dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)

        dec_output = self.trg_output_norm(self.trg_output_linear(dec_output))
        seq_logit = self.trg_output_linear2(dec_output) * self.x_logit_scale
        return seq_logit.view(-1, seq_logit.size(2))
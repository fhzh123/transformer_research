import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# Import custom modules
from .embedding.transformer_embedding import TransformerEmbedding
from .modules.layers import EncoderLayer, DecoderLayer
from .modules.sublayers import get_subsequent_mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

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

        self.d_model = d_model
        self.emb_src_trg_weight_sharing = emb_src_trg_weight_sharing

        # Source embedding part
        self.src_embedding = nn.Embedding(src_vocab_num, d_embedding, padding_idx=pad_idx)
        self.src_position_enc = PositionalEncoding(d_embedding, n_position=src_max_len)
        self.src_embedding_linear = nn.Linear(d_embedding, d_model)
        self.src_embedding_dropout = nn.Dropout(embedding_dropout)
        self.src_embedding_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Source Multi-head attention part
        self.encoder_norms = nn.ModuleList([
            nn.LayerNorm(d_model, eps=1e-6) for _ in range(self.n_common_layers)])
        
        # Source embedding part
        self.trg_embedding = nn.Embedding(trg_vocab_num, d_embedding, padding_idx=pad_idx)
        self.trg_position_enc = PositionalEncoding(d_embedding, n_position=trg_max_len)
        self.trg_embedding_linear = nn.Linear(d_embedding, d_model)
        self.trg_embedding_dropout = nn.Dropout(embedding_dropout)
        self.trg_embedding_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.trg_embedding = TransformerEmbedding(trg_vocab_num, d_model, d_embedding,
        #                         pad_idx=self.pad_idx, max_len=trg_max_len, 
        #                         embedding_dropout=embedding_dropout)

        # Target Multi-head attention part
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

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_output_linear2.weight = self.trg_embedding.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.src_embedding.weight = self.trg_embedding.weight

    def forward(self, src_seq, trg_seq, non_pad_position=None):
        src_mask = (src_seq != self.pad_idx).unsqueeze(-2)
        trg_mask = (trg_seq != self.pad_idx).unsqueeze(-2) & get_subsequent_mask(trg_seq)
        
        # Source Embedding
        enc_output = self.src_embedding(src_seq)
        if self.emb_src_trg_weight_sharing:
            enc_output *= self.d_model ** 0.5
        enc_output = self.src_embedding_dropout(self.src_position_enc(enc_output))
        enc_output = self.src_embedding_norm(self.src_embedding_linear(enc_output))

        # Target Embedding
        dec_output = self.trg_embedding(trg_seq)
        if self.emb_src_trg_weight_sharing:
            enc_output *= self.d_model ** 0.5
        dec_output = self.trg_embedding_dropout(self.trg_position_enc(dec_output))
        dec_output = self.trg_embedding_norm(self.trg_embedding_linear(dec_output))

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

        if non_pad_position is not None:
            dec_output = dec_output[non_pad_position]

        dec_output = self.trg_output_norm(self.trg_output_linear(dec_output))
        seq_logit = self.trg_output_linear2(dec_output) * self.x_logit_scale
        return seq_logit.log_softmax(dim=1)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

from .embedding.TransformerEmbedding import TransformerEmbedding,

class Transformer(nn.Module):
    def __init__(self, vocab_num, pad_idx=0, bos_idx=1, eos_idx=2, 
                 max_len=300, d_model=512, d_embedding=256, n_head=8, 
                 dim_feedforward=2048, dropout=0.1, n_layers=8, device=None):

        super(Transformer, self).__init__()

        # 
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.dropout = nn.Dropout(dropout)

        self.transformer_embedding = TransformerEmbedding(src_vocab_num, d_model, d_embedding,
                                        pad_idx=self.pad_idx, max_len=self.max_len)

        # Output model
        self.output_linear = nn.Linear(d_model.d_embedding, bias=False)
        self.output_norm = nn.LayerNorm(d_embedding)
        self.output_linear2 = nn.Linear(d_embedding, vocab_num, bias=False)

        # Transformer model
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward,
                activation='gelu', dropout=dropout) for i in range(n_layers)])

    def forward(self, src_input_sentence, non_pad_position=None):
        src_key_padding_mask = (src_input_sentence == self.pad_idx)

        encoder_out = self.src_embedding(src_input_sentence).transpose(0, 1)
        for i in range(len(self.encoders)):
            encoder_out = self.encoders[i](encoder_out, src_key_padding_mask=src_key_padding_mask)

        encoder_out = self.output_norm(self.dropout(F.gelu(self.output_linear(encoder_out))))
        encoder_out = self.output_linear2(encoder_out)
        return encoder_out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, dim_feedforward=2048, dropout=0.1, 
            activation="relu"):
        
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, self_attn, mask_attn, dim_feedforward=2048, dropout=0.1, 
            activation="relu"):
        
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.multihead_attn = mask_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
# Import PyTorch
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from .transformer_embedding import TransformerEmbedding

class Transformer(nn.Module):
    def __init__(self, src_vocab_num, trg_vocab_num, pad_idx=0, bos_idx=1, eos_idx=2, 
            d_model=512, d_embedding=256, n_head=8, dim_feedforward=2048, num_common_layer=10,
            num_encoder_layer=10, num_decoder_layer=10, src_max_len=100, trg_max_len=100, 
            trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=True,
            dropout=0.1, embedding_dropout=0.1, parallel=False):

        super(Transformer, self).__init__()

        # Hyper-paramter setting
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len
        
        # Parallel Transformer setting
        self.parallel = parallel
        self.num_common_layer = num_common_layer
        self.num_encoder_nonparallel = num_encoder_layer - num_common_layer

        # Dropout setting
        self.dropout = nn.Dropout(dropout)

        # Source embedding part
        self.src_embedding = TransformerEmbedding(src_vocab_num, d_model, d_embedding, 
            pad_idx=self.pad_idx, max_len=self.src_max_len, dropout=embedding_dropout)
        
        # Target embedding part
        self.trg_embedding = TransformerEmbedding(trg_vocab_num, d_model, d_embedding,
            pad_idx=self.pad_idx, max_len=self.trg_max_len, dropout=embedding_dropout)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Transformer Decoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        decoder_mask_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.decoders = nn.ModuleList([
            TransformerDecoderLayer(d_model, self_attn, decoder_mask_attn,
                dim_feedforward, dropout=dropout) for i in range(num_decoder_layer)])

        # Target linear part
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, trg_vocab_num)

        # Weight sharing
        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_output_linear2.weight = self.trg_embedding.token.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.src_embedding.token.weight = self.trg_embedding.token.weight
            

    @autocast()
    def forward(self, src_input_sentence, trg_input_sentence, tgt_mask, non_pad_position=None):
        src_key_padding_mask = (src_input_sentence == self.pad_idx)
        tgt_key_padding_mask = (trg_input_sentence == self.pad_idx)

        encoder_out = self.src_embedding(src_input_sentence).transpose(0, 1)
        decoder_out = self.trg_embedding(trg_input_sentence).transpose(0, 1)

        # Parallel Transformer
        if self.parallel:
            # [Non-parallel] Encoder
            for encoder in self.encoders[:self.num_encoder_nonparallel+1]:
                encoder_out = encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

            # [Parallel] Encoder -> Decoder
            for encoder, decoder in zip(
                    self.encoders[self.num_encoder_nonparallel+1:],
                    self.decoders[:self.num_common_layer-1]):
                decoder_out = decoder(decoder_out, encoder_out, tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
                encoder_out = encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

            # [Non-parallel] Decoder
            for decoder in self.decoders[self.num_common_layer-1:]:
                decoder_out = decoder(decoder_out, encoder_out, 
                    memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # Non-parallel Transformer
        else:
            # Encoder
            for encoder in self.encoders:
                encoder_out = encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

            # Decoder
            for decoder in self.decoders:
                decoder_out = decoder(decoder_out, encoder_out, tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)


        decoder_out = decoder_out.transpose(0, 1).contiguous()
        if non_pad_position is not None:
            decoder_out = decoder_out[non_pad_position]

        decoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(decoder_out))))
        decoder_out = self.trg_output_linear2(decoder_out)
        # decoder_out = decoder_out * self.x_logit_scale
        return decoder_out

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))#.transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    @autocast()
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
    def __init__(self, d_model, self_attn, mask_attn, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.multihead_attn = mask_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @autocast()
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

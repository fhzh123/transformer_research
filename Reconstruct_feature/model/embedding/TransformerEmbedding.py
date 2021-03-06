import torch
import torch.nn as nn
from torch.nn import functional as F
from .token import TokenEmbedding
from .segment import SegmentEmbedding
from .positional import PositionalEmbedding

class TransformerEmbedding(nn.Module):
    """
    Embedding which is consisted with under features
    1. TokenEmbedding : normal embedding matrix
    2. PositionalEmbedding : adding positional information using sin, cos
    sum of all these features are output of Embedding
    """

    def __init__(self, vocab_size, d_model, embed_size, pad_idx=0, max_len=512, 
                 embedding_dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size, pad_idx=pad_idx)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=max_len)
        self.segment = SegmentEmbedding(segment_size=3, embed_size=embed_size, pad_idx=pad_idx)
        self.linear_layer = nn.Linear(embed_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

    def forward(self, sequence, sequence_segment=None):
        x = self.token(sequence)
        
        if sequence_segment is None:
            x = self.embedding_dropout(x + self.position(sequence))
        else:
            x = self.embedding_dropout(x + self.position(sequence) + self.segment(sequence_segment))

        return self.norm(self.linear_layer(x))
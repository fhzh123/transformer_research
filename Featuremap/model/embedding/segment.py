import torch.nn as nn

class SegmentEmbedding(nn.Embedding):
    def __init__(self, segment_size, embed_size=512, pad_idx=0):
        super().__init__(segment_size, embed_size, padding_idx=pad_idx)
# Import PyTorch
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from .layers import PatchEmbedding, TransformerEncoderLayer
# from .layers import PatchEmbedding
# from .transformer import TransformerEncoder, ClassificationHead

class Vision_Transformer(nn.Module):
    def __init__(self, n_classes, d_model=512, d_embedding=256, n_head=8, dim_feedforward=2048,
            num_encoder_layer=10, num_decoder_layer=10, img_size=224, patch_size=16,
            dropout=0.3):
    
        super(Vision_Transformer, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # Image embedding part
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, d_embedding=d_embedding, img_size=img_size)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Target linear part (Not averaging)
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, n_classes)

    def forward(self, src_img: Tensor) -> Tensor:
        # Image embedding
        emb_out = self.patch_embedding(src_img).transpose(0, 1)
        
        # Transformer Encoder
        for encoder in self.encoders:
            encoder_out = encoder(emb_out)

        # Target linear
        encoder_out = encoder_out.transpose(0, 1)
        encoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(encoder_out))))
        encoder_out = self.trg_output_linear2(encoder_out)
        return encoder_out

# class Vision_Transformer(nn.Sequential):
#     def __init__(self,     
#                 in_channels: int = 3,
#                 patch_size: int = 16,
#                 emb_size: int = 768,
#                 img_size: int = 224,
#                 depth: int = 12,
#                 n_classes: int = 1000,
#                 **kwargs):
#         super().__init__(
#             PatchEmbedding(in_channels, patch_size, emb_size, img_size),
#             TransformerEncoder(depth, emb_size=emb_size, **kwargs),
#             ClassificationHead(emb_size, n_classes)
#         )

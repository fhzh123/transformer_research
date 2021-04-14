# Import modules
import random
import numpy as np
from itertools import combinations
# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
# Import Huggingface
from transformers import BertForTokenClassification, BertConfig

class kcBERT_reconstruct(nn.Module):
    def __init__(self, vocab_size=30000):

        super(kcBERT_reconstruct, self).__init__()

        # Hyper-parameter setting
        self.vocab_size = vocab_size
        # Model Initiating
        bert_config = BertConfig.from_pretrained('beomi/kcbert-base', 
                                                 num_labels=vocab_size, num_hidden_layers=1)
        self.bert = BertForTokenClassification(config=bert_config)

    def forward(self, src_input_sentence, src_segment):
        # Attention mask setting
        attention_mask = (src_input_sentence != 0)
        out = self.bert(src_input_sentence, token_type_ids=src_segment, 
                        attention_mask=attention_mask)
        out = out.logits.log_softmax(dim=-1)
        return out[attention_mask].contiguous()
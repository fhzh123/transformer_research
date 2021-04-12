# Import modules
import random
import numpy as np
from itertools import combinations
# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
# Import Huggingface
from transformers import BertForPreTraining, BertConfig

class kcBERT_pretraining(nn.Module):
    def __init__(self, vocab_size=30000):

        super(kcBERT_reconstruct, self).__init__()

        # Hyper-parameter setting
        self.vocab_size = vocab_size
        # Model Initiating
        bert_config = BertConfig.from_pretrained('beomi/kcbert-base', num_labels=vocab_size)
        self.bert = BertForPreTraining(config=bert_config)

    def forward(self, src_input_sentence):
        # Attention mask setting
        attention_mask = (src_input_sentence != 0)
        out = self.bert(src_input_sentence, attention_mask=attention_mask)
        out = out.logits.log_softmax(dim=-1)
        return out.view(-1, self.vocab_size).contiguous()
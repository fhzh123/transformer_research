# Import modules
import random
import numpy as np
from itertools import combinations
# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
# Import Huggingface
from transformers import BertForMaskedLM, BertConfig, BertForTokenClassification, BertTokenizer

class Custom_ConditionalBERT(nn.Module):
    def __init__(self, vocab_size=30000):

        super(Custom_ConditionalBERT, self).__init__()

        # Hyper-parameter setting
        self.vocab_size = vocab_size

        # Model Initiating
        # 1) NER Model
        ner_tokenizer = BertTokenizer.from_pretrained("dslim/bert-base-NER")
        ner_model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER")

        # 2) MLM Model
        mlm_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        mlm_model = BertForMaskedLM.from_pretrained('bert-base-cased')

    def forward(self, src_input_sentence, src_segment):
        # Attention mask setting
        attention_mask = (src_input_sentence != 0)
        
        out = self.bert(src_input_sentence, token_type_ids=src_segment, attention_mask=attention_mask)
        return out.prediction_logits, out.seq_relationship_logits
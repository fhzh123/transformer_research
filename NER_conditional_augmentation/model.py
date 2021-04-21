# Import modules
import random
import numpy as np
from itertools import groupby
# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
# Import Huggingface
from transformers import BertForMaskedLM, BertConfig, BertForTokenClassification, BertTokenizer

class Custom_ConditionalBERT(nn.Module):
    def __init__(self, mask_id_token=103, device=None):

        super(Custom_ConditionalBERT, self).__init__()

        # Hyper-parameter setting
        self.mask_id_token = mask_id_token
        self.device = device

        # Model Initiating
        # 1) NER Model
        self.ner_model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER")

        # 2) MLM Model
        self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-cased')

    def forward(self, src_input_sentence):
        # Attention mask setting
        attention_mask = (src_input_sentence != 0)

        # NER Output
        ner_out = self.ner_model(src_input_sentence, attention_mask=attention_mask)
        ner_results = ner_out.logits.max(dim=2)[1]

        # Pad remove
        for ix in (src_input_sentence == 0).nonzero():
            ner_results[ix[0]][ix[1]] = torch.tensor(0)

        # Replace NER token to MASK token
        src_input_sentence[ner_results != 0] = torch.LongTensor([self.mask_id_token]).to(self.device)

        # Padding tensor
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(0)], dim=dim)

        # Remove repetition of MASK token and concat tensor
        for i, src_list in enumerate(src_input_sentence.tolist()):
            src_list = ['remove_token' if g == self.mask_id_token and i!=0 else g for _, group_ in groupby(src_list) for i, g in enumerate(group_)]
            src_list = list(filter(lambda a: a != 'remove_token', src_list))
            src_tensor = pad_tensor(torch.LongTensor(src_list), src_input_sentence.size(1), 0)
            if i == 0:
                ner_masking_tensor = src_tensor.unsqueeze(0)
            else:
                ner_masking_tensor = torch.cat((ner_masking_tensor, src_tensor.unsqueeze(0)), dim=0)

        # MLM process to NER_Masking token
        ner_masking_tensor = ner_masking_tensor.to(self.device)
        ner_attention_mask = (ner_masking_tensor != 0)
        mlm_out = self.mlm_model(ner_masking_tensor, attention_mask=ner_attention_mask)

        return mlm_out.logits, ner_masking_tensor
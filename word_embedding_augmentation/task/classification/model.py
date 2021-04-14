# Import modules
import random
import numpy as np
from itertools import combinations
# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
# Import Huggingface
from transformers import BertForSequenceClassification, BertConfig
# Import custom modules
from ..reconstruction.model import kcBERT_reconstruct

class kcBERT_custom(nn.Module):
    def __init__(self, args, num_labels=3, noise_augment=True, mix_augment=True, 
                 augment_ratio=0.2, reconstruction_feature_use=True, device=None):

        super(kcBERT_custom, self).__init__()

        # Hyper-parameter setting
        self.num_labels = num_labels
        self.noise_augment = noise_augment
        self.mix_augment = mix_augment
        self.augment_ratio = augment_ratio
        self.reconstruction_feature_use = reconstruction_feature_use
        self.device = device
        # Model Initiating
        bert_config = BertConfig.from_pretrained('beomi/kcbert-base', num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', config=bert_config)
        if reconstruction_feature_use:
            self.recon_model = kcBERT_reconstruct(vocab_size=args.vocab_size)
            checkpoint = torch.load('./checkpoint_testing4.pth.tar')
            self.recon_model.load_state_dict(checkpoint['model'])
        # Split BERT embedding
        self.bert_embedding = self.bert.bert.embeddings
        for para in self.bert_embedding.parameters():
            para.requires_grad = False
        self.bert_embedding = self.bert_embedding.eval()
        # Split rest of BERT
        self.bert_encoder = self.bert.bert.encoder
        self.bert_pooler = self.bert.bert.pooler
        self.bert_dropout = self.bert.dropout
        self.bert_classifier = self.bert.classifier

    def forward(self, src_input_sentence, src_label=None):
        # Attention mask setting
        attention_mask = (src_input_sentence != 0)

        if not src_label==None:

            # Extract combinations
            batch_size = src_input_sentence.size(0)
            comb_ = list(combinations(range(batch_size), 2))
            comb_ = random.sample(comb_, int(len(comb_) * self.augment_ratio))
            # Original label processing
            processed_label = torch.zeros(batch_size, 3)
            processed_label[range(processed_label.shape[0]), src_label]=1

            #===================================#
            #====Reconstruct Feature Augment====#
            #===================================#

            if self.reconstruction_feature_use:

                # Reconstruct feature extract
                with torch.no_grad():
                    attention_mask = self.bert.get_extended_attention_mask(attention_mask, 
                                                                           attention_mask.shape, self.device)
                    recon_feature = self.recon_model.bert.bert.embeddings(src_input_sentence)
                    recon_feature = self.recon_model.bert.bert.encoder(recon_feature, attention_mask=attention_mask)
                    recon_feature = recon_feature.last_hidden_state

                new_label_recon = torch.zeros(int(len(comb_)), 3)
                for i, (comb_1, comb_2) in enumerate(comb_):
                    new_label_recon[i][src_label[comb_1]] = 1
                # Concat original label & noise label
                processed_label = torch.cat((processed_label, new_label_recon))
                # Attention mask setting
                for comb_1, comb_2 in comb_:
                    attention_mask = torch.cat((attention_mask, attention_mask[comb_1].unsqueeze(0)))
                # Add noise to word embedding
                with torch.no_grad():
                    for comb_1, comb_2 in comb_:
                        noise = (torch.rand(recon_feature[comb_1].size()) * 0.01).to(self.device)
                        emb1 = recon_feature[comb_1] + noise
                        recon_feature = torch.cat((recon_feature, (emb1).unsqueeze(0)))
                out = self.bert_encoder(recon_feature, attention_mask=attention_mask)
                out = self.bert_classifier(self.bert_dropout(self.bert_pooler(out.last_hidden_state)))
                return out, processed_label

            else:
                # Get BERT embedding values
                with torch.no_grad():
                    embedding_pred = self.bert_embedding(src_input_sentence)

                #===================================#
                #===========Noise Augment===========#
                #===================================#

                if self.noise_augment:
                    new_label_noise = torch.zeros(int(len(comb_)), 3)
                    for i, (comb_1, comb_2) in enumerate(comb_):
                        new_label_noise[i][src_label[comb_1]] = 1
                    # Concat original label & noise label
                    processed_label = torch.cat((processed_label, new_label_noise))
                    # Attention mask setting
                    for comb_1, comb_2 in comb_:
                        attention_mask = torch.cat((attention_mask, attention_mask[comb_1].unsqueeze(0)))
                    # Add noise to word embedding
                    with torch.no_grad():
                        for comb_1, comb_2 in comb_:
                            noise = (torch.rand(embedding_pred[comb_1].size()) * 0.01).to(self.device)
                            emb1 = embedding_pred[comb_1] + noise
                            embedding_pred = torch.cat((embedding_pred, (emb1).unsqueeze(0)))

                #===================================#
                #===========Mixup Augment===========#
                #===================================#

                if self.mix_augment:
                    mix_lam = np.random.beta(1, 1)
                    new_label_mix = torch.zeros(int(len(comb_)), 3)
                    for i, (comb_1, comb_2) in enumerate(comb_):
                        if src_label[comb_1] == src_label[comb_2]:
                            new_label_mix[i][src_label[comb_1]] = 1
                        else:
                            new_label_mix[i][src_label[comb_1]] = mix_lam
                            new_label_mix[i][src_label[comb_2]] = 1-mix_lam
                    # Concat original label & mixup label
                    processed_label = torch.cat((processed_label, new_label_mix))
                    # Attention mask setting
                    for comb_1, comb_2 in comb_:
                        if attention_mask[comb_1].sum() > attention_mask[comb_2].sum():
                            attention_mask = torch.cat((attention_mask, attention_mask[comb_1].unsqueeze(0)))
                        else:
                            attention_mask = torch.cat((attention_mask, attention_mask[comb_2].unsqueeze(0)))
                    # Mixup word embedding
                    with torch.no_grad():
                        for comb_1, comb_2 in comb_:
                            emb1 = embedding_pred[comb_1] * mix_lam
                            emb2 = embedding_pred[comb_2] * (1-mix_lam)
                            embedding_pred = torch.cat((embedding_pred, (emb1 + emb2).unsqueeze(0)))

                # BERT process
                attention_mask = self.bert.get_extended_attention_mask(attention_mask, 
                                                                    attention_mask.shape, self.device)
                out = self.bert_encoder(embedding_pred, attention_mask=attention_mask)
                out = self.bert_classifier(self.bert_dropout(self.bert_pooler(out.last_hidden_state)))

                return out, processed_label

        else:
            with torch.no_grad():
                embedding_pred = self.bert_embedding(src_input_sentence)
            attention_mask = self.bert.get_extended_attention_mask(attention_mask, 
                                                                attention_mask.shape, self.device)
            out = self.bert_encoder(embedding_pred, attention_mask=attention_mask)
            out = self.bert_classifier(self.bert_dropout(self.bert_pooler(out.last_hidden_state)))
            return out
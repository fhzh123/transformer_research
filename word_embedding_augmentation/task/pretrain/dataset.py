import random
# Import PyTorch
import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, comment_list, title_list):
        data = []
        for comment, title in zip(comment_list, title_list):
            data.append((comment, title))
        
        self.data = tuple(data)
        self.num_data = len(self.data)
        
    def __getitem__(self, index):
        comment, title = self.data[index]
        return comment, title
    
    def __len__(self):
        return self.num_data

class PadCollate:
    def __init__(self, pad_index=0, sep_index=3, dim=0):
        self.dim = dim
        self.pad_index = pad_index
        self.sep_index = sep_index

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)

        def masking_sentences(sentences):
            masking_list = list()
            for i, text in enumerate(sentences):
                seg1 = text[1:text.index(3)]
                seg2 = text[text.index(3)+1:-1]
                
                seg1_mask_ix = random.sample(range(len(seg1)), int(len(seg1)*0.15))
                seg2_mask_ix = random.sample(range(len(seg2)), int(len(seg2)*0.15))
                
                seg1 = [4 if i in seg1_mask_ix else x for i, x in enumerate(seg1)]
                seg2 = [4 if i in seg2_mask_ix else x for i, x in enumerate(seg2)]
                
                masking_list.append([2] + seg1 + [3] + seg2 + [3])
            return masking_list

        def pack_sentence(sentences, masking=False):
            sentences_len = max(map(lambda x: len(x), sentences))
            if masking:
                sentences = masking_sentences(sentences)
            sentences = [pad_tensor(torch.LongTensor(seq), sentences_len, self.dim) for seq in sentences]
            sentences = torch.cat(sentences)
            sentences = sentences.view(-1, sentences_len)
            return sentences
        
        comment, title = zip(*batch)
        # Data shuffle for Next Sentence Prediction (NSP)
        shuffle_count = int(len(comment) * 0.3)
        comment_split_1 = list(comment[:-shuffle_count])
        comment_split_2 = list(comment[-shuffle_count:])
        comment_split_2.insert(0, comment_split_2.pop())
        comment = comment_split_1 + comment_split_2
        nsp_label = [1 for _ in range(len(comment_split_1))] + [0 for _ in range(len(comment_split_2))]
        # Processing
        segment = [[0 for _ in t] + [1 for _ in c[:-1]] for c,t in zip(title, comment)]
        total = [t + c[1:] for c,t in zip(comment, title)]
        return pack_sentence(total, masking=True), pack_sentence(segment), \
               pack_sentence(total), torch.LongTensor(nsp_label)
        
    def __call__(self, batch):
        return self.pad_collate(batch)
import random
# Import PyTorch
import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, src_list, trg_list, min_len=4, src_max_len=300, trg_max_len=300):
        data = []
        exception_count = 0
        for src, trg in zip(src_list, trg_list):
            if min_len <= len(src) <= src_max_len and min_len <= len(trg) <= trg_max_len:
                data.append((src, trg))
            else:
                exception_count += 1
        
        self.data = tuple(data)
        self.num_data = len(self.data)
        print(f'Exception by minimum and maximum length: {exception_count}')
        
    def __getitem__(self, index):
        src, trg = self.data[index]
        return src, trg
    
    def __len__(self):
        return self.num_data

class PadCollate:
    def __init__(self, pad_index=0, dim=0):
        self.dim = dim
        self.pad_index = pad_index

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)

        def pack_sentence(sentences):
            sentences_len = max(map(lambda x: len(x), sentences))
            sentences = [pad_tensor(torch.LongTensor(seq), sentences_len, self.dim) for seq in sentences]
            sentences = torch.cat(sentences)
            sentences = sentences.view(-1, sentences_len)
            return sentences
        
        src, trg = zip(*batch)
        return pack_sentence(src), pack_sentence(trg)
        
    def __call__(self, batch):
        return self.pad_collate(batch)
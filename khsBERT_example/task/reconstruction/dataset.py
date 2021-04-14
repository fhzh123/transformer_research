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
        total = comment + title[1:]
        segment = [0 for _ in comment] + [1 for _ in title[:-1]]
        return total, segment
    
    def __len__(self):
        # return self.num_data
        return 100000 # For short training

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
        
        total, segment = zip(*batch)
        return pack_sentence(total), pack_sentence(segment)
        
    def __call__(self, batch):
        return self.pad_collate(batch)
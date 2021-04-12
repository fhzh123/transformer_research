import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, total_list, comment_list, title_list, label_list, max_len=300):
        data = []
        for total, comment, title, label in zip(total_list, comment_list, title_list, label_list):
            if len(total) <= max_len and len(comment) <= max_len and len(title) <= max_len:
                data.append((total, comment, title, label))
        
        self.data = tuple(data)
        self.num_data = len(self.data)
        
    def __getitem__(self, index):
        total, comment, title, label = self.data[index]
        segment = [1 for _ in comment] + [2 for _ in title[:-1]]
        return total, segment, label
    
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

        total, segment, label = zip(*batch)
        return pack_sentence(total), pack_sentence(segment), torch.LongTensor(label)

    def __call__(self, batch):
        return self.pad_collate(batch)

class TestDataset(Dataset):
    def __init__(self, total_list, comment_list, title_list, max_len=300):
        data = []
        for total, comment, title, label in zip(total_list, comment_list, title_list):
            if len(total) <= src_max_len and len(comment) <= len(title) <= max_len:
                data.append((total, comment, title))
        
        self.data = tuple(data)
        self.num_data = len(self.data)
        
    def __getitem__(self, index):
        total, comment, title = self.data[index]
        return total, comment, title
    
    def __len__(self):
        return self.num_data
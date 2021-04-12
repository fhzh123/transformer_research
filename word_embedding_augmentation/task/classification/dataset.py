import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, comment_list, title_list, label_list=None, phase='train'):
        data = []
        if phase != 'test':
            for comment, title, label in zip(comment_list, title_list, label_list):
                data.append((comment, title, label))
        if phase == 'test':
            for comment, title in zip(comment_list, title_list):
                data.append((comment, title))
        
        self.data = tuple(data)
        self.num_data = len(self.data)
        self.phase = phase.lower()
        
    def __getitem__(self, index):
        if self.phase in ['train', 'valid']:
            comment, title, label = self.data[index]
            return comment, title, label
        elif self.phase == 'test':
            comment, title = self.data[index]
            return comment, title
        else:
            raise Exception('[train, valid, test] Only!')
    
    def __len__(self):
        return self.num_data

class PadCollate:
    def __init__(self, pad_index=0, dim=0, isTest=False):
        self.dim = dim
        self.pad_index = pad_index
        self.isTest = isTest

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
        
        batch_out = zip(*batch)
        if self.isTest==False:
            comment, title, label = batch_out
            return pack_sentence(comment), pack_sentence(title), torch.LongTensor(label)
        if self.isTest==True:
            comment, title = batch_out
            return pack_sentence(comment), pack_sentence(title)

    def __call__(self, batch):
        return self.pad_collate(batch)
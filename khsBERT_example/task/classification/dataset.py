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

        def random_five_insert(vec, pad_size, max_len):
            vec_len = len(vec)
            i = 0
            while True:
                p = random.random()
                if i == 0:
                    i += 1
                    continue
                elif i == (max_len-1):
                    rest_tensor = torch.LongTensor()
                    pp = random.random()
                    for ii in range(pad_size[0]):
                        if 0<=pp<0.2:
                            vec = torch.cat([vec[:i+1], torch.LongTensor([30001]), vec[i+1:]], 0)
                        if 0.2<=pp<0.4:
                            vec = torch.cat([vec[:i+1], torch.LongTensor([30002]), vec[i+1:]], 0)
                        if 0.4<=pp<0.6:
                            vec = torch.cat([vec[:i+1], torch.LongTensor([30003]), vec[i+1:]], 0)
                        if 0.6<=pp<0.8:
                            vec = torch.cat([vec[:i+1], torch.LongTensor([30004]), vec[i+1:]], 0)
                        if 0.8<=pp<=1:
                            vec = torch.cat([vec[:i+1], torch.LongTensor([30005]), vec[i+1:]], 0)
                    break
                elif p >= 0.5:
                    i += 1
                    continue
                else:
                    pad_size[0] -= 1
                    if 0<=p<0.1:
                        vec = torch.cat([vec[:i], torch.LongTensor([30001]), vec[i:]], 0)
                    if 0.1<=p<0.2:
                        vec = torch.cat([vec[:i], torch.LongTensor([30002]), vec[i:]], 0)
                    if 0.2<=p<0.3:
                        vec = torch.cat([vec[:i], torch.LongTensor([30003]), vec[i:]], 0)
                    if 0.3<=p<0.4:
                        vec = torch.cat([vec[:i], torch.LongTensor([30004]), vec[i:]], 0)
                    if 0.4<=p<0.5:
                        vec = torch.cat([vec[:i], torch.LongTensor([30005]), vec[i:]], 0)
                    i += 2
            return vec

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
# Import PyTorch
import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, src_list, trg_list, min_len=4, src_max_len=300, trg_max_len=360):
        self.tensor_list = []
        for src, trg in zip(src_list, trg_list):
            if min_len <= len(src) <= src_max_len and min_len <= len(trg) <= trg_max_len:
                src_tensor = torch.zeros(src_max_len, dtype=torch.long)
                src_tensor[:len(src)] = torch.tensor(src, dtype=torch.long)
                trg_tensor = torch.zeros(trg_max_len, dtype=torch.long)
                trg_tensor[:len(trg)] = torch.tensor(trg, dtype=torch.long)
                self.tensor_list.append((src_tensor, trg_tensor))

        self.tensor_list = tuple(self.tensor_list)
        self.num_data = len(self.tensor_list)

    def __getitem__(self, index):
        return self.tensor_list[index]

    def __len__(self):
        return self.num_data
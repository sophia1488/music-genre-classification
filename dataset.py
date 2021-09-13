from torch.utils.data import Dataset, DataLoader
import pickle
import torch

class AudioDataset(Dataset):
    def __init__(self, pkl_dir, transforms=None):
        self.data = []
        self.length = 1500 
        self.transforms = transforms
        with open(pkl_dir, "rb") as f:
          self.data = pickle.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        entry = self.data[idx]
        output_data = {}
        values = entry["values"].reshape(-1, 128, self.length)
        values = torch.Tensor(values)
        if self.transforms:
            values = self.transforms(values)
        target = torch.LongTensor([entry["target"]])
        return (values, target)

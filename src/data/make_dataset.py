import torch
from torch.utils.data import Dataset

class Cifar10BatchDataset(Dataset):
    def __init__(self, d, transform=None):
        self.batch_label = d[b'batch_label']
        self.samples = self.add_from_dict(d)
        self.transform = transform
    
    def add_from_dict(self, d):
        samples = []
        for img, lbl in zip(d[b'data'], d[b'labels']):
            img = torch.tensor(img).reshape(3, 32, 32).div(255.0)
            samples.append((img, lbl))
        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.transform:
            img, ibl = self.transform((self.samples[idx]))
        else:
            img, lbl = self.samples[idx]
        return {'image': img, 'label': lbl}
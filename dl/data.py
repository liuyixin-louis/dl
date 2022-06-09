
import torch
from torch.utils.data import Dataset

class XYData(Dataset):
    def __init__(self,x ,y ):
        self.x = x
        self.y = y
    def __getitem__(self, item):
        return self.x[item],self.y[item]
    def __len__(self):
        return len(self.y)

class Loader():
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, num_workers=4):
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        self.iterator = None

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples
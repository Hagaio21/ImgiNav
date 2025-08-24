import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    """A dummy dataset that returns random tensors of a specified shape."""
    def __init__(self, num_samples, item_spec):
        self.num_samples = num_samples
        self.item_spec = item_spec

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = {}
        for key, shape in self.item_spec.items():
            item[key] = torch.randn(*shape)
        return item

def create_dummy_dataloader(batch_size, num_samples, item_spec):
    """Creates a dataloader with random data."""
    dataset = DummyDataset(num_samples, item_spec)
    return DataLoader(dataset, batch_size=batch_size)
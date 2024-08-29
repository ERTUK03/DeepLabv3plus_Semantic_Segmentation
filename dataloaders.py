from custom_dataset import CustomDataset
import torch
from torch.utils.data import DataLoader, random_split

def get_dataloaders(source_path, target_path, batch_size, train_size, palette = None, transforms = None):
    dataset = CustomDataset(source_path=source_path, target_path=target_path, palette=palette, transform=transforms)
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return DataLoader(train_dataset,
                      batch_size=batch_size,
                      shuffle=True),\
           DataLoader(test_dataset,
                      batch_size=batch_size,
                      shuffle=False)

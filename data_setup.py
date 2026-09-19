
import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path



class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.image_paths = list(Path(data_dir).glob('*/*.jpeg'))
        self.transform = transform
        self.classes = ['NORMAL', 'BACTERIA', 'VIRUS']

        self.class_to_idx = {'NORMAL': 0, 'BACTERIA': 1, 'VIRUS': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def _get_label(self, path):
        p_str = str(path)
        if 'NORMAL' in p_str:
            return self.class_to_idx['NORMAL']
        elif 'bacteria' in p_str.lower():
            return self.class_to_idx['BACTERIA']
        elif 'virus' in p_str.lower():
            return self.class_to_idx['VIRUS']
        else:
            raise ValueError(f'Label not recognized: {path}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self._get_label(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloaders(train_dir, test_dir, transform, batch_size, num_workers):
    """
    Create DataLoaders for training and testing datasets.

    Args:
        train_dir (str): Path to the training dataset directory.
        test_dir (str): Path to the testing dataset directory.
        transform (callable): Transformations to apply to the images.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    """

    

    train_dataset = ChestXRayDataset(data_dir = train_dir, transform = transform)
    test_dataset = ChestXRayDataset(data_dir = test_dir, transform = transform)


    class_names = train_dataset.classes


    train_dataloader = DataLoader(train_dataset,
                                  batch_size = batch_size,
                                  num_workers = num_workers,
                                  shuffle = True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size = batch_size,
                                  num_workers = num_workers,
                                   shuffle = False)

    
    return train_dataloader, test_dataloader, class_names

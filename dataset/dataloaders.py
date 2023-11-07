import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

import albumentations as A

from .datasets import SEA_AISegDatasetFiftyone, FVDataset, FVDatasetIS
from pathlib import Path

class SEA_AIDataLoader(DataLoader):
    """
    A dataloader fo the PyTorch SEA.AI dataset for maritime segmentation.
    """
    def __init__(self, dataset_name, batch_size=2, shuffle=True, num_workers=1, **kwargs):
        transform = A.Compose([
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

        self.dataset = SEA_AISegDatasetFiftyone(dataset_name, **kwargs)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class FVDataloader(DataLoader):
    """

    """
    def __init__(self, dataset_path, names_file, batch_size=2, shuffle=True, num_workers=1, **kwargs):
        transform = A.Compose([
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

        data_path = Path(dataset_path) / 'fac_imgs'
        label_path = Path(dataset_path) / 'fac_labels'

        self.dataset = FVDataset(
            str(data_path), str(label_path), 
            names_path=str( Path(dataset_path)/ names_file),
            **kwargs
        )
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def collate_fn(batch):
    return tuple(zip(*batch))   
 
class FVDataloaderIS(DataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, dataset_path, names_file, batch_size=2, shuffle=True, num_workers=1, **kwargs):
        transform = A.Compose([
            A.RandomCrop(width=450, height=450),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='coco'))
        
        data_path = Path(dataset_path) / 'fac_imgs'
        label_path = Path(dataset_path) / 'fac_labels'

        self.dataset = FVDatasetIS(
            str(data_path), str(label_path), 
            names_path=str( Path(dataset_path)/ names_file), 
            transform=transform,
            **kwargs
        )
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
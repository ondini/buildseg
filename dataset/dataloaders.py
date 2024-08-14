import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

import albumentations as A

from albumentations.pytorch import ToTensorV2

from .datasets import FVDataset, FVDatasetIS, SkyDataset
from .dataset_kpts import FVDatasetKPTS
from pathlib import Path


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
    def __init__(self, dataset_path, names_file, batch_size=2, shuffle=True, num_workers=1, augmentation=True, **kwargs):
        if augmentation:
            transform = A.Compose([
                A.CLAHE(),  
                A.HorizontalFlip(p=.5),
                A.VerticalFlip(p=.05),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=.55),
                A.HueSaturationValue(),
            ]) #, bbox_params=A.BboxParams(format='pascal_voc'))
        else :
            transform = A.Compose([
                A.NoOp()
            ])
            
        data_path = Path(dataset_path) / 'fac_imgs'
        label_path = Path(dataset_path) / 'fac_labels'

        self.dataset = FVDatasetIS(
            str(data_path), str(label_path), 
            names_path=str( Path(dataset_path)/ names_file), 
            transform=transform,
            **kwargs
        )
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        
class FVDataloaderKPT(DataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, coco_root, ann_file, img_prefix, batch_size=2, shuffle=True, num_workers=1, augmentation=False, **kwargs):
        if augmentation:
            transform = A.Compose([
                A.HorizontalFlip(p=.5),
                A.VerticalFlip(p=.5),
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.11, rotate_limit=12, p=.55),
                A.HueSaturationValue(),
            ]) #, bbox_params=A.BboxParams(format='pascal_voc'))
        else :
            transform = A.Compose([
                A.NoOp()
            ])
            
        self.dataset =  FVDatasetKPTS(
            coco_root, ann_file, 
            img_prefix, 
            transform=transform,
            **kwargs
        )
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

class CustomTransform:
    def __init__(self):
        tsfm_list = [A.HorizontalFlip(p=0.5), A.Affine(scale=(0.9, 1.1), rotate=(-15,15), 
                            shear=(-4,4), translate_percent={'x':(-0.1, 0.1), 'y':(0, 0.1)}, p=0.75), 
                            A.RandomBrightnessContrast(p=0.5), 
                            # A.CenterCrop(512,1024),
                            # A.Crop(0, 256, 512, 1024, p=1),
                            A.Resize(512, 1024),
                            ToTensorV2()]
        self.transforms_fin = A.Compose(tsfm_list)

    def __call__(self, img, target):
        img = (np.array(img)/255.).astype(np.float32)
        tgt = (np.array(target)==23).astype(np.uint8)
        fin = self.transforms_fin(image=img, mask=tgt)
        return fin['image'], fin['mask'].unsqueeze(0)
    
from torchvision.datasets import Cityscapes 


class FVDataloaderSkyCS(DataLoader):
    """

    """
    def __init__(self, dataset_path, split='train',  batch_size=2, shuffle=True, num_workers=1):


        self.dataset = Cityscapes(dataset_path, split=split, mode='fine',
                     target_type='semantic', transforms=CustomTransform())

        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class FVDataloaderSkyMV(DataLoader):
    def __init__(self, dataset_path, split='train',  batch_size=2, shuffle=True, num_workers=1):

        tsfm_list = [A.HorizontalFlip(p=0.5), A.Affine(scale=(0.9, 1.1), rotate=(-15,15), 
                                shear=(-4,4), translate_percent={'x':(-0.1, 0.1), 'y':(0, 0.1)}, p=0.75), 
                                A.RandomBrightnessContrast(p=0.5), 
                                A.Resize(752, 1008),
                                ToTensorV2()]
        tsfm = A.Compose(tsfm_list)

        self.dataset = SkyDataset(dataset_path, split=split, transform=tsfm)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

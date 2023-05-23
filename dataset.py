import torch
from PIL import Image, ImageFilter
import os
import numpy as np
import random
import torchvision.transforms as T

def one_hot(label, num_classes):
    """
    Converts label of a segmentation image to one-hot format
    # Args
        label: array(tensor) of shape [1, H, W] with class indices
        num_classes: number of classes in the dataset
    # Rets
        Array of shape[num_classes, W, H] but with num_classes channels
    """

    shape = tuple([num_classes] + list(label.shape)[1:])
    result = torch.zeros(shape)
    result = result.scatter_(0, label.long(), 1)

    return result

class FVDataset(torch.utils.data.Dataset):
    """A PyTorch dataset for roof detection task. Read images, apply augmentation and preprocessing transformations.
    # Args
        images_path: path to images folder
        labels_path: path to segmentation label masks folder
        size_coefficient: percentage of data to be used
        num_classes: number of classes in the dataset
        augmentation: transform for image and mask    
    """
    
    def __init__(self, images_path, labels_path, names_path='', size_coefficient=1, num_classes=2, augmentation=False):   
        self.images_path = images_path 
        self.labels_path = labels_path 

        if names_path != '':
            with open(names_path, 'r') as f:
                lns = f.readlines()
                self.image_fns = [image_fn.strip() for image_fn in lns]
                self.label_fns = [label_fn.strip() for label_fn in lns]
        else:
            self.image_fns = [image_fn for image_fn in sorted(os.listdir(images_path))]
            self.label_fns = [label_fn for label_fn in sorted(os.listdir(images_path))]

        self.num_classes = num_classes
        self.augmentation = augmentation

        self.coefficient = size_coefficient

    def __len__(self):
        return int(len(self.image_fns)*self.coefficient)
    
    def transform(self, imageI, label):
        if self.augmentation:
            imageI = T.TrivialAugmentWide()(imageI)
        image, label = T.ToTensor()(imageI), T.ToTensor()(label)
        if self.augmentation:
            k = random.randint(0, 3)
            image, label  = torch.rot90(image, k=k, dims=(1, 2)), torch.rot90(label, k=k, dims=(1, 2))
            
        return image, label
        
    def __getitem__(self, i):
        image_file_path = os.path.join(self.images_path, self.image_fns[i])
        label_file_path = os.path.join(self.labels_path, self.label_fns[i])

        imageI = Image.open(image_file_path)
        labelI = Image.open(label_file_path)
        label = np.array(labelI).astype('float') # float to make torch.Tensor map it to 1

        image, label = self.transform(imageI, label)
        
        if self.num_classes > 2:
            label = one_hot(label, self.num_classes)

        return image, label
        

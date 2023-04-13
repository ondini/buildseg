import torch
from PIL import Image, ImageFilter
import os
import numpy as np

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
    
    def __init__(self, images_path, labels_path, size_coefficient, num_classes=2, augmentation=None):   
        self.images_path = images_path 
        self.image_fns = [image_fn for image_fn in sorted(os.listdir(images_path))]
        self.labels_path = labels_path 
        self.label_fns = [label_fn for label_fn in sorted(os.listdir(images_path))]

        self.num_classes = num_classes
        self.augmentation = augmentation

        self.coefficient = size_coefficient

    def __len__(self):
        return int(len(self.image_fns)*self.coefficient)

    def __getitem__(self, i):
        image_file_path = os.path.join(self.images_path, self.image_fns[i])
        label_file_path = os.path.join(self.labels_path, self.label_fns[i])

        image = np.array(Image.open(image_file_path)) # cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
        labelI = Image.open(label_file_path)
        label = np.array(labelI).astype('float') # cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE))
        label_border = np.array(labelI.filter(ImageFilter.FIND_EDGES)).astype(bool).astype('float')

        if self.augmentation:
            image, label, label_border = self.augmentation(image), self.augmentation(label), self.augmentation(label_border)

        label = one_hot(label, self.num_classes)
        label_border = one_hot(label_border, self.num_classes)
        
        return image, label, label_border
        

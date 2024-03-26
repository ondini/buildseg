import random
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image
import os
import torch
import torchvision.transforms as T
import json
# import fiftyone as fo
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple

class FVDatasetKPTS(torch.utils.data.Dataset):
    """A PyTorch dataset for roof detection task. Read images, apply augmentation and preprocessing transformations.
    # Args
        images_path: path to images folder
        labels_path: path to segmentation label masks folder
        size_coefficient: percentage of data to be used
        num_classes: number of classes in the dataset
        augmentation: transform for image and mask
    """

    def __init__(
        self,
        coco_root,
        ann_file,
        img_prefix,
        sigma=3,
        transform=None,
        size_coefficient=1
    ):
        self.coco_root = coco_root
        self.img_path = os.path.join(coco_root, img_prefix)
        self.ann_path = os.path.join(coco_root, ann_file)
        self.transform = transform
        self.coefficient = size_coefficient

        # load the annotations
        with open(self.ann_path, "r") as json_file:
            anns = json.load(json_file)

        self.images = anns["images"]
        self.annotations = anns["annotations"]
        self.sigma = sigma
        

    def __len__(self):
        return int(len(self.annotations) * self.coefficient)

    def __getitem__(self, i):
        annotation = self.annotations[i]
        image_id = annotation["image_id"]
        image_file_path = os.path.join(self.img_path, self.images[image_id]["file_name"])
        
        imageI = Image.open(image_file_path)
        kpts = torch.tensor(annotation["keypoints"])
        if len(kpts) == 0:
            return T.ToTensor()(imageI), torch.zeros((1, imageI.size[1], imageI.size[0]))
        kpts = kpts[kpts[:,2]==1]
        mask = generate_channel_heatmap((imageI.size[1], imageI.size[0]), kpts, self.sigma, torch.device("cpu"))
        # create np array from mask 
        mask = mask.numpy()
        
        # apply transform
        result = self.transform(image=np.array(imageI), mask=mask)
        image, mask = T.ToTensor()(result['image']), torch.tensor(result['mask'])       
            
        return image, mask.unsqueeze(0) 
        
    

def generate_channel_heatmap(
    image_size: Tuple[int, int], keypoints: torch.Tensor, sigma: float, device: torch.device
) -> torch.Tensor:
    """
    Generates heatmap with gaussian blobs for each keypoint, using the given sigma.
    Max operation is used to combine the heatpoints to avoid local optimum surpression.
    Origin is topleft corner and u goes right, v down.

    Args:
        image_size: Tuple(int,int) that specify (H,W) of the heatmap image
        keypoints: a 2D Tensor K x 2,  with K keypoints  (u,v).
        sigma: (float) std deviation of the blobs
        device: the device on which to allocate new tensors

    Returns:
         Torch.tensor:  A Tensor with the combined heatmaps of all keypoints.
    """

    assert isinstance(keypoints, torch.Tensor)

    if keypoints.numel() == 0:
        # special case for which there are no keypoints in this channel.
        return torch.zeros(image_size, device=device)

    u_axis = torch.linspace(0, image_size[1] - 1, image_size[1], device=device)
    v_axis = torch.linspace(0, image_size[0] - 1, image_size[0], device=device)
    # create grid values in 2D with x and y coordinate centered aroud the keypoint
    v_grid, u_grid = torch.meshgrid(v_axis, u_axis, indexing="ij")  # v-axis -> dim 0, u-axis -> dim 1

    u_grid = u_grid.unsqueeze(0) - keypoints[..., 0].unsqueeze(-1).unsqueeze(-1)
    v_grid = v_grid.unsqueeze(0) - keypoints[..., 1].unsqueeze(-1).unsqueeze(-1)

    ## create gaussian around the centered 2D grids $ exp ( -0.5 (x**2 + y**2) / sigma**2)$
    heatmap = torch.exp(
        -0.5 * (torch.square(u_grid) + torch.square(v_grid)) / torch.square(torch.tensor([sigma], device=device))
    )
    heatmap = torch.max(heatmap, dim=0)[0]
    return heatmap

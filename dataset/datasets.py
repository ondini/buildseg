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



class CustomTransform:
    def __init__(self, joint_transform, img_transform=None):
        self.joint_transform = joint_transform
        self.img_transform = img_transform

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = T.TrivialAugmentWide()(image)
        image, label = T.ToTensor()(image), T.ToTensor()(label)

        k = random.randint(0, 3)
        image, label = torch.rot90(image, k=k, dims=(
            1, 2)), torch.rot90(label, k=k, dims=(1, 2))

        return {'image': image, 'mask': label}


class SEA_AISegDatasetFiftyone(Dataset):
    def __init__(self, dataset_name, view_name=None, background_list=["WATER", "HARBOUR", "SKY", "LAND"], transform=None, label_list=None, shape=[512, 640], **kwargs):
        """A PyTorch dataset for the SEA.AI segmentation dataset extending fiftyonedataset for PyTorch purposes

        Args:
            dataset_name (str): Name of Fiftyone segmentation dataset to be used.
            view_name (str, optional): Name of the view in the given Fiftyone dataset to be used. Defaults to None.
            background_list (list, optional): list of classes which are regarded as background. Defaults to ["WATER", "HARBOUR", "SKY", "LAND"].
            transform (A.Compose, optional): albumentations Compose defining augmentation. Defaults to None.
            label_list (list, optional): list of classes to be used. Defaults to None.
        """

        self.transform = transform
        self.dataset = fo.load_dataset(dataset_name)
        self.shape = shape

        if view_name is not None:
            self.dataset = self.dataset.load_saved_view(view_name)

        if label_list is None:  # TODO: implement label_list in some more reasonable way, probably start with one label?
            label_list = self.dataset.distinct(
                "ground_truth_det.detections.label")

        self.class_dict = {label: idx+1 for idx,
                           label in enumerate(label_list)}
        self.class_dict_i = {v: k for k, v in self.class_dict.items()}
        self.background_list = background_list
        self.foreground_list = list(set(label_list) - set(background_list))
        self.img_paths = self.dataset.values("filepath")

    def __len__(self):
        """returns the length of the dataset

        Returns:
            int: length of the dataset 
        """
        return len(self.img_paths)

    def __getitem__(self, index):
        """gets a sample on given index from the dataset

        Args:
            index (int): the index of the sample to get

        Returns:
            list: image, mask bboxes
        """

        filepath = self.img_paths[index]
        sample = self.dataset[filepath]
        group = self.dataset.get_group(sample.group.id)
        rgb_path = group['rgb_narrow'].filepath
        thermal_path = group['thermal_narrow'].filepath
        rgb_img = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        # rgb_img = self._standardize_image(rgb_img)

        thermal_img = np.array(Image.open(thermal_path).convert('L'), dtype=np.uint8)
        # thermal_img = self._standardize_image(thermal_img)

        img_shape = rgb_img.shape[:2]
        
        #image = np.concatenate((rgb_img, thermal_img[..., np.newaxis]), axis=2) 

        mask = np.zeros(img_shape, np.uint8)
        bboxes = []
        if sample.ground_truth_det is not None and len(sample.ground_truth_det.detections) > 0:
            for detection in sample.ground_truth_det.detections:
                box = [*detection.bounding_box, detection.label]
                if detection.label in self.foreground_list and box[2] != 0 and box[3] != 0:
                    bboxes.append([box[0]*img_shape[1], box[1]*img_shape[0], (box[0]+box[2])
                                  * img_shape[1], (box[1]+box[3])*img_shape[0], box[4]])

                    if detection.mask is not None:
                        y, x = np.where(detection.mask == 1)
                        y += int(detection.bounding_box[1]*img_shape[0])
                        x += int(detection.bounding_box[0]*img_shape[1])
                        mask[y, x] = self.class_dict[detection.label]
                    else:
                        x1, y1, x2, y2 = map(int, bboxes[-1][:4])
                        mask[y1:y2, x1:x2] = self.class_dict[detection.label]
        else:
            bboxes = [[0.0, 0.0, .5, .5, -1]]

        if self.transform is not None:
            augmentations = self.transform(
                image=image, mask=mask, bboxes=bboxes)
            image = augmentations["image"]
            mask = augmentations["mask"]
            bboxes = augmentations["bboxes"]

        if bboxes == [] or bboxes[0][-1] == -1:
            bboxes = []

        bboxes = np.array(bboxes)

        return rgb_img, thermal_img, mask #T.ToTensor()(image).float(), T.ToTensor()(mask)

    def get_item_from_path(self, path):
        '''gets a sample on index corresponding to given path from the dataset'''
        
        index = self.img_paths.index(path)
        return self.__getitem__(index)
        
        
    @staticmethod
    def collate_fn(batch):
        """collate function for the dataloader
        """
        img, mask, boxes = zip(*batch)

        img = torch.from_numpy(np.stack(img, axis=0))
        mask = torch.from_numpy(np.stack(mask, axis=0))

        bbox = [torch.from_numpy(arr) for arr in boxes]

        return img, mask, bbox

    def _standardize_image(self, img):
        """ 
            if image has a wrong dimension, resize it, if it is landscape, rotate it

            Args:
                img (np.array): image to be standardized
        """

        if img.shape[0] > img.shape[1]:
            img = np.rot90(img)

        if img.shape[0] != self.shape[0] or img.shape[1] != self.shape[1]:
            img = cv2.resize(img, (self.shape[1], self.shape[0]))

        return img
        

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

    def __init__(self, images_path, labels_path, names_path='', size_coefficient=1, num_classes=2, augmentation=True, **kwargs):
        self.images_path = images_path
        self.labels_path = labels_path

        if names_path != '':
            with open(names_path, 'r') as f:
                lns = f.readlines()
                self.image_fns = [image_fn.strip() for image_fn in lns]
                self.label_fns = [label_fn.strip() for label_fn in lns]
        else:
            self.image_fns = [image_fn for image_fn in sorted(
                os.listdir(images_path))]
            self.label_fns = [label_fn for label_fn in sorted(
                os.listdir(images_path))]

        self.num_classes = num_classes
        self.augmentation = augmentation

        self.coefficient = size_coefficient
        if size_coefficient < 1:
            # create shuffled ids array:
            self.ids = np.arange(len(self.image_fns))
            np.random.shuffle(self.ids, random_state=42)
            self.ids = self.ids[:int(len(self.image_fns)*self.coefficient)]

            self.image_fns = [self.image_fns[i] for i in self.ids]
            self.label_fns = [self.label_fns[i] for i in self.ids]
            

    def __len__(self):
        return int(len(self.image_fns)*self.coefficient)

    def transform(self, imageI, label):

        if self.augmentation:
            transform = T.Compose([
                T.TrivialAugmentWide(),
                T.GaussianBlur(kernel_size=3),
            ])
            imageI = transform(imageI)
        image, label = T.ToTensor()(imageI), T.ToTensor()(label)
        if self.augmentation:
            merged_tensor = torch.cat((image, label), dim=0)
            transform = T.Compose([
                T.RandomHorizontalFlip(0.1),
                T.RandomVerticalFlip(0.5),
                T.RandomRotation(degrees=15),
            ])
            transformed_tensor = transform(merged_tensor)
            # Extract the fcirst three channels for the image
            image = transformed_tensor[:3]
            # Extract the fourth channel for the label
            label = (transformed_tensor[3:] > 0).float()

        # if self.augmentation:
        #     k = random.randint(0, 3)
        #     image, label  = torch.rot90(image, k=k, dims=(1, 2)), torch.rot90(label, k=k, dims=(1, 2))

        return image.float(), label.float()

    def __getitem__(self, i):
        image_file_path = os.path.join(self.images_path, self.image_fns[i])
        label_file_path = os.path.join(self.labels_path, self.label_fns[i])

        imageI = Image.open(image_file_path)
        labelI = Image.open(label_file_path)
        # float to make torch.Tensor map it to 1
        label = np.array(labelI).astype('float')

        image, label = self.transform(imageI, label)

        if self.num_classes > 2:
            label = one_hot(label, self.num_classes)

        return image, label


class FVDatasetIS(torch.utils.data.Dataset):
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
        images_path,
        labels_path,
        names_path="",
        transform=None,
        size_coefficient=1,
        num_classes=5,
        augmentation=False,
    ):
        self.images_path = images_path
        self.labels_path = labels_path

        if names_path != "":
            with open(names_path, "r") as f:
                lns = f.readlines()
                self.image_fns = [image_fn.strip() for image_fn in lns]
                self.label_fns = [
                    ".".join(label_fn.strip().split(".")[:-1]) + ".json"
                    for label_fn in lns
                ]
        else:
            self.image_fns = [image_fn for image_fn in sorted(os.listdir(images_path))]
            self.label_fns = [label_fn for label_fn in sorted(os.listdir(images_path))]

        self.num_classes = num_classes
        self.augmentation = augmentation
        self.transform = transform

        self.coefficient = size_coefficient

    def __len__(self):
        return int(len(self.image_fns) * self.coefficient)

    def __getitem__(self, i):
        image_file_path = os.path.join(self.images_path, self.image_fns[i])
        label_file_path = os.path.join(self.labels_path, self.label_fns[i])

        imageI = Image.open(image_file_path)
        with open(label_file_path, "r") as json_file:
            annotations = json.load(json_file)

        bbs = []  # boudning boxes
        masks, labels, instance_ids = generate_annotations(annotations)

        labels = torch.tensor(labels).long()
        instance_ids = torch.tensor(instance_ids).long()
        transforms = self.transform(image=np.array(imageI), masks=masks)
        image, masks = T.ToTensor()(transforms["image"]), torch.tensor(np.array(transforms["masks"]))
        bbs, remove_ids = generate_bbs(masks)
        
        if len(remove_ids) > 0: # some bbs were removed, remove labels and masks on this id
            # print('removing', remove_ids)
            for remove_id in remove_ids:     # remove ids  from  tensors
                masks = torch.cat((masks[:remove_id], masks[remove_id+1:]))
                labels = torch.cat((labels[:remove_id], labels[remove_id+1:]))
                instance_ids = torch.cat((instance_ids[:remove_id], instance_ids[remove_id+1:]))

        bbs = (
            torch.tensor(bbs, dtype=torch.float32)
            if len(bbs) > 0
            else torch.tensor(np.zeros((0, 4)))
        )  # in bbs is expected a tensor of shape [N, 4] where N is the number of bounding boxes

        masks = (
            masks
            if len(masks) > 0
            else torch.tensor(np.zeros((0, 960, 540)), dtype=torch.uint8) # shape muset be the same as img
        )

        return image, {
            "masks": masks,
            "labels": labels,
            "boxes": bbs,
            "image_id": torch.tensor(i),
        }
        
    def collate_fn(batch):
        return tuple(zip(*batch))


def generate_bbs(masks):
    bbs = []
    remove_ids = []
    for i, mask in enumerate(masks):
        pos = np.nonzero(mask)
        if pos.numel() < 1:
            remove_ids.append(i)
            continue
        xmin = pos[:, 1].min()
        xmax = pos[:, 1].max()
        ymin = pos[:, 0].min()
        ymax = pos[:, 0].max()
        if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                remove_ids.append(i)
                continue  #   skip single-line masks, since bounding boxes muse be at least 2x2
        bbs.append([xmin, ymin, xmax, ymax])
    return bbs, remove_ids

def generate_annotations(annotations):
        # generate masks so that they are properly cropped
        roof_masks = []
        roof_labels = []
        roof_ids = []
        
        masks = []
        labels = []
        instance_ids = []
        
        for annotation in annotations:
            polygon = np.array(annotation["polygon"], dtype=np.int32)
            # skip polygons with less than 3 vertices and edge classes, as they are not polygonizable
            if len(polygon) < 3 or annotation["class_id"] in [3, 5, 6, 7]: 
                continue
            
            instance_mask = np.zeros(annotation["img_size_old"][:2], dtype=np.uint8)
            cv2.fillPoly(instance_mask, [polygon], (1, 1, 1))

            # must be resized afterwards, as otherwise the polygons are not as precise
            instance_mask_r = cv2.resize(
                instance_mask,
                dsize=(annotation["img_size_new"][:2]),
                interpolation=cv2.INTER_AREA,
            )

            if instance_mask_r.sum() < 8:
                continue  # skip empty masks -> MASKS WENt empty after resizing, if they were too small at the original image
            
            if annotation["class_id"] == 0: # roof
                roof_masks.append(instance_mask_r)
                roof_labels.append(annotation["class_id"] + 1)
                roof_ids.append(annotation["instance_id"])
            else:
                masks.append(instance_mask_r)
                labels.append(annotation["class_id"] + 1)
                instance_ids.append(annotation["instance_id"])
        
        # go through masks and crop out "remove" or "chimney" class masks from roof masks
        for i, mask in enumerate(masks):
            if not labels[i]-1 in [2, 8]: # remove or chimney
                continue
            for j, roof_mask in enumerate(roof_masks): # go through roof masks
                roof_masks[j] = roof_masks[j] * (1 - mask)
        
        # check if roof masks are empty after cropping, otherwise append to masks
        for i, roof_mask in enumerate(roof_masks):
            if roof_mask.sum() < 1:
                continue
            masks.append(roof_mask)
            labels.append(1)
            instance_ids.append(roof_ids[i])

        return masks, labels, instance_ids


def polygon_to_mask(polygon_annotations, image_shape):
    """
    Convert polygon annotations to a mask.

    Parameters:
    polygon_annotations (list of list of tuple): List of polygons, where each polygon is represented by a list of (x, y) tuples.
    image_shape (tuple): Shape of the image (height, width).

    Returns:
    dict: A dictionary with keys 'mask' containing the binary mask and 'annotations' containing the polygon annotations.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)

    for object in polygon_annotations:
        num = 1
        if object['classTitle'] != 'sky':
            num = 2
        polygon = object['points']['exterior']           

        polygon_np = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, polygon_np, num)

    
    return (mask == 1).astype(np.int16)

class SkyDataset(torch.utils.data.Dataset):
    """A PyTorch dataset for sky detection task. Read images, apply augmentation and preprocessing transformations.
    # Args
        type (str): testing or training or validation
    """

    def __init__(self, root_path, split='training', size_coefficient=1, transform=None, **kwargs):
        self.root_path = root_path

        self.transform = transform
        self.coefficient = size_coefficient

        self.img_path = os.path.join(root_path, split, 'img')
        self.names_list = os.listdir( self.img_path)
        self.ann_path = os.path.join(root_path, split, 'ann')



    def __len__(self):
        return int(len(self.names_list)*self.coefficient)

    def __getitem__(self, i):
        image_file_path = os.path.join(self.img_path, self.names_list[i])
        label_file_path = os.path.join(self.ann_path, self.names_list[i] +'.json')

        imageI = Image.open(image_file_path)
        #load ann file from json
        with open(label_file_path) as f:
            label = json.load(f)
        tgt = polygon_to_mask(label['objects'], imageI.size[::-1])
        
        img = (np.array(imageI)/255.).astype(np.float32)
        if self.transform:
            fin = self.transform(image=img, mask=tgt)
            img, tgt =  fin['image'], fin['mask'].unsqueeze(0).float()


        return img, tgt
import torch
from PIL import Image, ImageFilter
import os
import numpy as np
import random
import torchvision.transforms as T
import json
import cv2
import matplotlib.pyplot as plt


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
        size_coefficient=1,
        num_classes=5,
        augmentation=True,
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

        self.coefficient = size_coefficient

    def __len__(self):
        return int(len(self.image_fns) * self.coefficient)

    def transform(self, imageI, masksI):
        if self.augmentation:
            transform = T.Compose(
                [
                    T.TrivialAugmentWide(),
                ]
            )
            imageI = transform(imageI)
        image, masks = T.ToTensor()(imageI), torch.tensor(np.array(masksI))

        if False:  # self.augmentation:
            merged_tensor = torch.cat((image, masks), dim=0)
            transform = T.Compose(
                [
                    T.RandomHorizontalFlip(0.5),
                    T.RandomVerticalFlip(0.05),
                    T.RandomRotation(degrees=15),
                ]
            )
            transformed_tensor = transform(merged_tensor)
            image = transformed_tensor[
                :3
            ]  # Extract the first three channels for the image
            masksO = (
                transformed_tensor[3:] > 0
            )  # Extract the mask channels and make them binary

        return image.float(), masks.to(torch.uint8)

    def __getitem__(self, i):
        image_file_path = os.path.join(self.images_path, self.image_fns[i])
        label_file_path = os.path.join(self.labels_path, self.label_fns[i])

        imageI = Image.open(image_file_path)
        with open(label_file_path, "r") as json_file:
            annotations = json.load(json_file)

        masks = []
        labels = []
        instance_ids = []
        bbs = []  # boudning boxes

        for annotation in annotations:
            if (
                annotation["class_id"] != 0 and annotation["class_id"] % 3 == 0
            ):  # skip chimneys for now
                continue
            instance_mask = np.zeros(annotation["img_size_old"][:2], dtype=np.uint8)
            polygon = np.array(annotation["polygon"], dtype=np.int32)
            if len(polygon) < 3:
                continue
            cv2.fillPoly(instance_mask, [polygon], (1, 1, 1))

            instance_mask_r = cv2.resize(
                instance_mask,
                dsize=(annotation["img_size_new"][:2]),
                interpolation=cv2.INTER_AREA,
            )

            if instance_mask_r.sum() < 1:
                continue  # skip empty masks -> MASKS WENt empty after resizing

            masks.append(instance_mask_r)
            labels.append(annotation["class_id"])
            instance_ids.append(annotation["instance_id"])

        labels = torch.tensor(labels).long()
        instance_ids = torch.tensor(instance_ids).long()
        image, masks = self.transform(imageI, masks)

        for mask in masks:
            pos = np.nonzero(mask)
            if len(pos) < 1:
                print(mask.shape, pos)
            xmin = pos[:, 0].min()
            xmax = pos[:, 0].max()
            ymin = pos[:, 1].min()
            ymax = pos[:, 1].max()
            if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                continue  # skip single-line masks, since bounding boxes muse be at least 2x2
                print(mask.shape, pos)
            bbs.append([xmin, ymin, xmax, ymax])
        # print(image.shape, masks.shape)
        bbs = (
            torch.tensor(bbs, dtype=torch.float32)
            if len(bbs) > 0
            else torch.tensor(np.zeros((0, 4)))
        )  # cuz in bbs is expected a tensor of shape [N, 4] where N is the number of bounding boxes
        masks = (
            masks
            if len(masks) > 0
            else torch.tensor(
                np.zeros(annotation["img_size_new"][:2]), dtype=torch.uint8
            )
        )
        return image, {"masks": masks, "labels": labels, "boxes": bbs, "image_id": i}


def generate_bounding_box(polygon, frac):
    # Initialize min and max coordinates with extreme values
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    # Iterate through the vertices of the polygon to find min and max coordinates
    for x, y in polygon:
        min_x = min(min_x, x * frac[0])
        min_y = min(min_y, y * frac[1])
        max_x = max(max_x, x * frac[0])
        max_y = max(max_y, y * frac[1])

    # min_x = min_x - 1 if min_x > 0 else min_x
    # min_y = min_y - 1 if min_y > 0 else min_y
    # max_x = max_x + 1 if max_x < frac else max_x
    # max_y = max_y + 1 if max_y < frac else max_y

    # Define the bounding box as (x_min, y_min, width, height)
    bbox = [min_x, min_y, max_x, max_y]

    return bbox


if __name__ == "__main__":
    image_file_path = os.path.join(
        "/home/kafkaon1/Dev/FVAPP/data/CZ7/fac_imgs/20230428_161833.jpg"
    )
    label_file_path = os.path.join(
        "/home/kafkaon1/Dev/FVAPP/data/CZ7/fac_labels/20230428_161833.json"
    )

    imageI = Image.open(image_file_path)
    with open(label_file_path, "r") as json_file:
        annotations = json.load(json_file)

    masks = []
    labels = []
    instance_ids = []

    for annotation in annotations:
        instance_mask = np.zeros(annotation["img_size_old"][:2], dtype=np.uint8)
        polygon = np.array(annotation["polygon"], dtype=np.int32)
        if len(polygon) < 2:
            continue
        cv2.fillPoly(instance_mask, [polygon], (1, 1, 1))

        instance_mask = cv2.resize(
            instance_mask,
            dsize=(annotation["img_size_new"][:2]),
            interpolation=cv2.INTER_AREA,
        )
        masks.append(instance_mask)
        labels.append(annotation["class_id"])
        instance_ids.append(annotation["instance_id"])

    labels = np.array(labels, dtype=np.int64)
    instance_ids = np.array(instance_ids, dtype=np.int64)

    transform = T.Compose(
        [
            T.TrivialAugmentWide(),
            T.GaussianBlur(kernel_size=3),
        ]
    )

    imageI = transform(imageI)
    image, masks = T.ToTensor()(imageI), torch.tensor(masks)

    merged_tensor = torch.cat((image, masks), dim=0)
    transform = T.Compose(
        [
            T.RandomHorizontalFlip(0.1),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(degrees=15),
        ]
    )
    transformed_tensor = transform(merged_tensor)
    image = transformed_tensor[:3]  # Extract the first three channels for the image
    masks = (
        transformed_tensor[3:] > 0
    ).float()  # Extract the mask channels and make them binary

    print(image, masks, labels, instance_ids)
    # return image, masks, labels, instance_ids

from PIL import Image
import numpy as np
from scipy import ndimage
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

frame_size = 640
img_ratio = 0.075/0.1 # resolutions of given dataset and Google - (156543.03392 * math.cos(math.radians(lat))) / (2 ** zoom)

input_dir = '/home/ondin/Developer/FVAPP/data_big/archive/val/'
output_dir = '/home/ondin/Developer/FVAPP/data_big/archive/val/'

imgs_path = os.path.join(input_dir, 'image')
labels_path = os.path.join(input_dir, 'label')

out_imgs_path = os.path.join(output_dir, 'image_resized')
out_labels_path = os.path.join(output_dir, 'label_resized')

if not os.path.exists(out_imgs_path):
    os.makedirs(out_imgs_path)
if not os.path.exists(out_labels_path):
    os.makedirs(out_labels_path)

img_names = os.listdir(imgs_path)
for img_name in img_names:
    img_path = os.path.join(imgs_path, img_name)
    label_path = os.path.join(labels_path, img_name)

    image = np.array(Image.open(img_path))
    label = np.array(Image.open(label_path))

    new_shape = (int(label.shape[0] * img_ratio), int(label.shape[1] * img_ratio))
    label = cv2.resize(label, dsize=new_shape, interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, dsize=new_shape, interpolation=cv2.INTER_AREA)

    data = label
    # Find the positive label areas
    positive_label_areas = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)
    positive_label_areas[np.where(data == 1)] = 1

    # Generate a label mask that assigns a unique label to each positive area
    label_mask, num_labels = ndimage.label(positive_label_areas)
    print("Processing image: ", img_name)

    # Loop through each positive label area and generate a frame
    for i in tqdm(range(1, num_labels+1), desc='Generating frames'):
        # Find the bounding box of the positive label area
        rows, cols = np.where(label_mask == i)
        row_min, row_max = np.min(rows), np.max(rows)
        col_min, col_max = np.min(cols), np.max(cols)

        # Calculate the center of the positive label area
        center_row = (row_min + row_max) // 2
        center_col = (col_min + col_max) // 2

        # Calculate the frame boundaries

        x_min, x_max= (0, frame_size) if center_col - frame_size // 2 < 0 else (data.shape[1]-frame_size, data.shape[1]) if center_col + frame_size // 2 > data.shape[1] else (center_col - frame_size // 2, center_col + frame_size // 2)
        y_min, y_max= (0, frame_size) if center_row - frame_size // 2 < 0 else (data.shape[0]-frame_size, data.shape[0]) if center_row + frame_size // 2 > data.shape[0] else (center_row - frame_size // 2, center_row + frame_size // 2)

        # Extract the content
        img_content = image[y_min:y_max, x_min:x_max]
        label_content = label[y_min:y_max, x_min:x_max]

        # Save the frame content as a new image file
        img_out_path = os.path.join(out_imgs_path, f'{img_name[:-4]}_{i}.png')
        Image.fromarray(img_content).save(img_out_path)
        
        label_out_path = os.path.join(out_labels_path, f'{img_name[:-4]}_{i}.png')
        Image.fromarray(label_content).save(label_out_path)
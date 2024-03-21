import sqlite3
import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pdb

from PIL import Image, ImageDraw
import json
  
from collections import defaultdict
  
IMG_SHAPE_ = (1333, 800)

DB_PATH =  "/home/kafkaon1/Dev/data/db_updated_05_03_24.db" # "/local2/homes/zderaann/roof_annotations/database_updates.db" #"/local/homes/zderaann/roof_annotations/database.db"
IN_IMGS_PATH = "/local2/homes/zderaann/roof_annotations/annotated_imgs"
IN_SAT_PATH = "/local2/homes/zderaann/roof_annotations/satelite"

PATCH_SIZE = 256

OUT_PATH = '/home/kafkaon1/Dev/data/COCO_KPTS_1103'

COLORS = [
  "green",
  "blue",
  "red",
  "yellow",
  "brown",
  "pink",
  "orange",
  "purple",
  "black",
]

CLASSES = [
  "roof",
  "window",
  "remove",
  "chimnedge",
  "other",
  "ridge",
  "sideedge",
  "edge",
  "chimney",
]

CLASSES_COCO = [ # COCO classes
  "roof",
  "window",
  "other",
  "chimney",
]

COCO_MAPPING = {
    0: 1, # roof
    1: 2, # window
    4: 3, # other
    8: 4, # chimney
    2: 5, # remove
}

NUM_CLASSES = len(CLASSES)
NUM_COCO_CLASSES = len(CLASSES_COCO)
MAX_ANNOTS_PER_IMG = 100


import json
from pycocotools import mask

random.seed(42)

import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

register_all_modules()

min_area_threshold=400
arc_length_coeff=0.005     
        
models = {
    'point_rend': ('/home/kafkaon1/Dev/mmdetection/configs/point_rend/point-rend_r101-caffe_fpn_ms-3x_roofs.py', '/home/kafkaon1/Dev/out/pointRendOut1012/epoch_20.pth')
}

model_name = 'point_rend'
model = init_detector(*models[model_name], device='cuda:1') 

def get_new_size(old_size):
    # get new size by keeping the aspect ratio
    old_height, old_width = old_size
    tgt_width, tgt_height = IMG_SHAPE_
    if old_width / old_height > tgt_width / tgt_height:
        new_width = tgt_width
        new_height = int(old_height * tgt_width / old_width)
    else:
        new_height = tgt_height
        new_width = int(old_width * tgt_height / old_height)
        
    
    return (new_width, new_height)

def get_patch(img, x, y, pts):
    # create 256x256 patch with the point in the middle from img with prevence against overflowing
    
    x = x if x >= PATCH_SIZE//2 else PATCH_SIZE//2
    y = y if y >= PATCH_SIZE//2 else PATCH_SIZE//2
    x = x if x <= img.shape[1] - PATCH_SIZE//2 else img.shape[1] - PATCH_SIZE//2
    y = y if y <= img.shape[0] - PATCH_SIZE//2 else img.shape[0] - PATCH_SIZE//2
    x = int(x)
    y = int(y)
    
    # get all pts, that are in the patch
    pts_f = pts[(pts[:, 0] > x - PATCH_SIZE//2) & (pts[:, 0] < x + PATCH_SIZE//2) & (pts[:, 1] > y - PATCH_SIZE//2) & (pts[:, 1] < y + PATCH_SIZE//2)]
    pts_f[:,:2] = pts_f[:,:2] - [x-PATCH_SIZE//2, y-PATCH_SIZE//2]

    return img[y-PATCH_SIZE//2:y+PATCH_SIZE//2, x-PATCH_SIZE//2:x+PATCH_SIZE//2], pts_f

def generate(test=0.0005, train=0.0005):

    thrs = {'train': train, 'test': test + train}

    for filename in ['train', 'test']:
        out_imgs_path = os.path.join(OUT_PATH, f'{filename}_imgs') # image path

        if not os.path.exists(out_imgs_path):
            os.makedirs(out_imgs_path)
        
        # connect to database        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("SELECT * FROM annotation")

        rows = c.fetchall() # get all rows from the table = all annotations
        #random.shuffle(rows)

        #categories = [{"id": i + 1, "name": CLASSES_COCO[i]} for i in range(NUM_COCO_CLASSES)]

        images = []
        annotations = []
        img_id = 0
        print(f"Starting {filename} set \n")
    
        for irow, row in enumerate(rows): # go through all the annotations
            # if os.path.exists(os.path.join(OUT_PATH, row[0])): # image is already saved there
            #     continue
            if 'IMG_3258' in row[0]:
                continue
            print(f"{irow}/{len(rows)} = {irow/len(rows)*100:.2f}%, imgname: {row[0]}")
            
            img_fn = row[0] # filename
            img_path = os.path.join(IN_IMGS_PATH, img_fn)
            img = cv2.imread(img_path)[:,:,::-1]
            img_annotation = json.loads(row[2])
            
            res = inference_detector(model, img)
            
            estimate_points = []
            # go through all masks -> check if are roofs -> polygonize -> collect are polygon points
            for i, mask in enumerate(res.pred_instances['masks']):
                if res.pred_instances['scores'][i] < 0.5:
                    continue
                
                if res.pred_instances['labels'][i] != 0:
                    continue
                
                if mask.sum() < 1:
                    continue
                
                polygon_vertices = None
                mask = mask.cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cnt in contours:
                    epsilon = arc_length_coeff * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    area = cv2.contourArea(approx)

                    if len(approx) >= 3 and area > min_area_threshold:
                        polygon_vertices = approx.reshape(-1, 2).astype(np.float32)
                        estimate_points += [*polygon_vertices]
                
            new_shape = get_new_size(img.shape[:2])
            # pdb.set_trace()
            #img_r = cv2.resize(img, dsize=new_shape, interpolation=cv2.INTER_AREA)

            roof_points = []
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            stop = False
            # go through all the annotations for this image and create masks
            for ikey, key in enumerate(img_annotation.keys()):
                class_id = int(key) % NUM_CLASSES
                instance_id = img_id*MAX_ANNOTS_PER_IMG + ikey
                
                polygon  = list(map(lambda x: (x[1], x[2]), img_annotation[key]))

                # skip polygons with less than 3 vertices and edge classes, as they are not polygonizable
                if len(polygon) < 3 or class_id in [3, 5, 6, 7]: 
                    continue
                
                if len(polygon) > 10: # too "curly, probably not well annotated"
                    stop = True
                    continue
                
                if class_id == 0: # roof
                    roof_points += [*polygon]
                else:
                    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], (1, 1, 1))

            if stop:
                continue
            
            # # must be resized after fillpoly, as otherwise the polygons are not as precise
            # mask_r = cv2.resize(
            #     mask,
            #     dsize=(new_shape),
            #     interpolation=cv2.INTER_AREA,
            # )
            
            keypoints = []
            
            for i, pt in enumerate(roof_points):
                point = [int(pt[0]), int(pt[1])]
                point[0] = point[0] if point[0] < mask.shape[1] else mask.shape[1] - 1
                point[1] = point[1] if point[1] < mask.shape[0] else mask.shape[0] - 1
                
                try:
                    mask[point[1], point[0]] = 2
                except:
                    print('aa')
                if mask[point[1], point[0]] > 0:
                    keypoints += [(point[0], point[1], 1)]
                else:
                    keypoints += [(point[0], point[1], 2)]

            if len(keypoints) == 0:
                continue

            for estim_pt in estimate_points:
                # get img patch with keypoints that are inside
                patch, pts = get_patch(img, *estim_pt, np.array(keypoints))
                # draw point circles from pts into the patch using cv2
                
                new_name, ext = img_fn.split('.')
                
                new_name = f"{new_name}_{img_id}_.{ext}"
                images.append({"id": img_id, "file_name": new_name, "width": PATCH_SIZE, "height": PATCH_SIZE, "date_captured": "2013-11-18 02:53:27"}) # random date
                Image.fromarray(patch).save(os.path.join(out_imgs_path, new_name))  
                
                annotation = {
                    "image_id": img_id,
                    "keypoints": pts.tolist(),
                }
                
                annotations.append(annotation)
                img_id += 1
                

            if irow / len(rows) > thrs[filename]:
                break

        coco_data = {
            "images": images,
            "annotations": annotations
        }

        # Save the annotation JSON file
        with open(os.path.join(OUT_PATH, f'annotations_{filename}.json'), "w") as json_file:
            json.dump(coco_data, json_file, indent=4)


def generate_bbs(masks):
    bbs = []
    remove_ids = []
    for i, mask in enumerate(masks):
        pos = np.array(np.nonzero(mask)).T
        xmin = pos[:, 1].min()
        xmax = pos[:, 1].max()
        ymin = pos[:, 0].min()
        ymax = pos[:, 0].max()
        if (xmax - xmin) < 2 or (ymax - ymin) < 2:
            remove_ids.append(i)
            continue  #   skip single-line masks, since bounding boxes muse be at least 2x2
        bbs.append([float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)])
    return bbs, remove_ids

if __name__ == "__main__":
    generate()
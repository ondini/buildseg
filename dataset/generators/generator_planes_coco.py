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

DB_PATH =  "/home/kafkaon1/Dev/data/db_updated_28_01_24.db" # "/local2/homes/zderaann/roof_annotations/database_updates.db" #"/local/homes/zderaann/roof_annotations/database.db"
IN_IMGS_PATH = "/local2/homes/zderaann/roof_annotations/annotated_imgs"
IN_SAT_PATH = "/local2/homes/zderaann/roof_annotations/satelite"


OUT_PATH = '/home/kafkaon1/Dev/data/COCO_0904'

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
    

def generate(test=0.2, train=0.8):

    thrs = {'train': train, 'test': test + train}

    last_i = -1
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT * FROM annotation")

    rows = c.fetchall() # get all rows from the table = all annotations
    #random.shuffle(rows)
        
    categories = [{"id": i + 1, "name": CLASSES_COCO[i]} for i in range(NUM_COCO_CLASSES)]
    
    for filename in ['train', 'test']:
        out_imgs_path = os.path.join(OUT_PATH, f'{filename}_imgs') # image path

        if not os.path.exists(out_imgs_path):
            os.makedirs(out_imgs_path)
        
        # connect to database        
        
        images = []
        annotations = []

        for irow, row in enumerate(rows): # go through all the annotations
            if os.path.exists(os.path.join(OUT_PATH, row[0])): # image is already saved there
                continue
            
            if irow <= last_i:
                continue
            

            img_fn = row[0] # filename
            img_path = os.path.join(IN_IMGS_PATH, img_fn)
            img = cv2.imread(img_path)[:,:,::-1]
            img_annotation = json.loads(row[2])
            # print(img.shape)
            new_shape = get_new_size(img.shape[:2])
            # pdb.set_trace()
            img_r = cv2.resize(img, dsize=new_shape, interpolation=cv2.INTER_AREA)
            Image.fromarray(img_r).save(os.path.join(out_imgs_path, row[0]))

            img_id = irow + 1

            images.append({"id": img_id, "file_name": img_fn, "width": new_shape[0], "height": new_shape[1], "date_captured": "2013-11-18 02:53:27"}) # random date

            roof_masks = []
            roof_labels = []
            roof_ids = []
            
            masks = []
            labels = []
            inst_ids = []
            
            # go through all the annotations for this image and create masks
            for ikey, key in enumerate(img_annotation.keys()):
                class_id = int(key) % NUM_CLASSES
                instance_id = img_id*MAX_ANNOTS_PER_IMG + ikey
                
                polygon  = list(map(lambda x: (x[1], x[2]), img_annotation[key]))

                # skip polygons with less than 3 vertices and edge classes, as they are not polygonizable
                if len(polygon) < 3 or class_id in [3, 5, 6, 7]: 
                    continue
                
                instance_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(instance_mask, [np.array(polygon, dtype=np.int32)], (1, 1, 1))

                # must be resized after fillpoly, as otherwise the polygons are not as precise
                instance_mask_r = cv2.resize(
                    instance_mask,
                    dsize=(new_shape),
                    interpolation=cv2.INTER_AREA,
                )

                if instance_mask_r.sum() < 6:
                    continue  # skip empty masks -> MASKS WENt empty after resizing, if they were too small at the original image

                label = COCO_MAPPING[class_id] # map class id to COCO classes

                if class_id == 0: # roof
                    roof_masks.append(instance_mask_r)
                    roof_labels.append(1)
                    roof_ids.append(instance_id)
                else:
                    masks.append(instance_mask_r)
                    labels.append(label)
                    inst_ids.append(instance_id)
            
            # go through masks and crop out "remove" or "chimney" class masks from roof masks
            for i, mask in enumerate(masks):
                if not labels[i] in [2, 3, 4, 5]: # crop "remove" or "chimney", "other" or "window" from the roofplane
                    continue
                for j, roof_mask in enumerate(roof_masks): # go through roof masks
                    roof_masks[j] = roof_masks[j] * (1 - mask)
            
            # check if roof masks are empty after cropping, otherwise append to masks
            for i, roof_mask in enumerate(roof_masks):
                if roof_mask.sum() < 1:
                    continue
                masks.append(roof_mask)
                labels.append(1)
                inst_ids.append(roof_ids[i])
            
            bbs, remove_ids = generate_bbs(masks)
            
            if len(remove_ids) > 0: # some bbs were removed, remove labels and masks on this id
                # print('removing', remove_ids)
                for remove_id in remove_ids:     # remove ids  from  tensors
                    masks = masks[:remove_id] + masks[remove_id+1:]
                    labels = labels[:remove_id] + labels[remove_id+1:]
                    inst_ids = inst_ids[:remove_id] + inst_ids[remove_id+1:]

            for i, mask in enumerate(masks):
                if labels[i] > 4: # remove class - used only for cropping, not recognized by COCO
                    print('aaaaaaa')
                    continue
                # polygonize masks back as it is required by COCO format
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)

                segmentation = []

                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) > 4:
                        segmentation.append(contour)
                if len(segmentation) == 0:
                    continue
                
                annotation = {
                        "image_id": img_id,
                        "id": inst_ids[i],
                        "category_id": labels[i],
                        "segmentation": segmentation,
                        "bbox": bbs[i],
                        "area": int(mask.sum()),
                        "iscrowd": 0,
                    }
                
                annotations.append(annotation)

            
            if irow / len(rows) > thrs[filename]:
                last_i = irow
                break

        coco_data = {
            "info": {},
            "licenses": [],
            "categories": categories,
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
import sqlite3
import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

# from shapely.geometry import Polygon
# from shapely.ops import unary_union, cascaded_union

from PIL import Image, ImageDraw
import json
  
from collections import defaultdict
  
IMG_SHAPE = (960, 560)

DB_PATH =  "/home/kafkaon1/Dev/data/db_updated_28_01_24.db" # "/local2/homes/zderaann/roof_annotations/database_updates.db" #"/local/homes/zderaann/roof_annotations/database.db"
IN_IMGS_PATH = "/local2/homes/zderaann/roof_annotations/annotated_imgs"
IN_SAT_PATH = "/local2/homes/zderaann/roof_annotations/satelite"


OUT_PATH = '/home/kafkaon1/Dev/data/CZISUPD2lll'

# create output directories
out_sat_imgs_path = os.path.join(OUT_PATH, 'sat_imgs')
out_sat_labels_path = os.path.join(OUT_PATH, 'sat_labels')

out_fac_imgs_path = os.path.join(OUT_PATH, 'fac_imgs')
out_fac_labels_path = os.path.join(OUT_PATH, 'fac_labels')


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
];

CLASSES = [
  "roof",
  "window",
  "remove",
  "chimney",
  "other",
  "ridge",
  "sideedge",
  "edge",
  "chimedge",
];

NUM_CLASSES = len(CLASSES)


def generate():
    
    if not os.path.exists(out_sat_imgs_path):
        os.makedirs(out_sat_imgs_path)
        os.makedirs(out_sat_labels_path)

    if not os.path.exists(out_fac_imgs_path):
        os.makedirs(out_fac_imgs_path)
        os.makedirs(out_fac_labels_path)
    
    # connect to database        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT * FROM annotation")

    rows = c.fetchall()

    for row in rows:
        if os.path.exists(os.path.join(OUT_PATH, row[0])):
            continue

        img_fn = os.path.join(IN_IMGS_PATH, row[0])
        sat_fn = os.path.join(IN_SAT_PATH, row[1])

        img = cv2.imread(img_fn)[:,:,::-1]
        sat = cv2.imread(sat_fn)[:,:,::-1]

        img_annotation = json.loads(row[2])
        sat_annotation = json.loads(row[3])

        annotations = []
        instance_ids = defaultdict(int)
        
        # go through all the annotations and create a list of polygons
        for key in img_annotation.keys():
            class_id = int(key)%NUM_CLASSES
            
            instance_id = instance_ids[class_id]
            instance_ids[class_id] += 1
            polygon  = list(map(lambda x: [x[1], x[2]], img_annotation[key]))
            annotation = {
                "instance_id": instance_id,
                "class_id": class_id,
                "polygon": polygon,
                "img_size_old": img.shape,
                "img_size_new": IMG_SHAPE
            }
            annotations.append(annotation)
        
        # Save the annotation JSON file
        with open(os.path.join(out_fac_labels_path, row[0].split('.')[0] + '.json'), "w") as json_file:
            json.dump(annotations, json_file, indent=4)

        if (img.shape[1] < img.shape[0]):
            continue
        img = cv2.resize(img, dsize=IMG_SHAPE, interpolation=cv2.INTER_AREA)
        #plt.imsave(os.path.join(out_fac_imgs_path, row[0]), img)
        Image.fromarray(img).save(os.path.join(out_fac_imgs_path, row[0]))

        annotations = []
        instance_ids = defaultdict(int)
        for key in sat_annotation.keys():
            class_id = int(key)%5
            instance_id = instance_ids[class_id]
            instance_ids[class_id] += 1
            polygon = list(map(lambda x: [x[1], x[2]], sat_annotation[key]))
            annotation = {
                "instance_id": instance_id,
                "class_id": class_id,
                "polygon": polygon,
                "img_size_old": sat.shape,
                "img_size_new": IMG_SHAPE
            }

        #   Save the annotation JSON file
        with open(os.path.join(out_sat_labels_path, '.'.join(row[1].split('.')[:-1])+ '.json'), "w") as json_file:
            json.dump(annotations, json_file, indent=4)

        # plt.imsave(os.path.join(out_sat_imgs_path, row[1]), sat)
        Image.fromarray(sat).save(os.path.join(out_sat_imgs_path, row[1]))
        


def create_train_test_split(ratio=0.7):
    imgs = os.listdir(out_fac_imgs_path)
    random.shuffle(imgs)
    
    train_imgs = imgs[:int(len(imgs)*ratio)]
    test_imgs = imgs[int(len(imgs)*ratio):]

    with open(os.path.join(OUT_PATH, 'train_fac.txt'), 'w') as f:
        for img in train_imgs:
            f.write(img + '\n')

    with open(os.path.join(OUT_PATH, 'test_fac.txt'), 'w') as f:
        for img in test_imgs:
            f.write(img + '\n')

    imgs = os.listdir(out_sat_imgs_path)
    random.shuffle(imgs)

    train_imgs = imgs[:int(len(imgs)*ratio)]
    test_imgs = imgs[int(len(imgs)*ratio):]

    with open(os.path.join(OUT_PATH, 'train_sat.txt'), 'w') as f:
        for img in train_imgs:
            f.write(img + '\n')

    with open(os.path.join(OUT_PATH, 'test_sat.txt'), 'w') as f:
        for img in test_imgs:
            f.write(img + '\n')



if __name__ == "__main__":
    generate()
    create_train_test_split()
import sqlite3
import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

from shapely.geometry import Polygon
from shapely.ops import unary_union, cascaded_union

from PIL import Image, ImageDraw

DB_PATH = "/local/homes/zderaann/roof_annotations/database.db"
IN_IMGS_PATH = "/local/homes/zderaann/roof_annotations/annotated_imgs"
IN_SAT_PATH = "/local/homes/zderaann/roof_annotations/satelite"

OUT_PATH = '/home/kafkaon1/Dev/FVAPP/data/CZ99'

# create output directories
out_sat_imgs_path = os.path.join(OUT_PATH, 'sat_imgs')
out_sat_labels_path = os.path.join(OUT_PATH, 'sat_labels')

out_fac_imgs_path = os.path.join(OUT_PATH, 'fac_imgs')
out_fac_labels_path = os.path.join(OUT_PATH, 'fac_labels')

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

        polygons = []
        for key in img_annotation.keys():
            if (int(key)%5) != 0:
                continue
            polygon = img_annotation[key]
            polygons.append(list(map(lambda x: (x[1], x[2]), polygon)))
        
        merge_polygons(polygons)
        mask = create_mask(polygons, (img.shape[1], img.shape[0]))

        if (img.shape[1] < img.shape[0]):
            continue
        #img_ratio = 640/max(img.shape)
        new_shape = (960, 560) #(int(img.shape[1] * img_ratio), int(img.shape[0] * img_ratio))
        img = cv2.resize(img, dsize=new_shape, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, dsize=new_shape, interpolation=cv2.INTER_AREA)

        #plt.imsave(os.path.join(out_fac_imgs_path, row[0]), img)
        Image.fromarray(img).save(os.path.join(out_fac_imgs_path, row[0]))
        #plt.imsave(os.path.join(out_fac_labels_path, row[0]), mask)
        Image.fromarray(mask).save(os.path.join(out_fac_labels_path, row[0]))

        polygons = []
        for key in sat_annotation.keys():
            if (int(key)%5) != 0:
                continue
            polygon = sat_annotation[key]
            polygons.append(list(map(lambda x: (x[1], x[2]), polygon)))
        
        merge_polygons(polygons)
        mask = create_mask(polygons, (sat.shape[1], sat.shape[0]))
        # plt.imsave(os.path.join(out_sat_imgs_path, row[1]), sat)
        Image.fromarray(sat).save(os.path.join(out_sat_imgs_path, row[1]))
        #plt.imsave(os.path.join(out_sat_labels_path, row[1]), mask)
        Image.fromarray(mask).save(os.path.join(out_sat_labels_path, row[1]))
        


import math

def euclidean_distance(p1, p2):
    # Calculate the Euclidean distance between two points (vertices)
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def merge_vertices(poly1, poly2, distance_threshold=7):
    # Merge vertices between two polygons if they are closer than the distance_threshold

    for i in range(len(poly1)):
        for j in range(len(poly2)):
            dist = euclidean_distance(poly1[i], poly2[j])
            if dist <= distance_threshold:
                poly1[i] = poly2[j] = ((poly1[i][0] + poly2[j][0])/2, (poly1[i][1] + poly2[j][1])/2)

    return [poly1, poly2]

def merge_polygons(polygons):
    merged_polygons = []

    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            res = merge_vertices(polygons[i], polygons[j]) # try to merge with every other polygon
            polygons[i], polygons[j] = res


    return merged_polygons


def create_mask(polygons, image_size):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)

    for poly in polygons:
        if len(poly)<2:
            continue
        vertices = [(int(x), int(y)) for x, y in poly]
        draw.polygon(vertices, outline=1, fill=1)

    return np.array(mask)

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
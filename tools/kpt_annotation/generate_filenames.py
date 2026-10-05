import os
import sys
import json
import random
import numpy as np
from collections import defaultdict
import sqlite3

IN_IMGS_PATH = "/local2/homes/zderaann/roof_annotations/annotated_imgs"
NEW_PATH = "./kpts_imgs"
DB_PATH =  "/home/kafkaon1/Dev/data/db_updated_05_03_24.db"

def select_random_images():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM annotation")
    rows = c.fetchall() # get all rows from the table = all annotations
    c.close()
    
    # Load the images
    #imgs = os.listdir(IN_IMGS_PATH)
    imgs = [row[0] for row in rows]    
    imgs = imgs[:int(len(imgs)*0.6)] # only first 60% of images, to not interfere with val data
    
    #select randomly 100 images
    rimg = random.sample(imgs, 50) 
    
    used_satelite_imgs = set()

    # Save the filenames to a file
    with open("data/kp_filepaths.txt", "w") as f:
        for img in rimg:
            img_path = os.path.join(IN_IMGS_PATH, img)
            f.write(img_path + "\n")
            
    # and copy the files to NEW_PATH
    if not os.path.exists(NEW_PATH):
        os.makedirs(NEW_PATH)
        
    for img in rimg:
        img_path = os.path.join(IN_IMGS_PATH, img)
        new_img_path = os.path.join(NEW_PATH, img)
        os.system(f"cp {img_path} {NEW_PATH}")
        
def select_images_with_satellite():
    pass

if __name__ == "__main__":
    select_random_images()
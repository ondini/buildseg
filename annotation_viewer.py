import sqlite3
import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


DB_PATH = "/local/homes/zderaann/roof_annotations/database.db"
OUT_PATH = '/home/kafkaon1/FVAPP/vizes'
IMG_PATH = "/local/homes/zderaann/roof_annotations/annotated_imgs"
SAT_PATH = "/local/homes/zderaann/roof_annotations/satelite"

def visualize():
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT * FROM annotation WHERE img_pic_name='20230503_1934141.jpg'")

    rows = c.fetchall()

    for row in rows:
        # if os.path.exists(os.path.join(OUT_PATH, row[0])):
        #     continue

        img_fn = os.path.join(IMG_PATH, row[0])
        sat_fn = os.path.join(SAT_PATH, row[1])

        img = cv2.imread(img_fn)
        sat = cv2.imread(sat_fn)

        img_annotation = json.loads(row[2])
        sat_annotation = json.loads(row[3])


        plt.figure(figsize=(13,6))
        # plot images side by side with annotations on top
        plt.subplot(1,2,1)
        plt.imshow(img)
        for label in img_annotation.keys():
            for x in img_annotation[label]:
                if x[0] == 'N':
                    x[0] = -1
            poly = np.array(img_annotation[label])
            if poly.any():
                poly = np.vstack((poly, poly[0,:]))
                plt.plot(poly[:,1], poly[:,2])


        plt.axis('off')
        plt.tight_layout()

        plt.subplot(1,2,2)
        plt.imshow(sat)
        for label in sat_annotation.keys():
            for x in sat_annotation[label]:
                if x[0] == 'N':
                    x[0] = -1
            poly = np.array(sat_annotation[label])
            if poly.any():
                poly = np.vstack((poly, poly[0,:]))
                plt.plot(poly[:,1], poly[:,2])

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_PATH, row[0]), bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    visualize()
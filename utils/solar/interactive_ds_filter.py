import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import sqlite3
import torch
use_trips = False
from torch.utils.tensorboard import SummaryWriter


DB_PATH =  "/home/kafkaon1/Dev/data/db_updated_05_03_24.db" # "/local2/homes/zderaann/roof_annotations/database_updates.db" #"/local/homes/zderaann/roof_annotations/database.db"
IN_IMGS_PATH = "/local2/homes/zderaann/roof_annotations/annotated_imgs"
IN_SAT_PATH = "/local2/homes/zderaann/roof_annotations/satelite"
FIG_PATH = '/home/kafkaon1/Dev/pp.png'

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

NUM_CLASSES = len(CLASSES)

def save_seqs(seq_g, seq_b, save_root):
    seq_g_path = os.path.join(save_root, 'good_anns.txt')
    seq_b_path = os.path.join(save_root, 'bad_anns.txt')

    with open(seq_g_path, 'w') as f:
        for seq in seq_g:
            f.write(seq + '\n')
    
    with open(seq_b_path, 'w') as f:
        for seq in seq_b:
            f.write(seq + '\n')

def get_seqs(save_root):
    # get unique sequences in ds

    bad_seqs = set()
    good_seqs = set()
    seq_g_path = os.path.join(save_root, 'good_seqs.txt')
    seq_b_path = os.path.join(save_root, 'bad_seqs.txt')

    if os.path.exists(seq_g_path):
        with open(seq_g_path, 'r') as f:
            for line in f:
                good_seqs.add(line.strip())
    
    if os.path.exists(seq_b_path):
        with open(seq_b_path, 'r') as f:
            for line in f:
                bad_seqs.add(line.strip())


    return good_seqs, bad_seqs


def main(args):
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)


    good_seqs, bad_seqs = get_seqs(args.save_root)

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
    print(f"Starting")
    
    writer = SummaryWriter('log_dir')    
    # random.shuffle(rows)


    for irow, row in enumerate(rows): # go through all the annotations
        img_fn = row[0] # filename
        if img_fn in good_seqs or img_fn in bad_seqs:
            continue
        
        print(f"{irow/len(rows)*100:.2f}%, imgname: {img_fn}")
        
        img_path = os.path.join(IN_IMGS_PATH, img_fn)
        img = cv2.imread(img_path)[:,:,::-1]
        img_annotation = json.loads(row[2])
      
        fig, ax = plt.subplots(1,1, figsize=(20, 30))
        ax.axis('off')
        ax.imshow(img)
        
        for ikey, key in enumerate(img_annotation.keys()):
            class_id = int(key) % NUM_CLASSES
            if not class_id in [0, 2]:
                continue
            
            polygon = np.array(list((map(lambda x: (x[1], x[2]), img_annotation[key]))))
            color = (0,255,0) if class_id == 0 else (255,0,0)
            if polygon.any():
                poly = np.vstack((polygon, polygon[0,:]))
                #ax.plot(poly[:,0], poly[:,1], color)
                img = cv2.polylines(img.astype(np.uint8), [poly.astype(np.int32)], False, color, 2)
        
        # fig.tight_layout()
        # fig.savefig(FIG_PATH)
        # fig.show()

        # for some image "im"
        # imt = torch.from_numpy(img).permute(2,0,1)
        # writer.add_image('My image', imt, 0)
        # writer.close()
        # resize img to half the size
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        plt.imsave(FIG_PATH, img)

        act = input("Press...")
        if act == 'q':
            break
        elif act == 'g':
            bad_seqs.add(img_fn)
        else:
            good_seqs.add(img_fn)

        if irow % 10 == 0:
            print('Saving files...')
            save_seqs(good_seqs, bad_seqs, args.save_root)
            
    print('Saving files...')
    save_seqs(good_seqs, bad_seqs, args.save_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-id', '--input_seq', default='/app/data/offshore_dtst/seqs_filter/28_02_2024/good_seqs.txt', type=str,
                    help='dataset name')
    parser.add_argument('-s', '--save_root', default='/home/kafkaon1/Dev/data/ds_filtered', type=str,
                        help='root to save the filtered sequnces')
    
    args = parser.parse_args()
    main(args)
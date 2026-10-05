import cv2
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

# Function to collect keypoints
def collect_keypoints(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2)) 
    # Create a copy of the image for drawing
    image_copy = image.copy()

    # List to store keypoints
    keypoints = []

    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal keypoints
        if event == cv2.EVENT_LBUTTONDOWN:
            keypoints.append((x, y))
            cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('image', image_copy)

    # Create a window and set the mouse callback
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    # Display the image
    cv2.imshow('image', image)

    # Wait for the user to click 'q' to quit
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return keypoints

def collect_keypoints_mpl(image_path, image_name, points):

    # Load the image
    image = plt.imread(image_path)

    # List to store keypoints
    keypoints = []
    chimneypoints = []
    # Function to handle mouse click event
    def onclick(event):
        if event.button == 3:  # Left mouse button
            keypoints.append((event.xdata, event.ydata))
            plt.scatter(event.xdata, event.ydata, color='r', s=5)
            plt.draw()
        if event.button == 2:
            chimneypoints.append((event.xdata, event.ydata))
            plt.scatter(event.xdata, event.ydata, color='b', s=5)
            plt.draw()
            
            
    def on_key(event):
        if event.key == 'q':
            plt.close()
        if event.key == 'c':
            sys.exit()
        elif event.key == 'v':  # Switch to zoom tool
            fig.canvas.toolbar.zoom()
            
            
    # Display the imageW
    fig, ax = plt.subplots(figsize=(15,20))
    ax.imshow(image)
    if len(points) > 0:
        pts = np.array(points)
        ax.scatter(pts[:, 0], pts[:, 1], color='lime', s=5)
        
    ax.set_title(image_name)
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key)


    # Show the plot
    plt.show()
    kpts = np.array(keypoints)
    chpts = np.array(chimneypoints)
    # find closes point from pts to each point in kpts usin numpy matrix
    if len(points) > 0:
        for i, kpt in enumerate(kpts):
            dists = np.linalg.norm(kpt - pts, axis=1)
            min_idx = np.argmin(dists)
            min_dis = dists[min_idx]
            if min_dis < 25:
                pts[min_idx] = kpt
            else:
                pts = np.vstack([pts, kpt])
                
        for i, chpt in enumerate(chpts):
            dists = np.linalg.norm(chpt - pts, axis=1)
            min_idx = np.argmin(dists)
            min_dis = dists[min_idx]
            if min_dis < 25:
                print('aaa')
                pts = np.delete(pts, min_idx, axis=0)
    else:
        pts = kpts
        
    res = pts.tolist()
    rs = pts
    # show the final points again
    fig, ax = plt.subplots(figsize=(15,20))
    ax.imshow(image)
    if len(rs) > 0:
        ax.scatter(rs[:, 0], rs[:, 1], color='r', s=5)
    if len(chpts) > 0:
        ax.scatter(chpts[:, 0], chpts[:, 1], color='b', s=5)
    ax.set_title(image_name)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return res, chpts.tolist()

def show_pts(image_dir, pts):           
    for key, value in pts.items():
        print(key, value)
        image_path = os.path.join(image_dir, key)
        image = plt.imread(image_path)
        fig, ax = plt.subplots(figsize=(15,20))
        ax.imshow(image)
        for pt in value:
            ax.scatter(pt[0], pt[1], color='r', s=5)
        plt.title(key)
        plt.show()


if __name__ == "__main__":
    # Example usage
    image_dir = '/home/ondin/Developer/FVAPP/kpts_imgs'
    in_file = 'ptsR.json'
    out_file_chimney = 'chim.json'
    out_file = 'ptsRE.json'
    imgs = os.listdir(image_dir)
    with open(in_file) as f:
        pts = json.load(f)
    
    with open(out_file_chimney) as f:
        cpts = json.load(f)
        
    with open(out_file) as f:
        outpts = json.load(f)
        
    print(len(outpts))
    for i, img in enumerate(imgs):
        points = []
        if img in outpts:
            print(f'{img} already annotated.')
            continue
        if img in pts:
            points = pts[img]
        #print(cpts[img])
        image_path = os.path.join(image_dir, img)
        keypoints, chimpoints = collect_keypoints_mpl(image_path, img, points)
        print("Key Points:", keypoints)
        outpts[img] = keypoints
        cpts[img] = chimpoints
        
        if i%2 == 0:
            print('Saving')
            with open(out_file, 'w') as f:
                json.dump(outpts, f)
                
            with open(out_file_chimney, 'w') as f:
                json.dump(cpts, f)
    
    with open(out_file, 'w') as f:
        json.dump(outpts, f)

    with open(out_file_chimney, 'w') as f:
        json.dump(cpts, f)
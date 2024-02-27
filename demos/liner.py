

import torch
import os
import sys
import cv2



from utilss import  pred_lines


    
        
def find_polygon(model, img, size):
    imgr = cv2.resize(img, (512, 512))
    imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
    lines = pred_lines(imgr, modell, [512, 512], 0.1, 20)


    coeffs = img.shape[0]/512, img.shape[1]/512
    line_image = np.zeros(img.shape[:2])
    liness = []
    for line in lines:
        l = int(line[0]*coeffs[1]), int(line[1]*coeffs[0]), int(line[2]*coeffs[1]), int(line[3]*coeffs[0])
        x1, y1, x2, y2 = l

        # Check if at least 50% of the line lies within the mask
        line_mask = np.zeros_like(dilation)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

        intersection = line_mask * dilation
        
        # Calculate the percentage of the line within the mask
        percentage_inside_mask = (np.sum(intersection) / np.sum(line_mask)) * 100
        
        # If at least 50% of the line is inside the mask, draw it
        if percentage_inside_mask >= 20 and percentage_inside_mask < 100:
            liness += [l]
            cv2.line(line_image, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), 255, 2)
        
    
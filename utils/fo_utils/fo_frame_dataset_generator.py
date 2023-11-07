import fiftyone as fo
import fiftyone.core.labels as fol
import random
import numpy as np
import cv2
import os
import argparse

from tqdm import tqdm
random.seed(51)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-i', '--input_name', default="SEAAIVid", type=str,
                        help='input (video) dataset name')
    parser.add_argument('-o', '--output_name', default="SEAAIFr", type=str,
                        help='output (frames) dataset name')
    parser.add_argument('-r', '--sampling_rate', default=20, type=float,
                        help='sampling rate - every r-th image from input videos is taken (default: 10)')
    args = parser.parse_args()

    dataset = fo.load_dataset(args.input_name)
    dataset.group_slice = 'thermal_narrow'

    fps = dataset.values('filepath')
    frame_period = 10

    samples =  []    
    for i, fp in enumerate(fps):
        print(f'Processing video: {i}/{len(fps)}', fp)
        group = dataset.get_group(dataset[fp].group.id)
        if not 'rgb_narrow' in group or not 'thermal_narrow' in group:
            print('No rgb or thermal narrow group')
            continue
    
        if np.abs(len(group['thermal_narrow'].frames) - len(group['rgb_narrow'].frames)) > 2: 
            print('Frame count mismatch of', (len(group['thermal_narrow'].frames)- len(group['rgb_narrow'].frames)), ' on ', \
                  len(group['thermal_narrow'].frames), len(group['rgb_narrow'].frames))
            continue
    
        video_t, video_r = group['thermal_narrow'], group['rgb_narrow']
        path_t, path_r = video_t.filepath, video_r.filepath
        cap_t, cap_r = cv2.VideoCapture(path_t), cv2.VideoCapture(path_r)
        filepath_t, filepath_r = '.'.join(path_t.split('.')[:-1]), '.'.join(path_r.split('.')[:-1])
        os.makedirs(filepath_t, exist_ok=True)
        os.makedirs(filepath_r, exist_ok=True)
        
        for i in tqdm(video_t.frames):
            frame = video_t.frames[i]
            ret, vidframe_t = cap_t.read()
            if not ret:
                break
            ret, vidframe_r = cap_r.read()
            if not ret:
                break

            fo_group = fo.Group()
            detections = []

            if (i-1)%frame_period == 0 and frame.ground_truth_det is not None and len(frame.ground_truth_det.detections)>0:
                cv2.imwrite(f'{filepath_t}/{i}.png', vidframe_t)
                cv2.imwrite(f'{filepath_r}/{i}.png', vidframe_r)

                for det in frame.ground_truth_det.detections:
                    box = det.bounding_box
                    detections.append(
                        fo.Detection(label=det.label, bounding_box=det.bounding_box)
                    )
                
                sample_t = fo.Sample(filepath=f'{filepath_t}/{i}.png', group=fo_group.element('thermal_narrow'))
                sample_t["ground_truth_det"] = fo.Detections(detections=detections)

                sample_r = fo.Sample(filepath=f'{filepath_r}/{i}.png', group=fo_group.element('rgb_narrow'))
                samples.extend([sample_t, sample_r])
        
        cap_t.release()
        cap_r.release()

    dataset_fr = fo.Dataset(name=args.output_name)
    dataset_fr.add_samples(samples)
    dataset_fr.persistent = True 
    dataset_fr.save()
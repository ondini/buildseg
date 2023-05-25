import os
import datetime
import argparse
import logging
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader

from models import createDeepLabv3, createDeepLabv3Plus
from dataset import FVDataset
from metrics import GetMetricFunction, Metrics
from configs import model_weights

def evaluate(args):
    start = datetime.datetime.now()
    run_name = f"run_{start:%y%m%d-%H%M%S}"
    out_path = os.path.join(args.out_path, run_name)

    os.makedirs(out_path)

    val_data_path = os.path.join(args.dataset_path, 'val/image_resized')
    val_label_path = os.path.join(args.dataset_path, 'val/label_resized')
    val_list_path = os.path.join(args.dataset_path, 'val.txt')

    logging.basicConfig(filename=os.path.join(out_path, 'logging.log'), level=logging.INFO)

    valid_dataset = FVDataset(
        val_data_path, val_label_path, val_list_path,
        augmentation=False,
        size_coefficient=1/20
    )

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    results_final = {}
    for model_name in model_weights.keys():
        model_path = model_weights[model_name]
        model = torch.load(model_path)

        model.to(device)
        model.eval()

        desc = f'FV: {run_name} \n \
            Model: {model_name} \n \
            Checkpoint: {model_path} \n \
            Batch size: {args.batch_size} \n \n'

        logging.info(desc)

        y_true, y_pred = [], []

        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = model.predict(inputs)

            y_true.append(targets.cpu())
            y_pred.append(predictions.cpu())
        
        y_true, y_pred = torch.concat(y_true,0), torch.concat(y_pred, 0)

        metric_names = ["iou", "dice", "precision", "recall", "matthews"]
        metrics = {
            metric_name : GetMetricFunction(metric_name, reduction='none') for metric_name in metric_names
        } 
        metrics.update( {
            f"{metric_name}_b{1}" : GetMetricFunction(metric_name, 1, reduction='none') for metric_name in metric_names
        })
        metrics.update({
            f"{metric_name}_b{3}" : GetMetricFunction(metric_name, 3, reduction='none') for metric_name in metric_names
        } )
        metrics.update({
            f"{metric_name}_b{5}" : GetMetricFunction(metric_name, 5, reduction='none') for metric_name in metric_names
        } )
        metrics.update({
            f"{metric_name}_b{7}" : GetMetricFunction(metric_name, 7, reduction='none') for metric_name in metric_names
        } )

        results = {}
        for metric_name, metric in metrics.items():
            res = metric(y_true, y_pred)
            logging.info(f'{metric_name}: {res.mean()}')
            results[metric_name] = res
        
        results_final[model_name] = results
        
    np.save(os.path.join(out_path,"results.npy"), results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FVApp evaluation script')
    parser.add_argument("--device", type=str,  choices={'cuda:0', 'cuda:1', 'cpu'}, default='cuda:1')
    parser.add_argument("--out_path", type=str, default='/home/kafkaon1/FVAPP/out/eval')
    parser.add_argument("--dataset_path", type=str, default='/home/kafkaon1/FVAPP/data/FV')
    parser.add_argument("--batch_size", type=int, default=10)

    args = parser.parse_args()

    evaluate(args)
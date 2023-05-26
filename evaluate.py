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

from models import DLV3Reg, extractPolygons, labelFromPolygons
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
        size_coefficient=args.dataset_coefficient,
    )

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    results_final = {}
    for model_name in model_weights.keys():
        model_path = model_weights[model_name]
        if args.use_reg or args.use_sam or args.get_poly:
            model = DLV3Reg(model_path, do_reg=args.use_reg, do_sam=args.use_sam, do_poly=args.get_poly)
        else:
            model = torch.load(model_path)

        model.to(device)
        model.eval()

        desc = f'FV: {run_name} \n \
            Model: {model_name} \n \
            Checkpoint: {model_path} \n \
            Batch size: {args.batch_size} \n \
            Use reg: {args.use_reg} \n \
            Use sam: {args.use_sam} \n \
            Get poly: {args.get_poly} \n \
            Hard pred: {args.hard_pred} \n \
            Dataset coefficient: {args.dataset_coefficient} \n \
            Device: {device} \n \n'

        logging.info(desc)

        y_true, y_pred = [], []

        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = model.predict(inputs)

            y_true.append(targets.cpu().float())
            if args.hard_pred:
                y_pred.append((predictions.cpu()>0.5).float())
            else:
                y_pred.append(predictions.cpu().float())

        
        y_true, y_pred = torch.concat(y_true,0), torch.concat(y_pred, 0)

        metric_names = ["iou", "dice"] #, "matthews", "precision", "recall"]
        metrics = {
            metric_name : GetMetricFunction(metric_name) for metric_name in metric_names
        } 
        metrics.update( {
            f"{metric_name}_b{1}" : GetMetricFunction(metric_name, 1) for metric_name in metric_names[1:]
        })
        metrics.update({
            f"{metric_name}_b{3}" : GetMetricFunction(metric_name, 3) for metric_name in metric_names[1:]
        } )
        metrics.update({
            f"{metric_name}_b{5}" : GetMetricFunction(metric_name, 5) for metric_name in metric_names[1:]
        } )
        
        results = {}
        for metric_name, metric in metrics.items():
            res = metric(y_true, y_pred)
            logging.info(f'{metric_name}: {res.mean()}')
            results[metric_name] = res
        
        results_final[model_name] = results
        
    np.save(os.path.join(out_path,"results.npy"), results_final)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FVApp evaluation script')
    parser.add_argument("--device", type=str,  choices={'cuda:0', 'cuda:1', 'cpu'}, default='cuda:1')
    parser.add_argument("--out_path", type=str, default='/home/kafkaon1/FVAPP/out/eval')
    parser.add_argument("--dataset_path", type=str, default='/home/kafkaon1/FVAPP/data/FV')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_reg", type=bool, default=True)
    parser.add_argument("--use_sam", type=bool, default=False)
    parser.add_argument("--hard_pred", type=bool, default=False)
    parser.add_argument("--get_poly", type=bool, default=True)
    parser.add_argument("--dataset_coefficient", type=float, default=1/10)

    args = parser.parse_args()

    evaluate(args)
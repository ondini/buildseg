import argparse
import torch
from torch.utils.data import DataLoader
from os import makedirs
from sklearn.metrics import classification_report
from tqdm import tqdm
import pathlib
from matplotlib import pyplot as plt
import logging
from yaml import load, CLoader as Loader

from infrastructure import Experiment, load_yaml
import experiments # otherwise experiments will not be known
import datetime
import os
from dataset import FVDataset

def evaluate(args):
    start = datetime.datetime.now()
    run_name = f"run_{start:%y%m%d-%H%M%S}"
    out_log_path = os.path.join(args.out_path, run_name+'/logs/')

    os.makedirs(out_log_path)

    val_data_path = os.path.join(args.dataset_path, 'val/image_resized')
    val_label_path = os.path.join(args.dataset_path, 'val/label_resized')
    val_list_path = os.path.join(args.dataset_path, 'val.txt')

    # Configure the logger
    logging.basicConfig(filename=os.path.join(out_log_path, 'logging.log'), level=logging.INFO)

    valid_dataset = FVDataset(
        val_data_path, val_label_path, val_list_path,
        augmentation=False,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=4, persistent_workers=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    metrics = {
        "iou": IoU,
        "iou_b": IoU_b,
        "dice": dice_coefficient,
        "dice_b": dice_coefficient_boundary,
        "precision": precision,
        "precision_b": precision_boundary,
        "recall": recall,
        "recall_b": recall_boundary
    }

    checkpoint_metric = 'loss'

    if args.checkpoint_path:
        model = torch.load(args.checkpoint_path)
    else:
        model = createDeepLabv3Plus(1)
    model.to(device)
    model.eval()

    train_logs, valid_logs = [], []

    desc = f'FV: {run_name} \n \
         Model: {args.model} \n \
         Checkpoint: {args.checkpoint_path} \n \
         Dataset: {args.dataset_coeff:.2f} \n \
         Batch size: {args.batch_size_train} \n \
         Loss: {loss} \n \n'

         # log also the learning rate and if scheduler is used
    
    logging.info(desc)

    y_true, y_pred = [], []

    for inputs, targets in tqdm(valid_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        predictions = model.predict(inputs)

        y_true.append(targets.cpu())
        y_pred.append(predictions.cpu())
    
    y_true, y_pred = torch.hstack(y_true), torch.hstack(y_pred)




    makedirs(args["output"], exist_ok=True)
    with open(args["output"] / "eval-report.txt", "w") as fout:
        # TODO: Write your code here
        fout.write(classification_report(y_true.cpu().numpy(), y_pred.cpu().numpy()))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FVApp evaluation script')

    parser.add_argument('-i', '--input',type=pathlib.Path, required=True, help='input config')
    parser.add_argument('--dataset',type=str, default="test", help='which dataset to evalute')
    parser.add_argument('--ckpt',type=pathlib.Path, required=True, help='checkpoint path')
    parser.add_argument('--output',type=pathlib.Path, required=True, help='eval output path')
    parser.add_argument('--batchsize',type=int, default=32, help='batchsize')
    parser.add_argument('--device',type=str, default="cpu", help='device')
    parser.add_argument('--raw', action="store_true", default=False, help='plot raw unsmoothed data only')
    parser.add_argument('--num-dl-workers',type=int, default=1, help='num dataloader workers (converting images on the fly)')

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d ] %(message)s')
    
    args = parser.parse_args()

    evaluate(args)
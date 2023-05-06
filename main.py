import os
import random
import datetime
import argparse
import logging
import logging

import matplotlib.pyplot as plt
import numpy as np

# torch optim LR scheduler - one cycle LR 

import torch
from torch.utils.data import DataLoader

from models import createDeepLabv3, createDeepLabv3Plus
from dataset import FVDataset
from losses import CombLoss, BinaryDiceLoss, DistanceLoss, DistanceWeightBCELoss, BoundaryLoss, FocalLoss
from metrics import IoU, IoU_b, dice_coefficient, dice_coefficient_boundary, precision, precision_boundary, recall, recall_boundary

def run_epoch(model, device, loss_fn, dataloader, optimizer, metrics, mode, lr_scheduler):
    logs = {metric_name: [] for metric_name in metrics.keys()}
    logs['loss'] = []

    logging.info(f"> Starting {mode} epoch.")
    if mode == 'train':
        model.train() 
    else:
        model.eval()  

    for  i, (inputs , target) in enumerate(dataloader):
        logging.info(f"-> Epoch iteration {i}.")
        inputs = inputs.to(device)
        target = target.to(device)
        model.zero_grad()

        with torch.set_grad_enabled(mode == 'train'): # gradient calculation only in train mode
            outputs = model(inputs)
            loss = loss_fn(outputs.float(), target.float())

            for metric_name, metric_fn in metrics.items():
                metric_value = metric_fn(outputs, target).cpu().item()
                logs[metric_name].append(metric_value)

            logs['loss'].append(loss.cpu().item())

            if mode == 'train':
                loss.backward()
                # gradient clipping - to avoid exploding gradients 
                # test the clipping threshold in log scale
                # compute average gradient norm and log it
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                # gradient normalization


                optimizer.step()
                #lr_scheduler.step()
    
    logs = {metric_name: metric_values for metric_name, metric_values in logs.items()}
    return logs

def main(args):
    
    start = datetime.datetime.now()
    run_name = f"run_{start:%y%m%d-%H%M%S}"
    out_wgh_path = os.path.join(args.out_path, run_name+'/checkpoints/')
    out_log_path = os.path.join(args.out_path, run_name+'/logs/')

    os.makedirs(out_wgh_path)
    os.makedirs(out_log_path)

    train_data_path = os.path.join(args.dataset_path, 'train/image_resized')
    train_label_path = os.path.join(args.dataset_path, 'train/label_resized')
    val_data_path = os.path.join(args.dataset_path, 'val/image_resized')
    val_label_path = os.path.join(args.dataset_path, 'val/label_resized')

    # Configure the logger
    logging.basicConfig(filename=os.path.join(out_log_path, 'logging.log'), level=logging.INFO)

    train_dataset = FVDataset(
        train_data_path, train_label_path, 
        augmentation=True,
        size_coefficient = args.dataset_coeff
    )

    valid_dataset = FVDataset(
        val_data_path, val_label_path,
        augmentation=False,
        size_coefficient = args.dataset_coeff
    )

    # persisten workers - workers are kept alive between epochs
    # pin memory this can consume GPU space and accelerate transfers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=10, persistent_workers=True)
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

    num_epochs = 30

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001) # factor of 10 lower when I am funetuning
    lr_schedulre = None #torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=len(train_loader), epochs=num_epochs)
    loss = DistanceWeightBCELoss(alpha=0.8) #CombLoss(alpha=0.7) 
    
    best_val_score = float('inf')
    train_logs, valid_logs = [], []

    desc = f'FV: {run_name} \n \
         Model: {args.model} \n \
         Checkpoint: {args.checkpoint_path} \n \
         Dataset: {args.dataset_coeff:.2f} \n \
         Batch size: {args.batch_size_train} \n \
         Loss: {loss} \n \n'

         # log also the learning rate and if scheduler is used
    
    logging.info(desc)

    for epoch in range(1, num_epochs+1):
        logging.info(f'Epoch {epoch}/{num_epochs}.')
        train_log = run_epoch(model, device, loss, train_loader, optimizer, metrics, 'train', lr_schedulre)
        valid_log = run_epoch(model, device, loss, valid_loader, optimizer, metrics, 'valid', lr_schedulre)

        train_logs.append(train_log)
        valid_logs.append(valid_log)

        if best_val_score > np.mean(valid_log[checkpoint_metric]):
            best_val_score = np.mean(valid_log[checkpoint_metric])
            if not os.path.exists(out_wgh_path):
                os.makedirs(out_wgh_path)
            
            out_name =  f'{args.model}_err:{best_val_score:.3f}_ep:{epoch}.pth'
            torch.save(model, os.path.join(out_wgh_path,out_name))
            logging.info('Model saved!')
        
        np.save(os.path.join(out_log_path, f'train_logs.npy'), train_logs)
        np.save(os.path.join(out_log_path, f'val_logs.npy'), valid_logs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("FVApp training script")
    parser.add_argument("--model", type=str,  choices={'Deeplabv3+'}, default='Deeplabv3')
    parser.add_argument("--loss", type=str,  choices={'BinaryDice'}, default='BinaryDice')
    parser.add_argument("--device", type=str,  choices={'cuda:0', 'cuda:1', 'cpu'}, default='cuda:0')
    parser.add_argument("--checkpoint_path", type=str, default='/home/kafkaon1/FVAPP/out/run_230503-140231/checkpoints/Deeplabv3_err:0.176_ep:16.pth')
    parser.add_argument("--out_path", type=str, default='/home/kafkaon1/FVAPP/out/')
    parser.add_argument("--dataset_path", type=str, default='/home/kafkaon1/FVAPP/data/FV')
    parser.add_argument("--dataset_coeff", type=float, default=1/100)
    parser.add_argument("--batch_size_train", type=int, default=20)
    parser.add_argument("--batch_size_val", type=int, default=8)

    args = parser.parse_args()

    main(args)

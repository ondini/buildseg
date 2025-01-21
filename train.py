import os
import datetime
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import createDeepLabv3, createDeepLabv3Plus
from dataset import FVDataset
from losses import CombLoss, BinaryDiceLoss, DistanceLoss, DistanceWeightBCELoss, BoundaryLoss, FocalLoss
from metrics import GetMetricFunction, Metrics

def run_epoch(model, device, loss_fn, dataloader, optimizer, metrics, mode, lr_scheduler):

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

            metrics.add(outputs, target)

            if mode == 'train':
                loss.backward()
                # gradient clipping - to avoid exploding gradients 
                # test the clipping threshold in log scale
                # compute average gradient norm and log it
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()
    
def train(args):
    
    start = datetime.datetime.now()
    run_name = f"run_{start:%y%m%d-%H%M%S}"
    out_wgh_path = os.path.join(args.out_path, run_name+'/checkpoints/')
    out_log_path = os.path.join(args.out_path, run_name+'/logs/')

    os.makedirs(out_wgh_path, exist_ok=True)
    os.makedirs(out_log_path, exist_ok=True)

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

    # pin memory this can consume GPU space and accelerate transfers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=10, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=4, persistent_workers=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint_metric = 'loss'

    if args.checkpoint_path:
        model = torch.load(args.checkpoint_path)
    else:
        model = createDeepLabv3Plus(1)
    model.to(device)

    if args.freeze != 'none':
        for name, param in model.named_parameters():
            if name.startswith('encoder'):
                param.requires_grad = False
            if name.startswith('decoder.aspp') and args.freeze.startswith('encDec'):
                param.requires_grad = False
            if name.startswith('decoder.block') and args.freeze == 'encDec':
                param.requires_grad = False

    num_epochs = args.num_epochs

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001) # factor of 10 lower when I am funetuning
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=len(train_loader), epochs=num_epochs) if args.scheduler else None
    loss = nn.BCELoss() # DistanceWeightBCELoss(alpha=0.1, theta=3) #CombLoss(alpha=0.89) #DistanceWeightBCELoss(alpha=0.2, theta=5) #nn.BCELoss(reduction='mean') #CombLoss(alpha=0.84) #DistanceWeightBCELoss(alpha=0.5) #alpha=0.8) #CombLoss(alpha=0.7)  

    metric_names = ["iou", "dice", "precision", "recall", "matthews"]
    metrics = {
        metric_name : GetMetricFunction(metric_name) for metric_name in metric_names
    } 
    metrics.update( {
        f"{metric_name}_b{1}" : GetMetricFunction(metric_name, 1) for metric_name in metric_names
    })
    metrics.update({
        f"{metric_name}_b{3}" : GetMetricFunction(metric_name, 3) for metric_name in metric_names
    } )
    metrics.update( { 
        "loss" : lambda outputs, target : loss(outputs.float(), target.float())
    })

    train_metrics = Metrics(metrics)
    valid_metrics = Metrics(metrics)
    
    desc = f'FV: {run_name} \n \
         Model: {args.model} \n \
         Checkpoint: {args.checkpoint_path} \n \
         Dataset: {args.dataset_coeff:.2f} \n \
         Batch size: {args.batch_size_train} \n \
         Loss: {loss} \n \
         Epochs: {num_epochs} \n \
         Optimizer: {optimizer} \n \
         Scheduler: {lr_scheduler} \n \
         Frozen: {args.freeze} \n \n'
         
         # log also the learning rate and if scheduler is used
    
    logging.info(desc)

    best_val_score = float('inf')

    for epoch in range(1, num_epochs+1):
        logging.info(f'Epoch {epoch}/{num_epochs}.')
        run_epoch(model, device, loss, train_loader, optimizer, train_metrics, 'train', lr_scheduler)
        run_epoch(model, device, loss, valid_loader, optimizer, valid_metrics, 'valid', lr_scheduler)

        train_metrics.epoch_end()
        valid_metrics.epoch_end()

        if best_val_score > valid_metrics[checkpoint_metric]:
            best_val_score = valid_metrics[checkpoint_metric]            
            out_name =  f'{args.model}_err:{best_val_score:.5f}_ep:{epoch}.pth'
            torch.save(model, os.path.join(out_wgh_path,out_name))
            logging.info('Model saved!')
        
        train_metrics.save(os.path.join(out_log_path, f'train_logs.pkl'))
        valid_metrics.save(os.path.join(out_log_path, f'val_logs.pkl'))
        train_metrics.visualize_metrics(os.path.join(out_log_path, f'train_metrics.png'), n=1)
        valid_metrics.visualize_metrics(os.path.join(out_log_path, f'val_metrics.png'), n=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("FVApp training script")

    parser.add_argument("--model", type=str,  choices={'Deeplabv3+'}, default='Deeplabv3')
    parser.add_argument("--loss", type=str,  choices={'BinaryDice'}, default='BinaryDice')
    parser.add_argument("--num_epochs", type=int,  default=30)
    parser.add_argument("--device", type=str,  choices={'cuda:0', 'cuda:1', 'cpu'}, default='cuda:0')
    parser.add_argument("--checkpoint_path", type=str, default='') #/home/kafkaon1/Dev/FVAPP/out/train/run_230503-140231/checkpoints/Deeplabv3_err:0.194_ep:15.pth') #/home/kafkaon1/FVAPP/out/run_230503-140231/checkpoints/Deeplabv3_err:0.176_ep:16.pth')
    parser.add_argument("--out_path", type=str, default='/home/kafkaon1/Dev/FVAPP/out/train')
    parser.add_argument("--dataset_path", type=str, default='/home/kafkaon1/Dev/data/FV')
    parser.add_argument("--dataset_coeff", type=float, default=1/18)
    parser.add_argument("--batch_size_train", type=int, default=20)
    parser.add_argument("--batch_size_val", type=int, default=8)
    parser.add_argument("--scheduler", type=bool, default=True)
    parser.add_argument("--freeze", type=str, choices={'none', 'enc', 'encDec','encDecAspp'}, default='none')

    args = parser.parse_args()

    train(args)

import os
import random
import datetime
import argparse
import logging
import logging

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models import createDeepLabv3, createDeepLabv3Plus
from dataset import FVDataset
from losses import FVLoss
from metrics import intersection_over_union, intersection_over_union_boundary, dice_coefficient, dice_coefficient_boundary, precision, recall

def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def random_rot90(image):
    k = random.randint(0, 3)
    return torch.rot90(image, k=k, dims=(1, 2))

def run_epoch(model, device, loss_fn, dataloader, optimizer, metrics, mode):
    logs = {metric_name: [] for metric_name in metrics.keys()}
    logs['loss'] = []

    logging.info(f"> Starting {mode} epoch.")
    if mode == 'train':
        model.train() 
    else:
        model.eval()  

    for  i, (inputs , target, edges) in enumerate(dataloader):
        logging.info(f"-> Epoch iteration {i}.")
        inputs = inputs.to(device)
        target = target.to(device)
        edges = edges.to(device)
        model.zero_grad()

        with torch.set_grad_enabled(mode == 'train'): # gradient calculation only in train mode
            outputs = model(inputs)
            loss = loss_fn(outputs, target, edges)

            for metric_name, metric_fn in metrics.items():
                metric_value = metric_fn(outputs, target, edges).cpu().item()
                logs[metric_name].append(metric_value)

            logs['loss'].append(loss.cpu().item())

            if mode == 'train':
                loss.backward()
                optimizer.step()
    logs = {metric_name: np.mean(metric_values) for metric_name, metric_values in logs.items()}
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
    
    training_augmentation = T.Compose([
            T.ToTensor(),
            T.Lambda(random_rot90)
        ]
    )

    train_dataset = FVDataset(
        train_data_path, train_label_path, 
        augmentation=training_augmentation,
        size_coefficient = 1/20
    )

    valid_dataset = FVDataset(
        val_data_path, val_label_path,
        augmentation = T.ToTensor(),
        size_coefficient = 1/20
    )

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    metrics = {
        "dice": dice_coefficient,
        "dice_b": dice_coefficient_boundary,
        "precision": precision,
        "recall": recall,
    }


    if args.checkpoint_path:
        model = torch.load(args.checkpoint_path)
    else:
        model = createDeepLabv3Plus(2)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    loss = FVLoss()
    

    best_val_score = float('inf')
    train_logs, valid_logs = [], []

    num_epochs = 30

    for epoch in range(1, num_epochs+1):
        logging.info(f'Epoch {epoch}/{num_epochs}.')
        train_log = run_epoch(model, device, loss, train_loader, optimizer, metrics, 'train')
        valid_log = run_epoch(model, device, loss, valid_loader, optimizer, metrics, 'valid')

        train_logs.append(train_log)
        valid_logs.append(valid_log)

        if best_val_score > valid_log['loss']:
            best_val_score = valid_log['loss']
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
    parser.add_argument("--checkpoint_path", type=str, default='/home/kafkaon1/FVAPP/out/checkpoints/dl3_err:0.566_ep:2_04-06_12:24.pth')
    parser.add_argument("--out_path", type=str, default='/home/kafkaon1/FVAPP/out/')
    parser.add_argument("--dataset_path", type=str, default='/home/kafkaon1/FVAPP/data/FV')
    parser.add_argument("--dataset_coeff", type=float, default=1/200)
    args = parser.parse_args()

    main(args)

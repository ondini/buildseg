import os
import datetime
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import createDeepLabv3, createDeepLabv3Plus, createMaskRCNN
from datasetI import FVDatasetIS
from losses import (
    CombLoss,
    BinaryDiceLoss,
    DistanceLoss,
    DistanceWeightBCELoss,
    BoundaryLoss,
    FocalLoss,
)
from metrics import GetMetricFunction, Metrics

import utils
import math
import sys
from engine import evaluate


def train_one_epoch(
    model, device, dataloader, optimizer, epoch, print_freq, scaler=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(dataloader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(dataloader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def run_epoch(model, device, dataloader, optimizer, metrics, mode, lr_scheduler):
    logging.info(f"> Starting {mode} epoch.")
    if mode == "train":
        model.train()
    else:
        model.eval()

    for i, (inputs, target) in enumerate(dataloader):
        if inputs.shape[0] == 1:
            continue
        logging.info(f"-> Epoch iteration {i}.")
        inputs = inputs.to(device).float()
        target = target.to(device).float()
        model.zero_grad()

        with torch.set_grad_enabled(
            mode == "train"
        ):  # gradient calculation only in train mode
            outputs = model(inputs)
            loss = loss_fn(outputs.float(), target.float())

            metrics.add(outputs, target)

            if mode == "train":
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
    out_wgh_path = os.path.join(args.out_path, run_name + "/checkpoints/")
    out_log_path = os.path.join(args.out_path, run_name + "/logs/")

    os.makedirs(out_wgh_path, exist_ok=True)
    os.makedirs(out_log_path, exist_ok=True)

    # train_data_path = os.path.join(args.dataset_path, 'train/image_resized')
    # train_label_path = os.path.join(args.dataset_path, 'train/label_resized')
    # val_data_path = os.path.join(args.dataset_path, 'val/image_resized')
    # val_label_path = os.path.join(args.dataset_path, 'val/label_resized')

    train_data_path = os.path.join(args.dataset_path, "fac_imgs")
    train_label_path = os.path.join(args.dataset_path, "fac_labels")
    val_data_path = os.path.join(args.dataset_path, "fac_imgs")
    val_label_path = os.path.join(args.dataset_path, "fac_labels")

    # Configure the logger
    logging.basicConfig(
        filename=os.path.join(out_log_path, "logging.log"), level=logging.INFO
    )

    train_dataset = FVDatasetIS(
        train_data_path,
        train_label_path,
        names_path=os.path.join(args.dataset_path, "train_fac.txt"),
        augmentation=True,
        size_coefficient=args.dataset_coeff,
    )

    valid_dataset = FVDatasetIS(
        val_data_path,
        val_label_path,
        names_path=os.path.join(args.dataset_path, "test_fac.txt"),
        augmentation=False,
        size_coefficient=args.dataset_coeff,
    )

    # pin memory this can consume GPU space and accelerate transfers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=10,
        persistent_workers=True,
        collate_fn=utils.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        collate_fn=utils.collate_fn,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint_metric = "loss"

    if args.checkpoint_path:
        model = torch.load(args.checkpoint_path)
    else:
        model = createMaskRCNN(args.num_classes)

    model.to(device)

    num_epochs = args.num_epochs
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001) # factor of 10 lower when I am funetuning

    # and a learning rate scheduler which decreases the learning rate by10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=len(train_loader), epochs=num_epochs) if args.scheduler else None

    metric_names = ["iou", "dice", "precision", "recall", "matthews"]
    metrics = {
        metric_name: GetMetricFunction(metric_name) for metric_name in metric_names
    }
    metrics.update(
        {
            f"{metric_name}_b{1}": GetMetricFunction(metric_name, 1)
            for metric_name in metric_names
        }
    )
    metrics.update(
        {
            f"{metric_name}_b{3}": GetMetricFunction(metric_name, 3)
            for metric_name in metric_names
        }
    )

    train_metrics = Metrics(metrics)
    valid_metrics = Metrics(metrics)

    desc = f"FV: {run_name} \n \
         Model: {args.model} \n \
         Checkpoint: {args.checkpoint_path} \n \
         Dataset: {args.dataset_coeff:.2f} \n \
         Batch size: {args.batch_size_train} \n \
         Epochs: {num_epochs} \n \
         Optimizer: {optimizer} \n \
         Scheduler: {lr_scheduler} \n \
         Frozen: {args.freeze} \n \n"

    # log also the learning rate and if scheduler is used

    logging.info(desc)

    best_val_score = float("inf")

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Epoch {epoch}/{num_epochs}.")
        train_one_epoch(model, device, train_loader, optimizer, epoch, 10)
        # run_epoch(model, device, loss, valid_loader, optimizer, valid_metrics, 'valid', lr_scheduler)
        # evaluate(model, test_loader, device=device)

        # train_metrics.epoch_end()
        # valid_metrics.epoch_end()

        lr_scheduler.step()

        # if best_val_score > valid_metrics[checkpoint_metric]:
        #     best_val_score = valid_metrics[checkpoint_metric]
        #     out_name =  f'{args.model}_err:{best_val_score:.5f}_ep:{epoch}.pth'
        #     torch.save(model, os.path.join(out_wgh_path,out_name))
        #     logging.info('Model saved!')

        # train_metrics.save(os.path.join(out_log_path, f'train_logs.pkl'))
        # valid_metrics.save(os.path.join(out_log_path, f'val_logs.pkl'))
        # train_metrics.visualize_metrics(os.path.join(out_log_path, f'train_metrics.png'), n=1)
        # valid_metrics.visualize_metrics(os.path.join(out_log_path, f'val_metrics.png'), n=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FVApp training script")

    parser.add_argument(
        "--model", type=str, choices={"Deeplabv3+"}, default="Deeplabv3"
    )
    parser.add_argument(
        "--loss", type=str, choices={"BinaryDice"}, default="BinaryDice"
    )
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument(
        "--device", type=str, choices={"cuda:0", "cuda:1", "cpu"}, default="cuda:0"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=""
    )  # /home/kafkaon1/Dev/FVAPP/out/train/run_230525-125829/checkpoints/Deeplabv3_err:0.01101_ep:18.pth') #/home/kafkaon1/FVAPP/out/run_230503-140231/checkpoints/Deeplabv3_err:0.176_ep:16.pth')
    parser.add_argument(
        "--out_path", type=str, default="/home/kafkaon1/Dev/FVAPP/out/train"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="/home/kafkaon1/Dev/FVAPP/data/CZ7"
    )
    parser.add_argument("--dataset_coeff", type=float, default=0.1)
    parser.add_argument("--batch_size_train", type=int, default=6)
    parser.add_argument("--batch_size_val", type=int, default=8)
    parser.add_argument("--scheduler", type=bool, default=True)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument(
        "--freeze",
        type=str,
        choices={"none", "enc", "encDec", "encDecAspp"},
        default="none",
    )

    args = parser.parse_args()

    train(args)

import os, cv2
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models import createDeepLabv3
from dataset import FVDataset
from loss import BinaryDiceLoss

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

    if mode == 'train':
        model.train() 
    else:
        model.eval()  

    for  i, (inputs , target) in enumerate(dataloader):
        inputs = inputs.to(device)
        target = target.to(device)
        model.zero_grad()

        with torch.set_grad_enabled(mode == 'train'): # gradient calculation only in train mode
            outputs = model(inputs)['out']
            loss = loss_fn(outputs, target)

            for metric_name, metric_fn in metrics.items():
                metric_value = metric_fn(outputs, loss).cpu().detach().numpy()
                logs[metric_name].append(metric_value)

            logs['loss'].append(loss.cpu().item())

            if mode == 'train':
                loss.backward()
                optimizer.step()
    logs = {metric_name: np.mean(metric_values) for metric_name, metric_values in logs.items()}
    return logs


def main():
    out_wgh_path = './'
    train_data_path = '/home/ondin/Developer/FVAPP/data_big/archive/train/image_resized'
    train_label_path = '/home/ondin/Developer/FVAPP/data_big/archive/train/label_resized'
    val_data_path = '/home/ondin/Developer/FVAPP/data_big/archive/val/image_resized'
    val_label_path = '/home/ondin/Developer/FVAPP/data_big/archive/val/label_resized'

    #class_names = ['background', 'building']

    training_augmentation = T.Compose([
            T.ToTensor(),
            T.Lambda(random_rot90)
        ]
    )

    train_dataset = FVDataset(
        train_data_path, train_label_path, 
        augmentation=training_augmentation
    )

    valid_dataset = FVDataset(
        val_data_path, val_label_path,
        augmentation = T.ToTensor()
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    metrics = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = createDeepLabv3(2)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    loss = BinaryDiceLoss()

    best_val_score = 0.0
    train_logs, valid_logs = [], []

    num_epochs = 10

    for epoch in range(0, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}.')
        train_log = run_epoch(model, device, loss, train_loader, optimizer, metrics, 'train')
        valid_log = run_epoch(model, device, loss, valid_loader, optimizer, metrics, 'valid')

        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        if best_val_score < valid_logs['loss']:
            best_val_score = valid_logs['loss']
            if not os.path.exists(out_wgh_path):
                os.makedirs(out_wgh_path)
            
            now = datetime.datetime.now()
            out_name =  f'dl3_err:{best_val_score:.3f}_ep:{epoch}_{now:%m-%d_%H:%M}.pth'
            torch.save(model, os.path.join(out_wgh_path,out_name))
            print('Model saved!')
        
        np.save(os.path.join(out_wgh_path, 'train_logs.npy'), train_logs_list)
        np.save(os.path.join(out_wgh_path,'valid_logs.npy'), valid_logs_list)


# LOSSES:
# WEIGHTED BCE by class 
# WEIGHTED BCE by edges (Ronneberger et al)

# TODOS and ideas - need to choose right loss function, ideally weighed by edges
# Then we need to choose right metrics, probably normal mIoU, edge IoU and some other edge accuracy metric
# also there is a possibility of rotation augmentation added to the datasets

if __name__ == '__main__':
    main()
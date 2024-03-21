import numpy as np
import torch
from torchvision.utils import make_grid
from .base_trainer import BaseTrainer
import copy
from typing import List, Optional, Tuple

## TODOs - add learning  rate logging; add scaler

class SegTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, loss, metrics, optimizer, config, device,
                 data_loader, valid_data_loader, lr_scheduler=None):
        super().__init__(model, optimizer, config)

        self.device = device
        self.train_data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.loss = loss

        self.log_step = config['trainer'].get('logging_step', 1)
        self.metrics = metrics.with_writer(self.writer)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Args:
            epoch (int): Current training epoch.

        Returns:
            log (dict): A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        for i, (inputs , targets) in enumerate(self.train_data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad() # or model.zero_grad() ??? 
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)

            loss.backward()
            self.optimizer.step()

            if i % self.log_step == 0:
                progress = f"[{i+self.train_data_loader.batch_size}/{len(self.train_data_loader)}]"
                self.logger.info(f' -> {progress}: trainig epoch progress | loss: {loss.item():.6f}')
                self.writer.add_image('input', make_grid(inputs.cpu(), nrow=8, normalize=True)) ### <-
            
            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + i)
            
            self.metrics['loss'] = loss.item()
            self.metrics.add(outputs, targets)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return self.metrics.epoch_end()
    

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        Args:
            epoch (int): Current training epoch.

        Returns:
            log (dict): A log that contains average loss and metrics in this epoch.
        """
        self.model.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.valid_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)

                if i % self.log_step == 0:
                    progress = f"[{i+self.valid_data_loader.batch_size}/{len(self.valid_data_loader)}]"
                    self.logger.info(f' -> {progress}: validation epoch progress | loss: {loss.item():.6f}')

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + i, 'valid')
                self.writer.add_image('input/valid', make_grid(inputs.cpu(), nrow=8, normalize=True))

                self.metrics['loss'] = loss.item()
                self.metrics.add(outputs, targets, 'valid')

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return self.metrics.epoch_end()


class KPTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, loss, metrics, optimizer, config, device,
                 data_loader, valid_data_loader, lr_scheduler=None):
        super().__init__(model, optimizer, config)

        self.device = device
        self.train_data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.loss = loss

        self.log_step = config['trainer'].get('logging_step', 1)
        self.metrics = metrics.with_writer(self.writer)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Args:
            epoch (int): Current training epoch.

        Returns:
            log (dict): A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        for i, (inputs , targets) in enumerate(self.train_data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad() # or model.zero_grad() ??? 
            outputs = self.model(inputs)
            #  nn.functional.binary_cross_entropy_with_logits(
            #         predicted_unnormalized_maps[:, channel_idx, :, :], gt_heatmaps[channel_idx]
            #     )
            loss = self.loss(outputs, targets)

            loss.backward()
            self.optimizer.step()

            if i % self.log_step == 0:
                progress = f"[{i+self.train_data_loader.batch_size}/{len(self.train_data_loader)}]"
                self.logger.info(f' -> {progress}: trainig epoch progress | loss: {loss.item():.6f}')
                self.writer.add_image('input', make_grid(inputs.cpu(), nrow=8, normalize=True)) ### <-
            
            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + i)
            
            self.metrics['loss'] = loss.item()
            self.metrics.add(outputs, targets)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return self.metrics.epoch_end()
    

class ObjDetTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, loss, metrics, optimizer, config, device,
                 data_loader, valid_data_loader, lr_scheduler=None):
        super().__init__(model, optimizer, config)

        self.device = device
        self.train_data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.loss = loss

        self.log_step = config['trainer'].get('logging_step', 1)
        self.metrics = metrics.with_writer(self.writer)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Args:
            epoch (int): Current training epoch.

        Returns:
            log (dict): A log that contains average loss and metric in this epoch.
        """
        self.model.train()
    
        # TODO - add some smart approach to scheduler
        lr_scheduler = None 
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(self.train_data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
        

        for i, (inputs , targets) in enumerate(self.train_data_loader):
            images = list(image.to(self.device) for image in inputs)
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]
            
            self.optimizer.zero_grad()
            
            loss_dict = self.model.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            self.optimizer.step()
            
            if i % self.log_step == 0:
                progress = f"[{i+self.train_data_loader.batch_size}/{len(self.train_data_loader)}]"
                self.logger.info(f' -> {progress}: trainig epoch progress | loss: {losses.item():.6f}')
                self.writer.add_image('input', make_grid(torch.stack(inputs).cpu(), nrow=8, normalize=True)) ### <-
            
            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + i)
            
            self.metrics['loss'] = losses.item()
            # self.metrics.add(outputs, targets)
                
            if lr_scheduler is not None:
                lr_scheduler.step()

        return self.metrics.epoch_end()
    

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        Args:
            epoch (int): Current training epoch.

        Returns:
            log (dict): A log that contains average loss and metrics in this epoch.
        """
        self.model.eval()

        with torch.no_grad():
            
            for i, (inputs, targets) in enumerate(self.valid_data_loader):
                images = list(image.to(self.device) for image in inputs)
                targets = [
                    {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    }
                    for t in targets
                ]
                
                out = self.model.model(images, targets)
                
                # outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

                precisions, recalls, det_matched = calculate_mAP(out, targets, self.model.num_classes)

                det_labels, det_masks, det_targets = det_matched
                loss = self.loss(det_masks.float(), det_targets.float())
                
                if i % self.log_step == 0:
                    progress = f"[{i+self.valid_data_loader.batch_size}/{len(self.valid_data_loader)}]"
                    self.logger.info(f' -> {progress}: validation epoch progress | loss: {loss.item():.6f}') #mAP: {precisions.mean()*100:.1f}%')

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + i, 'valid')
                self.writer.add_image('input/valid', make_grid(torch.stack(inputs).cpu(), nrow=8, normalize=True))

                self.metrics['loss'] = loss.item()
                self.metrics.add(det_masks, det_targets, 'valid')
                
        return self.metrics.epoch_end()


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def calculate_mAP(detections, targets, n_classes, iou_threshold=0.5, score_threshold=0.8):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    Args:
        detections: list of tensors, one tensor for each image containing detected objects,
            each detection is of the form [box coords, class, score]
        targets: list of tensors, one tensor for each image containing ground-truth objects,
            each ground-truth object is of the form [box coords, class]
        n_classes: number of classes
        iou_threshold: intersection over union threshold used for determining whether a detection is correct
        score_threshold: threshold for a box to be considered a positive detection
        device: device on which tensors are located
    
    Returns:
        list of average precisions for all classes, mean average precision (mAP)
        list of average recalls for all classes, mean average recall (mAR)
     """
    
    class_tps = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    class_fps = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    n_dets = 0
    
    corr_labels_s = []
    corr_targets_s = []
    corr_detections_s = []
    
    for id in range(len(detections)): # go over all imgs in batch
        corr_labels = []
        corr_targets = []
        corr_detections = []
        for c in range(1, n_classes):
            # Extract only objects with this class
            target_ids = targets[id]['labels'] == c
            true_class_boxes = targets[id]['boxes'][target_ids,:]

            # Keep track of which true objects with this class have already been 'detected'
            true_class_boxes_detected = torch.zeros(true_class_boxes.shape[0], dtype=torch.uint8)

            # Extract only detections over some score
            det_filtered = detections[id]['scores'] > score_threshold

            # Extract only detections with this class
            det_labels = detections[id]['labels']
            det_ids = (det_labels == c) * det_filtered
            det_class_images = detections[id]['masks'][det_ids]  # (n_class_detections)
            det_class_boxes = detections[id]['boxes'][det_ids]  # (n_class_detections, 4)

            n_class_detections = det_class_boxes.size(0)
            if n_class_detections == 0:
                continue

            # In the order of decreasing scores, check if true or false positive
            true_positives = 0
            false_positives = 0
            
            for d in range(n_class_detections):
                this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)

                # If no such object in this image, then the detection is a false positive
                if true_class_boxes.size(0) == 0:
                    false_positives += 1
                    continue

                # Find maximum overlap of this detection with objects in this image of this class
                overlaps = find_jaccard_overlap(this_detection_box, true_class_boxes)  # (1, n_class_objects_in_img)
                max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

                # If the maximum overlap is greater than the threshold of 0.5, it's a match
                if max_overlap.item() > iou_threshold:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[ind] == 0:
                        true_positives += 1
                        true_class_boxes_detected[ind] = 1  # this object has now been detected/accounted for
                        corr_targets.append(targets[id]['masks'][target_ids][ind])
                        corr_labels.append(targets[id]['labels'][target_ids][ind])
                        corr_detections.append(det_class_images[d][0])
                        
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives += 1
                # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
                else:
                    false_positives += 1

            # Compute cumulative precision and recall at each detection in the order of decreasing scores
            class_tps[c - 1] += true_positives
            class_fps[c - 1] += false_positives
            n_dets += true_class_boxes.size(0)
        if len(corr_labels) > 0:
            corr_labels_s.append(torch.stack(corr_labels))
            corr_targets_s.append(torch.stack(corr_targets))
            corr_detections_s.append(torch.stack(corr_detections))
            

    precision = class_tps / (
            class_tps + class_fps + 1e-10)  # (n_class_detections)
    recall = class_tps / n_dets  # (n_class_detections)
    if len(corr_labels_s) > 0:
        correspondences = (torch.concat(corr_labels_s), 
                            torch.concat(corr_detections_s), 
                            torch.concat(corr_targets_s))
    else:
        correspondences = (torch.tensor([0]), torch.tensor(np.zeros((1, 256, 256))), torch.tensor(np.zeros((1, 256, 256))))

    return precision.mean(), recall.mean(), correspondences
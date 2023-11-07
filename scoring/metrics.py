import torch
import torch.nn.functional as F

import numpy as np

def moving_average(a, n=8):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# TODO - solve loss adding. Check tensorboard writing and what addimage does

class Metric():
    '''
    Metric class to store the metric for the experiment
    
    Args:
        function (callable): Metric function
        name (str): Name of the metric function
    '''
    def __init__(self, function, name):
        self._history = []
        self._function = function
        self._name = name
    
    def __str__(self) -> str:
        return f"Metric_object(function={self._name})"

    def add(self, prediction, target) -> None:
        """Add the prediction and target to the metric"""
        value = self._function(prediction, target)
        self._history.append(value)
        return value

    def compute(self, prediction, target) -> float:
        """Compute the metric"""
        return self._function(prediction, target)
    
    def epoch_end(self) -> None:
        """End of epoch"""
        retval = np.mean(self._history)
        self._history = []
        return retval

    @property
    def value(self) -> float:
        return np.mean(self._history)
    
    @value.setter
    def value(self, value) -> None:
        self._history.append(value)


class Metrics():
    '''
    Metrics class to store the metrics for the experiment

    Args:
        metrics (dict): Dictionary of metric functions
        phase (str): Phase of the experiment (train/val)
        tensorboard_writer (SummaryWriter): Tensorboard writer object
    '''
    def __init__(self, metric_fns, phase='train', tensorboard_writer=None):
        self.phase = phase
        self.writer = tensorboard_writer
        self._metrics = {}
        for metric_name, metric_fn in metric_fns.items():
            self._metrics[metric_name] = Metric(metric_fn, metric_name)
        self._metrics['loss'] = Metric(lambda x, y: 0, 'loss')
    
    def __str__(self) -> str:
        return f"Metrics_object(functions={[name for name in self._metrics.keys()]})"
    
    def with_writer(self, tensorboard_writer) -> 'Metrics':
        """Set the tensorboard writer"""
        self.writer = tensorboard_writer
        return self

    def add(self, prediction, target, phase='') -> None:
        """Add the prediction and target to all the metrics"""
        for metric in self._metrics.values():
            value = metric.add(prediction, target)
            if self.writer is not None:
                self.writer.add_scalar(f"{metric._name}/{self.phase if phase=='' else phase }", value) # possibly add moving average?
    
    def __getitem__(self, metric_name) -> float:
        return self._metrics[metric_name].value
    
    def __setitem__(self, metric_name, value) -> None:
        self._metrics[metric_name].value = value

    def epoch_end(self, phase='') -> dict:
        """End of epoch"""
        log = {}
        for metric in self._metrics.values():
            res = metric.epoch_end()
            log[metric._name] = res
            if self.writer is not None:
                self.writer.add_scalar(f"{metric._name}/{self.phase if phase=='' else phase}_epoch", res)
        
        if self.writer is not None:
            self.writer.add_scalars(f"{self.phase}_epoch", log)

        return log


def GetMetricFunction(name:str='iou', threshold:float=0.5, reduction:str='mean') -> callable:
    '''
    Generate the metric function based on the name

    Args:
        name (str): Name of the metric function
        border (int): Border width to obtain from the label and prediction for the metric (0=orignal prediction)
        threshold (float): Threshold to binarize the prediction
        reduction (str): Reduction method for the metric over the batch

    Returns:
        metric (function): Metric function
    
    '''
    [name, border] = name.split('_') if len(name.split('_')) > 1 else [name, '']
    border = int(border) if border != '' else 0

    if name == 'iou':
        metric_fn = _iou
    elif name == 'dice':
        metric_fn = _dice_coefficients
    elif name == 'matthews':
        metric_fn = _matthews_correlation_coefficient
    elif name == 'precision':
        metric_fn = _precision
    elif name == 'recall':
        metric_fn = _recall
    else:
        raise ValueError(f"Metric {name} not found")

    def metric(predict, targets):
        assert predict.shape == targets.shape, "Predict and targets must have the same shape"
        # Binarize the predict, targets based on the threshold
        predict = (predict > threshold).float()
        targets = (targets > threshold).float()

        # Get only the border part of the label a prediction
        if border > 0:
            targets_b, predict_b = F.max_pool2d(
                1 - targets, kernel_size=3, stride=1, padding=(3 - 1) // 2), F.max_pool2d(
                1 - predict, kernel_size=3, stride=1, padding=(3 - 1) // 2)
            
            targets_b -= 1 - targets
            predict_b -= 1 - predict

            targets, predict  = F.max_pool2d(
                targets_b, kernel_size=border, stride=1, padding=(border - 1) // 2), F.max_pool2d(
                predict_b, kernel_size=border, stride=1, padding=(border - 1) // 2)
        
        predict = predict.contiguous().view(predict.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)

        # Calculate the metric value
        value = metric_fn(predict, targets)
        value = value.cpu() if value.is_cuda else value
            
        if reduction == 'mean':
            return value.mean()
        elif reduction == 'sum':
            return value.sum()
        else:
            return value
    
    return metric


# Metric computation functions

def _iou(prediction, targets):
    # Calculate the intersection and union
    intersection = (prediction * targets).sum(dim=1)
    union = (prediction + targets).sum(dim=1) - intersection
    
    # Calculate the IoU for each image in the batch
    iou = intersection / (union + 1e-8)
    return iou

def _dice_coefficients(prediction, targets):
    # Calculate the intersection and union
    intersection = (prediction * targets).sum(dim=1)
    total = (prediction + targets).sum(dim=1)
    
    # Calculate the Dice coefficient for each image in the batch
    dice = (2 * intersection) / (total + 1e-8)
    return dice

def _precision(prediction, targets):
    # Calculate the true positives and false positives
    true_positives = (prediction * targets).sum(dim=1)
    false_positives = (prediction * (1 - targets)).sum(dim=1)
    
    # Calculate the precision for each image in the batch
    precision = true_positives / (true_positives + false_positives + 1e-8)
    return precision

def _recall(prediction, targets):
    # Calculate the true positives and false negatives
    true_positives = (prediction * targets).sum(dim=1)
    false_negatives = ((1 - prediction) * targets).sum(dim=1)
    
    # Calculate the recall for each image in the batch
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    return recall

def _matthews_correlation_coefficient(prediction, targets):
    # Calculate the true positives, true negatives, false positives and false negatives
    true_positives = (prediction * targets).sum(dim=1)
    true_negatives = ((1 - prediction) * (1 - targets)).sum(dim=1)
    false_positives = (prediction * (1 - targets)).sum(dim=1)
    false_negatives = ((1 - prediction) * targets).sum(dim=1)
    
    # Calculate the Matthews correlation coefficient for each image in the batch
    mcc = (true_positives * true_negatives - false_positives * false_negatives) / (
        torch.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives) + 1e-8))
    return mcc


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


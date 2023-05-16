import torch
import torch.nn.functional as F

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def moving_average(a, n=8):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Metric():
    '''
    Metric class to store the metric for the experiment
    
    Args:
        function (callable): Metric function
        name (str): Name of the metric function
    '''
    def __init__(self, function, name):
        self._history = []
        self._epoch_history = []
        self._current_epoch_history = []

        self.function = function
        self.name = name
    
    def __str__(self) -> str:
        return f"Metric_object(function={self.name})"

    def add(self, prediction, target) -> None:
        """Add the prediction and target to the metric"""
        self._current_epoch_history.append(self.function(prediction, target).cpu().item())

    def compute(self, prediction, target) -> float:
        """Compute the metric"""
        return self.function(prediction, target)
    
    def append(self, value) -> None:
        """Append the value to the metric"""
        self._current_epoch_history.append(value)
    
    @property
    def value(self) -> float:
        return self._epoch_history[-1]

    @property
    def history(self) -> list:
        return self._history

    def epoch_end(self) -> None:
        """End of epoch"""
        self._epoch_history.append(np.mean(self._current_epoch_history))
        self._history += self._current_epoch_history
        self._current_epoch_history = []


class Metrics():
    '''
    Metrics class to store the metrics for the experiment

    Args:
        metrics (dict): Dictionary of metrics

    '''
    def __init__(self, metrics):
        self._metrics = {}
        for metric_name, metric_fn in metrics.items():
            self._metrics[metric_name] = Metric(metric_fn, metric_name)
    
    def __str__(self) -> str:
        return f"Metrics_object(function={self.name})"

    def add(self, prediction, target) -> None:
        """Add the prediction and target to all the metrics"""
        for metric in self._metrics.values():
            metric.add(prediction, target)
    
    def __getitem__(self, metric_name) -> float:
        return self._metrics[metric_name].value
    
    def __setitem__(self, metric_name, value) -> None:
        self._metrics[metric_name].append(value)

    def epoch_end(self) -> None:
        """End of epoch"""
        for metric in self._metrics.values():
            metric.epoch_end()

    def save(self, path):
        tosave = {metric.name : {'history': metric._history, 'epoch_history': metric._epoch_history} for metric in self._metrics.values()}
        with open(path, "wb") as fout:
            pickle.dump(tosave, fout)
    
    def visualize_metrics(self, path, n=8):
        """Visualize the metrics"""
        plt.figure(figsize=(20,10))
        for metric in self._metrics.values():
            plt.plot(moving_average(metric.history, n), label=metric.name)
        
        plt.legend()
        plt.savefig(path)


def GetMetricFunction(name:str='iou', border:int=0, threshold:float=0.5, reduction:str='mean') -> callable:
    '''
    Generate the metric function based on the name

    Args:
        name (str): Name of the metric function
        border (int): Border width to obtain from the label and prediction for the metric (0=orignal prediction)
        threshold (float): Threshold to binarize the prediction
        reduction (str): Reduction method for the metric

    Returns:
        metric (function): Metric function
    
    '''
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

import torch

def intersection_over_union(outputs, targets, boundaries, threshold=0.5):
    # Binarize the outputs, targets based on the threshold
    outputs = (outputs > threshold).float()
    targets = (targets > threshold).float()

    # Calculate the intersection and union
    intersection = (outputs * targets).sum(dim=(1, 2))
    union = (outputs + targets).sum(dim=(1, 2)) - intersection
    
    # Calculate the IoU for each image in the batch
    iou = intersection / (union + 1e-8)
    return iou.mean().item()


def intersection_over_union_boundary(predict, targets, boundaries, threshold=0.5):
    # Binarize the outputs, targets based on the threshold
    predict = (predict > threshold).float()
    targets = (boundaries > threshold).float()

    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    # Calculate the intersection and union
    intersection = (predict * targets).sum(dim=1)
    union = (predict + targets).sum(dim=1) - intersection
    
    # Calculate the IoU for each image in the batch
    iou = intersection / (union + 1e-8)
    return iou.mean().item()


def dice_coefficient(outputs, targets, boundaries, threshold=0.5):
    # Binarize the outputs and targets based on the threshold
    outputs = (outputs > threshold).float()
    targets = (targets > threshold).float()
    
    # Calculate the intersection and union
    intersection = (outputs * targets).sum(dim=(1, 2))
    union = (outputs + targets).sum(dim=(1, 2))
    
    # Calculate the Dice coefficient for each image in the batch
    dice = (2 * intersection) / (union + 1e-8)
    return dice.mean().item()

def dice_coefficient_boundary(outputs, targets, boundaries, threshold=0.5):
    # Binarize the outputs and targets based on the threshold
    outputs = (outputs > threshold).float()
    targets = (boundaries > threshold).float()
    
    # Calculate the intersection and union
    intersection = (outputs * targets).sum(dim=(1, 2))
    union = (outputs + targets).sum(dim=(1, 2))
    
    # Calculate the Dice coefficient for each image in the batch
    dice = (2 * intersection) / (union + 1e-8)
    return dice.mean().item()

def precision(outputs, targets, threshold=0.5):
    # Binarize the outputs and targets based on the threshold
    outputs = (outputs > threshold).float()
    targets = (targets > threshold).float()
    
    # Calculate the true positives and false positives
    true_positives = (outputs * targets).sum(dim=(1, 2))
    false_positives = (outputs * (1 - targets)).sum(dim=(1, 2))
    
    # Calculate the precision for each image in the batch
    precision = true_positives / (true_positives + false_positives + 1e-8)
    return precision.mean().item()

def recall(outputs, targets, threshold=0.5):
    # Binarize the outputs and targets based on the threshold
    outputs = (outputs > threshold).float()
    targets = (targets > threshold).float()
    
    # Calculate the true positives and false negatives
    true_positives = (outputs * targets).sum(dim=(1, 2))
    false_negatives = ((1 - outputs) * targets).sum(dim=(1, 2))
    
    # Calculate the recall for each image in the batch
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    return recall.mean().item()

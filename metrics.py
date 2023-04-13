import torch

def intersection_over_union(predict, targets, boundaries, threshold=0.5):
    # Binarize the predict, targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > threshold).float()

    predict = predict.contiguous().view(predict.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)

    # Calculate the intersection and union
    intersection = (predict * targets).sum(dim=1)
    union = (predict + targets).sum(dim=1) - intersection
    
    # Calculate the IoU for each image in the batch
    iou = intersection / (union + 1e-8)
    return iou.mean()

def intersection_over_union_boundary(predict, targets, boundaries, threshold=0.5):
    # Binarize the predict, targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > boundaries).float()

    predict = predict.contiguous().view(predict.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)

    # Calculate the intersection and union
    intersection = (predict * targets).sum(dim=1)
    union = (predict + targets).sum(dim=1) - intersection
    
    # Calculate the IoU for each image in the batch
    iou = intersection / (union + 1e-8)
    return iou.mean().item()


def dice_coefficient(predict, targets, boundaries, threshold=0.5):
    # Binarize the predict and targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > threshold).float()

    predict = predict.contiguous().view(predict.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)
    
    # Calculate the intersection and union
    intersection = (predict * targets).sum(dim=1)
    sum = (predict + targets).sum(dim=1)
    
    # Calculate the Dice coefficient for each image in the batch
    dice = (2 * intersection) / (sum + 1e-8)
    return dice.mean()

def dice_coefficient_boundary(predict, targets, boundaries, threshold=0.5):
    # Binarize the predict and targets based on the threshold
    predict = (predict > threshold).float()
    targets = (boundaries > threshold).float()

    predict = predict.contiguous().view(predict.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)
    
    # Calculate the intersection and union
    intersection = (predict * targets).sum(dim=1)
    sum = (predict + targets).sum(dim=1)
    
    # Calculate the Dice coefficient for each image in the batch
    dice = (2 * intersection) / (sum + 1e-8)
    return dice.mean()

def precision(predict, targets, boundaries, threshold=0.5):
    # Binarize the predict and targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > boundaries).float()
    
    predict = predict.contiguous().view(predict.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)

    # Calculate the true positives and false positives
    true_positives = (predict * targets).sum(dim=1)
    false_positives = (predict * (1 - targets)).sum(dim=1)
    
    # Calculate the precision for each image in the batch
    precision = true_positives / (true_positives + false_positives + 1e-8)
    return precision.mean()

def recall(predict, targets, boundaries, threshold=0.5):
    # Binarize the predict and targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > boundaries).float()

    predict = predict.contiguous().view(predict.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)

    # Calculate the true positives and false negatives
    true_positives = (predict * targets).sum(dim=1)
    false_negatives = ((1 - predict) * targets).sum(dim=1)
    
    # Calculate the recall for each image in the batch
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    return recall.mean()

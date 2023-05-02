import torch
import torch.nn.functional as F


def IoU(predict, targets, threshold=0.5):
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

def IoU_b(predict, targets, threshold=0.5):
    # Binarize the predict, targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > threshold).float()


    targets_b = F.max_pool2d(
        1 - targets, kernel_size=3, stride=1, padding=(3 - 1) // 2)
    targets_b -= 1 - targets

    predict_b = F.max_pool2d(
        1 - predict, kernel_size=3, stride=1, padding=(3 - 1) // 2)
    predict_b -= 1 - predict

    predict_b = predict_b.contiguous().view(predict.shape[0], -1)
    targets_b = targets_b.contiguous().view(targets.shape[0], -1)
    
    # Calculate the intersection and union
    intersection = (predict_b * targets_b).sum(dim=1)
    union = (predict_b + targets_b).sum(dim=1) - intersection
    
    # Calculate the IoU for each image in the batch
    iou = intersection / (union + 1e-8)
    return iou.mean()


def dice_coefficient(predict, targets, threshold=0.5):
    # Binarize the predict and targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > threshold).float()

    predict = predict.contiguous().view(predict.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)
    
    # Calculate the intersection and union
    intersection = (predict * targets).sum(dim=1)
    total = (predict + targets).sum(dim=1)
    
    # Calculate the Dice coefficient for each image in the batch
    dice = (2 * intersection) / (total + 1e-8)
    return dice.mean()

def dice_coefficient_boundary(predict, targets, threshold=0.5):
    # Binarize the predict and targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > threshold).float()

    targets_b = F.max_pool2d(
        1 - targets, kernel_size=3, stride=1, padding=(3 - 1) // 2)
    targets_b -= 1 - targets

    predict_b = F.max_pool2d(
        1 - predict, kernel_size=3, stride=1, padding=(3 - 1) // 2)
    predict_b -= 1 - predict
    
    predict_b = predict_b.contiguous().view(predict.shape[0], -1)
    targets_b = targets_b.contiguous().view(targets.shape[0], -1)
    
    
    # Calculate the intersection and union
    intersection = (predict_b * targets_b).sum(dim=1)
    total = (predict_b + targets_b).sum(dim=1)
    
    # Calculate the Dice coefficient for each image in the batch
    dice = (2 * intersection) / (total + 1e-8)
    return dice.mean()

def precision(predict, targets, threshold=0.5):
    # Binarize the predict and targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > threshold).float()
    
    predict = predict.contiguous().view(predict.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)

    # Calculate the true positives and false positives
    true_positives = (predict * targets).sum(dim=1)
    false_positives = (predict * (1 - targets)).sum(dim=1)
    
    # Calculate the precision for each image in the batch
    precision = true_positives / (true_positives + false_positives + 1e-8)
    return precision.mean()

def precision_boundary(predict, targets, threshold=0.5):
    # Binarize the predict and targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > threshold).float()
    

    targets_b = F.max_pool2d(
        1 - targets, kernel_size=3, stride=1, padding=(3 - 1) // 2)
    targets_b -= 1 - targets

    predict_b = F.max_pool2d(
        1 - predict, kernel_size=3, stride=1, padding=(3 - 1) // 2)
    predict_b -= 1 - predict

    predict_b = predict_b.contiguous().view(predict.shape[0], -1)
    targets_b = targets_b.contiguous().view(targets.shape[0], -1)


    # Calculate the true positives and false positives
    true_positives = (predict_b * targets_b).sum(dim=1)
    false_positives = (predict_b * (1 - targets_b)).sum(dim=1)
    
    # Calculate the precision for each image in the batch
    precision = true_positives / (true_positives + false_positives + 1e-8)
    return precision.mean()

def recall(predict, targets, threshold=0.5):
    # Binarize the predict and targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > threshold).float()

    predict = predict.contiguous().view(predict.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)

    # Calculate the true positives and false negatives
    true_positives = (predict * targets).sum(dim=1)
    false_negatives = ((1 - predict) * targets).sum(dim=1)
    
    # Calculate the recall for each image in the batch
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    return recall.mean()

def recall_boundary(predict, targets, threshold=0.5):
    # Binarize the predict and targets based on the threshold
    predict = (predict > threshold).float()
    targets = (targets > threshold).float()


    targets_b = F.max_pool2d(
        1 - targets, kernel_size=3, stride=1, padding=(3 - 1) // 2)
    targets_b -= 1 - targets

    predict_b = F.max_pool2d(
        1 - predict, kernel_size=3, stride=1, padding=(3 - 1) // 2)
    predict_b -= 1 - predict

    predict_b = predict_b.contiguous().view(predict.shape[0], -1)
    targets_b = targets_b.contiguous().view(targets.shape[0], -1)


    # Calculate the true positives and false negatives
    true_positives = (predict_b * targets_b).sum(dim=1)
    false_negatives = ((1 - predict_b) * targets_b).sum(dim=1)
    
    # Calculate the recall for each image in the batch
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    return recall.mean()



##  Matthew’s correlation coefficient formula

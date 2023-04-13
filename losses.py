import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """ Binary-class dice loss 
    # Args
        smooth: Smoothing factor
        reduction: Reduction method after forward for batch

        predict: Batch of predictions, shape [batch_size, N_CLASSES, H, W]
        target: Target of the same shape
    # Rets
        Loss single value tensor
    """
    def __init__(self, smooth=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "batch sizes don't match"

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        AB_intersection = (predict * target).sum(dim=1)
        A_sum = (predict ** 2).sum(dim=1)
        B_sum = (target ** 2).sum(dim=1) 

        loss = 1 - (AB_intersection + self.smooth) / (A_sum + B_sum + self.smooth)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, targets, boundaries):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=boundaries)
        return loss.mean()

class FocalLoss(nn.Module):
    """ Binary-class dice loss 
    # Args
        alpha: Weighting factor for the rare class
        gamma: Focusing parameter for the rare class
        reduction: Reduction method after forward for batch
        
        predict: Batch of predictions, shape [batch_size, N_CLASSES, H, W]
        target: Target of the same shape
    # Rets
        Loss single value tensor
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predict, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(predict, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class FVLoss(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.BDC = BinaryDiceLoss()
        self.FL = FocalLoss()
        self.alpha = alpha
    
    def forward(self, predict, targets, boundaries):
        return self.BDC(predict, boundaries)*self.alpha + self.FL(predict, targets)
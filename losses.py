import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class BinaryDiceLoss(nn.Module):
    """ Binary-class dice loss 
    # Args
        smooth: Smoothing factor
        reduction: Reduction method after forward for batch

        predict: Batch of logits predictions, shape [batch_size, N_CLASSES, H, W], 
        target: Target of the same shape
    # Rets
        Loss single value tensor
    """
    def __init__(self, smooth=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def __str__(self):
        return f"BinaryDiceLoss(smooth={self.smooth}, reduction={self.reduction})"

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "batch sizes don't match"

        predict = predict.view(predict.shape[0], -1)
        target = target.view(target.shape[0], -1)

        AB_intersection = (predict * target).sum(dim=1)
        A_sum = (predict).sum(dim=1)
        B_sum = (target).sum(dim=1) 

        loss = 1 - (2 * AB_intersection + self.smooth) / (A_sum + B_sum + self.smooth)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class IOULoss(nn.Module):
    """ IOU loss 
    # Args
        smooth: Smoothing factor
        reduction: Reduction method after forward for batch

        predict: Batch of logits predictions, shape [batch_size, N_CLASSES, H, W], 
        target: Target of the same shape
    # Rets
        Loss single value tensor
    """
    def __init__(self, smooth=1, reduction='mean'):
        super(IOULoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def __str__(self):
        return f"IOULoss(smooth={self.smooth}, reduction={self.reduction})"

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "batch sizes don't match"

        predict = predict.view(predict.shape[0], -1)
        target = target.view(target.shape[0], -1)

        AB_intersection = (predict * target).sum(dim=1)
        A_sum = (predict + target).sum(dim=1)
        B_sum = (target).sum(dim=1) 

        loss = 1 - (AB_intersection + self.smooth) / (A_sum + B_sum - AB_intersection + self.smooth)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalLoss(nn.Module):
    """ Focal loss - BCE with balancing
    # Args
        alpha: Weighting factor for the rare class
        gamma: Focusing parameter for the rare class
        reduction: Reduction method after forward for batch
        
        predict: Batch of predictions, shape [batch_size, N_CLASSES, H, W]
        target: Target of the same shape
    # Rets
        Loss single value tensor
    """
    def __init__(self, alpha=0.8, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def __str__(self):
        return "FocalLoss(alpha={}, gamma={})".format(self.alpha, self.gamma)
        
    def forward(self, predict, targets):
        BCE_loss = F.binary_cross_entropy(predict, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def __str__(self):
        return "BoundaryLoss(theta0={}, theta={})".format(self.theta0, self.theta)

    def forward(self, predict, targets):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = predict.shape

        # optionally add multiclass onehot of targets

        # boundary map
        gt_b = F.max_pool2d(
            1 - targets, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - targets

        pred_b = F.max_pool2d(
            1 - predict, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - predict

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
    
class DistanceLoss(nn.Module):
    # TODO - need to add some better balance
    def __init__(self, device='cpu', theta=9):
        super().__init__()
        self.theta = theta
        self.device = device
    
    def forward(self, predict, targets):

        n, c, _, _ = predict.shape

        # make distance weight field
        p = (self.theta - 1) // 2
        t_p = F.pad(targets, (p, p, p, p), mode='constant', value=0)  

        gt_d = F.conv2d(
                    1-t_p, torch.ones([1,1,self.theta,self.theta]).to(self.device), stride=1)

        gt_d *= 1-targets

        fn = (1-predict) * targets

        fn = fn.view(n, c, -1)
        gt_d = gt_d.view(n, c, -1)
        predict = predict.view(n, c, -1)

        fp = gt_d*predict

        loss = torch.mean(torch.sum(fp + fn*4 , dim=2))
        return loss

class TverskyLoss(nn.Module):
    def __init__(self, smooth=1, alpha=1, beta=1, reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "batch sizes don't match"

        predict = predict.view(predict.shape[0], -1)
        target = target.view(target.shape[0], -1)

        TP = (predict * target).sum(dim=1)    
        FP = ((1-target) * predict).sum(dim=1) 
        FN = (target * (1-predict)).sum(dim=1) 
       
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        
        loss = 1 - Tversky

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()   
        else:
            return loss

class DistanceWeightBCELoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1, theta0=3, theta=9):
        super().__init__()
        self.alpha = alpha
        self.theta0 = theta0
        self.theta = theta
        self.reduction = reduction
        self.gb = T.GaussianBlur(theta)
        self.BCE = nn.BCELoss(reduction='none')

    def __str__(self):
        return "DistanceWeightBCELoss(alpha={}, theta0={}, theta={})".format(self.alpha, self.theta0, self.theta)
    
    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "batch sizes don't match"

        target_boundary = F.max_pool2d(
            1 - target, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        target_boundary -= 1 - target

        weights = self.gb(target_boundary) + self.alpha # smoothen the boundary and add "base" weight
        bce_values = self.BCE(predict, target) # alternative is to use F.binary_cross_entropy_with_logits(predict, target, reduction='none', pos_weight=weights) directly here

        weights = weights.view(predict.shape[0], -1)
        bce_values = bce_values.view(predict.shape[0], -1)

        loss = bce_values*weights # weight the loss with boundary-distance weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 

class CombLoss(nn.Module):
    def __init__(self, alpha=0.6):
        # check for the same magnitute of loss values
        super().__init__()
        self.AreaLoss = BinaryDiceLoss() #nn.BCELoss(reduction='none') #FocalLoss()
        self.BoundaryLoss = BoundaryLoss()
        self.alpha = alpha

    def __str__(self):
        return "CombLoss(alpha={}) : alpha*{} + (1-alpha)*{}".format(self.alpha, self.BoundaryLoss, self.AreaLoss)
    
    def forward(self, predict, targets):
        return self.BoundaryLoss(predict, targets)*self.alpha + (1-self.alpha)*self.AreaLoss(predict, targets)


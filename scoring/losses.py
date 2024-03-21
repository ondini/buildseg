import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch.nn import BCEWithLogitsLoss


def nll_loss(output, target):
    return F.nll_loss(output, target)

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
        A_sum = (predict).sum(dim=1)
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

        gt_d = F.conv2d(1-t_p, torch.ones([1,1,self.theta,self.theta]).to(self.device), stride=1)

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
        self.AreaLoss = BinaryDiceLoss() #BinaryDiceLoss() #nn.BCELoss(reduction='none') #FocalLoss()
        self.BoundaryLoss = BoundaryLoss()
        self.alpha = alpha

    def __str__(self):
        return "CombLoss(alpha={}) : alpha*{} + (1-alpha)*{}".format(self.alpha, self.BoundaryLoss, self.AreaLoss)
    
    def forward(self, predict, targets):
        return self.BoundaryLoss(predict, targets)*self.alpha + (1-self.alpha)*self.AreaLoss(predict, targets)

### SPECIFIC LOSS REIMPLEMENTATION FOR MASKRCNN

def make_fasrcnn_loss(loss_name, fast=False):
    # get loss function by its name from this file 
    loss_fn = globals()[loss_name]()
    
    def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Computes the loss for Faster R-CNN.

        Args:
            class_logits (Tensor)
            box_regression (Tensor)
            labels (list[BoxList])
            regression_targets (Tensor)

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss
    
    def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
        """
        Args:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        # print('aaaaaaaaa')
        discretization_size = mask_logits.shape[-1]
        labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
        mask_targets = [
            project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
        ]

        labels = torch.cat(labels, dim=0)
        mask_targets = torch.cat(mask_targets, dim=0)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
        )
        return mask_loss

    return maskrcnn_loss if not fast else fastrcnn_loss

## UTILS FOR OFFICIAL MASKRCNN LOSS

from torchvision.ops import boxes as box_ops, roi_align

def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]
import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    """ Binary-class dice loss 
    # Args
        predict: Batch of predictions, shape [batch_size, N_CLASSES, H, W]
        target: Target of the same shape
    # Rets
        Loss single value tensor
    """
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "batch sizes don't match"

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        AB_intersection = (predict * target).sum(dim=1)
        A_sum = (predict ** 2).sum(dim=1)
        B_sum = (target ** 2).sum(dim=1) 

        loss = 1 - (AB_intersection + self.smooth) / (A_sum + B_sum + self.smooth)

        return loss.mean()



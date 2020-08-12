
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, targets):
        num = output.size(0)
        smooth = 1

        assert output.size() == targets.size()

        probs = F.sigmoid(output)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        loss = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        loss = 1 - loss.sum() / num
        return loss






class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = input.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        # dice = DiceLoss()
        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss



def DFL(y_true, y_pred):
    # Code written by Seung hyun Hwang
    gamma = 1.1
    alpha = 0.48
    smooth = 1.
    epsilon = 1e-7
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    # dice loss
    intersection = (y_true * y_pred).sum()
    dice_loss = (2. * intersection + smooth) / ((y_true * y_true).sum() + (y_pred * y_pred).sum() + smooth)

    # focal loss
    y_pred = torch.clamp(y_pred, epsilon)

    pt_1 = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred))
    pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))
    focal_loss = -torch.mean(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - \
                 torch.mean((1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))


    return focal_loss - torch.log(dice_loss)
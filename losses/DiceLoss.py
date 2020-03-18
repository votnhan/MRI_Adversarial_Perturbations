import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import class_to_one_hot


def _cal_dice_score(output, target, num_classes, smooth):
    axes = (0, -1, -2)
    target_one_hot = class_to_one_hot(target, num_classes)
    softmax_op = F.softmax(output, dim=1)
    intersection = (softmax_op * target_one_hot).sum(dim=axes)
    softmax_sum = softmax_op.sum(dim=(axes))
    target_one_hot_sum = target_one_hot.sum(dim=axes)
    dice_score = (2. * intersection + smooth) / (softmax_sum + target_one_hot_sum + smooth)
    return dice_score


class DiceLoss2dLinear(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-6):
        super(DiceLoss2dLinear, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, output, target):
        dice_score = _cal_dice_score(output, target, self.num_classes, self.smooth)
        negative = 1 - dice_score
        per_class_ds = negative.mean()
        return per_class_ds


class DiceLoss2dLogarithmic(nn.Module):
    def __init__(self, num_classes=4, gamma=1, smooth=1e-6):
        super(DiceLoss2dLogarithmic, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, output, target):
        dice_score = _cal_dice_score(output, target, self.num_classes, self.smooth)
        logarithmic_exp = (-1 * torch.log(dice_score)) ** self.gamma
        per_class_ds = logarithmic_exp.mean()
        return per_class_ds

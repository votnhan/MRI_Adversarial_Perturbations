import torch.nn as nn
import torch.nn.functional as F
from .utils import class_to_one_hot


class DiceLoss2d(nn.Module):
    def __init__(self, num_classes=4, smooth=1.0):
        super(DiceLoss2d, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, output, target):
        target_one_hot = class_to_one_hot(target, self.num_classes)
        softmax_op = F.softmax(output, dim=1)
        intersection = (softmax_op * softmax_op).sum(dim=(-1, -2))
        softmax_sum = softmax_op.sum(dim=(-1, -2))
        target_one_hot_sum = target_one_hot.sum(dim=(-1, -2))
        dice_score = (2.* intersection + self.smooth) / (softmax_sum + target_one_hot_sum + self.smooth)
        per_class_ds = dice_score.mean(1)
        avg_batches = per_class_ds.mean()
        return 1 - avg_batches

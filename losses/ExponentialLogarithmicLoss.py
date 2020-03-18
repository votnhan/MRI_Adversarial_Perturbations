import torch.nn as nn
import torch
from .DiceLoss import DiceLoss2dLogarithmic
from .CrossEntropyLoss import ExponentialCrossEntropyLoss2d


class ExponentialLogarithmicLoss(nn.Module):
    def __init__(self, frequency=None, alpha=0.5, beta=0.5, gamma=1, num_classes=4, smooth=1e-6):
        super(ExponentialLogarithmicLoss, self).__init__()
        if frequency is None:
            self.weight = [1.0 / num_classes for i in range(num_classes)]
        else:
            self.weight = frequency
        self.weight = torch.FloatTensor(self.weight)
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss2dLogarithmic(num_classes, gamma, smooth)
        self.weighted_exp_ce_loss = ExponentialCrossEntropyLoss2d(self.weight, gamma)

    def forward(self, output, target):
        dice_loss = self.dice_loss(output, target)
        exp_ce_loss = self.weighted_exp_ce_loss(output, target)
        return self.alpha*dice_loss + self.beta*exp_ce_loss


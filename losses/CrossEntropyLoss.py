import torch.nn.functional as F
import torch.nn as nn
import torch
from .utils import class_to_one_hot


class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, frequency=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        if type(frequency) is list:
            frequency = torch.FloatTensor(frequency)
        weight = 1.0 / frequency
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.name = 'ce_loss'

    def forward(self, inputs, targets):
        long_ts_target = targets.type(torch.LongTensor).to(inputs.get_device())
        return self.nll_loss(F.log_softmax(inputs), long_ts_target)


class ExponentialCrossEntropyLoss2d(nn.Module):
    def __init__(self, frequency, gamma=1, num_classes=4):
        super(ExponentialCrossEntropyLoss2d, self).__init__()
        if type(frequency) is list:
            frequency = torch.FloatTensor(frequency)
        self.weight = 1.0 / frequency
        self.gamma = gamma
        self.num_classes = num_classes
        self.name = 'exp_ce_loss'

    def forward(self, output, target):
        axes = (0, -1, -2)
        log_softmax = F.log_softmax(output, dim=1)
        one_hot_target = class_to_one_hot(target, self.num_classes)
        nll_loss = (-1. * log_softmax * one_hot_target).mean(dim=axes)
        weighted_exp_loss = self.weight.to(output.get_device()) * (nll_loss ** self.gamma)
        per_class_nll = weighted_exp_loss.mean()
        return per_class_nll

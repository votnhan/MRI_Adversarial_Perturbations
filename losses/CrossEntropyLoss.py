import torch.nn.functional as F
import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class ExponentialCrossEntropyLoss2d(nn.Module):
    def __init__(self, frequency, gamma):
        super(ExponentialCrossEntropyLoss2d, self).__init__()
        self.weight = 1.0 / frequency
        self.gamma = gamma

    def forward(self, output, target):
        axes = (0, -1, -2)
        log_softmax = F.log_softmax(output, dim=1)
        nll_loss = (-1. * log_softmax * target).sum(dim=axes)
        weighted_exp_loss = self.weight.to(output.get_device()) * (nll_loss ** self.gamma)
        per_class_nll = weighted_exp_loss.mean()
        return per_class_nll

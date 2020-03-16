import torch
import torch.nn.functional as F


def pre_process(output):
    softmax_op = F.softmax(output, dim=1)
    output_pred = torch.argmax(softmax_op, dim=1)
    return output_pred


def dice_score(output, target, smooth=1.0):
    axis = (-1, -2)
    intersection = torch.sum(output*target, dim=axis)
    sum_output = torch.sum(output, dim=axis)
    sum_target = torch.sum(target, dim=axis)
    dice = (2*intersection + smooth) / (sum_output + sum_target + smooth)
    return torch.mean(dice)


def dice_whole_tumor(output, target):
    cls_output = pre_process(output)
    mask_op = cls_output > 0
    mask_target = target > 0
    return dice_score(mask_op, mask_target)


def dice_tumor_core(output, target):
    cls_output = pre_process(output)
    mask_op = (cls_output == 1) | (cls_output == 3)
    mask_target = (target == 1) | (target == 3)
    return dice_score(mask_op, mask_target)


def dice_enhancing_tumor(output, target):
    cls_output = pre_process(output)
    mask_op = cls_output == 3
    mask_target = target == 3
    return dice_score(mask_op, mask_target)


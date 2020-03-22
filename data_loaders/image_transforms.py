import cv2
import torch
import numpy as np
from utils import scale_intensity_input


class ToTensor:
    def __call__(self, np_data):
        cp_data = np_data.copy().astype(np.float32)
        return torch.from_numpy(cp_data)


class Normalization:
    def __init__(self, means, stds):
        self.means = torch.FloatTensor(means)
        self.stds = torch.FloatTensor(stds)

    def __call__(self, np_data):
        means = self.means.view(-1, 1, 1)
        stds = self.stds.view(-1, 1, 1)
        return (np_data - means) / stds


class GaussianBlur:
    def __init__(self, kernel_size, border_type):
        self.kernel_size = kernel_size
        self.border_type = border_type

    def __call__(self, image):
        blur_image = cv2.GaussianBlur(image, self.kernel_size, borderType=self.border_type)
        return blur_image


class IntensityScale:
    def __init__(self, range_scale=None):
        if range_scale is None:
            range_scale = [0, 1]
        self.range_scale = range_scale
        self.new_range = self.range_scale[1] - self.range_scale[0]

    def __call__(self, image):
        axes = (-1, -2)
        max_val = np.amax(image, axis=axes)
        min_val = np.amin(image, axis=axes)
        scale_arr = scale_intensity_input(image, min_val, max_val, self.range_scale)
        return scale_arr

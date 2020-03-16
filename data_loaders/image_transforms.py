import cv2
import torch


class ToTensor:
    def __call__(self, np_data):
        return torch.from_numpy(np_data)


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
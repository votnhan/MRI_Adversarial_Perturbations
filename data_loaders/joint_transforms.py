import PIL
import cv2
import random
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomSizeAndCrop:
    def __init__(self, size, range_scale=(0.8, 1.2), interpolation=PIL.Image.BILINEAR):
        self.size = size
        self.scale_min = range_scale[0]
        self.scale_max = range_scale[1]
        self.interpolation = interpolation
        self.crop = RandomCrop(size)

    def __call__(self, np_data, np_label):
        scale_factor = random.uniform(self.scale_min, self.scale_max)
        w, h = np_data.shape[1:]
        new_w = int(scale_factor*w)
        new_h = int(scale_factor*h)

        np_data = resize(np_data, (new_w, new_h), self.interpolation)
        np_label = resize(np_label, (new_w, new_h), cv2.INTER_NEAREST)
        np_data, np_label = self.crop(np_data, np_label)
        return np_data, np_label


class RandomCrop:
    def __init__(self, crop_size=224):
        self.crop_size = crop_size

    def __call__(self, np_data, np_label):
        w, h = np_data.shape[1:]

        if (w == self.crop_size) and (h == self.crop_size):
            return np_data, np_label

        if (w < self.crop_size) or (h < self.crop_size):
            return np_data, np_label

        if w == self.crop_size:
            x1 = 0
        else:
            x1 = random.randint(0, w - self.crop_size)

        if h == self.crop_size:
            y1 = 0
        else:
            y1 = random.randint(0, h - self.crop_size)

        cropped_data = np_data[:, x1: x1+self.crop_size, y1: y1+self.crop_size]
        cropped_label = np_label[:, x1: x1+self.crop_size, y1: y1+self.crop_size]
        return cropped_data, cropped_label


class RandomHorizonFlip:
    def __init__(self, percent=0.5):
        self.percent = percent

    def __call__(self, np_data, np_label):
        flip = random.random() < self.percent
        if flip:
            np_data = np.flip(np_data, axis=1)
            np_label = np.flip(np_label, axis=1)
        return np_data, np_label


def resize(np_arr, size, interpolation=cv2.INTER_LINEAR):
    np_image = np.transpose(np_arr, axes=(1, 2, 0))
    resize_image = cv2.resize(np_image, dsize=size, interpolation=interpolation)
    np_data = np.transpose(resize_image, axes=(2, 0, 1))
    return np_data

import pandas as pd
import json
import torch.nn.functional as F
import torch
import numpy as np
import os
import copy
from itertools import repeat
from pathlib import Path
from collections import OrderedDict
from PIL import Image

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)


def inf_loop(data_loader):
    """ wrapper function for endless data loader. """
    for loader in repeat(data_loader):
        yield from loader


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def result2class(tensor_output):
    softmax_op = F.softmax(tensor_output, dim=1)
    class_op = torch.argmax(softmax_op, dim=1)
    return class_op


def save_output(tensor_output, file_names, epoch, output_dir, percent=0.5):
    class_op = result2class(tensor_output)
    num_output = tensor_output.size(0)
    num_save = int(percent*num_output)
    for i in range(num_save):
        output = class_op[i].cpu().numpy()
        name, ext = os.path.splitext(file_names[i])
        new_name = '{}_ep{}{}'.format(name, str(epoch), ext)
        path_save = os.path.join(output_dir, new_name)
        np.savez_compressed(path_save, output)
        print('Save output of sample: {} for track'.format(name))


# Use for both output and target tensor
def save_mask2image(tensors, tensor_names, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if len(tensors.size()) > 3:
        class_op = result2class(tensors)
    else:
        class_op = tensors
    n_samples = tensors.size(0)
    for i in range(n_samples):
        output = class_op[i]
        tensor_name, ext = os.path.splitext(tensor_names[i])
        output_path = os.path.join(output_dir, '{}.png'.format(tensor_name))
        output_image = mask2image(output.cpu().numpy())
        save_image(output_image, output_path)
        print('Save output: {}'.format(output_path))


def mask2image(label):
    label_clone = copy.deepcopy(label)
    label_clone[label == 4] = 3
    w, h = label.shape
    result = np.zeros((w, h, 3))
    result[label_clone == 2] = red
    result[label_clone == 1] = green
    result[label_clone == 3] = blue
    return result


def save_image(np_array, file_path):
    image = Image.fromarray(np_array.astype(np.uint8))
    image.save(file_path)


def cal_frequency_of_label(label_dir):
    label_dict = {
        0: 0,
        1: 0,
        2: 0,
        4: 0
    }
    labels = os.listdir(label_dir)
    num_labels = len(labels)
    for i, label in enumerate(labels):
        print('{}/{}'.format(i, num_labels))
        path_label = os.path.join(label_dir, label)
        label = np.load(path_label)['arr_0']
        for k, v in label_dict.items():
            label_dict[k] += np.sum(label == k)
    sum_voxel = sum([v for k, v in label_dict.items()])
    for k, v in label_dict.items():
        label_dict[k] = label_dict[k] / sum_voxel
    return label_dict


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
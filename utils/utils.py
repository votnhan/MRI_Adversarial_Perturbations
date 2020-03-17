import pandas as pd
import json
import torch.nn.functional as F
import torch
import numpy as np
import os
from itertools import repeat
from pathlib import Path
from collections import OrderedDict


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


def save_output(tensor_output, file_names, epoch, output_dir, percent=0.5):
    softmax_op = F.softmax(tensor_output, dim=1)
    class_op = torch.argmax(softmax_op, dim=1)
    num_output = tensor_output.size(0)
    num_save = int(percent*num_output)
    for i in range(num_save):
        output = class_op[i].cpu().numpy()
        name, ext = os.path.splitext(file_names[i])
        new_name = '{}_ep{}{}'.format(name, str(epoch), ext)
        path_save = os.path.join(output_dir, new_name)
        np.savez_compressed(path_save, output)
        print('Save output of sample: {} for track'.format(name))


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
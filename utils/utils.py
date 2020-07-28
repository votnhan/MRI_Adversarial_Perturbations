import pandas as pd
import json
import torch.nn.functional as F
import torch
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import nibabel as nib
import glob
import shutil
from itertools import repeat
from pathlib import Path
from collections import OrderedDict
from PIL import Image

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)


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


def show_sample(arr, figure_size=(12, 24), v_mins=None, v_maxs=None):
    if v_mins is None:
      v_mins = np.amin(arr, axis=(-1, -2))
    
    if v_maxs is None:
      v_maxs = np.amax(arr, axis=(-1, -2))
    n_modals = arr.shape[0]
    fig, axes = plt.subplots(1, n_modals, figsize=figure_size)
    for i in range(n_modals):
      axes[i].imshow(arr[i], cmap='gray', vmin=v_mins[i], vmax=v_maxs[i])
      axes[i].axis('off')


def get_data_from_subject_dir(subject_path, modals, modal_ext):
    subject_name = os.path.basename(subject_path)
    list_gather = []
    list_affine = []
    for modal in modals:
        modal_file = '{}_{}.{}'.format(subject_name, modal, modal_ext)
        modal_path = os.path.join(subject_path, modal_file)
        modal_image = nib.load(modal_path)
        modal_arr = modal_image.get_data()
        list_gather.append(np.expand_dims(modal_arr, 0))
        list_affine.append(modal_image.get_affine())

    all_modals_arr = np.concatenate(list_gather, 0)
    arr_temp = np.expand_dims(all_modals_arr, 0)
    arr_temp = np.transpose(arr_temp, (4, 1, 2, 3, 0))
    result = arr_temp[:, :, :, :, 0]
    return result, list_affine


def attack_dataset(train_dir, output_dir, attack_model, dataset_name ='Brats2018'):
    modals = ['t1', 't1ce', 'flair', 't2']
    modal_ext = 'nii.gz'
    range_scale = [-1, 1]
    epsilon = 0.1
    n_folder_keep = 3
    pattern = os.path.join(train_dir, '*', '*')
    paths_folder = glob.glob(pattern)
    n_subjects = len(paths_folder)
    print('Attacking dataset {} start'.format(dataset_name))
    for i, path_folder in enumerate(paths_folder):
        print('{}/{}'.format(i+1, n_subjects))
        split_path = path_folder.split('/')
        new_path_folder = os.path.join(output_dir, *split_path[-n_folder_keep:])
        result = attack_patient_data(path_folder, new_path_folder, attack_model, modals, modal_ext, range_scale,
                                     epsilon)
        if not result:
            print('Folder {} existed !'.format(new_path_folder))
        else:
            subject_name = os.path.basename(path_folder)
            label_file = '{}_seg.{}'.format(subject_name, modal_ext)
            label_path_src = os.path.join(path_folder, label_file)
            label_path_des = os.path.join(new_path_folder, label_file)
            shutil.copy(label_path_src, label_path_des)

    print('Attacking dataset {} end'.format(dataset_name))


def attack_patient_data(patient_dir, output_dir, model, modals, modal_extension, range_scale, epsilon=0.1):
    axes = (-1, -2)
    subject_name = os.path.basename(patient_dir)
    if not os.path.exists(patient_dir):
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)

    data_patient, list_affine = get_data_from_subject_dir(patient_dir, modals, modal_extension)
    mask_not_brain = data_patient <= 0
    min_val = np.amin(data_patient, axis=axes)
    max_val = np.amax(data_patient, axis=axes)
    intensity_scaled_input = scale_intensity_input(data_patient, min_val, max_val, range_scale)
    noise_data, _, _, _ = adversarial_attack(intensity_scaled_input, model, range_scale, epsilon)
    np_noise_data = noise_data.detach().cpu().numpy()
    reversed_intensity_output = reverse_intensity_scale(np_noise_data, min_val, max_val, range_scale)
    reversed_intensity_output[mask_not_brain] = 0.0
    np_noise_data_tp = np.transpose(reversed_intensity_output, (1, 2, 3, 0))
    for i in range(np_noise_data_tp.shape[0]):
        new_image = nib.Nifti1Image(np_noise_data_tp[i].astype(np.int16), list_affine[i])
        new_image_name = '{}_{}.{}'.format(subject_name, modals[i], modal_extension)
        new_image_path = os.path.join(output_dir, new_image_name)
        new_image.to_filename(new_image_path)
        print('---- modal: {}'.format(modals[i]))

    print('Attack {}: done'.format(patient_dir))
    return True


def scale_intensity_input(arr, min_val, max_val, range_scale):
    old_range = max_val - min_val
    new_range = range_scale[1] - range_scale[0]
    n_samples = arr.shape[0]
    n_channels = arr.shape[1]
    scaled_input = np.zeros(arr.shape)
    for i in range(n_samples):
        for j in range(n_channels):
            if old_range[i][j] == 0:
                scaled_input[i, j][:, :] = range_scale[0]
            else:
                scaled_input[i, j][:, :] = (arr[i][j] - min_val[i][j]) * new_range / old_range[i][j] + range_scale[0]

    return scaled_input


def reverse_intensity_scale(arr, min_val, max_val, range_scale):
    new_range = max_val - min_val
    old_range = range_scale[1] - range_scale[0]
    n_channels = arr.shape[1]
    n_samples = arr.shape[0]
    scaled_input = np.zeros(arr.shape)
    for i in range(n_samples):
        for j in range(n_channels):
            if new_range[i][j] == 0:
                scaled_input[i, j][:, :] = min_val[i][j]
            else:
                scaled_input[i, j] = (arr[i][j] - range_scale[0]) * new_range[i][j] / old_range + min_val[i][j]

    return scaled_input


def adversarial_attack(arr, model, range_scale, epsilon=0.05):
    model.eval()
    tensor_input = torch.from_numpy(arr).type(torch.FloatTensor)

    if len(tensor_input.size()) == 3:
        tensor_input = tensor_input.unsqueeze(0)

    tensor_input = tensor_input.to(next(model.parameters()).device)
    noise = model(tensor_input)
    noise_clamped = inf_norm_adjust(noise, epsilon)
    noise_input = noise_clamped + tensor_input
    result = torch.clamp(noise_input, range_scale[0], range_scale[1])
    return result, noise_input, noise_clamped, noise


def demo_attack(data, model, range_scale, epsilon):
    model.eval()
    mask_not_brain = data <= 0
    axes = (-1, -2)
    min_val = np.amin(data, axis=axes)
    max_val = np.amax(data, axis=axes)
    scaled_input = scale_intensity_input(data, min_val, max_val, range_scale)
    results = adversarial_attack(scaled_input, model, range_scale, epsilon)
    np_results = [x.detach().cpu().numpy() for x in results]
    np_results_rerv_scale = [reverse_intensity_scale(x, min_val, max_val, range_scale) for x in np_results]
    
    for x in np_results_rerv_scale:
        x[mask_not_brain] = 0.0
    
    return np_results_rerv_scale


def get_adv_for_one_sample(data, model, range_scale, epsilon):
    new_data = np.expand_dims(data, axis=0)
    results = demo_attack(new_data, model, range_scale, epsilon)
    result = results[0]
    return result.astype(data.dtype)


def get_adv_for_multiple_samples(data, model, range_scale, epsilon):
    results = demo_attack(data, model, range_scale, epsilon)
    noise_input_clamped = results[0]
    return noise_input_clamped.astype(data.dtype)


def attack_list_samples(container, list_samples, output_container, range_scale, model, epsilon=0.1):
    if not os.path.exists(output_container):
        os.makedirs(output_container)

    data = []
    for sample in list_samples:
        sample_path = os.path.join(container, '{}.npz'.format(sample))
        sample_data = np.load(sample_path)['arr_0']
        data.append(np.expand_dims(sample_data, axis=0))

    data_arr = np.concatenate(data, axis=0)
    results = get_adv_for_multiple_samples(data_arr, model, range_scale, epsilon)

    for i, x in enumerate(list_samples):
        noise_sample = results[i]
        output_path = os.path.join(output_container, '{}.npz'.format(x))
        np.savez_compressed(output_path, noise_sample)
        print('Save adversary of sample: {} for track'.format(x))


def demo_segmentation(data, model, range_scale, over_classes=False):
    model.eval()
    axes = (-1, -2)
    min_val = np.amin(data, axis=axes)
    max_val = np.amax(data, axis=axes)
    scaled_input = scale_intensity_input(data, min_val, max_val, range_scale)
    tensor_input = torch.from_numpy(scaled_input).type(torch.FloatTensor)
    device = next(model.parameters()).device
    output_ts = model(tensor_input.to(device))
    output_cls = result2class(output_ts)
    output_np = output_cls.detach().cpu().numpy()

    if over_classes:
        one_hot = np.eye(4)[output_np]
        output_np = np.transpose(one_hot, (0, 3, 1, 2)).astype(np.bool_)

    if output_np.shape[0] == 1:
        output_np = output_np[0]

    return output_np


def demo_result_after_attack(data, label, trained_model, attacking_model, range_scale, epsilon):
    trained_model.eval()
    attacking_model.eval()
    tumor_segmented = demo_segmentation(data, trained_model, range_scale)
    image_tumor_segmented = mask2image(tumor_segmented)
    noise_input = demo_attack(data, attacking_model, range_scale, epsilon)
    noise_output = demo_segmentation(noise_input, trained_model, range_scale)
    image_noise_tumor = mask2image(noise_output)
    image_label = mask2image(label)
    return image_label, image_tumor_segmented, image_noise_tumor


def result2class(tensor_output):
    softmax_op = F.softmax(tensor_output, dim=1)
    class_op = torch.argmax(softmax_op, dim=1)
    return class_op


def save_output(tensor_output, file_names, epoch, output_dir, percent=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

def mask2image_over_classes(one_hot_mask):
    list_colors = [white, green, red, blue]
    n_classes = one_hot_mask.shape[0]
    w, h = one_hot_mask.shape[1:]
    results = []
    for i in range(n_classes):
        mask = np.zeros((w, h, 3))
        mask[one_hot_mask[i]] = list_colors[i]
        results.append(mask)

    return results

def save_image(np_array, file_path):
    image = Image.fromarray(np_array.astype(np.uint8))
    image.save(file_path)


def inf_norm_adjust(noises, epsilon=0.05):
    n_samples, n_channels = noises.size()[:2]
    for i in range(n_samples):
        for j in range(n_channels):
            inf_norm = noises[i].data[j].abs().max()
            scale_factor = min(1.0, epsilon / inf_norm)
            noises[i].data[j] *= scale_factor

    return noises


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
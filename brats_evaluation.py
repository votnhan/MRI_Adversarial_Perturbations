import numpy as np
import pandas as pd
import os
import argparse
from scipy.spatial.distance import directed_hausdorff


def read_mask(mask_path):
    arr = np.load(mask_path, allow_pickle=True)['arr_0']
    return arr


def get_whole_tumor_mask(data):
    return data > 0


# Class 4 <=> class 3
def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_score(output, label, eps=1e-6):
    intersection = np.sum(output*label)
    sum_output = np.sum(output)
    sum_target = np.sum(label)
    dice = (2*intersection + eps)/(sum_output + sum_target + eps)
    return dice


def sensitivity(output, label, eps=1e-6):
    tp = np.sum(output*label)
    return (tp + eps)/(np.sum(label) + eps)


def specificity(output, label, eps=1e-6):
    non_output = output == 0
    non_label = label == 0
    tn = np.sum(non_output*non_label)
    return (tn+eps)/(np.sum(non_label) + eps)


def hausdorff_distance(output, label):
    coor_label = np.where(label == 1)
    coor_label = np.asarray(coor_label).T
    coor_output = np.where(output == 1)
    coor_output = np.asarray(coor_output).T
    d_ab = directed_hausdorff(coor_label, coor_output)[0]
    d_ba = directed_hausdorff(coor_output, coor_label)[0]
    hd_ab = max(d_ab, d_ba)
    return hd_ab


metrics_dict = {
    'dice_score': dice_score,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'hd_distance': hausdorff_distance
}

masking_function = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
header_default = ('Whole tumor', 'Tumor core', 'Enhancing tumor')
np_extension = 'npz'


def evaluation(metric_names, output_path, label_path, output_metrics):
    file_names = os.listdir(output_path)
    file_names.sort()
    rows = [[] for i in range(len(metric_names))]
    subject_names = []
    n_samples = len(file_names)
    for idx, name in enumerate(file_names):
        subject_name = name.split('_ep')[0]
        output_file = os.path.join(output_path, name)
        label_file = os.path.join(label_path, '{}.{}'.format(subject_name, np_extension))
        output = read_mask(output_file)
        output[output == 3] = 4
        label = read_mask(label_file)
        for i, metric in enumerate(metric_names):
            if metric not in metrics_dict:
                print('{} is not implemented !'.format(metric))
                continue
            line = [metrics_dict[metric](func(output), func(label)) for func in masking_function]
            rows[i].append(line)

        subject_names.append(subject_name)
        
        if idx % 1000 == 0:
            print('{}/{}'.format(idx, n_samples))

    for i, metric in enumerate(metric_names):
        export_csv_file(rows[i], header_default, subject_names, metric, output_metrics)


def export_csv_file(rows, header, index, metric_name, output_fd):
    if not os.path.exists(output_fd):
        os.makedirs(output_fd)

    df = pd.DataFrame.from_records(rows, columns=header, index=index)
    output_file = '{}.csv'.format(metric_name)
    output_path = os.path.join(output_fd, output_file)
    df.to_csv(output_path)


parser = argparse.ArgumentParser(description='Evaluation of model')
parser.add_argument('--metrics_name', type=str, nargs='+',
                    default=['dice_score', 'sensitivity', 'specificity', 'hd_distance'])

parser.add_argument('--output_path', type=str)
parser.add_argument('--label_path', type=str)
parser.add_argument('--output_metrics', type=str, default='output')

if __name__ == '__main__':
    args = parser.parse_args()
    evaluation(args.metrics_name, args.output_path, args.label_path, args.output_metrics)


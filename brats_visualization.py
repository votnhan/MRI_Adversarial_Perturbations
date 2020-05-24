import numpy as np
import matplotlib.pyplot as plt
import os

modalities = ['t1', 't1ce', 'flair', 't2']
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)


def scaled_slice(arr, new_range):
    arr = arr.astype(np.float32)
    min_val = np.min(arr)
    max_val = np.max(arr)
    new_min, new_max = new_range
    result = (arr - min_val) * (new_max - new_min)/ (max_val - min_val) + new_min
    return result


def show_channel(channel):
    rotated_channel = np.rot90(channel, k=3)
    fig, axe = plt.subplots(1, 1)
    axe.imshow(rotated_channel, cmap='gray', origin='lower')
    axe.axes.xaxis.set_visible(False)
    axe.axes.yaxis.set_visible(False)


def show_image(image):
    rotated_image = np.rot90(image, k=3)
    fig, axe = plt.subplots(1, 1)
    axe.imshow(rotated_image, origin='lower')
    axe.axes.xaxis.set_visible(False)
    axe.axes.yaxis.set_visible(False)


def visualize_sample(path_container, sample_name, output_container):
    if not os.path.exists(output_container):
        os.makedirs(output_container)

    sample_file = '{}.npz'.format(sample_name)
    sample_path = os.path.join(path_container, sample_file)
    arr = np.load(sample_path)['arr_0']
    n_channels = arr.shape[0]
    for i in range(n_channels):
        show_channel(arr[i])
        output_file = '{}_{}.png'.format(sample_name, modalities[i])
        plt.savefig(os.path.join(output_container, output_file), bbox_inches='tight')

    print('Saved channel figures for {}.'.format(sample_name))


def overlay_label_for_sample(path_container, sample_name, label_container, output_container, modal_name, epoch=None):
    if not os.path.exists(output_container):
        os.makedirs(output_container)

    sample_file = '{}.npz'.format(sample_name)
    sample_path = os.path.join(path_container, sample_file)
    arr = np.load(sample_path)['arr_0']
    modal = arr[modalities.index(modal_name)]
    scaled_modal = scaled_slice(modal, (0, 255)).astype(np.uint8)

    w, h = scaled_modal.shape
    new_slice = np.broadcast_to(scaled_modal, (3, w, h))
    image_arr = np.transpose(new_slice, (1, 2, 0))
    image_arr.setflags(write=True)

    if epoch:
        label_file = '{}_ep{}.npz'.format(sample_name, epoch)
        label_path = os.path.join(label_container, label_file)
    else:
        label_path = os.path.join(label_container, sample_file)

    label = np.load(label_path)['arr_0']
    label[label == 3] = 4

    ed = label == 2
    ncr = label == 1
    eh = label == 4

    mask_image = np.zeros((w, h, 3), dtype=np.uint8)
    mask_image[ed] = red
    mask_image[ncr] = green
    mask_image[eh] = blue

    image_arr[ed] = 0
    image_arr[ncr] = 0
    image_arr[eh] = 0
    overlay = image_arr + mask_image
    show_image(overlay)
    if epoch:
        output_image = '{}_{}_overlay_ep{}.png'.format(sample_name, modal_name, epoch)
    else:
        output_image = '{}_{}_overlay.png'.format(sample_name, modal_name)
    output_path = os.path.join(output_container, output_image)
    plt.savefig(output_path, bbox_inches='tight')
    print('Overlay for {} on {} modality.'.format(sample_name, modal_name))


def generate_rows_for_visualization(path_container, list_names, label_container, prediction_container,
                                    output_container, modal_name):

    for name in list_names:
        output_for_case = os.path.join(output_container, name)
        if not os.path.exists(output_for_case):
            os.makedirs(output_for_case)
        visualize_sample(path_container, name, output_for_case)
        overlay_label_for_sample(path_container, name, label_container, output_for_case, modal_name)
        overlay_label_for_sample(path_container, name, prediction_container, output_for_case, modal_name, epoch=1)
        print('Visualized for {} sample.'.format(name))

path_container = 'data/train/slices'
label_container = 'data/train/labels'
sample_name = 'Brats18_CBICA_ALN_1_80'
output_container = 'output_overlay'
# visualize_sample(path_container, sample_name, output_container)
# overlay_label_for_sample(path_container, sample_name, label_container, output_container, 't2')

# For visualization of chosen sample in validation set
# from brats_visualization import generate_rows_for_visualization
#
# path_container = 'data/val/slices'
# list_names = [
#               'Brats18_CBICA_ASG_1_80','Brats18_CBICA_ASY_1_66',
#               'Brats18_TCIA01_150_1_53', 'Brats18_TCIA04_479_1_100',
#               'Brats18_TCIA06_332_1_47', 'Brats18_CBICA_ASG_1_123',
#               'Brats18_CBICA_ASY_1_83'
#               ]
#
# label_container = 'data/val/labels'
# prediction_container = 'saved/models/Brain_Tumor_Segmentation_2D/0524_042314/tracker'
# output_container = 'visualization_val'
# modal_name = 't2'
# 
# generate_rows_for_visualization(path_container, list_names, label_container,
#                                 prediction_container, output_container,
#                                 modal_name)


# For visualization of chosen sample in test set
# from brats_visualization import generate_rows_for_visualization
#
# path_container = 'data/test/slices'
# list_names = [
#                 'Brats18_TCIA09_312_1_96',
#                 'Brats18_TCIA10_629_1_85',
#                 'Brats18_TCIA13_624_1_51',
#                 'Brats18_TCIA13_653_1_110',
#                 'Brats18_TCIA10_449_1_95',
#                 'Brats18_TCIA10_152_1_58',
#                 'Brats18_TCIA10_408_1_65'
#               ]
#
# label_container = 'data/test/labels'
# prediction_container = '0522_134208/tracker'
# output_container = 'visualization_test'
# modal_name = 't2'
#
# generate_rows_for_visualization(path_container, list_names, label_container,
#                                 prediction_container, output_container,
#                                 modal_name)
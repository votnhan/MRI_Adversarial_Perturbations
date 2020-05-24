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
    output_image = '{}_{}_overlay.png'.format(sample_name, modal_name)
    output_path = os.path.join(output_container, output_image)
    plt.savefig(output_path, bbox_inches='tight')
    print('Overlay for {} on {} modality.'.format(sample_name, modal_name))


path_container = 'data/train/slices'
label_container = 'data/train/labels'
sample_name = 'Brats18_CBICA_ALN_1_80'
output_container = 'output_overlay'
# visualize_sample(path_container, sample_name, output_container)
# overlay_label_for_sample(path_container, sample_name, label_container, output_container, 't2')



import copy
from torchvision import transforms as std_transforms
from .data_loaders import AxialMRIDataLoader
from .joint_transforms import RandomSizeAndCrop, Resize, RandomHorizonFlip, Compose
from .image_transforms import ToTensor, Normalization, IntensityScale


def _create_transforms(config):
    joint_tf_cfg = config['transforms']['joint_transforms']
    image_tf_cfg = config['transforms']['image_transforms']

    # joint transforms
    random_resize_crop = RandomSizeAndCrop(joint_tf_cfg['crop_size'], range_scale=tuple(joint_tf_cfg['range_scale']))
    resize = Resize((joint_tf_cfg['crop_size'], joint_tf_cfg['crop_size']))
    flip = RandomHorizonFlip()
    joint_transforms_list = [random_resize_crop, resize, flip]

    # image transforms
    to_tf = ToTensor()
    # Turn off means, stds normalization
    # normalization = Normalization(image_tf_cfg['means'], image_tf_cfg['stds'])
    scale = IntensityScale(image_tf_cfg['range_scale'])
    image_transforms_list = [scale, to_tf]

    # target transforms
    target_transforms_list = [ToTensor()]

    # validation transforms
    val_transforms_list = copy.deepcopy(image_transforms_list)

    joint_transforms_cp = Compose(joint_transforms_list)
    image_transforms_cp = std_transforms.Compose(image_transforms_list)
    target_transforms_cp = std_transforms.Compose(target_transforms_list)
    val_transforms_cp = std_transforms.Compose(val_transforms_list)

    return joint_transforms_cp, image_transforms_cp, target_transforms_cp, val_transforms_cp

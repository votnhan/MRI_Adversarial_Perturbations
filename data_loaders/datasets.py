from torch.utils.data import Dataset
from functools import reduce


class VisionDataset(Dataset):
    def __init__(self, root_dir, joint_transforms, image_transforms, target_transforms):
        self.root_dir = root_dir
        self.joint_transforms = joint_transforms
        self.image_transforms = image_transforms
        self.target_transform = target_transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def info(self):
        head = 'Dataset: {}'.format(self.__class__.__name__)
        body = ['Number of samples: {}'.format(self.__len__())]
        if self.root_dir:
            body.append('Root data directory: {}'.format(self.root_dir))
        _ = reduce(lambda x, y: x.append(repr(y)), self.join_transforms, body)
        _ = reduce(lambda x, y: x.append(repr(y)), self.image_transforms, body)
        _ = reduce(lambda x, y: x.append(repr(y)), self.target_transform, body)
        lines = [head] + body
        return '\n'.join(lines)

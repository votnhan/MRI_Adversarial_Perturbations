import os
import numpy as np
import copy
from .datasets import VisionDataset


class AxialMRIDataset(VisionDataset):
    def __init__(self, root_dir='data', split='train', num_samples=-1, joint_transforms=None, image_transform=None,
                 target_transform=None):
        super().__init__(root_dir, joint_transforms, image_transform, target_transform)
        data_path = os.path.join(root_dir, split, 'slices')
        label_path = os.path.join(root_dir, split, 'labels')

        self.slice_paths = [os.path.join(data_path, x) for x in os.listdir(data_path)]
        self.label_paths = [os.path.join(label_path, x) for x in os.listdir(label_path)]

        if num_samples != -1:
            self.slice_paths = self.slice_paths[:num_samples]
            self.label_paths = self.label_paths[:num_samples]

    def __getitem__(self, index):
        img_path = self.slice_paths[index]
        lbl_path = self.label_paths[index]
        subject_name = os.path.basename(img_path)
        img = np.load(img_path)['arr_0']
        lbl = np.load(lbl_path)['arr_0']

        lbl_clone = copy.deepcopy(lbl)
        lbl_clone[lbl == 4] = 3
        lbl_clone = np.expand_dims(lbl_clone, axis=0)

        if self.joint_transforms:
            img, lbl_clone = self.joint_transforms(img, lbl_clone)

        if self.image_transforms:
            img = self.image_transforms(img)

        if self.target_transform:
            lbl_clone = self.target_transform(lbl_clone)

        return img, lbl_clone, subject_name

    def __len__(self):
        return len(self.slice_paths)


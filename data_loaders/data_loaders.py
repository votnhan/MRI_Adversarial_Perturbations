from torch.utils.data import DataLoader
from .axial_mri_dataset import AxialMRIDataset


class AxialMRIDataLoader(DataLoader):
    def __init__(self, root, batch_size, shuffle=True, num_workers=0, split='train', num_samples=-1,
                 joint_transforms=None, image_transforms=None, target_transforms=None):
        self.dataset = AxialMRIDataset(root, split, num_samples, joint_transforms, image_transforms, target_transforms)
        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

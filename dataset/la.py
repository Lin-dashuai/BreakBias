"""LA H03 dataset with role-aware AP/BE views."""
from dataset.transform import RandomCrop_3D, LabelGuidedCrop_3D, ToTensor_3D_BB
from rvp import build_shared_hard_reference, anatomy_preserving_altitude_jitter
from copy import deepcopy
import math
import os
from pathlib import Path
import random
import numpy as np
import h5py
from torch.utils.data import Dataset

class BreakBiasLADataset(Dataset):

    def __init__(self, name, root, mode, output_size=None, id_path=None, nsample=None, rvp_enabled=True, rvp_prob=0.7, rvp_altitude_jitter=7.5):
        self.name, self.root, self.mode = (name, root, mode)
        self.output_size = output_size
        self.rvp_enabled = rvp_enabled
        self.rvp_prob = rvp_prob
        self.rvp_altitude_jitter = rvp_altitude_jitter
        assert name == 'LA' and mode in ('train_l', 'train_u', 'val')
        if mode in ('train_l', 'train_u'):
            self.crop_transform = RandomCrop_3D(output_size) if mode == 'train_u' else LabelGuidedCrop_3D(output_size, foreground_prob=0.9)
        self.to_tensor_transform = ToTensor_3D_BB()
        self.ids = Path(id_path).read_text().splitlines()
        assert self.ids
        if mode == 'train_l' and nsample is not None:
            self.ids *= math.ceil(nsample / len(self.ids))
            random.shuffle(self.ids)
            self.ids = self.ids[:nsample]

    def __getitem__(self, item):
        case_id = self.ids[item]
        file_path = os.path.join(self.root, case_id, 'mri_norm2.h5')
        try:
            with h5py.File(file_path, 'r') as hf:
                original_image = hf['image'][:]
                label = np.zeros_like(original_image, dtype=np.uint8) if self.mode == 'train_u' else hf['label'][:]
        except Exception as e:
            raise
        if self.mode in ['train_l', 'train_u']:
            sample_to_crop = {'image_ap': deepcopy(original_image), 'image_be': deepcopy(original_image), 'label': label}
            cropped_sample = self.crop_transform(sample_to_crop)
            image_ap = cropped_sample['image_ap']
            image_be = cropped_sample['image_be']
            cropped_label = cropped_sample['label']
            image_weak = image_ap.copy()
            flip_code = random.randint(0, 7)
            if flip_code == 1:
                image_weak = np.flip(image_weak, 0).copy()
            elif flip_code == 2:
                image_weak = np.flip(image_weak, 1).copy()
            elif flip_code == 3:
                image_weak = np.flip(image_weak, 2).copy()
            elif flip_code == 4:
                image_weak = np.flip(image_weak, (0, 1)).copy()
            elif flip_code == 5:
                image_weak = np.flip(image_weak, (0, 2)).copy()
            elif flip_code == 6:
                image_weak = np.flip(image_weak, (1, 2)).copy()
            elif flip_code == 7:
                image_weak = np.flip(image_weak, (0, 1, 2)).copy()
            ap_max, ap_min = (np.max(image_ap), np.min(image_ap))
            if ap_max - ap_min < 1e-07:
                image_ap_norm = np.zeros_like(image_ap, dtype=np.float32)
            else:
                image_ap_norm = (image_ap - ap_min) / (ap_max - ap_min + 1e-07)
            be_max, be_min = (np.max(image_be), np.min(image_be))
            if be_max - be_min < 1e-07:
                image_be_norm = np.zeros_like(image_be, dtype=np.float32)
            else:
                image_be_norm = (image_be - be_min) / (be_max - be_min + 1e-07)
            rotate_angle = 0
            try:
                if self.rvp_enabled and random.random() < self.rvp_prob:
                    selected_tree = 'maxtree' if random.random() > 0.5 else 'mintree'
                    shared_reference = build_shared_hard_reference(image_ap_norm, selected_tree, rotate_angle=0, hard_strength='base')
                    image_ap_norm = anatomy_preserving_altitude_jitter(image_ap_norm, shared_reference, self.rvp_altitude_jitter)
                    image_be_norm = shared_reference
                    rotate_angle = 0
            except Exception as e:
                raise
            image_ap = image_ap_norm * (ap_max - ap_min) + ap_min
            image_be = image_be_norm * (be_max - be_min) + be_min
            final_sample = {'image_ap': image_ap, 'image_be': image_be, 'image_weak': image_weak, 'label': cropped_label, 'id': case_id, 'rotate_angle': rotate_angle}
            return self.to_tensor_transform(final_sample)
        image = original_image.astype(np.float32)
        label = label.astype(np.uint8)
        return {'image': image, 'label': label, 'id': case_id}

    def __len__(self):
        return len(self.ids)

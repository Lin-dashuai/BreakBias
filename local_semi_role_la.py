"""LA H03 shared-reference input pipeline; historical tensor keys are preserved."""
from dataset.transform import RandomCrop_3D, LabelGuidedCrop_3D, ToTensor_3D_BB
from tree_process import tree_process_3D_higra
from copy import deepcopy
import math
import os
from pathlib import Path
import random
import numpy as np
from scipy import ndimage
import h5py
from torch.utils.data import Dataset

def _clip01(image):
    return np.clip(image, 0.0, 1.0).astype(np.float32)

def _tree_hard_view(image, tree_type, area_threshold, topoprob, rotate_angle, hard_strength='base'):
    out = tree_process_3D_higra(image, tree_type, rotate_angle, hard_strength='base')
    if not isinstance(out, np.ndarray):
        return image
    return _clip01(out)

def _tree_change_mask(image, hard_view):
    diff = np.abs(hard_view - image)
    if float(diff.max()) < 1e-06:
        return np.zeros_like(image, dtype=bool)
    thr = max(0.01, float(np.percentile(diff, 85)))
    mask = diff > thr
    if mask.mean() < 0.002:
        mask = diff > max(0.003, float(np.percentile(diff, 70)))
    return mask

def _stable_altitude(image, hard_view, jitter=5.0):
    mask = _tree_change_mask(image, hard_view)
    if not mask.any():
        return image
    delta = random.choice([-1.0, 1.0]) * random.uniform(0.5 * jitter / 255.0, jitter / 255.0)
    out = image.copy()
    out[mask] = out[mask] + delta
    return _clip01(out)


class BreakBias_LA_Role(Dataset):
    def __init__(self, name, root, mode, output_size=None, id_path=None, nsample=None,
                 tree_mode=1, tree_areathrehold=100, tree_topoprob=0.5,
                 role_trigger_prob=0.7, role_altitude_jitter=7.5):
        self.name, self.root, self.mode = name, root, mode
        self.output_size = output_size
        self.tree_mode = tree_mode
        self.tree_areathrehold = tree_areathrehold
        self.tree_topoprob = tree_topoprob
        self.role_trigger_prob = role_trigger_prob
        self.role_altitude_jitter = role_altitude_jitter
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
            sample_to_crop = {'image_similar': deepcopy(original_image), 'image_dissimilar': deepcopy(original_image), 'label': label}
            cropped_sample = self.crop_transform(sample_to_crop)
            image_similar = cropped_sample['image_similar']
            image_dissimilar = cropped_sample['image_dissimilar']
            cropped_label = cropped_sample['label']
            image_weak = image_similar.copy()
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
            img_s_max, img_s_min = (np.max(image_similar), np.min(image_similar))
            if img_s_max - img_s_min < 1e-07:
                image_similar_norm = np.zeros_like(image_similar, dtype=np.float32)
            else:
                image_similar_norm = (image_similar - img_s_min) / (img_s_max - img_s_min + 1e-07)
            img_d_max, img_d_min = (np.max(image_dissimilar), np.min(image_dissimilar))
            if img_d_max - img_d_min < 1e-07:
                image_dissimilar_norm = np.zeros_like(image_dissimilar, dtype=np.float32)
            else:
                image_dissimilar_norm = (image_dissimilar - img_d_min) / (img_d_max - img_d_min + 1e-07)
            rotate_angle = 0
            try:
                if self.tree_mode == 1 and random.random() < self.role_trigger_prob:
                    selected_tree = 'maxtree' if random.random() > 0.5 else 'mintree'
                    shared_reference = _tree_hard_view(image_similar_norm, selected_tree, self.tree_areathrehold, self.tree_topoprob, rotate_angle=0, hard_strength='base')
                    image_similar_norm = _stable_altitude(image_similar_norm, shared_reference, self.role_altitude_jitter)
                    image_dissimilar_norm = shared_reference
                    rotate_angle = 0
            except Exception as e:
                raise
            image_similar = image_similar_norm * (img_s_max - img_s_min) + img_s_min
            image_dissimilar = image_dissimilar_norm * (img_d_max - img_d_min) + img_d_min
            final_sample = {'image_similar': image_similar, 'image_dissimilar': image_dissimilar, 'image_weak': image_weak, 'label': cropped_label, 'id': case_id, 'rotate_angle': rotate_angle}
            return self.to_tensor_transform(final_sample)
        image = original_image.astype(np.float32)
        label = label.astype(np.uint8)
        return {'image': image, 'label': label, 'id': case_id}

    def __len__(self):
        return len(self.ids)

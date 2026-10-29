import random
import math
import numpy as np
import torch
from scipy import ndimage

class RandomCrop_3D(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_ap, image_be, label = (sample['image_ap'], sample['image_be'], sample['label'])
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image_ap = np.pad(image_ap, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image_be = np.pad(image_be, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        w, h, d = image_ap.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_ap = image_ap[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_be = image_be[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image_ap': image_ap, 'image_be': image_be, 'label': label}

class LabelGuidedCrop_3D(object):
    """
    Crop the image in a sample, ensuring the crop is centered on a foreground voxel.
    If no foreground is available, it falls back to random cropping.

    Args:
    output_size (tuple or list): Desired output size.
    foreground_prob (float): Probability of cropping a patch containing a foreground class.
                             Default is 1.0 (always crop on foreground if possible).
    """

    def __init__(self, output_size, foreground_prob=1.0):
        assert isinstance(output_size, (list, tuple)), 'output_size must be a list or tuple'
        self.output_size = output_size
        self.foreground_prob = foreground_prob

    def __call__(self, sample):
        image_ap, image_be, label = (sample['image_ap'], sample['image_be'], sample['label'])
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image_ap = np.pad(image_ap, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image_be = np.pad(image_be, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        w, h, d = image_ap.shape
        perform_foreground_crop = random.random() < self.foreground_prob
        foreground_coords = np.argwhere(label > 0)
        if perform_foreground_crop and len(foreground_coords) > 0:
            center_coord = random.choice(foreground_coords)
            w1 = center_coord[0] - self.output_size[0] // 2
            h1 = center_coord[1] - self.output_size[1] // 2
            d1 = center_coord[2] - self.output_size[2] // 2
            w1 = max(0, w1)
            h1 = max(0, h1)
            d1 = max(0, d1)
            w1 = min(w1, w - self.output_size[0])
            h1 = min(h1, h - self.output_size[1])
            d1 = min(d1, d - self.output_size[2])
        else:
            w1 = np.random.randint(0, w - self.output_size[0] + 1)
            h1 = np.random.randint(0, h - self.output_size[1] + 1)
            d1 = np.random.randint(0, d - self.output_size[2] + 1)
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_ap = image_ap[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_be = image_be[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image_ap': image_ap, 'image_be': image_be, 'label': label}

class ToTensor_3D_BB(object):

    def __call__(self, sample):
        result = sample.copy()
        for key, value in sample.items():
            if 'image' in key:
                if not isinstance(value, torch.Tensor):
                    img = value.reshape(1, *value.shape).astype(np.float32)
                    result[key] = torch.from_numpy(img.copy())
            elif 'label' in key:
                if not isinstance(value, torch.Tensor):
                    result[key] = torch.from_numpy(value.copy()).long()
        return result

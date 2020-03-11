import torch
import numpy as np
import torch.nn as nn
import albumentations as albu
import cv2
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
from albumentations.augmentations import functional as F
import torch
import random

class ToTensor:
    def __call__(self, data):
        if isinstance(data, tuple):
            return tuple([self._to_tensor(image) for image in data])
        else:
            return self._to_tensor(data)

    def _to_tensor(self, data):
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))

class Normalize:
    def __init__(self, mean, std):
        self.mean = np.average(mean)
        self.std = np.average(std)

    def __call__(self, image):
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image


class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.

    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value

                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')


class RandomMorph(ImageOnlyTransform):

    def __init__(self, _min=2, _max=6, element_shape=cv2.MORPH_ELLIPSE, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self._min = _min
        self._max = _max
        self.element_shape = element_shape

    def apply(self, image, **params):
        arr = np.random.randint(self._min, self._max, 2)
        kernel = cv2.getStructuringElement(self.element_shape, tuple(arr))

        if random.random() > 0.5:
            # make it thinner
            image = cv2.erode(image, kernel, iterations=1)
        else:
            # make it thicker
            image = cv2.dilate(image, kernel, iterations=1)

        return image


def get_augs(gridmask=False, randommorph=True):
    augs = [albu.ShiftScaleRotate(p=0.2,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=1,
                          scale_limit=0.2,
                          rotate_limit=30)]
    if gridmask:
        augs.append(albu.OneOf([
                        GridMask(num_grid=3, rotate=15),
                        GridMask(num_grid=3),
                        ], p=0.5)) #Low probability
    if randommorph:
        augs.append(RandomMorph()) #Stolen from kaggle

    return albu.Compose(augs)

def avg(alpha=1):
    l, lc, s, sc = 0, 0, 0, 0
    for x in range(1000):
        r = np.random.beta(alpha, alpha)
        if r > 0.5:
            l+=r
            lc+=1
        else:
            s+=r
            sc+=1
    print(l/lc)
    print(s/sc)

def mixup_data(data, labels, alpha, device):
    indices = torch.randperm(data.size(0))
    lam = np.random.beta(alpha, alpha)
    data = data * lam + data[indices] * (1 - lam)
    labels = ([labels[0].to(device), labels[1].to(device), labels[2].to(device)],
                      [labels[0][indices].to(device),
                       labels[1][indices].to(device),
                       labels[2][indices].to(device)], lam)
    return data, labels

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(data, labels, alpha, device):
    indices = torch.randperm(data.size()[0])
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    labels = ([labels[0].to(device), labels[1].to(device), labels[2].to(device)],
                          [labels[0][indices].to(device),
                           labels[1][indices].to(device),
                           labels[2][indices].to(device)], lam)
    return data, labels

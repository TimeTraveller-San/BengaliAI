import torch
import numpy as np
import torch.nn as nn
import albumentations as albu
import cv2

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


def get_augs():
    return albu.Compose([
        albu.ShiftScaleRotate(p=0.8, border_mode=cv2.BORDER_CONSTANT, value =1),
    #     albu.OneOf([
    #         albu.ElasticTransform(p=0.1, alpha=1, sigma=10, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT,value =1),
    #         albu.GridDistortion(distort_limit =0.01 ,border_mode=cv2.BORDER_CONSTANT,value =1, p=0.1),
    #         albu.OpticalDistortion(p=0.1, distort_limit= 0.01, shift_limit=0.1, border_mode=cv2.BORDER_CONSTANT,value =1)
    #         ], p=0.3),
    #     albu.OneOf([
    # #         albu.GaussNoise(var_limit=0.5),
    #         albu.Blur(),
    #         albu.GaussianBlur(blur_limit=1)
    #         ], p=0.4),
    #     albu.RandomGamma(p=0.8)
    ])

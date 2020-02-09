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
        albu.ShiftScaleRotate(p=0.7,
                              border_mode=cv2.BORDER_CONSTANT,
                              value=1,
                              scale_limit=0.2,
                              rotate_limit=30),
        # albu.OneOf([
        #     albu.ElasticTransform(p=0.1, alpha=1, sigma=10, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT,value =1),
        #     albu.GridDistortion(distort_limit =0.01 ,border_mode=cv2.BORDER_CONSTANT,value =1, p=0.1),
        #     albu.OpticalDistortion(p=0.1, distort_limit= 0.01, shift_limit=0.1, border_mode=cv2.BORDER_CONSTANT,value =1)
        #     ], p=0.3),
        # albu.OneOf([
        #     albu.Blur(),
        #     albu.GaussianBlur(blur_limit=1)
        #     ], p=0.4),
    ])



def mixup_data(data, labels, alpha, device):
    indices = torch.randperm(data.size(0))
    lam = np.random.beta(alpha, alpha)
    data = data * lam + data[indices] * (1 - lam)
    labels = ([labels[0].to(device), labels[1].to(device), labels[2].to(device)],
                      [labels[0][indices].to(device),
                       labels[1][indices].to(device),
                       labels[2][indices].to(device)], lam)
    return data, labels


class Mixed_CrossEntropyLoss():
    def __init__(self):
        pass

    def __call__(self, preds, labels, val=True):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        if val:
            return criterion(preds, labels)
        l1, l2, lam = labels
        return lam * criterion(preds, l1) + (1 - lam) * criterion(preds, l2)


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

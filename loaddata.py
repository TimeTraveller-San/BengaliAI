import os
import gc
import cv2
import time
import torch
import sklearn
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as albu
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from augmentations import *

class BengaliAI(Dataset):
    def __init__(self, data, details=False, transform=None, imgsize=(128, 128)):
        self.images = data.iloc[:, 5:].values
        self.grapheme_roots = data['grapheme_root'].values
        self.vowel_diacritics = data['vowel_diacritic'].values
        self.consonant_diacritics = data['consonant_diacritic'].values
        self.imgsize = imgsize
        self.transform = transform
        if details:
            self.mean, self.std = details
        else:
            self.mean, self.std = 0.5, 0.5
        self.reqtransform = transforms.Compose([
            Normalize(self.mean, self.std),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img = self.images[idx].reshape(self.imgsize).astype(np.float)
        grapheme_root = self.grapheme_roots[idx]
        vowel_diacritic = self.vowel_diacritics[idx]
        consonant_diacritic = self.consonant_diacritics[idx]
        if self.transform is not None:
            img = self.transform(image=img)['image']
        img = self.reqtransform(img)
        label = (grapheme_root, vowel_diacritic, consonant_diacritic)
        return img, label

    def __len__(self):
        return len(self.images)


# if debug:
#     dataset = BengaliAI(train_df, transform=train_aug)
#     i = 0
#     LIMIT = 10
#
#     for img, (l1, l2, l2) in dataset:
#         plt.imshow(img.numpy().reshape(128, 128))
#         plt.show()
#         i += 1
#         if i > LIMIT:
#             break

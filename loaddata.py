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



def load_df(debug=True, random_state=42):
    # Load Feather Data
    df = 'data/train.csv'
    files = [f'data/train_128_feather/train_{i}.feather' for i in range(4)]
    df = pd.read_csv(df)
    if debug:
        data0 = pd.read_feather(files[0])
        data_full = data0
        del data0
        gc.collect()
        data_full = df.merge(data_full, on='image_id', how='inner')
    else:
        data0 = pd.read_feather(files[0])
        data1 = pd.read_feather(files[1])
        data2 = pd.read_feather(files[2])
        data3 = pd.read_feather(files[3])
        data_full = pd.concat([data0,data1,data2,data3], ignore_index=True)
        del data0, data1, data2, data3
        gc.collect()
        data_full = df.merge(data_full, on='image_id', how='inner')
    del df
    gc.collect()
    print(data_full.shape)
    train_df , valid_df = train_test_split(data_full,
                    test_size=0.20, random_state=random_state,
                    shuffle=True)
    del data_full
    gc.collect()
    return train_df, valid_df


if __name__ == "__main__":
    """Unit tests"""
    train_df, valid_df = load_df(True)
    dataset = BengaliAI(train_df)
    for img, label in dataset:
        print(label)
        print(img.shape)
        plt.imshow(img.numpy().reshape(128, 128))
        plt.show()
        break

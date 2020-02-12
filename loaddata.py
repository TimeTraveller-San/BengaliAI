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
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from augmentations import *
from utils import *



seed_everything()

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



def load_df(debug=True, random_state=42, root="data/"):
    # Load Feather Data
    # df = 'data/train.csv'
    df = os.path.join(root, 'train.csv')
    # files = [f'data/train_128_feather/train_{i}.feather' for i in range(4)]
    files = [os.path.join(root, 'train_128_feather', f'train_{i}.feather') for i in range(4)]
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

    msss = MultilabelStratifiedShuffleSplit(n_splits=1,
                                            test_size=0.2,
                                            random_state=random_state)
    y = data_full.iloc[:, 1:4]
    for train_index, test_index in msss.split(data_full, y):
        train_df, valid_df = data_full.iloc[train_index, :], data_full.iloc[test_index, :]
    # train_df , valid_df = train_test_split(data_full,
    #                 test_size=0.20, random_state=random_state,
    #                 shuffle=True)
    del data_full, y
    gc.collect()
    return train_df, valid_df


def load_toy_df(random_state=42, root="/home/timetraveller/Entertainment/BengaliAI_Data"):
    # Load Feather Data
    df = os.path.join(root, 'toy_data.csv')
    files = [os.path.join(root, 'train_128', f'train_{i}.feather') for i in range(4)]
    df = pd.read_csv(df)
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
    msss = MultilabelStratifiedShuffleSplit(n_splits=1,
                                            test_size=0.2,
                                            random_state=random_state)
    y = data_full.iloc[:, 1:4]
    for train_index, test_index in msss.split(data_full, y):
        train_df, valid_df = data_full.iloc[train_index, :], data_full.iloc[test_index, :]
    del data_full, y
    gc.collect()
    return train_df, valid_df


if __name__ == "__main__":
    """Unit tests"""
    # train_df, valid_df = load_df(True)
    train_df, valid_df = load_toy_df()
    dataset1 = BengaliAI(train_df)
    dataset2 = BengaliAI(train_df,
                             transform=get_augs(),
                        )
    ncol = 2
    nrow = 15
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2 * nrow), squeeze=True)
    for i, ((img1, _), (img2, _)) in enumerate(zip(dataset1, dataset2)):
        for j, img in enumerate([img1, img2]):
            print(i, j)
            ax = axes[i, j]
            ax.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(img.numpy().reshape(128, 128))
        if i >= nrow-1:
            break
    plt.show()



#
# def show_images(aug_dict, ncol=6):
#     nrow = len(aug_dict)
#
#     fig, axes = plt.subplots(nrow, ncol, figsize=(20, 2 * nrow), squeeze=False)
#
#     for i, (key, aug) in enumerate(aug_dict.items()):
#         for j in range(ncol):
#             ax = axes[i, j]
#             if j == 0:
#                 ax.text(0.5, 0.5, key, horizontalalignment='center', verticalalignment='center', fontsize=15)
#                 ax.get_xaxis().set_visible(False)
#                 ax.get_yaxis().set_visible(False)
#                 ax.axis('off')
#             else:
#                 image, label = train_dataset[j-1]
#                 if aug is not None:
#                     image = aug(image=image)['image']
#                 ax.imshow(image, cmap='Greys')
#                 ax.set_title(f'label: {label}')

import os
import gc
import cv2
import time
import torch
import sklearn
import torchvision
import numpy as np
import pandas as pd
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

from utils import *

import pretrainedmodels
from efficientnet_pytorch import EfficientNet

n_grapheme = 168
n_vowel = 11
n_consonant = 7
num_classes = [n_grapheme, n_vowel, n_consonant]

class ClassifierCNN(nn.Module):
    def __init__(self, model_name, num_classes=num_classes, pretrained='imagenet'):
        super(ClassifierCNN, self).__init__()

        self.inconv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        if model_name.split('-')[0] == 'efficientnet':
            self.effnet = True
            if pretrained:
                self.model = EfficientNet.from_pretrained(model_name)
            else:
                self.model = EfficientNet.from_name(model_name)
            in_features = 1280 #TODO: Write a lazy linear to find this, for now, I do it by getting an error
        else:
            self.effnet = False
            self.model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
            in_features = self.model.last_linear.in_features

        self.head_grapheme_root = nn.Linear(in_features, num_classes[0])
        self.head_vowel_diacritic = nn.Linear(in_features, num_classes[1])
        self.head_consonant_diacritic = nn.Linear(in_features, num_classes[2])

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x, logit=True):
        x  = self.inconv(x)
        if self.effnet:
            features = self.model.extract_features(x)
        else:
            features = self.model.features(x)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)

        logit_grapheme_root = self.head_grapheme_root(features)
        logit_vowel_diacritic = self.head_vowel_diacritic(features)
        logit_consonant_diacritic = self.head_consonant_diacritic(features)

        if logit:
            return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic
        else:
            grapheme_root = F.softmax(logit_grapheme_root, 1)
            vowel_diacritic = F.softmax(logit_vowel_diacritic, 1)
            consonant_diacritic = F.softmax(logit_consonant_diacritic, 1)
            return grapheme_root, vowel_diacritic, consonant_diacritic

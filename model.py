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
import SEResNeXt_mish
import SEResNeXt_vanilla
# from efficientnet_pytorch import EfficientNet
from effnet import EfficientNet
from mixup import *
from activations import Mish

n_grapheme = 168
n_vowel = 11
n_consonant = 7
num_classes = [n_grapheme, n_vowel, n_consonant]


seed_everything()

def residual_add(lhs, rhs):
    lhs_ch, rhs_ch = lhs.shape[1], rhs.shape[1]
    if lhs_ch < rhs_ch:
        out = lhs + rhs[:, :lhs_ch]
    elif lhs_ch > rhs_ch:
        out = torch.cat([lhs[:, :rhs_ch] + rhs, lhs[:, rhs_ch:]], dim=1)
    else:
        out = lhs + rhs
    return out

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

def mean_std(model_name):
    try:
        mean = pretrainedmodels.__dict__['pretrained_settings'][model_name]['imagenet']['mean']
        std = pretrainedmodels.__dict__['pretrained_settings'][model_name]['imagenet']['std']
    except:
        mean, std = IMAGE_RGB_MEAN, IMAGE_RGB_STD
    return (mean, std)

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p),
                    (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class RGB(nn.Module):
    def __init__(self, model_name):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1,3,1,1))
        self.register_buffer('std', torch.ones(1,3,1,1))
        mean, std = mean_std(model_name)
        self.mean.data = torch.FloatTensor(mean).view(self.mean.shape)
        self.std.data = torch.FloatTensor(std).view(self.std.shape)

    def forward(self, x):
        x = (x-self.mean)/self.std
        return x

class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, use_bn=True,
                    activation=F.relu, dropout_ratio=-1, residual=False):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h


def lin_head(indim, outdim, bias=True, use_bn=True, activation=F.relu,
                                        dropout_ratio=-1, residual=False):
    hdim = int((indim+outdim)/2)
    l1 = LinearBlock(indim, hdim, use_bn=use_bn, activation=activation,
                                    residual=residual)
    l2 = LinearBlock(hdim, outdim, use_bn=use_bn, activation=None,
                                    residual=residual)
    return nn.Sequential(l1, l2)


# class AdaptiveHead(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(AdaptiveHead, self).__init__()
#         self.pool = GeM()
#         self.l1 = nn.Linear(in_features, out_features)
#         # self.l1 = nn.Linear(in_features, in_features//2)
#         # self.bn = nn.BatchNorm1d(in_features//2)
#         # self.mish = Mish()
#         # self.l2 = nn.Linear(in_features//2, out_features)
#
#     def forward(self, x):
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.l1(x)
#         # x = self.bn(x)
#         # x = self.mish(x)
#         # x = self.l2(x)
#         return x

class AdaptiveHead(nn.Module):
    """WORKS THE BEST TILL NOW - 0.714"""
    def __init__(self, in_features, out_features):
        super(AdaptiveHead, self).__init__()
        self.pool = GeM()
        self.fc1 = nn.Conv2d(in_features, in_features//2, 2)
        self.bn = nn.BatchNorm2d(in_features//2)
        self.mish = Mish()
        self.l1 = nn.Linear(in_features//2, out_features)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.mish(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)

        return x

# class AdaptiveHead(nn.Module):
#       """BADH DOESNT WORK"""
#     def __init__(self, in_features, out_features, p=0.3):
#         super(AdaptiveHead, self).__init__()
#         self.mish = Mish()
#         self.fc1 = nn.Conv2d(in_features, in_features//2, 4)
#         self.bn = nn.BatchNorm2d(in_features//2)
#         self.pool = GeM()
#         self.dropout = nn.Dropout2d(p)
#         self.l1 = nn.Linear(in_features//2, out_features)
#
#
#     def forward(self, x):
#         x = self.mish(x)
#         x = self.fc1(x)
#         x = self.bn(x)
#         x = self.pool(x)
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         x = self.l1(x)
#         return x

class AdaptiveHead_Heavy(nn.Module):
    def __init__(self, in_features, out_features, factor):
        super(AdaptiveHead_Heavy, self).__init__()
        self.fc1 = nn.Conv2d(in_features, in_features//factor, 4)
        self.bn = nn.BatchNorm2d(in_features//factor)
        self.mish = Mish()
        self.pool = GeM()
        self.l1 = nn.Linear(in_features//factor, out_features)


    def forward(self, x):
        x = self.fc1(x) #bottleneck
        x = self.bn(x)
        x = self.mish(x) #sigmoid instead?
        x = self.pool(x) #4x4->1x1
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        return x



class ClassifierCNN(nn.Module):
    def __init__(self, model_name, num_classes=num_classes,
            rgb=False, pretrained=None, activation=None, heavy_head=False):
        super(ClassifierCNN, self).__init__()

        if rgb:
            self.first = RGB(model_name)
        else:
            self.first = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        if not pretrained:
            pretrained = None
        else:
            print(f"Loading pretrained weights for {model_name}")
            pretrained = 'imagenet'
        if activation == 'mish':
            print("Using mish")
            if pretrained is not None:
                print("PRETRAINTED")
                self.model = SEResNeXt_vanilla.se_resnext50_32x4d(pretrained=pretrained) #Only one model supported for now
            else:
                print("NOT PRETRIANED")
                self.model = SEResNeXt_mish.se_resnext50_32x4d(pretrained=pretrained) #Only one model supported for now
        else:
            self.model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        self.size = (224, 224) # __dict__ for se_resnext50_32x4d

        # in_features = self.model.last_linear.in_features
        in_features = 2048

        # self.head_grapheme_root = lin_head(in_features, num_classes[0])
        # self.head_vowel_diacritic = lin_head(in_features, num_classes[1])
        # self.head_consonant_diacritic = lin_head(in_features, num_classes[2])

        # self.head_grapheme_root = nn.Linear(in_features, num_classes[0])
        # self.head_vowel_diacritic = nn.Linear(in_features, num_classes[1])
        # self.head_consonant_diacritic = nn.Linear(in_features, num_classes[2])

        if heavy_head:
            self.head_grapheme_root = AdaptiveHead_Heavy(in_features, num_classes[0], factor=2)
            self.head_vowel_diacritic = AdaptiveHead_Heavy(in_features, num_classes[1], factor=4)
            self.head_consonant_diacritic = AdaptiveHead_Heavy(in_features, num_classes[2], factor=4)
        else:
            self.head_grapheme_root = AdaptiveHead(in_features, num_classes[0])
            self.head_vowel_diacritic = AdaptiveHead(in_features, num_classes[1])
            self.head_consonant_diacritic = AdaptiveHead(in_features, num_classes[2])

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x = F.interpolate(x, size=self.size, mode='bilinear')
        x = self.first(x)
        x = self.model.features(x)
        logit_grapheme_root = self.head_grapheme_root(x)
        logit_vowel_diacritic = self.head_vowel_diacritic(x)
        logit_consonant_diacritic = self.head_consonant_diacritic(x)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic

class ClassifierCNN_effnet(nn.Module):
    def __init__(self, model_name, num_classes=num_classes, rgb=False, pretrained=False, activation=None):
        super(ClassifierCNN_effnet, self).__init__()

        if rgb:
            self.first = RGB(model_name)
        else:
            self.first = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name, activation=activation)
        else:
            self.model = EfficientNet.from_name(model_name, activation=activation)
        self.pool = GeM()
        # in_features = 1280 #TODO: Write a lazy linear to find this, for now, I do it by getting an error
        in_features = 1408 #TODO: Write a lazy linear to find this, for now, I do it by getting an error
        # in_features = 1536 #TODO: Write a lazy linear to find this, for now, I do it by getting an error

        # self.head_grapheme_root = lin_head(in_features, num_classes[0])
        # self.head_vowel_diacritic = lin_head(in_features, num_classes[1])
        # self.head_consonant_diacritic = lin_head(in_features, num_classes[2])

        self.head_grapheme_root = AdaptiveHead_Heavy(in_features, num_classes[0], factor=2)
        self.head_vowel_diacritic = AdaptiveHead_Heavy(in_features, num_classes[1], factor=2)
        self.head_consonant_diacritic = AdaptiveHead_Heavy(in_features, num_classes[2], factor=2)



    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.first(x)
        x = self.model.extract_features(x)

        logit_grapheme_root = self.head_grapheme_root(x)
        logit_vowel_diacritic = self.head_vowel_diacritic(x)
        logit_consonant_diacritic = self.head_consonant_diacritic(x)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic





if __name__ == "__main__":
    """Unit tests"""
    #1. Check forward pass
    model_name = 'se_resnext50_32x4d'
    # model_name = 'efficientnet-b0'
    pretrained = False
    activation = 'mish'
    mixup = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ClassifierCNN_effnet(model_name, pretrained=pretrained, activation=activation).to(device)
    model = ClassifierCNN(model_name, pretrained=pretrained, activation=activation).to(device)
    x = torch.zeros((8, 1, 128, 128))
    with torch.no_grad():
        output1, output2, output3 = model(x.cuda())
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)
    print("Forward pass successful.")

    #2. Does the model converge?
    from loaddata import BengaliAI, load_df
    from tqdm import tqdm
    df, _ = load_df(True)
    dataset = BengaliAI(df[:10000])
    dataloader = DataLoader(dataset, batch_size=32)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    if mixup:
        print("Using mixup")
        criterion = Mixup_CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    losses = []
    epochs = 5 #Increase this if you have good resources/a lot of time
    for epoch in tqdm(range(epochs)):
        for i, (img, label) in enumerate(tqdm(dataloader)):
            model.train()
            if mixup:
                alpha = 1
                img, labels = mixup_data(img, label, alpha, device)
                labels, shuffled_labels, lam = labels
            img = img.to(device)
            optimizer.zero_grad()
            out = model(img)
            if mixup:
                loss = 0.5*criterion(out[0], (label[0].to(device), shuffled_labels[0], lam)) +\
                       0.25*criterion(out[1], (label[1].to(device), shuffled_labels[1], lam)) +\
                       0.25*criterion(out[2], (label[2].to(device), shuffled_labels[2], lam))
            else:
                loss = 0.5*criterion(out[0], label[0].to(device)) +\
                       0.25*criterion(out[1], label[1].to(device)) +\
                       0.25*criterion(out[2], label[2].to(device))
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
    plt.plot(losses)
    plt.show()
    print("Did the model converge?")
    result = input()
    if result in "yes y 1 hai haan yep yup fuckYEAH yeah ye".split():
        print("yay! Good luck for training.")
    else:
        print("Don't give up.")

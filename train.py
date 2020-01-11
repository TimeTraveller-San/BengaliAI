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
from tqdm import tqdm_notebook as tqdm2
import matplotlib.pyplot as plt

import albumentations as albu
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils import *
import augmentations
from model import *
from loaddata import *

import argparse

seed_everything()

def train(n_epochs=5, name='test', pretrained=False):
    train_df , valid_df = load_df(False)
    train_aug = augmentations.get_augs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    # model_name = 'se_resnext101_32x4d'
    model_name = 'efficientnet-b0'
    model = ClassifierCNN(model_name, pretrained=pretrained).to(device)
    lr = 1e-3

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr
    )
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.3)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs=n_epochs, steps_per_epoch=3139, pct_start=0.0,
    #                                    anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0)

    criterion = nn.CrossEntropyLoss()

    train_dataset = BengaliAI(train_df, transform=train_aug, details=mean_std(model_name))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    val_dataset = BengaliAI(valid_df, details=mean_std(model_name))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # model.freeze()
    ws = [0.5, 0.25, 0.25]
    history = pd.DataFrame()

    if n_epochs:
        for epoch in tqdm2(range(n_epochs)):

            running_loss = 0
            running_loss0 = 0
            running_loss1 = 0
            running_loss2 = 0

            running_acc0 = 0.0
            running_acc1 = 0.0
            running_acc2 = 0.0

            running_recall = 0.0
            running_recall0 = 0.0
            running_recall1 = 0.0
            running_recall2 = 0.0

            recall = 0

            bar = tqdm(train_loader)
            for i, (img, label) in enumerate(bar):
                img = img.to(device)
                optimizer.zero_grad()
                out = model(img)
                label[0] = label[0].to(device)
                label[1] = label[1].to(device)
                label[2] = label[2].to(device)
                loss0 = criterion(out[0], label[0])
                loss1 = criterion(out[1], label[1])
                loss2 = criterion(out[2], label[2])
                loss = ws[0]*loss0 + ws[1]*loss1 + ws[2]*loss2
    #             loss.backward()
    #             optimizer.step()
                bar.set_description(f"Recall: {recall:.3f}")
                with torch.no_grad():

                    running_loss += loss.item()/len(train_loader)
                    running_loss0 += loss0.item()/len(train_loader)
                    running_loss1 += loss1.item()/len(train_loader)
                    running_loss2 += loss2.item()/len(train_loader)

                    recall, recall_grapheme, recall_vowel, recall_consonant = macro_recall_multi(out, label)

                    running_recall += recall/len(train_loader)
                    running_recall0 += recall_grapheme/len(train_loader)
                    running_recall1 += recall_vowel/len(train_loader)
                    running_recall2 += recall_consonant/len(train_loader)

                    running_acc0 += (out[0].argmax(1)==label[0]).float().mean()/len(train_loader)
                    running_acc1 += (out[1].argmax(1)==label[1]).float().mean()/len(train_loader)
                    running_acc2 += (out[2].argmax(1)==label[2]).float().mean()/len(train_loader)

                loss.backward()
                optimizer.step()
    #             scheduler.step()

            print(f"Epoch: [{epoch+1}/{n_epochs}] Training...")
            print(f"Recall: {running_recall:.3f} | [{running_recall0:.3f} | {running_recall1:.3f} | {running_recall2:.3f}]")
            print(f"Acc:  [{100*running_acc0:.3f}% | {100*running_acc1:.3f}% | {100*running_acc2:.3f}%]")
            print(f"Loss: {running_loss:.3f} | [{running_loss0:.3f} | {running_loss1:.3f} | {running_loss2:.3f}]")

            history.loc[epoch, 'train_loss'] = running_loss
            history.loc[epoch, 'train_recall'] = running_recall
            history.loc[epoch, 'train_recall_grapheme'] = running_recall0
            history.loc[epoch, 'train_recall_vowel'] = running_recall1
            history.loc[epoch, 'train_recall_consonant'] = running_recall2
            history.loc[epoch, 'train_acc_grapheme'] = running_acc0.cpu().numpy()
            history.loc[epoch, 'train_acc_vowel'] = running_acc1.cpu().numpy()
            history.loc[epoch, 'train_acc_consonant'] = running_acc2.cpu().numpy()

            with torch.no_grad():
                running_loss = 0
                running_loss0 = 0
                running_loss1 = 0
                running_loss2 = 0

                running_acc0 = 0.0
                running_acc1 = 0.0
                running_acc2 = 0.0

                running_recall = 0.0
                running_recall0 = 0.0
                running_recall1 = 0.0
                running_recall2 = 0.0


                for i, (img, label) in enumerate(val_loader):
                    img = img.to(device)
                    out = model(img)
                    label[0] = label[0].to(device)
                    label[1] = label[1].to(device)
                    label[2] = label[2].to(device)
                    recall, recall_grapheme, recall_vowel, recall_consonant = macro_recall_multi(out, label)
                    running_recall += recall/len(val_loader)
                    running_recall0 += recall_grapheme/len(val_loader)
                    running_recall1 += recall_vowel/len(val_loader)
                    running_recall2 += recall_consonant/len(val_loader)
                    running_acc0 += (out[0].argmax(1)==label[0]).float().mean()/len(val_loader)
                    running_acc1 += (out[1].argmax(1)==label[1]).float().mean()/len(val_loader)
                    running_acc2 += (out[2].argmax(1)==label[2]).float().mean()/len(val_loader)
                print(f"\nEpoch: [{epoch+1}/{n_epochs}] Validating...")
                print(f"Recall: {running_recall:.3f} | [{running_recall0:.3f} | {running_recall1:.3f} | {running_recall2:.3f}]")
                print(f"Acc:  [{100*running_acc0:.3f}% | {100*running_acc1:.3f}% | {100*running_acc2:.3f}%]")

                history.loc[epoch, 'val_recall'] = running_recall
                history.loc[epoch, 'val_recall_grapheme'] = running_recall0
                history.loc[epoch, 'val_recall_vowel'] = running_recall1
                history.loc[epoch, 'val_recall_consonant'] = running_recall2
                history.loc[epoch, 'val_acc_grapheme'] = running_acc0.cpu().numpy()
                history.loc[epoch, 'val_acc_vowel'] = running_acc1.cpu().numpy()
                history.loc[epoch, 'val_acc_consonant'] = running_acc2.cpu().numpy()

    history.to_csv(f'logs/{name}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", "-p", default=False,
                        help="use pretrained weights of not")
    parser.add_argument("--epochs", "-e", default=50,
                        help="number of epochs")
    parser.add_argument("--name", "-n", default="effnet-b0",
                            help="name of output csv")
    args = parser.parse_args()


    train(int(args.epochs), args.name, bool(args.pretrained))

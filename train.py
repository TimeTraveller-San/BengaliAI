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
from tqdm import tqdm_notebook as tqdm2

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
import logging #TODO

seed_everything()
check_dirs()


def train(n_epochs=5, name='test', pretrained=False, debug=False,
        continue_train=False, model_name='efficientnet-b0', run_name=False):

    if not run_name: run_name = model_name
    SAVE_DIR = f'logs/models/{run_name}'
    make_dir(SAVE_DIR)


    train_df , valid_df = load_df(debug)
    if debug:
        LIMIT = 500
        train_df = train_df[:LIMIT]
        valid_df = valid_df[:LIMIT]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    model_name = 'efficientnet-b0'
    # model_name = 'se_resnext101_32x4d'

    if model_name.split('-')[0] == 'efficientnet':
        model = ClassifierCNN_effnet(model_name, pretrained=pretrained).to(device)
    else:
        model = ClassifierCNN(model_name, pretrained=pretrained).to(device)
    lr = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if continue_train:
        try:
            if os.path.exists(str(continue_train)):
                path = continue_train
            else:
                path = os.path.join(SAVE_DIR, 'best.pth')
            print(path)
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded model from: {path}")
        except:
            continue_train = False
            print("Can't continue training. Starting again.")

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.3)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs=n_epochs, steps_per_epoch=3139, pct_start=0.0,
    #                                    anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0)
    criterion = nn.CrossEntropyLoss()
    train_aug = augmentations.get_augs()
    train_dataset = BengaliAI(train_df,
                             transform=train_aug,
                             details=mean_std(model_name)
                        )
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=4,
                              shuffle=False
                        )
    val_dataset = BengaliAI(valid_df,
                            details=mean_std(model_name)
                        )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            shuffle=False
                        )
    assert(len(train_dataset)>=batch_size)
    # model.freeze()
    ws = [0.5, 0.25, 0.25]
    history = pd.DataFrame()
    current, best = 0., -1.
    epoch = 0
    if continue_train:
        print(f"\n\nWILL CONTINUE FROM EPOCH: {start_epoch}\n\n")
        n_epochs += start_epoch
        epoch = start_epoch
    pbar = tqdm(total=n_epochs)
    pbar.update(epoch)
    # for epoch in pbar:
    while epoch < n_epochs:
        pbar.update(1)
        for phase in ['train', 'valid', 'save']:
            if phase == 'save':
                min_save_epoch = 3
                save_freq = 5
                if epoch < min_save_epoch:
                    continue
                if current > best:
                    best = current
                    save_model(os.path.join(SAVE_DIR, 'best.pth'),
                                    epoch, model, optimizer, True)
                elif (epoch+1) % save_freq == 0:
                    save_model(os.path.join(SAVE_DIR, f'{run_name}_{epoch+1}.pth'),
                                    epoch, model, optimizer, True)

            if phase == 'train':
                model.train()
                loader = train_loader

            if phase == 'valid':
                model.eval()
                loader = val_loader

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

            bar = tqdm(loader)
            for i, (img, label) in enumerate(bar):
                img = img.to(device)
                if phase == 'train':
                    optimizer.zero_grad()
                out = model(img)
                label[0] = label[0].to(device)
                label[1] = label[1].to(device)
                label[2] = label[2].to(device)
                loss0 = criterion(out[0], label[0])
                loss1 = criterion(out[1], label[1])
                loss2 = criterion(out[2], label[2])
                loss = ws[0]*loss0 + ws[1]*loss1 + ws[2]*loss2
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                bar.set_description(f"Recall: {recall:.3f}")
                # Evaluation
                with torch.no_grad():
                    running_loss += loss.item()/len(loader)
                    running_loss0 += loss0.item()/len(loader)
                    running_loss1 += loss1.item()/len(loader)
                    running_loss2 += loss2.item()/len(loader)
                    recall, recall_grapheme, recall_vowel, recall_consonant = macro_recall_multi(out, label)
                    running_recall += recall/len(loader)
                    running_recall0 += recall_grapheme/len(loader)
                    running_recall1 += recall_vowel/len(loader)
                    running_recall2 += recall_consonant/len(loader)
                    running_acc0 += (out[0].argmax(1)==label[0]).float().mean()/len(loader)
                    running_acc1 += (out[1].argmax(1)==label[1]).float().mean()/len(loader)
                    running_acc2 += (out[2].argmax(1)==label[2]).float().mean()/len(loader)
            print(f"Epoch: [{epoch+1}/{n_epochs}] {phase}...")
            print(f"Recall: {running_recall:.3f} | [{running_recall0:.3f} | {running_recall1:.3f} | {running_recall2:.3f}]")
            print(f"Acc:  [{100*running_acc0:.3f}% | {100*running_acc1:.3f}% | {100*running_acc2:.3f}%]")
            print(f"Loss: {running_loss:.3f} | [{running_loss0:.3f} | {running_loss1:.3f} | {running_loss2:.3f}]")
            if phase == 'valid':
                current = recall
            history.loc[epoch, f'{phase}_loss'] = running_loss
            history.loc[epoch, f'{phase}_recall'] = running_recall
            history.loc[epoch, f'{phase}_recall_grapheme'] = running_recall0
            history.loc[epoch, f'{phase}_recall_vowel'] = running_recall1
            history.loc[epoch, f'{phase}_recall_consonant'] = running_recall2
            history.loc[epoch, f'{phase}_acc_grapheme'] = running_acc0.cpu().numpy()
            history.loc[epoch, f'{phase}_acc_vowel'] = running_acc1.cpu().numpy()
            history.loc[epoch, f'{phase}_acc_consonant'] = running_acc2.cpu().numpy()
        epoch += 1
        # history.to_csv(f'logs/{name}_{epoch}.csv')
    # history.to_csv(f'logs/{name}_{FINAL}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", "-p", default=False,
                        help="use pretrained weights of not")
    parser.add_argument("--epochs", "-e", default=50,
                        help="number of epochs")
    parser.add_argument("--name", "-n", default="effnet-b0",
                            help="name of output csv")
    args = parser.parse_args()

    debug = True
    model_name = 'efficientnet-b0'
    run_name = 'test_phase1'
    train(int(args.epochs), args.name, True, debug=debug,
            continue_train=True, model_name=model_name, run_name=run_name)

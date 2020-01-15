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

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils import *
from model import *
from mixup import *
import augmentations
from loaddata import *

import logging
import argparse



seed_everything()
check_dirs()




def train(n_epochs=5, pretrained=False, debug=False, rgb=False,
        continue_train=False, model_name='efficientnet-b0', run_name=False,
        weights=[2, 1, 1], activation=None, mixup=False, alpha=1,
        min_save_epoch=3, save_freq=3):

    if not run_name: run_name = model_name
    SAVE_DIR = f'logs/models/{run_name}'
    make_dir(SAVE_DIR)
    logfile = os.path.join(SAVE_DIR, 'logs.txt')
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=logfile,
                        level=logging.DEBUG,
                        filemode='a'
                        )
    logging.info(f"\n\n---------------- [LOGS for {run_name}] ----------------")


    train_df , valid_df = load_df(debug)
    if debug:
        LIMIT = 500
        train_df = train_df[:LIMIT]
        valid_df = valid_df[:LIMIT]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    if model_name.split('-')[0] == 'efficientnet':
        model = ClassifierCNN_effnet(model_name,
                                    pretrained=pretrained,
                                    rgb=rgb,
                                    activation=activation).to(device)
    else:
        model = ClassifierCNN(model_name,
                              pretrained=pretrained,
                              rgb=rgb,
                              activation=activation).to(device)

    lr = 3e-4 # Andrej must be proud of me
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer, mode='min', factor=0.7,
                                        patience=3, min_lr=1e-10, verbose=True)

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
            scheduler = checkpoint['scheduler']
            start_epoch = checkpoint['epoch']
            print(f"Loaded model from: {path}")
            logging.info(f"Loaded model from: {path}")
        except:
            continue_train = False
            print("Can't continue training. Starting again.")
            logging.info("Can't continue training. Starting again.")

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.3)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs=n_epochs, steps_per_epoch=3139, pct_start=0.0,
    #                                    anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0)

    if mixup:
        criterion = Mixup_CrossEntropyLoss()
    else:
        # criterion = Mixup_CrossEntropyLoss()
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
    ws = get_weights(weights)
    history = pd.DataFrame()
    current, best = 0., -1.
    epoch = 0

    logging.info(f"Project path: {SAVE_DIR}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Model class: {type(model)}")
    logging.info(f"Debug: {debug}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"LR: {lr}")
    logging.info(f"Optimizer: {type(optimizer)}")
    logging.info(f"Weights: [{ws[0]} | {ws[1]} | {ws[2]}]")
    logging.info(f"Activation: {activation}")
    logging.info(f"Mixup: {mixup}")
    logging.info(f"Train dataset: {train_dataset}")
    logging.info(f"Validation dataset: {val_dataset}")
    logging.info(f"Continue: {continue_train}")
    logging.info(f"Model: {model}")
    logging.info("------------------------------------------------------------")

    logging.info("Starting training...")
    if continue_train:
        logging.info(f"WILL CONTINUE FROM EPOCH: {start_epoch}\n\n")
        n_epochs += start_epoch
        epoch = start_epoch
    pbar = tqdm(total=n_epochs)
    pbar.update(epoch)
    while epoch < n_epochs:
        pbar.update(1)
        for phase in ['train', 'valid', 'save']:
        # for phase in ['valid', 'train', 'save']:
            if phase == 'save':

                if epoch < min_save_epoch:
                    continue
                if current > best:
                    best = current
                    save_model(os.path.join(SAVE_DIR, 'best.pth'),
                                    epoch, model, optimizer, scheduler, True)
                elif (epoch+1) % save_freq == 0:
                    save_model(os.path.join(SAVE_DIR, f'{run_name}_{epoch+1}.pth'),
                                    epoch, model, optimizer, scheduler, True)
                continue

            if phase == 'train':
                model.train()
                logging.info("----------------------------------------------------------\n")
                loaders = [train_loader]

            if phase == 'valid':
                model.eval()
                if mixup:
                    logging.info("++++++++++++++ VALIDATING ON BOTH, IGNORE ABOVE TRAIN METRICS ++++++++++++++")
                    logging.info("++++++++++++++ FIRST IS TRAIN, THEN IS VAL ++++++++++++++")
                    loaders = [train_loader, val_loader] #For mixup, train_loader while training doesn't have the actual train data so validation needs to validate both. (to see if I am overfitting)
                else:
                    loaders = [val_loader]

            running_loss = 0.
            running_loss0 = 0.
            running_loss1 = 0.
            running_loss2 = 0.

            running_acc0 = 0.
            running_acc1 = 0.
            running_acc2 = 0.

            running_recall = 0.
            running_recall0 = 0.
            running_recall1 = 0.
            running_recall2 = 0.

            recall = 0.

            for loader in loaders:
                bar = tqdm(loader)
                for i, (img, label) in enumerate(bar):
                    with torch.set_grad_enabled(phase == 'train'):
                        img = img.to(device)
                        if phase == 'train':
                            optimizer.zero_grad()
                            if mixup:
                                img, labels = mixup_data(img, label, alpha, device)
                                labels, shuffled_labels, lam = labels
                        out = model(img)
                        label[0] = label[0].to(device)
                        label[1] = label[1].to(device)
                        label[2] = label[2].to(device)

                        if mixup and phase == 'train':
                            loss0 = criterion(out[0], (label[0], shuffled_labels[0], lam), False)
                            loss1 = criterion(out[1], (label[1], shuffled_labels[1], lam), False)
                            loss2 = criterion(out[2], (label[2], shuffled_labels[2], lam), False)
                        else:
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

                logging.info(f"Epoch: [{epoch+1}/{n_epochs}] {phase}...")
                logging.info(f">> Recall: {running_recall:.3f} | [{running_recall0:.3f} | {running_recall1:.3f} | {running_recall2:.3f}] <<")
                logging.info(f"Acc:  [{100*running_acc0:.3f}% | {100*running_acc1:.3f}% | {100*running_acc2:.3f}%]")
                logging.info(f"Loss: {running_loss:.3f} | [{running_loss0:.3f} | {running_loss1:.3f} | {running_loss2:.3f}]\n")


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

        history.to_csv(os.path.join(SAVE_DIR, f"{run_name}_{epoch}.csv"))
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", default=10,
                        help="number of epochs")
    parser.add_argument("--pretrained", "-p", default=False,
                        help="use pretrained weights of not")
    parser.add_argument("--debug", "-d", default=False,
                        help="if debug, run small model")
    parser.add_argument("--continue_train", "-c", default=False,
                        help="continue training or not")
    parser.add_argument("--model_name", "-mn", default="efficientnet-b0",
                            help="name of the model")
    parser.add_argument("--run_name", "-rn", default=False,
                            help="name of run")
    parser.add_argument("--rgb", "-rbg", default=False,
                            help="rgb or not?")
    parser.add_argument("--w1", "-w1", default=2,
                            help="weight for grapheme (ratio)")
    parser.add_argument("--w2", "-w2", default=1,
                            help="weight for vowel (ratio)")
    parser.add_argument("--w3", "-w3", default=1,
                            help="weight for consonant (ratio)")
    parser.add_argument("--activation", "-a", default=None,
                            help="None is default, mish is mish")
    parser.add_argument("--mixup", "-mx", default=False,
                            help="mixup augmentations, only on input for now")
    parser.add_argument("--alpha", "-alpha", default=1,
                            help="alpha for mixup")
    parser.add_argument("--min_save_epoch", "-mse", default=3,
                            help="minimum epoch to start saving models")
    parser.add_argument("--save_freq", "-sf", default=3,
                            help="frequency of saving epochs")



    args = parser.parse_args()

    # debug = False
    # pretrained = False
    # # model_name = 'efficientnet-b0'
    # model_name = 'se_resnext50_32x4d'
    # run_name = 'senet50'
    weights = [int(args.w1), int(args.w2), int(args.w3)]

    if args.debug:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++ DEBUG MODE +++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    train(int(args.epochs),
        args.pretrained,
        args.debug,
        args.rgb,
        args.continue_train,
        args.model_name,
        args.run_name,
        weights,
        args.activation,
        args.mixup,
        int(args.alpha),
        args.min_save_epoch,
        args.save_freq,
        )

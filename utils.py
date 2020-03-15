import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn
import time
import os
import pickle
import pretrainedmodels
from sklearn.model_selection import train_test_split
import sklearn
import random
import gc
import torch


def mean_std(model_name):
    try:
        mean = pretrainedmodels.__dict__['pretrained_settings'][model_name]['imagenet']['mean']
        std = pretrainedmodels.__dict__['pretrained_settings'][model_name]['imagenet']['std']
    except:
        mean, std = 0.5, 0.5
    return (mean, std)

# def seed_everything(seed=42):
def seed_everything(seed=2890):
    """
    42 is the answer to everything.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def macro_recall_multi(preds, labels):
    pred_graphemes, pred_vowels, pred_consonants = preds
    true_graphemes, true_vowels, true_consonants = labels
    n_grapheme = 168
    n_vowel = 11
    n_consonant = 7
    pred_label_graphemes = torch.argmax(pred_graphemes, dim=1).cpu().numpy()
    true_label_graphemes = true_graphemes.cpu().numpy()
    pred_label_vowels = torch.argmax(pred_vowels, dim=1).cpu().numpy()
    true_label_vowels = true_vowels.cpu().numpy()
    pred_label_consonants = torch.argmax(pred_consonants, dim=1).cpu().numpy()
    true_label_consonants = true_consonants.cpu().numpy()

    recall_grapheme = sklearn.metrics.recall_score(true_label_graphemes, pred_label_graphemes, average='macro')
    recall_vowel = sklearn.metrics.recall_score(true_label_vowels, pred_label_vowels, average='macro')
    recall_consonant = sklearn.metrics.recall_score(true_label_consonants, pred_label_consonants, average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score, recall_grapheme, recall_vowel, recall_consonant


def calc_macro_recall(solution, submission):
    # solution df, submission df
    scores = []
    for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
        y_true_subset = solution[solution[component] == component]['target'].values
        y_pred_subset = submission[submission[component] == component]['target'].values
        scores.append(sklearn.metrics.recall_score(
            y_true_subset, y_pred_subset, average='macro'))
    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_dirs():
    directories = ["logs/", "logs/models/"]
    for dir in directories:
        make_dir(dir)


def save_model(PATH, epoch, model, optimizer, scheduler, vocal=False):
    try:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler,
                }, PATH)
        if vocal:
            print(f"Saved model: {PATH} for epoch: {epoch}")
    except:
        print(f"Disk must be full... RIP")

def get_weights(weights):
    """convert ratio to actual weights. Sum should be one. Although, I dont
    think the sum=1 has any effect on optimization. I could just get away with
    ratio but... that isn't fancy.. or clean.
    or is it not?
    I might remove this reduntant thing in future.
    `feeling ~cute~ confused, might delete later.`
    """
    return [weights[0]/sum(weights),
            weights[1]/sum(weights),
            weights[2]/sum(weights)]

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

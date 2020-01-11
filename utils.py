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

def seed_everything(seed=42):
    """
    42 is the answer to everything.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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

    recall_grapheme = sklearn.metrics.recall_score(pred_label_graphemes, true_label_graphemes, average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_label_vowels, true_label_vowels, average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_label_consonants, true_label_consonants, average='macro')
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

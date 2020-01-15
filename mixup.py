import numpy as np
import torch
import torch.nn as nn

def mixup_data(data, labels, alpha, device):
    indices = torch.randperm(data.size(0))
    lam = np.random.beta(alpha, alpha)
    data = data * lam + data[indices] * (1 - lam)
    labels = ([labels[0].to(device), labels[1].to(device), labels[2].to(device)],
                      [labels[0][indices].to(device),
                       labels[1][indices].to(device),
                       labels[2][indices].to(device)], lam)
    return data, labels


class Mixup_CrossEntropyLoss():
    def __init__(self):
        pass

    def __call__(self, preds, labels, val=True):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        if val:
            return criterion(preds, labels)
        l1, l2, lam = labels
        return lam * criterion(preds, l1) + (1 - lam) * criterion(preds, l2)

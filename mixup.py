import numpy as np
import torch
import torch.nn as nn

def mixup(data, labels, alpha):
    indices = torch.randperm(data.size(0))
    lam = np.random.beta(alpha, alpha)
    data = data * lam + data[indices] * (1 - lam)
    labels = (labels, labels[indices], lam)
    return data, labels


class Mixup_CrossEntropyLoss():
    def __init__():
        pass

    def __cal__(preds, labels):
        l1, l2, lam = labels
        criterion = nn.CrossEntropyLoss(reduction='mean')
        return lam * criterion(preds, l1) + (1 - lam) * criterion(preds, l2)

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

def ohem_loss(cls_pred, cls_target, rate=0.3 ):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate) )
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


class CrossEntropyLoss_OHEM(torch.nn.CrossEntropyLoss):
    """ Online hard example mining.
    Needs input from nn.LogSotmax() """

    def __init__(self, ratio=0.3):
        super(CrossEntropyLoss_OHEM, self).__init__(None, True)
        self.ratio = ratio
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label]
        #loss_incs = -x_.sum(1)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return self.criterion(x_hn, y_hn)



# class Mixed_CrossEntropyLoss():
#     def __init__(self, ohem=False):
#         if ohem:
#             print("USING OHEM")
#             self.criterion = ohem_loss
#         else:
#             self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
#         pass
#
#     def __call__(self, preds, labels, val=True):
#         if val:
#             return self.criterion(preds, labels)
#         l1, l2, lam = labels
#         return lam * self.criterion(preds, l1) + (1 - lam) * self.criterion(preds, l2)

def ohem_loss_from_loss(loss, rate=0.3):
    bs = loss.shape[0]
    sorted_loss, idx = torch.sort(loss, descending=True)
    keep_num = min(sorted_loss.size()[0], int(bs*rate) )

    if keep_num < sorted_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        loss = loss[keep_idx_cuda]
    cls_loss = loss.sum() / keep_num
    return cls_loss


class CEntropy():
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.ohem_criterion = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, preds, labels, val=True, ohem=False):
        if val:
            return self.criterion(preds, labels)
        l1, l2, lam = labels
        if ohem:
            return lam * self.ohem_criterion(preds, l1) + (1 - lam) * self.ohem_criterion(preds, l2)
        return lam * self.criterion(preds, l1) + (1 - lam) * self.criterion(preds, l2)


class Mixed_CrossEntropyLoss():
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.ohem_criterion = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, preds, labels, val=True, ohem=False):
        if ohem: criterion = self.ohem_criterion
        else: criterion = self.criterion
        if val:
            return criterion(preds, labels)
        l1, l2, lam = labels
        return lam * criterion(preds, l1) + (1 - lam) * criterion(preds, l2)

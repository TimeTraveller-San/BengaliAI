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
# from mixup import *
from loaddata import *
from optimizers import *
from augmentations import *
from losses import *

import logging
import argparse
import warnings
import sys
import math

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

warnings.filterwarnings('ignore')

seed_everything()
check_dirs()


def save_cond(phase, li, num_loaders):
    if num_loaders == 1 and phase == 'valid':
        return True
    elif num_loaders == 2 and phase == 'valid' and li == 1:
        return True
    return False


def find_lr(model, optimizer, criterion, trainloader, ws, final_value=10, init_value=1e-8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train() # setup model for training configuration

    num = len(trainloader) - 1 # total number of batches
    mult = (final_value / init_value) ** (1/num)

    losses = []
    lrs = []
    best_loss = 0.
    avg_loss = 0.
    beta = 0.98 # the value for smooth losses
    lr = init_value

    for batch_num, (img, label) in enumerate(trainloader):
        img = img.to(device)
        optimizer.param_groups[0]['lr'] = lr
        batch_num += 1 # for non zero value
        optimizer.zero_grad() # clear gradients

        out = model(img)
        label[0] = label[0].to(device)
        label[1] = label[1].to(device)
        label[2] = label[2].to(device)

        loss0 = criterion(out[0], label[0])
        loss1 = criterion(out[1], label[1])
        loss2 = criterion(out[2], label[2])

        loss = ws[0]*loss0 + ws[1]*loss1 + ws[2]*loss2

        #Compute the smoothed loss to create a clean graph
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss

        # append loss and learning rates for plotting
        lrs.append(math.log10(lr))
        losses.append(smoothed_loss)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        # backprop for next step
        loss.backward()
        optimizer.step()

        # update learning rate
        lr = mult*lr

    plt.xlabel('Learning Rates')
    plt.ylabel('Losses')
    plt.plot(lrs,losses)
    plt.show()
    sys.exit()

def get_sched(schd, optimizer, train_df, batch_size, n_epochs):
    if schd is None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   # milestones=[60,75,85,90],
                                                   # milestones=[30,40,50,70], #alt
                                                   # milestones=[30,50,80,150], #alt3
                                                   milestones=[30,40,50,75], #alt4
                                                   gamma=0.5)
    elif schd == "clr":
        stepsz = int(10*train_df.shape[0]//batch_size)
        if lr < 0.01:
            max_lr = 100*lr
        else:
            max_lr = 2*lr
        scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                                base_lr=lr, max_lr=max_lr,
                                                step_size_up=stepsz)

    elif schd == "oclr":
        import math
        steps_per_epoch = math.ceil(train_df.shape[0]/batch_size)
        print(f"\n\nScjed: {steps_per_epoch}")
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=n_epochs, pct_start=0.33)
    elif schd == "rlrp":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                            patience=5, min_lr=1e-8)
    return scheduler

def get_optim(model, schd, optmzr, lr, train_df, batch_size, n_epochs, momentum=0.0, weight_decay=0.0):
    lr = 3e-4
    if optmzr == 'swats':
        print(f"\n\n\n Using SWATS")
        optimizer = SWATS(model.parameters(), lr=lr, logger=logger)
    elif optmzr == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = Adam16(model.parameters(), lr=lr) #For half precision
    elif optmzr == 'radam':
        optimizer = RAdam(model.parameters(), lr=lr)
    elif optmzr == 'sgd':
        lr = 0.1
        optimizer = optim.SGD(model.parameters(),
                        lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = None
        print(f"{optmzr} not defined")

    scheduler = get_sched(schd, optimizer, train_df, batch_size, n_epochs)

    return optimizer, scheduler, lr

def get_criterion(mixup, cutmix, ohem):
    if mixup or cutmix:
        criterion = Mixed_CrossEntropyLoss(ohem)
    else:
        if ohem:
            criterion = CrossEntropyLoss_OHEM()
        else:
            criterion = nn.CrossEntropyLoss()
    return criterion

def plot_lr(optimizer, scheduler, epochs):
    lrs = {}
    # epochs = 6600
    for epoch in range(epochs):
        lrs[epoch] = get_learning_rate(optimizer)
        scheduler.step()
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.plot(list(lrs.keys()),list(lrs.values()))
    plt.show()
    sys.exit()

def train(n_epochs=5, pretrained=False, debug=False, rgb=False,
        continue_train=False, model_name='se_resnext50_32x4d', run_name=False,
        weights=[2, 1, 1], activation=None, mixup=False, cutmix=False, alpha=1,
        min_save_epoch=3, save_freq=3, data_root="/data", save_dir=None,
        use_wandb=False, optmzr=None, heavy_head=False, toy_set=False,
        gridmask=False, morph=False, schd=None, momentum=0., weight_decay=0.,
        use_apex=False, batch_size=32, grad_acc=0, ohem=False,
        min_loss_cutoff=100000, loss_skips=3, verbose=False):

    if not run_name: run_name = model_name

    if save_dir is None:
        SAVE_DIR = f'logs/models/{run_name}'
    else:
        SAVE_DIR = os.path.join(save_dir, run_name)

    if use_wandb:
        import wandb
        wandb.init(project="bengali-ai2")

    make_dir(SAVE_DIR)
    logfile = os.path.join(SAVE_DIR, 'logs.txt')
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=logfile,
                        level=logging.DEBUG,
                        filemode='a'
                        )
    logger = logging.getLogger(__name__)
    logger.info(f"\n\n---------------- [LOGS for {run_name}] ----------------")

    if mixup and cutmix:
        augs = ['mixup', 'cutmix']
        p = [0.25, 0.75] #I want more cutmix than mixup
    elif mixup:
        augs = ['mixup']
        p = [1]
    elif cutmix:
        augs = ['cutmix']
        p = [1]
    else:
        augs = None
    if toy_set:
        train_df , valid_df = load_toy_df()
    else:
        train_df , valid_df = load_df(debug, root=data_root)

    if debug:
        LIMIT = 500
        train_df = train_df[:LIMIT]
        valid_df = valid_df[:LIMIT]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    freezed = False

    if model_name.split('-')[0] == 'efficientnet':
        model = ClassifierCNN_effnet(model_name,
                                    pretrained=pretrained,
                                    rgb=rgb,
                                    activation=activation).to(device)
        if pretrained:
            model.freeze()
            freezed = True
            logger.info("Freezing model")

    else:
        model = ClassifierCNN(model_name,
                              pretrained=pretrained,
                              rgb=rgb,
                              activation=activation,
                              heavy_head=heavy_head).to(device)
        if pretrained:
            model.freeze()
            freezed = True
            logger.info("Freezing model")


    lr = 3e-4 # Andrej must be proud of me
    # lr = 0.01 # For SGD
    if use_wandb:
        wandb.watch(model)

    # optimizer, lr = get_optim(model, optmzr, lr, momentum, weight_decay)
    optimizer, scheduler, lr = get_optim(model, schd, optmzr, lr, train_df,
                                                batch_size, n_epochs,
                                                momentum, weight_decay)

    # plot_lr(optimizer, scheduler, n_epochs)
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
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded model from: {path}")
            print(f"Continuing from epoch: {start_epoch}")
            logger.info(f"Loaded model from: {path}")
            logger.info(f"Continuing from epoch: {start_epoch}")
        except:
            continue_train = False
            start_epoch = 0
            print("Can't continue training. Starting again.")
            logger.info("Can't continue training. Starting again.")

    criterion = get_criterion(mixup, cutmix, ohem)

    train_aug = get_augs(gridmask, morph)
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

    # find_lr(model, optimizer, criterion, train_loader, ws)

    logger.info(f"Project path: {SAVE_DIR}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Model class: {type(model)}")
    logger.info(f"Debug: {debug}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"LR: {lr}")
    logger.info(f"Optimizer: {type(optimizer)}")
    try:
        logger.info(f"Scheduler: {type(scheduler)}")
    except:
        logger.info(f"Scheduler: None")
    logger.info(f"Weights: [{ws[0]} | {ws[1]} | {ws[2]}]")
    logger.info(f"Activation: {activation}")
    logger.info(f"Mixup: {mixup}")
    logger.info(f"Cutmix: {cutmix}")
    logger.info(f"Train dataset: {train_dataset}")
    logger.info(f"Validation dataset: {val_dataset}")
    logger.info(f"Continue: {continue_train}")
    logger.info(f"Momentum: {momentum}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation: {grad_acc}")
    logger.info(f"Model: {model}")
    logger.info("------------------------------------------------------------")

    logger.info("Starting training...")
    if continue_train:
        logger.info(f"WILL CONTINUE FROM EPOCH: {start_epoch}\n\n")
        n_epochs += start_epoch
        epoch = start_epoch
    if verbose:
        pbar = tqdm(total=n_epochs, initial=epoch)
    unfreerze_cutoff = 0 #HARDCODED HP

    if APEX_AVAILABLE and use_apex:
        model, optimizer = amp.initialize(
               model, optimizer, opt_level="O3",
               keep_batchnorm_fp32=True
            )
    while epoch < n_epochs:
        # Epoch start
        if epoch >= unfreerze_cutoff and freezed:
            logger.info("Unfreezing model")
            model.unfreeze()
            freezed = False
        for phase in ['train', 'valid', 'save']:
            # Phase start
            if phase == 'save':
                if epoch < min_save_epoch:
                    continue
                if current >= best:
                    best = current
                    mname = os.path.join(SAVE_DIR, 'best.pth')
                    save_model(mname,
                                    epoch, model, optimizer, scheduler, True)
                    logger.info(f"Saving model {mname}")
                elif (epoch+1) % save_freq == 0:
                    mname = os.path.join(SAVE_DIR, f'{run_name}_{epoch+1}.pth')
                    save_model(mname,
                                    epoch, model, optimizer, scheduler, True)
                    logger.info(f"Saving model {mname}")
                continue

            if phase == 'train':
                model.train()
                # logger.info("----------------------------------------------------------\n")
                loaders = [train_loader]

            if phase == 'valid':
                model.eval()
                if mixup or cutmix:
                    # logger.info("++++++++++++++ VALIDATING ON BOTH, IGNORE ABOVE TRAIN METRICS ++++++++++++++")
                    # logger.info("++++++++++++++ FIRST IS TRAIN, THEN IS VAL ++++++++++++++")
                    loaders = [train_loader, val_loader] #For mixup, train_loader while training doesn't have the actual train data so validation needs to validate both. (to see if I am overfitting)
                else:
                    loaders = [val_loader]

            for li, loader in enumerate(loaders):
                # if phase == "valid" and not li:
                #     # if (epoch + 1) % 3 is not 0:
                #     #     continue #Only calculate train metric for every 3rd epoch so save time. X_X
                #     # logger.info("[Train metrics]")
                # if phase == "valid" and li:
                #     # logger.info("[Validation metrics]")

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

                if verbose:
                    bar = tqdm(loader)
                else:
                    bar = loader
                for i, (img, label) in enumerate(bar):
                    with torch.set_grad_enabled(phase == 'train'):
                        if mixup or cutmix:
                            aug = np.random.choice(augs, p=p)
                        img = img.to(device)
                        if phase == 'train':

                            if not grad_acc:
                                optimizer.zero_grad()
                            elif (epoch+1) % grad_acc == 0:
                                optimizer.zero_grad()

                            if mixup or cutmix:
                                if aug == 'mixup':
                                    img, labels = mixup_data(img, label, alpha, device)
                                else:
                                    img, labels = cutmix_data(img, label, alpha, device)
                                labels, shuffled_labels, lam = labels

                        out = model(img)
                        label[0] = label[0].to(device)
                        label[1] = label[1].to(device)
                        label[2] = label[2].to(device)

                        if (mixup or cutmix) and phase == 'train':
                            loss0 = criterion(out[0], (label[0], shuffled_labels[0], lam), False)
                            loss1 = criterion(out[1], (label[1], shuffled_labels[1], lam), False)
                            loss2 = criterion(out[2], (label[2], shuffled_labels[2], lam), False)
                        else:
                            loss0 = criterion(out[0], label[0])
                            loss1 = criterion(out[1], label[1])
                            loss2 = criterion(out[2], label[2])

                        if epoch < min_loss_cutoff:
                            loss = ws[0]*loss0 + ws[1]*loss1 + ws[2]*loss2
                        elif (epoch+1) % loss_skips == 0:
                            loss = ws[0]*loss0 + ws[1]*loss1 + ws[2]*loss2
                        else:
                            loss = loss0

                        if phase == 'train':
                            if APEX_AVAILABLE and use_apex:
                                with amp.scale_loss(loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                loss.backward()

                            current_lr = get_learning_rate(optimizer)

                            if not grad_acc:
                                optimizer.step()
                            elif (epoch+1) % grad_acc == 0:
                                optimizer.step()

                            if schd == "clr" or schd == "oclr":
                                scheduler.step()

                        if verbose:
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

                # if phase == 'valid' and li == 1: #For the validation dataset
                if save_cond(phase, li, len(loaders)):
                    current = recall #Update current score
                    if schd == "rlrp":
                        print("\nplatue")
                        scheduler.step(recall) #Step for val loss only

                epoch_str = f"[{epoch+1}/{n_epochs}] | {phase[0]}_{li} | "
                recall_str = f"R: {running_recall:.3f} | [{running_recall0:.3f} | {running_recall1:.3f} | {running_recall2:.3f}] | "
                acc_str = f"A: [{100*running_acc0:.3f}% | {100*running_acc1:.3f}% | {100*running_acc2:.3f}%] | "
                loss_str = f"L: {running_loss:.3f} | [{running_loss0:.3f} | {running_loss1:.3f} | {running_loss2:.3f}]"
                lr_str = f"LR: {get_learning_rate(optimizer)}"

                if verbose:
                    print(epoch_str)
                    print(recall_str)
                    print(acc_str)
                    print(loss_str)

                # logger.info(f"Epoch: [{epoch+1}/{n_epochs}] {phase}...")
                # logger.info(f"Learning rate: {current_lr}")
                # logger.info(f">> Recall: {running_recall:.3f} | [{running_recall0:.3f} | {running_recall1:.3f} | {running_recall2:.3f}] <<")
                # logger.info(f"Acc:  [{100*running_acc0:.3f}% | {100*running_acc1:.3f}% | {100*running_acc2:.3f}%]")
                # logger.info(f"Loss: {running_loss:.3f} | [{running_loss0:.3f} | {running_loss1:.3f} | {running_loss2:.3f}]\n")
                logger.info(epoch_str+recall_str+acc_str+loss_str+lr_str)

                if use_wandb:
                    wandb.log({f"{phase}_{li}_loss": running_loss})
                    wandb.log({f"{phase}_{li}_recall": running_recall})
                    wandb.log({f"{phase}_{li}_recall_grapheme": running_recall0})
                    wandb.log({f"{phase}_{li}_recall_vowel": running_recall1})
                    wandb.log({f"{phase}_{li}_recall_consonant": running_recall2})


                history.loc[epoch, f'{phase}_{li}_loss'] = running_loss
                history.loc[epoch, f'{phase}_{li}_recall'] = running_recall
                history.loc[epoch, f'{phase}_{li}_recall_grapheme'] = running_recall0
                history.loc[epoch, f'{phase}_{li}_recall_vowel'] = running_recall1
                history.loc[epoch, f'{phase}_{li}_recall_consonant'] = running_recall2
                history.loc[epoch, f'{phase}_{li}_acc_grapheme'] = running_acc0.cpu().numpy()
                history.loc[epoch, f'{phase}_{li}_acc_vowel'] = running_acc1.cpu().numpy()
                history.loc[epoch, f'{phase}_{li}_acc_consonant'] = running_acc2.cpu().numpy()
                # Loader end
            # # this part new
            if (phase == "valid") and (schd not in ["clr", "oclr", "rlrp"]):
                scheduler.step() #Step for val loss only
            # Phase end
        # Epoch end
        save_hist = False
        if save_hist:
            history.to_csv(os.path.join(SAVE_DIR, f"{run_name}_{epoch}.csv"))
        epoch += 1
        if verbose:
            pbar.update(1)


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
    parser.add_argument("--model_name", "-mn", default="se_resnext50_32x4d",
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
    parser.add_argument("--cutmix", "-cx", default=False,
                            help="cutmix augmentations, only on input for now")
    parser.add_argument("--alpha", "-alpha", default=1,
                            help="alpha for mixup and cutmix")
    parser.add_argument("--min_save_epoch", "-mse", default=3,
                            help="minimum epoch to start saving models")
    parser.add_argument("--save_freq", "-sf", default=3,
                            help="frequency of saving epochs")
    parser.add_argument("--data_root", "-dr", default="data/",
                            help="location of data")
    parser.add_argument("--save_dir", "-sr", default=None,
                            help="directory to save model")
    parser.add_argument("--use_wandb", "-wb", default=False,
                            help="use wandb or not?")
    parser.add_argument("--optmzr", "-optim", default="adam",
                            help="what optimizer to use")
    parser.add_argument("--heavy_head", "-hh", default=False,
                            help="head for network end, heavy (Conv) or not.")
    parser.add_argument("--toy_set", "-ts", default=False,
                            help="use toy dataset or not.")
    parser.add_argument("--gridmask", "-grm", default=False,
                            help="gridmask augmentation.")
    parser.add_argument("--morph", "-mrp", default=False,
                            help="morphological augmentation: bengaliai-cv19/discussion/128198")
    parser.add_argument("--scheduler", "-sch", default=None,
                            help="type of scheduler.")
    parser.add_argument("--momentum", "-mom", default=0.0,
                            help="for SGD")
    parser.add_argument("--weight_decay", "-wde", default=0.0,
                            help="for adam and others")
    parser.add_argument("--use_apex", "-ax", default=False,
                            help="nvidia apex")
    parser.add_argument("--batch_size", "-bs", default=32,
                            help="batch size")
    parser.add_argument("--grad_acc", "-gac", default=0,
                            help="gradient accumulation")
    parser.add_argument("--ohem", "-ohm", default=False,
                            help="Online Hard Example Mining")
    parser.add_argument("--min_loss_cutoff", "-mlc", default=1000000,
                            help="Min epochs to start cutting loss to just root")
    parser.add_argument("--loss_skips", "-lskips", default=3,
                            help="No. epochs to skip for constant and vowel")
    parser.add_argument("--verbose", "-v", default=False,
                            help="print loss on screen or not?")



    args = parser.parse_args()
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
        args.cutmix,
        float(args.alpha),
        int(args.min_save_epoch),
        args.save_freq,
        args.data_root,
        args.save_dir,
        args.use_wandb,
        args.optmzr,
        args.heavy_head,
        args.toy_set,
        args.gridmask,
        args.morph,
        args.scheduler,
        float(args.momentum),
        float(args.weight_decay),
        args.use_apex,
        int(args.batch_size),
        int(args.grad_acc),
        args.ohem,
        int(args.min_loss_cutoff),
        int(args.loss_skips),
        args.verbose,
        )

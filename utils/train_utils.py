#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.metric import dice_coeff
import logging
import os
    
def write_metric_values(net, output, epoch, train_metric, val_metric):
    file = open(output+'epoch-%0*d.txt'%(2,epoch),'w') 
    if net.n_classes > 1:
        logging.info('training cross-entropy: {}'.format(train_metric[-1]))
        logging.info('validation cross-entropy: {}'.format(val_metric[-1]))
        file.write('train cross-entropy = %f\n'%(train_metric[-1]))
        file.write('val cross-entropy = %f'%(val_metric[-1]))
    else:
        logging.info('training dice: {}'.format(train_metric[-1]))
        logging.info('validation dice: {}'.format(val_metric[-1]))
        file.write('train dice = %f\n'%(train_metric[-1]))
        file.write('val dice = %f'%(val_metric[-1]))
    file.close()

def launch_training(epochs, net, train_loader, val_loader, n_train, device, net_id, criterion, optimizer, output):
    ''' generic training launcher '''
    if net.n_classes > 1:
        train_cross_entropy, val_cross_entropy = [], []
        best_val_cross_entropy = 1e6
    else :
        train_dices, val_dices = [], []
        best_val_dice = -1

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, masks = batch[0], batch[1]
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                masks = masks.to(device=device, dtype=mask_type)
                preds = net(imgs)
                
                if net.n_classes > 1:
                    if net_id in [8, 9, 10, 11, 12, 13]:
                        loss = supervision_loss(net.attention, preds, torch.squeeze(masks,1), criterion)
                    else:
                        loss = criterion(preds, torch.squeeze(masks,1))
                else:
                    if net_id in [8, 9, 10, 11, 12, 13]:
                        loss = supervision_loss(net.attention, preds, masks,1, criterion)
                    else:
                        loss = criterion(preds, masks)                    
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
        if net.n_classes > 1:
            train_cross_entropy.append(eval_net(net, train_loader, device))
            val_cross_entropy.append(eval_net(net, val_loader, device))
            write_metric_values(net, output, epoch, train_cross_entropy, val_cross_entropy)
            if epoch>0:
                if val_cross_entropy[-1]<best_val_cross_entropy:
                    os.remove(output+'epoch.pth')
                    torch.save(net.state_dict(), output+'epoch.pth')
                    best_val_cross_entropy = val_cross_entropy[-1]
                    logging.info(f'checkpoint {epoch + 1} saved !')
            else:
                torch.save(net.state_dict(), output+'epoch.pth')
        else:
            train_dices.append(eval_net(net, train_loader, device))
            val_dices.append(eval_net(net, val_loader, device))
            write_metric_values(net, output, epoch, train_dices, val_dices)
            if epoch>0:
                if val_dices[-1]>best_val_dice:
                    os.remove(output+'epoch.pth')
                    torch.save(net.state_dict(), output+'epoch.pth')
                    best_val_dice = val_dices[-1]
                    logging.info(f'checkpoint {epoch + 1} saved !')
            else:
                torch.save(net.state_dict(), output+'epoch.pth') 
    if net.n_classes > 1: 
        return train_cross_entropy, val_cross_entropy
    else:
        return train_dices, val_dices
            
def dice_history(epochs, train_dices, val_dices, output):
    ''' display and save dices after training '''
    plt.plot(range(epochs), train_dices)
    plt.plot(range(epochs), val_dices)
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.xlim([0,epochs-1]) 
    plt.ylim([0,1]) 
    plt.grid()
    plt.savefig(output+'dices.png')
    plt.close()
    np.save(output+'train_dices.npy', np.array(train_dices))
    np.save(output+'val_dices.npy', np.array(val_dices))
    
def cross_entropy_history(epochs, train_cross_entropy, val_cross_entropy, output):
    ''' display and save dices after training '''
    plt.plot(range(epochs), train_cross_entropy)
    plt.plot(range(epochs), val_cross_entropy)
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('cross-entropy')
    plt.xlim([0,epochs-1]) 
    # plt.ylim([0,1]) 
    plt.grid()
    plt.savefig(output+'cross-entropy.png')
    plt.close()
    np.save(output+'train_cross_entropy.npy', np.array(train_cross_entropy))
    np.save(output+'val_dcross_entropy.npy', np.array(val_cross_entropy))
    
def eval_net(net, loader, device):
    ''' evaluation with dice coefficient '''
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)
    tot = 0
    for batch in loader:
        imgs, masks = batch[0], batch[1]
        imgs = imgs.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=mask_type)
        with torch.no_grad():
            preds = net(imgs)
        if net.n_classes > 1:
            tot += nn.functional.cross_entropy(preds,torch.squeeze(masks,1))
        else:
            preds = torch.sigmoid(preds)            
            preds = (preds > 0.5).float()
            tot += dice_coeff(preds, masks).item()
    return tot / n_val

def supervision_loss(attention, preds, masks, criterion):
    if attention:
        loss1 = criterion(preds[0], masks)
        loss2 = criterion(preds[1], masks)
        loss3 = criterion(preds[2], masks)
        loss4 = criterion(preds[3], masks)
        loss5 = criterion(preds[4], masks)
        loss6 = criterion(preds[5], masks)
        loss7 = criterion(preds[6], masks)
        loss8 = criterion(preds[7], masks)
        loss9 = criterion(preds[8], masks)
        loss = loss1+0.8*loss2+0.7*loss3+0.6*loss4+0.5*loss5+0.8*loss6+0.7*loss7+0.6*loss8+0.5*loss9
        loss /= 6.2
    else:
        loss1 = criterion(preds[0], masks)
        loss2 = criterion(preds[1], masks)
        loss3 = criterion(preds[2], masks)
        loss4 = criterion(preds[3], masks)
        loss5 = criterion(preds[4], masks)
        loss = loss1+0.8*loss2+0.7*loss3+0.6*loss4+0.5*loss5    
        loss /= 3.6        
    return loss
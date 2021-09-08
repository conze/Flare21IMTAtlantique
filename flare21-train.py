#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: Pierre-Henri Conze | IMT Atlantique, LaTIM UMR 1101, Inserm

from datasets.dataset_flare21 import dataset_flare21
from manage.manage_flare21 import create_flare21_dataset, flare21_split
import argparse
import logging
import sys
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import distutils.dir_util
from nets.whichnet import whichnet
from utils.train_utils import launch_training, dice_history, cross_entropy_history

def train_flare21(root,
                  net_id, 
                  net,
                  anatomy,
                  output,
                  device,
                  vgg,
                  epochs,
                  batch,
                  lr,
                  size):
    
    train_ids, val_ids = flare21_split()
    
    logging.info(f'''training ids: {train_ids}''')
    
    logging.info(f'''validation ids: {val_ids}''')
                 
    if os.path.isfile(output+'imgs-id-train.npy') == False:
        create_flare21_dataset(root, output, train_ids, 'train', size, anatomy)
    
    if os.path.isfile(output+'imgs-id-val.npy') == False:
        create_flare21_dataset(root, output, val_ids, 'val', size, anatomy)
    
    train_dataset = dataset_flare21(output, 'train', anatomy, vgg)

    val_dataset = dataset_flare21(output, 'val', anatomy, vgg)    

    n_train, n_val = len(train_dataset), len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
  
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    logging.info(f'''starting training:
        vgg:             {vgg}
        epochs:          {epochs}
        batch size:      {batch}
        learning rate:   {lr}
        training size:   {n_train}
        validation size: {n_val}
        size:            {size}
        device:          {device.type}''')
    
    optimizer = optim.Adam(net.parameters(), lr=1e-6*lr)
    
    if net.n_channels>1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()        
    
    return launch_training(epochs, net, train_loader, val_loader, n_train, device, net_id, criterion, optimizer, output)

def get_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', '--input', type=str, dest='input')

    parser.add_argument('-o', '--output', type=str, dest='output')
    
    parser.add_argument('-a', '--anatomy', type=str, default='all', dest='anatomy') # liver, kidneys, spleen, pancreas or all
    
    parser.add_argument('-e', '--epochs', type=int, default=100, dest='epochs')
    
    parser.add_argument('-b', '--batch', type=int, default=4, dest='batch')
    
    parser.add_argument('-l', '--learning', type=int, default=10, dest='lr')
    
    parser.add_argument('-n', '--network', type=int, default=1, dest='network')
    
    parser.add_argument('-s', '--size', type=int, default=512, dest='size')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info('using device: ' + str(device))
    
    if args.anatomy == 'all':
        n_classes = 5 # 4 organs + background
    else:
        n_classes = 1        
    net, vgg = whichnet(args.network, n_classes) 
    
    logging.info(f'network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    net.to(device=device)

    try:
        
        args.output += 'flare21-'+args.anatomy+'-n-'+str(args.network)+'-e-'+str(args.epochs)+'-b-'+str(args.batch)
        args.output += '-l-'+str(int(args.lr))+'-s-'+str(args.size)+'/'

        distutils.dir_util.mkpath(args.output)

        train_metric, val_metric = train_flare21(root = args.input,
                                                 net_id = args.network, 
                                                 net = net,
                                                 anatomy = args.anatomy,
                                                 output = args.output,
                                                 device = device,
                                                 vgg = vgg,
                                                 epochs = args.epochs,
                                                 batch = args.batch,
                                                 lr = args.lr,
                                                 size = args.size)
        
        if net.n_classes > 1:
            cross_entropy_history(args.epochs, train_metric, val_metric, args.output)            
        else:
            dice_history(args.epochs, train_metric, val_metric, args.output)

    except KeyboardInterrupt:
        logging.info('keyboard interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
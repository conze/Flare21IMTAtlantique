#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
import torch
from utils.utils import get_array_affine_header, get_largest_connected_region, get_2_largest_connected_region
from torch.utils.data import DataLoader
import tqdm
import distutils.dir_util
import nibabel
from skimage.transform import resize, rotate
from datasets.dataset_flare21 import tiny_dataset_flare21
from nets.whichnet import whichnet
import os
    
def listdir_nohidden(path):
    l = []
    for f in np.sort(os.listdir(path)) :
        if f.startswith('.') == False :
            l.append(f)
    return l

def infer_flare21(net,
                  net_id,
                  anatomy,
                  output,
                  device,
                  vgg,
                  size):

    list_ = listdir_nohidden('./inputs/')
    test_ids = []
    for elem_ in list_:
        if elem_.split('.')[1] == 'nii':
            test_ids.append(elem_.split('_')[1])
    test_ids = np.array(test_ids, dtype = np.int)
    
    for index, id_ in enumerate(tqdm.tqdm(test_ids)):
        
        test_dataset = tiny_dataset_flare21(id_, size, anatomy, vgg)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        array, affine, header = get_array_affine_header(test_dataset, 'CT')
        
        array_liver = np.copy(array)
        array_kidneys = np.copy(array)
        array_spleen = np.copy(array)
        array_pancreas = np.copy(array)        

        with torch.no_grad():
            
            for idx, data in enumerate(test_loader):
                
                image = data
                image = image.to(device=device, dtype=torch.float32)
                
                net.training = False
                
                prob = torch.softmax(net(image), dim=1)
                pred = torch.argmax(prob, dim=1).float()
                full_mask = pred.squeeze().cpu().numpy().swapaxes(0,1).astype(np.uint8)
                
                mask_liver = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_kidneys = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_spleen = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_pancreas = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                
                mask_liver[np.where(full_mask==1)] = 1
                mask_kidneys[np.where(full_mask==2)] = 1
                mask_spleen[np.where(full_mask==3)] = 1
                mask_pancreas[np.where(full_mask==4)] = 1                
                
                mask_liver = resize(rotate(mask_liver, -90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_kidneys = resize(rotate(mask_kidneys, -90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_spleen = resize(rotate(mask_spleen, -90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_pancreas = resize(rotate(mask_pancreas, -90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                    
                mask_liver[np.where(mask_liver>0.95)] = 1
                mask_liver[np.where(mask_liver!=1)] = 0
                
                mask_kidneys[np.where(mask_kidneys>0.95)] = 1
                mask_kidneys[np.where(mask_kidneys!=1)] = 0
                
                mask_spleen[np.where(mask_spleen>0.95)] = 1
                mask_spleen[np.where(mask_spleen!=1)] = 0
                
                mask_pancreas[np.where(mask_pancreas>0.95)] = 1
                mask_pancreas[np.where(mask_pancreas!=1)] = 0
                
                array_liver[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_liver[::-1,::]
                array_kidneys[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_kidneys[::-1,::]
                array_spleen[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_spleen[::-1,::]
                array_pancreas[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_pancreas[::-1,::]
                
        array_liver = get_largest_connected_region(array_liver)
        array_kidneys = get_2_largest_connected_region(array_kidneys)
        array_spleen = get_largest_connected_region(array_spleen)
        array_pancreas = get_largest_connected_region(array_pancreas)
        array[np.where(array_liver==1)] = 1
        array[np.where(array_kidneys==1)] = 2
        array[np.where(array_spleen==1)] = 3
        array[np.where(array_pancreas==1)] = 4
        
        prediction = nibabel.Nifti1Image(array.astype(np.uint16), affine=affine, header=header)
        nibabel.save(prediction, output+'test_%0*d'%(3,id_)+'.nii.gz')
        
        del prediction, test_dataset, array, array_liver, array_kidneys, array_spleen, array_pancreas

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
    model = './weights/epoch.pth'
    
    output = './outputs/'
    
    distutils.dir_util.mkpath(output)
    
    net_id = 1
    
    n_classes = 5 # 4 organs + background   
    
    size = 512   

    net, vgg = whichnet(net_id, n_classes)

    logging.info("loading model {}".format(model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'using device {device}')
    
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    logging.info("model loaded !")

    infer_flare21(net = net,
                  net_id = net_id,
                  anatomy = 'all',
                  output = output, 
                  device = device,
                  vgg = vgg,
                  size = size)
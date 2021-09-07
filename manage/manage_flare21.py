#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from exams.exam_flare21 import exam_flare21
import distutils.dir_util
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.transform import resize, rotate
from skimage import io
import tqdm
from utils import utils
import random

def create_flare21_dataset(output, ids, scheme, size, anatomy):
    ''' create flare21 dataset with scheme={train,val} and anatomy={all, liver, kidneys, spleen, pancreas} '''
    
    list_id = []
    folder = output+scheme+'/'
    distutils.dir_util.mkpath(folder)
    
    for idx, id_ in enumerate(tqdm.tqdm(ids)):
        exam = exam_flare21(id_, 'train', anatomy)
        exam.normalize()  
        for xyz in range(exam.CT.shape[2]): 
            img, mask = extract_flare21_slice(exam, xyz, size, anatomy)
            if len(np.unique(img))>1: # to avoid blank images
                list_id.append('CT-%0*d-'%(3,id_)+anatomy+'-%0*d'%(3,xyz+1))
                io.imsave(folder+list_id[-1]+'-src.png', img)
                io.imsave(folder+list_id[-1]+'-mask.png', mask, check_contrast=False)
                if len(np.unique(mask))>1:
                    io.imsave(folder+list_id[-1]+'-bound.png', img_as_ubyte(utils.boundaries(img, None, mask)))
        del exam
    np.save(output+'imgs-id-'+scheme+'.npy', list_id)
    del list_id
    
def extract_flare21_slice(exam, xyz, size, anatomy, mask_available=True):
    
    img = rotate(resize(np.squeeze(exam.CT.get_fdata()[:,:,xyz])[::-1,:], output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
    min_greyscale, max_greyscale = np.percentile(img,(1,99)) 
    img = rescale_intensity(img, in_range=(min_greyscale,max_greyscale), out_range=(0,1))
    
    if mask_available:
        if anatomy == 'liver' or anatomy == 'all':
           mask_liver = rotate(resize(exam.mask_liver.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        if anatomy == 'kidneys' or anatomy == 'all':
            mask_kidneys = rotate(resize(exam.mask_kidneys.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        if anatomy == 'spleen' or anatomy == 'all':
            mask_spleen = rotate(resize(exam.mask_spleen.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
        if anatomy == 'pancreas' or anatomy == 'all':
            mask_pancreas = rotate(resize(exam.mask_pancreas.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)
            
        if anatomy == 'liver':
            mask_liver[np.where(mask_liver>0)] = 255
            return img_as_ubyte(img), mask_liver.astype(np.uint8)
        elif anatomy == 'kidneys':
            mask_kidneys[np.where(mask_kidneys>0)] = 255
            return img_as_ubyte(img), mask_kidneys.astype(np.uint8)
        elif anatomy == 'spleen':
            mask_spleen[np.where(mask_spleen>0)] = 255
            return img_as_ubyte(img), mask_spleen.astype(np.uint8)
        elif anatomy == 'pancreas':
            mask_pancreas[np.where(mask_pancreas>0)] = 255
            return img_as_ubyte(img), mask_pancreas.astype(np.uint8) 
        elif anatomy == 'all':
            mask = np.zeros(shape=mask_liver.shape, dtype=np.uint8)
            mask[np.where(mask_liver>0)] = 1
            mask[np.where(mask_kidneys>0)] = 2
            mask[np.where(mask_spleen>0)] = 3
            mask[np.where(mask_pancreas>0)] = 4
            return img_as_ubyte(img), mask.astype(np.uint8)
    else:
        return img_as_ubyte(img)

def flare21_split():
    ids = list(range(0,360))
    random.seed(4)
    random.shuffle(ids)
    train_ids = ids[:350]
    val_ids = ids[350:]
    return list(train_ids), list(val_ids)
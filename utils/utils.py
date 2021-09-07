#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.exposure import rescale_intensity
from skimage.measure import label
import xlrd
import nibabel

def boundaries(img, pred, groundtruth):
    img = rescale_intensity(img, in_range=(np.min(img),np.max(img)), out_range=(0,1))
    if type(pred) == np.ndarray and type(groundtruth) == np.ndarray:
        out = mark_boundaries(img, groundtruth, color=(0, 1, 0), background_label=4)
        out = mark_boundaries(out, pred, color=(1, 0, 0), background_label=2)
    else:
        if type(pred) == np.ndarray:
            out = mark_boundaries(img, pred, color=(1, 0, 0), background_label=2)
        if type(groundtruth) == np.ndarray:
            out = mark_boundaries(img, groundtruth, color=(0, 1, 0), background_label=4)            
    return out

def get_all_metaHep_ID_serie(root, xlsx):
    xlsx = root+xlsx
    wb = xlrd.open_workbook(xlsx)
    sh = wb.sheet_by_name('main')
    list_ids = [list(sh.row_values(rownum))[0] for rownum in range(1,sh.nrows)]
    list_ids = np.array(list_ids, dtype=np.int)
    list_series = [list(sh.row_values(rownum))[8] for rownum in range(1,sh.nrows)]    
    list_series = np.array(list_series, dtype=np.int)
    return list(list_ids), list(list_series)

# =======
def metahep_resec_get_array_affine_header(test_dataset):
    array = np.zeros(test_dataset.exam.VE.shape, dtype=np.uint16)
    affine, header = test_dataset.exam.VE.affine, test_dataset.exam.VE.header
    return array, affine, header
# =======

def print_ods_(scores, test_ids, output, name):
    resfile = open(output+name, "a")
    resfile.write('exam\t'+'dice\t'+'sens\t'+'spec\t'+'jacc\t'+'avd\t'+'assd\t'+'mssd\t\n')
    for index, id_ in enumerate(test_ids):
        resfile.write('%0*d'%(3,id_)+'\t')
        for idx in range(scores.shape[1]):
            resfile.write(str('%.3f'%scores[index,idx]).replace(".", ",")+'\t')
        resfile.write('\n')
    resfile.write('mean\t') 
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.mean(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\nstd\t')
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.std(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\n')
    resfile.close()
    
def print_ods(scores, test_ids, test_series, output, name):
    resfile = open(output+name, "a")
    resfile.write('exam\t'+'serie\t'+'dice\t'+'sens\t'+'spec\t'+'jacc\t'+'avd\t'+'assd\t'+'mssd\t\n')
    for index, id_ in enumerate(test_ids):
        resfile.write('%0*d'%(3,id_)+'\t')
        resfile.write('%0*d'%(2,test_series[index])+'\t')        
        for idx in range(scores.shape[1]):
            resfile.write(str('%.3f'%scores[index,idx]).replace(".", ",")+'\t')
        resfile.write('\n')
    resfile.write('mean\t\t') 
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.mean(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\nstd\t\t')
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.std(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\n')
    resfile.close()

# =======
def metahep_resec_print_ods(scores, test_ids, test_series, output, name):
    resfile = open(output+name, "a")
    resfile.write('exam\t'+'serie\t'+'dice\t'+'sens\t'+'spec\t'+'jacc\t'+'avd\t'+'assd\t'+'mssd\t\n')
    for index, id_ in enumerate(test_ids):
        resfile.write('%0*d'%(3,id_)+'\t')
        resfile.write('%0*d'%(2,test_series[index])+'\t')
        for idx in range(scores.shape[1]):
            resfile.write(str('%.3f'%scores[index,idx]).replace(".", ",")+'\t')
        resfile.write('\n')
    resfile.write('mean\t\t') 
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.mean(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\nstd\t\t')
    for idx in range(scores.shape[1]):
        resfile.write(str('%.3f'%np.std(scores[:,idx])).replace(".", ",")+'\t')
    resfile.write('\n')
    resfile.close()
# =======
    
def normalization_imgs(imgs):
    ''' centering and reducing data structures '''
    imgs = imgs.astype(np.float32, copy=False)
    mean = np.mean(imgs) # mean for data centering
    std = np.std(imgs) # std for data normalization
    if np.int32(std) != 0:
        imgs -= mean
        imgs /= std
    return imgs

def normalization_masks(imgs_masks):
    imgs_masks = imgs_masks.astype(np.float32, copy=False)
    imgs_masks /= 255.
    imgs_masks = imgs_masks.astype(np.uint8)
    return imgs_masks

def get_array_affine_header(test_dataset, modality):
    if modality == 'T2':
        array = np.zeros(test_dataset.exam.T2.shape, dtype=np.uint16)
        affine, header = test_dataset.exam.T2.affine, test_dataset.exam.T2.header
    elif modality == 'CT':
        array = np.zeros(test_dataset.exam.CT.shape, dtype=np.uint16)
        affine, header = test_dataset.exam.CT.affine, test_dataset.exam.CT.header        
    return array, affine, header
    
def img_init(img):
    return nibabel.Nifti1Image(img.get_fdata().astype(np.float32), affine=img.affine, header=img.header)

def img_zero(mask):
    return nibabel.Nifti1Image(np.zeros(shape=mask.shape).astype(np.float32), affine=mask.affine, header=mask.header)

def img_init_array(img, array):
    return nibabel.Nifti1Image(array.astype(np.float32), affine=img.affine, header=img.header)
    
def mask_init(mask):
    return nibabel.Nifti1Image(mask.get_fdata().astype(np.uint8), affine=mask.affine, header=mask.header)

def mask_zero(mask):
    return nibabel.Nifti1Image(np.zeros(shape=mask.shape).astype(np.uint8), affine=mask.affine, header=mask.header)

def mask_init_array(img, array):
    return nibabel.Nifti1Image(array.astype(np.uint8), affine=img.affine, header=img.header)

def get_largest_connected_region(segmentation):
    if len(np.unique(segmentation)) == 1:
        return segmentation
    else:
        labels = label(segmentation,connectivity=1)
        unique, counts = np.unique(labels, return_counts=True)
        list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
        largest=max(list_seg, key=lambda x:x[1])[0]
        labels_max=(labels == largest).astype(int)
        return labels_max
    
def get_2_largest_connected_region(segmentation):
    if len(np.unique(segmentation)) == 1:
        return segmentation
    else:
        labels = label(segmentation,connectivity=1)
        unique, counts = np.unique(labels, return_counts=True)
        list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
        largest=max(list_seg, key=lambda x:x[1])[0]
        labels_max1=(labels == largest)
        labels[np.where(labels == largest)] = 0
        unique, counts = np.unique(labels, return_counts=True)
        list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
        largest=max(list_seg, key=lambda x:x[1])[0]
        labels_max2=(labels == largest)
        return labels_max1+labels_max2
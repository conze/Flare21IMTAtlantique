#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
from skimage import io
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, affine
from exams.exam_flare21 import exam_flare21
from manage.manage_flare21 import extract_flare21_slice
from utils.utils import normalization_imgs
 
class dataset_flare21(Dataset):

    def __init__(self, path:str, scheme:str, anatomy:str, vgg:bool=False):
        self.path = path
        self.scheme = scheme
        self.anatomy = anatomy
        self.vgg = vgg
        self.ids = np.load(self.path+'imgs-id-'+self.scheme+'.npy')

    def __len__(self):
        return len(self.ids)

    def transform(self, img, mask):
        (d,t,sc,sh) = transforms.RandomAffine.get_params(degrees=(-20,20), translate=(0.2,0.2), scale_ranges=(0.8,1.2), shears=(-20,20), img_size=img.shape)
        img = affine(to_pil_image(img), angle=d, translate=t, scale=sc, shear=sh)
        mask = affine(to_pil_image(mask), angle=d, translate=t, scale=sc, shear=sh)
        return (np.array(img), np.array(mask))

    def __getitem__(self, idx:int):
        folder = self.path+self.scheme+'/'
        img = io.imread(folder+self.ids[idx]+'-src.png')
        mask = io.imread(folder+self.ids[idx]+'-mask.png')
        if self.scheme == 'train':
            img, mask = self.transform(img, mask)	
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.float32)
        img_[:,:,0] = img
        if self.vgg:
            img_[:,:,1], img_[:,:,2] = img, img
        mask_ = np.zeros(shape=(mask.shape[0],mask.shape[1],1), dtype=np.uint8)
        mask_[:,:,0] = mask
        img_ = normalization_imgs(img_)
        return (img_.swapaxes(2,0), mask_.swapaxes(2,0))
    
class tiny_dataset_flare21(Dataset):
    ''' one single examination for prediction purposes '''
    
    def __init__(self, id_, size, anatomy:str, vgg:bool=False):
        self.id = id_
        self.size = size
        self.anatomy = anatomy
        self.vgg = vgg
        self.exam = exam_flare21('./inputs/', self.id, 'test', self.anatomy, upload=True)
        self.exam.normalize()
            
    def __len__(self):
        return self.exam.CT.shape[2]

    def __getitem__(self, idx:int):
        img = extract_flare21_slice(self.exam, idx, self.size, self.anatomy, mask_available=False)
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.float32)
        img_[:,:,0] = img
        if self.vgg:
            img_[:,:,1], img_[:,:,2] = img, img
        img_ = normalization_imgs(img_)
        return img_.swapaxes(2,0)

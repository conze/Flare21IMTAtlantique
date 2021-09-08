#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel
import logging
from utils.utils import normalization_imgs, mask_zero
import numpy as np

class exam_flare21:

    def __init__(self, root, id_, scheme:str, anatomy, upload=True):
        
        self.root = './inputs/'
        self.scheme = scheme
        if self.scheme == 'train':
            self.folder_src = self.root + 'TrainingImg/'
            self.folder_mask = self.root + 'TrainingMask/' 
        elif self.scheme == 'val':
            self.folder_src = self.root #+ 'ValidationImg/'
        elif self.scheme == 'test':
            self.folder_src = self.root #+ 'TestImg/'
        self.id = '%0*d'%(3,id_)
        self.anatomy = anatomy
        if upload:
            self.exam_upload()
            self.print_info()
            
    def separate_organs(self):
        
        self.mask_liver = mask_zero(self.mask)
        self.mask_liver.get_fdata()[np.where(self.mask.get_fdata()==1.)] = 1
    
        self.mask_kidneys = mask_zero(self.mask)
        self.mask_kidneys.get_fdata()[np.where(self.mask.get_fdata()==2.)] = 1
    
        self.mask_spleen = mask_zero(self.mask)
        self.mask_spleen.get_fdata()[np.where(self.mask.get_fdata()==3.)] = 1
    
        self.mask_pancreas = mask_zero(self.mask)
        self.mask_pancreas.get_fdata()[np.where(self.mask.get_fdata()==4.)] = 1

    def exam_upload(self): 
        
        if self.scheme == 'train':
            self.CT = nibabel.as_closest_canonical(nibabel.load(self.folder_src+'train_'+self.id+'_0000.nii.gz'))
            self.mask = nibabel.as_closest_canonical(nibabel.load(self.folder_mask+'train_'+self.id+'.nii.gz'))
            self.separate_organs()
        elif self.scheme == 'val':
            self.CT = nibabel.as_closest_canonical(nibabel.load(self.folder_src+'validation_'+self.id+'_0000.nii.gz'))
        elif self.scheme == 'test':
            self.CT = nibabel.as_closest_canonical(nibabel.load(self.folder_src+'test_'+self.id+'_0000.nii.gz'))
            
        
    def normalize(self):
        self.CT.get_fdata()[:,:,:] = normalization_imgs(self.CT.get_fdata())[:,:,:]
            
    def print_info(self):
        
        logging.basicConfig(level=logging.INFO, format='\n %(levelname)s: %(message)s')
        logging.info(f'''exam {self.id} uploaded:                   
        shape:         {self.CT.shape}
        anatomy:       {self.anatomy}
        ''')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 08:44:59 2017

@author: eli
"""


import os
import numpy as np
from sklearn.cross_validation import train_test_split
from skimage import io
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from common import utils
from datasets import base_dataset
from scipy.misc import imresize
from scipy import interpolate
import matplotlib.pyplot as plt
import glob
import rawpy

class  MSR_Demosaicing(base_dataset.BaseDataset):
    
    def __init__(self,
                 dir_path = '/data/datasets/MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_panasonic/'):
        
        assert os.path.isdir(dir_path), 'dir not exist: ' + dir_path
        self.dir_path = dir_path
        return
        
    def get_training_samples(self):
        X = []
        y = []
        for line in open(os.path.join(self.dir_path, 'train.txt'), 'r'):
            line = line.rstrip()
            gt = plt.imread(os.path.join(self.dir_path,'groundtruth',line + '.png'))[:,:,:3]
            mosaiced = plt.imread(os.path.join(self.dir_path,'input',line + '.png'))
            demosaiced = utils.mosaic_then_demosaic(mosaiced, 'rggb')
            
            X.append(demosaiced)
            y.append(gt)
            
        for line in open(os.path.join(self.dir_path, 'validation.txt'), 'r'):
            line = line.rstrip()
            gt = plt.imread(os.path.join(self.dir_path,'groundtruth',line + '.png'))[:,:,:3]
            mosaiced = plt.imread(os.path.join(self.dir_path,'input',line + '.png'))
            demosaiced = utils.mosaic_then_demosaic(mosaiced, 'rggb')
            
            X.append(demosaiced)
            y.append(gt)
            
        X = np.stack(X)
        y = np.stack(y)
        
        X -= 0.5
        y -= 0.5
        
#        X = np.transpose(X, [0,3,1,2])
        y = np.transpose(y, [0,3,1,2])
            
        return X, y

    def get_test_samples(self):
        X = []
        y = []
        fnames = []
        for line in open(os.path.join(self.dir_path, 'test.txt'), 'r'):
            line = line.rstrip()
            gt = plt.imread(os.path.join(self.dir_path,'groundtruth',line + '.png'))[:,:,:3]
            mosaiced = plt.imread(os.path.join(self.dir_path,'input',line + '.png'))
            demosaiced = utils.mosaic_then_demosaic(mosaiced, 'rggb')
            
            X.append(demosaiced)
            y.append(gt)
            fnames.append(line)
            
        X = np.stack(X)
        y = np.stack(y)
        
        X -= 0.5
        y -= 0.5
        
#        X = np.transpose(X, [0,3,1,2])
        y = np.transpose(y, [0,3,1,2])
            
        return X, y, fnames

        
class Raw2RGB_fullimage_testset(base_dataset.BaseDataset):
    
    def __init__(self,
                 dir_path = '../../../data/fi_dataset/testset/'):
        
        assert os.path.isdir(dir_path), 'dir not exist: ' + dir_path

        self.dir_path = dir_path
        self.test_scene_names = glob.glob(dir_path + '*.png')
        return
 
    def get_test_samples(self, num_samples = None):
        X = []
        fnames = []
        if num_samples is None:
            num_samples = len(self.test_scene_names)
            
        for scene in self.test_scene_names[:num_samples]:
            lin = utils.imread(scene).astype(np.float32) / 1023
            lin = utils.mosaic_then_demosaic(lin, 'grbg')
            lin = utils.srgb_gamma(lin) 
            #lin[0,:,:] /= 2.1
            #lin[1,:,:] /= 1.9
            #lin[2,:,:] /= 2.1
            lin -= 0.5

            X.append(lin)
            
            scene = os.path.basename(scene)[:-4]
            fnames.append(scene)
            
        X = np.stack(X)
        y = X.copy()
        
        return X, y, fnames

        
class Raw2RGB_fullimage_Dataset(base_dataset.BaseDataset):
    
    def __init__(self,
                 dir_path = '/data/eli/datasets/fi_dataset/orig/',
                 in_image = 'short_exposure1.png',
                 target_image = 'medium_exposure.jpg',
                 test_train_split = 0.1
                 ):
        
        assert os.path.isdir(dir_path), 'dir not exist: ' + dir_path

        self.dir_path = dir_path
        self.in_image = in_image
        self.target_image = target_image
        # have an independent random generator
        self.local_random = np.random.RandomState(76312735)
                
        scene_names = os.listdir(dir_path)
        self.train_scene_names, self.test_scene_names = train_test_split(scene_names, 
                                                                     test_size = test_train_split,
                                                                     random_state = self.local_random)
        return
    
    def get_training_samples(self, num_samples = None):
        X = []
        y = []
        if num_samples is None:
            num_samples = len(self.train_scene_names)
            
        for scene in self.train_scene_names[:num_samples]:
            lin = utils.imread(os.path.join(self.dir_path,scene,self.in_image)).astype(np.float32) / 1023
#            lin = lin[:,504:-504]
            lin = utils.mosaic_then_demosaic(lin, 'grbg')
            lin = utils.srgb_gamma(lin) - 0.5
#            lin = np.expand_dims(lin, 0)
            
            sam = utils.imread(os.path.join(self.dir_path,scene,self.target_image)).astype(np.float32) / 255
#            sam = sam[:,504:-504,:]
            sam -= 0.5
            sam = np.transpose(sam, [2,0,1])
            
            X.append(lin)
            y.append(sam)
            
        X = np.stack(X)
        y = np.stack(y)
        return X, y
        
    def get_test_samples(self, num_samples = None):
        X = []
        y = []
        fnames = []
        if num_samples is None:
            num_samples = len(self.test_scene_names)
            
        for scene in self.test_scene_names[:num_samples]:
            lin = utils.imread(os.path.join(self.dir_path,scene,self.in_image)).astype(np.float32) / 1023
#            lin = lin[:,504:-504]
            lin = utils.mosaic_then_demosaic(lin, 'grbg')
            lin = utils.srgb_gamma(lin) - 0.5
#            lin = np.expand_dims(lin, 0)
            
            sam = utils.imread(os.path.join(self.dir_path,scene,self.target_image)).astype(np.float32) / 255
#            sam = sam[:,504:-504,:]
            sam -= 0.5
            sam = np.transpose(sam, [2,0,1])
            
            X.append(lin)
            y.append(sam)
            fnames.append(scene)
            
        X = np.stack(X)
        y = np.stack(y)
        return X, y, fnames
        
class Raw2RGB_smallimage_Dataset(base_dataset.BaseDataset):
    
    def __init__(self,
                 dir_path = '../../../data/fi_dataset/small/',
#                 in_image = 'short_exposure1.png',
#                 target_image = 'medium_exposure.jpg',
                 test_train_split = 0.1
                 ):
        
        assert os.path.isdir(dir_path), 'dir not exist: ' + dir_path

        self.dir_path = dir_path
#        self.in_image = in_image
#        self.target_image = target_image
        # have an independent random generator
        self.local_random = np.random.RandomState(76312735)
        
        import glob

        scene_names = [os.path.basename(x) for x in glob.glob(dir_path + '/*_raw_low.png')]
        self.train_scene_names, self.test_scene_names = train_test_split(scene_names, 
                                                                     test_size = test_train_split,
                                                                     random_state = self.local_random)
        return
    
    def get_training_samples(self, num_samples = None):
        X = []
        y = []
        if num_samples is None:
            num_samples = len(self.train_scene_names)
            
        for scene in self.train_scene_names[:num_samples]:
            lin = io.imread(os.path.join(self.dir_path,scene)).astype(np.float32) / 255 - 0.5
            lin = np.transpose(lin, [2,0,1])
#            lin = lin[:,504:-504]
#            lin = utils.mosaic_then_demosaic(lin, 'grbg')
#            lin = np.expand_dims(lin, 0)
            
            sam = io.imread(os.path.join(self.dir_path, scene[:-12] + '_sam_normal.png')).astype(np.float32) / 255
#            sam = sam[:,504:-504,:]
            sam = utils.srgb_gamma_inv(sam)
            sam -= 0.5
            sam = np.transpose(sam, [2,0,1])
            
            X.append(lin)
            y.append(sam)
            
        X = np.stack(X)
        y = np.stack(y)
        return X, y
        
    def get_test_samples(self, num_samples = None):
        X = []
        y = []
        fnames = []
        if num_samples is None:
            num_samples = len(self.test_scene_names)
            
        for scene in self.test_scene_names[:num_samples]:
            lin = io.imread(os.path.join(self.dir_path,scene)).astype(np.float32) / 255 - 0.5
            lin = np.transpose(lin, [2,0,1])
#            lin = lin[:,504:-504]
#            lin = utils.mosaic_then_demosaic(lin, 'grbg')
#            lin = np.expand_dims(lin, 0)
            
            sam = io.imread(os.path.join(self.dir_path, scene[:-12] + '_sam_normal.png')).astype(np.float32) / 255
#            sam = sam[:,504:-504,:]
#            sam = utils.srgb_gamma_inv(sam) 
            sam -= 0.5
            sam = np.transpose(sam, [2,0,1])
            
            X.append(lin)
            y.append(sam)
            fnames.append(scene[:-12])
            
        X = np.stack(X)
        y = np.stack(y)
        return X, y, fnames

        
class Raw2RGB_Dataset(base_dataset.BaseDataset):
    
    def __init__(self,
                 dir_path = '../../../data/fi_dataset/color_dataset'
                 ):
        
        assert os.path.isdir(dir_path), 'dir not exist: ' + dir_path

        self.dir_path = dir_path
        

        return
    
    def get_training_samples(self):
        X = []
        y = []
        lin_dir = os.path.join(self.dir_path,'train','lin')
        sam_dir = os.path.join(self.dir_path,'train','sam')
        
        for fname in os.listdir(lin_dir):
            lin = io.imread(os.path.join(lin_dir,fname)).astype(np.float32) / 255 - 0.5
            lin = np.transpose(lin, [2,0,1])
            
            sam = io.imread(os.path.join(sam_dir,fname)).astype(np.float32) / 255
            sam = utils.srgb_gamma_inv(sam)
            sam -= 0.5
            sam = np.transpose(sam, [2,0,1])
            
            X.append(lin)
            y.append(sam)
            
        X = np.stack(X)
        y = np.stack(y)
        return X, y
        
    def get_test_samples(self):
        X = []
        y = []
        fnames = []
        lin_dir = os.path.join(self.dir_path,'test','lin')
        sam_dir = os.path.join(self.dir_path,'test','sam')
        
        for fname in os.listdir(lin_dir):
            lin = io.imread(os.path.join(lin_dir,fname)).astype(np.float32) / 350 - 0.5
            lin = np.transpose(lin, [2,0,1])
            
            sam = io.imread(os.path.join(sam_dir,fname)).astype(np.float32) / 255
            sam = utils.srgb_gamma_inv(sam)
            sam -= 0.5
            sam = np.transpose(sam, [2,0,1])
            
            X.append(lin)
            y.append(sam)
            fnames.append(fname)
            
        X = np.stack(X)
        y = np.stack(y)
        return X, y, fnames
        

        
class Grey2RGB_Dataset(base_dataset.BaseDataset):
    
    def __init__(self,
                 dir_path = '/data/datasets/PASCAL_VOC_Context/VOCdevkit/VOC2010/JPEGImages/',
                 training_patch_size = 256,
                 noise_type = 'gaussian',
                 std = 25,
                 receptive_field = 40, # for padding test set
                 test_train_split = 0.05, # 0 for only training, 1 for only test
                 ):
        
        assert os.path.isdir(dir_path), 'dir not exist: ' + dir_path
        assert noise_type in ('gaussian'), 'the requested noise type is not implemented (' + noise_type + ')'  

        self.dir_path = dir_path
        self.training_patch_size = training_patch_size
        self.noise_type = noise_type
        self.std = std
        self.receptive_field = receptive_field
        self.half_receptive_field = receptive_field//2
        
        # have an independent random generator
        self.local_random = np.random.RandomState(76312735)
                
        img_file_names = os.listdir(dir_path) #[f for f in os.listdir(dir_path) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.train_img_file_names, self.test_img_file_names = train_test_split(img_file_names, 
                                                                     test_size = test_train_split,
                                                                     random_state = self.local_random)
        self.cur_test_ind = 0
    
    def get_training_samples(self, num_samples = 1):
        Xs = []
        ys = []
        ind = self.local_random.randint(0, len(self.train_img_file_names)-1)
        rgb = io.imread(os.path.join(self.dir_path,self.train_img_file_names[ind])).astype(np.float32)
        if np.min(rgb.shape[0:2]) < self.training_patch_size:
            return np.empty([0]), np.empty([0])
        ycbcr = utils.rgb2ycbcr(rgb)
        ycbcr = utils.change_range(ycbcr, [0,255], [-0.5,0.5])
        ycbcr = np.transpose(ycbcr, [2,0,1])
        y = ycbcr[0:1,:,:]
        cbcr = ycbcr[1:3,:,:]
        while len(Xs) < num_samples:
            patch_ind_m = self.local_random.randint(0, rgb.shape[0] - self.training_patch_size)
            patch_ind_n = self.local_random.randint(0, rgb.shape[1] - self.training_patch_size)
            im_p = y[:,
                     patch_ind_m:patch_ind_m + self.training_patch_size,
                    patch_ind_n:patch_ind_n + self.training_patch_size]
            gt_p = cbcr[:,
                     patch_ind_m:patch_ind_m + self.training_patch_size,
                    patch_ind_n:patch_ind_n + self.training_patch_size]
#            gt_p = gt_p[:,
#                    self.half_receptive_field:-self.half_receptive_field,
#                    self.half_receptive_field:-self.half_receptive_field]
            Xs.append(im_p)
            ys.append(gt_p)
        Xs = np.stack(Xs)
        ys = np.stack(ys)
        return Xs, ys
        
    def get_test_samples(self, num_samples):
        raise NotImplementedError()
        
    def get_next_test_samples(self, num_samples = 1):     
        Xs = []
        ys = []
        fnames = []
        for _ in range(num_samples):
            fname = self.test_img_file_names[self.cur_test_ind]
            self.cur_test_ind = (self.cur_test_ind + 1)%len(self.test_img_file_names)
            full_fname = os.path.join(self.dir_path,fname)
            fnames.append(fname)
            rgb = io.imread(full_fname).astype(np.float32)
            rgb = rgb[:256,:256,:]
            ycbcr = utils.rgb2ycbcr(rgb)
            ycbcr = utils.change_range(ycbcr, [0,255], [-0.5,0.5])
            ycbcr = np.transpose(ycbcr, [2,0,1])
            y = ycbcr[0:1,:,:]
            cbcr = ycbcr[1:3,:,:]
            Xs.append(y)
            ys.append(cbcr)

        return Xs, ys, fname
        
        
        
class CorrectRGB_Dataset(base_dataset.BaseDataset):
    
    def __init__(self,
                 dir_path = '/data/datasets/PASCAL_VOC_Context/VOCdevkit/VOC2010/SegmentationObject/',
                 training_patch_size = 256,
                 noise_type = 'gaussian',
                 std = 25,
                 receptive_field = 40, # for padding test set
                 test_train_split = 0.05, # 0 for only training, 1 for only test
                 ):
        
        assert os.path.isdir(dir_path), 'dir not exist: ' + dir_path
        assert noise_type in ('gaussian'), 'the requested noise type is not implemented (' + noise_type + ')'  

        self.dir_path = dir_path
        self.training_patch_size = training_patch_size
        self.noise_type = noise_type
        self.std = std
        self.receptive_field = receptive_field
        self.half_receptive_field = receptive_field//2
        
        # have an independent random generator
        self.local_random = np.random.RandomState(76312735)
                
        img_file_names = os.listdir(dir_path) #[f for f in os.listdir(dir_path) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.train_img_file_names, self.test_img_file_names = train_test_split(img_file_names, 
                                                                     test_size = test_train_split,
                                                                     random_state = self.local_random)
        self.cur_test_ind = 0
    
    def get_training_samples(self, num_samples = 1):
        Xs = []
        ys = []
        ind = self.local_random.randint(0, len(self.train_img_file_names)-1)
        rgb = io.imread(os.path.join(self.dir_path,'../JPEGImages', self.train_img_file_names[ind][:-3] + 'jpg')).astype(np.float32)
        labels = io.imread(os.path.join(self.dir_path, self.train_img_file_names[ind]))[:,:,0]
        if np.min(rgb.shape[0:2]) < self.training_patch_size:
            return np.empty([0]), np.empty([0])
        labels[labels==255] = 0    
        uniq_labels = np.sort(np.unique(labels))
        ycbcr = utils.rgb2ycbcr(rgb)
        ycbcr = utils.change_range(ycbcr, [0,255], [-0.5,0.5])
        ycbcr = np.transpose(ycbcr, [2,0,1])
        # pointer to cb & cr channels
        cb, cr = ycbcr[1,:,:], ycbcr[2,:,:]
        cbcr_gt = ycbcr[1:3,:,:].copy()
        # altering the colors
        for val in uniq_labels:#[:-1]: # exclude boundaries which are labeld 255
            mask = labels == val
            cb[mask] += self.local_random.uniform(-0.1,0.1) * cb[mask] + self.local_random.uniform(-0.1,0.1) * cr[mask] + self.local_random.uniform(-0.1,0.1)
            cr[mask] += self.local_random.uniform(-0.1,0.1) * cb[mask] + self.local_random.uniform(-0.1,0.1) * cr[mask] + self.local_random.uniform(-0.1,0.1)
#        ycbcr = ycbcr.clip(-0.5,0.5)    
        while len(Xs) < num_samples:
            patch_ind_m = self.local_random.randint(0, rgb.shape[0] - self.training_patch_size)
            patch_ind_n = self.local_random.randint(0, rgb.shape[1] - self.training_patch_size)
            im_p = ycbcr[:,
                     patch_ind_m:patch_ind_m + self.training_patch_size,
                    patch_ind_n:patch_ind_n + self.training_patch_size]
            gt_p = cbcr_gt[:,
                     patch_ind_m:patch_ind_m + self.training_patch_size,
                     patch_ind_n:patch_ind_n + self.training_patch_size]
#            gt_p = gt_p[:,
#                    self.half_receptive_field:-self.half_receptive_field,
#                    self.half_receptive_field:-self.half_receptive_field]
            Xs.append(im_p)
            ys.append(gt_p)
        Xs = np.stack(Xs)
        ys = np.stack(ys)
        return Xs, ys
        
    def get_test_samples(self, num_samples):
        raise NotImplementedError()
        
    def get_next_test_samples(self, num_samples = 1):     
        Xs = []
        ys = []
        fnames = []
        for _ in range(num_samples):
            fname = self.test_img_file_names[self.cur_test_ind]
            self.cur_test_ind = (self.cur_test_ind + 1)%len(self.test_img_file_names)
            fnames.append(fname)
            full_fname = os.path.join(self.dir_path,fname)
            labels = io.imread(full_fname)[:,:,0]
            labels[labels==255] = 0    
            full_fname = os.path.join(self.dir_path,'../JPEGImages',fname[:-3] + 'jpg')
            rgb = io.imread(full_fname).astype(np.float32)
            rgb = rgb[:256,:256,:]
            labels = labels[:256,:256]
            ycbcr = utils.rgb2ycbcr(rgb)
            ycbcr = utils.change_range(ycbcr, [0,255], [-0.5,0.5])
            ycbcr = np.transpose(ycbcr, [2,0,1])
            cb, cr = ycbcr[1,:,:], ycbcr[2,:,:]
            cbcr_gt = ycbcr[1:3,:,:].copy()
            # altering the colors
            uniq_labels = np.sort(np.unique(labels))
            for val in uniq_labels:#[:-1]: # exclude boundaries which are labeld 255
                mask = labels == val
                cb[mask] += self.local_random.uniform(-0.1,0.1) * cb[mask] + self.local_random.uniform(-0.1,0.1) * cr[mask] + self.local_random.uniform(-0.1,0.1)
                cr[mask] += self.local_random.uniform(-0.1,0.1) * cb[mask] + self.local_random.uniform(-0.1,0.1) * cr[mask] + self.local_random.uniform(-0.1,0.1)
#            ycbcr = ycbcr.clip(-0.5,0.5) 
            Xs.append(ycbcr)
            ys.append(cbcr_gt)

        return Xs, ys, fname
        
        
class CorrectRGB_uniform_Dataset(base_dataset.BaseDataset):
    
    def __init__(self,
                 dir_path = '/data/datasets/PASCAL_VOC_Context/VOCdevkit/VOC2010/JPEGImages/',
                 training_patch_size = 256,
                 noise_type = 'gaussian',
                 std = 25,
                 receptive_field = 40, # for padding test set
                 test_train_split = 0.05, # 0 for only training, 1 for only test
                 ):
        
        assert os.path.isdir(dir_path), 'dir not exist: ' + dir_path
        assert noise_type in ('gaussian'), 'the requested noise type is not implemented (' + noise_type + ')'  

        self.dir_path = dir_path
        self.training_patch_size = training_patch_size
        self.noise_type = noise_type
        self.std = std
        self.receptive_field = receptive_field
        self.half_receptive_field = receptive_field//2
        
        # have an independent random generator
        self.local_random = np.random.RandomState(76312735)
                
        img_file_names = os.listdir(dir_path) #[f for f in os.listdir(dir_path) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.train_img_file_names, self.test_img_file_names = train_test_split(img_file_names, 
                                                                     test_size = test_train_split,
                                                                     random_state = self.local_random)
        self.cur_test_ind = 0
    
    def get_training_samples(self, num_samples = 1):
        Xs = []
        ys = []
        ind = self.local_random.randint(0, len(self.train_img_file_names)-1)
        rgb = io.imread(os.path.join(self.dir_path,'../JPEGImages', self.train_img_file_names[ind][:-3] + 'jpg')).astype(np.float32)
        labels = np.zeros(rgb.shape[:2], np.int32)#io.imread(os.path.join(self.dir_path, self.train_img_file_names[ind]))[:,:,0]
        if np.min(rgb.shape[0:2]) < self.training_patch_size:
            return np.empty([0]), np.empty([0])
            
        uniq_labels = np.sort(np.unique(labels))
        ycbcr = utils.rgb2ycbcr(rgb)
        ycbcr = utils.change_range(ycbcr, [0,255], [-0.5,0.5])
        ycbcr = np.transpose(ycbcr, [2,0,1])
        # pointer to cb & cr channels
        cb, cr = ycbcr[1,:,:], ycbcr[2,:,:]
        cbcr_gt = ycbcr[1:3,:,:].copy()
        # altering the colors
        for val in uniq_labels:#[:-1]: # exclude boundaries which are labeld 255
            mask = labels == val
            cb[mask] += self.local_random.uniform(-0.1,0.1) * cb[mask] + self.local_random.uniform(-0.1,0.1) * cr[mask] + self.local_random.uniform(-0.1,0.1)
            cr[mask] += self.local_random.uniform(-0.1,0.1) * cb[mask] + self.local_random.uniform(-0.1,0.1) * cr[mask] + self.local_random.uniform(-0.1,0.1)
#        ycbcr = ycbcr.clip(-0.5,0.5)    
        while len(Xs) < num_samples:
            patch_ind_m = self.local_random.randint(0, rgb.shape[0] - self.training_patch_size)
            patch_ind_n = self.local_random.randint(0, rgb.shape[1] - self.training_patch_size)
            im_p = ycbcr[:,
                     patch_ind_m:patch_ind_m + self.training_patch_size,
                    patch_ind_n:patch_ind_n + self.training_patch_size]
            gt_p = cbcr_gt[:,
                     patch_ind_m:patch_ind_m + self.training_patch_size,
                     patch_ind_n:patch_ind_n + self.training_patch_size]
#            gt_p = gt_p[:,
#                    self.half_receptive_field:-self.half_receptive_field,
#                    self.half_receptive_field:-self.half_receptive_field]
            Xs.append(im_p)
            ys.append(gt_p)
        Xs = np.stack(Xs)
        ys = np.stack(ys)
        return Xs, ys
        
    def get_test_samples(self, num_samples):
        raise NotImplementedError()
        
    def get_next_test_samples(self, num_samples = 1): 
        Xs = []
        ys = []
        fnames = []
        for _ in range(num_samples):
            fname = self.test_img_file_names[self.cur_test_ind]
            self.cur_test_ind = (self.cur_test_ind + 1)%len(self.test_img_file_names)
            fnames.append(fname)
#            full_fname = os.path.join(self.dir_path,fname)
#            labels = io.imread(full_fname)[:,:,0]
            full_fname = os.path.join(self.dir_path,'../JPEGImages',fname[:-3] + 'jpg')
            rgb = io.imread(full_fname)
            rgb = imresize(rgb, size = 256.0/min(rgb.shape[:2]))
            rgb = rgb.astype(np.float32)
            rgb = rgb[:256,:256,:]
            ycbcr = utils.rgb2ycbcr(rgb)
            ycbcr = utils.change_range(ycbcr, [0,255], [-0.5,0.5])
            ycbcr = np.transpose(ycbcr, [2,0,1])
            cb, cr = ycbcr[1,:,:], ycbcr[2,:,:]
            cbcr_gt = ycbcr[1:3,:,:].copy()
            
            cb += self.local_random.uniform(-0.1,0.1) * cb + self.local_random.uniform(-0.1,0.1) * cr + self.local_random.uniform(-0.1,0.1)
            cr += self.local_random.uniform(-0.1,0.1) * cb + self.local_random.uniform(-0.1,0.1) * cr + self.local_random.uniform(-0.1,0.1)

            Xs.append(ycbcr)
            ys.append(cbcr_gt)

        return Xs, ys, fname
        
class CorrectRGB_uniform_w_noise_Dataset(base_dataset.BaseDataset):
    
    def __init__(self,
                 dir_path = '/data/datasets/PASCAL_VOC_Context/VOCdevkit/VOC2010/JPEGImages/',
                 training_patch_size = 256,
                 noise_type = 'gaussian',
                 std = 25,
                 receptive_field = 40, # for padding test set
                 test_train_split = 0.05, # 0 for only training, 1 for only test
                 mosaiced = False,
                 quadratic_color_trans = False
                 ):
        
        assert os.path.isdir(dir_path), 'dir not exist: ' + dir_path
        assert noise_type in ('gaussian'), 'the requested noise type is not implemented (' + noise_type + ')'  

        self.dir_path = dir_path
        self.training_patch_size = training_patch_size
        self.noise_type = noise_type
        self.std = std
        self.receptive_field = receptive_field
        self.half_receptive_field = receptive_field//2
        self.mosaiced = mosaiced
        self.quadratic_color_trans = quadratic_color_trans
        
        # have an independent random generator
        self.local_random = np.random.RandomState(76312735)
                
        img_file_names = os.listdir(dir_path) #[f for f in os.listdir(dir_path) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.train_img_file_names, self.test_img_file_names = train_test_split(img_file_names, 
                                                                     test_size = test_train_split,
                                                                     random_state = self.local_random)
        self.cur_test_ind = 0
        

    def make_noisy(self, im):
        assert im.shape[0]==3
        noisy = im.copy()
        if self.std:
            noisy += (self.std / 255.0) * np.random.standard_normal(im.shape).astype(np.float32)
            
          
        color_trans = np.eye(3, 10 if self.quadratic_color_trans else 4, dtype=np.float32)
        color_trans +=  0.1 * np.random.standard_normal(color_trans.shape).astype(np.float32)
        
        noisy = noisy.reshape([3,-1])
        
        elements = [noisy, np.ones_like(noisy[0,:])]
        if self.quadratic_color_trans:
            for i in range(3):
                for j in range(i,3):
                    elements.append(noisy[i,:]*noisy[j,:])
        elements = np.vstack(elements)
        
        noisy = np.dot(color_trans,elements)
        noisy = noisy.reshape(im.shape)
        return noisy
        
    def get_training_samples(self, num_samples=None):
        Xs = []
        ys = []
#        fnames = []
        for ind in range(len(self.train_img_file_names)):
            if num_samples and len(Xs)==num_samples:
                break
            fname = self.train_img_file_names[ind]
#            fnames.append(fname)
            full_fname = os.path.join(self.dir_path,fname)
            rgb = io.imread(full_fname)
            
            # don't train on gray scale images
            if np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,0],rgb[:,:,2]):
                continue
            
            if self.training_patch_size is not None:
                if min(rgb.shape[:2])<self.training_patch_size:
                    continue
                scale = float(self.training_patch_size)/min(rgb.shape[:2])
                new_size = [round(scale*v) for v in rgb.shape[:2]]
                rgb = imresize(rgb, size = new_size)
            rgb = rgb.astype(np.float32)
            rgb = np.transpose(rgb, [2,0,1])
            if self.training_patch_size is not None:
                rgb = utils.get_central_pixels(rgb, [self.training_patch_size,self.training_patch_size])

            rgb = utils.change_range(rgb, [0,255], [-0.5,0.5])

            
            noisy = self.make_noisy(rgb)
            if self.mosaiced:
                noisy = utils.mosaic_then_demosaic(noisy)
                
            Xs.append(noisy)
            ys.append(rgb)
            
        if self.training_patch_size is not None:
            Xs = np.stack(Xs)
            ys = np.stack(ys)
        
        return Xs, ys
        
    def get_test_samples(self):
        Xs = []
        ys = []
        fnames = []
        for ind in range(len(self.test_img_file_names)):
            fname = self.test_img_file_names[ind]
            fnames.append(fname)
            full_fname = os.path.join(self.dir_path,fname)
            rgb = io.imread(full_fname)
            if min(rgb.shape[:2])<self.training_patch_size:
                continue
            if np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,0],rgb[:,:,2]):
                continue
            scale = float(self.training_patch_size)/min(rgb.shape[:2])
            new_size = [round(scale*v) for v in rgb.shape[:2]]
            rgb = imresize(rgb, size = new_size)
            rgb = rgb.astype(np.float32)
            rgb = np.transpose(rgb, [2,0,1])
            rgb = utils.get_central_pixels(rgb, [self.training_patch_size,self.training_patch_size])
#            rgb = rgb[:256,:256,:]
            rgb = utils.change_range(rgb, [0,255], [-0.5,0.5])
            
            
            noisy = self.make_noisy(rgb)
            if self.mosaiced:
                noisy = utils.mosaic_then_demosaic(noisy)
                
            Xs.append(noisy)    
            ys.append(rgb)

        Xs = np.stack(Xs)
        ys = np.stack(ys)
        return Xs, ys, fnames
        
    def get_next_test_samples(self, num_samples = 1): 
        raise NotImplementedError()
        

        
class CorrectRAW_uniform_Dataset(base_dataset.BaseDataset):
    
    def __init__(self,
                 dir_path = '/data/eli/datasets/fi_dataset/processed/',
                 training_patch_size = 256,
                 noise_type = 'gaussian',
                 std = 25,
                 receptive_field = 40, # for padding test set
                 test_train_split = 0.05, # 0 for only training, 1 for only test
                 ):
        
        assert os.path.isdir(dir_path), 'dir not exist: ' + dir_path
        assert noise_type in ('gaussian'), 'the requested noise type is not implemented (' + noise_type + ')'  

        self.dir_path = dir_path
        self.training_patch_size = training_patch_size
        self.noise_type = noise_type
        self.std = std
        self.receptive_field = receptive_field
        self.half_receptive_field = receptive_field//2
        
        # have an independent random generator
        self.local_random = np.random.RandomState(76312735)
                
#        img_file_names = os.listdir(dir_path) #[f for f in os.listdir(dir_path) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.train_img_file_names = os.listdir(dir_path + 'train')
        self.test_img_file_names = os.listdir(dir_path + 'test')
#        train_test_split(img_file_names, 
#                                                                     test_size = test_train_split,
#                                                                     random_state = self.local_random)
        self.cur_test_ind = 0
    
    def get_training_samples(self, num_samples = 1):
        Xs = []
        ys = []
        for _ in range(num_samples):
            ind = self.local_random.randint(0, len(self.train_img_file_names)-1)
            fname = self.train_img_file_names[ind]
            full_fname = os.path.join(self.dir_path,'train',fname,'medium_exposure.jpg')
            rgb = io.imread(full_fname)
            rgb = imresize(rgb,1024.0/rgb.shape[0])
            rgb = rgb.astype(np.float32)
            patch_ind_m = self.local_random.randint(0, rgb.shape[0] - self.training_patch_size)
            patch_ind_n = self.local_random.randint(0, rgb.shape[1] - self.training_patch_size)
            rgb = rgb[patch_ind_m:patch_ind_m+256,patch_ind_n:patch_ind_n+256,:]
#            rgb = rgb * 255.0 / rgb.max()
            ycbcr = utils.rgb2ycbcr(rgb)
            ycbcr = utils.change_range(ycbcr, [0,255], [-0.5,0.5])
            ycbcr = np.transpose(ycbcr, [2,0,1])
#            cb, cr = ycbcr[1,:,:], ycbcr[2,:,:]
            cbcr_gt = ycbcr[1:3,:,:].copy()
            
            r = io.imread(os.path.join(self.dir_path,'train',fname,'gt_r.png'))
            g = io.imread(os.path.join(self.dir_path,'train',fname,'gt_g.png'))
            b = io.imread(os.path.join(self.dir_path,'train',fname,'gt_b.png'))
            rgb = np.stack((r,g,b), axis=2)
            rgb = imresize(rgb,1024.0/rgb.shape[0])
            patch_ind_m = self.local_random.randint(0, rgb.shape[0] - self.training_patch_size)
            patch_ind_n = self.local_random.randint(0, rgb.shape[1] - self.training_patch_size)
            rgb = rgb[patch_ind_m:patch_ind_m+256,patch_ind_n:patch_ind_n+256,:]
            rgb = rgb.astype(np.float32)
            rgb = rgb / 1023.0
            rgb = rgb ** (1.0/2.2)
            rgb *= 255.0
            ycbcr = utils.rgb2ycbcr(rgb)
            y = ycbcr[:,:,0]
            y = 255*(y/255)**(2.2)
            ycbcr = utils.change_range(ycbcr, [0,255], [-0.5,0.5])
            ycbcr = np.transpose(ycbcr, [2,0,1])
            
            
            Xs.append(ycbcr)
            ys.append(cbcr_gt)
        Xs = np.stack(Xs)
        ys = np.stack(ys)
        return Xs, ys
        
    def get_test_samples(self, num_samples):
        raise NotImplementedError()
        
    def get_next_test_samples(self, num_samples = 1): 
        Xs = []
        ys = []
        fnames = []
        for _ in range(num_samples):
            fname = self.test_img_file_names[self.cur_test_ind]
            self.cur_test_ind = (self.cur_test_ind + 1)%len(self.test_img_file_names)
            fnames.append(fname)
            full_fname = os.path.join(self.dir_path,'test',fname,'medium_exposure.jpg')
            rgb = io.imread(full_fname)
            rgb = imresize(rgb,256.0/rgb.shape[0])
            rgb = rgb.astype(np.float32)
            rgb = rgb[:256,:256,:]
            ycbcr = utils.rgb2ycbcr(rgb)
            ycbcr = utils.change_range(ycbcr, [0,255], [-0.5,0.5])
            ycbcr = np.transpose(ycbcr, [2,0,1]).astype(np.float32)
#            cb, cr = ycbcr[1,:,:], ycbcr[2,:,:]
            cbcr_gt = ycbcr[1:3,:,:].copy()
            
            r = io.imread(os.path.join(self.dir_path,'test',fname,'gt_r.png'))
            g = io.imread(os.path.join(self.dir_path,'test',fname,'gt_g.png'))
            b = io.imread(os.path.join(self.dir_path,'test',fname,'gt_b.png'))
            rgb = np.stack((r,g,b), axis=2)
            rgb = imresize(rgb,256.0/rgb.shape[0])
            rgb = rgb[:256,:256,:]
            rgb = rgb.astype(np.float32)
            rgb = rgb / 1023.0
            rgb = rgb ** (1.0/2.2)
            rgb *= 255.0
            ycbcr = utils.rgb2ycbcr(rgb)
            ycbcr = utils.change_range(ycbcr, [0,255], [-0.5,0.5])
            ycbcr = np.transpose(ycbcr, [2,0,1]).astype(np.float32)
            
            
            Xs.append(ycbcr)
            ys.append(cbcr_gt)

        return Xs, ys, fname

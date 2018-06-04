# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 12:23:14 2016

@author: t-elshw
"""

OVERRIDE_OUTPUT_DIR = False


train_params = {
    'MAX_EPOCHS'                    :   700             ,
    'LEARNING_RATE_INITIAL'         :   5e-5             ,
    'BATCH_SIZE'                    :   2**0             ,
    'TRAIN_VALID_SPLIT'             :   0.1             ,
}

params = {
    'TRAIN_PARAMS'                  :   train_params    ,
    'IN_PATCH_SIZE'                 :   1024               ,
    'TEST_EVERY_N_EPOCHS'           :   train_params['MAX_EPOCHS']//10,
    'NOISE_TYPE'                    :   'Gaussian'      ,
    'SIGMA'                         :   25.0              ,
    }

    
#import sys
import os
import shutil
from common import utils, models, custom_losses
#from common import new_lasagne_features
import numpy as np
import theano
import lasagne as lsgn
import nolearn.lasagne as nl
#import cPickle as pickle
from sklearn.utils import shuffle
from threading import Thread, Event
import queue
from scipy.io import loadmat
from skimage import io, color, exposure
import matplotlib.pyplot as plt
from theano import tensor as T
from matplotlib import colors
from datasets import raw2rgb_dataset
import pickle
from lasagne.layers import Pool2DLayer as PoolLayer
from scipy.misc import imresize
import csv
import h5py
import time


cur_file_name = os.path.basename(__file__)[:-3]
output_dir = '../../../data/output/' + cur_file_name

if os.path.exists(output_dir) and OVERRIDE_OUTPUT_DIR:
    shutil.rmtree(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    

params_path = os.path.join(output_dir,'params.pickle')    
#net_path = os.path.join(output_dir,'net.pickle')  
training_stats_path = os.path.join(output_dir,'training_stats.csv') 
if os.path.exists(training_stats_path):
    with open(training_stats_path) as f:
        training_stats = [{k: float(v) for k, v in row.items()}
                for row in csv.DictReader(f, skipinitialspace=True)]
else:
    training_stats = []

    
#ds = raw2rgb_dataset.Raw2RGB_fullimage_Dataset(
#            dir_path = '/data/eli/datasets/fi_dataset/orig/',
#            in_image = 'short_exposure1.png',
#            target_image = 'medium_exposure.jpg',)
#ds = raw2rgb_dataset.CorrectRGB_uniform_w_noise_Dataset(std=0, quadratic_color_trans = True)
#        in_image='short_exposure1.png', target_image = 'short_exposure.jpg')

#Xt = Xt[:,:,:,504:-504]
#yt = yt[:,:,:,504:-504]

#Xt = Xt[:,:,:2048,:2048]
#yt = yt[:,:,:2048,:2048]
#X = np.random.standard_normal([10,3,256,256]).astype(np.float32)
#y = np.random.standard_normal([10,3,256,256]).astype(np.float32)
    
def test_net(nn, train_history=[]):
    
    epoch = len(train_history)
    if params['TEST_EVERY_N_EPOCHS'] and not epoch%params['TEST_EVERY_N_EPOCHS'] == 0:
        return     

#    testing_net.load_params_from(params_path)
    
    print('Starting testing on selected test images \n') 
    out_dir = os.path.join(output_dir, str(epoch))
    output_test_imgs_dir = os.path.join(out_dir, 'denoised_imgs')
    if not os.path.exists(output_test_imgs_dir):
        os.makedirs(output_test_imgs_dir)
    
    t = time.time()
    y_h = nn.predict(Xt)
    presition_lasted = time.time() - t
    print('Prediction on test set took {0:.2f}s ({1:.2f}s per image)'.format(
            presition_lasted, presition_lasted/Xt.shape[0]))

    y_h = y_h.reshape(Xt.shape)
    
    X_ = np.transpose(Xt, [0,2,3,1])
    y_ = np.transpose(yt, [0,2,3,1])
    y_h = np.transpose(y_h, [0,2,3,1])
  
    X_ = utils.change_range(X_, [-0.5,0.5], [0,1])
    y_ = utils.change_range(y_, [-0.5,0.5], [0,1])
    y_h = utils.change_range(y_h, [-0.5,0.5], [0,1])
    
    with open(out_dir + '/results.csv', 'w') as f:
        f.write('image,psnr\n')
        for i in range(X_.shape[0]):
            f.write(test_scene_names[i] + ',{}\n'.format(utils.calc_psnr(y_[i,...],y_h[i,...], max_val=1.0)))
    
        
    X_ = X_.clip(0, 1)
    y_ = y_.clip(0, 1)
    y_h = y_h.clip(0, 1)
    
#    X_ = utils.srgb_gamma(X_)
#    y_ = utils.srgb_gamma(y_)
#    y_h = utils.srgb_gamma(y_h)

    X_ = (255*X_).astype(np.uint8)
    y_ = (255*y_).astype(np.uint8)
    y_h = (255*y_h).astype(np.uint8)
    
    for i in range(X_.shape[0]):
#        io.imsave(
#            os.path.join(output_test_imgs_dir, test_scene_names[i] + '_noisy.png'), 
#            X_[i,...])
        io.imsave(
            os.path.join(output_test_imgs_dir, test_scene_names[i] + '_out.png'), 
            y_h[i,...])
#        io.imsave(
#            os.path.join(output_test_imgs_dir, test_scene_names[i] + '_gt.png'), 
#            y_[i,...])
    
    nn.save_params_to(out_dir + '/params.pickle')        
#    pickle.dump( nn, open(out_dir + '/net.pickle', "wb" ) )
    
        
        
on_epoch_finished = [
#        nl.handlers.SaveWeights(path = params_path, every_n_epochs = 10, pickle = False),
#        nl.handlers.RememberBestWeights()
        utils.SaveNetState(
            training_stats_path, 
            params_path,
            1),
        test_net
        ]
                 

class LoadPatchesBatchIterator(nl.BatchIterator):

    def transform(self, Xb, yb):
        if yb is None:
            return Xb, yb
        
        
        half = Xb.shape[0]//2
        Xb, yb = shuffle(Xb,yb)
        Xb[:half,...] = Xb[:half,:,:,::-1]  
        yb[:half,...] = yb[:half,:,:,::-1]  
#        Xb, yb = shuffle(Xb,yb)
#        Xb[:half,...] = Xb[:half,:,::-1,:]  
#        yb[:half,...] = yb[:half,:,::-1,:]  
        
        out_x = np.zeros(Xb.shape[:2] + (params['IN_PATCH_SIZE'], params['IN_PATCH_SIZE']), dtype=np.float32)
        out_y = np.zeros_like(out_x)
        for i in range(Xb.shape[0]):
            m = 2 * (np.random.randint(0,1 + Xb.shape[-2]-params['IN_PATCH_SIZE']) // 2)
            n = 2 * (np.random.randint(0,1 + Xb.shape[-1]-params['IN_PATCH_SIZE']) // 2)
            out_x[i,...] = Xb[i,:,m:m+params['IN_PATCH_SIZE'],n:n+params['IN_PATCH_SIZE']]
            out_y[i,...] = yb[i,:,m:m+params['IN_PATCH_SIZE'],n:n+params['IN_PATCH_SIZE']]
        return out_x, out_y#.reshape([out_y.shape[0], -1])
        
num_epochs_to_do = max(params["TRAIN_PARAMS"]["MAX_EPOCHS"] - len(training_stats), 0)

print('Loading dataset...')   
t = time.time()
hf = h5py.File('/data/eli/datasets/full_images.h5', 'r')
X = hf['X'][:]
y = hf['y'][:]
Xt = hf['Xt'][:]
yt = hf['yt'][:]

Ts = hf['train_oracle_color_trans'][:]
color_trans_init = np.median(Ts,axis=0)

test_scene_ids = hf['test_scene_ids'][:]
test_scene_names = [str(scene[0]) + '_' + str(scene[1]) for scene in test_scene_ids]
hf.close()
print('Finished Loading dataset (took {}s)'.format(int(time.time()-t)))


layers = models.denoising_and_coloring( (None,3,params['IN_PATCH_SIZE'],params['IN_PATCH_SIZE']),
                                        flatten=False, 
                                        num_denoise_layers=15,
                                        num_coloring_layers=3,
                                        color_trans_init=color_trans_init)

#from common import models
net = nl.NeuralNet(
    layers                  =   layers,#models.color_trans_estim((None,3,256,256)),

    update                  =   lsgn.updates.adam, #nesterov_momentum,
    update_learning_rate    =   theano.shared(utils.float32(params["TRAIN_PARAMS"]["LEARNING_RATE_INITIAL"])),
    #update_momentum         =   theano.shared(utils.float32(params["TRAIN_PARAMS"]["MOMENTUM_INITIAL"])),

#        objective = custom_losses.objective_with_vgg(vgg_alpha=1.0),
    objective_loss_function =   custom_losses.loss_Lab(vgg_alpha=0.5),
    
    regression              =   True,
    batch_iterator_train    =   LoadPatchesBatchIterator(batch_size = params["TRAIN_PARAMS"]["BATCH_SIZE"]),#LoadPatchesBatchIterator(batch_size = params["TRAIN_PARAMS"]["BATCH_SIZE"]),
    batch_iterator_test     =   LoadPatchesBatchIterator(batch_size = params["TRAIN_PARAMS"]["BATCH_SIZE"]),
    max_epochs              =   num_epochs_to_do,
    train_split             =   nl.TrainSplit(eval_size=params["TRAIN_PARAMS"]["TRAIN_VALID_SPLIT"]),
    on_epoch_finished       =   on_epoch_finished,
#    on_training_started     =   on_training_started,
    y_tensor_type           =   T.TensorType(theano.config.floatX, (False,)*4),
    check_input             =   False,
    verbose                 =   1,
    )

net.initialize()

if os.path.exists(params_path):
    print('Initializing with saved weights')  
    net.load_params_from(params_path)
net.train_history_ = training_stats

print('Starting training {} epochs\n'.format(num_epochs_to_do))
net.fit(X, y)
#else:
#print('Runing test set', flush = True)
#    Xt, yt, fnames = ds.get_test_samples()  
#    Xt, yt = ds.get_training_samples(3)
#    fnames = ['train0', 'train1', 'train2']

    
#### Testing net ##################

#layers = models.denoising_and_coloring((None,3,Xt.shape[-2],Xt.shape[-1]), 
#                                       flatten=False, num_denoise_layers=0,
#                                       num_coloring_layers=5)
#
#net = nl.NeuralNet(
#    layers                  =   layers,#models.color_trans_estim((None,3,256,256)),
#    update                  =   lsgn.updates.adam, #nesterov_momentum,
#    update_learning_rate    =   theano.shared(utils.float32(params["TRAIN_PARAMS"]["LEARNING_RATE_INITIAL"])),
#    regression              =   True,
#    check_input             =   False,
#    y_tensor_type           =   T.TensorType(theano.config.floatX, (False,)*4),
#    batch_iterator_test     =   nl.BatchIterator(batch_size = 1),
#    )
#net.initialize()
#net.load_params_from(params_path)

test_net(net)   
   

print('Done!\n')

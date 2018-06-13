#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:40:44 2017

@author: eli
"""

import theano
from theano import tensor as T
import numpy as np
import lasagne
import scipy.ndimage.filters as fi
from theano.sandbox.cuda.dnn import dnn_conv
from theano.tensor.signal import downsample
from common import models
import os
import pickle

def rgb2lab ( rgb ) :
#    rgb += 0.5 # assumin input is in [-0.5,0.5]
    rgb = T.clip(rgb, 0, 1)
#    rgb = T.switch(T.gt(rgb,0.04045), ((rgb+0.055)/1.055)**2.4, rgb/12.92)
    rgb *= 100
   
    x = rgb [:,0:1,:,:] * 0.4124 + rgb [:,1:2,:,:] * 0.3576 + rgb [:,2:3,:,:] * 0.1805
    y = rgb [:,0:1,:,:] * 0.2126 + rgb [:,1:2,:,:] * 0.7152 + rgb [:,2:3,:,:] * 0.0722
    z = rgb [:,0:1,:,:] * 0.0193 + rgb [:,1:2,:,:] * 0.1192 + rgb [:,2:3,:,:] * 0.9505

    x /= 95.047
    y /= 100.0
    z /= 108.883
    
    def f(X):
        return T.switch(T.gt(X,0.008856), X**(1/3), 7.787*X + 16/116)
        
    x = f(x)
    y = f(y)
    z = f(z)
    
    L = 116*y - 16
    a = 500*(x - y)
    b = 200*(y-z)

    return T.concatenate([L,a,b], axis=1)

class loss_Lab():
    def __init__(self, vgg_alpha=0.5):
        self.vgg_alpha = vgg_alpha
        
    def __call__(self, y_true, y_pred):
        lab_true = rgb2lab(y_true)
        lab_pred = rgb2lab(y_pred)

        loss = l1_loss(lab_true,lab_pred)
        
        if self.vgg_alpha:
            loss *= (1-self.vgg_alpha)
            loss += self.vgg_alpha * loss_MSSSIM(lab_true[:,[0],:,:], lab_pred[:,[0],:,:])
            
        return loss

class loss_for_ablation():
    def __init__(self, vgg_alpha=0.5):
        self.vgg_alpha = vgg_alpha
        
    def __call__(self, y_true, y_pred):

        loss = l1_loss(y_true,y_pred)
        
        if self.vgg_alpha:
            loss *= (1-self.vgg_alpha)
            loss += self.vgg_alpha * (  0.33 * loss_MSSSIM(y_true[:,[0],:,:], y_pred[:,[0],:,:]) +
                                        0.33 * loss_MSSSIM(y_true[:,[1],:,:], y_pred[:,[1],:,:]) +
                                        0.33 * loss_MSSSIM(y_true[:,[2],:,:], y_pred[:,[2],:,:]) )
            
        return loss
        
def loss_SSIM(y_true, y_pred): 
    
    # There are additional parameters for this function
    # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
    #   and cannot be used for learning
    
    ns = 5
    patches_true = T.nnet.neighbours.images2neibs(y_true, [ns, ns], mode = 'ignore_borders')
    patches_pred = T.nnet.neighbours.images2neibs(y_pred, [ns, ns], mode = 'ignore_borders')

    u_true = T.mean(patches_true, axis=-1)
    u_pred = T.mean(patches_pred, axis=-1)
    var_true = T.var(patches_true, axis=-1)
    var_pred = T.var(patches_pred, axis=-1)
    eps = 1e-9
    std_true = T.sqrt(var_true + eps) # adding eps to avoid very small number that results in gradient exploding
    std_pred = T.sqrt(var_pred + eps)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    
    ssim /= T.clip(denom, eps, np.inf)
    #ssim = T.switch(T.isnan(ssim), ssim, T.zeros_like(ssim)) #tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    
    return T.mean((1.0 - ssim) / 2.0)

    
def loss_MSSSIM(y_true, y_pred):
#    num_samples = -1#y_true.shape[0]
#    im_size = 128 #T.sqrt(y_true.shape[1]/3)
#    y_true = T.reshape(y_true, [-1,3,im_size,im_size])
#    y_pred = T.reshape(y_pred, [-1,3,im_size,im_size])
    num_scales = 2
     
    loss = loss_SSIM(y_true, y_pred)
    
    if num_scales==1:
        return loss
    
    # generate blur kernel
    blur_kernel_sz = 3
    blur_std = 2.0
    ker = np.zeros((blur_kernel_sz,blur_kernel_sz),dtype='float32')
    ker[blur_kernel_sz//2,blur_kernel_sz//2] = 1
    ker = fi.gaussian_filter(ker, sigma=blur_std)
    ker = np.expand_dims(np.expand_dims(ker,0),0)
#    cker = np.asanyarray(np.zeros((1,1,blur_kernel_sz,blur_kernel_sz)),dtype='float32')
#    for i in range(3):
#    cker[0][0] = ker
    blur_ker = theano.shared(ker,'kernels')
    
    
   
    for i in range(num_scales-1):
        # downsample by factor 2
        y_true = T.nnet.conv2d(y_true, blur_ker,
                                  subsample=(2,2),
                                  filter_flip=True,
                                  border_mode='half')
        y_pred = T.nnet.conv2d(y_pred, blur_ker,
                                  subsample=(2,2),
                                  filter_flip=True,
                                  border_mode='half')
#        y_true = y_true[:,:,0::2,0::2] + y_true[:,:,1::2,0::2] + y_true[:,:,0::2,1::2] + y_true[:,:,1::2,1::2]
#        y_true /= 4
#        y_pred = y_pred[:,:,0::2,0::2] + y_pred[:,:,1::2,0::2] + y_pred[:,:,0::2,1::2] + y_pred[:,:,1::2,1::2]
#        y_pred /= 4
        #T.signal.pool.pool_2d(y_true, (2,2), mode = 'average_exc_pad')
#        y_pred = T.signal.pool.pool_2d(y_pred, (2,2), mode = 'average_exc_pad')
        loss += loss_SSIM(y_true, y_pred)
    
    return loss
    
    
def l1_loss(a, b):
    return lasagne.objectives.aggregate(T.abs_(a - b))
    
    
class loss_MSSSIM_L2_class():
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        
    def __call__(self, y_true, y_pred):
        l2 = lasagne.objectives.aggregate(lasagne.objectives.squared_error(y_true,y_pred))
        msssim = loss_MSSSIM(y_true,y_pred)
    #    alpha = 0.1
        
        loss = (1-self.alpha)*l2 + self.alpha*msssim
        return loss
    
    
class loss_MSSSIM_L1_class():
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        
    def __call__(self, y_true, y_pred):
        l1 = lasagne.objectives.aggregate(l1_loss(y_true,y_pred))
        msssim = loss_MSSSIM(y_true,y_pred)
    #    alpha = 0.1
        
        loss = (1-self.alpha)*l1 + self.alpha*msssim
        return loss
        
def align_targets(predictions, targets):
    """Helper function turning a target 1D vector into a column if needed.
    This way, combining a network of a single output unit with a target vector
    works as expected by most users, not broadcasting outputs against targets.
    Parameters
    ----------
    predictions : Theano tensor
        Expression for the predictions of a neural network.
    targets : Theano tensor
        Expression or variable for corresponding targets.
    Returns
    -------
    predictions : Theano tensor
        The predictions unchanged.
    targets : Theano tensor
        If `predictions` is a column vector and `targets` is a 1D vector,
        returns `targets` turned into a column vector. Otherwise, returns
        `targets` unchanged.
    """
    if (getattr(predictions, 'broadcastable', None) == (False, True) and
            getattr(targets, 'ndim', None) == 1):
        targets = lasagne.utils.as_theano_expression(targets).dimshuffle(0, 'x')
    return predictions, targets
    
def huber_loss(predictions, targets, delta=0.1):
    """ Computes the huber loss between predictions and targets.
    .. math:: L_i = \\frac{(p - t)^2}{2},  |p - t| \\le \\delta
        L_i = \\delta (|p - t| - \\frac{\\delta}{2} ), |p - t| \\gt \\delta
    Parameters
    ----------
    predictions : Theano 2D tensor or 1D tensor
        Prediction outputs of a neural network.
    targets : Theano 2D tensor or 1D tensor
        Ground truth to which the prediction is to be compared
        with. Either a vector or 2D Tensor.
    delta : scalar, default 1
        This delta value is defaulted to 1, for `SmoothL1Loss`
        described in Fast-RCNN paper [1]_ .
    Returns
    -------
    Theano tensor
        An expression for the element-wise huber loss [2]_ .
    Notes
    -----
    This is an alternative to the squared error for
    regression problems.
    References
    ----------
    .. [1] Ross Girshick et al (2015):
           Fast RCNN
           https://arxiv.org/pdf/1504.08083.pdf
    .. [2] Huber, Peter et al (1964)
           Robust Estimation of a Location Parameter
           https://projecteuclid.org/euclid.aoms/1177703732
    """
    predictions, targets = align_targets(predictions, targets)
    abs_diff = abs(targets - predictions)
    ift = 0.5 * lasagne.objectives.squared_error(targets, predictions)
    iff = delta * (abs_diff - delta / 2.)
    return lasagne.objectives.aggregate(theano.tensor.switch(abs_diff <= delta, ift, iff))


def total_variation_loss(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)).sum()


class objective():
    """
    Default implementation of the NeuralNet objective.

    :param layers: The underlying layers of the NeuralNetwork
    :param loss_function: The callable loss function to use
    :param target: the expected output

    :param aggregate: the aggregation function to use
    :param deterministic: Whether or not to get a deterministic output
    :param l1: Optional l1 regularization parameter
    :param l2: Optional l2 regularization parameter
    :param get_output_kw: optional kwargs to pass to
                          :meth:`NeuralNetwork.get_output`
    :return: The total calculated loss
    """
    
    def __init__(self, tv=0,
              tv_layer_name = 'output_image',
              l2_on_blur = 0,
              blur_kernel_sz = 5,
              blur_std = 2.0,
              img_var_loss = 0.0):
        self.tv = tv
        self.tv_layer_name = tv_layer_name
        self.l2_on_blur = l2_on_blur
        self.img_var_loss = img_var_loss
        
        if l2_on_blur:
            # generate blur kernal
            ker = np.zeros((blur_kernel_sz,blur_kernel_sz),dtype='float32')
            ker[blur_kernel_sz//2,blur_kernel_sz//2] = 1
            ker = fi.gaussian_filter(ker, sigma=blur_std)
            cker = np.asanyarray(np.zeros((3,3,blur_kernel_sz,blur_kernel_sz)),dtype='float32')
            for i in range(3):
                cker[i][i] = ker
            self.blur_ker = theano.shared(cker,'kernels')
        return
    
    def __call__(self,
              layers,
              loss_function,
              target,
              aggregate=lasagne.objectives.aggregate,
              deterministic=False,
              l1=0,
              l2=0,
              get_output_kw=None):
        if get_output_kw is None:
            get_output_kw = {}
            
        output_layer = layers[-1]
        network_output = lasagne.layers.get_output(
            output_layer, deterministic=deterministic, **get_output_kw)
        
        loss = aggregate(loss_function(network_output, target))
    
        if l1:
            loss += lasagne.regularization.regularize_layer_params(
                layers.values(), lasagne.regularization.l1) * l1
        if l2:
            loss += lasagne.regularization.regularize_layer_params(
                layers.values(), lasagne.regularization.l2) * l2
        if self.tv:
            loss += aggregate(
                        total_variation_loss(
                            lasagne.layers.get_output(
                                layers[self.tv_layer_name], 
                                deterministic=deterministic, **get_output_kw)) * self.tv)
        
        if self.l2_on_blur:
            loss += aggregate(
                        T.sum((dnn_conv(lasagne.layers.get_output(
                                            layers[self.tv_layer_name], 
                                            deterministic=deterministic, **get_output_kw),
                                        self.blur_ker) - 
                               dnn_conv(lasagne.layers.get_output(
                                            layers['input'], 
                                            deterministic=deterministic, **get_output_kw),
                                        self.blur_ker))**2) * self.l2_on_blur)
        
        if self.img_var_loss:
            loss += aggregate(T.abs_(
                                T.var(lasagne.layers.get_output(
                                            layers[self.tv_layer_name], 
                                            deterministic=deterministic, **get_output_kw), [-2, -1]) - 
                                T.var(lasagne.layers.get_output(
                                            layers['input'], 
                                            deterministic=deterministic, **get_output_kw), [-2, -1])) * self.img_var_loss)
        return loss
    
    
class objective_with_vgg():
    """
    Default implementation of the NeuralNet objective.

    :param layers: The underlying layers of the NeuralNetwork
    :param loss_function: The callable loss function to use
    :param target: the expected output

    :param aggregate: the aggregation function to use
    :param deterministic: Whether or not to get a deterministic output
    :param l1: Optional l1 regularization parameter
    :param l2: Optional l2 regularization parameter
    :param get_output_kw: optional kwargs to pass to
                          :meth:`NeuralNetwork.get_output`
    :return: The total calculated loss
    """
    
    def __init__(self, vgg_alpha=0,
              vgg_layer_name = 'conv2_2',
              ):
        self.vgg_alpha = vgg_alpha
        self.vgg_layer_name = vgg_layer_name
        
        if vgg_alpha:
            if not os.path.isfile('vgg16.pkl'):
                os.system(u'wget -N https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl')
            d = pickle.load(open('vgg16.pkl','rb'), 
                            encoding='latin1'   # using latin1 to fix unpickling python2 in python3
                                                # as suggested in http://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
                            )
            self.vgg_weights = d['param values']
            self.vgg_weights_shared = []
        return
    
           
    def get_vgg_features(self,X):
        
        def conv(X, W, b, W_shape):
            X = T.nnet.conv2d(X, W,
                                 None, W_shape,
                                  subsample=(1,1),
                                  filter_flip=True,
                                  border_mode='half')
            X += b #theano.shared(b).dimshuffle(('x', 0) + ('x',) * 2)
            X = T.nnet.relu(X)
            return X
        
        def pool(X):
            return downsample.max_pool_2d(X,
                                        ds=(2,2),
                                        st=(2,2),
                                        ignore_border=True,
                                        padding=(0, 0),
                                        mode='max',
                                        )
#        X = X[:,:,768:-768,768:-768]
        X = X[:,::-1,:,:] # rgb2bgr
        
        # block 1
        if len(self.vgg_weights_shared) < 2:
            self.vgg_weights_shared.append(theano.shared(self.vgg_weights[0])) # W
            self.vgg_weights_shared.append(theano.shared(self.vgg_weights[1]).dimshuffle(('x', 0) + ('x',) * 2) )  # b
        X = conv(X, self.vgg_weights_shared[0], self.vgg_weights_shared[1], self.vgg_weights[0].shape)
        if self.vgg_layer_name == 'conv1_1': return X
        
        if len(self.vgg_weights_shared) < 4:
            self.vgg_weights_shared.append(theano.shared(self.vgg_weights[2])) # W
            self.vgg_weights_shared.append(theano.shared(self.vgg_weights[3]).dimshuffle(('x', 0) + ('x',) * 2) )  # b
        X = conv(X, self.vgg_weights_shared[2], self.vgg_weights_shared[3], self.vgg_weights[2].shape)
        if self.vgg_layer_name == 'conv1_2': return X
        
        X = pool(X)
        if self.vgg_layer_name == 'pool1': return X
        
        # block 2
        if len(self.vgg_weights_shared) < 6:
            self.vgg_weights_shared.append(theano.shared(self.vgg_weights[4])) # W
            self.vgg_weights_shared.append(theano.shared(self.vgg_weights[5]).dimshuffle(('x', 0) + ('x',) * 2) )  # b
        X = conv(X, self.vgg_weights_shared[4], self.vgg_weights_shared[5], self.vgg_weights[4].shape)
        if self.vgg_layer_name == 'conv2_1': return X
        
        if len(self.vgg_weights_shared) < 8:
            self.vgg_weights_shared.append(theano.shared(self.vgg_weights[6])) # W
            self.vgg_weights_shared.append(theano.shared(self.vgg_weights[7]).dimshuffle(('x', 0) + ('x',) * 2) )  # b
        X = conv(X, self.vgg_weights_shared[6], self.vgg_weights_shared[7], self.vgg_weights[6].shape)
        if self.vgg_layer_name == 'conv2_2': return X
        
        X = pool(X)
        if self.vgg_layer_name == 'pool2': return X
        
#        # block 3
#        X = conv(X, self.vgg_weights[8], self.vgg_weights[9])
#        if self.vgg_layer_name == 'conv3_1': return X
#        
#        X = conv(X, self.vgg_weights[10], self.vgg_weights[11])
#        if self.vgg_layer_name == 'conv3_2': return X
#        
#        X = conv(X, self.vgg_weights[12], self.vgg_weights[13])
#        if self.vgg_layer_name == 'conv3_3': return X
#        
#        X = pool(X)
#        if self.vgg_layer_name == 'pool3': return X
#        
#        # block 4
#        X = conv(X, self.vgg_weights[14], self.vgg_weights[15])
#        if self.vgg_layer_name == 'conv4_1': return X
#        
#        X = conv(X, self.vgg_weights[16], self.vgg_weights[17])
#        if self.vgg_layer_name == 'conv4_2': return X
#        
#        X = conv(X, self.vgg_weights[18], self.vgg_weights[19])
#        if self.vgg_layer_name == 'conv4_3': return X
#        
#        X = pool(X)
#        if self.vgg_layer_name == 'pool4': return X
#        
#        # block 5
#        X = conv(X, self.vgg_weights[20], self.vgg_weights[21])
#        if self.vgg_layer_name == 'conv5_1': return X
#        
#        X = conv(X, self.vgg_weights[22], self.vgg_weights[23])
#        if self.vgg_layer_name == 'conv5_2': return X
#        
#        X = conv(X, self.vgg_weights[24], self.vgg_weights[25])
#        if self.vgg_layer_name == 'conv5_3': return X
#        
#        X = pool(X)
#        if self.vgg_layer_name == 'pool5': return X
        
        return X
    
    def __call__(self,
              layers,
              loss_function,
              target,
              aggregate=lasagne.objectives.aggregate,
              deterministic=False,
              l1=0,
              l2=0,
              get_output_kw=None):
        if get_output_kw is None:
            get_output_kw = {}
            
        output_layer = layers[-1]
        network_output = lasagne.layers.get_output(
            output_layer, deterministic=deterministic, **get_output_kw)
        if self.vgg_alpha:
            target_im = target[:,:network_output.shape[1]*network_output.shape[2]*network_output.shape[3]]
            target_im = T.reshape(target_im,network_output.shape)
        else:
            target_im = target
        loss = aggregate(loss_function(network_output, target_im))
    
        if l1:
            loss += lasagne.regularization.regularize_layer_params(
                layers.values(), lasagne.regularization.l1) * l1
        if l2:
            loss += lasagne.regularization.regularize_layer_params(
                layers.values(), lasagne.regularization.l2) * l2
               
        if self.vgg_alpha:
#            loss += self.vgg_alpha * aggregate(
#                                        T.mean((self.get_vgg_features(T.reshape(target,network_output.shape)) - 
#                                                self.get_vgg_features(network_output)
#                                              )**2))
            out_vgg = self.get_vgg_features(network_output)
            target_vgg = target[:,network_output.shape[1]*network_output.shape[2]*network_output.shape[3]:]
            target_vgg = T.reshape(target_vgg, out_vgg.shape)
            loss += self.vgg_alpha * aggregate(
                                        T.mean((target_vgg - out_vgg)**2))
        
     
        return loss

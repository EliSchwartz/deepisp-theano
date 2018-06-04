#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:58:10 2017

@author: eli
"""

from lasagne.layers import InputLayer, Upscale2DLayer, ConcatLayer, Conv2DLayer, FlattenLayer, BatchNormLayer, SliceLayer, ElemwiseSumLayer, \
        GaussianNoiseLayer, GlobalPoolLayer, ReshapeLayer, FeaturePoolLayer, ExpressionLayer, set_all_param_values, NonlinearityLayer, \
        DenseLayer, MaxPool2DLayer, BiasLayer
        
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import rectify, tanh, softmax
from theano import tensor as T
from common import custom_layers, utils
import os
import pickle
import numpy as np

def pretrained_vgg16(incoming, freeze = True):
    if not os.path.isfile('vgg16.pkl'):
        os.system(u'wget -N https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl')
    d = pickle.load(open('vgg16.pkl','rb'), 
                    encoding='latin1'   # using latin1 to fix unpickling python2 in python3
                                        # as suggested in http://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
                    )
    p = d['param values']
    if isinstance(incoming, (list,tuple)):
        incoming = InputLayer(incoming)
    net = {}
    net['input'] = SliceLayer(incoming, indices=slice(-1,None,-1), axis=1) # rgb2bgr
    net['conv1_1'] = ConvLayer(incoming, 64, 3, pad='same', W=p[0], b=p[1])
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad='same', W=p[2], b=p[3])
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad='same', W=p[4], b=p[5])
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad='same', W=p[6], b=p[7])
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad='same', W=p[8], b=p[9])
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad='same', W=p[10], b=p[11])
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad='same', W=p[12], b=p[13])
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad='same', W=p[14], b=p[15])
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad='same', W=p[16], b=p[17])
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad='same', W=p[18], b=p[19])
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad='same', W=p[20], b=p[21])
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad='same', W=p[22], b=p[23])
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad='same', W=p[24], b=p[25])
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
#    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
#    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
#    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
#    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    
    if freeze:
        for key,val in net.items():
            if not('pool' or 'input' in key):
                net[key].params[net[key].W].remove("trainable")
                net[key].params[net[key].b].remove("trainable")
                
#    set_all_param_values(net['pool5'], d['param values'][:-6])
    
    return net


def denoising(input_shape, pad = 'image', flatten = True, num_denoise_layers = 20):
    # pad = 'image' or 'features'
    
    
    in_layer = InputLayer(shape=input_shape, name= 'input')
#    in_layer = GaussianNoiseLayer(in_layer, sigma=0.1)
    
    
    if pad == 'image':
        in_layer = custom_layers.ReflectLayer(in_layer, width=num_denoise_layers)
#    else:
#        in_layer = custom_layers.ReflectLayer(in_layer, width=1)

    l_image = in_layer

    l_conv = Conv2DLayer(
            in_layer if pad == 'image' else custom_layers.ReflectLayer(in_layer, width=1), 
            num_filters = 64, 
            filter_size = (3,3), 
            stride=(1, 1), 
            pad = 'same' if pad == 'image' else 'valid',
            nonlinearity=None,
            name = 'denoiser_block0_conv')
        
    for i in range(num_denoise_layers-2):
        l_conv, l_image = custom_layers.gen_res_conv_layer(
            l_conv, l_image,
            num_filters = 64,
            pad = 'reflect',
            name = 'denoiser_block{}'.format(i+1))
    
    l_conv, l_image = custom_layers.gen_res_conv_layer(
            l_conv, l_image,
            num_filters = 3,
            pad = 'reflect',
            name = 'denoiser_block{}'.format(num_denoise_layers-1))
    
    _, l_image = custom_layers.gen_res_conv_layer(
            l_conv, l_image,
            num_filters = 0,
            pad = 'reflect')
    
    if pad == 'image':
        l_image = utils.cropLayer(l_image, num_denoise_layers)
    
    output_layer = l_image
    if flatten:
        output_layer = FlattenLayer(output_layer)
    return output_layer


def denoising_and_coloring(input_shape, flatten = True,
                            num_denoise_layers = 20,
                            num_coloring_layers = 5,
                            color_trans_init = np.eye(10,3, dtype=np.float32),
                            disconnect_ll_and_hl = False):
    
    in_layer = InputLayer(shape=input_shape, name= 'input')

    l_image = in_layer

    l_conv = Conv2DLayer(
            custom_layers.ReflectLayer(in_layer, width=1), 
            num_filters = 64, 
            filter_size = (3,3), 
            stride=(1, 1), 
            pad = 'valid',
            nonlinearity=None,
            name = 'denoiser_block0_conv')
        
    for i in range(num_denoise_layers):
        l_conv, l_image = custom_layers.gen_res_conv_layer(
            l_conv, l_image,
            num_filters = 64,
            pad = 'reflect',
            name = 'denoiser_block{}'.format(i+1))
    
    
    
#    hl = Conv2DLayer(l_conv, num_filters=64, filter_size=3, stride=2, pad='valid', nonlinearity=rectify,
#            name = 'hl_conv{}'.format(0))
    if disconnect_ll_and_hl:
        hl = l_image
    else:
        hl = l_conv
    for i in range(num_coloring_layers):
        hl = Conv2DLayer(hl, num_filters=64, filter_size=3, stride=2, pad='valid', nonlinearity=rectify,
            name = 'hl_conv{}'.format(i))
        hl = MaxPool2DLayer(hl, pool_size=2)
#    hl = Conv2DLayer(hl, num_filters=64, filter_size=3, stride=1, pad='valid', nonlinearity=rectify,
#            name = 'hl_conv{}'.format(num_coloring_layers+1))
#    hl = SliceLayer(hl, indices=0)
#    hl = SliceLayer(hl, indices=0)
    hl = GlobalPoolLayer(hl)

    img_with_2nd_order_elem = custom_layers.second_order_elements(l_image)
    transformation_params = DenseLayer(hl, num_units=img_with_2nd_order_elem.output_shape[1]*3, nonlinearity=None,
            name = 'transformation_params_flat')
    transformation_params = ReshapeLayer(transformation_params, [[0], img_with_2nd_order_elem.output_shape[1], 3],
            name = 'transformation_params')
#    transformation_params = ExpressionLayer(transformation_params, lambda X: 0*X) # TODO remove
    # adding the average over the training set
    transformation_params = BiasLayer(transformation_params, 
                                      b=color_trans_init, 
                                      shared_axes=[0])
    transformation_params.params[transformation_params.b].remove("trainable")
    
    output_layer = custom_layers.batched_tensordot(transformation_params, img_with_2nd_order_elem)
    
#    transformed = custom_layers.LinearTransPerPix((img_with_2nd_order_elem, transformation_params))
#    output_layer = ElemwiseSumLayer((l_image, transformed)) # skip connection
    if flatten:
        output_layer = FlattenLayer(output_layer)
    return output_layer

    
def iizuka_colorization(input_shape = (None, 1, 256, 256)):
    # implementing "Let there be Color" http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/
    # doing it with 256x256 input so I'll be able to turn it into fully convolutional
    # also, right now I'm ignoring the classification part
    in_layer = InputLayer(shape=input_shape)
    
    ll = Conv2DLayer(in_layer, num_filters=64, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=128, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=128, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=256, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=256, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    
    gl = Conv2DLayer(ll, num_filters=512, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=1024, filter_size=8, stride=1, pad='valid', nonlinearity=rectify) # FC that is implemented as conv
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=1, stride=1, pad='valid', nonlinearity=rectify) # FC that is implemented as conv
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=256, filter_size=1, stride=1, pad='valid', nonlinearity=rectify) # FC that is implemented as conv
    gl = BatchNormLayer(gl)
    
    ml = Conv2DLayer(ll, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ml = BatchNormLayer(ml)
    ml = Conv2DLayer(ml, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ml = BatchNormLayer(ml)
    
    gl_duplicated = Upscale2DLayer(gl, scale_factor=32) 
    fl = ConcatLayer((ml, gl_duplicated))
    fl = Conv2DLayer(fl, num_filters=256, filter_size=1, stride=1, pad='same', nonlinearity=rectify)
    fl = BatchNormLayer(fl)
    
    upsample = Conv2DLayer(fl, num_filters=128, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    upsample = BatchNormLayer(upsample)
    upsample = Upscale2DLayer(upsample, scale_factor=2)
    upsample = Conv2DLayer(upsample, num_filters=64, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    upsample = BatchNormLayer(upsample)
    upsample = Conv2DLayer(upsample, num_filters=64, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    upsample = BatchNormLayer(upsample)
    upsample = Upscale2DLayer(upsample, scale_factor=2)
    upsample = Conv2DLayer(upsample, num_filters=32, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    upsample = BatchNormLayer(upsample)
    
    output_layer = Conv2DLayer(upsample, num_filters=2, filter_size=3, stride=1, pad='same', nonlinearity=tanh)
    output_layer = Upscale2DLayer(output_layer, scale_factor=2)
    output_layer = FlattenLayer(output_layer)
    
    return output_layer

    
def iizuka_colorization_res(input_shape = (None, 3, 256, 256)):
    # implementing "Let there be Color" http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/
    # doing it with 256x256 input so I'll be able to turn it into fully convolutional
    # also, right now I'm ignoring the classification part
    in_layer = InputLayer(shape=input_shape)
    in_cbcr = SliceLayer(in_layer, indices=slice(1,3), axis=1)
    ll = Conv2DLayer(in_layer, num_filters=64, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=128, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=128, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=256, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=256, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    
    gl = Conv2DLayer(ll, num_filters=512, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=1024, filter_size=8, stride=1, pad='valid', nonlinearity=rectify) # FC that is implemented as conv
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=1, stride=1, pad='valid', nonlinearity=rectify) # FC that is implemented as conv
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=256, filter_size=1, stride=1, pad='valid', nonlinearity=rectify) # FC that is implemented as conv
    gl = BatchNormLayer(gl)
    
    ml = Conv2DLayer(ll, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ml = BatchNormLayer(ml)
    ml = Conv2DLayer(ml, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ml = BatchNormLayer(ml)
    
    gl_duplicated = Upscale2DLayer(gl, scale_factor=32) 
    fl = ConcatLayer((ml, gl_duplicated))
    fl = Conv2DLayer(fl, num_filters=256, filter_size=1, stride=1, pad='same', nonlinearity=rectify)
    fl = BatchNormLayer(fl)
    
    upsample = Conv2DLayer(fl, num_filters=128, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    upsample = BatchNormLayer(upsample)
    upsample = Upscale2DLayer(upsample, scale_factor=2)
    upsample = Conv2DLayer(upsample, num_filters=64, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    upsample = BatchNormLayer(upsample)
    upsample = Conv2DLayer(upsample, num_filters=64, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    upsample = BatchNormLayer(upsample)
    upsample = Upscale2DLayer(upsample, scale_factor=2)
    upsample = Conv2DLayer(upsample, num_filters=32, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    upsample = BatchNormLayer(upsample)
    
    output_layer = Conv2DLayer(upsample, num_filters=2, filter_size=3, stride=1, pad='same', nonlinearity=tanh)
    output_layer = Upscale2DLayer(output_layer, scale_factor=2)
    output_layer = ElemwiseSumLayer((output_layer,in_cbcr))
    output_layer = FlattenLayer(output_layer)
    
    return output_layer
    
    
    
def color_trans_estim(input_shape = (None, 3, 256, 256)):
    # implementing a model that estimates a 3X4 color matrix correction a nd applying it to the input image
    # doing it with 256x256 input so I'll be able to turn it into fully convolutional
    # 
    in_layer = InputLayer(shape=input_shape)
    in_cbcr = SliceLayer(in_layer, indices=slice(1,3), axis=1)
    ll = Conv2DLayer(in_layer, num_filters=64, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=128, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=128, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=256, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=256, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    ll = Conv2DLayer(ll, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    ll = BatchNormLayer(ll)
    
    gl = Conv2DLayer(ll, num_filters=512, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=3, stride=2, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=512, filter_size=3, stride=1, pad='same', nonlinearity=rectify)
    gl = BatchNormLayer(gl)
    gl = Conv2DLayer(gl, num_filters=1024, filter_size=8, stride=1, pad='valid', nonlinearity=rectify) # FC that is implemented as conv

    transformation_params = Conv2DLayer(gl, num_filters=6, filter_size=1, stride=1, pad='valid', nonlinearity=None) # FC that is implemented as conv
    transformation_params = Upscale2DLayer(transformation_params, scale_factor=256)
    
    output_layer = custom_layers.LinearTransPerPix_cbcr((in_cbcr, transformation_params))
    output_layer = FlattenLayer(output_layer)
    
    return output_layer
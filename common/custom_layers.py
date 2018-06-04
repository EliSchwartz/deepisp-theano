#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 13:52:00 2017

@author: eli
"""
import lasagne
from theano import tensor as T

## local linear transformation
#class _LinearTransPerPix(lasagne.layers.MergeLayer):
#    def __init__(self, inputs , other, **kwargs):
#        super(LinearTransPerPix, self).__init__([inputs, other], **kwargs)
#
#    def get_output_shape_for(self, input_shapes):
#        in_shape, weights_shape = input_shapes
#        return in_shape
#
#
#    def get_output_for(self, inputs):
#
#        in_im, transformation = inputs
#        other_input = T.repeat(other_input.dimshuffle(0, 'x', 1), lstm_input.shape[1], axis=1)  # repeat along time dimension
#
#        return  T.tanh( T.sum( lstm_input*other_input , axis=2 ) 

      
      
class batched_tensordot(lasagne.layers.MergeLayer):
    def __init__(self, incoming1, incoming2, batched_tensordot_axes=[[1],[1]], **kwargs):
        super(batched_tensordot, self).__init__([incoming1, incoming2], **kwargs)
        self.batched_tensordot_axes = batched_tensordot_axes
        # optional: check here that self.input_shapes make sense for a dot product
        # self.input_shapes will be populated by the super() call above
        
    def get_output_shape_for(self, input_shapes):
        # (rows of first input x columns of second input)
        axes_a = [self.input_shapes[0][i] for i in range(1,len(self.input_shapes[0])) if i not in self.batched_tensordot_axes[0]]
        axes_b = [self.input_shapes[1][i] for i in range(1,len(self.input_shapes[1])) if i not in self.batched_tensordot_axes[1]]
        
        return (self.input_shapes[0][0],) + tuple(axes_a) + tuple(axes_b)
        
    def get_output_for(self, inputs, **kwargs):
        return T.batched_tensordot(inputs[0], inputs[1], axes=self.batched_tensordot_axes)
        
class DotLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming1, incoming2, **kwargs):
        super(batched_tensordot, self).__init__([incoming1, incoming2], **kwargs)
        # optional: check here that self.input_shapes make sense for a dot product
        # self.input_shapes will be populated by the super() call above
        assert incoming1.output_shape[0] == incoming2.output_shape[0], "Batch dim don't match {}!={}".format(incoming1.output_shape[0],incoming2.output_shape[0])
        assert incoming1.output_shape[-1] == incoming2.output_shape[-2], "length of the dim summed over should match {}!={}".format(incoming1.output_shape[-1],incoming2.output_shape[-2])
        
    def get_output_shape_for(self, input_shapes):
        # (rows of first input x columns of second input)
        axes_a = list(self.input_shapes[0])
        axes_a.pop(-1)
        axes_b = list(self.input_shapes[1])
        axes_b.pop(0)
        axes_b.pop(-2)
        
        return tuple(axes_a + axes_b)
        
    def get_output_for(self, inputs, **kwargs):
        return T.dot(inputs[0], inputs[1], axes=self.batched_tensordot_axes)

        
def second_order_elements(in_im):
    d = in_im.output_shape[1]
    new_features = []
    new_features.append(in_im)
    for i in range(d):
        first_chan = lasagne.layers.SliceLayer(in_im, axis=1, indices=slice(i,(i+1)))
#        new_features.append(first_chan)
        for j in range(i,d):
            second_chan = lasagne.layers.SliceLayer(in_im, axis=1, indices=slice(j,(j+1)))
            new_feature = lasagne.layers.ElemwiseMergeLayer((first_chan, second_chan),T.mul)
            new_features.append(new_feature)
    new_features.append(lasagne.layers.ExpressionLayer(new_features[-1], lambda X: T.ones_like(X))) # adding a constant channel
    new_features = lasagne.layers.concat(new_features)
    return new_features
    
def LinearTransPerPix(inputs):
    in_im, transformation = inputs
    in_im_shape = in_im.output_shape
    transformation_shape = transformation.output_shape
    assert len(transformation_shape)==5
    M = transformation_shape[1] # in dim
    N = transformation_shape[2] # out dim 
    assert in_im_shape[1] == M

    scale = 1
    if in_im_shape[-2:] != transformation_shape[-2:]:
        assert float(in_im_shape[-2]) / in_im_shape[-1] == float(transformation_shape[-2]) / transformation_shape[-1]
        scale = float(in_im_shape[-1]) / transformation_shape[-1]
        assert scale == int(scale), 'scaling should be integer but is {}={}/{}'.format(scale,transformation_shape[-1],in_im_shape[-1])
        scale = int(scale)


#    biases = UpscaleBilinear2DLayer(lasagne.layers.SliceLayer(transformation, axis=1, indices=M), scale)
#    transformation = lasagne.layers.SliceLayer(transformation, axis=1, indices=slice(M))
    transformation = lasagne.layers.concat((transformation,)*in_im_shape[-1], axis=-1)
    transformation = lasagne.layers.concat((transformation,)*in_im_shape[-2], axis=-2)
    new_features = []
    for i in range(N):
        trans_row = lasagne.layers.SliceLayer(transformation, axis=2, indices=i)
        new_feature = lasagne.layers.FeaturePoolLayer(lasagne.layers.ElemwiseMergeLayer((trans_row, in_im),T.mul),pool_size=M, pool_function=T.sum)
        new_features.append(new_feature)
        
    
    new_features = lasagne.layers.concat(new_features)
#    new_features = lasagne.layers.ElemwiseSumLayer((new_features, biases))
    
    return new_features
    
    
    
def LinearTransPerPix_cbcr(inputs):
    in_im, transformation = inputs
    
    trans_cb = lasagne.layers.SliceLayer(transformation, axis=1, indices=slice(0,2))
    trans_cr = lasagne.layers.SliceLayer(transformation, axis=1, indices=slice(3,5))
    
    bias_cb = lasagne.layers.SliceLayer(transformation, axis=1, indices=slice(2,3))
    bias_cr = lasagne.layers.SliceLayer(transformation, axis=1, indices=slice(5,6))
    
    new_cb = lasagne.layers.FeaturePoolLayer(lasagne.layers.ElemwiseMergeLayer((trans_cb, in_im),T.mul),pool_size=2, pool_function=T.sum)
    new_cr = lasagne.layers.FeaturePoolLayer(lasagne.layers.ElemwiseMergeLayer((trans_cr, in_im),T.mul), pool_size=2, pool_function=T.sum)
    
    new_cb = lasagne.layers.ElemwiseSumLayer((new_cb, bias_cb))
    new_cr = lasagne.layers.ElemwiseSumLayer((new_cr, bias_cr))
    
    new_im = lasagne.layers.concat((new_cb,new_cr))
    new_im = lasagne.layers.ElemwiseSumLayer((in_im, new_im))
    return new_im
    

def WeightedSumLayer(features, weights):
    # features shape [batchsize, num_trans_params*d, m, n]
    # weights shape [batchsize, d, m, n]
    
    d = weights.output_shape[1]
    num_trans_params = int(features.output_shape[1]/d)
    
    # reshape features to 5 dimensions where the 2nd dimension is splitted to 2 dims
    features = lasagne.layers.ReshapeLayer(features,[[0],num_trans_params,d,[2],[3]])    
    # features shape [batchsize, num_trans_params, d, m, n]
    
    sliced_layers = []
    for i in range(num_trans_params):
        sliced_layer = lasagne.layers.SliceLayer(features, axis=1, indices=i)
        # sliced_layer shape [batchsize, d, m, n]
        sliced_layer = lasagne.layers.ElemwiseMergeLayer((sliced_layer,weights), merge_function=T.mul)
        sliced_layer = lasagne.layers.FeaturePoolLayer(sliced_layer, pool_size=d, axis=1, pool_function=T.sum)
#        sliced_layer = lasagne.layers.ReshapeLayer(sliced_layer, [[0],1,[1],[2],[3]])
        # sliced_layer shape [batchsize, 1, d, m, n]
        sliced_layers.append(sliced_layer)
    out = lasagne.layers.ConcatLayer(sliced_layers, axis=1)
#    features = lasagne.layers.ReshapeLayer(features,[[0],num_trans_params,d,[2],[3]])    
#    out = lasagne.layers.FeaturePoolLayer(weighted_features, pool_size=d, axis=1, pool_function=T.sum)
#    out = lasagne.layers.ReshapeLayer(out, [[0],[2],[3],
    return out
    

def PerPixelSoftmax(inputs):
    b,d,m,n = inputs.output_shape
    b_m_n_d = lasagne.layers.DimshuffleLayer(inputs, (0,2,3,1))
    bmn_d = lasagne.layers.ReshapeLayer(b_m_n_d,[-1,d])
    bmn_d = lasagne.layers.NonlinearityLayer(bmn_d, nonlinearity=lasagne.nonlinearities.softmax)
    b_m_n_d = lasagne.layers.ReshapeLayer(bmn_d,[b,m,n,d])
    b_d_m_n = lasagne.layers.DimshuffleLayer(b_m_n_d, (0,3,1,2))
    return b_d_m_n
    

    
    
class UpscaleBilinear2DLayer(lasagne.layers.Layer):
    """
    2D bilinear upsampling layer

    Performs 2D upsampling (using bilinear interpolation) over the two trailing
    axes of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    scale_factor : integer
        The scale factor in each dimension.

    use_1D_kernel : bool
        Upsample rows and columns separately using 1D kernels, otherwise
        use a 2D kernel.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    References
    -----
    .. [1] Augustus Odena, Vincent Dumoulin, Chris Olah (2016):
           Deconvolution and checkerboard artifacts. Distill.
           http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, incoming, scale_factor, use_1D_kernel=True, **kwargs):
        super(UpscaleBilinear2DLayer, self).__init__(incoming, **kwargs)
        self.scale_factor = scale_factor
        self.use_1D_kernel = use_1D_kernel

        if self.scale_factor < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))
        if isinstance(self.scale_factor, tuple):
            raise ValueError('Scale factor must be a scalar, not a tuple')

    def get_output_shape_for(self, input_shape):
        h = input_shape[2]*self.scale_factor \
            if input_shape[2] != None else None
        w = input_shape[3]*self.scale_factor \
            if input_shape[3] != None else None
        return input_shape[0:2] + tuple([h, w])

    def get_output_for(self, input, **kwargs):
        return T.nnet.abstract_conv.bilinear_upsampling(
            input,
            self.scale_factor,
            batch_size=self.input_shape[0],
            num_input_channels=self.input_shape[1],
            use_1D_kernel=self.use_1D_kernel)
        
        
def gen_res_conv_layer(conv_layer, image_layer,
              num_filters = 32,
              nonlinearity = lasagne.nonlinearities.rectify, pad = 'zeros',
              name = None):
               
#    image_layer = cropLayer(image_layer)
    num_channles = image_layer.output_shape[1]
    # update image
    res_img = lasagne.layers.SliceLayer(conv_layer, indices=slice(0,num_channles), axis=1)
    res_img = lasagne.layers.NonlinearityLayer(res_img, nonlinearity=lasagne.nonlinearities.tanh)
    res_img = lasagne.layers.ExpressionLayer(res_img, lambda X : X/10)
    image_layer = lasagne.layers.ElemwiseSumLayer((image_layer, res_img),
                        name = None if name is None else name + '_img')
    
    if num_filters:
        num_filters_prev = conv_layer.output_shape[1]
        conv_layer = lasagne.layers.SliceLayer(conv_layer, indices=slice(num_channles, num_filters_prev), axis=1)    
        conv_layer = lasagne.layers.NonlinearityLayer(conv_layer, nonlinearity=nonlinearity)
        
        conv_layer = lasagne.layers.concat((
                                         image_layer,
                                         conv_layer),
                                        axis=1)
        if pad == 'reflect':
            conv_layer = ReflectLayer(conv_layer, 1)
        
        conv_layer = lasagne.layers.Conv2DLayer(
            conv_layer,
            num_filters = num_filters, 
            filter_size = (3,3), 
            stride=(1, 1), 
            pad = 'same' if pad == 'zeros' else 'valid',
            nonlinearity=None,
            name = None if name is None else name + '_conv')
    else:
        conv_layer = None
        
    return conv_layer, image_layer


  
    
class Upscale2DLayer(lasagne.layers.Layer):
    """
    2D upscaling layer
    Performs 2D upscaling over the two trailing axes of a 4D input tensor.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    scale_factor : integer or iterable
        The scale factor in each dimension. If an integer, it is promoted to
        a square scale factor region. If an iterable, it should have two
        elements.
    mode : {'repeat', 'dilate'}
        Upscaling mode: repeat element values or upscale leaving zeroes between
        upscaled elements. Default is 'repeat'.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    Notes
    -----
    Using ``mode='dilate'`` followed by a convolution can be
    realized more efficiently with a transposed convolution, see
    :class:`lasagne.layers.TransposedConv2DLayer`.
    """

    def __init__(self, incoming, scale_factor, mode='repeat', **kwargs):
        super(Upscale2DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = lasagne.utils.as_tuple(scale_factor, 2)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

        if mode not in {'repeat', 'dilate'}:
            msg = "Mode must be either 'repeat' or 'dilate', not {0}"
            raise ValueError(msg.format(mode))
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[-2] is not None:
            output_shape[-2] *= self.scale_factor[0]
        if output_shape[-1] is not None:
            output_shape[-1] *= self.scale_factor[1]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b = self.scale_factor
        upscaled = input
        if self.mode == 'repeat':
            if b > 1:
                upscaled = T.extra_ops.repeat(upscaled, b, -1)
            if a > 1:
                upscaled = T.extra_ops.repeat(upscaled, a, -2)
        elif self.mode == 'dilate':
            if b > 1 or a > 1:
                output_shape = self.get_output_shape_for(input.shape)
                upscaled = T.zeros(shape=output_shape, dtype=input.dtype)
                upscaled = T.set_subtensor(upscaled[:, :, ::a, ::b], input)
        return upscaled
    
    
def reflect_pad(x, width, batch_ndim=1):
    """
    Pad a tensor with a constant value.
    Parameters
    ----------
    x : tensor
    width : int, iterable of int, or iterable of tuple
        Padding width. If an int, pads each axis symmetrically with the same
        amount in the beginning and end. If an iterable of int, defines the
        symmetric padding width separately for each axis. If an iterable of
        tuples of two ints, defines a seperate padding width for each beginning
        and end of each axis.
    batch_ndim : integer
        Dimensions before the value will not be padded.
    """
    
    # Idea for how to make this happen: Flip the tensor horizontally to grab horizontal values, then vertically to grab vertical values
    # alternatively, just slice correctly
    input_shape = x.shape
    input_ndim = x.ndim

    output_shape = list(input_shape)
    indices = [slice(None) for _ in output_shape]

    if isinstance(width, int):
        widths = [width] * (input_ndim - batch_ndim)
    else:
        widths = width

    for k, w in enumerate(widths):
        try:
            l, r = w
        except TypeError:
            l = r = w
        output_shape[k + batch_ndim] += l + r
        indices[k + batch_ndim] = slice(l, l + input_shape[k + batch_ndim])

    # Create output array
    out = T.zeros(output_shape)
    
    # Vertical Reflections
    out=T.set_subtensor(out[:,:,:width,width:-width], x[:,:,width:0:-1,:])# out[:,:,:width,width:-width] = x[:,:,width:0:-1,:]
    out=T.set_subtensor(out[:,:,-width:,width:-width], x[:,:,-2:-(2+width):-1,:])#out[:,:,-width:,width:-width] = x[:,:,-2:-(2+width):-1,:]
    
    # Place X in out
    # out = T.set_subtensor(out[tuple(indices)], x) # or, alternative, out[width:-width,width:-width] = x
    out=T.set_subtensor(out[:,:,width:-width,width:-width],x)#out[:,:,width:-width,width:-width] = x
   
   #Horizontal reflections
    out=T.set_subtensor(out[:,:,:,:width],out[:,:,:,(2*width):width:-1])#out[:,:,:,:width] = out[:,:,:,(2*width):width:-1]
    out=T.set_subtensor(out[:,:,:,-width:],out[:,:,:,-(width+2):-(2*width+2):-1])#out[:,:,:,-width:] = out[:,:,:,-(width+2):-(2*width+2):-1]
    
    
    return out
    
class ReflectLayer(lasagne.layers.Layer):

    def __init__(self, incoming, width, batch_ndim=2, **kwargs):
        super(ReflectLayer, self).__init__(incoming, **kwargs)
        self.width = width
        self.batch_ndim = batch_ndim

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        if isinstance(self.width, int):
            widths = [self.width] * (len(input_shape) - self.batch_ndim)
        else:
            widths = self.width

        for k, w in enumerate(widths):
            if output_shape[k + self.batch_ndim] is None:
                continue
            else:
                try:
                    l, r = w
                except TypeError:
                    l = r = w
                output_shape[k + self.batch_ndim] += l + r
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return reflect_pad(input, self.width,  self.batch_ndim)
      

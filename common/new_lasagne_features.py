# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:45:07 2016

@author: Eli
"""
from lasagne import init
from lasagne.layers import Layer, BiasLayer
from lasagne.utils import floatX


class ScaleLayer(Layer):
    """
    lasagne.layers.ScaleLayer(incoming, scales=lasagne.init.Constant(1),
    shared_axes='auto', **kwargs)
    A layer that scales its inputs by learned coefficients.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    scales : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for the scale.  The scale
        shape must match the incoming shape, skipping those axes the scales are
        shared over (see the example below).  See
        :func:`lasagne.utils.create_param` for more information.
    shared_axes : 'auto', int or tuple of int
        The axis or axes to share scales over. If ``'auto'`` (the default),
        share over all axes except for the second: this will share scales over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    Notes
    -----
    The scales parameter dimensionality is the input dimensionality minus the
    number of axes the scales are shared over, which matches the bias parameter
    conventions of :class:`DenseLayer` or :class:`Conv2DLayer`. For example:
    >>> layer = ScaleLayer((20, 30, 40, 50), shared_axes=(0, 2))
    >>> layer.scales.get_value().shape
    (30, 50)
    """
    def __init__(self, incoming, scales=init.Constant(1), shared_axes='auto',
                 **kwargs):
        super(ScaleLayer, self).__init__(incoming, **kwargs)

        if shared_axes == 'auto':
            # default: share scales over all but the second axis
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes

        # create scales parameter, ignoring all dimensions in shared_axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ScaleLayer needs specified input sizes for "
                             "all axes that scales are not shared over.")
        self.scales = self.add_param(
            scales, shape, 'scales', regularizable=False)

    def get_output_for(self, input, **kwargs):
        axes = iter(range(self.scales.ndim))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes) for input_axis in range(input.ndim)]
        return input * self.scales.dimshuffle(*pattern)


def standardize(layer, offset, scale, shared_axes='auto'):
    """
    Convenience function for standardizing inputs by applying a fixed offset
    and scale.  This is usually useful when you want the input to your network
    to, say, have zero mean and unit standard deviation over the feature
    dimensions.  This layer allows you to include the appropriate statistics to
    achieve this normalization as part of your network, and applies them to its
    input.  The statistics are supplied as the `offset` and `scale` parameters,
    which are applied to the input by subtracting `offset` and dividing by
    `scale`, sharing dimensions as specified by the `shared_axes` argument.
    Parameters
    ----------
    layer : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    offset : Theano shared variable or numpy array
        The offset to apply (via subtraction) to the axis/axes being
        standardized.
    scale : Theano shared variable or numpy array
        The scale to apply (via division) to the axis/axes being standardized.
    shared_axes : 'auto', int or tuple of int
        The axis or axes to share the offset and scale over. If ``'auto'`` (the
        default), share over all axes except for the second: this will share
        scales over the minibatch dimension for dense layers, and additionally
        over all spatial dimensions for convolutional layers.
    Examples
    --------
    Assuming your training data exists in a 2D numpy ndarray called
    ``training_data``, you can use this function to scale input features to the
    [0, 1] range based on the training set statistics like so:
    >>> import lasagne
    >>> import numpy as np
    >>> training_data = np.random.standard_normal((100, 20))
    >>> input_shape = (None, training_data.shape[1])
    >>> l_in = lasagne.layers.InputLayer(input_shape)
    >>> offset = training_data.min(axis=0)
    >>> scale = training_data.max(axis=0) - training_data.min(axis=0)
    >>> l_std = standardize(l_in, offset, scale, shared_axes=0)
    Alternatively, to z-score your inputs based on training set statistics, you
    could set ``offset = training_data.mean(axis=0)`` and
    ``scale = training_data.std(axis=0)`` instead.
    """
    # Subtract the offset
    layer = BiasLayer(layer, -offset, shared_axes)
    # Do not optimize the offset parameter
    layer.params[layer.b].remove('trainable')
    # Divide by the scale
    layer = ScaleLayer(layer, floatX(1.)/scale, shared_axes)
    # Do not optimize the scales parameter
    layer.params[layer.scales].remove('trainable')
    return layer
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:47:56 2016

@author: t-elshw
"""
import os
import sys
#import cv2
from skimage import io, feature, color, exposure
from scipy.misc import imresize
import numpy as np
from sklearn.utils import shuffle
from sklearn import linear_model
import nolearn.lasagne as nl
import csv
import lasagne as lsgn
from collections import OrderedDict
from theano import tensor as T
import shutil
import pickle
import rawpy
#import inspect

#import matplotlib.pyplot as plt

def imread(filepath):
    if filepath.endswith('dng'):
        return rawpy.imread(filepath).raw_image
    else:
        return io.imread(filepath)
        
        

def is_ec2():
    import socket
    try:
        socket.gethostbyname('instance-data.ec2.internal.')
        return True
    except socket.gaierror:
        return False       

        
class TrainSplit(object):
    def __init__(self, eval_size):
        self.eval_size = eval_size

    def __call__(self, X, y, net):
        if self.eval_size:
            if isinstance(y, list):
                train_test_split(range())
                kf = KFold(len(y), round(1. / self.eval_size))
            else:
                kf = StratifiedKFold(y, round(1. / self.eval_size))

            train_indices, valid_indices = next(iter(kf))
            X_train = _sldict(X, train_indices)
            y_train = _sldict(y, train_indices)
            X_valid = _sldict(X, valid_indices)
            y_valid = _sldict(y, valid_indices)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = None, None

        return X_train, X_valid, y_train, y_valid        
        
def mosaic_then_demosaic(rgb, pattern = 'grbg'):
    from scipy.ndimage.filters import convolve
    rgb = np.squeeze(rgb)
    if rgb.ndim == 2: # if input is in CFA format, same code should work with just stacking the input
        rgb = np.stack([rgb]*3)
        
    in_shape = rgb.shape
    if in_shape[2]==3:
        rgb = np.transpose(rgb, [2,0,1])
    
    mask = np.zeros_like(rgb)
    if pattern == 'grbg':
        mask[0,0::2,1::2] = 1 # r
        mask[1,0::2,0::2] = 1 # g1
        mask[1,1::2,1::2] = 1 # g2
        mask[2,1::2,0::2] = 1 # b
    elif pattern == 'rggb':
        mask[0,0::2,0::2] = 1 # r
        mask[1,0::2,1::2] = 1 # g1
        mask[1,1::2,0::2] = 1 # g2
        mask[2,1::2,1::2] = 1 # b
    else:
        raise NotImplementedError
        
    
    H_G = np.asarray(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]]) / 4   # yapf: disable

    H_RB = np.asarray(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]) / 4  # yapf: disable

    rgb[0,...] = convolve(rgb[0,...] * mask[0,...], H_RB, mode = 'mirror')
    rgb[1,...] = convolve(rgb[1,...] * mask[1,...], H_G,  mode = 'mirror')
    rgb[2,...] = convolve(rgb[2,...] * mask[2,...], H_RB, mode = 'mirror')

    if in_shape[2]==3:
        rgb = np.transpose(rgb, [1,2,0])
    return rgb     

        
        
def rgb2y(rgb_img):
    return rgb_img[:,:,0]*0.299 + rgb_img[:,:,1]*0.587 + rgb_img[:,:,2]*0.114

def RGB2YCbCr_Layer(incoming):
    W = np.array([[.299, .587, .114],
                  [-.168736, -.331364, .5],
                  [.5, -.418688, -.081312]]).reshape([3,3,1,1]).astype(np.float32)
#    b = np.array([0, 128, 128]).astype(np.float32)
    l = lsgn.layers.Conv2DLayer(incoming,
                                num_filters = 3,
                                filter_size = 1,
                                W = W,
                                b = lsgn.init.Constant()
                                )
    l.params[l.W].remove("trainable")
    l.params[l.b].remove("trainable")
    
    return l

def rgb2ycbcr(rgb): # in (0,255) range
    channel_first = rgb.shape[0]==3 
    if channel_first:
        r = rgb[0,:,:]
        g = rgb[1,:,:]
        b = rgb[2,:,:]
    else:
        r = rgb[:,:,0]
        g = rgb[:,:,1]
        b = rgb[:,:,2]
    y = .299*r + .587*g + .114*b
    cb = 128 -.168736*r -.331364*g + .5*b
    cr = 128 +.5*r - .418688*g - .081312*b
    ycbcr = np.stack((y,cb,cr), 0 if channel_first else 2)
    return ycbcr

def ycbcr2rgb(ycbcr):
    channel_first = ycbcr.shape[0]==3 
    if channel_first:
        y = ycbcr[0,:,:]
        cb = ycbcr[1,:,:]
        cr = ycbcr[2,:,:]  
    else:
        y = ycbcr[:,:,0]
        cb = ycbcr[:,:,1]
        cr = ycbcr[:,:,2]    
    r = y + 1.402 * (cr-128)
    g = y - .34414 * (cb-128) -  .71414 * (cr-128)
    b = y + 1.772 * (cb-128)
    rgb = np.stack((r,g,b), 0 if channel_first else 2)
    return rgb

def srgb_gamma(im):
    # implements the gamma correction for srgb color space
    # according to https://en.wikipedia.org/wiki/SRGB
    
    im = im.astype(np.float32)
    if im.max() > 1:
        im /= 255

    a = .055
    threshold = .0031308
    factor = 12.92

    near_zero_mask = im<=threshold
    im[near_zero_mask] *= factor

    im[~near_zero_mask] **= 1/2.4
    im[~near_zero_mask] *= 1+a
    im[~near_zero_mask] -= a

    return im

def srgb_gamma_inv(im):
    # implements the inverse gamma correction for srgb color space
    # according to https://en.wikipedia.org/wiki/SRGB
    
    im = im.astype(np.float32)
    if im.max() > 1:
        im /= 255

    a = .055
    threshold = .04045
    factor = 12.92
    
    near_zero_mask = im<=threshold
    im[near_zero_mask] /= factor

    im[~near_zero_mask] += a
    im[~near_zero_mask] /= 1+a
    im[~near_zero_mask] **= 2.4

    return im

    
def anscombe(img):
    return 2 * np.sqrt(np.array(img) + 3.0/8)

def ianscombe(img):
    return (np.array(img) / 2) ** 2 - 3.0/8
    
def imnoise(img, params, return_clean = False):
    img = img.astype(np.float32)
    if not params["NOISE_TYPE"]:
        pass
    elif params["NOISE_TYPE"] == "Poisson":
        peak = np.random.uniform(params['POISSON_PEAK_MIN'], params['POISSON_PEAK_MAX'])
        img_max = img.max()
        img_max = max([img_max, 1.0])
        img *= (peak / img_max)
        noisy = np.random.poisson(img)
    else: # gaussian
        sigma = params['SIGMA']
        noisy = img + sigma * np.random.standard_normal(img.shape).astype(np.float32)
        
    if return_clean:
        return noisy, img
    else:
        return noisy
        
    
def read_image(img_path):
    img = io.imread(img_path, as_grey=True)
    if img.max() <= 1:
        img = 255 * img
    img = img.astype(np.float32)
    return img


def crop_search_win(img, row, col, half_search_win):
    min_row = max(0,row-half_search_win)
    max_row = min(img.shape[0],row+half_search_win)
    min_col = max(0,col-half_search_win)
    max_col = min(img.shape[1],col+half_search_win)
    return img[min_row:max_row,min_col:max_col]
        
def get_best_matching_patches(img, patch, Nsp):
#    sqdiff = cv2.matchTemplate(img, patch, cv2.TM_SQDIFF)
    corr = feature.match_template(img, patch)
    res = np.zeros([Nsp, patch.shape[0], patch.shape[1]]).astype(img.dtype)
    for i in range(Nsp):
        sub = np.unravel_index(np.argmax(corr), corr.shape)
#        sub = np.unravel_index(np.argmin(sqdiff), sqdiff.shape)
        res[i,:,:] = img[sub[0]:sub[0]+patch.shape[0],sub[1]:sub[1]+patch.shape[1]]
        corr[sub[0],sub[1]] = -1
#        sqdiff[sub[0],sub[1]] = np.inf
    return res

def change_range(x, orig_range, new_range):
    x_ = x.copy()
    x_ -= orig_range[0]
    x_ *= float(new_range[1] - new_range[0]) / (orig_range[1] - orig_range[0])
    x_ += new_range[0]
    return x_

def get_corr_heatmaps(patch, ker_sizes = (9, 13)):
    assert(len(patch.shape) == 2), "2D patch"
    assert(patch.shape[0] == patch.shape[1]), "square patch"
    assert(patch.shape[0] % 2 == 1), "odd patch size"
    
    patch_siz = patch.shape[0]
    half_patch = int(patch_siz / 2)
    res = np.zeros([len(ker_sizes) + 1, patch.shape[0], patch.shape[1]]).astype(patch.dtype)
    res[0 ,:, :] = patch
    for i, ker_siz in enumerate(ker_sizes):
        assert(ker_siz % 2 == 1), "odd kernel size"
        half_ker = int(ker_siz/2)
        ker = patch[half_patch - half_ker: half_patch + half_ker + 1, 
                    half_patch - half_ker: half_patch + half_ker + 1]
        corr = feature.match_template(patch, ker, pad_input = True)
        res[i + 1,:,:] = change_range(corr, [-1, 1], [0, 255])
    max_half_ker = max(ker_sizes) / 2
    res = res[:, 
              max_half_ker : -max_half_ker, 
              max_half_ker : -max_half_ker]
    return res


def load_image_patches(img, patch_size = 13, num_patches = None, noisy_img = None, Nsp = 1, sigma = 0):

    if isinstance(img, str): # otherwise assuming it's image
        img = read_image(img)
    # pad image
    img = np.pad(img, pad_width = int(patch_size/2), mode = 'reflect')
    
    h, w = img.shape
    
    noisy_img = img + sigma * np.random.standard_normal(img.shape).astype(np.float32)
    
    if num_patches is None:
        x = np.zeros([(h-patch_size+1)*(w-patch_size+1), Nsp * patch_size**2]).astype(np.float32)
        y = np.zeros([(h-patch_size+1)*(w-patch_size+1), patch_size**2]).astype(np.float32)
        ind = 0
        for i in range(h-patch_size+1):
            for j in range(w-patch_size+1):
                noisy_patchs = noisy_img[i: i + patch_size, j: j + patch_size]
                patch = img[i: i + patch_size, j: j + patch_size]
                if Nsp>1:
                    search_win = crop_search_win(noisy_img, i, j, 16)
                    noisy_patchs = get_best_matching_patches(search_win, noisy_patchs, Nsp).reshape([1,-1])
                x[ind,:] = noisy_patchs.reshape([1,-1])
                y[ind,:] = patch.reshape([1,-1])
                ind += 1
    else:
        x = np.zeros([num_patches, Nsp * patch_size**2]).astype(np.float32)
        y = np.zeros([num_patches, patch_size**2]).astype(np.float32)
        for i in range(num_patches):
            ii = np.random.randint(h-patch_size+1)
            jj = np.random.randint(w-patch_size+1)
            noisy_patch = noisy_img[ii: ii + patch_size, jj: jj + patch_size]
            patch = img[ii: ii + patch_size, jj: jj + patch_size]
            if Nsp>1:
                search_win = crop_search_win(noisy_img, ii, jj, 16)
                noisy_patch = get_best_matching_patches(search_win, noisy_patch, Nsp)
#            x[i,:] = get_best_matching_patches(noisy_img, patch, Nsp, noisy_img).reshape([1,-1])
            x[i,:] = noisy_patch.reshape([1,-1])
            y[i,:] = patch.reshape([1,-1])
#    y = x[:,:patch_size**2]
    return x, y

    
def get_central_pixels(img, out_shape):
    assert len(img.shape)>=2
    assert len(out_shape)==2 or len(out_shape)==len(img.shape), "out_shpae is " + str(len(out_shape)) + " dimensions"
    half_diff_shape = [int((img.shape[-2] - out_shape[-2]) / 2),
                       int((img.shape[-1] - out_shape[-1]) / 2)]
    return img[ ...,
                half_diff_shape[0] : half_diff_shape[0] + out_shape[0],
                half_diff_shape[1] : half_diff_shape[1] + out_shape[1]]

    
def load_image_patches_w_cor_heatmaps(
        img, 
        params,
        num_patches = None):
    in_patch_size       =   params["IN_PATCH_SIZE"]
    out_patch_size      =   params["OUT_PATCH_SIZE"]
    ker_sizes           =   params["SIMILIRATY_HEATMAPS_KER_SIZES"]

    if isinstance(img, str): # otherwise assuming it's image
        img = read_image(img)
    h, w = img.shape
    
#    noisy_img = img + sigma * np.random.standard_normal(img.shape).astype(np.float32)
    noisy_img = imnoise(img, params)
    
    padded_patch_size = in_patch_size + max(ker_sizes) - 1 if ker_sizes else in_patch_size
    padded_patch_size = int(padded_patch_size) # just to avoid warnings
    if num_patches is None: # all patches
        row_inds, col_inds = np.meshgrid(
            range(h-padded_patch_size+1), 
            range(w-padded_patch_size+1), 
            indexing='ij')
        row_inds = row_inds.reshape([-1,1])
        col_inds = col_inds.reshape([-1,1])
        num_patches = len(col_inds)
    else:
        row_inds = np.random.randint(
            low = 0,
            high = h-padded_patch_size+1,
            size = num_patches)
        col_inds = np.random.randint(
            low = 0,
            high = w-padded_patch_size+1,
            size = num_patches)
    coords = zip(row_inds, col_inds)    
    x = np.zeros([num_patches, (1 + len(ker_sizes)), in_patch_size, in_patch_size]).astype(np.float32)
    y = np.zeros([num_patches, out_patch_size**2]).astype(np.float32)
    ind = 0
    for (i,j) in coords:
        patch = get_central_pixels(
            img[i: i + padded_patch_size, j: j + padded_patch_size], 
            [out_patch_size, out_patch_size])
        noisy_patch = noisy_img[i: i + padded_patch_size, j: j + padded_patch_size]
        noisy_patchs = get_corr_heatmaps(noisy_patch, ker_sizes)
        x[ind,:] = noisy_patchs
        y[ind,:] = patch.reshape([1,-1])
        ind += 1

    return x, y

def load_patches_w_cor_heatmaps(
        params,
        training_set_path = '../../../data/BSDS300/images/train'
        ):
    in_patch_size       =   params["IN_PATCH_SIZE"]
    out_patch_size      =   params["OUT_PATCH_SIZE"]
    num_patches         =   params["NUM_PATCHES"]
    ker_sizes           =   params["SIMILIRATY_HEATMAPS_KER_SIZES"]

    num_imgs = len(os.listdir(training_set_path))
    num_patches_per_img = int(num_patches / num_imgs)
    X = np.empty((num_patches, (1 + len(ker_sizes)), in_patch_size, in_patch_size), np.float32)
    y = np.empty((num_patches, out_patch_size**2), np.float32)
    for (i, file) in enumerate(os.listdir(training_set_path)):
        X_, y_ = load_image_patches_w_cor_heatmaps(
                                    img = os.path.join(training_set_path,file),
                                    params = params,
                                    num_patches = num_patches_per_img)
        X[i*num_patches_per_img:(i+1)*num_patches_per_img,:] = X_ 
        y[i*num_patches_per_img:(i+1)*num_patches_per_img,:] = y_
        print ('Finished loading patches from image ' + str(i+1) + ' out of ' + str(num_imgs))
        sys.stdout.flush()
    X, y = shuffle(X, y, random_state=42)  # shuffle train data
    return X, y


def load(patch_size, 
         Nsp = 1, 
         num_patches = 1e6, 
         training_set_path = '../../../data/BSDS300/images/train',
         sigma = 0):
#    X, y = load_image_patches('Lena512.png')
    print('Perecent of patches loaded:')
    print('-' * 100)
    
    num_imgs = len(os.listdir(training_set_path))
    num_patches_per_img = int(num_patches / num_imgs)
    X = np.empty((num_patches,int(Nsp*patch_size**2)), np.float32)
    y = np.empty((num_patches,int(patch_size**2)), np.float32)
    for (i, file) in enumerate(os.listdir(training_set_path)):
        X_, y_ = load_image_patches(os.path.join(training_set_path,file),
                                    patch_size,
                                    num_patches_per_img, 
                                    Nsp = Nsp,
                                    sigma = sigma)
        X[i*num_patches_per_img:(i+1)*num_patches_per_img,:] = X_ 
        y[i*num_patches_per_img:(i+1)*num_patches_per_img,:] = y_
        
        percentage_loaded_change = int(100*(i+1)/num_imgs) - int(100*(i)/num_imgs)
        print("#" * percentage_loaded_change, end="", flush=True)
#    X = shuffle(X, random_state=42)  # shuffle train data
#    X, y = shuffle(X, y, random_state=42)  # shuffle train data
    print('\nFinished loading patches')
    return X, y
    
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

    
def load_image_patches_fi_dataset(
        input_imgs, 
        params, 
        num_patches = None, 
        num_channels = 1, 
        stride = 2
        ):
            
    patch_size = params["IN_PATCH_SIZE"]
    
    if isinstance(input_imgs, str):
        noisy_img = io.imread(os.path.join(input_imgs, params['FI_NET_INPUT_IMG'])).astype(np.float32)
        img = io.imread(os.path.join(input_imgs, params['FI_NET_GT_IMG'])).astype(np.float32)
    else:
        img = input_imgs["img"]
        noisy_img = input_imgs["noisy_img"]
        
    if params['FI_NET_INPUT_IMG'] == 'short_exposure.jpg':
        noisy_img = np.amax(noisy_img, axis=2) # V channel
        img = np.amax(img, axis=2) # V channel
        img = hist_match(img, noisy_img)

#    if num_patches:
#        noisy_img = imresize(noisy_img, 0.5, 'nearest')
#        img = imresize(img, 0.5, 'bicubic')
 
    
    # pad image
    img = np.pad(img, pad_width = int(patch_size/2), mode = 'reflect')
    noisy_img = np.pad(noisy_img, pad_width = int(patch_size/2), mode = 'reflect')
    h, w = img.shape[0], img.shape[1]
        
    if num_patches is None:
        x = np.zeros([(h-patch_size+1)*(w-patch_size+1)/(stride**2), num_channels, patch_size, patch_size]).astype(np.float32)
        y = np.zeros_like(x)
        ind = 0
        for i in range(0, h-patch_size+1, stride):
            for j in range(0, w-patch_size+1, stride):
                noisy_patchs = noisy_img[i: i + patch_size, j: j + patch_size, ...]
                patch = img[i: i + patch_size, j: j + patch_size, ...]
                x[ind,...] = noisy_patchs
                y[ind,...] = patch
                ind += 1
    else:
        x = np.zeros([num_patches, num_channels, patch_size, patch_size]).astype(np.float32)
        y = np.zeros_like(x)
        for i in range(num_patches):
            ii = stride * np.random.randint((h-patch_size+1)/stride)
            jj = stride * np.random.randint((w-patch_size+1)/stride)
            noisy_patch = noisy_img[ii: ii + patch_size, jj: jj + patch_size,...]
            patch = img[ii: ii + patch_size, jj: jj + patch_size,...]
            x[i,...] = noisy_patch
            y[i,...] = patch
    if params["SAVE_IMAGE_TYPE"]=='uint16':
        max_range = 1023
    else:
        max_range = 255    
    x = change_range(x, [0,max_range], [-params["NET_ABS_RANGE"], params["NET_ABS_RANGE"]])
    y = change_range(y, [0,max_range], [-params["NET_ABS_RANGE"], params["NET_ABS_RANGE"]])
    
    return x, y
    
def load_fi_dataset_train(
        params,
        num_patches = 1e6,
        training_set_path = '../../../data/FI_Dataset/train/',
        ):
            
    patch_size = params["IN_PATCH_SIZE"]
    num_channels = 1
#    X, y = load_image_patches('Lena512.png')
    print('Perecent of patches loaded:')
    print('-' * 100, flush=True)
    
    num_imgs = len(os.listdir(training_set_path))
    num_patches_per_img = int(num_patches / num_imgs)
    X = np.empty((num_patches,num_channels,patch_size,patch_size), np.float32)
    y = np.zeros_like(X)
    for (i, file) in enumerate(os.listdir(training_set_path)):
        X_, y_ = load_image_patches_fi_dataset(os.path.join(training_set_path,file),
                                    params,
                                    num_patches_per_img, 
                                    num_channels,
                                    )
        X[i*num_patches_per_img:(i+1)*num_patches_per_img,...] = X_ 
        y[i*num_patches_per_img:(i+1)*num_patches_per_img,...] = y_
        
        percentage_loaded_change = int(100*(i+1)/num_imgs) - int(100*(i)/num_imgs)
        print("#" * percentage_loaded_change, end="", flush=True)
#    X = shuffle(X, random_state=42)  # shuffle train data
#    X, y = shuffle(X, y, random_state=42)  # shuffle train data
    print('\nFinished loading patches', flush=True)
    return X, y
    
def std2psnr(std, max_val = 255.0):
    std = float(std)
    max_val = float(max_val)
    return 20.0 * np.log10(max_val / std)

def psnr2std(psnr, max_val = 255.0):
    psnr = float(psnr)
    max_val = float(max_val)
    return max_val / (10.0 ** (psnr / 20.0))
    
def calc_psnr(clean, noisy, max_val=255.0):
    if max_val is None:
        max_val = clean.max()
    diff2 = (noisy - clean) ** 2
    RMSE = np.sqrt(diff2.mean())
    PSNR = std2psnr(RMSE, max_val)
    return PSNR

def get_gaussian_2d(std, size):
    from scipy import signal
    gauss_1d = signal.get_window(('gaussian',std),size)
    gauss_1d = gauss_1d.reshape([1, size]) # make it 2d so 'dot' will perform matrix mult
    gauss_2d = np.dot(gauss_1d.transpose() ,gauss_1d )
    gauss_2d /= gauss_2d.sum()
    return gauss_2d
    
def reconst_image_from_patches(patches, img_shape, stride = 1):
    patch_size = int(np.sqrt(patches.shape[1]))
    assert patches.shape[0] == \
        len(range(0, img_shape[0]-patch_size+1, stride)) * \
        len(range(0, img_shape[1]-patch_size+1, stride)) , \
        "number of patches doesn't fit img_shape"
    img = np.zeros(img_shape,patches.dtype)
    weights = np.zeros(img_shape,patches.dtype)
    gaussian = get_gaussian_2d(float(patch_size)/4, patch_size)
    ind = 0
    for i in range(0, img_shape[0]-patch_size+1, stride):
        for j in range(0, img_shape[1]-patch_size+1, stride):
            img[i: i + patch_size, j: j + patch_size] += \
                gaussian * patches[ind,:].reshape([patch_size, patch_size])
            weights[i: i + patch_size, j: j + patch_size] += gaussian
            ind += 1
    img /= weights        
    return img
     
def test_net_single_img_old(net, input_img, get_img_patches_method, 
                        output_img_path = None, sigma = 25, net_abs_range = 0.1, Nsp = 1, patch_size = 13):
    if isinstance(input_img, str): # otherwise assuming it's image
        input_img = read_image(input_img)
        
        
    noisy_patches, _ = get_img_patches_method(input_img, sigma = sigma, Nsp = Nsp, patch_size = patch_size)
    noisy_patches = change_range(noisy_patches,[0,255],[-net_abs_range,net_abs_range])
    net_input_shape = net.get_all_layers()[0].shape
    net_input_shape = [-1 if v is None else v for v in net_input_shape] # replace none with -1
    noisy_patches = noisy_patches.reshape(net_input_shape)
    denoised_patches = net.predict(noisy_patches)
    denoised_patches = denoised_patches.reshape([denoised_patches.shape[0], -1])
    denoised_patches = denoised_patches[:, -patch_size**2:]
    denoised_img = reconst_image_from_patches(denoised_patches, input_img.shape)
    denoised_img = change_range(denoised_img,[-net_abs_range,net_abs_range],[0,255])
#    patch_size = int(np.sqrt(denoised_patches.shape[1]))
    psnr = calc_psnr(input_img[patch_size:-patch_size,patch_size:-patch_size], 
                    denoised_img[patch_size:-patch_size,patch_size:-patch_size],
                    255)
    if output_img_path is not None:
        denoised_clipped = np.minimum(np.maximum(denoised_img,0),255)
        denoised_img_uint = denoised_clipped.astype(np.uint8)
        io.imsave(output_img_path, denoised_img_uint)
#        cv2.imwrite(output_img_path, denoised_img_uint)
    return psnr, denoised_img

def find_bound_width(img_shape, num_pix_wo_bound):
    bound_width = 0
    new_img_shape = np.array(img_shape)
    num_pix = new_img_shape[0] * new_img_shape[1]
    while num_pix > num_pix_wo_bound :
        bound_width += 1
        new_img_shape -= 2
        num_pix = new_img_shape[0] * new_img_shape[1]
        
    assert(num_pix == num_pix_wo_bound), "couldn''t find an appropriate bound width img_shape={} num_pix_wo_bound={}".format(img_shape, num_pix_wo_bound)
    return bound_width
    

def test_net_single_img( 
       net, 
       input_img,
       output_img_path,
       params):
    if isinstance(input_img, str): # otherwise assuming it's image
        input_img = read_image(input_img)
    
    if params['USE_SIMILIRATY_HEATMAPS']:
        noisy_patches, _ = load_image_patches_w_cor_heatmaps(
                            img             =   input_img  , 
                            params          =   params     , 
                            num_patches     =   None       )
    elif params['USE_SIM_PATCHES']:
        noisy_patches, _ = load_image_patches(
                                    img = input_img,
                                    patch_size = params["IN_PATCH_SIZE"],
                                    num_patches = None, 
                                    Nsp = params['NUM_OF_SIM_PATCHES'],
                                    sigma = params['SIGMA'])
    else:
        assert(0)
    
    if params["NORM_NOISE_STD_TO_1"]:
        net_abs_range = 256.0 / (2*params["SIGMA"]) 
    elif params["NET_ABS_RANGE"]:
        net_abs_range = params["NET_ABS_RANGE"]
    else:
        1
#     
#    if params["NOISE_TYPE"] == 'Poisson':
#        noisy_patches[:,1:,:,:] = change_range(noisy_patches[:,1:,:,:], [0,255], [-net_abs_range, net_abs_range])
#        noisy_patches[:,0,:,:] = anscombe(noisy_patches[:,0,:,:])
#        noisy_patches[:,0,:,:] = change_range(
#                                    noisy_patches[:,0,:,:], 
#                                    anscombe([0, params['POISSON_PEAK']]), 
#                                    [-net_abs_range,net_abs_range])
#    elif not params["NO_NORMALIZATION"]:
#        noisy_patches = change_range(noisy_patches,[0,255],[-net_abs_range,net_abs_range])
    net_input_shape = net.get_all_layers()[0].shape
    net_input_shape = [-1 if v is None else v for v in net_input_shape] # replace none with -1
    noisy_patches = noisy_patches.reshape(net_input_shape)
    noisy_patches = change_range(noisy_patches, [0,255], [-net_abs_range,net_abs_range])
    denoised_patches = net.predict(noisy_patches)
    denoised_patches = change_range(denoised_patches, [-net_abs_range,net_abs_range], [0,255])
    denoised_patches = denoised_patches.reshape([denoised_patches.shape[0], -1])
    bound_width = find_bound_width(np.array(input_img.shape), denoised_patches.shape[0])
    denoised_img = reconst_image_from_patches(
                        denoised_patches, 
                        np.array(input_img.shape) - 2 * bound_width + params["OUT_PATCH_SIZE"] - 1)
    half_out_patch_size = int(params["OUT_PATCH_SIZE"]/2)
    if  half_out_patch_size:
        denoised_img = denoised_img[half_out_patch_size:-half_out_patch_size,
                                    half_out_patch_size:-half_out_patch_size]
    
#    if params["NOISE_TYPE"] == 'Poisson':
#        denoised_img = change_range(
#            denoised_img,
#            [-net_abs_range,net_abs_range], 
#            anscombe([0, params['POISSON_PEAK']]))
#        denoised_img = ianscombe(denoised_img)
#        denoised_img = change_range(
#            denoised_img,
#            [0, params['POISSON_PEAK']], 
#            [0, 255])
#    elif not params["NO_NORMALIZATION"]:
#        denoised_img = change_range(denoised_img,[-net_abs_range,net_abs_range],[0,255])
    input_img_bound_width = find_bound_width(
                                np.array(input_img.shape), 
                                denoised_img.shape[0]*denoised_img.shape[1])
    if input_img_bound_width:
        input_img = input_img[input_img_bound_width:-input_img_bound_width,
                               input_img_bound_width:-input_img_bound_width]
    psnr = calc_psnr(input_img, 
                    denoised_img,
                    255)
    if output_img_path is not None:
        if params['SAVE_IMAGE_TYPE'] == 'uint16':
            denoised_clipped = np.minimum(np.maximum(denoised_img,0),65535)
            denoised_img_uint = denoised_clipped.astype(np.uint16)
        else:
            denoised_clipped = np.minimum(np.maximum(denoised_img,0),255)
            denoised_img_uint = denoised_clipped.astype(np.uint8)
        io.imsave(output_img_path, denoised_img_uint)
#        cv2.imwrite(output_img_path, denoised_img_uint)
    return psnr, denoised_img 
    
def test_net(net,
             input_dir_path, 
             output_file_path, 
             output_imgs_path,
             params):
    with open(output_file_path,'w') as f:
        f.write('image_file_name,psnr\n')
    i = 1
    for file in os.listdir(input_dir_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            output_img_path = output_imgs_path + '/' + file if output_imgs_path else None
            print('Testing image number ' + str(i) + ' (' + file + ')\n')
            i += 1
            psnr, _ = test_net_single_img(
                net, 
                os.path.join(input_dir_path,file), 
                output_img_path,            
                params)
            with open(output_file_path,'a') as f:
                f.write(file + ',' + str(psnr) + '\n' )


def test_net_single_img_fi_dataset( 
       net, 
       input_scene_path,
       output_img_path,
       params):
    num_channels = 1
    net_abs_range = params["NET_ABS_RANGE"]
    stride = 2;
    
    img_shape = io.imread(os.path.join(input_scene_path, params['FI_NET_INPUT_IMG'])).shape

    noisy_patches, _ = load_image_patches_fi_dataset(
                            input_scene_path, 
                            params, 
                            num_patches = None, 
                            num_channels = num_channels, 
                            )
                                    
    net_input_shape = net.get_all_layers()[0].shape
    net_input_shape = [-1 if v is None else v for v in net_input_shape] # replace none with -1
    noisy_patches = noisy_patches.reshape(net_input_shape)
    denoised_patches = net.predict(noisy_patches)
    if params['SAVE_IMAGE_TYPE'] == 'uint16':
        denoised_patches = change_range(denoised_patches, [-net_abs_range,net_abs_range], [0,1023])
    else:
        denoised_patches = change_range(denoised_patches, [-net_abs_range,net_abs_range], [0,255])
    
    denoised_patches = denoised_patches.reshape([-1, params["OUT_PATCH_SIZE"]**2])
    bound_width = find_bound_width(np.array(img_shape[0:2]), denoised_patches.shape[0] * (stride**2))
    denoised_img = reconst_image_from_patches(
                        denoised_patches, 
                        np.array(img_shape[0:2]) - 2 * bound_width + params["OUT_PATCH_SIZE"] - 1,
                        stride)
                            
    half_out_patch_size = int(params["OUT_PATCH_SIZE"]/2)
    if  half_out_patch_size:
        denoised_img = denoised_img[half_out_patch_size:-half_out_patch_size,
                                    half_out_patch_size:-half_out_patch_size]
#    if num_channels == 1:
#        hsv = color.rgb2hsv(noisy_img.astype(np.uint8))
#        hsv[:,:,2] = denoised_img/255 #exposure.equalize_hist(denoised_img/255, 256)
#        denoised_img = 255 * color.hsv2rgb(hsv)
    if output_img_path is not None:
        if params['SAVE_IMAGE_TYPE'] == 'uint16':
            denoised_clipped = np.minimum(np.maximum(denoised_img,0),65535)
            denoised_img_uint = denoised_clipped.astype(np.uint16)
        else:
            denoised_clipped = np.minimum(np.maximum(denoised_img,0),255)
            denoised_img_uint = denoised_clipped.astype(np.uint8)
        
        io.imsave(output_img_path, denoised_img_uint)
#        cv2.imwrite(output_img_path, denoised_img_uint)
#    return psnr, denoised_img 

def test_net_fi_dataset( 
       net, 
       dataset_test_path,
       output_path,
       params):
    for (i, file) in enumerate(os.listdir(dataset_test_path)):
        test_net_single_img_fi_dataset(
            net, 
            input_scene_path = os.path.join(dataset_test_path, file),
            output_img_path = os.path.join(output_path, file + '.png'),
            params = params)






                

def float32(k):
    return np.cast['float32'](k)
        


        

class NeuralNetResidual(nl.NeuralNet):
    def fit(self, X, y, epochs=None):
        out_patch_size = int(np.sqrt(y.shape[1]))
        y = get_central_pixels(X[:,0,:,:], [out_patch_size,out_patch_size]).reshape(y.shape) - y
        return super(NeuralNetResidual, self).fit(X, y, epochs)
        
    def predict(self, X):
        noise = super(NeuralNetResidual, self).predict(X)
        noise_shape = list(noise.shape)
        out_patch_size = int(np.sqrt(noise_shape[1]))
        denoised = get_central_pixels(X[:,0,:,:], [out_patch_size,out_patch_size]).reshape(noise_shape) - noise
        return denoised
 
 
class NeuralNetWithGradClip(nl.NeuralNet):
    
    max_norm = 1000
    
    def set_max_norm(self, max_norm):
        self.max_norm = max_norm
        self.initialize()
        
    def _create_iter_funcs(self, layers, objective, update, output_type):
        import theano
        from theano import tensor as T
        from lasagne.layers import InputLayer
        from lasagne.layers import get_output
        from lasagne.updates import norm_constraint
        
        
        y_batch = output_type('y_batch')

        output_layer = layers[-1]
        objective_kw = self._get_params_for('objective')

        loss_train = objective(
            layers, target=y_batch, **objective_kw)
        loss_eval = objective(
            layers, target=y_batch, deterministic=True, **objective_kw)
        predict_proba = get_output(output_layer, None, deterministic=True)
        if not self.regression:
            predict = predict_proba.argmax(axis=1)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval

        all_params = self.get_all_params(trainable=True)
        grads = theano.grad(loss_train, all_params)
        if self.max_norm:
            grads = [norm_constraint(grad, self.max_norm, range(grad.ndim))
                for grad in grads]
        for idx, param in enumerate(all_params):
            grad_scale = getattr(param.tag, 'grad_scale', 1)
            if grad_scale != 1:
                grads[idx] *= grad_scale
        update_params = self._get_params_for('update')
        updates = update(grads, all_params, **update_params)

        input_layers = [layer for layer in layers.values()
                        if isinstance(layer, InputLayer)]

        X_inputs = [theano.In(input_layer.input_var, name=input_layer.name)
                    for input_layer in input_layers]
        inputs = X_inputs + [theano.In(y_batch, name="y")]

        train_iter = theano.function(
            inputs=inputs,
            outputs=[loss_train],
            updates=updates,
            allow_input_downcast=True,
            )
        eval_iter = theano.function(
            inputs=inputs,
            outputs=[loss_eval, accuracy],
            allow_input_downcast=True,
            )
        predict_iter = theano.function(
            inputs=X_inputs,
            outputs=predict_proba,
            allow_input_downcast=True,
            )

        return train_iter, eval_iter, predict_iter

class NeuralNetWithDataNormalization(nl.NeuralNet):
    def set_normaliztion_params(self, offset, scale):
        self.offset = offset
        self.scale = scale
        self.initialize()
    
    def predict(self, X):
        X -= self.offset
        X *= 1/self.scale
        y = super(NeuralNetWithDataNormalization, self).predict(X)
        if isinstance(y, tuple):
            y = np.concatenate(y)
        y = y.reshape([-1, y.shape[-1]])
        y *= self.scale
        y += self.offset
        return y
        
#    def fit(self, X, y, epochs = None):
#        X -= self.offset
#        X *= 1/self.scale
#        y -= self.offset
#        y *= 1/self.scale
#        super(NeuralNetWithDataNormalization, self).fit(X, y, epochs)
        
    

class AdjustVariable(object):
    def __init__(self, name, start=0.01, factor=0.5, update_every_n=1):
        self.name = name
        self.start, self.factor, self.update_every_n = start, factor, update_every_n
        self.ls = None

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch']
        value = float32(self.start * self.factor**(epoch//self.update_every_n))
        getattr(nn, self.name).set_value(value)

class EarlyStopping(object):
    def __call__(self, nn, train_history):
        current_epoch = train_history[-1]['epoch']
        if current_epoch >= nn.max_epochs:
            raise StopIteration()

       
class SaveNetState(object):    
    def __init__(self, training_stats_csv_path, 
                 params_pickle_path, n_epochs_to_save=1, 
                 net_pickle_path = None):
        self.training_stats_csv_path = training_stats_csv_path
        self.params_pickle_path = params_pickle_path
        self.n_epochs_to_save = n_epochs_to_save
        self.net_pickle_path = net_pickle_path
        self.keys = None
        sys.setrecursionlimit(7400) # to avoid problems with pickling the net
        return
        
    def __call__(self, nn, train_history):
        # sort keys to get constincy between runnings
        if self.keys is None: 
            soerted_train_history = OrderedDict(sorted(train_history[0].items(), key=lambda t: t[0]))
            self.keys = soerted_train_history.keys()

        # for the first epoch create csv file if it doesn't exist, in which case we only append new data
        if not os.path.exists(self.training_stats_csv_path):
            with open(self.training_stats_csv_path, 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, self.keys)
                dict_writer.writeheader()
                    
        if train_history and (0 == len(train_history)%self.n_epochs_to_save):
            dict_to_save = train_history[-self.n_epochs_to_save:]
            for i in range(len(dict_to_save)):
                dict_to_save[i]['valid_loss_best'] = float(dict_to_save[i]['valid_loss_best'])
                dict_to_save[i]['train_loss_best'] = float(dict_to_save[i]['train_loss_best'])
            with open(self.training_stats_csv_path, 'a') as output_file:
                dict_writer = csv.DictWriter(output_file, self.keys)
                dict_writer.writerows(dict_to_save)
            nn.save_params_to(self.params_pickle_path)
            if train_history[-1]['valid_loss_best']:
                shutil.copyfile(self.params_pickle_path, self.params_pickle_path[:-7] + '_best.pickle')
            if self.net_pickle_path:
                pickle.dump( nn, open( self.net_pickle_path, "wb" ) )
                if train_history[-1]['valid_loss_best']:
                    shutil.copyfile(self.net_pickle_path, self.net_pickle_path[:-7] + '_best.pickle')

        
class LoadNetState(object):    
    def __init__(self, training_stats_csv_path, params_pickle_path):
        self.training_stats_csv_path = training_stats_csv_path
        self.params_pickle_path = params_pickle_path
        return
        
    def __call__(self, nn, train_history):
        if os.path.exists(self.training_stats_csv_path):
            with open(self.training_stats_csv_path) as f:
                reader = csv.reader(f, skipinitialspace=True)
                header = next(reader)
                nn.train_history_ = [dict(zip(header, row)) for row in reader]
        if os.path.exists(self.params_pickle_path):
            nn.load_params_from(self.params_pickle_path)


def cropLayer(layer, crop_width = 1):
    layer = lsgn.layers.SliceLayer(layer, indices=slice(crop_width, -crop_width), axis=-1)
    layer = lsgn.layers.SliceLayer(layer, indices=slice(crop_width, -crop_width), axis=-2)
    return layer
    
    
#class PerPixelSoftMaxLayer(lsgn.layers.Layer):
#    """
#    
#    """
#
#    def __init__(self, incoming, **kwargs):
#        super(PerPixelSoftMaxLayer, self).__init__(incoming, **kwargs)
#
#    def get_output_shape_for(self, input_shape):
#        return input_shape
#
#    def get_output_for(self, input, **kwargs):
#        input_shape = self.input_shape
#        b, ch, r, c = input_shape
# 
#        input_exp = T.exp(input)
#        scale = input_exp[:, 0, :, :]
#        for i in range(1,ch):
#            scale += input_exp[:, i, :, :]
#        scale = T.tile(scale, [1, ch, 1, 1])
#        
#        input_exp /= scale
#
#        return input_exp   
        
def PerPixelSoftMaxLayer(l):
    chs = 2#l.input_shape[1]
    l = lsgn.layers.ExpressionLayer(l, lambda X: T.exp(X))
    l_scale = lsgn.layers.ExpressionLayer(l, lambda X: T.sum(X, axis=1))
    l_scale_tiled = lsgn.layers.concat((l_scale,l_scale) , axis = 1)
    l = lsgn.layers.ElemwiseMergeLayer((l,l_scale_tiled), T.true_div)
#    l = l_scale_tiled
    return l
    
        
#class ConstLayer(lsgn.layers.Layer):
#     def __init__(self, incoming, const_val = 1, **kwargs):
#        super(ConstLayer, self).__init__(incoming, **kwargs)
#        self.shared_const_array = None
#        self.const_val = const_val
#        
#    def get_output_for(self, input, **kwargs):
#        if self.shared_const_array is None:
#            self.shared_const_array = theano.shared(np.ones(input.shape, dtype=theano.config.floatX))
#        return input.sum(axis=-1)

def conv_input_length(output_length, filter_size, stride, pad=0):
    """Helper function to compute the input size of a convolution operation
    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.
    Parameters
    ----------
    output_length : int or None
        The size of the output.
    filter_size : int
        The size of the filter.
    stride : int
        The stride of the convolution operation.
    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        both borders.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.
    Returns
    -------
    int or None
        The smallest input size corresponding to the given convolution
        parameters for the given output size, or ``None`` if `output_size` is
        ``None``. For a strided convolution, any input size of up to
        ``stride - 1`` elements larger than returned will still give the same
        output size.
    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.
    Notes
    -----
    This can be used to compute the output size of a convolution backward pass,
    also called transposed convolution, fractionally-strided convolution or
    (wrongly) deconvolution in the literature.
    """
    if output_length is None:
        return None
    if pad == 'valid':
        pad = 0
    elif pad == 'full':
        pad = filter_size - 1
    elif pad == 'same':
        pad = filter_size // 2
    if not isinstance(pad, int):
        raise ValueError('Invalid pad: {0}'.format(pad))
    return (output_length - 1) * stride - 2 * pad + filter_size
    
class TransposedConv2DLayer(lsgn.layers.Conv1DLayer):
    """
    lasagne.layers.TransposedConv2DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), crop=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False, **kwargs)
    2D transposed convolution layer
    Performs the backward pass of a 2D convolution (also called transposed
    convolution, fractionally-strided convolution or deconvolution in the
    literature) on its input and optionally adds a bias and applies an
    elementwise nonlinearity.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
    num_filters : int
        The number of learnable convolutional filters this layer has.
    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.
    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        transposed convolution operation. For the transposed convolution, this
        gives the dilation factor for the input -- increasing it increases the
        output size.
    crop : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the transposed convolution is computed where the input and
        the filter overlap by at least one position (a full convolution). When
        ``stride=1``, this yields an output that is larger than the input by
        ``filter_size - 1``. It can be thought of as a valid convolution padded
        with zeros. The `crop` argument allows you to decrease the amount of
        this zero-padding, reducing the output size. It is the counterpart to
        the `pad` argument in a non-transposed convolution.
        A single integer results in symmetric cropping of the given size on all
        borders, a tuple of two integers allows different symmetric cropping
        per dimension.
        ``'full'`` disables zero-padding. It is is equivalent to computing the
        convolution wherever the input and the filter fully overlap.
        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.
        ``'valid'`` is an alias for ``0`` (no cropping / a full convolution).
        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.
    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).
        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_input_channels, num_filters, filter_rows, filter_columns)``.
        Note that the first two dimensions are swapped compared to a
        non-transposed convolution.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    flip_filters : bool (default: False)
        Whether to flip the filters before sliding them over the input,
        performing a convolution, or not to flip them and perform a
        correlation (this is the default). Note that this flag is inverted
        compared to a non-transposed convolution.
    output_size : int or iterable of int or symbolic tuple of ints
        The output size of the transposed convolution. Allows to specify
        which of the possible output shapes to return when stride > 1.
        If not specified, the smallest shape will be returned.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.
    b : Theano shared variable or expression
        Variable or expression representing the biases.
    Notes
    -----
    The transposed convolution is implemented as the backward pass of a
    corresponding non-transposed convolution. It can be thought of as dilating
    the input (by adding ``stride - 1`` zeros between adjacent input elements),
    padding it with ``filter_size - 1 - crop`` zeros, and cross-correlating it
    with the filters. See [1]_ for more background.
    Examples
    --------
    To transpose an existing convolution, with tied filter weights:
    >>> from lasagne.layers import Conv2DLayer, TransposedConv2DLayer
    >>> conv = Conv2DLayer((None, 1, 32, 32), 16, 3, stride=2, pad=2)
    >>> deconv = TransposedConv2DLayer(conv, conv.input_shape[1],
    ...         conv.filter_size, stride=conv.stride, crop=conv.pad,
    ...         W=conv.W, flip_filters=not conv.flip_filters)
    References
    ----------
    .. [1] Vincent Dumoulin, Francesco Visin (2016):
           A guide to convolution arithmetic for deep learning. arXiv.
           http://arxiv.org/abs/1603.07285,
           https://github.com/vdumoulin/conv_arithmetic
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 crop=0, untie_biases=False,
                 W=lsgn.init.GlorotUniform(), b=lsgn.init.Constant(0.),
                 nonlinearity=lsgn.nonlinearities.rectify, flip_filters=False,
                 output_size=None, **kwargs):
        # output_size must be set before calling the super constructor
        if (not isinstance(output_size, T.Variable) and
                output_size is not None):
            output_size = lsgn.utils.as_tuple(output_size, 2, int)
        self.output_size = output_size
        super(TransposedConv2DLayer, self).__init__(
                incoming, num_filters, filter_size, stride, crop, untie_biases,
                W, b, nonlinearity, flip_filters, n=2, **kwargs)
        # rename self.pad to self.crop:
        self.crop = self.pad
        del self.pad

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        # first two sizes are swapped compared to a forward convolution
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        if self.output_size is not None:
            size = self.output_size
            if isinstance(self.output_size, T.Variable):
                size = (None, None)
            return input_shape[0], self.num_filters, size[0], size[1]

        # If self.output_size is not specified, return the smallest shape
        # when called from the constructor, self.crop is still called self.pad:
        crop = getattr(self, 'crop', getattr(self, 'pad', None))
        crop = crop if isinstance(crop, tuple) else (crop,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_input_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, crop)))

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.crop == 'same' else self.crop
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=self.get_W_shape(),
            subsample=self.stride, border_mode=border_mode,
            filter_flip=not self.flip_filters)
        output_size = self.output_shape[2:]
        if isinstance(self.output_size, T.Variable):
            output_size = self.output_size
        elif any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]
        conved = op(self.W, input, output_size)
        return conved
     
def reinit_iter_funcs(net):
    if net is None:
        return
    # re-initialized iter_funcs
#    if getattr(net, '_initialized', False):
    net.initialize()
        
    iter_funcs = net._create_iter_funcs(
            net.layers_, net.objective, net.update,
            net.y_tensor_type,
            )
    net.train_iter_, net.eval_iter_, net.predict_iter_ = iter_funcs
            
def set_trainable(output_layer, trainable = False, nets = None):
    for layer in lsgn.layers.get_all_layers(output_layer):
        for param in layer.params:
            if isinstance(layer, lsgn.layers.BatchNormLayer) and (param.name in ('mean','inv_std')):
                # ignore the mean and std variables as they are not suppose to be trained
                continue
            if trainable:
                layer.params[param].add('trainable')
            else:             
                layer.params[param].discard('trainable')
    
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        reinit_iter_funcs(net)
    
def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.sum(targets * log_predictions, axis=1)
    
def labels2probs(labels):
    dim0 = labels.shape[0]
    dim1 = max(labels.max() + 1,2)
    probs = np.zeros([dim0,dim1], np.float32)
    probs[np.arange(len(probs)),labels] = 1
    return probs
    
def gen_res_conv_layer(conv_layer, image_layer,
              num_filters = 32,
              nonlinearity = lsgn.nonlinearities.rectify):
               
    image_layer = cropLayer(image_layer)
    num_channles = image_layer.output_shape[1]
    # update image
    res_img = lsgn.layers.SliceLayer(conv_layer, indices=slice(0,num_channles), axis=1)
    res_img = lsgn.layers.NonlinearityLayer(res_img, nonlinearity=lsgn.nonlinearities.tanh)
    res_img = lsgn.layers.ExpressionLayer(res_img, lambda X : X/10)
    image_layer = lsgn.layers.ElemwiseSumLayer((image_layer, res_img))
    
    if num_filters:
        num_filters_prev = conv_layer.output_shape[1]
        conv_layer = lsgn.layers.SliceLayer(conv_layer, indices=slice(num_channles, num_filters_prev), axis=1)    
        conv_layer = lsgn.layers.NonlinearityLayer(conv_layer, nonlinearity=nonlinearity)
        
        conv_layer = lsgn.layers.concat((
                                         image_layer,
                                         conv_layer),
                                        axis=1)
        
        conv_layer = lsgn.layers.Conv2DLayer(
            conv_layer,
            num_filters = num_filters, 
            filter_size = (3,3), 
            stride=(1, 1), 
            pad = 'valid',
            nonlinearity=None)
    else:
        conv_layer = None
        
    return conv_layer, image_layer
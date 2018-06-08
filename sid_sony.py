import os
import os.path

import numpy as np
# import torch.utils.data as data
# from PIL import Image
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
import glob
from common import utils
import rawpy
from tqdm import tqdm, trange
# import h5py

def pack_raw(raw):

    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = utils.mosaic_then_demosaic(im, 'gbrg')

    return im


class SID_Sony():
    """`Learning to see in the dark' Sony Dataset.

    Args:
        root (string): Root directory of dataset where directory ``Dataset_LINEAR_with_noise`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in a pair of PIL images
            and returns a transformed version. E.g, ``transforms.RandomCrop``

    """



    def __init__(self, root='/data/datasets/SID/Learning-to-See-in-the-Dark/dataset/Sony', subset='train', transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.subset = subset  # 'train', 'val' or 'test'

        self.input_dir = os.path.join(root,'short/')
        self.gt_dir = os.path.join(root,'long/')



        # get train and test IDs
        first_digit = {'train':'0', 'test':'1', 'val':'2'}

        self.fns = glob.glob(self.gt_dir + first_digit[subset] + '*.ARW')
        self.ids = []
        for i in range(len(self.fns)):
            _, fn = os.path.split(self.fns[i])
            self.ids.append(int(fn[0:5]))
        # self.ids = self.ids[:2]

        self.gt_images = []#[0] * len(self.ids)
        self.input_images = []
        self.gt_ind = []
        self.fnames = []

        # self.input_images = {}
        # self.input_images['300'] = [0] * len(self.ids)
        # self.input_images['250'] = [0] * len(self.ids)
        # self.input_images['100'] = [0] * len(self.ids)

        # save_data_path = os.path.join(root, subset + '.h5')
        # if os.path.exists(save_data_path):
        #     with h5py.File(save_data_path, 'r') as f:
        #         self.gt_images = f['gt_images'][:]
        #         self.input_images = f['input_images'][:]
        #         self.gt_ind = f['gt_ind'][:]
        #         self.fnames = f['fnames'][:]
        #     return


        print('Preparing dataset:\n')
        for ind in tqdm(range(len(self.ids))):
            train_id = self.ids[ind]

            gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % train_id)
            gt_path = gt_files[0]
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)


            in_files = glob.glob(self.input_dir + '%05d_00*.ARW' % train_id)
            for in_path in in_files:
            # in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
                self.gt_images.append(np.transpose(np.float32(im / 65535.0), (2, 0, 1)))
                _, in_fn = os.path.split(in_path)
                _, gt_fn = os.path.split(gt_path)
                in_exposure = float(in_fn[9:-5])
                gt_exposure = float(gt_fn[9:-5])
                ratio = min(gt_exposure / in_exposure, 300)
                # if self.input_images[str(ratio)[0:3]][ind] is None:
                raw = rawpy.imread(in_path)
                self.input_images.append(pack_raw(raw) * ratio)
                self.gt_ind.append(ind)
                self.fnames.append((in_fn.encode(),gt_fn.encode()))
        self.gt_images = np.stack(self.gt_images)
        self.input_images = np.stack(self.input_images)
        # with h5py.File(save_data_path, 'w') as f:
        #     f.create_dataset('gt_images',data=self.gt_images)
        #     f.create_dataset('input_images',data=self.input_images)
        #     f.create_dataset('gt_ind',data=self.gt_ind)
        #     f.create_dataset('fnames',data=self.fnames)





    def __getitem__(self, ind):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img = self.input_images[ind]
        target = self.gt_images[self.gt_ind[ind]]
        in_fn = self.fnames[ind][0].decode()

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target, in_fn

    def __len__(self):
        return len(self.input_images)

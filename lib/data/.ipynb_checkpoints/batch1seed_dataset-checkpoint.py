"""

Should output:
    - img: B X 3 X H X W

    - mask: B X H X W
    - pose: B X 7 (s, tr, q)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.linalg
import scipy.ndimage.interpolation
from skimage.io import imread
from absl import flags, app

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from lib.utils import image as image_utils
from lib.utils import transformations
from lib.utils.inversion_dist import *
import cv2
import os

import pycocotools.mask as mask_util


# -------------- Dataset ------------- #
# ------------------------------------ #
class Batch1SeedDataset(Dataset):


    def __init__(self, args, filter_key=None):
        self.args = args
        self.img_size = args.img_size
        self.jitter_frac = args.jitter_frac
        self.padding_frac = args.padding_frac
        self.filter_key = filter_key

        self.estimated_poses_path = osp.join('datasets', 'batch1seed_test','cache','poses_estimated_singletpl.bin')
        self.detections_path = osp.join('datasets', 'batch1seed_test','cache','detections.npy')

        self.estimated_poses = torch.load(self.estimated_poses_path)
        self.detections = np.load(self.detections_path, allow_pickle = True)
            
    def __len__(self):
        return len(self.estimated_poses['indices'])

    def __getitem__(self, index):
        id = self.estimated_poses['indices'][index]
        
        for row in self.detections:
            if row['id']==id:
                img_info = row
        
        
        pose = [self.estimated_poses['s'][id], self.estimated_poses['t'][id], self.estimated_poses['R'][id]]
        img_path = img_info['image_path']

        file_name = os.path.basename(img_path)
        if file_name.startswith('goodtest'):
            category = np.asarray(1)
            img_path = os.path.join('datasets', 'batch1seed_test', 'batch1seed','images','GoodSeed', file_name)
        else:
            category = np.asarray(2)
            img_path = os.path.join('datasets', 'batch1seed_test', 'batch1seed','images', 'BadSeed', file_name)
        
        img = imread(img_path)
        img = torch.from_numpy(img)
        img = torch.permute(img, (2,0,1))
        mask = mask_util.decode(img_info['mask']).astype(np.bool)
        mask_dts = image_utils.compute_dt_barrier(mask)
        
        basename = os.path.basename(img_path)
  
        img = img/255.0
    
        img = (img - 0.5)/0.5
        elem = {
            'idx': index,
            'img': img,
            'mask': mask,
            'sfm_pose': np.concatenate(pose),
            'inds': id,
            'img_path': img_path,
            'mask_dt': mask_dts,
            'class': category
        }

        return elem
    
def data_loader(args, shuffle=False):
    return base_loader(Batch1SeedDataset, args.batch_size, args, filter_key=None, shuffle=shuffle)
# ------------ Data Loader ----------- #
# ------------------------------------ #
def base_loader(d_set_func, batch_size, args, filter_key=None, shuffle=True):

    dset = d_set_func(args)

    return DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=False)
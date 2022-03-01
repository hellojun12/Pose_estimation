import torch
from torch.utils.data import Dataset
import os
import cv2
import json

import numpy as np
from glob import glob
import scipy.io

import copy

from transform import Transform

class LeadSportsDataset(Dataset):

    def __init__(self, is_train=True, **kwargs):
        super(LeadSportsDataset, self).__init__()
        
        self.img_list = self._load_img_list('data/lsp/images', is_train)
        self.anno_list = scipy.io.loadmat('data/lsp/joints.mat')
        
        self.len = len(self.img_list)
        self.hm_size = kwargs['output_size']
        self.data_name = kwargs['dataset']
        self.resize = kwargs['resize']
        self.n_landmarks = kwargs['n_landmarks']
        self.sigma = kwargs['sigma']

        self.transform = Transform(self.data_name, self.resize)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        anno_num = self._load_img_ID(img_path) - 1

        # Image Loading
        img = cv2.imread(img_path) #opencv (height width channel)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        org_size = img.shape[:2] #height ,width

        if self.transform != None:
            img = self.transform(image=img)['image']
        # Ground Truth
        heatmap = self._get_heatmaps_from_json(anno_num, org_size)
        
        # 이미지의 resizes는 transform에서 해주게 된다.
        return img, heatmap

    def __len__(self):
        return self.len

    def _load_img_list(self, data_root, is_train):
        # Change the name of directory which has inconsistent naming rule.
        full_img_list = glob(f'{data_root}/*')

        # ID < 1500 for Training
        # 500 < ID for Validation
        if is_train:
            return [path for path in full_img_list if (self._load_img_ID(path) < 1500)]

        else:
            return [path for path in full_img_list if (1500 < self._load_img_ID(path))]

    def _load_img_ID(self, path):
        return int(path.split(os.sep)[-1].lstrip('im').rstrip('.jpg'))

    def _get_heatmaps_from_json(self, anno_num, org_size):

        # Parse point annotation
        joints = copy.deepcopy(self.anno_list['joints'])
        joints = np.transpose(joints[:2, :, anno_num])

        joints[:,0] = joints[:,0] / org_size[1] * self.hm_size #x축
        joints[:,1] = joints[:,1] / org_size[0] * self.hm_size #y축

        heatmap = np.zeros((self.n_landmarks, self.hm_size, self.hm_size), dtype=np.float32)

        for i, jt in enumerate(joints):
          heatmap[i] = self._draw_labelmap(heatmap[i], jt, self.sigma)

        return heatmap

    def _draw_labelmap(self, heatmap, jt, sigma):
        # Draw a 2D gaussian
        # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
        H, W = heatmap.shape[:2]

        # Check that any part of the gaussian is in-bounds
        ul = [int(jt[0] - 3 * sigma), int(jt[1] - 3 * sigma)]
        br = [int(jt[0] + 3 * sigma + 1), int(jt[1] + 3 * sigma + 1)]
        if (ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] or
                br[0] < 0 or br[1] < 0):
            # If not, just return the image as is
            return heatmap, 0

        # Generate gaussian
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1

        g = np.exp(- ((x - x0)**2 + (y - y0)**2) / (2 * sigma **2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]

        # Image range
        heatmap_x = max(0, ul[0]), min(br[0], heatmap.shape[1])
        heatmap_y = max(0, ul[1]), min(br[1], heatmap.shape[0])

        heatmap[heatmap_y[0]:heatmap_y[1], heatmap_x[0]:heatmap_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return heatmap

def lsp(**kwargs):
    return LeadSportsDataset(**kwargs)

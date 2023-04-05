#!/usr/bin/env python
# coding:utf8
import nibabel as nib
import os
# import matplotlib.pyplot as plt
import numpy as np
import random
# from torchvision import  transforms as T
from PIL import Image
from torch.utils import data
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


###
class Brats17(data.Dataset):
    def __init__(self, root='/userhome/ZYQ/NPY_HGG/', transforms=None, train=True, test=False, val=False):
        self.test = test
        self.train = train
        self.val = val

        if self.train:
            self.root = root
            # self.root = '/userhome/ZYQ/nii2NPY/'  # directly
            # self.root = '/userhome/ZYQ/png2NPY/'
            self.folderlist = list(os.listdir(self.root))

        elif self.val:
            self.root = root
            self.folderlist = list(os.listdir(self.root))

        elif self.test:
            self.root = ''
            self.folderlist = os.listdir(os.path.join(self.root))

    def __getitem__(self, index):

        if self.train:
            if 1 > 0:
                ss = 96
                # print(self.folderlist[index])
                path = self.root
                img = np.load(os.path.join(path, self.folderlist[index]))
                img = np.asarray(img)
                # print(img.shape)
                index_x = np.random.randint(ss, img.shape[1] - ss, size=1)
                index_y = np.random.randint(ss, img.shape[2] - ss, size=1)
                # print(index_x,index_y,index_z)

                '''if random.random() < 0.8:
                    while np.sum(img[1, index_x[0] - ss:index_x[0] + ss, index_y[0] - ss:index_y[0] + ss] == 1.0) < 1:
                        index_x = np.random.randint(ss, img.shape[1] - ss, size=1)
                        index_y = np.random.randint(ss, img.shape[2] - ss, size=1)'''

                if np.max(img[0, :, :]) != 0 and np.max(img[1, :, :]) != 0 and np.max(img[2, :, :]) != 0 and np.max(img[3, :, :]) != 0 and np.max(img[4, :, :]) != 0:
                    img_in = img[:, index_x[0] - ss:index_x[0] + ss, index_y[0] - ss:index_y[0] + ss]
                    img_out = img_in[0:4, :, :].astype(float)
                    label_out = img_in[4, :, :].astype(float)
                    # print(img_in.shape)
                    img = torch.from_numpy(img_out).float()
                    label = torch.from_numpy(label_out).long()
                    # print(label_out.max(),label_out.min())

        elif self.val:
            path = self.root
            img = np.load(os.path.join(path, self.folderlist[index]))
            img = np.asarray(img)
            img_out = img[0:4, :, :].astype(float)
            label_out = img[4, :, :].astype(float)
            # print(img.shape)
            img = torch.from_numpy(img_out).float()
            label = torch.from_numpy(label_out).long()
        else:
            print('###$$$$$$$$$$$$$$$$$$$^^^^^^^^^^^^^')

        return img, label

    def __len__(self):
        return len(self.folderlist)

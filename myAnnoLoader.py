'''
Author: your name
Date: 2021-08-03 14:41:37
LastEditTime: 2021-08-03 17:19:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \MAST_superpixel\myAnnoLoader.py
'''
import os
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random

import cv2
import numpy as np
import functional.utils.io as davis_io
import torch.nn.functional as F

M = 1

def squeeze_index(image, index_list):
    for i, index in enumerate(index_list):
        image[image == index] = i
    return image

def l_loader(path):
    image = cv2.imread(path)
    image = np.float32(image) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return image

def a_loader(path):
    anno, _ = davis_io.imread_indexed(path)
    return anno

def l_prep(image):

    image = transforms.ToTensor()(image)
    image = transforms.Normalize([50,0,0], [50,127,127])(image)

    h,w = image.shape[1], image.shape[2]
    # print('image shape before: ', image.shape)
    if w%M != 0: image = image[:,:,:-(w%M)]
    if h%M != 0: image = image[:,:,-(h%M)]
    # print('image shape after: ', image.shape)
    return image

def a_prep(image):
    h,w = image.shape[0], image.shape[1]
    if w % M != 0: image = image[:,:-(w%M)]
    if h % M != 0: image = image[:-(h%M),:]
    image = np.expand_dims(image, 0)

    return torch.Tensor(image).contiguous().long()

class myImageFloder(data.Dataset):
    def __init__(self, annos, training=False):

        self.annos = annos
        self.training = training

    def __getitem__(self, index):
        anno = self.annos[index]

        annotations = a_prep(a_loader(anno))

        return annotations

    def __len__(self):
        return len(self.annos)
'''
Author: your name
Date: 2021-07-30 14:37:17
LastEditTime: 2021-08-04 15:47:34
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \MAST_superpixel\mask_refine\superpixel_tools.py
'''
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import mark_boundaries  # 导入mark_boundaries 以绘制实际的超像素分割
from skimage.segmentation import find_boundaries  # 导入包以使用SLIC superpixel segmentation
from skimage.feature import local_binary_pattern
import skimage.segmentation as seg
from skimage.util import img_as_float

from PIL import Image
import numpy as np

import scipy.io
import numpy as np


import argparse
import os
import cv2 as cv


def find_major_class(elements, class_num):
    elements = np.array(elements)
    cnts = []
    for i in range(class_num):
        cnt = np.sum(elements==i)
        cnts.append(cnt)
    index_max = cnts.index(max(cnts))
    return index_max, cnts[index_max]


def read_annotation(annotation='00001.png'):
    im = Image.open(annotation)
    anno = np.atleast_3d(im)[...,0]
    return anno.copy()

def load_seglabels(matpath='../../DAVIS/matdata2048/shooting/00001.mat'):
    mat = scipy.io.loadmat(matpath)
    return mat['labels']

def anno_of_region(annotation, segments):
    label_num = np.max(segments)
    class_num = np.max(annotation)+1
    res = np.zeros(label_num+1)
    for i in range(1, label_num+1):
        index = (segments==i)
        anno_arr = annotation[index]
        major_class,_ = find_major_class(anno_arr, class_num)
        res[i] = major_class
    return res


def fix_with_major(annotation, segments, p=0.7):
    anno = annotation.copy()
    label_num = np.max(segments)
    for i in range(1, label_num+1):
        index = (segments==i)
        c = anno[index]
        count = np.bincount(c)
        d = np.argmax(count)
        n = count[d]
        if n/c.size >= p:
            anno[index] = d
    return anno

'''
@description: 返回超像素的平均lbp值，lbp值的标准差以及灰度均值
@param {*} segments 超像素分割后的结果
@param {*} imagepath 对应的图片路径
@return {*}
'''
def extract_feature_lbp(segments, imagepath='00001.jpg'):
    # labels = (segments==seglabel)
    boundary_index = find_boundaries(segments)
    # region = labels& (~boundary_index)

    # settings for LBP
    radius = 1 # LBP算法中范围半径的取值
    n_points = 8 * radius # 领域像素点数
    
    image = cv.imread(imagepath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius)

    label_num = np.max(segments)

    res = np.zeros((label_num+1, 3))
    for i in range(1, label_num+1):
        index = (segments==i)
        region = index& (~boundary_index)
        val = np.mean(lbp[region])
        res[i][1] = np.std(lbp[region])
        lbp[index]=val
        res[i][0] = val
        res[i][2] = np.mean(gray[index])
        
    # plt.imshow(lbp)
    # plt.show()
    return res
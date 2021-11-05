'''
Author: your name
Date: 2021-10-29 15:24:37
LastEditTime: 2021-11-05 16:49:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \GraphCut\img_cut.py
'''
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def read_annotation(annotation='00001.png'):
    im = Image.open(annotation)
    anno = np.atleast_3d(im)[...,0]
    return anno.copy()

# GC_BGD    = 0,  //背景
# GC_FGD    = 1,  //前景
# GC_PR_BGD = 2,  //可能背景
# GC_PR_FGD = 3   //可能前景

def mark_label(anno):
    r = 50
    map = np.zeros_like(anno).astype(np.float)
    h, w = anno.shape
    for x in range(h):
        for y in range(w):
            x0 = (x - r) if x - r >= 0 else 0
            x1 = (x + r) if x + r < h else h-1
            y0 = (y - r) if y - r >= 0 else 0
            y1 = (y + r) if y + r < w else w-1
            arr = anno[x0:x1,y0:y1]
            tmp = np.sum(arr)
            if tmp == 0:
                map[x][y] = 0
            elif tmp > len(arr) * 0.8:
                map[x][y] = 1
            elif tmp > len(arr) * 0.5:
                map[x][y] = 3
            else:
                map[x][y] = 2

    return map
    
def quality_of_long_memory(l_anno, s_anno):
    l_index = (l_anno>0)
    s_index = (s_anno>0)
    ls = l_index&s_index
    l_or_s = l_index|s_index
    IoU = len(ls[ls>0])/len(l_or_s[l_or_s>0])
    ratio = len(l_index[l_index>0])/len(s_index[s_index>0])
    return IoU, ratio

def my_grabcut(img, long_m, short_m):
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    mask = blend_mask(long_m, short_m)
    plt.figure()
    plt.imshow(mask)

    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    
    plt.figure()
    plt.imshow(mask)
    plt.show()

def blend_mask(long_m, short_m):
    index_s = short_m > 0
    index_l = long_m > 0

    map = np.zeros_like(short_m).astype(np.uint8)
    ind1 = index_l & index_s # 长短记忆的重合部分为前景
    ind2 = (index_l & (~ind1)) # 长记忆不与短记忆重合的部分为可能背景
    ind3 = (index_s & (~ind1)) # 短记忆不与长记忆重合的部分为可能前景
    
    map[ind1] = 1 # 前景
    map[ind2] = 2 # 可能背景
    map[ind3] = 3 # 可能前景

    map1 = map.copy()
    
    res = np.where(index_s)

    # plt.figure()
    # plt.imshow(map)

    h, w = map.shape
    r = 9
    for i in range(len(res[0])):
        x = res[0][i]
        y = res[1][i]
        x0 = (x - r) if x - r >= 0 else 0
        x1 = (x + r) if x + r < h else h-1
        y0 = (y - r) if y - r >= 0 else 0
        y1 = (y + r) if y + r < w else w-1
        map1[x0:x1,y0:y1] = 3
    
    map1[map1!=3] = 0
    plt.figure('map1') 
    plt.imshow(map1)
    tmp_index = (map1>0) & ((map==0)|(map==2))
    map[tmp_index] = 3
    return map

if __name__ == '__main__':
    img = plt.imread('./00082.jpg')
    long_m = read_annotation('./long_00082.png')
    short_m = read_annotation('./short_00082.png')
    my_grabcut(img, long_m, short_m)

'''
Author: your name
Date: 2021-11-08 16:09:30
LastEditTime: 2021-11-17 15:50:50
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \dynamic_refers_mast\tools.py
'''
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import os
# import torch
# import torch.nn.functional as F



def dataloader(filepath, videoname):
    catnames = [videoname]
    annotation_all = []
    jpeg_all = []
    for catname in catnames:
        anno_path = os.path.join(filepath, 'Annotations/480p/' + catname.strip())
        cat_annos = [os.path.join(anno_path,file) for file in sorted(os.listdir(anno_path))]
        annotation_all.append(cat_annos)

        jpeg_path = os.path.join(filepath, 'JPEGImages/480p/' + catname.strip())
        cat_jpegs = [os.path.join(jpeg_path, file) for file in sorted(os.listdir(jpeg_path))]
        jpeg_all.append(cat_jpegs)
    return annotation_all, jpeg_all
    

def read_annotation(annotation='00001.png'):
    im = Image.open(annotation)
    anno = np.atleast_3d(im)[...,0]
    return anno.copy()

def get_centroid(img, target=1):
    xx,yy = np.where(img==target)
    x = np.round(np.mean(xx)).astype(np.int32)
    y = np.round(np.mean(yy)).astype(np.int32)
    return x, y

def get_anno_by_ref(model, outputs, images_rgb, long_ref_index, target_frame, dil_int, dil=-1):
    
    pad =  ((0,0), (0,0))
    
    rgb_1 = images_rgb[target_frame]
    l_rgb_0 = [images_rgb[ind] for ind in long_ref_index]
    l_anno_0 = [outputs[ind] for ind in long_ref_index]
    _, _, h, w = l_anno_0[0].size()

    _long_output = model(l_rgb_0, l_anno_0, rgb_1, long_ref_index, target_frame, dil_int)
    _long_output = F.interpolate(_long_output, (h,w), mode='bilinear', align_corners=True)
    long_output = torch.argmax(_long_output, 1, keepdim=True).float()

    long_out_img = long_output[0, 0].cpu().numpy().astype(np.uint8)
    long_out_img = np.pad(long_out_img, pad, 'edge').astype(np.uint8)
    return long_out_img, long_output


# 返回远期、近期参考帧结果的标签数量和他们的IoU
def quality_of_long_memory(l_anno, s_anno):
    l_index = (l_anno>0)
    s_index = (s_anno>0)
    l_num = len(l_index[l_index>0])
    s_num = len(s_index[s_index>0])
    ls = l_index&s_index
    l_or_s = l_index|s_index
    IoU = len(ls[ls>0])/len(l_or_s[l_or_s>0])
    ratio = l_num/s_num
    return IoU, ratio, l_num, s_num


if __name__ == '__main__':
    anno = read_annotation('./resource/shooting/long_00026.png')
    plt.figure()
    plt.imshow(anno)
    plt.show()
    # get_centroid(anno, 1)
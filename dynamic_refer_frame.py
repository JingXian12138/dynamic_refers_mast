'''
Author: your name
Date: 2021-10-31 14:49:11
LastEditTime: 2021-11-06 15:15:49
LastEditors: Please set LastEditors
Description: 有机结合局部搜索和全局搜索
FilePath: \dynamic_refers\dynamic_refer_frame.py
'''


import numpy as np
import functional.feeder.dataset.DavisLoaderLab as DL
from functional.utils.io import imwrite_indexed

import argparse
import os
from models.mast import MAST
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn
import numpy as np
import myGrabcut
import matplotlib.pyplot as plt



def _dataloader(filepath, videoname):
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


def dynamic_refers(videoname='drift-straight'):
    # 加载模型和参数
    parser = argparse.ArgumentParser(description='MAST')
    parser.add_argument('--ref', type=int, default=1)

    args = parser.parse_args()
    print('开始测试：')
    filepath = '/dataset/dusen/DAVIS/'
    TrainData = _dataloader(filepath, videoname)
    TrainImgLoader = torch.utils.data.DataLoader(
        DL.myImageFloder(TrainData[0], TrainData[1], False),
        batch_size=1, shuffle=False,num_workers=0,drop_last=False
    )
    args.training = False
    
    model = MAST(args)
    checkpoint = torch.load('../checkpoint.pt')
    model.load_state_dict(checkpoint['state_dict'])
    print('checkpoint loaded')

    device_ids = [0]
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.eval()
    torch.backends.cudnn.benchmark = True

    # 保存结果
    if not os.path.exists('./mask_refine'):
        os.mkdir('./mask_refine')
    output_folder = os.path.join('./mask_refine', videoname)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for b_i, (images_rgb, annotations) in enumerate(TrainImgLoader):
        images_rgb = [r.cuda() for r in images_rgb]
        annotations = [q.cuda() for q in annotations]
        outputs = [annotations[0].contiguous()]

        # 将第0帧的anno保存
        first_anno = os.path.join(output_folder, '%s.png' % str(0).zfill(5))
        pad =  ((0,0), (0,0))

        # 统计每张图预测结果的标记个数(用于表示目标是否变大或者变小)
        length = len(images_rgb)
        label_num = np.zeros(length)
        
        if not os.path.exists(first_anno):
            # output first mask
            out_img = annotations[0][0, 0].cpu().numpy().astype(np.uint8)

            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            label_num[0] = len(out_img[out_img>0])
            imwrite_indexed(first_anno, out_img )
        
        
        
        long_ref_index = [0]
        
        f=open("quota_of_videos.txt","a")
        f.write('#'+ str(videoname) + '\n')
        for target_frame in range(1, length):
            mem_gap = 2
            short_ref_index = list(filter(lambda x: x > 0, range(target_frame-1, target_frame -1- mem_gap * 3, -mem_gap)))[::-1]
            
            ref_index = long_ref_index + list(filter(lambda x: x > 0, range(target_frame-1, target_frame -1- mem_gap * 3, -mem_gap)))[::-1]

            
            with torch.no_grad():
                
                if target_frame < 5:
                
                    out_img, output = get_anno_by_ref(model, outputs, images_rgb, ref_index, target_frame, 15)
                    outputs.append(output)
                    
                    label_num[target_frame] = len(out_img[out_img>0])
                    output_file = os.path.join(output_folder, '%s.png' % str(target_frame).zfill(5))
                    imwrite_indexed(output_file, out_img)
                else:
                    # 长期记忆的结果
                    long_out_img, _ = get_anno_by_ref(model, outputs, images_rgb, long_ref_index, target_frame, min(target_frame-long_ref_index[0]-1,15))
                    # 短期记忆的结果
                    short_out_img, _ = get_anno_by_ref(model, outputs, images_rgb, short_ref_index, target_frame, 15)
                    # 计算指标
                    IoU, ratio, l_num, s_num = quality_of_long_memory(long_out_img, short_out_img)
                    print(target_frame, ': ',IoU, ratio, l_num, s_num)
                    f.write(str(target_frame).zfill(2) + ': ' + str(IoU)+','+str(ratio)+','+str(l_num)\
                        +','+str(s_num)+'\n')
                    
                    if IoU > 0.9 and 1.05 > ratio and ratio > 0.9:
                        print('#############')
                        ref_index = long_ref_index
                    elif IoU < 0.4:
                        print('IoU<0.4重新选择参考帧')
                        for i in range(0, target_frame - 6):
                            tmp = [i]
                            dil = min(target_frame-long_ref_index[0]-1,15)
                            long_out_img, _ = get_anno_by_ref(model, outputs, images_rgb, tmp, target_frame, dil)
                            IoU_n, ratio_n, l_num_n, s_num_n = quality_of_long_memory(long_out_img, short_out_img)
                            dev = np.abs(IoU - ratio_n)
                            if IoU_n > IoU and ratio_n > 0.8 and ratio_n < 1.1 and np.abs(IoU_n-ratio_n) < dev:
                                dev = np.abs(IoU_n-ratio_n)
                                print('new IoU ', IoU_n)
                                long_ref_index = [i]
                                # output_file = os.path.join(output_folder, 'long_%s.png' % str(target_frame).zfill(5))
                                # imwrite_indexed(output_file, long_tmp_img)
                                ref_index = long_ref_index + short_ref_index
                    elif 0.7> IoU and IoU > 0.2 and ratio > 0.9:
                        # 有一定重合但是很可能散布到了背景上
                        print('grabcut: ', target_frame)
                        img = plt.imread(TrainData[0][0][target_frame])
                        # img = images_rgb[target_frame]
                        # img = img.squeeze(0)
                        # img = img.permute(1,2,0)
                        # print(img.shape)
                        # img = img.cpu().numpy().astype(np.uint8)
                        # print(type(img[0][0][0]))
                        mask_grab = myGrabcut.my_grabcut(img, long_out_img, short_out_img)
                        grab_output_file = os.path.join(output_folder, 'grab_%s.png' % str(target_frame).zfill(5))
                        imwrite_indexed(grab_output_file, mask_grab)
                    

                    long_output_file = os.path.join(output_folder, 'long_%s.png' % str(target_frame).zfill(5))
                    imwrite_indexed(long_output_file, long_out_img)
                    short_output_file = os.path.join(output_folder, 'short_%s.png' % str(target_frame).zfill(5))
                    imwrite_indexed(short_output_file, short_out_img)

                    # 最终进行预测
                    print(target_frame, ': ', ref_index)
                    out_img, output = get_anno_by_ref(model, outputs, images_rgb, ref_index, target_frame, 15)
                    outputs.append(output)

                    label_num[target_frame] = len(out_img[out_img>0])
                    output_file = os.path.join(output_folder, '%s.png' % str(target_frame).zfill(5))
                    imwrite_indexed(output_file, out_img)
                    
        print(label_num)

def get_anno_by_ref(model, outputs, images_rgb, long_ref_index, target_frame, dil_int):
    
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



if __name__ == '__main__':

    vns = ['bike-packing','blackswan','bmx-trees','breakdance','camel','car-roundabout',\
           'car-shadow','cows','dog','drift-chicane','drift-straight','goat','gold-fish',\
           'horsejump-high','judo','kite-surf','lab-coat','libby','loading','mbike-trick',\
           'motocross-jump','paragliding-launch','parkour','pigs','scooter-black','shooting','soapbox']
    #vns = ['drift-straight']
    #for vn in vns:
    #    #frame_arr = [0]
    #    #tar = 1
    #    #N = first_frame(videoname=vn, frames=frame_arr, target_frame=tar, source=2)
    #    for tar in range(20,50):
    #        frame_arr = [0]
            #mem_gap = 2
            #frame_arr = [0] + list(filter(lambda x: x > 0, range(tar-1, tar -1- mem_gap * 3, -mem_gap)))[::-1]
    #        get_next_anno(videoname=vn, frames=frame_arr, target_frame=tar)
    # for vn in vns:
    dynamic_refers('goat')
    
    # filepath = '/dataset/dusen/DAVIS/'
    # TrainData = _dataloader(filepath, 'goat')
    # print(TrainData[0][0][33])

    
'''
Author: your name
Date: 2021-10-31 14:49:11
LastEditTime: 2021-11-17 22:18:08
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
import torch.backends.cudnn
import numpy as np
import myGrabcut
import matplotlib.pyplot as plt
import tools


def dynamic_refers(videoname='drift-straight'):
    # 加载模型和参数
    parser = argparse.ArgumentParser(description='MAST')
    parser.add_argument('--ref', type=int, default=1)

    args = parser.parse_args()
    print('开始测试：')
    filepath = '/dataset/dusen/DAVIS/'
    TrainData = tools.dataloader(filepath, videoname)
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
        _, _, h, w = annotations[0].size()
        span = (np.sqrt(h*h+w*w)/8).astype(np.int32)
        print('_______________span_________________', span)
        

        # 统计每张图预测结果的标记个数(用于表示目标是否变大或者变小)
        length = len(images_rgb)
        label_num = np.zeros(length)
        centroids = np.zeros((length,2))
        centroids_long = np.zeros((length,2))
        centroids_short = np.zeros((length,2))
        
        
        #if not os.path.exists(first_anno):
            # output first mask
        out_img = annotations[0][0, 0].cpu().numpy().astype(np.uint8)

        out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
        label_num[0] = len(out_img[out_img>0])
        cx, cy = tools.get_centroid(out_img, 1)
        centroids[0][0] = cx 
        centroids[0][1] = cy
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
                
                    out_img, output = tools.get_anno_by_ref(model, outputs, images_rgb, ref_index, target_frame, 1, [1])
                    outputs.append(output)
                    # 质心
                    cx, cy = tools.get_centroid(out_img, 1)
                    centroids[target_frame][0] = cx 
                    centroids[target_frame][1] = cy

                    label_num[target_frame] = len(out_img[out_img>0])
                    output_file = os.path.join(output_folder, '%s.png' % str(target_frame).zfill(5))
                    imwrite_indexed(output_file, out_img)
                else:
                    # 短期记忆的结果
                    short_out_img, _ = tools.get_anno_by_ref(model, outputs, images_rgb, short_ref_index, target_frame, 0)
                    # 质心
                    cx_s, cy_s = tools.get_centroid(short_out_img, 1)
                    centroids_short[target_frame][0] = cx_s 
                    centroids_short[target_frame][1] = cy_s
                    print('短期记忆的质心：(', cx_s, ',', cy_s,')')
                    
                    cx_o = centroids[long_ref_index[0]][0]
                    cy_o = centroids[long_ref_index[0]][1]
                    print('参考长期记忆的质心：(', cx_o, ',', cy_o,')')
                    dil_sp = (np.sqrt((cx_s-cx_o)**2 + (cy_s-cy_o)**2) // span + 1).astype(np.int32)
                    print('********: dil_sp', dil_sp)
                    # 长期记忆的结果
                    long_out_img, _ = tools.get_anno_by_ref(model, outputs, images_rgb, long_ref_index, target_frame, 1, [dil_sp])
                    # 质心
                    cx, cy = tools.get_centroid(long_out_img, 1)
                    centroids_long[target_frame][0] = cx 
                    centroids_long[target_frame][1] = cy
                    # print('长期记忆的质心：(', cx, ',', cy,')')

                    

                    # 计算指标
                    IoU, ratio, l_num, s_num = tools.quality_of_long_memory(long_out_img, short_out_img)
                    print(target_frame, ': ',IoU, ratio, l_num, s_num)
                    f.write(str(target_frame).zfill(2) + ': ' + str(IoU)+','+str(ratio)+','+str(l_num)\
                        +','+str(s_num)+'\n')
                    
                    flag = False
                    if IoU > 0.9 and 1.05 > ratio and ratio > 0.9:
                        print('#############')
                        ref_index = long_ref_index + long_ref_index + [target_frame-3]
                    elif IoU < 0.6:
                        print('IoU<0.6重新选择参考帧')
                        for i in range(0, target_frame - 6):
                            tmp = [i]
                            dil = min(target_frame-long_ref_index[0]-1,15)
                            long_out_img_t, _ = tools.get_anno_by_ref(model, outputs, images_rgb, tmp, target_frame, 1, [1])
                            IoU_n, ratio_n, l_num_n, s_num_n = tools.quality_of_long_memory(long_out_img_t, short_out_img)
                            dev = np.abs(IoU - ratio_n)
                            if IoU_n > IoU and ratio_n > 0.6 and ratio_n < 1.1 and np.abs(IoU_n-ratio_n) < dev:
                                IoU = IoU_n
                                ratio = ratio_n
                                dev = np.abs(IoU_n-ratio_n)
                                print('new IoU ', IoU_n)
                                long_ref_index = [i]
                                # output_file = os.path.join(output_folder, 'long_%s.png' % str(target_frame).zfill(5))
                                # imwrite_indexed(output_file, long_tmp_img)
                                ref_index = long_ref_index + [target_frame-1, target_frame-3]
                    
                        # 对新的long_ref进行前背景分割
                        print('grabcut: ', target_frame)
                        img = plt.imread(TrainData[1][0][target_frame])
                        
                        mask_grab = myGrabcut.my_grabcut(img, long_out_img, short_out_img)
                        print('mask_grab shape', mask_grab.shape)
                        grab_output_file = os.path.join(output_folder, 'grab_%s.png' % str(target_frame).zfill(5))
                        imwrite_indexed(grab_output_file, mask_grab)
                        
                        mask_grab_output = torch.Tensor(mask_grab)
                        mask_grab_output = mask_grab_output.unsqueeze(0).unsqueeze(0)
                        flag = True

                        # tar = long_ref_index[0]
                        # img_l = plt.imread(TrainData[1][0][target_frame])
                        # long_out_img_l, _ = tools.get_anno_by_ref(model, outputs, images_rgb, [0], tar, 1, [1])
                        # short_out_img_l, _ = tools.get_anno_by_ref(model, outputs, images_rgb, [tar-1,tar-3,tar-5], tar, 0, [1])
                        # mask_grab_l = myGrabcut.my_grabcut(img_l, long_out_img_l, short_out_img_l)
                        # print('mask_grab shape_l', mask_grab_l.shape)
                        # grab_output_file = os.path.join(output_folder, 'grab_%s.png' % str(tar).zfill(5))
                        # imwrite_indexed(grab_output_file, mask_grab_l)
                        # mask_grab_output_l = torch.Tensor(mask_grab_l)
                        # mask_grab_output_l = mask_grab_output_l.unsqueeze(0).unsqueeze(0)
                        # outputs[tar] = mask_grab_output_l


                    # elif 0.7> IoU and IoU > 0.2 and ratio > 0.9:
                    #     # 有一定重合但是很可能散布到了背景上
                    #     print('grabcut: ', target_frame)
                    #     img = plt.imread(TrainData[1][0][target_frame])
                    #     mask_grab = myGrabcut.my_grabcut(img, long_out_img, short_out_img)
                    #     print('mask_grab shape', mask_grab.shape)
                    #     grab_output_file = os.path.join(output_folder, 'grab_%s.png' % str(target_frame).zfill(5))
                    #     imwrite_indexed(grab_output_file, mask_grab)
                        
                    #     mask_grab_output = torch.Tensor(mask_grab)
                    #     mask_grab_output = mask_grab_output.unsqueeze(0).unsqueeze(0)
                    #     flag = True
                    

                    long_output_file = os.path.join(output_folder, 'long_%s.png' % str(target_frame).zfill(5))
                    imwrite_indexed(long_output_file, long_out_img)
                    short_output_file = os.path.join(output_folder, 'short_%s.png' % str(target_frame).zfill(5))
                    imwrite_indexed(short_output_file, short_out_img)

                    # 最终进行预测
                    if flag:
                        print(target_frame, ' mask_grab')
                        out_img = mask_grab
                        output = mask_grab_output
                    else:
                        print(target_frame, ': ', ref_index)
                        out_img, output = tools.get_anno_by_ref(model, outputs, images_rgb, ref_index, target_frame, 1, [3])
                        # 质心
                        cx, cy = tools.get_centroid(out_img, 1)
                        centroids[target_frame][0] = cx 
                        centroids[target_frame][1] = cy
                        
                    outputs.append(output)
                    label_num[target_frame] = len(out_img[out_img>0])
                    output_file = os.path.join(output_folder, '%s.png' % str(target_frame).zfill(5))
                    imwrite_indexed(output_file, out_img)
                    
        print(label_num)
        # print('质心: ')
        # print(centroids)





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
    dynamic_refers('car-roundabout')
    
    # filepath = '/dataset/dusen/DAVIS/'
    # TrainData = tools.dataloader(filepath, 'goat')
    # print(TrainData[1][0][33])


    
#!/usr/bin/python3
# coding=utf-8

import cv2
import numpy as np
import pylab as plt

## 1阶IIR时域低通滤波
def time_iir_lpf(frame_in, alpha):
    f0=np.zeros_like(frame_in[0])
    frame_out=[]
    for f in frame_in:
        f0=f0*alpha+(1.0-alpha)*f
        frame_out.append(f0)
    return frame_out

## 1阶IIR时域高通滤波
def time_iir_hpf(frame_in, alpha):
    frame_lf=time_iir_lpf(frame_in,alpha)
    frame_hf=[f1-f2 for f1,f2 in zip(frame_in,frame_lf)]
    return frame_hf

## 1阶时域中值滤波
def time_med_lpf(frame_in,sz=3):
    frame_out=[np.median(frame_in[n:n+sz],axis=0) for n in range(len(frame_in)-sz)]
    return frame_out


## 从图像帧数据生成GIF动图  
def create_gif(frames,fname,duration=0.025):
    import imageio  
    imageio.mimsave(fname, frames, 'GIF', duration = duration)  

## 将深度图序列存成GIF
def save_dep_frames(dep_frames,dmin,dmax,fname='frames.gif',cmap=cv2.COLORMAP_JET,duration=0.025):
    rgb_frames=[cv2.applyColorMap(((f-dmin)/dmax*255.0).astype(np.uint8), cmap) for f in dep_frames]
    create_gif(rgb_frames,fname,duration)
    return rgb_frames
    
    
## 显示连续帧
def view_dep_frames(frames,intv=0.001):
    for n,img in enumerate(frames):
        plt.clf()
        plt.imshow(img,cmap='jet')
        plt.title('%d/%d'%(n,len(frames)))
        plt.show(block=False)
        plt.pause(intv)
    plt.clf()
    return

def view_rgb_frames(frames,intv=0.001):
    for n,img in enumerate(frames):
        plt.clf()
        plt.imshow(img)
        plt.title('%d/%d'%(n,len(frames)))
        plt.show(block=False)
        plt.pause(intv)
    plt.clf()
    return

###################
# 算法测试
###################
if __name__=='__main__':
    DMIN,DMAX=0.5,2.2   # 伪彩色距离范围
    PN=0.1              # 噪声功率
    
    # 加载旋转立方体图像
    print('[INF] loading npy data...')
    dep_frames=np.load('cube.npy')
    dep_frames+=np.random.randn(*dep_frames.shape)*PN        # 加入噪声
    print('[INF] saving GIF...')
    save_dep_frames(dep_frames,DMIN,DMAX,'spin_cube.gif')   # 原图序列保存动态GIF
    if False: view_dep_frames(dep_frames)
    
    # 时域中值滤波
    print('[INF] MED filter...')
    dep_frames_med=time_med_lpf(dep_frames,sz=5)
    print('[INF] saving GIF...')
    save_dep_frames(dep_frames_med,DMIN,DMAX,'spin_cube_med.gif') # 序列转成伪彩色保存动态GIF
    if False: view_dep_frames(dep_frames_med)
    
    # 时域低通滤波
    print('[INF] IIR-LPF...')
    dep_frames_lf=time_iir_lpf(dep_frames,alpha=0.9)
    print('[INF] saving GIF...')
    save_dep_frames(dep_frames_lf,DMIN,DMAX,'spin_cube_lf.gif') # 序列转成伪彩色保存动态GIF
    if False: view_dep_frames(dep_frames_lf)
    
    # 时域高通滤波
    print('[INF] IIR-HPF...')
    dep_frames_hf=time_iir_hpf(dep_frames,alpha=0.9)
    print('[INF] saving GIF...')
    save_dep_frames(dep_frames_hf,DMIN,DMAX,'spin_cube_hf.gif') # 序列转成伪彩色保存动态GIF
    if False: view_dep_frames(dep_frames_hf)
    

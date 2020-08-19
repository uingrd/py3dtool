#!/usr/bin/python3
# coding=utf-8

import math
import numpy as np
import cv2

import pylab as plt

import sys
sys.path.append('../pc_to_dep')
sys.path.append('../pc_trans' )
from pc_to_dep import *
from pc_trans  import *

# 生成立方体
def get_cube_pc(N=5000):
    p0=np.ones(N)*0.5
    p1=np.random.rand(N)-0.5
    p2=np.random.rand(N)-0.5
    pc=np.hstack((np.array([ p0, p1, p2]), np.array([-p0, p1, p2]),\
                  np.array([ p1, p0, p2]), np.array([ p1,-p0, p2]),\
                  np.array([ p1, p2, p0]), np.array([ p1, p2,-p0]))).T
    return pc

# 生成立方体点云
pc=get_cube_pc(50000)*0.5
R=calc_matrix_rot(math.radians(45),math.radians(45))
pc=np.dot(pc,R)
pc=pc+[0,0,1]   # 点云移后，方便投影到深度图

# 相机参数
CAM_WID,CAM_HGT = 320,240
CAM_FX,CAM_FY   = 200,200
CAM_CX,CAM_CY   = CAM_WID//2,CAM_HGT//2
IMG_HGT,IMG_WID = CAM_HGT,CAM_WID
DMAX=2.0

# 点云到深度图转换器
conv=pc_to_dep_c(CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT)
    
# 生成移动立方体深度图序列
print('[INF] generating frames...')
frame=[]
for a in range(0,360,1):
    print('%d'%a)
    pc_new=pc_roty_mov(math.radians(a),0,0,1,pc)
    img_dep=conv.to_dep(pc_new)
    img_dep[np.isinf(img_dep)]=DMAX
    frame.append(img_dep)
np.save('cube.npy', np.array(frame))

# 显示
for n,img_dep in enumerate(frame_dep):
    plt.clf()
    plt.imshow(img_dep,cmap='jet')
    plt.title(str(n))
    plt.show(block=False)
    plt.pause(0.01)


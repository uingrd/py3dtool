#!/usr/bin/python3
# coding=utf-8

import numpy as np

# 相机参数
CAM_WID,CAM_HGT = 640,480

img_dep=np.genfromtxt('head_dep.csv', delimiter=',').astype(np.float32).reshape(CAM_HGT,CAM_WID)
img_amp=np.genfromtxt('head_amp.csv', delimiter=',').astype(np.float32).reshape(CAM_HGT,CAM_WID)

########## 调用opencv计算几种滤波 ###########
imgf=img_dep.copy()

if True: imgf=cv2.GaussianBlur(imgf,(5,5),2)
if True: imgf=cv2.blur(imgf,(5,5))
if True: imgf=cv2.medianBlur(imgf,3)         
if True: imgf=cv2.bilateralFilter(imgf,5,1,1)

# 显示原始数据
import pylab as plt
import matplotlib
matplotlib.use('tkagg')

plt.clf()
plt.subplot(1,2,1)
plt.imshow(img_dep/img_dep.max(),cmap='gray')
plt.title('depth')
plt.subplot(1,2,2)
plt.imshow(img_amp/img_amp.max(),cmap='gray')
plt.title('amplitude')
plt.show()

# 显示滤波结果
plt.clf()
plt.subplot(1,2,1)
plt.imshow(np.clip(img_dep,DMIN,DMAX),cmap='jet')
plt.title('original')
plt.subplot(1,2,2)
plt.imshow(np.clip(imgf,DMIN,DMAX),cmap='jet')
plt.title('filter')
plt.show()

## 以点云形式显示结果
CAM_FX,CAM_FY   = 795.209,793.957
CAM_CX,CAM_CY   = 332.031,231.308

tab_x,tab_y=np.meshgrid(range(CAM_WID),range(CAM_HGT))  # 构建点云变换的速算表格
tab_x=tab_x.astype(np.float32)-CAM_CX
tab_y=tab_y.astype(np.float32)-CAM_CY

tab_dep_to_z=1.0/np.sqrt(tab_x**2/(CAM_FX**2)+tab_y**2/(CAM_FY**2)+1)
tab_dep_to_x=tab_x/CAM_FX*tab_dep_to_z
tab_dep_to_y=tab_y/CAM_FX*tab_dep_to_z

pc_z=imgf*tab_dep_to_z  # 深度转点云坐标
pc_x=imgf*tab_dep_to_x
pc_y=imgf*tab_dep_to_y
pc=np.array([pc_x.ravel(),pc_y.ravel(),pc_z.ravel()]).T

# 点云形式显示结果
import sys
sys.path.append("../viewer")    # 同级的viewer目录
from pc_view import *

DMIN,DMAX=0.5,0.8
pc_view(pc,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,dmin=DMIN,dmax=DMAX)



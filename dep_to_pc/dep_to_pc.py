#!/usr/bin/python3
# coding=utf-8

import numpy as np
import cv2

class dep_to_pc_c:
    def __init__(self,fx,fy,cx,cy,w,h,dvec=None):
        # 构建速算表格
        tab_x,tab_y=np.meshgrid(range(w),range(h))
        tab_x=tab_x.astype(np.float32)-cx
        tab_y=tab_y.astype(np.float32)-cy
        
        self.tab_dep_to_z=1.0/np.sqrt(tab_x**2/(CAM_FX**2)+tab_y**2/(CAM_FY**2)+1)
        self.tab_dep_to_x=tab_x/fx*self.tab_dep_to_z
        self.tab_dep_to_y=tab_y/fy*self.tab_dep_to_z
        
        self.cam_mat=np.array([fx,0,cx,0,fy,cy,0,0,1]).reshape(3,3)
        self.cam_dvec=dvec
        return
    
    def to_pc(self,img_dep):
        if self.cam_dvec is not None:
            img_dep=cv2.undistort(img_dep,self.cam_mat,self.cam_dvec)
            
        # 深度转点云坐标
        pc_z=img_dep*self.tab_dep_to_z
        pc_x=img_dep*self.tab_dep_to_x
        pc_y=img_dep*self.tab_dep_to_y

        pc=np.array([pc_x.ravel(),pc_y.ravel(),pc_z.ravel()]).T
        return pc

if __name__=='__main__':
    # 参数
    if False:
        CAM_WID,CAM_HGT = 320,240
        CAM_CX,CAM_CY = 157.262,122.083
        CAM_FX,CAM_FY = 210.783,204.817
        CAM_DVEC=np.array([-0.378924, 0.145035, -0.000970854, 6.20728e-005, -0.0133954])

        # 点云转换器
        conv=dep_to_pc_c(CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,CAM_DVEC)

        # 加载数据
        img_dep=np.genfromtxt('dep_320x240.csv', delimiter=',').astype(np.float32)
    else:
        CAM_WID,CAM_HGT = 640,480        
        CAM_FX,CAM_FY   = 795.209,793.957
        CAM_CX,CAM_CY   = 332.031,231.308

        # 点云转换器
        conv=dep_to_pc_c(CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT)

        # 加载数据
        img_dep=np.genfromtxt('dep.csv', delimiter=',').astype(np.float32)


    # 从CSV文件加载点云并显示
    pc=conv.to_pc(img_dep)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    ax = plt.figure().gca(projection='3d')
    ax.plot(pc[:,0],pc[:,1],pc[:,2],'k.',markersize=0.02)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('point cloud')
    plt.show()    


#!/usr/bin/python3
# coding=utf-8

import numpy as np

class dep_to_pc_c:
    def __init__(self,fx,fy,cx,cy,w,h):
        # 构建速算表格
        tab_x,tab_y=np.meshgrid(range(w),range(h))
        tab_x=tab_x.astype(np.float32)-cx
        tab_y=tab_y.astype(np.float32)-cy
        
        self.tab_dep_to_z=1.0/np.sqrt(tab_x**2/(CAM_FX**2)+tab_y**2/(CAM_FY**2)+1)
        self.tab_dep_to_x=tab_x/fx*self.tab_dep_to_z
        self.tab_dep_to_y=tab_y/fy*self.tab_dep_to_z
        return
    
    def to_pc(self,img_dep):
        # 深度转点云坐标
        pc_z=img_dep*self.tab_dep_to_z
        pc_x=img_dep*self.tab_dep_to_x
        pc_y=img_dep*self.tab_dep_to_y

        pc=np.array([pc_x.ravel(),pc_y.ravel(),pc_z.ravel()]).T
        return pc

if __name__=='__main__':
    # 参数
    CAM_WID,CAM_HGT = 640,480        
    CAM_FX,CAM_FY   = 795.209,793.957
    CAM_CX,CAM_CY   = 332.031,231.308

    # 点云转换器
    conv=dep_to_pc_c(CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT)

    # 加载数据
    img_dep=np.genfromtxt('img_dep_640x480.csv', delimiter=',').astype(np.float32)

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


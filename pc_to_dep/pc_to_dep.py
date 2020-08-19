#!/usr/bin/python3
# coding=utf-8

import numpy as np

class pc_to_dep_c:
    def __init__(self,fx,fy,cx,cy,w,h,eps=1.0e-8):
        # 构建速算表格
        tab_x,tab_y=np.meshgrid(range(w),range(h))
        tab_x=tab_x.astype(np.float32)-cx
        tab_y=tab_y.astype(np.float32)-cy        
        self.tab_dep_to_z=1.0/np.sqrt(tab_x**2/(fx**2)+tab_y**2/(fy**2)+1)
        
        self.fx,self.fy=fx,fy
        self.cx,self.cy=cx,cy
        self.w,self.h=w,h
        
        self.eps=eps
        return


    def to_dep(self,pc,fill_hole=True,z_to_dep=True):
        pc=pc.reshape(-1,3)
        
        # 滤除镜头后方的点
        valid=pc[:,2]>self.eps
        z=pc[valid,2]
        
        # 点云反向映射到像素坐标位置
        u=np.round(pc[valid,0]*self.fx/z+self.cx).astype(int)
        v=np.round(pc[valid,1]*self.fy/z+self.cy).astype(int)
    
        # 滤除超出图像尺寸的无效像素
        valid=np.bitwise_and(np.bitwise_and((u>=0),(u<self.w)),
                             np.bitwise_and((v>=0),(v<self.h)))
        u,v,z=u[valid],v[valid],z[valid]

        # 按距离填充生成深度图，近距离覆盖远距离
        img_z=np.full((self.h, self.w),np.inf)        
        for ui,vi,zi in zip(u,v,z):
            img_z[vi,ui]=min(img_z[vi,ui],zi)   # 近距离像素屏蔽远距离像素

        # 小洞和“透射”消除
        if fill_hole:
            img_z_shift=np.array([img_z,\
                                  np.roll(img_z, 1,axis=0),\
                                  np.roll(img_z,-1,axis=0),\
                                  np.roll(img_z, 1,axis=1),\
                                  np.roll(img_z,-1,axis=1)])
            img_z=np.min(img_z_shift,axis=0)
        
        return img_z/self.tab_dep_to_z if z_to_dep else img_z
            
if __name__ == '__main__':
    # 参数
    CAM_WID,CAM_HGT = 640,480        
    CAM_FX,CAM_FY   = 795.209,793.957
    CAM_CX,CAM_CY   = 332.031,231.308

    EPS=1.0e-12

    # 转换器
    conv=pc_to_dep_c(CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,EPS)
    
    # 加载点云数据
    pc=np.genfromtxt('pc.csv', delimiter=',').astype(np.float32)
    
    # 转换
    img_dep=conv.to_dep(pc)
    
    import matplotlib.pyplot as plt
    plt.imshow(img_dep,cmap='jet')
    plt.show()



#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 查看3D点云
import numpy as np
import cv2
import time

####################
# 显示点云
####################
def pc_view(pc,\
            cam_fx,cam_fy,cam_cx,cam_cy,\
            img_wid,img_hgt,\
            cz=None,dmin=0,dmax=5,name='point colud viewer',
            update_pc_callback=None):

    # 移动到(0,0,tz)位置
    def _pc_trans_movz(tz): 
        return _pc_trans_mov(0,0,tz)

    # 生成绕X轴旋转矩阵
    def _pc_trans_rotx(b):    
        return np.array([[1,        0 ,       0 ,0],
                         [0, np.cos(b),np.sin(b),0],
                         [0,-np.sin(b),np.cos(b),0],
                         [0,        0 ,       0 ,1]])

    # 生成绕Y轴旋转矩阵
    def _pc_trans_roty(b):    
        return np.array([[np.cos(b),0,-np.sin(b),0],
                        [       0 ,1,        0 ,0],
                        [np.sin(b),0, np.cos(b),0],
                        [       0 ,0,        0 ,1]])

    # 生成点云平移矩阵
    def _pc_trans_mov(tx,ty,tz):    
        return np.array([[ 1, 0, 0,0],
                        [ 0, 1, 0,0],
                        [ 0, 0, 1,0],
                        [tx,ty,tz,1]])


    cv2.namedWindow(name)
    
    eps=1.0e-8
    
    # 点云变换参数
    ax=ay=0.0                       # 点云旋转角度
    mz=0                            # 点云观察点位置:(0,0,mz)
    if cz is None: 
        cz=np.mean(pc[:,2])         # 点云旋转中心:(0,0,cz)
    update_M=False                  # 标示变换矩阵是否更新了
    
    # 鼠标动作跟踪
    mouse_x=mouse_y=0
    mouse_down=False
    def mouse_callback(evt,x,y,flags,param):
        nonlocal mouse_x,mouse_y,mouse_down
        nonlocal ax,ay,mz,update_M
        
        if evt==cv2.EVENT_LBUTTONDOWN:
            mouse_down=True
            mouse_x,mouse_y=x,y
        elif evt==cv2.EVENT_LBUTTONUP:
            mouse_down=False
        elif evt==cv2.EVENT_MOUSEMOVE:
            if mouse_down:
                if flags&cv2.EVENT_FLAG_SHIFTKEY==0:
					# 鼠标拖动对应旋转
                    dx,dy=x-mouse_x,y-mouse_y
                    mouse_x,mouse_y=x,y
                    if not update_M:
                        ax+=dy/50.0
                        ay-=dx/50.0
                        update_M=True
                else:
					# 按下shift键时鼠标拖动对应观察者前后移动
                    dy=y-mouse_y
                    mouse_x,mouse_y=x,y
                    if not update_M:
                        mz+=dy/10
                        update_M=True
                
    cv2.setMouseCallback(name,mouse_callback)
    
    pc_view=pc.copy()
    cnt=0   # 帧计数
    while True:            
        # 点云变换，并将变换后的点云映射回深度图
        if update_M:
            M=np.dot(_pc_trans_movz(-cz), _pc_trans_rotx(ax))
            M=np.dot(M, _pc_trans_roty(ay))
            M=np.dot(M, _pc_trans_movz(cz+mz))
            
            pc_view=np.dot(pc,M[:3,:3])+M[3,:3]
            update_M=False
        
        # 去除镜头后方的点
        valid=pc_view[:,2]>eps
        z=pc_view[valid,2]
        
        # 点云反向映射到像素坐标位置
        u=np.round(pc_view[valid,0]*cam_fx/z+cam_cx).astype(int)
        v=np.round(pc_view[valid,1]*cam_fy/z+cam_cy).astype(int)
    
        # 滤除超出图像尺寸的无效像素
        valid=np.bitwise_and(np.bitwise_and((u>=0),(u<img_wid)),
                             np.bitwise_and((v>=0),(v<img_hgt)))
        u,v,z=u[valid],v[valid],z[valid]
    
        # 按距离填充生成深度图，近距离覆盖远距离
        img_z=np.full((img_hgt, img_wid),np.inf)        
        for ui,vi,zi in zip(u,v,z):
            img_z[vi,ui]=min(img_z[vi,ui],zi)   # 近距离像素屏蔽远距离像素
        mask=np.isinf(img_z)                    # mask标示出未填充的深度图像素
        
        # 将深度图转换成伪彩色，并更新显示
        img_u8=np.uint8(np.clip((img_z-dmin)/float(dmax-dmin),0.0,1.0)*255)
        img_rgb=cv2.applyColorMap(img_u8,cv2.COLORMAP_RAINBOW)
        img_rgb[mask,:]=0
        
        # 刷新屏幕显示
        cv2.imshow(name, img_rgb)
        
        # 检查用户界面操作
        key=cv2.waitKey(1)&0xFF
        if key==ord('q') or key==27 or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key==ord('r'):
            ax=ay=0.0   # 点云旋转角度
            mz=0        # 点云观察点位置:(0,0,mz)
            update_M=True
    
        if update_pc_callback is not None:
            pc_new=update_pc_callback(cnt)      # 获取新的图像帧回调函数
            if pc_new is not None:
                pc=pc_new
                update_M=True
        
        time.sleep(0.001)
        cnt+=1
    return

####################
# 单元测试
####################
if __name__=='__main__':
	# 相机参数
    CAM_WID,CAM_HGT = 320,240
    CAM_FX,CAM_FY   = 160,120
    CAM_CX,CAM_CY   = CAM_WID//2,CAM_HGT//2

    # 生成测试点云
    N=50000 # 点云计数
    # 下面函数每次调用生成一个端面被截平的椭球体表面点云，cnt决定椭球体形状
    def generate_pc(cnt=0):    
        # 生成球面点云
        pc=np.random.rand(N,3)-0.5
        pc=pc/np.linalg.norm(pc,axis=1).reshape(N,1) 
        
        # 生成变化了的椭球三轴长度
        p=cnt/60.0*np.pi
        a=0.5*(1.0+np.cos(p)*0.5)
        b=0.3*(1.0+np.cos(p*1.3+0.5)*0.5)
        c=0.4*(1.0+np.cos(p*0.4+1.0)*0.5)
        
        # 球面点云编程椭球面
        pc=pc*np.array([a,b,c])+[0,0,0.6]
        pc[pc[:,0]>0.2,0]=0.2
        
        return pc
    
	# 显示点云
    pc0=generate_pc()
    pc_view(pc0,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,
            dmin=0,dmax=1,
            update_pc_callback=generate_pc)



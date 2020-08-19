#!/usr/bin/python3
# coding=utf-8

import numpy as np
import math

## 弧度和角度互换
def deg_to_rad(d): return math.radians(d)   # =d*(np.pi/180.0)
def rad_to_deg(r): return math.degrees(r)   # =r*(180.0/np.pi)

# 计算绕X轴旋转的旋转矩阵
def calc_matrix_rotx(b):
    return np.array([[1,        0 ,       0 ],
                     [0, np.cos(b),np.sin(b)],
                     [0,-np.sin(b),np.cos(b)]]) 

# 计算绕Y轴旋转的旋转矩阵
def calc_matrix_roty(b):    
    return np.array([[np.cos(b),0,-np.sin(b)],
                     [       0 ,1,        0 ],
                     [np.sin(b),0, np.cos(b)]])

# 计算绕Z轴旋转的旋转矩阵
def calc_matrix_rotz(b):    
    return np.array([[ np.cos(b),np.sin(b),0],
                     [-np.sin(b),np.cos(b),0],
                     [        0 ,       0 ,1]])
                     
def calc_matrix_rot(ax=0,ay=0,az=0):
    Rx=calc_matrix_rotx(ax)
    Ry=calc_matrix_roty(ay)
    Rz=calc_matrix_rotz(az)
    return np.dot(Rx,np.dot(Ry,Rz))

## 功能描述：
#     点云变换
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
def pc_trans(T,pc):      
    T_rot=T[0:3,0:3]        # 截取旋转部分
    pc_out=np.dot(pc,T_rot) # 计算旋转
    pc_out[:,0]+=T[3,0]     # 计算平移
    pc_out[:,1]+=T[3,1]
    pc_out[:,2]+=T[3,2]

    return pc_out


## 功能描述：
#     点旋转平移变换
# 输入参数：
#     x,y,z: 输入点坐标(标量）
#     T    : 变换矩阵
# 输出参数：
#     x_out,y_out,z_out：输出点坐标
def pnt_trans(T,x,y=None,z=None):
    if y is None: x,y,z=x[0],x[1],x[2]  # 输入坐标封装在x里面了
    xout=x*T[0,0]+y*T[1,0]+z*T[2,0]+T[3,0]
    yout=x*T[0,1]+y*T[1,1]+z*T[2,1]+T[3,1]
    zout=x*T[0,2]+y*T[1,2]+z*T[2,2]+T[3,2]
    return xout,yout,zout

## 功能描述：
#     点旋转变换
# 输入参数：
#     x,y,z: 输入点坐标(标量）
#     R    : 变换矩阵
# 输出参数：
#     x_out,y_out,z_out：输出点坐标
def pnt_rot(R,x,y=None,z=None):
    if y is None: x,y,z=x[0],x[1],x[2]  # 输入坐标封装在x里面了
    xout=x*R[0,0]+y*R[1,0]+z*R[2,0]
    yout=x*R[0,1]+y*R[1,1]+z*R[2,1]
    zout=x*R[0,2]+y*R[1,2]+z*R[2,2]
    return xout,yout,zout


## 功能描述：
#     点云平移
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     tx,ty,tz: 各个方向移动距离
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_mov(tx,ty,tz,pc=None):    
    T=np.array([[ 1, 0, 0,0],
                [ 0, 1, 0,0],
                [ 0, 0, 1,0],
                [tx,ty,tz,1]])    # 移动坐标
    return pc_trans(T,pc) if pc is not None else T


## 功能描述：
#     点云沿x方向平移
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     tx: 沿x方向移动距离
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_movx(tx,pc=None): return pc_mov(tx,0,0,pc)


## 功能描述：
#     点云沿y方向平移
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     ty: 沿y方向移动距离
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_movy(ty,pc=None): return pc_mov(0,ty,0,pc)    


## 功能描述：
#     点云沿z方向平移
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     tz: 沿z方向移动距离
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out点云变换类
def pc_movz(tz,pc=None): return pc_mov(0,0,tz,pc)


## 功能描述：
#     点云沿着x轴旋转
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_rotx(b,pc=None):    
    T=np.array([[1,        0 ,       0 ,0],
                [0, np.cos(b),np.sin(b),0],
                [0,-np.sin(b),np.cos(b),0],
                [0,        0 ,       0 ,1]])    # 绕X轴旋转
    return pc_trans(T,pc) if pc is not None else T


## 功能描述：
#     点云沿着y轴旋转
# 用法：
#     [pc_out,T]=pc_roty(pc,b)
# 输入参数：
#     pc: 输入点云集depth_img合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵
def pc_roty(b,pc=None):    
    T=np.array([[np.cos(b),0,-np.sin(b),0],
                [       0 ,1,        0 ,0],
                [np.sin(b),0, np.cos(b),0],
                [       0 ,0,        0 ,1]])    # 绕Y轴旋转
    return pc_trans(T,pc) if pc is not None else T


## 功能描述：
#     点云沿着z轴旋转
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数0,1)据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_rotz(b,pc=None):    
    T=np.array([[ np.cos(b),np.sin(b),0,0],
                [-np.sin(b),np.cos(b),0,0],
                [        0 ,       0 ,1,0],
                [        0 ,       0 ,0,1]])    # 绕Z轴旋转
    return pc_trans(T,pc) if pc is not None else T


## 功能描述：
#     点云以(x,y,z)为中心，沿着z轴旋转
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_rotx_mov(b,x,y,z,pc=None):
    T1=pc_mov(-x,-y,-z)
    T2=pc_rotx(b)
    T3=pc_mov(x,y,z)
    T=np.dot(np.dot(T1,T2),T3)    
    return pc_trans(T,pc) if pc is not None else T


## 功能描述：
#     点云以(x,y,z)为中心，沿着z轴旋转
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_roty_mov(b,x,y,z,pc=None):
    T1=pc_mov(-x,-y,-z)
    T2=pc_roty(b)
    T3=pc_mov(x,y,z)
    T=np.dot(np.dot(T1,T2),T3)
    return pc_trans(T,pc) if pc is not None else T


## 功能描述：
#     点云以(x,y,z)为中心，沿着z轴旋转
# 输入参数：test_cam_cv()
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度点云变换类
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_rotz_mov(b,x,y,z,pc=None):
    T1=pc_mov(-x,-y,-z)
    T2=pc_rotz(b)
    T3=pc_mov(x,y,z)
    T=np.dot(np.dot(T1,T2),T3)
    return pc_trans(T,pc) if pc is not None else T

## 功能描述：
#   合并旋转矩阵和平移向量得到完整的变换矩阵
def gen_trans_mat(R,t):
    T=np.eye(4)
    T[:3,:3]=R
    T[3,:3]=t
    return T


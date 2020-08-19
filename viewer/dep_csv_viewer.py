#coding:utf-8
#!/usr/bin/python3

import numpy as np
import pylab as plt
import cv2

from IPython import embed

## 功能描述：
#   显示点云的辅助函数
def plot_pc(pc,title='',show=True,ax=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if ax is None: ax=plt.figure().gca(projection='3d')
    ax.plot(pc[:,0],pc[:,1],pc[:,2],'.k',markersize=0.1)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if show: plt.show()
    return ax

# 通过窗口选择需要查看的文件
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

fname=''
win = Tk()
def open_file():
    global fname
    global win
    fname=filedialog.askopenfilename(initialdir='./',
                                     title='open file',
                                     filetypes=(('CSV file','*.csv'),('All files','*')))
    win.destroy() 
ttk.Button(win, text="Select CSV File to view", command=open_file).grid()
win.mainloop()

print('[INF] fname:',fname)

CAM_WID,CAM_HGT = 320,240
CAM_CX,CAM_CY = 157.262,122.083
CAM_FX,CAM_FY = 210.783,204.817
CAM_DIST=np.array([-0.378924, 0.145035, -0.000970854, 6.20728e-005, -0.0133954])
CAM_MAT=np.array([CAM_FX,     0,CAM_CX,
                       0,CAM_FY,CAM_CY,
                       0,     0,     1]).reshape(3,3)
                  

# 构建速算表格
tab_x,tab_y=np.meshgrid(range(CAM_WID),range(CAM_HGT))
tab_x=tab_x.astype(np.float32)-CAM_CX
tab_y=tab_y.astype(np.float32)-CAM_CY

tab_dep_to_z=1.0/np.sqrt(tab_x**2/(CAM_FX**2)+tab_y**2/(CAM_FY**2)+1)
tab_dep_to_x=tab_x/CAM_FX*tab_dep_to_z
tab_dep_to_y=tab_y/CAM_FX*tab_dep_to_z

# 加载数据
img_dep=np.genfromtxt(fname,delimiter=',').reshape(CAM_HGT,CAM_WID).astype(np.float32)
img_dep=cv2.undistort(img_dep,CAM_MAT,CAM_DIST)
# 深度转点云坐标
pc_z=img_dep*tab_dep_to_z
pc_x=img_dep*tab_dep_to_x
pc_y=img_dep*tab_dep_to_y

pc=np.array([pc_x.ravel(),pc_y.ravel(),pc_z.ravel()]).T
if False:
    plot_pc(pc)
else:
    from pc_view import pc_view
    pc_view(pc+[-0.5,-0.5,0.5],
            CAM_CX,CAM_CY,CAM_FX,CAM_FY,CAM_WID,CAM_HGT,
            dmin=0,dmax=2)

#!/usr/bin/python3
#coding:utf-8

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

IMG_WID,IMG_HGT = 320,240
PAUSE=0.03
DMIN,DMAX=0,3.0

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


pc=np.genfromtxt(fname,delimiter=',').reshape(-1,3)
plot_pc(pc)


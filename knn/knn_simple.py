#!/usr/bin/python3
# coding=utf-8

import numpy as np

####################
# 最近邻查找，
# 使用简单但低效的遍历法
####################

## 简易的最近邻搜索，从点云pc中寻找点p的k近邻
# 返回点p的k近邻的点云序号和计算得到的距离
def find_knn_simple(p,pc,k=1,r=np.inf):
    if np.size(p)>3:    # 处理待检索点超过1个的情况
        return [find_knn_simple(p0,pc,k,r) for p0 in p.reshape(-1,3)]
        
    dist=np.sum((pc-p)**2,axis=1)
    idx=np.argsort(dist)[:k]
    return [pc[i] for i in idx if dist[i]<r**2]


if __name__=='__main__':
    np.random.seed(1234)
    pc=np.random.rand(1000,3)*2.0-1.0
    p =np.random.rand(3)*2.0-1.
    pnn=find_knn_simple(p,pc,k=50,r=0.5)
    dist=[np.linalg.norm(p-n) for n in pnn]
    for c,d in enumerate(dist):
        print('[%d] %f'%(c,d))

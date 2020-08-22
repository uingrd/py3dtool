#!/usr/bin/python3
# -*- coding: utf-8 -*-

########################################
# KNN
# 基于opencv库的KDtree
########################################

import numpy as np
import cv2

#   pc:     待查询的点云，每行是一个点的XYZ坐标
#   p:      检索点的XYZ坐标
# 输出:
#   pc_out  如果检索点只有1个的话，输出近邻点坐标列表
#           如果检索点有多个的话，pc_out是列表，每个元素是一个点的近邻坐标列表
#   flann   查询树
def find_knn(p,pc,k=1,r=np.inf,flann=None,ret_flann=False,verbose=False):
    if flann is None:
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=8),   # algorithm=1:KD-tree， trees=5:构建5棵树，加快搜索(1-16)
                                      dict(checks=50))              # 查询递归次数

    matches = flann.knnMatch(np.array(p,dtype=np.float32,ndmin=2),pc.astype(np.float32),k=k)   # KNN搜索
    pc_out=[pc[m.trainIdx].tolist() for m in matches[0] if m.distance<r]
    return pc_out if not ret_flann else (pc_out,flann)

def find_knn_batch(pc0,pc,k=1,r=np.inf,flann=None,ret_flann=False,verbose=False):
    if flann is None:
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=8),   # algorithm=1:KD-tree， trees=5:构建5棵树，加快搜索(1-16)
                                      dict(checks=50))              # 查询递归次数
    matches=flann.knnMatch(np.array(pc0,dtype=np.float32).reshape(-1,3),pc.astype(np.float32),k=k)   # KNN搜索
    pc_out_list=[]
    for m in matches:
        pc_out_list.append([pc[n.trainIdx].tolist() for n in m if n.distance<r])
    return pc_out_list if not ret_flann else(pc_out_list,flann)

        
####################
# 单元测试
####################
if __name__=='__main__':
    np.random.seed(1234)
    import knn_simple

    pc =np.random.rand(1000,3).astype(np.float32)               # 被检索的点云
    pc0=np.random.rand(4,3).astype(np.float32)                  # 搜索中心

    # 打印搜索结果
    p=pc0[0]
    print('nn search, p:',p)
    print('       ',find_knn(p,pc))
    print('expect:',knn_simple.find_knn_simple(p,pc))
    
    print('knn search')
    pc_nn_list1=find_knn_batch(pc0,pc,k=10)
    pc_nn_list2=knn_simple.find_knn_simple(pc0,pc,k=10)
    
    for p0,pc_nn1,pc_nn2 in zip(pc0,pc_nn_list1,pc_nn_list2):
        print('p0:',p0)
        for p1,p2 in zip(pc_nn1,pc_nn2):
            print('      ',np.array(p1,dtype=np.float32))
            print('expect',p2)
            

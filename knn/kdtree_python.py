#!/usr/bin/python3
# coding=utf-8

import heapq
import numpy as np

# http://code.activestate.com/recipes/579104-simple-python-point-kd-tree-no-scipynumpy-needed/

## 递归构建KD树
# points待分割的点集
# dim   总的维度
# i     对应当前切分的维度
def make_kd_tree(points, dim, i=0):
    if len(points) > 1:
        points=points[np.argsort(points[:,i]),:]
        i = (i+1) % dim
        half = len(points)//2
        return (make_kd_tree(points[: half   ,:], dim, i),    # 左树
                make_kd_tree(points[half + 1:,:], dim, i),    # 右树
                points[half,:])
    elif len(points) == 1:
        return (None, None, points[0,:])

## KNN搜索，使用了优先队列
# 注意：K大于节点数一半时，可能漏掉点
def get_knn(kd_node, point, k, dim, dist_func, return_distances=False, i=0, heap=None):
    is_root = not heap
    if is_root: heap = []
    
    if kd_node:
        dist = dist_func(point, np.array(kd_node[2])) # 节点距离(进入队列时将array变成了list，需要恢复)
        dx = kd_node[2][i] - point[i]       # 相对分割平面的位置
        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2].tolist()))   # 直接入队(heapq不能直接处理array)
        elif dist < -heap[0][0]:            # 替换队列中最远点
            heapq.heappushpop(heap, (-dist, kd_node[2].tolist()))
            
        i = (i+1) % dim
        
        # 根据相对分割平面的位置，选择搜索一侧子树
        get_knn(kd_node[1 if dx<0 else 0], point, k, dim,
                dist_func, return_distances, i, heap)

        # 如果到分割平面的距离平方(dx**2)小于当前近邻集合中最大距离平方(-heap[0][0])，则需要搜索另一侧树
        if dx**2 < -heap[0][0] or len(heap) < k: 
            get_knn(kd_node[1 if dx>=0 else 0], point, k, dim,
                    dist_func, return_distances, i, heap)
    
    if is_root: # 搜索完成
        idx=np.argsort([-h[0] for h in heap])
        neighbors=[(-heap[n][0],np.array(heap[n][1])) for n in idx]
        
        return neighbors if return_distances else [n[1] for n in neighbors]

# 最近邻搜索
def get_nearest(kd_node, point, dim, dist_func, return_distances=False, i=0, best=None):
    if kd_node:
        dist = dist_func(point, kd_node[2]) # 到节点的距离
        dx = kd_node[2][i] - point[i]       # 相对分割平面的位置
        # 保存最优结果
        if not best:                        
            best = [dist, kd_node[2]]       # 搜索刚开始
        elif dist < best[0]:
            best[0], best[1] = dist, kd_node[2]
        i = (i + 1) % dim
        
        # 根据相对分割平面的位置，选择搜索一侧子树
        get_nearest(kd_node[1 if dx<0 else 0], point, dim, dist_func, return_distances, i, best)
        
        # 如果到分割平面的距离平方(dx**2)小于当前最小距离平方(best[0])，则需要搜索另一侧树
        if dx**2 < best[0]:
            get_nearest(kd_node[1 if dx>=0 else 0], point, dim, dist_func, return_distances, i, best)
    return best if return_distances else best[1]

####################
# 单元测试
####################
if __name__=='__main__':
    from knn_simple import *
    np.random.seed(1234)

    N,D=5000,3

    # 生成随机点
    points=(np.random.rand(N,D)-0.5)*2.0  

    # 生成查询树
    kd_tree = make_kd_tree(points=points, dim=D)

    # 最近邻查询
    if True:
        print('\n[INF] kd-tree nn testing...')
        M=100
        pt=(np.random.rand(M,D)-0.5)*2.0    # 待查寻点集
        nn_ref=find_knn_simple(pt,points)   # 参考答案
        for p_ref,p in zip(nn_ref,pt):
            d,res=get_nearest(kd_node=kd_tree,
                              point=p,
                              dim=D,
                              dist_func=lambda a, b: np.sum((a-b)**2),
                              return_distances=True)
            # 检验       
            err=np.max(np.abs(res-p_ref))
            if err>0: 
                print('********** ',end='')
                print('err:',err)
            else:
                print('.',end='')

    # K近邻查询
    if True:
        print('\n[INF] kd-tree knn testing...')
        M,K=100,8
        pt=(np.random.rand(M,D)-0.5)*2.0        # 待查寻点集
        nn_ref=find_knn_simple(pt,points,k=K)   # 参考答案
        for p_ref,p in zip(nn_ref,pt):
            res=get_knn(kd_node=kd_tree,
                          point=p,
                          k=K,
                          dim=D,
                          dist_func=lambda a, b: np.sum((a-b)**2),
                          return_distances=True)

            # 检验
            err=np.sum([np.linalg.norm(p1[1]-p2) for p1,p2 in zip(res,p_ref)])
            if err>0: 
                print('********** ',end='')
                print(err)
            else:
                print('.',end='')

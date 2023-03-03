#!/usr/bin/python3
# coding=utf-8

import numpy as np
import knn_simple

####################
# 基于随机投影的近似最近邻查询
####################

# 建立随机投影树
def build_tree(pc,threshold=2,deg=0,deg_max=10,it_max=4):
    if len(pc)<threshold or deg>deg_max: return pc
    
    # 随机分割点云
    for _ in range(it_max):
        v=np.random.randn(pc.shape[-1]) # 生成随机向量
        sel=(pc*v).sum(axis=1)             
        pc0, pc1 = pc[sel>=0], pc[sel<0]# 点云划分
        if len(pc0) and len(pc1):
            return (v,
                    build_tree(pc0,threshold,deg+1,deg_max),
                    build_tree(pc1,threshold,deg+1,deg_max))
    # 无法分割点云
    return pc

# 在投影树上搜索近邻点云子集
def tree_search(p,tree):
    if not isinstance(tree,tuple): return tree  # 叶节点
    return tree_search(p, tree[1] if (p*tree[0]).sum()>=0 else tree[2])

# k近邻搜索，基于多棵随机投影树
def find_knn(p,pc,k=1,trees=None,num_tree=10):
    # 建树林
    if not trees:
        trees=[build_tree(pc) for _ in range(num_tree)]
    
    # 初次筛选查找
    pc0=[tuple(p0) for t in trees for p0 in tree_search(p,t)]
    
    # 去除重复的点
    pc0=np.array(list(set(pc0)))
    
    # 二次筛选
    return knn_simple.find_knn_simple(p,pc0,k)

    
####################
# 单元测试
####################
if __name__=='__main__':
    np.random.seed(1234)
    import knn_simple

    pc =np.random.randn(1000,3).astype(np.float32)  # 被检索的点云
    pc0=np.random.randn(4,3).astype(np.float32)     # 搜索中心

    # 打印搜索结果
    for p in pc0:
        print('find_knn:')
        for p0 in find_knn(p,pc,k=4):
            print('    ',p0)
        
        print('  expect:')
        for p0 in knn_simple.find_knn_simple(p,pc,4):
            print('    ',p0)

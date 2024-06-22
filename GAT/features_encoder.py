# -*- coding: utf-8 -*-
# @Time : 2023/7/10 22:42
# @Author : Crush
# @Version: 3.9.5
import rdkit
from rdkit import Chem
import numpy as np
import torch
import pickle
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
'''
rdkit中：
原子符号返回的是字符串 str  size=15
原子度返回的是 整型 int
原子手性 size=1 [0/1]
手性类型 one-hot size=3  同样用str()转换一下类型
原子杂化类型是内置数据类型，要用str()函数转换一下
芳香性 size=1 [0/1]
原子是否在环上 size=1
'''

atom_symbol = [['B'], ['C'], ['N'], ['O'],
               ['F'], ['Si'], ['P'], ['S'],
               ['Cl'], ['As'], ['Se'], ['Br'],
               ['Te'], ['I'], ['At']]

atom_degree = [[1], [2], [3], [4], [5]]

hybridizationType = [['S'], ['SP'], ['SP2'],
                     ['SP2D'], ['SP3'], ['SP3D'],
                     ['SP3D2'], ['OTHER'], ['UNSPECIFIED']]

hydrogens = [[0], [1],
             [2], [3], [4]]

chiralType = [['CHI_UNSPECIFIED'], ['CHI_TETRAHEDRAL_CW'], ['CHI_TETRAHEDRAL_CCW']]
'''
CW 顺时针R构型
CCW 逆时针S构型
'''
##################### bond encoder  #################
'''
化学键类型 size=4 ont-hot编码
立体结构       one-hot编码
'''
bondType = [['SINGLE'], ['DOUBLE'],
            ['TRIPLE'], ['AROMATIC']]

bondStereo = [['STEREONONE'], ['STEREOANY'],
              ['STEREOE'], ['STEREOZ']]


def features_encode(atom_features, encoder_name):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(atom_features)
    with open(f'{encoder_name}.pickle', 'wb') as f:
        pickle.dump(encoder, f)

features_encode(bondStereo, 'bondStereo_encoder')

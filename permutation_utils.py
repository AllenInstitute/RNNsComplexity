#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:09:35 2023

@author: dana.mastrovito
"""

import numpy as np
import os
import networkx as nx



def get_degree_distribution(CM):
    DG = nx.from_numpy_array(CM, create_using=nx.DiGraph)
    degree_sequence = sorted((d for n, d in DG.degree()), reverse=True)
    return degree_sequence
    
def get_indegree_distribution(CM):
    DG = nx.from_numpy_array(CM, create_using=nx.DiGraph)
    degree_sequence = sorted((d for n, d in DG.in_degree()), reverse=True)
    return degree_sequence

def get_outdegree_distribution(CM):
    DG = nx.from_numpy_array(CM, create_using=nx.DiGraph)
    degree_sequence = sorted((d for n, d in DG.out_degree()), reverse=True)
    return degree_sequence


def block_permute(file,rng,suffix = "",save = False):
    CM = np.load(file)
    permuted = np.zeros_like(CM)
    permuted = permuted.flatten()
    flat = CM.flatten()
    pos = np.where(flat >0)[0]
    ppermute = rng.choice(pos, pos.size, replace = False)  
    permuted[pos] = ppermute
    neg = np.where(flat <0)[0]
    npermute = rng.choice(neg, neg.size, replace = False)
    permuted[neg]  =  npermute
    permuted = permuted.reshape(CM.shape)
    if save:
        np.save(os.path.join(os.path.dirname(file), os.path.basename(file) +"_block_permuted"+suffix+".npy"),permuted)
    return permuted
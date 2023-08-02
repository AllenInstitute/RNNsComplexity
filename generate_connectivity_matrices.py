#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:21:00 2023

@author: dana mastrovito
"""
#exec(open("generate_connectivity_matrices.py").read())
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(cell1, cell2):
    return np.sum(np.square(cell1 - cell2))
    

post = pd.read_feather("post_cell_table_v1dd_all_proofread.feather")
synapse = pd.read_feather("synapse_table_v1dd_all_proofread.feather")
pre = pd.read_feather("pre_cell_table_v1dd_all_proofread.feather")

layer = ["23","4"]
strlayer = "_".join(layer)
ncells = 198

pre_layer = pre[pre['soma_layer'].isin(layer)]
pre_positions =  np.stack(pre_layer['pt_position'].to_numpy())
xmin = np.min(pre_positions[:,0])
xmax = np.max(pre_positions[:,0])
ymin = np.min(pre_positions[:,1])
ymax = np.max(pre_positions[:,1])
zmin = np.min(pre_positions[:,2])
zmax = np.max(pre_positions[:,2])
pre_mid = (np.mean([xmin,xmax]),np.mean([ymin, ymax]),np.mean([zmin, zmax]))

pre_distance = []
for cell in range(len(pre_layer)):
    pre_distance.append(euclidean_distance(pre_mid,pre_positions[cell]))


pre_distance = np.array(pre_distance)
pre_order = list(np.argsort(pre_distance)[:ncells])


pre_cell_idx = pre_layer['pre_column_idx'].iloc[pre_order]
pre_cell_pt_index = pre_layer['pt_root_id'][pre_layer['pre_column_idx'].isin(pre_cell_idx)]
synapses = synapse[(synapse['pre_pt_root_id'].isin(pre_cell_pt_index)) & (synapse['post_pt_root_id'].isin(pre_cell_pt_index))]
pre_cell_pt_index = pre_cell_pt_index.to_list()


CM = np.zeros((ncells, ncells))
for synapse in range(len(synapses)):
    preidx = pre_cell_pt_index.index(synapses.iloc[synapse]['pre_pt_root_id'])
    postidx = pre_cell_pt_index.index(synapses.iloc[synapse]['post_pt_root_id'])
    if pre[pre['pt_root_id'] == synapses.iloc[synapse]['pre_pt_root_id']]['cell_type'].item() == 'PYC':
        CM[preidx,postidx] += synapses.iloc[synapse]['size']
    else:
        CM[preidx,postidx] -= synapses.iloc[synapse]['size']

np.save(strlayer+".npy",CM)

CM = CM/np.max(np.abs(CM))

CMf = CM.flatten()
CMf = CMf[np.nonzero(CMf)]
plt.clf()
plt.hist(CMf)
plt.savefig(strlayer+".png")


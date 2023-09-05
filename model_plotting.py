#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:14:39 2023

@author: dana.mastrovito
"""

from utils import GetModelFiles, GetModelFileParameters
import matplotlib
import numpy as np


params = GetModelFileParameters()
v1params = GetModelFileParameters(v1dd = True, add_dir = "Dales",suffix="23_4_")
k = 1.2
snr = 2.2


MESOSCOPIC = True
SUPPLEMENTARY = False
V1DD = True

nNN28p1_files = [GetModelFiles(nNN = 28, p=1.0,gain =g) for g in params['gain']]
nNN28p1D_files = [GetModelFiles(nNN = 28, p=1.0,gain =g,add_dir = "Dales") for g in params['gain']]
nNN198p0_files = [GetModelFiles(nNN = 198, p=0.0,gain =g) for g in params['gain']]



#EM_23_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix = "23_") for g in params['gain']]
EM_23_4_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix = "23_4_",add_dir="Dales") for g in v1params['gain']]
#EM_23_permuted_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_permuted_") for g in params['gain']]
EM_23_4_permuted_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_permuted_") for g in params['gain']]
EM_23_4_permuted_topology_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_permutedT_") for g in params['gain']]

EM_23_4_block_permuted_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_block_permuted_",add_dir="Dales") for g in params['gain']]
EM_23_4_block_permuted_topology_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_block_permuted_",add_dir="Dales/topology") for g in params['gain']]
EM_23_4_flipped_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_flipped_",add_dir="Dales") for g in v1params['gain']]

EM_23_4_topology_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_topology_") for g in params['gain']]



#EM_23_4_transformed_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_transformed_") for g in params['gain']]
dd_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,degree_distribution=True) for g in params['gain']]


density_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,density = True) for g in params['gain']]
density_thresh_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,density = True,suffix="_thresh") for g in params['gain']]
density_sparse_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,density = True,suffix="_sparsify") for g in params['gain']]
density_threshto_files = [GetModelFiles(nNN = 198, p=0.0, gain = g, density = True, suffix="_threshto5573") for g in params['gain']]

ps = ["p = 0","p = 0.2","p = 0.5","p = 0.8","p = 1"]
#"Density"
if MESOSCOPIC:
    model_types = ['nNN = 28', 'nNN = 198' ,"Meso","Meso_threshtoEM"]        
    model_files = [nNN28p1_files,nNN198p0_files,density_files,density_threshto_files] 
    suffixes = ["","","","_threshto5573"]
    add_dirs = ["","","",""] 
    nNN =[28,198,198,198]
    probs = [1.0,1.0, 0.0,0.0]
    file_add = "meso"
    nNN28cm = matplotlib.colormaps['Blues']   
    nNN198cm = matplotlib.colormaps['Oranges']
    nNN28colors = [nNN28cm(g) for g in np.linspace(.25,1,len(params['p']))]  
    nNN198colors = [nNN198cm(g) for g in np.linspace(.25,1,len(params['p']))]  
    mesocm = matplotlib.colormaps['Greens'] 
    mesocolors = [mesocm(g) for g in np.linspace(.5,1.0,2)]  
    colors = [nNN28colors[-1],nNN198colors[-1],mesocolors[0],mesocolors[1]]
    linestyles = ['solid','solid','solid','solid']
elif V1DD:
    model_types = ['nNN = 28D',"V1 2/3 4","V1 2/3 4 permuted","V1 Dales BP","V1 Dales +-"]
    model_files = [nNN28p1D_files,EM_23_4_files, EM_23_4_permuted_files,EM_23_4_block_permuted_files, EM_23_4_flipped_files]
    suffixes = ["","23_4_","23_4_permuted_","23_4_block_permuted_", "23_4_flipped_"]
    add_dirs = ["Dales","Dales","","Dales","Dales"]
    nNN = [28,198,198,198,198,198]
    probs = [1.0,0.0,0.0,0.0,0.0,0.0]
    autumn = matplotlib.colormaps['RdPu']  
    modelcolors = [autumn(g) for g in np.linspace(.45,.9,4)]  
    colors = ["deepskyblue",modelcolors[3],modelcolors[0],modelcolors[1],modelcolors[2]]
    linestyles = ["solid","dashdot","solid","solid","dotted"] 
elif SUPPLEMENTARY:
    model_types = ["DD","V1 2/3 4 permutedT", "V1 Dales BPT",'V1 T']
    model_files = [dd_files, EM_23_4_permuted_topology_files, EM_23_4_block_permuted_topology_files,EM_23_4_topology_files] 
    add_dirs = ["","Dales","","Dales/topology","Dales"]
    suffixes = ["","23_4_permutedT_", "23_4_block_permuted_","23_4_topology_"]           
    nNN  = [198,198,198,198]
    probs = [0.0,0.0,0.0,0.0]
    file_add = "supp"
    autumn = matplotlib.colormaps['RdPu']  
    modelcolors = [autumn(g) for g in np.linspace(.45,.9,4)]  
    colors  = [ "lime",modelcolors[0],modelcolors[1],modelcolors[2]]
    linestyles = ['solid','dashdhot','dashdot','dashed']        
        

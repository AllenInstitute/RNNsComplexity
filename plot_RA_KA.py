# -*- coding: utf-8 -*-
"""
Code to load saved data and generate plots for representation alignment and tangent kernel alignment

@author: hyliu
"""



# get transition point

# from utils import *
import numpy as np
import pandas as pd
# import torch
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import argparse

# dir = "/allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity"
# params = GetModelFileParameters(Density = False,em = False,dir = dir)

params = {'nNN': [4, 8, 16, 27, 32, 64, 128, 198], 
          'p': [0.0, 0.2, 0.5, 0.8, 1.0], 
          'gain': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]}
transition_df = pd.read_csv('gain_transitions.csv')
gain_list = params['gain']

for nNN_idx in range(7):
        
    all_rep_sim_list = []
    all_kernel_alignment_list = []
    all_rep_sim_list_std = []
    all_kernel_alignment_list_std = []
    
    # load data
    nNN = params['nNN'][nNN_idx]
    for pp in range(len(params['p'])):
        prob = params['p'][pp]        
        
        rep_sim_list = np.load('./saved_rep_align/rep_align_mean_nNN' + str(nNN) + '_p' + str(prob) + '.npy')
        rep_sim_list_std = np.load('./saved_rep_align/rep_align_std_nNN' + str(nNN) + '_p' + str(prob) + '.npy')
        kernel_alignment_list = np.load('./saved_NTK_align/NTK_align_mean_nNN' + str(nNN) + '_p' + str(prob) + '.npy')
        kernel_alignment_list_std = np.load('./saved_NTK_align/NTK_align_std_nNN' + str(nNN) + '_p' + str(prob) + '.npy')
        
        all_rep_sim_list.append(rep_sim_list)
        all_kernel_alignment_list.append(kernel_alignment_list)
        all_rep_sim_list_std.append(rep_sim_list_std)
        all_kernel_alignment_list_std.append(kernel_alignment_list_std)
    
    
    # plot rep align
    fig = plt.figure(figsize=(6, 3))
    cmap = plt.cm.cool
    plt.title('nNN = ' + str(nNN))
    for pp, rep_sims in enumerate(all_rep_sim_list):
        # plt.plot(gain_list, rep_sims, marker='s', linestyle='-', color=cmap(pp * 80), label=str(params['p'][pp]))
        rep_sims_std = all_rep_sim_list_std[pp]
        plt.errorbar(gain_list, rep_sims, yerr=rep_sims_std, fmt='-s', color=cmap(pp * 80), \
                      label=str(params['p'][pp]), zorder=1)
        # label transition
        if nNN > 4:
            transition_val = transition_df.iloc[nNN_idx * len(params['p']) + pp, 3]
            transition_idx = gain_list.index(transition_val)
            # plt.plot(transition_val, rep_sims[transition_idx], 'r*', markersize=20, zorder=2) # mark the transition point 
            plt.axvline(x=transition_val, linestyle='dotted', color=cmap(pp * 80), zorder=2)
    if (nNN==4) or (nNN==8): 
        plt.legend(title="Rewiring prob")
        plt.ylabel('Representation alignment')    
    plt.show()
    fig.savefig('rep_align_nNN' + str(nNN) + '.pdf')
    
    # plot NTK align
    fig2 = plt.figure(figsize=(6, 3))
    cmap = plt.cm.cool
    plt.title('nNN = ' + str(nNN))
    for pp, kernel_align in enumerate(all_kernel_alignment_list):
        # plt.plot(gain_list, kernel_align, marker='s', linestyle='-', color=cmap(pp * 80), label=str(params['p'][pp]))
        kernel_align_std = all_kernel_alignment_list_std[pp]    
        if nNN > 4:
            transition_val = transition_df.iloc[nNN_idx * len(params['p']) + pp, 3]
            transition_idx = gain_list.index(transition_val)
            
            # Plot the points before the transition th element with full opacity
            plt.errorbar(gain_list[:transition_idx+1], kernel_align[:transition_idx+1], yerr=kernel_align_std[:transition_idx+1], fmt='-s', \
                          color=cmap(pp * 80), label=str(params['p'][pp]), zorder=1)        
            # Plot the points from the  transition th element onwards with lower opacity
            plt.errorbar(gain_list[transition_idx:], kernel_align[transition_idx:], yerr=kernel_align_std[transition_idx:], fmt='-s', \
                          color=cmap(pp * 80), alpha=0.2, zorder=1)
            
            # plt.plot(transition_val, kernel_align[transition_idx], 'r*', markersize=20, zorder=2) # mark the transition point  
            plt.axvline(x=transition_val, linestyle='dotted', color=cmap(pp * 80), zorder=2)
        else:
            plt.errorbar(gain_list, kernel_align, yerr=kernel_align_std, fmt='-s', color=cmap(pp * 80), \
                          label=str(params['p'][pp]), zorder=1)
    if (nNN==4) or (nNN==8): 
        plt.legend(title="Rewiring prob")
        plt.ylabel('Tangent kernel alignment')    
    plt.show()
    fig2.savefig('NTK_align_nNN' + str(nNN) + '.pdf')

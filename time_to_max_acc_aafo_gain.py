#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:52:11 2023

@author: dana
"""
#exec(open("time_to_max_acc_aafo_gain.py").read())
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import pickle
import numpy as np
from utils import  GetModelFiles,GetModelFileParameters
from utils import GetModelWeights,GetInitializedModel
import torch
import pickle
import pandas as pd
import matplotlib

device = 'cpu'
NOISE = False
FULLY_TRAINED = True

dir ="/allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity/RNN/Gaussian"
if NOISE:
    dir = os.path.join(dir,"noise")

if FULLY_TRAINED:
    acc = 90
    dir = os.path.join(dir, "fully_trained")
    readfile = "time_to_acc_"+ str(acc)+"_"
    outfile = "time_to_acc_" + str(acc) + "_aafo_gain"
else:
    acc = ""
    readfile = "time_to_max_acc_"
    outfile = "time_to_max_acc_aafo_gain"

pcistdir = os.path.join(dir, "pcist")


#gain_transitions = [[3.0],[3.0],[2.0],[2.0],[1.5],[1.5]]
gain_transitions = pd.read_csv(os.path.join(dir,"jacobian_product_trajectory",'gain_transitions_init.csv'))

nfiles = 10
params = GetModelFileParameters()


autumn = matplotlib.colormaps['cool']
#colors = [autumn(g) for g in np.linspace(0,1,len(params['nNN']))]
colors = [autumn(g) for g in np.linspace(0,1,len(params['p']))]


plt.clf()
fix, ax = plt.subplots(4,2,sharex = "all",sharey='all')
axes = ax.flat
for n, nNN in enumerate(params['nNN']):
    with open(os.path.join(pcistdir,readfile+"nNN"+str(nNN)+".pkl"),'rb') as f:
        time_to_max = pickle.load(f)# dict max_acc[gain,p,file], nepochs[gain,p, file], files[gain, p,file]
    for p, prob in enumerate(params['p']):
        ttmm = [np.mean(time_to_max['nepochs'][g,p,:])/39.0 + 1.0 for g,gain  in enumerate(params['gain'])]
        ttmsd =[np.std(time_to_max['nepochs'][g,p,:]/39.0 + 1.0)/np.sqrt(10) for g,gain in enumerate(params['gain'])] 
       
        '''
        wg = np.where(np.isnan(ttmm) == False)[0]
        ncolors = [colors[n] for i in range(len(wg))]
   
        if nNN != 198 and gain_transitions[n] in list(np.array(params['gain'])[wg]):
            idx = list(np.array(params['gain'])[wg]).index(gain_transitions[n])
            ncolors[idx] = (1.0,0.0,0.0,1.0)
            print(ncolors)
        '''
        gain_transition = gain_transitions['Transition Point'][(gain_transitions['nNN'] == nNN) & (gain_transitions['p'] == float(prob))].item()
        gidx = params['gain'].index(gain_transition)
        axes[n].errorbar(np.array(params['gain']),np.array(ttmm),yerr= np.array(ttmsd),color = colors[p],alpha = 0.6,marker = 's',label = str(prob))
        #axes[n].scatter(np.array(params['gain'])[wg][idx],np.array(ttmm)[wg][idx],color = ncolors[idx],marker = "*" )
        ylim = axes[n].get_ylim()
        axes[n].plot(np.repeat(np.array(params['gain'])[gidx],5),np.linspace(ylim[0],ylim[1],5),color = colors[p],linestyle = 'dashed')
    axes[n].set_xticks(np.arange(6),labels = [str(gain) for gain in np.arange(6)])
    axes[n].set_title(label = "nNN = " + str(nNN),fontsize = 14)
    axes[n].tick_params(axis="both",labelsize = 14)
    

    #plt.ylim(0,100)
    #axes[n].set_yscale("log")
axes[2].set_ylabel("epochs till max acc",fontsize = 14)
axes[6].set_xlabel("Gain",fontsize = 14)
#axes[5].set_xlabel("Gain")
#plt.ylabel("n Batches",fontsize = 14)
#plt.title("Trained PCIst")
#axes[5].legend( title = "Rewiring p",fontsize = 14)
#plt.gca().add_artist(legend)
[ax.set_yscale("log") for ax in axes]
plt.tight_layout()
plt.savefig(os.path.join(pcistdir,outfile+"_legend.png"),dpi = 600)
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:14:08 2023

@author: dana.mastrovito
"""
#exec(open("MaxACC_afo_Gain.py").read())
import os
from utils import  GetModelFiles,GetModelFileParameters, GetAdjacencyRank
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import matplotlib.markers as mmark
import matplotlib.patches as mpatches
import pickle
import pandas as pd


NOISE = False
DENSITY = False
EM = False
FC = False
V1DD = False
ZOOM = True
if ZOOM:
    ymin = 99

suffix = ""

dir= "/allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity/RNN"
if EM:
    dir = os.path.join(dir, "EM_column")
elif V1DD:
    dir = os.path.join(dir, "EM_column",'v1dd','Dales','relu')
    add_dir = "Dales/relu"
    suffix = "23_4_"
elif DENSITY:
    dir = os.path.join(dir,"Density")
    suffix = "_thresh"
else:
    dir = os.path.join(dir, "Gaussian")
    add_dir = ""



#plt.rcParams.update({'font.size': 10})

params =  GetModelFileParameters(em=EM,Density = DENSITY,v1dd= V1DD, add_dir = add_dir, suffix = suffix)
#plt.clf()
markers = ['v',"^","<",">","o"] 

autumn = matplotlib.colormaps['cool']
colors = [autumn(g) for g in np.linspace(0,1,len(params['p']))]
strnNN = [str(nNN) for nNN in params['nNN']]
strprobs = [str(prob) for prob in params['p']]

legendp = [mmark.MarkerStyle(m) for m in markers]
legendnNN = []
legendps = []

if NOISE:
    dir = os.path.join(dir, "noise")

if FC:
    params['nNN'] = [198]
    suffix = "_FC_198"
else:
    if not (DENSITY or EM):
        #params['nNN'] = [nNN for nNN in params['nNN'] if nNN not in [27,198]]
        pass
    

pcistdir = os.path.join(dir, "pcist")


AllACCs = []
AllTrainACCs = []
if not (EM or DENSITY or FC or V1DD):
    fig, ax = plt.subplots(4,2,sharex = "all",sharey='row')
    axes = ax.flat
else:
    fig, ax = plt.subplots(1,1,sharex = "all",sharey='all')
    axes  = [ax]

gain_transitions = pd.read_csv(os.path.join(dir,'jacobian_product_trajectory','gain_transitions_init.csv'))
for n,nNN in enumerate(params['nNN']):
    with open(os.path.join(pcistdir,"time_to_max_acc_nNN"+str(nNN)+suffix+".pkl"),'rb') as f:
        time_to_max = pickle.load(f)# dict max_acc[gain,p,file], nepochs[gain,p, file], files[gain, p,file]
    AllACCs.append(time_to_max['max_acc'])
    AllTrainACCs.append(time_to_max['train_acc'])
    if not (EM or DENSITY or FC or V1DD):
        axes[n].set_title('nNN = ' + str(nNN))
    if n < 2:
        if ZOOM:
            axes[n].set_ylim(96,101)
        else:
            pass
        #axes[n].set_ylim((50,110))
    elif n < 4:
        if ZOOM:
           axes[n].set_ylim(ymin,101)
        else:
            axes[n].set_ylim((60,110))
    elif n < 6:
        if ZOOM:
            axes[n].set_ylim(ymin,101)
        else:
            axes[n].set_ylim((70,110))
    else:
        if ZOOM:
            axes[n].set_ylim(ymin,101)
        else:
            axes[n].set_ylim((80,110))
    #axes[n].set_yscale('log')
    axes[n].tick_params(axis = "x",labelsize = 14) 
    axes[n].tick_params(axis = "y",labelsize = 14) 
   
    for p,prob in enumerate(params['p']):
        #if p ==0:
        #    legendnNN.append(axes[n].errorbar(params['gain'],np.mean(time_to_max['max_acc'][:,p,:],1), \
        #            yerr = np.std(time_to_max['max_acc'][:,p,:],1),marker = markers[p],color = colors[p],alpha =0.5,label = strnNN[n] ))
        std = np.nanstd(time_to_max['max_acc'][:,p,:],1)
        train_std = np.nanstd(time_to_max['train_acc'][:,p,:],1)
        if n == 0:
            #print(p,"plotted")
            #legendps.append(axes[n].errorbar(params['gain'],np.nanmean(time_to_max['max_acc'][:-1,p,:],1), \
            #         yerr = np.nanstd(time_to_max['max_acc'][:-1,p,:],1),marker= markers[p],color = colors[p],alpha = 0.5, label = strprobs[p]))
            axes[n].plot(params['gain'],np.mean(time_to_max['max_acc'][:,p,:],1),marker= markers[p],color = colors[p],alpha = 0.5, label = strprobs[p])
            legendps.append(axes[n].plot(params['gain'],np.mean(time_to_max['max_acc'][:,p,:],1),marker= 's',color = colors[p],alpha = 0.5, label = strprobs[p]))            
        else:
            #axes[n].errorbar(params['gain'],np.nanmean( time_to_max['max_acc'][:-1,p,:],1),\
            #        yerr = np.nanstd(time_to_max['max_acc'][:-1,p,:],1),marker= markers[p],color = colors[p],alpha = 0.5, label = strprobs[p])
            axes[n].plot(params['gain'],np.mean(time_to_max['max_acc'][:,p,:],1),marker= markers[p],color = colors[p],alpha = 0.5, label = strprobs[p])
        axes[n].fill_between(params['gain'],np.mean(time_to_max['max_acc'][:,p,:],1)-std, np.mean(time_to_max['max_acc'][:,p,:],1)+std,alpha = 0.1,color = colors[p]) 
        #axes[n].plot(params['gain'],np.nanmean(time_to_max['train_acc'][:,p,:],1),\
        #             color = colors[p],alpha = 0.5,linestyle='dashed')
        #axes[n].fill_between(params['gain'],np.mean(time_to_max['train_acc'][:,p,:],1)-train_std, np.mean(time_to_max['train_acc'][:,p,:],1)+train_std,alpha = 0.1,color = colors[p]) 
        
        gain_transition = gain_transitions['Transition Point'][(gain_transitions['nNN'] == nNN) & (gain_transitions['p'] == float(prob))].item()
        gidx = params['gain'].index(gain_transition)
        ylim = axes[n].get_ylim()
        axes[n].plot(np.repeat(params['gain'][gidx],5),np.linspace(ylim[0],ylim[1],5),linestyle='dashed',color=colors[p])
        mean = np.mean(time_to_max['max_acc'][:,p,:],1)
        print(nNN, prob, "transition = ", gidx, "max acc at ", np.where(mean == np.max(mean))[0])
        #axes[n].set_yscale('log')
        #axes[n].plot(params['gain'][gidx],np.nanmean(time_to_max['max_acc'][:,p,:],1)[gidx],marker="*",color=colors[p],markersize=15)                    
#nNN_legend = plt.legend(handles = legendnNN, title = "nNN",loc=(0,0.2))
#nNN_legend = plt.legend(legendnNN,labels = strnNN,title="Nearest Neigbhors",loc="lower left")
#plt.gca().add_artist(nNN_legend)

#plt.xscale('log')
#plt.ylim((0,105))
axes[0].set_ylabel("Accuracy",fontsize  = 14)
#axes[0].legend(handles = legendps, title = "p Rewiring",loc ="lower right",fontsize=6)
if not (EM or DENSITY or FC or V1DD):
    axes[-2].set_xlabel("Gain",fontsize = 14)
          
with open(os.path.join(dir,'Accuracies.csv'), 'w') as f:
    for inn, inNN in enumerate(params['nNN']):
        for ig,g in enumerate(params['gain']):
            for ip, p in enumerate(params['p']):
                macc = np.mean(AllACCs[inn][ig],1)
                sd = np.std(AllACCs[inn][ig],1)
                f.write(",".join((str(inNN),str(g),str(p),"{:.3f}".format(macc[ip]),"{:.3f}".format(sd[ip]))))
                f.write('\n')
          
'''           
    
#transitions =[[5,3,3,3,3],list(np.repeat(3,5)),[3,2,2,2,2],[1.75,1.75,2.0,1.75,1.75],list(np.repeat(1.75,5)),list(np.repeat(1.5,5)),list(np.repeat(1.5,5))]           
#gain_transitions = [3.0,3.0,2.0,1.75,1.75,1.5,1.5]
for inn, inNN in enumerate(params['nNN']):
    for p, prob in enumerate(params['p']):
        macc = np.mean(AllACCs[inn][:,p,:],1)
        wg = np.where(macc == np.nanmax(macc))[0][0]
        #gidx = params['gain'].index(gain_transitions[inn])
        gain_transition = gain_transitions['Transition Point'][(gain_transitions['nNN'] == inNN) & (gain_transitions['p'] == float(prob))].item()
        gidx = params['gain'].index(gain_transition)
        if wg != gidx:
            print(inNN, prob, params['gain'][gidx],"max at",params['gain'][wg])
            #print(inNN,prob,"max acc not at GcnNN")
            #print(inNN,prob,macc[gidx-1],macc[gidx],macc[gidx+1])
        else:
            print(inNN, prob, "max at transition")
        #axes[inn].plot(params['gain'][wg],macc[wg],marker='X',markersize=10,color= colors[p])
'''        
 
    
if ZOOM:
    suffix = suffix+"_zoom"
plt.tight_layout()        
plt.savefig(os.path.join(dir,"MaxAcc_afo_Gain_scatter"+ suffix+".png"),dpi = 600)

'''
4 True
4 98.21875 98.7890625 97.4375
8 True
8 99.3359375 99.4296875 96.890625
16 True
16 99.28125 99.625 99.609375
32 True
32 99.7109375 99.78125 99.6640625
64 True
64 99.1484375 99.8125 99.796875
128 True
128 99.5703125 99.9140625 99.890625
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 07:26:19 2023

@author: dana.mastrovito
"""
#exec(open("rich_lazy_perprob.py").read())
import matplotlib.pyplot as plt
import torch
import Network
import numpy as np 
import os
import glob
import pickle
import matplotlib

from utils import GetModelWeights
from utils import GetModelFiles,GetModelFileParameters
from utils import GetModelReadout,GetModelInputWeights
import matplotlib
import matplotlib as mpl
import pandas as pd

def plot_afo_gain():
    plt.clf()    
    
def plot_afo_Ipcist():
    plt.clf()
 
    
NARROW = False
PIXEL = False
DIGITS = False
V1DD = False
suffix = ""
if DIGITS:    
    dir="/allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity/RNN/digits/Gaussian"
elif NARROW:
    dir="/allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity/RNN/narrow/Gaussian"
elif PIXEL:
    dir="/allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity/RNN/pixel_by_pixel/Gaussian"
elif V1DD:
    dir="/allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity/RNN/EM_column/v1dd"
    suffix = "23_4_"
else:
    dir="/allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity/RNN/Gaussian"
    
outdir = os.path.join(dir,"lazy")
if not os.path.exists(outdir):
    os.mkdir(outdir)

fontsize = 14   
params =  GetModelFileParameters(digits = DIGITS,pixel = PIXEL,narrow = NARROW,v1dd = V1DD,suffix = suffix)

#params['gain'] = [gain for gain in params['gain'] if gain not in [0.5,1.0]]
#params['nNN'] = [nNN for nNN in params['nNN'] if nNN not in [16]]

#transitions =[[5,3,3,3,3],list(np.repeat(3,5)),[3,2,2,2,2],list(np.repeat(2,5)),list(np.repeat(1.5,5)),
#              list(np.repeat(1.5,5))]
if os.path.exists(os.path.join(dir,'jacobian_product_trajectory','gain_transitions_init.csv')):
    gain_transitions = pd.read_csv(os.path.join(dir,'jacobian_product_trajectory','gain_transitions_init.csv'))
    GTs = True
else:
    GTs = False

#gains = [0.5, 0.75, 1.0, 1.5, 2.0,3.0,5.0,25.0]
#probs = [0.0,0.2,0.5,0.8,1.0]
cmap = matplotlib.colormaps['cool']
bounds = params['gain']
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


autumn = matplotlib.colormaps['tab10']
colors = [autumn(nNN) for nNN in np.linspace(0,1,len(params['nNN']))]
markers = ['v',"^","<",">","o"] 



for p,prob in enumerate(params['p']):
    plt.clf()
    fig1,ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    for n,nNN in enumerate(params['nNN']):
        gIdiffs = []
        gHdiffs = []
        gRdiffs = []
        gIstd = []
        gHstd = []
        gRstd = []
        for g,gain in enumerate(params['gain']):
            files = GetModelFiles(nNN, prob,gain=gain,NetworkType='RNN',digits = DIGITS,narrow=NARROW, pixel = PIXEL,v1dd = V1DD, suffix = suffix )
            Idiffs = []
            Hdiffs = []
            Rdiffs = []
            for file in files:
                print(file)
                Ireadout = np.sqrt(np.sum(GetModelReadout(file,initial = True)[0].numpy()**2))# zero is the weight rather than bias
                Freadout = np.sqrt(np.sum(GetModelReadout(file,initial = False)[0].numpy()**2))
                #diffReadout = torch.mean(torch.abs(Freadout - Ireadout)).numpy()
                diffReadout = np.sqrt(np.sum((Freadout - Ireadout)**2)) #np.sqrt(np.sum((Freadout - Ireadout).numpy()**2))
                IWeight  = GetModelWeights(file,initial = True)
                mask = np.nonzero(IWeight)
                FWeight  =  GetModelWeights(file,initial = False)
                diffHidden = np.sqrt(np.sum((FWeight[mask] - IWeight[mask])**2))
                IinputWeight = GetModelInputWeights(file,initial = True).numpy().flatten()
                FinputWeight =GetModelInputWeights(file,initial = False).numpy().flatten()
                diffInput = np.sqrt(np.sum((FinputWeight - IinputWeight)**2))
                Idiffs.append(diffInput)
                Hdiffs.append(diffHidden)
                Rdiffs.append(diffReadout)
            #plt.subplot(2,3,1)
            
            #plt.subplot(1,3,1)
            gIdiffs.append(np.mean(np.array(Idiffs)))
            gHdiffs.append(np.mean(np.array(Hdiffs)))
            gRdiffs.append(np.mean(np.array(Rdiffs)))
            gIstd.append(np.std(np.array(Idiffs)))
            gHstd.append(np.std(np.array(Hdiffs)))
            gRstd.append(np.std(np.array(Rdiffs)))
        ax1.errorbar(params['gain'],gIdiffs,yerr=gIstd,color=colors[n],marker='o',label = str(nNN))
        #ax1.plot(params['gain'],gIdiffs,color=colors[n],alpha = 0.5)
        #ax1.plot(params['gain'][gidx],gIdiffs[gidx],color='red',marker="*",alpha = 0.5,markersize = 15)
        
        #plt.subplot(2,3,2)
        #axes=plt.gca()
        #axes.set_aspect("equal",'box')            

        #plt.subplot(1,3,2)
        ax2.errorbar(params['gain'],gRdiffs,yerr=gRstd,color=colors[n],marker ='o')
        #ax2.plot(params['gain'],gRdiffs,color=colors[n],alpha = 0.5)
        #ax2.plot(params['gain'][gidx],gRdiffs[gidx],color='red',marker="*",alpha=0.5,markersize = 15)
        
        #ax2.title("Readout Layer")
        #ax2.xlabel("gain")
        #plt.subplot(2,3,3)
        #axes=plt.gca()
        #axes.set_aspect("equal",'box')            

        #plt.subplot(1,3,3)
        ax3.errorbar(params['gain'],gHdiffs,yerr=gHstd,color=colors[n],marker='o') 
        #ax3.plot(params['gain'],gHdiffs,color=colors[n],alpha = 0.5) 
        #ax3.plot(params['gain'][gidx],gHdiffs[gidx],color='red',marker="*",alpha = 0.5,markersize = 15)

        if GTs:
            gain_transition = gain_transitions['Transition Point'][(gain_transitions['nNN'] == nNN) & (gain_transitions['p'] == float(prob))].item()
            gidx = params['gain'].index(gain_transition)
            transition = params['gain'][gidx]
        
            ax1.plot(np.repeat(transition,5),np.linspace(0,8.5,5),color = colors[n],linestyle = 'dashed',alpha = 0.5)
            ax2.plot(np.repeat(transition,5),np.linspace(0,3.5,5),color = colors[n],linestyle = 'dashed',alpha = 0.5)
            ax3.plot(np.repeat(transition,5),np.linspace(0,5.5,5),color = colors[n],linestyle = 'dashed',alpha = 0.5)  
            #ax1.plot(params['gain'][gidx],gIdiffs[gidx],color='red',marker="*",alpha = 0.5,markersize = 15)
            #ax2.plot(params['gain'][gidx],gRdiffs[gidx],color='red',marker="*",alpha=0.5,markersize = 15)
            #ax3.plot(params['gain'][gidx],gHdiffs[gidx],color='red',marker="*",alpha = 0.5,markersize = 15)
 
    
        
       

        ax1.set_title("Input Layer",fontsize = fontsize)
        ax1.set_ylabel("Norm Weight Change",fontsize = fontsize)
        ax1.set_xlabel("Gain",fontsize = fontsize)            
        if prob in [ 0.0, 1.0]:
            ax1.legend(title = "nNN",fontsize = fontsize)
        ax1.tick_params(axis = "both",labelsize = fontsize) 
        fig1.tight_layout()
        fig1.savefig(os.path.join(outdir,'input_weight_change_p'+str(prob)+'.png'),dpi = 300)            
        
        ax2.set_title("Readout Layer",fontsize = fontsize)
        ax2.set_ylabel("Norm Weight Change",fontsize = fontsize)
        ax2.set_xlabel("Gain",fontsize = fontsize)
        ax2.tick_params(axis = "both",labelsize = fontsize) 
        fig2.tight_layout()
        fig2.savefig(os.path.join(outdir,'readout_weight_change_p'+str(prob)+'.png'),dpi = 300)            
        
        ax3.set_title("Hidden Layer",fontsize = fontsize)
        ax3.set_ylabel("Norm Weight Change",fontsize = fontsize)
        ax3.set_xlabel("Gain",fontsize = fontsize)
        ax3.tick_params(axis = "both",labelsize = fontsize) 
        fig3.tight_layout()
        fig3.savefig(os.path.join(outdir,'hidden_weight_change_p'+str(prob)+'.png'),dpi = 300)



'''
plt.clf()
fig = plt.gcf()
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
              orientation='vertical', label='Gain')
plt.savefig(os.path.join(outdir,"Gain_colorbar.png"))
'''    
            
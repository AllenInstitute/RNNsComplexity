#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 06:17:43 2023

@author: dana.mastrovito
"""
#exec(open("plot_jacobian_product.py").read())

import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib
import pickle
import numpy as np
from utils import  GetModelFiles,GetModelFileParameters
from utils import GetModelWeights,GetInitializedModel
import torch
import pandas as pd
from scipy import stats as st
import socket

plt.rcParams['font.family'] = 'sans-serif'
hostname = socket.gethostname()    
if 'zuul' in hostname:
    home = "/home/dana/"
    
else:
    home = "/"

suffix = ""
add_dir = ""#"fully_trained"#"gain25.0"

ndigits = 10

DENSITY  = True
EM = False
DIGITS = False
NARROW = False
PIXEL = False
V1DD = False
coeff_var = False



plot_kaplan_york = False


INITIAL = True
if INITIAL:
    outsuffix=suffix+"_init"
else:
    outsuffix = suffix


NOISE = False

assert not ( EM and DENSITY)
complexity_dir = os.path.join(home,'allen','programs','mindscope','workgroups','tiny-blue-dot',"RNN",'Complexity')
dir = os.path.join(complexity_dir,"RNN")
nhidden = 198

if DENSITY:
    outdir = os.path.join(dir, "Density","jacobian_product_trajectory")
    suffix = "_threshto5573"
elif EM:
    outdir = os.path.join(dir, "EM_column","jacobian_product_trajectory")
elif NOISE:
    outdir = os.path.join(dir,"Gaussian","noise","jacobian_product_trajectory")
elif DIGITS:
    outdir = os.path.join("RNN","digits","Gaussian","jacobian_product_trajectory")
    suffix = ""
    ndigits = 2
elif PIXEL:
    outdir = os.path.join("RNN","pixel_by_pixel","Gaussian","jacobian_product_trajectory")
    ninputs = 14
    suffix = ""
elif NARROW:
    outdir = os.path.join("RNN","narrow","Gaussian","jacobian_product_trajectory")
    suffix = ""  
    nhidden = 28
elif V1DD:
    add_dir = "Dales"
    outdir = os.path.join("RNN","EM_column","v1dd",add_dir,"jacobian_product_trajectory")
    suffix = "23_4_"  
    nhidden = 198
elif coeff_var:
    '''
    
   
    _mean_0.05_std_0.025_ii_0.05_0.05
    '''
    add_dir = ""
    suffix = "mean_0.05_std_0.05_ii_0.05_0.075_"
    outdir = os.path.join("RNN","coefficient_variation","jacobian_product_trajectory")
else:
    outdir = os.path.join(dir,"Gaussian",add_dir,"jacobian_product_trajectory")



def get_colour_name(rgb_triplet):
    min_colours = {}
    for key, name in webcolors.css21_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def plot_max_jacobies_afo_gain_FC(jacobies,outdir, suffix,p =0):
    autumn = matplotlib.colormaps['cool']
    colors = [autumn(nNN) for nNN in np.linspace(0,1,len(params['nNN'][:-1]))]
    if p == 0:
        plt.plot(params['gain'],np.repeat(0,len(params['gain'])),color='red',alpha = 0.2)
    for n,nNN in enumerate(params['nNN'][:-1]):
        #w =  np.min(np.array(params['gain'])[np.where(np.amax(jacobies[n,p,:,:,:],2) >0)[0]])
        #wgt0=[]
        w = params['gain'][np.where(np.nanmean(jacobies[n,p,:,0],1) >0)[0][0]]
        print(nNN, params['p'][p],w)
        plt.errorbar(params['gain'],np.nanmean(np.amax(jacobies[n,p,:,:,:],2),1),\
                    yerr = np.nanstd(np.amax(jacobies[n,p,:,:,:],2),1),color = colors[n],label = str(nNN)+'/'+str(w),\
                        linestyle = 'dashed')
    plt.legend(title = "nNN")
    plt.savefig(os.path.join(outdir,'max_lyapunov_exp_aafo_gain_prob'+str(params['p'][p])+suffix+'.png'),dpi = 600)
    plt.savefig(os.path.join(outdir,'max_lyapunov_exp_aafo_gain_prob'+str(params['p'][p])+suffix+'.pdf'),dpi = 600)

#wgt0.append(np.where(jacobies[n,p,:,f,0] >0)[0][0])
#transitions.append(params['gain'][st.mode(np.array(wgt0))[0][0]])


def plot_jacobies_FC(jacobies,std,outdir,suffix):
    cool = matplotlib.colormaps['Blues']
    colors = [cool(f,alpha = .2) for f in np.linspace(0,1,nf)]
    red = matplotlib.colors.to_rgba('red',alpha = .2)
    strgains = [str(gain) for gain in params['gain']]
    strprobs = [str(prob) for prob in params['p']]
    markers = ['v',"^","<",">","o"] 
    legendp = []
    #legendp = [mmark.MarkerStyle(m) for m in markers]
    legendg = []
    p = -1
    nshow = 50
    mx = np.nanmax(jacobies[np.where(np.isinf(jacobies[:,:,:,:,:nshow])==False)])
    mn = np.nanmin(jacobies[np.where(np.isinf(jacobies[:,:,:,:,:nshow])==False)])
    
    for n,nNN in enumerate(params['nNN']):
        print(nNN)
        if ng <= 12:
            fig,ax = plt.subplots(4,3)#,sharex = "col",sharey='row')
        else:
            fig,ax = plt.subplots(4,np.ceil(ng/4).astype(int))#,sharex = "col",sharey='row')
        axes = ax.flat 
        for g in range(ng):
            for f in range(10):
                if np.all(np.isnan(jacobies[n,0,g,f,:])) == False and np.all(np.isinf(jacobies[n,p,g,f,:])) == False:
                    cc = np.array([colors[f] for i in range(nhidden)])
                    if np.any(jacobies[n,p,g,f,:]>0):
                        wgt0 = np.where(jacobies[n,p,g,f,:nshow] > 0)
                        cc[wgt0[0]] = red
                    axes[g].bar(np.arange(nshow),jacobies[n,0,g,f,:nshow],color=cc[:nshow])
            axes[g].set_title(" ".join(("nNN =",str(params['nNN'][n]),"gain =",strgains[g])  ))    
            plt.xlabel("EV #",fontsize = 20)
            plt.ylabel("Mean: 100 batches",fontsize = 20)
            plt.ylim((mn,mx))
            plt.tick_params(axis = "x",labelsize = 20) 
            plt.tick_params(axis = "y",labelsize = 20) 
            
            plt.tight_layout()
            plt.savefig(os.path.join(outdir,"Jacobian_products_afo_nNN"+str(nNN) +"_prob_"+str(params['p'][p])+"gain_"+str(gain)+suffix +".png"),dpi = 600)


def plot_jacobies_per_file(jacobies,std,outdir,suffix,file,mn,mx,nshow):
    
    if np.all(np.isnan(jacobies[:nshow])) == False and np.all(np.isinf(jacobies[:nshow])) == False:
        plt.clf()
        plt.figure(figsize=(6.4, 4.8), dpi=600)
        cc = np.array(['blue' for i in range(nshow)])
        if np.any(jacobies[:nshow]>0):
            wgt0 = np.where(jacobies[:nshow] > 0)
            cc[wgt0[0]] = 'red'
        plt.bar(np.arange(nshow),jacobies[:nshow],color=cc)
        #axes[g].set_title(" ".join(("nNN =",str(params['nNN'][n]),"gain =",strgains[g])  ))    
        plt.xlabel("Exponents",fontsize = 20)
        plt.ylabel("Mean: 100 batches",fontsize = 20)
        ax = plt.gca()
        plt.tick_params(axis = "x",labelsize = 20) 
        plt.tick_params(axis = "y",labelsize = 20) 
        plt.ylim((mn,mx))
        plt.tight_layout()
        print(os.path.join(outdir,"Jacobian_products_"+os.path.basename(file)+".png"))
        plt.savefig(os.path.join(outdir,"Jacobian_products_"+os.path.basename(file)+".png"),dpi = 600)



def plot_max_jacobies_afo_gain(jacobies,outdir, suffix,p =0,accs = None):
    autumn = matplotlib.colormaps['tab10']
    colors = [autumn(nNN) for nNN in np.linspace(0,1,len(params['nNN']))]
    plt.plot(params['gain'],np.repeat(0,len(params['gain'])),'r--',alpha = 0.5)
    for n,nNN in enumerate(params['nNN']):
        #w =  np.min(np.array(params['gain'])[np.where(np.amax(jacobies[n,p,:,:,:],2) >0)[0]])
        gtz = np.where(np.nanmean(jacobies[n,p,:,:,0],1) >0)
        if len(gtz[0]) >0:
            w = params['gain'][gtz[0][0]]
        else:
            w = np.nan
        if accs is not None:
            jacob_gain_means = np.zeros(len(params['gain']))
            jacob_gain_stds = np.zeros(len(params['gain']))
            for g in range(len(params['gain'])):
                thresh = np.where(accs[n,p,g,:] > 90.)
                if len(thresh[0]) > 0:
                    jacob_gain_means[g] = np.nanmean(jacobies[n,p,g,thresh,0])
                    jacob_gain_stds[g] = np.nanstd(jacobies[n,p,g,thresh,0])/np.sqrt(len(thresh[0]))
            plt.errorbar(params['gain'],jacob_gain_means,alpha = 0.8,color = colors[n],\
                    yerr = jacob_gain_stds,marker='o',label = str(nNN)+'/'+str(w))
        else:
            plt.errorbar(params['gain'],np.nanmean(jacobies[n,p,:,:,0],1),alpha = 0.8,color = colors[n],\
                    yerr = np.nanstd(jacobies[n,p,:,:,0],1)/np.sqrt(10),marker='o',label = str(nNN)+'/'+str(w))
        print(nNN, params['p'][p],w)
    
    plt.ylim(-.8,1)
    plt.title("Rewiring Probability = "+str(params['p'][p]),fontsize = 14)
    plt.ylabel("Maximum Lyapunov Exponent",fontsize = 14)
    plt.xlabel("Gain",fontsize = 14)
    plt.legend(title = "nNN/Transition",fontsize = 14)
    plt.tick_params(axis = "x",labelsize = 14) 
    plt.tick_params(axis = "y",labelsize = 14) 
    plt.tight_layout()
    if accs is not None:
        plt.savefig(os.path.join(outdir,'max_lyapunov_exp_aafo_gain_prob'+str(params['p'][p])+suffix+'_thresh.png'),dpi = 600)
    else:
        plt.savefig(os.path.join(outdir,'max_lyapunov_exp_aafo_gain_prob'+str(params['p'][p])+suffix+'.png'),dpi = 600)
    cf = plt.gcf()
    print(cf)



def plot_mean_jacobies_afo_gain(jacobies,outdir, suffix,gain_transitions,p =0):
    plt.clf()
    autumn = matplotlib.colormaps['tab10']
    colors = [autumn(nNN) for nNN in np.linspace(0,1,len(params['nNN']))]
    for n,nNN in enumerate(params['nNN']):
        #w =  np.min(np.array(params['gain'])[np.where(np.amax(jacobies[n,p,:,:,:],2) >0)[0]])
        mean = np.stack([np.nanmean(jacobies[n,p,:,f,:],1) for f in range(10)])
        plt.errorbar(params['gain'],np.nanmean(mean,0),alpha = 0.8,color = colors[n],\
                    yerr = np.nanstd(mean,0)/np.sqrt(10),marker='o',label = str(nNN))
    ax = plt.gca()
    ylim = ax.get_ylim()
    for n,nNN in enumerate(params['nNN']):
        gain_transition = gain_transitions['Transition Point'][(gain_transitions['nNN'] == nNN) & (gain_transitions['p'] == params['p'][p])].item()
        plt.plot(np.repeat(gain_transition,10),np.linspace(ylim[0],ylim[1],10),color = colors[n],linestyle='dashed',alpha = 0.8)
    plt.title("Rewiring Probability = "+str(params['p'][p]))
    plt.ylabel("Mean Lyapunov Exponent")
    plt.xlabel("Gain")
    plt.legend(title = "nNN")
    plt.savefig(os.path.join(outdir,'mean_lyapunov_exp_aafo_gain_prob'+str(params['p'][p])+suffix+'.png'),dpi = 600)



def get_Kaplan_York(jacobies):
    j = np.where(np.cumsum(jacobies)>0)[0]
    if j.size > 0:
        j = np.max(j) + 1
    else:
        j = 1
    lyapunov_sum = np.sum(jacobies[:j])
    return j + ((lyapunov_sum)/np.abs(jacobies[j]))


def plot_Kaplan_York(jacobies,n,outdir,transitions,axes, suffix,ndigits):
    #transitions =[[5,3,3,3,3],list(np.repeat(3,5)),[3,2,2,2,2],[1.75,1.75,2.0,1.75,1.75],list(np.repeat(1.75,5)),list(np.repeat(1.5,5)),list(np.repeat(1.5,5)),list(np.repeat(1.5,5))]
    markers = ['v',"^","<",">","o"] 
    autumn = matplotlib.colormaps['cool']
    colors = [autumn(g) for g in np.linspace(0,1,len(params['p']))]
    #plt.clf()
    pdims = []
    for p,prob in enumerate(params['p']):
        gdims = []
        gdstd = []
        gidx = params['gain'].index(transitions[p])
        for g,gain in enumerate(params['gain']):
            dims = []
            for f in range(10):
                dims.append(get_Kaplan_York(jacobies[n,p,g,f]))
            gdims.append(dims)
        axes[n].errorbar(params['gain'],np.nanmean(np.stack(gdims),1),yerr = np.std(np.stack(gdims),1),color=colors[p],marker='s',label = str(prob),alpha = 0.5)
        axes[n].plot(np.repeat(params['gain'][gidx],5),np.linspace(0,120,5),color = colors[p],linestyle ='dashed',alpha = 0.5)
        axes[n].plot(params['gain'],np.repeat(ndigits,len(params['gain'])),color ='red',linestyle = 'dotted')
        pdims.append(gdims)
    #if n ==0:
    #    axes[n].legend(fontsize =14)
        #axes[n].set_ylabel("Lyapunov Dimension")
    if params['nNN'][n] in [128,198]:
        axes[n].set_xlabel('Gain',fontsize = 18)
    axes[n].set_title("nNN = "+str(params['nNN'][n]),fontsize = 18)
    axes[n].tick_params(axis = "both",labelsize = 18)
    if ndigits == 2:
        axes[n].set_ylim((0,10))
    else:
        #axes[n].set_yscale("log")
        axes[n].set_ylim((0,33))
    with open(os.path.join(outdir,'KY_nNN'+str(nNN)+suffix+"_ndigits_"+str(ndigits)+".pkl"),'wb') as f:
        pickle.dump(pdims, f)
    return pdims
         
        

def plot_jacobies_bar(jacobies,std,p,outdir,suffix):
    cool = matplotlib.colormaps['Blues']
    colors = [cool(f,alpha = .2) for f in np.linspace(0,1,nf)]
    red = matplotlib.colors.to_rgba('red',alpha = 0.2)
    strgains = [str(gain) for gain in params['gain']]
    strprobs = [str(prob) for prob in params['p']]
    markers = ['v',"^","<",">","o"] 
    legendp = []
    #legendp = [mmark.MarkerStyle(m) for m in markers]
    legendg = []
    nshow = 50
    mx = np.nanmax(jacobies[np.where(np.isinf(jacobies[:,:,:,:,:nshow])==False)])
    mn = np.nanmin(jacobies[np.where(np.isinf(jacobies[:,:,:,:,:nshow])==False)])
    
    for n,nNN in enumerate(params['nNN']):
        print(nNN)
        plt.clf()
        fig,ax = plt.subplots(4,3,sharex = "col",sharey='row')
        axes = ax.flat 
        for g in range(ng):
            for f in range(10):
                if np.all(np.isnan(jacobies[n,p,g,f,:])) == False and np.all(np.isinf(jacobies[n,p,g,f,:])) == False:
                    cc = np.array([colors[f] for i in range(nhidden)])
                    if np.any(jacobies[n,p,g,f,:]>0):
                        wgt0 = np.where(jacobies[n,p,g,f,:nshow] > 0)
                        cc[wgt0[0]] = red
                    axes[g].bar(np.arange(nshow),jacobies[n,p,g,f,:nshow],color=cc[:nshow])
            axes[g].set_title(" ".join(("nNN =",str(params['nNN'][n]),"gain =",strgains[g])  ))   
            axes[g].set_ylim((mn,mx))
        plt.xlabel("EV #")
        plt.ylabel("Mean: 100 batches")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,"Jacobian_products_afo_gain"+str(nNN) +"_prob_"+str(params['p'][p])+suffix +".png"),dpi = 600)



def plot_jacobies(jacobies,std,p,outdir,suffix):
    #bins  = np.linspace(np.nanmin(jacobies),np.nanmax(jacobies),50)
    bins = [-25,-10,-5,-2,-1,-.75,-.5,-.2,0.,0.01,.02,.03,.05,.06,.07,.08,.09,.1,.2,.3,.4]
    cool = matplotlib.colormaps['Blues']
    strgains = [str(gain) for gain in params['gain']]
    strprobs = [str(prob) for prob in params['p']]  
    for n,nNN in enumerate(params['nNN']):
        print(nNN)
        plt.clf()
        fig,ax = plt.subplots(4,3,sharex = "col",sharey='row')
        axes = ax.flat 
        for g in range(ng):
            for f in range(10):
                axes[g].hist(jacobies[n,p,g,f,:],bins = bins, alpha = 0.2)
            axes[g].set_title(" ".join(("nNN =",str(params['nNN'][n]),"gain =",strgains[g])  ))  
            axes[g].set_ylim((0,10))
            #axes[g].set_xlim((-1,0.5))
            ylim = axes[g].get_ylim()
            axes[g].axvline(x = 0,ymin =0,ymax = ylim[1],color='red',alpha = 0.5, linestyle = 'dashed')
        plt.xlabel("EV #")
        plt.ylabel("Mean: 100 batches")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,"Jacobian_products_afo_gain"+str(nNN) +"_prob_"+str(params['p'][p])+suffix +".png"),dpi = 600)


hostname = socket.gethostname()
if 'zuul' in hostname:
    home ="/home/dana/"
else:
    home = ""

params = GetModelFileParameters(em = EM,Density = DENSITY,digits = DIGITS,v1dd = V1DD,pixel = PIXEL,
                                coeff_var = coeff_var, narrow  = NARROW , suffix=suffix,add_dir = add_dir,home = home)

'''
params['nNN'] = [nhidden]
params['p'] = [0.0]
params['gain'] = [0.5,0.75,0.88,0.94,1.0,1.02,1.05,1.10,1.20,1.25]
'''

if not V1DD or DENSITY:
    params['nNN'] = [4, 8, 16, 28, 32, 64, 128, 198]
    params['gain'] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 5.0]



nnNN = len(params['nNN'])
nprob = len(params['p'])
ng = len(params['gain'])
nf = 10

jacobies = np.zeros((nnNN, nprob,ng, nf,nhidden))
std = np.zeros_like(jacobies)
rvals = np.zeros_like(jacobies)
jacobies.fill(np.nan)
std.fill(np.nan)
rvals.fill(np.nan)

for n, nNN in enumerate(params['nNN']):
    for p, prob in enumerate(params['p']):
        for g, gain in enumerate(params['gain']):
            files = GetModelFiles(nNN, prob, gain = gain,em = EM, density = DENSITY,digits = DIGITS,narrow = NARROW,v1dd = V1DD, pixel = PIXEL, 
                                  coeff_var = coeff_var, suffix=suffix,add_dir= add_dir,home = home)
            for f, file in enumerate(files):
                if V1DD or coeff_var or DENSITY :
                    jacoby_file = os.path.join(os.path.dirname(file),"jacobian_product_trajectory", os.path.basename(file)+suffix+outsuffix)
                else:
                    jacoby_file = os.path.join(os.path.dirname(file),"jacobian_product_trajectory", os.path.basename(file)+outsuffix)
                if os.path.exists(jacoby_file):
                    print(jacoby_file)
                    with open(jacoby_file,'rb') as jf:
                        jacoby = pickle.load(jf) #mean, std, all, rvals
                        order = np.argsort(jacoby['mean'].cpu().numpy())
                        order = order[::-1] #biggest will now be first
                        jacobies[n,p,g,f] = jacoby['mean'].cpu().numpy()[order]
                        std[n,p,g,f] = jacoby['std'].cpu().numpy()[order]
                        #print(jacoby_file)
                        #rvals[n,p,g,f] = jacoby['rvals']
nshow = 50            
#plot_nNN = [8,28,64,198]
#plot_prob = [1.0]
plot_nNN = [198]
plot_prob = [0.0]
plot_gain = [0.5,1.0,3.0]#
mn = np.nanmin(jacobies[:,:,:,:,:nshow])
mx = np.nanmax(jacobies[:,:,:,:,:nshow])
for n, nNN in enumerate(params['nNN']):
    if nNN in plot_nNN:
        for p, prob in enumerate(params['p']):
            if prob in plot_prob:
                for g, gain in enumerate(params['gain']):
                    if gain in plot_gain:
                        files = GetModelFiles(nNN, prob, gain = gain,em = EM, density = DENSITY,digits = DIGITS,narrow = NARROW,v1dd = V1DD, pixel = PIXEL, 
                                  coeff_var = coeff_var, suffix=suffix,add_dir= add_dir,home = home)
                        for f, file in enumerate(files):
                            plot_jacobies_per_file(jacobies[n,p,g,f] ,std[n,p,g,f] ,outdir,suffix,file,mn,mx,nshow)

print(os.path.join(os.path.dirname(file),"jacobies"+"_"+suffix+"_"+outsuffix+".pkl"))
#with open (os.path.join(os.path.dirname(file),"jacobies"+"_"+suffix+"_"+outsuffix+".pkl"),'wb') as f:
with open (os.path.join(os.path.dirname(file),"jacobies"+"_nhidden"+str(nhidden)+"_"+outsuffix+".pkl"),'wb') as f:  
    pickle.dump(jacobies,f)
                        
for p, prob in enumerate(params['p']):
    if EM or DENSITY or V1DD or coeff_var:
        plot_jacobies_FC(jacobies,std,outdir,suffix+"_"+outsuffix)
    else:
        #plot_jacobies(jacobies,std,p,outdir,outsuffix)
        plot_jacobies_FC(jacobies,std,outdir,"_"+outsuffix)

        
'''        
for n, nNN in enumerate(params['nNN']):        
    plot_Kaplan_York(jacobies,n,dir,outsuffix)
'''
#gt0 = np.zeros((nnNN,ng,nf)) 

plt.clf()
fig, ax = plt.subplots(4,2, sharex = True, sharey = True)    
fig.set_figheight(13)
fig.set_figwidth(13)

axes = ax.flat

nNNs = []
ps = []
transitions = []
for n,nNN in enumerate(params['nNN']):
    nNNtransitions = []
    nNNstdtrans = []
    for p,prob in enumerate(params['p']): 
        gtz = np.where(np.nanmean(jacobies[n,p,:,:,0],1) >0)
        if len(gtz[0]) >0:
            transition_point = params['gain'][gtz[0][0]]
        else:
            transition_point = np.nan
        nNNtransitions.append(transition_point)
        transitions.append(transition_point)
        nNNs.append(nNN)
        ps.append(prob)
            

gain_transitions = {'nNN':nNNs, 'p':ps,'Transition Point':transitions}
df = pd.DataFrame(data = gain_transitions)
df.to_csv(os.path.join(outdir,"gain_transitions"+"_"+suffix+outsuffix+".csv") )


gain_transitions = pd.read_csv(os.path.join(outdir,"gain_transitions"+"_"+suffix+"_init.csv"))
if plot_kaplan_york:
    
    transitions = []
    for n,nNN in enumerate(params['nNN']):
        nNNtransitions = []
        for p,prob in enumerate(params['p']): 
            gain_transition = gain_transitions['Transition Point'][(gain_transitions['nNN'] == nNN) & (gain_transitions['p'] == float(prob))].item()
            if not  np.isnan(gain_transition):
                gidx = params['gain'].index(gain_transition)
                transition = params['gain'][gidx]
            else:
                transition = np.nan
            nNNtransitions.append(transition)
        pdims = plot_Kaplan_York(jacobies,n,outdir,nNNtransitions,axes,outsuffix,ndigits)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,"KY_nNN"+"_"+suffix+"_"+outsuffix+".png"),dpi = 600)            

if ndigits == 2:
    KYs = np.zeros((len(params['nNN']),len(params['p']),len(params['gain']),nf))
    KYs.fill(np.nan)
    for n,nNN in enumerate(params['nNN']):
        for p,prob in enumerate(params['p']): 
            for g, gain in enumerate(params['gain']):
                for f in range(nf):
                    KYs[n,p,g,f] = get_Kaplan_York(jacobies[n,p,g,f])
            KYnpg = np.nanmean(KYs[n,p,:,:],1)
            gt = gain_transitions[(gain_transitions['nNN']==nNN) & (gain_transitions['p']==prob)]['Transition Point'].item()
            
plt.clf()
autumn = matplotlib.colormaps['tab10']
colors = [autumn(g) for g in np.linspace(0,1,len(params['nNN']))]
for n,nNN in enumerate(params['nNN']):
    nNNtrans = []
    std = []
    for p,prob in enumerate(params['p']): 
        trans = [np.where(jacobies[n,p,:,f,0] >0)[0][0]  if np.where(jacobies[n,p,:,f,0] >0)[0].size >0 else np.nan for f in range(10) ]
        gtz = np.where(np.nanmean(jacobies[n,p,:,:,0],1) >0)
        if len(gtz[0]) >0:
            transition_point = params['gain'][np.where(np.nanmean(jacobies[n,p,:,:,0],1) >0)[0][0]]
        else:
            transiton_point = np.nan
        nNNtrans.append(transition_point)
        nnan = np.where(np.isnan(trans) == False)[0].size
        std.append(np.std(np.array(params['gain'])[np.array(trans)[np.where(np.isnan(trans)==False)].astype(int)])/np.sqrt(nnan))
        #plt.plot(params['p'],nNNtransitions,label = "nNN = "+ str(nNN),alpha = 0.5, marker='.')
    plt.errorbar(params['p'],nNNtrans,yerr=std,label = str(nNN),alpha = 0.8,marker='o',color= colors[n])

plt.tick_params(axis = "x",labelsize = 14) 
plt.tick_params(axis = "y",labelsize = 14) 
plt.title("Transition Points",fontsize = 14)
plt.xlabel("Rewiring Probability",fontsize = 14)
plt.ylabel("Gain",fontsize = 14)
plt.legend(fontsize = 14, title = "nNN")
plt.tight_layout()
plt.savefig(os.path.join(outdir,'gain_transitions'+"_"+suffix+"_"+outsuffix+'.png'),dpi = 600)                


'''
ngt0_mean = np.mean(gt0,2)
ngt0_std = np.std(gt0,2)
plt.close("all")
plt.clf()
for n in range(nnNN):
    plt.errorbar(x=params['gain'],y=ngt0_mean[n,:],yerr = ngt0_std[n,:],label = str(params['nNN'][n]))
    
plt.xscale('log')       
plt.ylabel("mean number of EVs gt 1")
plt.xlabel("gain")
plt.legend(title="nNN")
plt.savefig(os.path.join(dir,"mean_summary_ngt0"+suffix+".png"),dpi = 300)


accs = np.zeros(jacobies.shape[:-1])
for n, nNN in enumerate(params['nNN']):
    with open("RNN/Gaussian/pcist/time_to_max_acc_nNN"+str(nNN)+".pkl",'rb') as accf:
        acc = pickle.load(accf)
        accs[n] = np.swapaxes(acc['max_acc'],0,1)
'''        
        
accs = None
for p in range(len(params['p'])):
    plt.clf()
    plt.close('all')
    plot_max_jacobies_afo_gain(jacobies,outdir, suffix+outsuffix,p = p,accs = accs)
    
for p in range(len(params['p'])):
    plt.clf()
    plt.close('all')
    plot_mean_jacobies_afo_gain(jacobies,outdir, suffix+outsuffix,gain_transitions,p = p)
 
for n in range(len(params['nNN'])):
    for p in range(len(params['p'])):
        for g in range(len(params['gain'])):
            print(params['nNN'][n],params['p'][p],params['gain'][g],[np.where(jacobies[n,p,g,i,:] >0)[0].size for i in range(10)])    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:45:17 2021

@author: danamastrovito
"""
#exec(open("ModelPCIst.py").read())
#This code uses published PCIst computation code from 
#https://github.com/renzocom/PCIst

import glob
import os
import numpy as np
import torch
import Network
import PCIst
import matplotlib.pyplot as plt
import pickle
import socket

hostname = socket.gethostname()    
if 'zuul' in hostname:
    home = "/home/dana/"
    
else:
    home = "/"

from utils import  GetModelFiles,GetModelFileParameters

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def plot_pcist_singleprob(IPCIst,SPCIst,TPCIst,gains): 
    plt.clf()
    fig, axs = plt.subplots(5,1,sharex = True, sharey= True)
    for g,gain in enumerate(gains):
        ax = axs[g]
        ax.boxplot([IPCIst[g][0],SPCIst[g][0],TPCIst[g][0]],labels =  ['Initial','Shuffled','Trained'])
        ax.set_title("Shuffled Weights Gain " + str(gain))
        if g == 0:
            ax.set_ylabel("PCIst")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,"PCIst_nNN"+str(n)+".png"),dpi = 300) 
 
    
def plot_pcist(IPCIst,SPCIst, TPCIst,probs,gains): 
    IPCIst = np.stack(IPCIst)
    SPCIst = np.stack(SPCIst)#[gain,prob,file]
    TPCIst = np.stack(TPCIst)
    maxy = np.max([np.max(TPCIst),np.max(IPCIst),np.max(SPCIst)])
    miny = np.min([np.min(TPCIst),np.min(IPCIst),np.min(SPCIst)])
    rng = (miny,maxy)
    plt.clf()
    fig, axs = plt.subplots(5,3,sharex = True, sharey= True)
    for g,gain in enumerate(gains):
        ax = axs[g,0]
        ax.boxplot(IPCIst[g,:,:].T,labels =  probs)
        if g == 0:
            ax.set_ylabel("PCIst")
        elif g== 5:
            ax.set_xlabel("probability of rewiring")
        ax = axs[g,1]
        ax.boxplot(SPCIst[g,:,:].T,labels=probs)
        #plt.ylim(rng)
        #plt.xlabel("probability of rewiring")
        #plt.ylabel("PCIst")
        ax.set_title("Shuffled Weights Gain " + str(gain))
        ax = axs[g,2]
        ax.boxplot(TPCIst[g,:,:].T,labels = probs)
        ax.set_title("Trained")
        if g == 5:
            ax.set_xlabel("probability of rewiring")   
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,"PCIst_nNN"+str(n)+".png"),dpi = 300) 


def get_noise(noise_mean, noise_std,nOsc,dt,device='cpu'):
    noise = np.random.normal(noise_mean, noise_std,nOsc)*np.sqrt(dt)
    return torch.tensor(noise,dtype = torch.float64).to(device)

def plot_weight_dist(Iweights,Tweights,fig=None,ax=None,start = False, finished= False, params=None,file =None):
    if start:
        fig, ax = plt.subplots(2,1)
    bins = np.linspace(-1,1,50)
    xlim = (-1,1)
    I = np.histogram(Iweights,bins = bins)
    F = np.histogram(Tweights,bins=bins)
    ax[0].hist(WhI,bins = bins, alpha = .2,color = colors[ifile])
    #hax[0].set_ylim(ylim)
    ax[0].set_xlim(xlim)
    ax[0].set_title("Initial Weights")
    
    ax[1].hist(Wh,bins = bins,alpha=.2,color = colors[ifile])
    ax[1].set_xlim(xlim)
    #hax[1].set_ylim(ylim)
    ax[1].set_title("Learned Weights")
    
    if finished:
        fig.tight_layout()
        fig.savefig(file)
    return fig, ax
    
k = 1.2
snr = 2.2
EM = False                                                      
DENSITY = False
V1DD = False
NOISE = False
DEGREE_DISTRIBUTION = False

colors=['red','orange','green','blue','purple','lightblue','pink','brown','darkgreen','grey']

dir = os.path.join(home,"allen","programs","braintv","workgroups","tiny-blue-dot","RNN","Complexity","RNN")


if DENSITY:
    dir = os.path.join(dir,'Density')
    files = glob.glob(os.path.join(dir,"Density_198*ninputs_28*Run_[0-9]*"))
    #suffix = "_sparsify"
    #suffix = "_thresh"
    suffix = "_threshto5573"
    #suffix = ""
elif EM:
    dir = os.path.join(dir,'EM_column',"microns")
    suffix = "permuted_"
    files = glob.glob(os.path.join(dir,"EM_column_198*ninputs_28*"+suffix+"Run_[0-9]*"))
elif V1DD:
    add_dir = ""#"Dales"
    Dales = False
    dir = os.path.join(dir,'EM_column',"v1dd",add_dir)
    suffix = "23_4_topology_"
    files = glob.glob(os.path.join(dir,"v1dd_198*ninputs_28*"+suffix+"Run_[0-9]*"))
elif DEGREE_DISTRIBUTION:
    dir = os.path.join(dir, "degree_dist")
    files = glob.glob(os.path.join(dir, "DegreeDistribution_198_ninputs_28*Run_[0-9]*"))
    suffix = ""
    add_dir =""
else:
    add_dir =""
    Dales = False
    dir = os.path.join(dir, "Gaussian",add_dir)
    #nhidden = 500
    #files = glob.glob(os.path.join(dir,"WattsStrogatz_"+str(nhidden)+"ninputs_28*_Run_[0-9]*"))
    suffix = ""
    
if NOISE:
    dir = os.path.join(dir,"noise")


outdir = os.path.join(dir,"new_pcist")   
if not os.path.exists(outdir):
    os.mkdir(outdir)


print(outdir)    
#files =[f for f in files if  "png" not in f and 'npy' not in f]

nPCIpts = 28

if not (DENSITY or EM or V1DD or DEGREE_DISTRIBUTION):
    params  =  GetModelFileParameters(add_dir = add_dir,home  = home)
    '''
    params['nNN'] = [500,1000] #np.flip(np.sort(np.array(list(set([int(os.path.basename(f).split('_')[3].split('nNN')[1]) for f in files])))))
    params['p'] =  [0.0] #np.sort(np.array(list(set([float(os.path.basename(f).split('_')[5]) for f in files]))))
    params['gain'] = [0.5,0.75,0.88,0.94,1.0,1.02,1.05,1.1,1.2,1.25]
    '''
    
    params['nNN'] = [n for n in params['nNN'] if n in [500,1000]]
    nNN = params['nNN']#np.flip(np.sort(np.array(list(set([int(os.path.basename(f).split('_')[3].split('nNN')[1]) for f in files])))))
    ps =  params['p'] #np.sort(np.array(list(set([float(os.path.basename(f).split('_')[5]) for f in files]))))
    gains = params['gain']
    
    '''
    nNN = [n for n in nNN if n != 198]
    print(nNN)
    gains = []
    for f in files:
        if 'gain' in f:
            gains.append(float(os.path.basename(f).split('_')[7]))
        else:
            gains.append(1.0)
    gains = np.sort(np.array(list(set(gains))))
    '''            
else:
    params = GetModelFileParameters(em= EM, v1dd = V1DD,Density = DENSITY, degree_distribution = DEGREE_DISTRIBUTION, suffix = suffix,add_dir = add_dir)
    nNN = params['nNN']
    ps = params['p']
    gains = params['gain']
    '''
    for f in files:
        if 'gain' in f:
            gains.append(float(os.path.basename(f).split('_')[6]))
    gains = np.sort(np.array(list(set(gains))))
    '''    
    
couplingStrength =  np.linspace(0,1.5,num = 30)
dcS = np.diff(couplingStrength)
#gains = [gain for gain in gains if gain not in 25.0]]



for n in nNN:
    print("nNN",n)
    SPCI = []
    TPCI = []
    IPCI = []
    plt.clf()
    fig, axs = plt.subplots(nrows = len(gains),ncols = 2, sharex = True, sharey = True)
    fps = []
    for g,gain in enumerate(gains):
        pSPCI = []
        pTPCI = []
        pIPCI = []
        for i,p in enumerate(ps):
            print("p",p)
            files = GetModelFiles(n,p,gain = gain, density = DENSITY,v1dd= V1DD, em = EM, degree_distribution = DEGREE_DISTRIBUTION, suffix = suffix,add_dir = add_dir,home = home)
            if not (EM or DENSITY or V1DD or DEGREE_DISTRIBUTION):
                #file = [f for f in files if 'nNN'+str(n) in f and 'p_'+"{:.1f}".format(p) in f and 'gain_'+str(gain) in f]
                file = GetModelFiles(n,p,gain = gain, density = DENSITY,v1dd= V1DD, em = EM, degree_distribution = DEGREE_DISTRIBUTION, suffix = suffix,add_dir = add_dir,home = home)
            else:
                #file = [f for f in files if 'gain_'+str(gain) in f]
                file = GetModelFiles(n,p,gain = gain, density = DENSITY,v1dd= V1DD, em = EM, degree_distribution = DEGREE_DISTRIBUTION, suffix = suffix,add_dir = add_dir)
            print(file)
            Spci = []
            Tpci = []
            Ipci = []
            if len(file) >0:
                params = {"nNN":n,"p":p,"gain":gain}
                hfig,hax = plt.subplots(2,1)
                fps.append(p)
                if len(file) != 10:
                    print(n,p,"Not 10 files !")
                for ifile, f in enumerate(file):
                    tpci = []
                    spci = []
                    ipci = []
                    model = torch.load(f,map_location=device)
          
                    net = Network.Net(model['config']['ModelType'],model['config']['ninputs'],model['config']['nhidden'],\
                                      model['config']['batch_size'],input_bias=False,device = device,noise = NOISE) 
                    
                    if 'Gain' in model.keys():
                        net.initialize(model['ConnType']['ConnType'],nNN = model['ConnType']['nNN'],p=model['ConnType']['p'],gain = model['Gain'],suffix = suffix,Dales = Dales)
                    else:
                        net.initialize(model['ConnType']['ConnType'],nNN = model['ConnType']['nNN'],p=model['ConnType']['p'],gain = 1.0,suffix = suffix)
                     
                    #Re-initialize all other parameters to that of final state
                    net.to(device)
                    net.Reinitialize(f,device=device)
                    
                    
                    IparamIdx =  [idx for idx,param in enumerate(model['InitialState']) if 'Wh' in param[0] or 'weight_hh' in param[0]][0]
                    paramIdx = [idx for idx,param in enumerate(model['Parameters']) if 'Wh' in param[0] or 'weight_hh' in param[0]][0]
                    
                    
                    WhI = np.array(model['InitialState'][IparamIdx][1].detach().cpu()).flatten()
                    nzI = np.nonzero(WhI)
                    WhI = WhI[nzI]
                    
                    Wh = np.array(model['Parameters'][paramIdx][1].detach().cpu()).flatten()
                    nel = len(Wh)
                    CM = np.zeros(Wh.shape)
                    nz = np.nonzero(Wh)
                    Wh = Wh[nz]
                    
                    if ifile ==0:
                        fig, ax = plot_weight_dist(WhI,Wh,start = True,finished = False)
                    elif ifile == len(file) - 1:
                        fig, ax = plot_weight_dist(WhI,Wh,fig,ax,start = False,finished = True,params= params,\
                                         file = os.path.join(dir,"dists",os.path.basename(f).split("_Run")[0]+".png"))
                    else:
                        fig, ax = plot_weight_dist(WhI,Wh,fig,ax,start = False,finished = False)
                    
                    #Re-Initialize model with weights sampled from the final learned weights
                    sample = np.random.choice(Wh,size = len(nz[0]),replace=False)
                    #locations = np.random.choice(np.arange(nel),len(nz[0]),replace=False)
                    CM[nz] = sample
                    CM = torch.from_numpy(CM.reshape((model['config']['nhidden'],model['config']['nhidden']))).type(torch.float32)
                    
                      
                    if model['config']['ModelType'] == 'RNN':
                        net.model.cell.weight_hh.data = CM
                    elif model['config']['ModelType'] == 'Kuramoto':
                        net.model.cell.Wh.data = CM
                    
                    (images, labels) = next(iter(net.dl.train_loader))
                    images = images.reshape(net.batch_size, net.ninputs, int(images.shape[1]/net.ninputs))
                    images = images.permute(2,0,1)
                    input = []
                    for b in range(net.batch_size):
                        input.append( torch.normal(mean = 0.0,std = 0.01,size = (28,28)))
                    
                    input = torch.stack(input).swapaxes(0,1)
                    '''
                    input = torch.zeros((images.shape[0],images.shape[1],images.shape[2]) )
                    input = input + (torch.normal(mean = input,std = 0.1)**2)
                    '''
                    net.to(device)
                    input,images = input.to(device), images.to(device)
                    net.model.forward(input)
                    istate = net.model.state.clone().detach()
                    net.model.forward(images)
                    state = net.model.state.clone().detach()
                    
                    time = np.arange(-nPCIpts,nPCIpts)
                    zt = np.where(time ==0.0)[0][0]
                    
                    #par = {'baseline_window':(-images.shape[2],-2), 'response_window':(0,images.shape[2]-1), 'k':1.2,'max_var':99, 'embed':False,'n_steps':10,'min_snr':1.1}
                    #par = {'baseline_window':(-nPCIpts ,0), 'response_window':(0,nPCIpts ), 'k':k,'max_var':99, 'embed':False,'n_steps':10,'min_snr':snr}
                    par = {'baseline_window':(-nPCIpts ,0), 'response_window':(0,nPCIpts ), 'k':k,'max_var':99, 'embed':False,'n_steps':10,'min_snr':snr}
                    
                    for batch in range(model['config']['batch_size']):
                        mis = istate[batch,:,:nPCIpts]
                        ms = state[batch,:,:nPCIpts]
                        sigI = torch.hstack((mis,ms))
                        mean = torch.mean(sigI[:,:nPCIpts],1)
                        sigI = (sigI - torch.unsqueeze(mean,1)).numpy()
                        #print(sigI.shape)
                        spci.append(PCIst.calc_PCIst(sigI, time, full_return=False, **par))
                       
                            
                    net.Reinitialize(f,device=device)
                    net.to(device)
                    net.model.forward(input)
                    istate = net.model.state.clone().detach()
                    net.model.forward(images)
                    state = net.model.state.clone().detach()
                    
                    for batch in range(model['config']['batch_size']):
                        mis = istate[batch,:,:nPCIpts]
                        ms = state[batch,:,:nPCIpts]
                        sigT = torch.hstack((mis,ms))
                        mean = torch.mean(sigT[:,:nPCIpts],1)
                        sigT = (sigT - torch.unsqueeze(mean, 1)).numpy()
                        tpci.append(PCIst.calc_PCIst(sigT, time, full_return=False, **par))
                    
                        
                    net.Reinitialize(f,initial = True,device=device)
                    net.to(device)
                    net.model.forward(input)
                    istate = net.model.state.clone().detach()
                    net.model.forward(images)
                    state = net.model.state.clone().detach()
                    
                    for batch in range(model['config']['batch_size']):
                        mis = istate[batch,:,:nPCIpts]
                        ms = state[batch,:,:nPCIpts]
                        sigT = torch.hstack((mis,ms))
                        mean = torch.mean(sigT[:,:nPCIpts],1)
                        sigT = (sigT - torch.unsqueeze(mean, 1)).numpy()
                        ipci.append(PCIst.calc_PCIst(sigT, time, full_return=False, **par))
                    
                    Spci.append(np.mean(spci)) #pci values over runs with same parameters
                    Tpci.append(np.mean(tpci))
                    Ipci.append(np.mean(ipci))
        
            pSPCI.append(Spci) #pci values over all ps for a given gain
            pTPCI.append(Tpci)
            pIPCI.append(Ipci)
        SPCI.append(pSPCI)   
        TPCI.append(pTPCI)  
        IPCI.append(pIPCI)
    #plot_pcist(IPCI,SPCI,TPCI,list(set(fps)), gains)
    PCI = {"SPCI":SPCI,"TPCI":TPCI,"IPCI":IPCI}
    pcifile = open(os.path.join(outdir,"PCIst_nNN"+str(n)+suffix+"_k"+str(k)+"_snr"+str(snr)+".pkl"), 'wb')
    # source, destination
    pickle.dump(PCI, pcifile)                     
    pcifile.close()
        




              
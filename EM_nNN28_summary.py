#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:10:07 2023

@author: dana.mastrovito
"""
#exec(open("EM_nNN28_summary.py").read())

from utils import GetModelFileParameters, GetModelFiles
from utils import GetModelWeights
from utils import GetModelReadout,GetModelInputWeights

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
from matplotlib import cm
import matplotlib
import sys
plt.rcParams['font.family'] = 'sans-serif'
#plt.rc('text', usetex=True)

MESOSCOPIC = False
SUPPLEMENTARY = True
V1DD = False

if MESOSCOPIC:
    outfile = "_meso"
elif SUPPLEMENTARY:
    outfile = "_supp"
elif V1DD:
    outfile =""

def GetWeightChange(models,device = 'cpu'):
    Idiffs = []
    Hdiffs = []
    Rdiffs  = []
    for file in models:
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
    return Idiffs, Hdiffs, Rdiffs
  

def GetAccuracy(models,device = 'cpu'):
    max_acc = []
    time_to_max = []
    for file in models:
        model = torch.load(file,map_location=device)
        valacc = np.array(model['ValidationAccuracy'])
        if len(valacc)/100 != 39.0:
            print(file,len(valacc))
        assert len(valacc)/100 == 39.0
        max_acc.append(np.max(valacc))
        time_to_max.append(np.where(valacc == np.max(valacc))[0][0]) #/39.0 # 39 is number of batches per epoch
    return max_acc, time_to_max

def GetKaplanYork(jacobies):
    cs = np.stack([np.cumsum(jacobies[i,:]) for i in range(10)])
    gtz = [np.where(cs[i,:] >0)[0] for i in range(10)]
    lyapunov_sum = []
    js = []
    for i in range(10):
        if len(gtz[i]) >0:
            j = np.max(gtz[i])
            
        else:
            j = 1
        js.append(j)
        lyapunov_sum.append(np.sum(jacobies[i,:j])) 
    KY = [js[i] + (lyapunov_sum[i]/np.abs(jacobies[i,js[i]])) for i in range(10)]
    return np.mean(KY), np.std(KY)


def GetGainTransition(jacobies):
    transition = np.where(jacobies)[0]
    if transition.size >0:
        return transition[0]
    else:
        return np.nan

def GetMaxMeanJacobies(jacobies):
        max_mean_jacs = np.mean(jacobies[:,0])
        std = np.std(jacobies[:,0])
        return max_mean_jacs,std   

def GetJacobies(files,suffix="_init"):
    jacobies = []
    for file in files:
        jacoby_file = os.path.join(os.path.dirname(file),"jacobian_product_trajectory", os.path.basename(file)+suffix)
        with open(jacoby_file,'rb') as jf:
            jacoby = pickle.load(jf) #mean, std, all, rvals
        order = np.argsort(jacoby['mean'].numpy())
        order = order[::-1] #biggest will now be first
        jacobies.append(jacoby['mean'].numpy()[order])
    return np.stack(jacobies)

def GetPCIst(dir,nNN,suffix,k=None,snr=None):
    file = os.path.join(dir,"pcist","PCIst_nNN"+str(nNN)+suffix)
    if k is not None:
        file =file + "_k"+str(k)
        
    if snr is not None:
        file = file +"_snr"+str(snr)
    
    with open(os.path.join(file +".pkl"),'rb') as f:
        pci = pickle.load(f)
    return pci



def GetAlignment(dir, nNN,p, suffix):
    with open(os.path.join(dir,"RA",suffix+"Alignment.pkl"),'rb') as f:
        alignment = pickle.load(f)
    return alignment['RA'][nNN,p], alignment['NTK'][nNN,p]
    

params = GetModelFileParameters()
v1params = GetModelFileParameters(v1dd = True, add_dir = "Dales",suffix="23_4_")
params['gain'] = [g for g in params['gain'] if g not in [0.88,0.94,1.02,1.05,1.1,1.2]]

k = 1.2
snr = 2.2

fontsize = 14


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
    model_types = ["Meso","Meso_threshtoEM",'nNN = 28', 'nNN = 198' ]  
    model_labels = ["Meso","Meso_threshtoEM",'nNN = 28', 'nNN = 198' ]        
    model_files = [density_files,density_threshto_files,nNN28p1_files,nNN198p0_files] 
    suffixes = ["","_threshto5573","",""]
    add_dirs = ["","","",""] 
    nNN =[198,198,28,198]
    probs = [0.0,0.0,1.0,1.0]
    file_add = "meso"
elif V1DD:
    model_types = ["V1 2/3 4 Dales",'nNN = 28D',"V1 2/3 4 permuted","V1 Dales BP","V1 Dales +-"]
    model_labels = ["V1 2/3 4 Dales",r'$\mathbf{1}$' + ' nNN = 28D',r'$\mathbf{2}$'+ " V1 2/3 4 permuted",r'$\mathbf{6}$'+ " V1 Dales BP",r'$\mathbf{8}$' + " V1 Dales +-"]
    model_files = [EM_23_4_files,nNN28p1D_files, EM_23_4_permuted_files,EM_23_4_block_permuted_files, EM_23_4_flipped_files]
    suffixes = ["23_4_","","23_4_permuted_","23_4_block_permuted_", "23_4_flipped_"]
    add_dirs = ["Dales","Dales","","Dales","Dales"]
    nNN = [198,28,198,198,198]
    probs = [0.0,1.0,0.0,0.0,0.0]
    
    #model_types = ["V1 2/3 4 Dales","V1 2/3 4 permuted","V1 Dales BP","V1 Dales +-",'nNN = 28D']
    #model_labels = ["V1 2/3 4 Dales","r'$\mathbf{2} V1 2/3 4 permuted","r'$\mathbf{6} V1 Dales BP","r'$\mathbf{8} V1 Dales +-",'r'$\mathbf{1} nNN = 28D']
    #model_files = [EM_23_4_files, EM_23_4_permuted_files,EM_23_4_block_permuted_files, EM_23_4_flipped_files,nNN28p1D_files]
    #suffixes = ["23_4_","23_4_permuted_","23_4_block_permuted_", "23_4_flipped_",""]
    #add_dirs = ["Dales","","Dales","Dales","Dales"]
    #nNN = [198,198,198,198,28]
    #probs = [0.0,0.0,0.0,0.0,1.0]
elif SUPPLEMENTARY:
    model_types = ["DD",'V1 T',"V1 2/3 4 permutedT", "V1 Dales BPT"]
    model_files = [dd_files,EM_23_4_topology_files, EM_23_4_permuted_topology_files, EM_23_4_block_permuted_topology_files] 
    model_labels = [r'$\mathbf{3}$'+ " DD",r'$\mathbf{4}$'+' V1 T',r'$\mathbf{5}$'+" V1 2/3 4 permutedT", r'$\mathbf{7}$'+ " V1 Dales BPT"]
    add_dirs = ["","Dales","Dales","","Dales/topology"]
    suffixes = ["","23_4_topology_","23_4_permutedT_", "23_4_block_permuted_"]   
    #model_types = ["DD","V1 2/3 4 permutedT", "V1 Dales BPT",'V1 T']
    #model_labels = ["r'$\mathbf{3} DD","r'$\mathbf{5} V1 2/3 4 permutedT", "r'$\mathbf{7} V1 Dales BPT",'r'$\mathbf{4} V1 T']
    #model_files = [dd_files, EM_23_4_permuted_topology_files, EM_23_4_block_permuted_topology_files,EM_23_4_topology_files] 
    #add_dirs = ["","Dales","","Dales/topology","Dales"]
    #suffixes = ["","23_4_permutedT_", "23_4_block_permuted_","23_4_topology_"]           
    nNN  = [198,198,198,198]
    probs = [0.0,0.0,0.0,0.0]
    file_add = "supp"

'''    
suffixes = ["",      "",    "", "23_4_", "23_4_permuted_", "23_4_permutedT_","23_4_block_permuted_","23_4_block_permuted_","23_4_flipped_","23_4_topology_","","", #_thresh, _sparse,"23_4_transformed_",
add_dirs = ["", "Dales",    "", "Dales" ,"Dales","Dales",  "Dales/topology","Dales","Dales","","","",""]
nNN = [28,28,198,198,198,198,198,198,198,198,198,198,198]
probs = [1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
'''


pci_fig, pci_ax = plt.subplots()
lambda_fig, lambda_ax = plt.subplots()
dim_fig, dim_ax = plt.subplots()
Idim_fig, Idim_ax = plt.subplots()
acc_fig, acc_ax = plt.subplots()
ttmacc_fig, ttm_ax = plt.subplots()
trans_fig, trans_ax = plt.subplots()
Iweight_fig, Iweight_ax = plt.subplots()
Hweight_fig, Hweight_ax = plt.subplots()
Rweight_fig, Rweight_ax = plt.subplots()
pcilambda_fig, pcilambda_ax = plt.subplots()

ra_fig, ra_ax = plt.subplots()
ntk_fig, ntk_ax = plt.subplots()

axes = [pci_ax,  lambda_ax,  dim_ax,  Idim_ax,  acc_ax,  ttm_ax, Iweight_ax, Hweight_ax, Rweight_ax, ra_ax, ntk_ax, pcilambda_ax]#,     trans_ax]
figs = [pci_fig, lambda_fig, dim_fig, Idim_fig, acc_fig, ttmacc_fig,Iweight_fig,Hweight_fig,Rweight_fig,ra_fig, ntk_fig, pcilambda_fig]#, trans_fig]

measures = ["PCIst", "Largest Lyapunov", "Dimensionality", "Init Dimensionality", "Accuracy","Time to Max Acc",
            "IWeight Change","HWeight Change","RNorm Weight Change",
            "Representational Alignment","NTK Alignment","PCIst Largest Lyapunov"]#, "Gain Transition"]



plt.close("all")
plt.clf()
lambda_ax.plot(params['gain'],np.repeat(0,len(params['gain'])),linestyle='dashed',color='red',alpha =0.5)    

for m, type in enumerate(model_types):
    if model_types[m] in ["V1 2/3 4","V1 Dales +-",'V1 2/3 4 Dales']:
        gains = v1params['gain']
    else:
        gains = params['gain'] 
    
    linestyle = 'solid'
    model_acc = []
    acc_std = []
    model_lambda = []
    lambda_std = []
    model_ttm = []
    ttm_std = []
    model_dim = []
    dim_std = []
    model_Idim = []
    Idim_std = []
    dWi = []
    dWi_std = []
    dWh = []
    dWh_std = []
    dWr = []
    dWr_std = []
    model_gain_transition = []
    
    for g, gain in enumerate(gains):
        Tjacs = GetJacobies(model_files[m][g],suffix=suffixes[m]) 
        ky,stdky = GetKaplanYork(Tjacs)
        Ijacs = GetJacobies(model_files[m][g],suffix=suffixes[m]+"_init") 
        mmj,stdmj = GetMaxMeanJacobies(Ijacs)
        kyI, stdkyI = GetKaplanYork(Ijacs)
        model_lambda.append(mmj)
        lambda_std.append(stdmj)
        model_gain_transition.append(mmj>0)
        #if not np.isnan(gain_transition):
        #    gain_transition = params['gain'][gain_transition]
        #model_gain_transition.append(gain_transition)
        model_dim.append(ky)
        dim_std.append(stdky)
        model_Idim.append(kyI)
        Idim_std.append(stdkyI)
        max_acc, ttm = GetAccuracy(model_files[m][g])
        model_acc.append(np.mean(max_acc))
        acc_std.append(np.std(max_acc))
        model_ttm.append(np.mean(ttm))
        ttm_std.append(np.std(ttm))
        i, h, r = GetWeightChange(model_files[m][g])
        dWi.append(np.mean(i))
        dWi_std.append(np.std(i)/np.sqrt(10))
        dWh.append(np.mean(h))
        dWh_std.append(np.std(h)/np.sqrt(10))
        dWr.append(np.mean(r))
        dWr_std.append(np.std(r)/np.sqrt(10))
    print(model_gain_transition)
    dWi = np.array(dWi)
    dWh = np.array(dWh)
    dWr = np.array(dWr)
    dWi_std = np.array(dWi_std)
    dWh_std = np.array(dWh_std)
    dWr_std = np.array(dWr_std)
    model_lambda = np.array(model_lambda)
    lambda_std =  np.array(lambda_std)/np.sqrt(len(lambda_std))
    model_acc = np.array(model_acc)
    acc_std = np.array(acc_std)/np.sqrt(len(acc_std))
    model_ttm = np.array(model_ttm)
    ttm_std = np.array(ttm_std)/np.sqrt(len(ttm_std))
    model_dim = np.array(model_dim)
    dim_std = np.array(dim_std)/np.sqrt(len(dim_std))
    model_Idim = np.array(model_Idim)
    Idim_std = np.array(Idim_std)
    if np.any(model_gain_transition):
        gt = gains[GetGainTransition(model_gain_transition)]
    else:
        gt = 'NA'
    print(model_types[m],' ',"Gain Transition",' ',gt)
    linewidth = 1.0
    if ("microns" in model_types[m] or 'V1' in model_types[m]) and "Meso" not in model_types[m]:
        #model_types.extend(["V1 2/3 4","V1 2/3 4 permuted","V1 Dales BP","V1 Dales BPT","V1 Dales +-"])#,"V1 Box-Cox"])
        autumn = matplotlib.colormaps['RdPu']  
        pci = GetPCIst(os.path.dirname(model_files[m][g][0]),nNN[m],suffixes[m],k = k, snr = snr)
        model_pci = np.mean(np.stack(pci['IPCI'])[:,-1,:],1)
        pci_std = np.std(np.stack(pci['IPCI'])[:,-1,:],1)
        pci_std = pci_std/np.sqrt(len(pci_std))
        colors = [autumn(g) for g in np.linspace(0.25,1,4)]  
        ra, ntk = GetAlignment(os.path.dirname(model_files[m][g][0]),-1,-1,suffixes[m])
        linewidth = 1.0
        if 'V1 Dales BP' == model_types[m]:
            color = colors[1]
            #color = 'pink'
        elif 'V1 2/3 4 Dales' == model_types[m]:
            color = colors[3]
            linestyle = 'dashdot'
            linewidth=3.0
            #color = 'mediumorchid'
        elif 'V1 2/3 4 permuted' == model_types[m]:
            color = colors[0]
            #color ='plum'
        elif 'V1 2/3 4 permutedT' == model_types[m]:
            color = colors[0]
            linestyle = 'dashdot'
            #color ='plum'
        elif 'V1 Dales BPT' == model_types[m]:
            color = colors[1]
            linestyle = 'dashdot'
        elif "V1 Dales +-" == model_types[m]:
            #color = 'fuchsia'
            color = colors[2]
            linestyle = 'dotted'
        elif "'V1 T" == model_types[m]:
            color = colors[2]
            linestyle = 'dashed'
    elif "Meso" in model_types[m] :
        autumn = matplotlib.colormaps['Greens'] 
        pci = GetPCIst(os.path.dirname(model_files[m][g][0]),nNN[m],suffixes[m],k=k,snr = snr)
        model_pci = np.mean(np.stack(pci['IPCI'])[:,-1,:],1)
        pci_std = np.std(np.stack(pci['IPCI'])[:,-1,:],1)
        pci_std = pci_std/np.sqrt(len(pci_std))
        colors = [autumn(g) for g in np.linspace(.5,1.0,2)]  
        ra, ntk = GetAlignment(os.path.dirname(model_files[m][g][0]),-1,-1,suffixes[m])
        if 'threshto' in suffixes[m]:
            color = colors[1]
            linestyle = 'dashed'
            linewidth = 2.0
        elif 'thresh' in suffixes[m]:
            color = colors[3]
        elif 'sparsify' in suffixes[m]:
            color = colors[2]
        else:
            color = colors[0]
            linewidth = 3.0
            linestyle = 'dashdot'
    elif "DD" == model_types[m]:
        pci = GetPCIst(os.path.dirname(model_files[m][g][0]),nNN[m],suffixes[m],k=k,snr = snr)
        ra, ntk = GetAlignment(os.path.dirname(model_files[m][g][0]),-1,-1,suffixes[m])
        model_pci = np.mean(np.stack(pci['IPCI'])[:,-1,:],1)
        pci_std = np.std(np.stack(pci['IPCI'])[:,-1,:],1)
        pci_std = pci_std/np.sqrt(len(pci_std))
        color = "lime"
    else:
        if nNN[m] == 28 and add_dirs[m]!="Dales":
            autumn = matplotlib.colormaps['tab10']   
            colors = [autumn(g) for g in np.linspace(.25,1,len(params['nNN'][:-2]))]  
            ra, ntk = GetAlignment(os.path.dirname(model_files[m][g][0]),params['nNN'].index(28),-1,suffixes[m])
        elif  nNN[m] == 28 and add_dirs[m] == "Dales":
                colors = ["deepskyblue"]
                color = colors[0]
                ra, ntk = GetAlignment(os.path.dirname(model_files[m][g][0]),-1,-1,suffixes[m])
        elif nNN[m] == 198:
            autumn = matplotlib.colormaps['tab10']
            colors = [autumn(g) for g in np.linspace(.25,1,len(params['nNN'][:-2]))] 
            ra, ntk = GetAlignment(os.path.dirname(model_files[m][g][0]),-1,-1,suffixes[m])
        else:
            n = params['nNN'].index(nNN[m])
            color = colors[n]
            
        #p = params['p'].index(float(model_files[m][0][0].split("_p_")[1].split("_")[0]))
        if probs[m] == 1:
            p = -1
        else:
            p = params['p'].index(probs[m])
        pci = GetPCIst(os.path.dirname(model_files[m][g][0]),nNN[m],suffixes[m],k = k, snr = snr)
        model_pci = np.mean(np.stack(pci['IPCI'])[:,p,:],1)
        pci_std = np.std(np.stack(pci['IPCI'])[:,p,:],1)
        pci_std = pci_std/np.sqrt(len(pci_std))
        
        
    
    lambda_ax.plot(np.repeat(gt,5),np.linspace(-2,1,5),color = color,linestyle ='dashed', alpha = 0.7)
    #lambda_ax.plot(gains, model_lambda,label = "/".join((model_types[m],str(gt))),color = color,linestyle = linestyle)   
    #lambda_ax.fill_between(gains,model_lambda-lambda_std, model_lambda+lambda_std,alpha = 0.2,color = color)
    lambda_ax.errorbar(gains,model_lambda,yerr = lambda_std,color = color,linestyle = linestyle,label = "/".join((model_labels[m],str(gt))),linewidth = linewidth)
    if model_types[m] == "V1 2/3 4 Dales":
        lyaps = model_lambda
    
    pci_ax.plot(np.repeat(gt,5),np.linspace(0,120,5),color = color,linestyle ='dashed', alpha = 0.7)
    #pci_ax.plot(gains,model_pci,color = color,linestyle = linestyle)
    #pci_ax.fill_between(gains,model_pci-pci_std,model_pci+pci_std,alpha = 0.2,color = color)
    model_label = "/".join((model_labels[m],str(gt)))
    if model_types[m] =='V1 2/3 4 Dales' or model_types[m] == "Meso":
        model_label = '** '+ model_labels[m] +' **'
    pci_ax.errorbar(gains, model_pci,yerr = pci_std,color = color,linestyle = linestyle,label = model_label,linewidth = linewidth)
    
    acc_ax.plot(np.repeat(gt,5),np.linspace(20,100,5),color = color,linestyle ='dashed', alpha = 0.7)
    #acc_ax.plot(gains, model_acc,color = color,linestyle = linestyle)
    #acc_ax.fill_between(gains, model_acc -acc_std, model_acc+acc_std, alpha = 0.2,color = color)
    acc_ax.errorbar(gains, model_acc, yerr = acc_std, color = color,linestyle = linestyle,linewidth = linewidth)
    
    ttm_ax.plot(np.repeat(gt,5),np.linspace(1000,4000,5),color = color,linestyle ='dashed', alpha = 0.7)
    #ttm_ax.plot(gains, model_ttm,color = color,linestyle = linestyle)    
    #ttm_ax.fill_between(gains, model_ttm-ttm_std, model_ttm+ttm_std, alpha = 0.2,color = color)
    ttm_ax.errorbar(gains, model_ttm, yerr = ttm_std, color = color,linestyle = linestyle)
    
    pcilambda_ax.errorbar(model_lambda, model_pci,xerr= lambda_std,yerr=pci_std, color = color,linestyle = linestyle,label = model_label,linewidth = linewidth)
    
    dy = dim_ax.get_ylim()
    dim_ax.plot(np.repeat(gt,5),np.linspace(0,dy[1],5),color = color,linestyle =linestyle, alpha = 0.7)
    #dim_ax.plot(gains,model_dim,color = color,linestyle = linestyle)
    #dim_ax.fill_between(gains,model_dim- dim_std, model_dim+dim_std, alpha = 0.2,color = color)
    dim_ax.errorbar(gains, model_dim, yerr = dim_std, color = color,linestyle = linestyle,linewidth = linewidth)
    if V1DD:
        dim_ax.set_ylim((0,30))
    
    
    Idim_ax.plot(np.repeat(gt,5),np.linspace(0,80,5),color = color,linestyle ='dashed', alpha = 0.7)
    #Idim_ax.plot(gains,model_Idim,color = color,linestyle = linestyle)
    #Idim_ax.fill_between(gains,model_Idim- Idim_std, model_Idim+Idim_std, alpha = 0.2,color = color)
    Idim_ax.errorbar(gains, model_Idim, yerr = Idim_std, color = color,linestyle = linestyle,linewidth = linewidth)
    #Idim_ax.set_ylim((0,10))
    #trans_ax.plot(params['gain'],model_gain_transition)
    
    #Iweight_ax.plot(gains,dWi,color = color,linestyle = linestyle)
    #Iweight_ax.fill_between(gains,dWi - dWi_std, dWi + dWi_std, alpha = 0.2, color = color)
    dy = Iweight_ax.get_ylim()
    if dy[1] < 5:
        dy = (0.0,5.0)
    Iweight_ax.plot(np.repeat(gt,5),np.linspace(0,dy[1],5),color = color,linestyle = linestyle, alpha = 0.7)
    Iweight_ax.errorbar(gains, dWi, yerr = dWi_std, color = color,linestyle = linestyle,linewidth = linewidth)
    
    
   # Hweight_ax.plot(gains,dWh,color = color,linestyle = linestyle)
    #Hweight_ax.fill_between(gains,dWh - dWh_std, dWh + dWh_std, alpha = 0.2, color = color)
    dy = Hweight_ax.get_ylim()
    if dy[1] < 5:
        dy = (0.0,5.0)
    Hweight_ax.plot(np.repeat(gt,5),np.linspace(0,dy[1],5),color = color,linestyle = linestyle, alpha = 0.7)
    Hweight_ax.errorbar(gains, dWh, yerr = dWh_std, color = color,linestyle = linestyle,linewidth = linewidth)
    
    
    #Rweight_ax.plot(gains,dWr,color = color,linestyle = linestyle)
    #Rweight_ax.fill_between(gains,dWr - dWr_std, dWr + dWr_std, alpha = 0.2, color = color)
    dy = Rweight_ax.get_ylim()
    if dy[1] < 5:
        dy = (0.0,5.0)
    Rweight_ax.plot(np.repeat(gt,5),np.linspace(0,dy[1],5),color = color,linestyle = linestyle, alpha = 0.7)
    Rweight_ax.errorbar(gains, dWr, yerr = dWr_std, color = color,linestyle = linestyle,linewidth = linewidth)

    if gt == 'NA':
        ntk_ax.errorbar(gains, np.mean(ntk,1), yerr = np.std(ntk,1)/np.sqrt(10),color = color, linestyle = linestyle,linewidth = linewidth)
    else:
        dy = ra_ax.get_ylim()
        wgt = gains.index(gt)
        ntk_ax.plot(np.repeat(gt,5),np.linspace(0,dy[1],5),color = color,linestyle = linestyle, alpha = 0.7)
        ntk_ax.errorbar(gains[:wgt+1], np.mean(ntk[:wgt+1,:],1), yerr = np.std(ntk[:wgt+1,:],1)/np.sqrt(10),color = color, linestyle = linestyle,linewidth = linewidth)
        ntk_ax.errorbar(gains[wgt:], np.mean(ntk[wgt:,:],1), yerr = np.std(ntk[wgt:,:],1)/np.sqrt(10),color = color, linestyle = linestyle,alpha=0.3,linewidth = linewidth)
    
    dy = ra_ax.get_ylim()
    ra_ax.plot(np.repeat(gt,5),np.linspace(0,dy[1],5),color = color,linestyle = linestyle, alpha = 0.7)
    ra_ax.errorbar(gains, np.mean(ra,1),yerr = np.std(ra,1)/np.sqrt(10), color = color, linestyle = linestyle,linewidth = linewidth)
    
lambda_ax.set_ylim(-2.2, 1.0)
lambda_ax.set_xlim((0.5,5))    


pci_ax.legend(fontsize=fontsize,loc="upper right")
pci_legend = pci_ax.get_legend()
texts = pci_legend.get_texts()
if not SUPPLEMENTARY:
    texts[0].set_weight("bold")

pcilambda_ax.legend(fontsize=fontsize,loc="lower left")
pcilambda_ax.set_ylim((0,80))
pcilambda_legend = pcilambda_ax.get_legend()
texts = pcilambda_legend.get_texts()
if not SUPPLEMENTARY:
    texts[0].set_weight("bold")



[ax.set_xlabel('Gain',fontsize = fontsize) for ax in axes if ax != pcilambda_ax]
pcilambda_ax.set_xlabel("Maximum Lyapunov Exponent")
[ax.set_ylabel(measures[m],fontsize = fontsize) for m,ax in enumerate(axes) if ax != pcilambda_ax]
pcilambda_ax.set_ylabel("PCIst")

[ax.tick_params(axis = "x",labelsize = fontsize) for m,ax in enumerate(axes)]
[ax.tick_params(axis = "y",labelsize = fontsize) for m,ax in enumerate(axes)]
[fig.tight_layout() for fig in figs]

[fig.savefig(os.path.join("RNN","EM_column","v1dd_nNN28_"+measures[m].replace(" ","_")+outfile+".png"), dpi = 600) for m,fig in enumerate(figs)]


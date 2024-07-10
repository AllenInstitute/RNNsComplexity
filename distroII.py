#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:22:18 2022

@author: danamastrovito
"""
#exec(open("distroII.py").read())


from utils import GetInitializedModel, GetModelWeights, GetEigenAdjacency
from utils import  GetModelFiles,GetModelFileParameters, GetAdjacencyRank
from utils import Shuffle_Weights,ParticipationRatio, InverseParticipationRatio
import os 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import pickle
import glob
import itertools
import collections
import socket
import networkx as nx
import torch

def flatten(iterable):
    for el in iterable:
        if isinstance(el, collections.Iterable) and not isinstance(el, str): 
            yield from flatten(el)
        else:
            yield el



Test = False
Init = True
Density = False
Include_Participation_Ratio = False
V1DD = False
coeff_var = False
SPLIT = False
plt.rcParams['font.family'] = 'sans-serif'

rng = np.random.default_rng(8)

fontsize = 20

    

if Init:
    suffix = "_initial"
else:
    suffix = suffix

if Include_Participation_Ratio:
    suffix = suffix+"_PR"
else:
    suffix = suffix


def plot_spectral_radius(spectral_radius,outdir,suffix=""):
    plt.clf()
    mpl.rcParams.update({'font.size': fontsize})
    mnI = np.min(np.imag( spectral_radius))
    mxI = np.max(np.imag( spectral_radius))
    mnR = np.min(np.real( spectral_radius))
    mxR = np.max(np.real( spectral_radius))
    ylim = (mnI-(.2*np.abs(mxI-mnI)),mxI+(.1*np.abs(mxI)))
    xlim = (mnR- (.2*np.abs(mxR - mnR)),mxR + (.1*np.abs(mxR)))
       
    plotidx = 1
    axes =[]
    for n,nNN in enumerate(params['nNN']):
        for p,prob in enumerate(params['p']):
            axes.append(plt.subplot(6,5,plotidx))
            plt.scatter(np.real(spectral_radius[n,p,:]),np.imag(spectral_radius[n,p,:]),marker='.')
            plt.title('nNN='+str(nNN)+ ' p='+str(prob),fontsize=fontsize)
            plotidx +=1
            plt.xlim(xlim)
            plt.ylim(ylim)
        
    
    for ax,axis in enumerate(axes):
        if ax %5 !=0:
            axis.get_shared_y_axes().join(axes[0],axis)
            axis.set_yticklabels([])
        else:
            axis.set_yticklabels(axis.get_yticks(),fontsize=fontsize)
            axis.set_ylabel("Imag")
        if ax < len(axes) -5:
            axis.get_shared_x_axes().join(axes[-1],axis)
            axis.set_xticklabels([])
            
        else:
            axis.set_xticklabels(axis.get_xticks(),rotation=90,fontsize=fontsize)
            axis.set_xlabel("Real")
            
    plt.tight_layout()    
    plt.savefig(os.path.join(outdir,"Spectral_radius"+suffix+".png"),transparent = False,dpi =300)    




def plot_eigen_spectrum(eigen_values,rank ,filename=None):
    plt.clf()
    mpl.rcParams.update({'font.size': fontsize})
    axes = []
    mnI = np.min(np.imag(eigen_values))
    mxI = np.max(np.imag(eigen_values))
    mnR = np.min(np.real(eigen_values))
    mxR = np.max(np.real(eigen_values))
    ylim = (mnI-(.1*np.abs(mnI)),mxI+(.1*np.abs(mxI)))
    xlim = (mnR-(.1*np.abs(mnR)),mxR + (.1*np.abs(mxR)))
    for run,eig in enumerate(eigen_values):    
        axes.append(plt.subplot(5,2,run+1))
        plt.scatter(np.real(eig),np.imag(eig),marker='.')
        if run >=len(eigen_values) -2:
            plt.xlabel("Real")
        if run % 2 ==0:
            plt.ylabel('Imag')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title("Rank " + str(rank[run]))
    
    for ax,axis in enumerate(axes):
        if ax %2 !=0:
            axis.get_shared_y_axes().join(axes[0],axis)
            axis.set_yticklabels([])
        if ax < len(axes) -2:
            axis.get_shared_x_axes().join(axes[-1],axis)
            axis.set_xticklabels([])
    plt.tight_layout()
    plt.savefig( filename +"_eigen_spectrum.png",transparent = False, dpi = 300)



def plot_eigen_spectrumII(eigen_values,filename=None,PR= None,xlim = None,ylim = None,Mvals = None, Wvals = None,outdir = '"'):
    plt.clf()
    if PR is not None:
        cmap = cm.cool
        bounds = np.linspace(0,1,256)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    else:
        cm_subsection = np.linspace(0,198,10) 
        colors = [ cm.tab10(x) for x in cm_subsection ]
    axes = []
    print(xlim)
    if xlim is  None and ylim is None:
        if Mvals is not None and Wvals is not None:
            I = list(np.imag(np.array(list(flatten(eigen_values)))))
            I.extend(np.imag(np.array(list(flatten(Mvals)))))
            I.extend(np.imag(np.array(list(flatten(Wvals)))))
            mnI = np.min(I)
            mxI = np.max(I)
            R = list(np.real(np.array(list(flatten(eigen_values)))))
            R.extend(np.real(np.array(list(flatten(Mvals)))))
            R.extend(np.real(np.array(list(flatten(Wvals)))))
            mnR = np.min(R)
            mxR = np.max(R)
            mn = np.min([mnI, mnR])
            mx = np.max([mxI, mxR])
            ylim = (mn-(.1*np.abs(mn)),mx+(.1*np.abs(mx)))
            xlim = (mn-(.1*np.abs(mn)),mx + (.1*np.abs(mx)))
        else:
            mnI = np.min(np.imag(np.array(list(flatten(eigen_values)))))
            mxI = np.max(np.imag(np.array(list(flatten(eigen_values)))))
            mnR = np.min(np.real(np.array(list(flatten(eigen_values)))))
            mxR = np.max(np.real(np.array(list(flatten(eigen_values)))))
            mn = np.min([mnI, mnR])
            mx = np.max([mxI, mxR])
            ylim = (mnI-(.1*np.abs(mnI)),mxI+(.1*np.abs(mxI)))
            xlim = (mnR-(.1*np.abs(mnR)),mxR + (.1*np.abs(mxR)))
    if np.abs(xlim[0]) < xlim[1]:
        xlim = (-1*xlim[1],xlim[1])
    if xlim[0] > -1. or xlim[1] < 1:
        xlim = (-1.1,1.1)
    if ylim[0] > -1. or ylim[1] < 1:
        ylim = (-1.1,1.1)
    #lim = (mn-(.1*np.abs(mn)),mx +(.1*np.abs(mx)))
    plotidx = 1
    axis = plt.subplot(1,1,1)     
    max_eig = np.max(np.abs(eigen_values))
    for eig in np.arange(eigen_values.size): 
        if PR is not None:
            color = cmap(PR[eig].astype(np.int32)/np.max(list(flatten(PR))))
        else:
            color = 'blue'
        circle = plt.Circle((0, 0), 1.0, linestyle='dashed',fill = False)
        #if np.abs(eigen_values[eig]) == max_eig:
        #    plt.scatter(np.real(eigen_values[eig]),np.imag(eigen_values[eig]),color = color,marker='+')
        #else:
        plt.scatter(np.real(eigen_values[eig]),np.imag(eigen_values[eig]),color = color,marker='.')
    '''
    if Mvals is not None:
        plt.scatter(np.real(Mvals[n][p][r]),np.imag(Mvals[n][p][r]),color = 'r',marker='x',alpha = 0.3)
    if Wvals is not None:
        plt.scatter(np.real(Wvals[n][p][r]),np.imag(Wvals[n][p][r]),color = 'r',marker='*',alpha = 0.3)
    '''    
    #plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plotidx += 1 
    axis.add_patch(circle)
    axis.set_aspect('equal')
    axis.set_facecolor('xkcd:white')
    axis.set_xlabel(r'Re($\lambda$)',fontsize = 20)
    axis.set_ylabel(r'Im($\lambda$)',fontsize = 20)
    plt.tick_params(axis = "x",labelsize = 20) 
    plt.tick_params(axis = "y",labelsize = 20) 
    plt.tight_layout()
 
    plt.savefig( os.path.join(outdir,os.path.basename(filename) +"_eigen_spectrum.png"),transparent = False, dpi = 300)
    '''
    if PR is not None:
        plt.clf()
        cb = mpl.colorbar.ColorbarBase(plt.gca(),cmap=cmap,norm=norm,orientation='vertical')
        plt.tight_layout()
        plt.savefig(filename+"_legend.png",dpi = 300,transparent = False)
    '''
    plt.clf()
        




hostname = socket.gethostname()
if 'zuul' in hostname:
    home ="/home/dana/"
else:
    home = ""

params =  GetModelFileParameters(home = home)

if Test:
    params = {"nNN":[58],'p':[.2]}
elif Density:
    params={'nNN':[198],'p':[0]}
    params = GetModelFileParameters(Density = True,home = home)
    outdir = os.path.join("RNN","Density","eigen_spectrum")
    Density = True
    add_dir = ""
    suffix = "_threshto5573"
    Dales = False
elif coeff_var:
    SPLIT = True
    '''
    _mean_0.05_std_0.05_ii_0.075_0.075', 
    '_mean_0.05_std_0.05_ii_0.05_0.05',
    '_mean_0.05_std_0.025_ii_0.025_0.025',
    '_mean_0.05_std_0.025_ii_0.075_0.075',
    '_mean_0.05_std_0.025_ii_0.05_0.05',
    '_mean_0.05_std_0.025_ii_0.05_0.025', 
    '_mean_0.05_std_0.05_ii_0.025_0.025'
    '''
    suffix = "mean_0.05_std_0.05_ii_0.05_0.025_"
    add_dir = ""
    params = GetModelFileParameters(coeff_var = coeff_var, add_dir  = add_dir, suffix = suffix,home = home)
    outdir =  os.path.join("RNN","coefficient_variation","eigen_spectrum")
    title = suffix
elif V1DD:
    suffix = "23_4_"
    Dales = True
    add_dir = os.path.join('Dales')
    params = GetModelFileParameters(v1dd = V1DD, add_dir  = add_dir, suffix = suffix,home = home)
    outdir = os.path.join("RNN","EM_column",'v1dd',add_dir,"eigen_spectrum")
    title = "V1 2/3 4"
else:
    add_dir = ""
    Dales = False
    params =  GetModelFileParameters(home = home)
    params['nNN'] = [nNN for nNN in params['nNN'] if nNN in [ 8,28,64,198]]
    params['p'] = [1.0]
     #[0.5, 0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,5.0] #[0.5,0.75, 0.88, 0.94, 1.0, 1.02, 1.05, 1.1, 1.2, 1.25]
    #'gain': [0.5, 0.75, 0.88, 0.94, 1.0, 1.02, 1.05, 1.1, 1.2, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 5.0]}
    outdir = os.path.join("RNN","Gaussian","eigen_spectrum")

if not os.path.exists(outdir):
    os.mkdir(outdir)

params['gain']  = [0.5,1.0,3.0]
for nNN in params['nNN']:
    if nNN in [8,28,64,198,500,1000]:
        p = 0.0
    else:
        p = 1.0
    for gain in params['gain']:
        Files  = GetModelFiles(nNN, p, gain = gain, NetworkType='RNN')#,v1dd = V1DD,add_dir = add_dir, suffix= suffix,density = Density, home = home)
        for file in Files:
            print(file)
            A = GetModelWeights(file,initial = Init )
            EigVals,EigVecs = GetEigenAdjacency(A)
            IPR = InverseParticipationRatio(EigVecs)
            PR =ParticipationRatio(EigVals)
            print('plotting...')
            plot_eigen_spectrumII(EigVals, filename=file,outdir = outdir)#,PR = IPR)





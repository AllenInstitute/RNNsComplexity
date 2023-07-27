#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:44:43 2022

@author: danamastrovito
"""

import glob
import os
import numpy as np
import torch
import Network
from numpy import linalg as LA
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LinearRegression
from sympy import limit, oo, Symbol
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def ParticipationRatio(eigenvalues):
    return np.sum(eigenvalues)**2/np.sum(eigenvalues**2)

def InverseParticipationRatio(eigenvectors):
    I = [np.sum(np.abs(eigenvectors[:,i])**2)**2/(np.sum(np.abs(eigenvectors[:,i])**4)) for i in np.arange(eigenvectors.shape[1])]
    return np.array(I)


def normalize_std(x):
    return x/(2*np.std(x))

def modified_chaos_test(x,c=1.7):
    sigma = 0.5
    N = len(x)
    eta = np.random.uniform(-.5,.5,N)
    p = lambda n: np.sum(x[:n-1]*np.cos(np.arange(1,n)*c))
    q = lambda n: np.sum(x[:n-1]*np.sin(np.arange(1,n)*c))
    #((p(j + n) - p(j))**2 + (q(j+n)-q(j))**2) + sigma*eta[n]
    #(p(j+n) - p(j))**2 + (q(j +n)-q(j))**2 + sigma*eta
    #K = np.corrcoef(n,M(n))




def chaos_zero_one_test(x,time = None,c = 1.7,tau = 29,plot = False,file =""):
    #x should be one dimenasional or as [samples,variables] 
    #which will be reduced to a single dimension
    if len(np.squeeze(x).shape) >1 :
        pca = PCA(n_components=1)
        x = pca.fit_transform(x)
    if file != "":
        file = file+"_"
    theta = lambda t: (c*t) + np.sum(x[:t])
    p = lambda t: np.sum(x[:t]*np.cos(theta(t)))
    nt = len(x)
    if time is None:
        time = np.arange(1,nt)
    MSD = (np.array([p(t) for t in range(nt)]) - p(tau))**2
    #MSD = np.diff(P,n=tau)**2
    M = 1/nt*np.array([np.sum(MSD[:t]) for t in np.arange(1,nt)])
    reg = LinearRegression().fit(np.log10(time).reshape(-1,1),np.log10(M+1).reshape(-1,1))    
    K = reg.coef_
    if K >.5:
        title="Chaotic"
    else:
        title = "non-Chaotic"
    if plot:
        plt.clf()
        plt.plot(np.log10(time),np.log10(M+1))
        plt.plot(np.log10(time),reg.predict(np.log10(time).reshape(-1,1)),color='red')
        plt.xlabel("log time")
        plt.ylabel("log MSD")
        plt.title("K = %.3f "%K+title)
        plt.legend(['Data','MSD'])
        plt.savefig(file+"zero_one_test.png")
    return K
    

def threshold_matrix(mat,thresh):
    '''
    mat = np.around(mat,decimals = 10)
    thresholded = mat.copy()
    thresholded[np.where(np.abs(thresholded) <= np.abs(thresh))] = 0.001
    '''
    thresholded = mat.copy()
    thresholded[np.where(thresholded <thresh)] = 0
    return thresholded

def power_adjacency(mat,beta):
    wij = ((mat+1)/2.)**beta
    return wij

def compute_percentile(mat,percentile):
    sorted_mat = np.sort(mat.flatten())
    idx = int(np.round(percentile*len(sorted_mat)))
    return(sorted_mat[idx])
 

def UniquePartitions(partitions):
    boolean_partitions = [partition.astype(bool) for partition in partitions]
    for part in boolean_partitions:
        w = [i for i,partition in enumerate(boolean_partitions) if ((part == partition).all()) | ((part == ~partition).all())  ]
        if len(w) >1:
            w.reverse()
            for wi in w[1:]:
                boolean_partitions.pop(wi)
    unique_partitions = [partition.astype(int) for partition in boolean_partitions]
    return unique_partitions       
                
def GetPartitions(Aij):
    beta = np.logspace(0,1,num = 10)
    percentiles = np.arange(.0,1,.25)
    sc = SpectralClustering(2, affinity='precomputed', n_init=100,assign_labels='discretize')
    Partitions = []
    for b in beta:
        poweradj = power_adjacency(Aij,b)
        for p in percentiles:
            thresh = compute_percentile(poweradj,p)
            adj_thresh = threshold_matrix(poweradj,thresh)
            scgt = sc.fit_predict(adj_thresh)             
            if len(np.where(scgt==0)[0]) != 1 and len(np.where(scgt==1)[0]) != 1:
                Partitions.append(scgt)
    for b in beta:
        poweradj = power_adjacency(Aij.T,b)
        for p in percentiles:
            thresh = compute_percentile(poweradj,p)
            adj_thresh = threshold_matrix(poweradj,thresh)
            scgt = sc.fit_predict(adj_thresh)             
            if len(np.where(scgt==0)[0]) != 1 and len(np.where(scgt==1)[0]) != 1:
                Partitions.append(scgt)
    return Partitions




def GetEigenAdjacency(mat,order = False):
    vals,vecs = LA.eig(mat)
    if order:
        valorder = np.flip(np.argsort(np.abs(vals)))
        vals = vals[valorder]
        vecs = vecs[:,valorder]
    return vals,vecs


def GetAdjacencyRank(mat):
    rank  = LA.matrix_rank(mat)
    return rank


def SetModelHiddenWeights(net,mat):
    '''
    Sets Model Hidden weights

    Parameters
    ----------
    net : Network class instance
        DESCRIPTION.
    mat : numpy array
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if net.ModelType == 'RNN':
        net.model.cell.weight_hh.data = torch.from_numpy(mat).type(torch.float32)
        print("Hidden Weights reset")
    elif net.ModelType == 'Kuramoto':
        net.model.cell.Wh.data = torch.from_numpy(mat).type(torch.float32)
        
        


def Shuffle_Weights(mat):
    '''
    Returns shuffled weight matrix, while preserving edge locations.

    Parameters
    ----------
    mat : np.array [nhidden,nhidden]
        weight matrix to shuffle.

    Returns
    -------
    shuffled : np.array
        shuffled matrix weights.

    '''
    shape = mat.shape
    mat = mat.flatten()
    shuffled = np.zeros(mat.shape)
    nz = np.nonzero(mat)
    mat = mat[nz]
    sample = np.random.choice(mat,size = len(nz[0]),replace=False)
    #locations = np.random.choice(np.arange(nel),len(nz[0]),replace=False)
    shuffled[nz] = sample
    shuffled = shuffled.reshape(shape)
    return shuffled


def GetModelInputWeights(file,initial = False,device='cpu'):
    model = torch.load(file,map_location = device)
    if initial:
        ih = [param[1].detach() for param in model['InitialState'] if 'weight_ih' in param[0]][0]
    else:
        ih = [param[1].detach() for param in model['Parameters'] if 'weight_ih' in param[0]][0]
    return ih


def GetModelReadout(file,initial = False,device='cpu'):
    model = torch.load(file,map_location = device)
    if initial:
        readout = [param[1].detach() for param in model['InitialState'] if 'readout' in param[0]]
    else:
        readout = [param[1].detach() for param in model['Parameters'] if 'readout' in param[0]]
    return readout
        
                
def GetModelWeights(file,initial = False,device='cpu'):
    '''
    Get Model hidden layer weights

    Parameters
    ----------
    file : str
        filename from which to get model weights.
    intial : boolean, optional
        If True return initial pre-trained weights. The default is False.

    Returns
    -------
    Wh : np.array
        Model hiddden weights.

    '''
    model = torch.load(file,map_location = device)
    if initial:
        paramIdx = [idx for idx,param in enumerate(model['InitialState']) if 'Wh' in param[0] or 'weight_hh' in param[0]][0]  
        Wh = np.array(model['InitialState'][paramIdx][1].detach())
    else:
        paramIdx = [idx for idx,param in enumerate(model['Parameters']) if 'Wh' in param[0] or 'weight_hh' in param[0]][0]  
        Wh = np.array(model['Parameters'][paramIdx][1].detach())
                
    return Wh
    


def GetModelFileParameters(NetworkType = 'RNN',Density = False,em = False,digits = False, pixel = False, narrow = False, suffix="", add_dir=None):
    '''
    
    
    Parameters
    ----------
    NetworkType : TYPE, optional
        DESCRIPTION. The default is 'RNN'.
    
    Returns
    -------
    params : dictionary with keys 'nNN' and 'p'
        run parameters: .
    
    '''
    assert not (em and Density)
    
    if NetworkType == 'RNN':
        dir = "RNN"
    elif NetworkType == "Kuramoto":
        dir = "Kuramoto"
    
    if digits or narrow or pixel:
        if digits:
            dir = os.path.join(dir, "digits")
        elif pixel:
            dir = os.path.join(dir, "pixel_by_pixel")
        elif narrow:
            dir = os.path.join(dir, "narrow")
     
    if add_dir is not None:
        add_dir =  add_dir
    else:
        add_dir = ""
        
    nNN = []
    p = []
    gain = []
    if Density:
        dir = os.path.join(dir,"Density")
        files = glob.glob(os.path.join(dir,"Density_198*ninputs_28*Run_[0-9]*"))
        if len(files) >0:
            nNN = [198]
            p = [0]
            gain = list(set([float(file.split("gain_")[1].split("_")[0]) for file in files]))
            gain.sort()
    elif em:
        files = glob.glob(os.path.join(dir,"EM_column","EM_column_198*ninputs_28_gain_[0-9].[0-9][0-9]_"+suffix+"Run_[0-9]*"))
        files.extend(glob.glob(os.path.join(dir,"EM_column","EM_column_198*ninputs_28_gain_[0-9][0-9].[0-9]_"+suffix+"Run_[0-9]*")))
        files.extend(glob.glob(os.path.join(dir,"EM_column","EM_column_198*ninputs_28_gain_[0-9].[0-9]_"+suffix+"Run_[0-9]*")))
        if len(files) >0:
            nNN = [198]
            p = [0]
            gain = list(set([float(file.split("gain_")[1].split("_")[0]) for file in files]))
            gain.sort()
    else:
        files = glob.glob(os.path.join(dir,"Gaussian",add_dir,"WattsStrogatz_*ninputs_*"+suffix+"Run_[0-9]*"))
        nNN = list(set([ int(file.split('nNN')[1].split("_")[0]) for file in files] ))
        p = list(set([ float(file.split('p_')[1].split("_")[0]) for file in files] ))
        gain = list(set([float(file.split("gain_")[1].split("_")[0]) for file in files]))
        nNN.sort()
        p.sort()
        gain.sort()
        
    return {'nNN':nNN,'p':p,'gain':gain}
  
def GetModelFiles(nNN, p,gain=None,NetworkType='RNN',noise = False,density=False,em = False,digits = False,pixel = False, narrow = False, add_dir = None,suffix=""):
    '''
    Return Trained Model Filenames with particular number of nearest neighbors and probability of rewiring

    Parameters
    ----------
    type: str
        Network type. Can be one of 'RNN' or "Kuramoto"
    nNN : int
        Number of nearest neighbors in Watts Strogatz Ring.
    p : float
        probability of rewiring initial nearest-neighbor structure.

    Returns
    -------
    files : list
        list of files.

    '''
    assert not (em and density)
    if NetworkType == 'RNN':
        dir = "RNN"
    elif NetworkType == "Kuramoto":
        dir = "Kuramoto"
    
    if digits or narrow or pixel:
        if digits:
            dir = os.path.join(dir, "digits")
        elif pixel:
            dir = os.path.join(dir, "pixel_by_pixel")
        elif narrow:
            dir = os.path.join(dir, "narrow")
    
    if density:
        dir = os.path.join(dir, "Density")
    elif em:
        dir = os.path.join(dir, "EM_column")
    else:
        dir = os.path.join(dir, "Gaussian")
        
    if noise:
        dir = os.path.join(dir, "noise")
    
    if add_dir is not None:
        dir = os.path.join(dir,add_dir)
        
    if not density and not em:
        files = glob.glob(os.path.join(dir,"WattsStrogatz_*ninputs_*nNN*_p*_gain_???"+suffix+"_Run_[0-9]*"))
        files.extend(glob.glob(os.path.join(dir,"WattsStrogatz_*ninputs_*nNN*_p_*_gain_????"+suffix+"_Run_[0-9]*")))
        files = [f for f in files if 'nNN'+str(nNN) in f and 'p_'+"{:.1f}".format(p) in f and 'gain_'+str(gain) in f]
    elif density:
        files = glob.glob(os.path.join(dir,"Density_198*ninputs_28_Density_gain_???"+suffix+"_Run_[0-9]*"))
        files.extend(glob.glob(os.path.join(dir,"Density_198*ninputs_28_Density_gain_????"+suffix+"_Run_[0-9]*")))
        files = [f for f in files if str(nNN) in f  and 'gain_'+str(gain) in f]
    elif em:
        print(dir)
        files = glob.glob(os.path.join(dir,"EM_column_198*ninputs_28_gain_[0-9].[0-9][0-9]_"+suffix+"Run_[0-9]*"))
        files.extend(glob.glob(os.path.join(dir,"EM_column_198*ninputs_*_gain_[0-9][0-9].[0-9]_"+suffix+"Run_[0-9]*")))
        files.extend(glob.glob(os.path.join(dir,"EM_column_198*ninputs_*_gain_[0-9].[0-9]_"+suffix+"Run_[0-9]*")))      
        files = [f for f in files if str(nNN) in f  and 'gain_'+str(gain) in f]
       
        
    files =[f for f in files if  ".png" not in f]
    files =[f for f in files if  ".npy" not in f]
    runs = [f.split("Run_")[1] for f in files]
    runs = np.array(runs).astype(int)
    order = np.argsort(runs)
    files = list(np.array(files)[order])
    
    if len(files) < 1:
        print("No Files found with specified run params")
    return files


def GetInitializedModel(file,initial = False,batch_size = None,noise = False, device='cpu'):
    '''
    Load a model and return model instance initialized and ready to run:
    net.forward(images)
    net.train()
    etc

    Parameters
    ----------
    file : str
        filename from which to initialize model.
        
    intial : boolean, optional
        If True return model reinitialized with pre-trained weights. The default is False.

    Returns
    -------
    net : Network class instance
        network model reinitialized with parametersr.

    '''
    model = torch.load(file,map_location=device)
    if batch_size is None:
        batch_size = model['config']['batch_size']
    
    net = Network.Net(model['config']['ModelType'],model['config']['ninputs'],model['config']['nhidden'],\
                      batch_size,input_bias=False,noise = noise,device = device) 
    if 'gain' in model.keys():
        net.initialize(model['ConnType']['ConnType'],nNN = model['ConnType']['nNN'],p=model['ConnType']['p'],gain = model['Gain'])
    else:
        net.initialize(model['ConnType']['ConnType'],nNN = model['ConnType']['nNN'],p=model['ConnType']['p'],gain = 1.0)
         
    
    if initial:
        net.Reinitialize(file,initial = True,device = device)
    else:
        net.Reinitialize(file,device=device)
    return net
              



def GetMNIST_TestData(net):
    '''
    
    Parameters
    ----------
    net : Network class instance
        DESCRIPTION.

    Returns
    -------
    One sMNIST Batch.

    '''
    for images,labels in iter(net.dl.test_loader):
        images = images.reshape(net.batch_size, net.ninputs, int(images.shape[1]/net.ninputs))
        images = images.permute(2,0,1)
        yield images,labels
   



def GetMNIST_TrainData(net):
    '''
    
    Parameters
    ----------
    net : Network class instance
        DESCRIPTION.

    Returns
    -------
    One sMNIST Batch.

    '''
    for images,labels in iter(net.dl.train_loader):
        images = images.reshape(net.batch_size, net.ninputs, int(images.shape[1]/net.ninputs))
        images = images.permute(2,0,1)
        return images,labels
                    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:47:29 2021

@author: danamastrovito
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.utils
import matplotlib.pyplot as plt
import os 
import networkx as nx
import numpy as np
import pickle
import Parameters
import argparse
import sys
import socket
from scipy import stats

hostname = socket.gethostname()
if 'zuul' in hostname:
    sys.path.append("/home/dana/allen/programs/braintv/workgroups/tiny-blue-dot/GLIFS_ASC/main")
else:
    sys.path.append("/allen/programs/braintv/workgroups/tiny-blue-dot/GLIFS_ASC/main")
from models.networks import BNNFC

nout = 10    #For sMNIST this should not change

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# if batch is size 5
#Valid numbers for inputs include: 8,14,16,28,49,56,98,112,196,392

def transform_log_normal(weights):
    return torch.exp(weights)


def block_permutation(CM):
    excitatory = list(np.where(np.array([np.all(CM[row,np.nonzero(CM[row,:])] > 0) for row in range(CM.shape[0])]) == True)[0])
    inhibitory = list(np.where(np.array([np.all(CM[row,np.nonzero(CM[row,:])] < 0) for row in range(CM.shape[0])]) == True)[0])
    locs = np.ix_(excitatory, excitatory)
    E_E = CM[locs]
    nz = np.nonzero(E_E)
    permuted = rng.choice(E_E[nz],nz[0].size, replace = False)
    CM[locs][nz] = permuted
    locs = np.ix_(inhibitory, inhibitory)    
    I_I = CM[locs]
    nz = np.nonzero(I_I)
    permuted = rng.choice(I_I[nz],nz[0].size, replace = False)
    CM[locs][nz] = permuted
    locs = np.ix_(excitatory, inhibitory)
    E_I = CM[locs]
    nz = np.nonzero(E_I)
    permuted = rng.choice(E_I[nz],nz[0].size, replace = False)
    CM[locs][nz] = permuted
    locs = np.ix_(inhibitory, excitatory)
    I_E = CM[locs]
    nz = np.nonzero(I_E)
    permuted = rng.choice(I_E[nz],nz[0].size, replace = False)
    CM[locs][nz] = permuted
    return CM
    

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience and metrics > 90:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def BinaryConnectedNeighbors(nOsc,k,p,Dists,Neighborhood = "Target",reverse=False):
    ConnMatrix = torch.zeros((nOsc,nOsc))
    prob = torch.ones(k)*p
    TargetNeighbors, SourceNeighbors = Parameters.NodeNeighbors(Dists,reverse = reverse)
    probs=torch.bernoulli(prob)
    #This distinction is only meaningful if Neighborhood is based on Density or
    #directional Connections
    #It should not make a difference for Distance or spatial Neighborhoods
    
    if Neighborhood == "Target":
        Neighbors = torch.tensor(TargetNeighbors)
    else:
        Neighbors = torch.tensor(SourceNeighbors)
    
    for r in range(nOsc):
        probs=torch.bernoulli(prob)
        if not reverse:
            RegionNeighborhood = Neighbors[r][1:(k+1)].long()
            RNeighbors =  Neighbors[r][1:]
        else:
            RegionNeighborhood = Neighbors[r][:k].long()
            RNeighbors =  Neighbors[r][:-1]
        
        keep = torch.where(probs ==0)[0]
        rewire = torch.where(probs !=0)[0]
        for kr in RegionNeighborhood[keep]:
            RNeighbors = RNeighbors[RNeighbors !=kr]
        #wrewire = RNeighbors[torch.randint(0,len(RNeighbors),(len(rewire),))]
        wrewire = RNeighbors[np.random.choice(len(RNeighbors),len(rewire),replace=False)]    
        if Neighborhood == "Target":
            ConnMatrix[r,RegionNeighborhood[keep]] = 1
            ConnMatrix[r,wrewire] = 1
        elif Neighborhood == "Source":
            ConnMatrix[RegionNeighborhood[keep],r] = 1 
            ConnMatrix[r,wrewire] = 1
        else:#Both
            ConnMatrix[r,RegionNeighborhood[keep]] = 1
            ConnMatrix[r,wrewire] = 1
            ConnMatrix[RegionNeighborhood[keep],r] = 1 
            ConnMatrix[r,wrewire] = 1
    return ConnMatrix




#this sets the sum of each row to be the same as the as_mat 
def Sum_normalization_as(mat,as_mat,by='column'):
    
    if by == 'column':
        mat = mat.T
        as_mat = as_mat.T 
    for r in range(mat.shape[0]):
        rowsum = np.sum(mat[r,:])
        if rowsum !=0:
            norm = np.sum(mat[r,:])/np.sum(as_mat[r,:])
            #print(norm)
            mat[r,:] = np.divide(mat[r,:],norm)
        
    #mat[np.where(np.isnan(mat)==True)] = 0 
    #mat[np.where(np.isinf(mat)==True)] = 0
    if by == 'column':
        mat = mat.T
        
    return mat

def get_ConnMatrix(self,Conn = "Density"):
    '''
    Gets connectivity from Mouse Regionalized-model. Requires nhiddden = 198

    Parameters
    ----------
    Conn : str, optional
        The default is "Density".

    Returns
    -------
    np.array
        Returns requested Mouse Connectivity values.

    '''
    #df = {'Strength':strength, 'Density':density, 'NormalizedStrength':Nstrength,'Dists':Dists,'Delays':Delays,'names':Tnames}
    f = open('Cortex_Thal_Cla_HippN198.sav', 'rb')
    df = pickle.load(f)
    f.close()
    
    if Conn == 'Density':
        return df['Density']
    elif Conn == 'Strength':
        return df['Strength']
    elif Conn == 'NormalizedStrength':
        return df['NormalizedStrength']
    elif Conn == 'Dists':
        return df['Dists'] #in mm
    elif Conn == 'Delays':
        return df['Delays'] # based on dt of 1/1000
    elif Conn == 'Names':
        return df['names']
        
class RNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out,OutputDir, batch_size,input_bias,noise,device,nonlinearity='tanh'):
        super(RNN, self).__init__()
        self.n_hid = n_hid
        self.batch_size = batch_size
        self.cell = nn.RNNCell(n_inp,n_hid)
        self.readout = nn.Linear(n_hid, n_out)
        self.device = device
        self.OutputDir = OutputDir
        self.InitialState = self.cell.weight_hh.data
        self.noise = noise
        
    def initialize(self,mat):
       self.cell.weight_hh.data = mat 
       
       self.Connections = torch.zeros(self.cell.weight_hh.shape)
       self.Connections[mat!=0] = 1
       InitialParameters = list(self.named_parameters())
       InitialState = []
       for param in InitialParameters:
            InitialState.append((param[0],param[1].detach().clone()))
       self.InitialState = InitialState
    
    def Reinitialize(self,model):
        ModelParams = list(self.named_parameters())
        with torch.no_grad():
            for MParam in ModelParams:
                savedval = [param for param in model if param[0] == MParam[0]]
                MParam[1].copy_(savedval[0][1])
       
    def PermuteInputOutput(self):
        Wih  = self.cell.weight_ih.data.clone()
        Wr = self.readout.weight.data.clone()
        permute = self.rng.choice(Wih.flatten(), Wih.numel(), replace = False).reshape(Wih.shape)
        self.cell.weight_ih.data = torch.from_numpy(permute).type(torch.float32)
        permute = self.rng.choice(Wr.flatten(), Wr.numel(), replace = False).reshape(Wr.shape)
        self.readout.weight.data = torch.from_numpy(permute).type(torch.float32)
        
    def save(self):
        pass
    
    def plot(self,Outfile):
        pass
    
    def Clamp_Grads(self):
        self.cell.weight_hh.grad[self.Connections ==0] = 0
        
    def Clamp_Weights(self):
        self.cell.weight_hh.data.clamp_(0)
        
    def forward(self, x, initial_state = None,stateOnly=None):
        if initial_state is None:
            state = torch.zeros(x.size(1),self.n_hid).to(self.device)
        else:
            state = initial_state.to(self.device)
            
        self.state = torch.zeros((self.batch_size,self.n_hid,x.shape[0]))
        
        if self.noise:
            x = x + (torch.normal(mean = x,std = 0.1)**2)
            
        for t in range(x.size(0)):
            state = self.cell(x[t],state)
            self.state[:,:,t] = state.clone()
        
        output = self.readout(state)
        if stateOnly:
            return state
        else:
            return output
    
    def forward_special_Jacobian_single_step(self, params):
        x = params[0]
        state = params[1]
        state = self.cell(x,state)
        
        return state




class psRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid):
        super(psRNNCell, self).__init__()
        self.Wi = nn.Linear(n_inp, n_hid, bias=True)
        self.Wh = nn.Linear(n_hid, n_hid, bias=False)
        self.A = nn.Parameter(torch.rand(n_hid,requires_grad=True))
        #self.omega = nn.Parameter(torch.zeros(n_hid).uniform_(-1*pi2,pi2))
        self.omega = nn.Parameter(torch.rand(n_hid,requires_grad=True))
        
    def Clamp_Grads(self):
        self.Wh.weight.grad[self.Connections ==0] = 0
        
    def initialize(self,mat):
        self.Wh.weight.data = mat
        self.Connections = torch.zeros(self.Wh.weight.shape)
        self.Connections[mat!=0] = 1
        
    #ToDo reset zero elements during training
    '''        
    def setWeights(self):
        with torch.no_grad():
            self.Wh.weight[] 
            '''       
       
    def forward(self,x,state):
        state = self.A*torch.cos(self.omega*state + self.Wi(x)) + self.Wh(state)
        
        return state
   
 
class psRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, OutputDir,batch_size,device):
        super(psRNN, self).__init__()
        self.batch_size = batch_size
        self.n_hid = n_hid
        self.cell = psRNNCell(n_inp,n_hid)
        self.readout = nn.Linear(n_hid, n_out)
        self.device = device
        self.Order = []
        self.OutputDir = OutputDir
        
        
        
    def Reinitialize(self,model):
        ModelParams = list(self.named_parameters())
        with torch.no_grad():
            for MParam in ModelParams:
                savedval = [param for param in model if param[0] == MParam[0]]
                MParam[1].copy_(savedval[0][1])
       
    
    def Clamp_Grads(self):
        self.cell.Clamp_Weights()
        
    def initialize(self,mat):
        self.cell.initialize(mat)
        InitialParameters = list(self.named_parameters())
        InitialState = []
        for param in InitialParameters:
            InitialState.append((param[0],param[1].detach().clone()))
        self.InitialState = InitialState
        
         
    def plot(self,OutFile):
        epochs = np.arange(len(self.Order))
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(epochs,np.array(self.Order))
        plt.title("Order Over Training")
        plt.xlabel("Epochs")
        plt.subplot(2,1,2)
        plt.plot(epochs[1:],np.diff(np.array(self.Order)))
        plt.title("dO/dE Training")
        plt.xlabel("Epochs")
        plt.tight_layout() 
        plt.savefig(os.path.join(self.OutputDir,OutFile + "_OrdeParam.png"),transparent=False,dpi =300) 
    
    def save(self):
        self.ComputeOrder()
        return {'Order':self.order}
        
    def forward(self, x):
        state = torch.ones(x.size(1),self.n_hid).to(self.device)
        self.state = torch.zeros((self.batch_size,self.n_hid,x.shape[0]))
        for t in range(x.size(0)):
            state = self.cell(x[t],state)
            self.state[:,:,t] = state.clone()
        output = self.readout(state)
        
        return output#self.prob(output)

    def ComputeOrder(self):
        with torch.no_grad():
            phasediff = self.state.unsqueeze(2) - self.state.unsqueeze(1)
            self.order = (1/(self.n_hid**2))*torch.sum(torch.mean(torch.cos(phasediff),3),(1,2))
            self.Order.append(self.order.detach().numpy())
 

class KuramotoCell(nn.Module):
    def __init__(self, n_inp, n_hid,rng,input_bias,freq):
        super(KuramotoCell, self).__init__()
        if input_bias:
            self.Wi = nn.Linear(n_inp, n_hid, bias=True )
        else:
            self.Wi = nn.Linear(n_inp, n_hid, bias=False )
        self.Wh = nn.Parameter(torch.from_numpy(rng.random(size=(n_hid, n_hid))).type(torch.float32))
        if n_inp != 28:
            factor = 2*28./n_inp
        else:
            factor = 1
        self.dt = 1#*(n_inp/28.)
        
        self.omega = nn.Parameter(torch.from_numpy(freq).type(torch.float32))
        print(self.omega)
        
   
    
    def forward(self,x,state):
        #state = torch.sum(self.Wh*torch.sin((state.unsqueeze(2) - state.unsqueeze(1)) + self.Wi(x).unsqueeze(2)),2) + self.omega   + state
        
        state = torch.sum(self.Wh*torch.sin(state.unsqueeze(2) - state.unsqueeze(1)),2) + self.Wi(x) + self.omega + state
        return torch.remainder(state,2*np.pi) 

    def Clamp_Grads(self):
        self.Wh.grad[self.Connections ==0] = 0

    def Clamp_Weights(self):
        self.Wh.data.clamp_(0)
        
    def initialize(self,mat):
        self.Wh.data = mat
        self.Wi.weight.data = self.Wi.weight.data/mat.shape[0]
        self.Connections = torch.zeros(self.Wh.shape)
        self.Connections[mat!=0] = 1
        '''
        InitialParameters = list(self.named_parameters(prefix='cell'))
        InitialState = []
        for param in InitialParameters:
            InitialState.append((param[0],param[1].detach()))
        self.InitialState = InitialState
        ''' 
        
class Kuramoto(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, OutputDir,batch_size,rng,input_bias,freq,device):
        super(Kuramoto, self).__init__()
        self.batch_size = batch_size
        self.n_hid = n_hid
        self.cell = KuramotoCell(n_inp,n_hid,rng,input_bias,freq)
        self.readout = nn.Linear(n_hid, n_out)
        self.device = device
        self.Order = []
        self.OutputDir = OutputDir
        self.state = torch.zeros((self.batch_size,self.n_hid,self.n_hid))
        
    def Clamp_Weights(self):
        self.cell.Clamp_Weights()
    
    def Clamp_Grads(self):
        self.cell.Clamp_Grads()    

    def initialize(self,mat):
       self.cell.initialize(mat)
       InitialParameters = list(self.named_parameters())
       InitialState = []
       for param in InitialParameters:
            InitialState.append((param[0],param[1].detach().clone()))
       self.InitialState = InitialState
       
       
    def Reinitialize(self,model):
        ModelParams = list(self.named_parameters())
        with torch.no_grad():
            for MParam in ModelParams:
                savedval = [param for param in model if param[0] == MParam[0]]
                if len(savedval) >0:
                    MParam[1].copy_(savedval[0][1])
                    print("reinitialized ",savedval[0][0])
                else:
                    print('no saved value for ',MParam[0])
       
      
    
    def plot(self,OutFile):
        epochs = np.arange(len(self.Order))
        
        if epochs.size > 1:
            if len(self.Order[-1]) > 1:
                Order = np.mean(np.array(self.Order),1)
            else:
                Order = np.array(self.Order).flatten()
            plt.clf()
            plt.subplot(2,1,1)
            plt.plot(epochs,Order)
            plt.title("Order Over Training")
            plt.xlabel("Epochs")
            plt.subplot(2,1,2)
            plt.plot(epochs[1:],np.diff(np.array(Order)))
            plt.title("dO/dE Training")
            plt.xlabel("Epochs")
            plt.tight_layout() 
            plt.savefig(os.path.join(self.OutputDir,OutFile + "_OrdeParam.png"),transparent=False,dpi =300) 
          
    def save(self):
       self.ComputeOrder()
       return {'Order':self.order}
   
      
    def forward(self, x,random_initial_state = False):
       '''
        Forward pass

        Parameters
        ----------
        x : torch tensor
            Model input
        random_initial_state : boolean, optional
            Initialize network state from uniform distribution instead of all zeros. The default is False.

        Returns
        -------
        output : torch tensor
            Network predictions.

       '''
      
       if random_initial_state:
           state = torch.zeros(self.batch_size,self.n_hid).uniform_(0,np.pi*2).to(self.device)
       else:
           state = torch.zeros(self.batch_size,self.n_hid)
       
       
       
       self.state = torch.zeros((self.batch_size,self.n_hid,x.shape[0]))
       for t in range(x.size(0)):
           state = self.cell(x[t],state)
           self.state[:,:,t] = state.clone()
       output = self.readout(state)
       
       return output
    
    def ComputeOrder(self):
        '''
        Computes Universal and Kuramoto Order Parameter for Kuramoto Networks

        Returns
        -------
        universal_order : float
           

        '''
        with torch.no_grad():
            '''
            for o in torch.range(self.n_hid):
                self.state[:,o,:] - self.state[:,self.connections[o]]
            '''    
            connections = self.cell.Connections.unsqueeze(0).unsqueeze(3)
            phasediff = self.state.unsqueeze(2) - self.state.unsqueeze(1)
            
            #universal order
            #k = torch.sum(self.cell.Wh)
            k = torch.sum(self.cell.Connections)
            
            #universal_order = torch.sum(self.cell.Wh*torch.mean(torch.cos(phasediff),3),(1,2))
            universal_order  = torch.sum(self.cell.Connections*torch.mean(torch.cos(phasediff),3),(1,2))
            if k !=0:
                universal_order = universal_order/k
           
            '''
            i = torch.sqrt(torch.zeros(1,dtype = torch.cfloat)-1)
            universal_order = torch.sum(self.cell.Wh*torch.mean(torch.real(torch.exp(phasediff *i)),3),(1,2))
            if k !=0:
                universal_order = universal_order/k
            '''
            #kuramoto order
            i = torch.sqrt(torch.zeros(1,dtype = torch.cfloat)-1)
            #self.order = torch.mean(torch.mean(torch.exp(i*self.state),1).abs(),1)
            
            #self.order = torch.sqrt(torch.mean(torch.mean(torch.exp(i*self.state),1).abs(),1))
            #self.order = torch.sqrt(torch.mean( 1/(self.n_hid*2)*torch.sum(torch.exp(i*phasediff),(1,2)).abs(),1))
            
            #self.order = 1/self.n_hid*torch.sum(torch.mean(torch.exp(i*self.state).abs(),2))
            #self.order = torch.mean(torch.mean(torch.cos(phasediff),(1,2)),1)
            #self.order = torch.sum(torch.mean(torch.real(torch.exp(self.state*i)).abs(),2),1)/self.n_hid
            self.order  = torch.mean(torch.mean(torch.cos(phasediff),(1,2)),1)
            self.Order.append(self.order.detach().numpy())
            return universal_order
        
 

class Net(nn.Module):
    #Main network class
    def __init__(self,ModelType,ninputs, nhidden,batch_size,input_bias=True,posW = False,digits = None,noise=False,device=device,nonlinearity = 'tanh',pixel_by_pixel= False,run = None):
        '''
        Params
        
        ModelType: str
            Can be one of 'RNN' 'Kuramoto' 'psRNN' 
        
        ninputs: int
            number of inputs to the network, generally 28
        
        nhidden: int
            number of hidden units 198 matches number of regions in mouse
                 meso-scale model
        
        batch_size: int
            number of MNIST images per forward pass

        input_bias: boolean
            if True include bias term in network
            
        posW: boolean
            if True restrict model hidden weights to be positive 
        
        '''
        
        
        super(Net,self).__init__()
        self.OutputDir = os.path.join(ModelType)
        if digits is not None:
            self.OutputDir = os.path.join(self.OutputDir,"digits")
        if pixel_by_pixel or ninputs < 28:
            self.OutputDir = os.path.join(self.OutputDir, "pixel_by_pixel")
        if nhidden < 109:
            self.OutputDir = os.path.join(self.OutputDir, "narrow")
        
        if noise:
            self.noise = True
        else:
            self.noise = False
        self.ModelType = ModelType
        self.batch_size= batch_size
        self.ninputs = ninputs
        self.nhidden = nhidden
        self.posW = posW
        if run is not None:
            seed = 8 + run
        else:
            seed = 8
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        
        if ModelType =="RNN":
            self.model = RNN(ninputs,nhidden,nout,self.OutputDir,batch_size,input_bias,noise,device,nonlinearity)
        elif ModelType=="ps":
            self.model = psRNN(ninputs,nhidden,nout,self.OutputDir,batch_size,self.rng,input_bias,device)
        elif ModelType=="Kuramoto":
            self.model = Kuramoto(ninputs,nhidden,nout,self.OutputDir,batch_size,self.rng,input_bias,device)
        elif ModelType == "GLIFR":
            self.model = BNNFC(ninputs, nhidden, nout, batch_size,num_ascs = 2,hetinit = True, ascs= True, \
                        learnparams = True, output_weight = True,dropout_prob = 0,delay = 1, device = device)
          
        self.config = {"ModelType":ModelType,\
                       "ninputs":ninputs,\
                           "nhidden":nhidden,\
                           "batch_size":batch_size}
        
        self.dl  = sMNIST(batch_size,digits = digits)
        self.TrainingLoss = []
        self.ValidationLoss = []
        self.TrainingAccuracy = []
        self.ValidationAccuracy = []
        if not os.path.exists(self.OutputDir):
            os.makedirs(self.OutputDir)
    
        
    def plot(self):
        '''
        plot incremental training and validation loss

        Returns
        -------
        None.

        '''
        
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(np.array(self.TrainingLoss))
        plt.title("Training Loss")
        plt.xlabel("Batches")
        plt.subplot(2,1,2)
        plt.plot(np.array(self.TrainingAccuracy))
        plt.title("Training Accuracy %")
        plt.xlabel("Batches")
        plt.tight_layout() 
        plt.savefig(os.path.join(self.OutputDir,self.OutFile + "_TrainingLoss.png"),transparent=False,dpi =300)
        
        if len(self.ValidationLoss) > 0 :
            plt.clf()
            plt.subplot(2,1,1)
            plt.plot(np.array(self.ValidationLoss))
            plt.title("Validation Loss")
            plt.xlabel("Epochs")
            plt.subplot(2,1,2)
            plt.plot(np.array(self.ValidationAccuracy))
            plt.title("Validation Accuracy %")
            plt.xlabel("Epochs")
            plt.tight_layout() 
            plt.savefig(os.path.join(self.OutputDir,self.OutFile + "_ValidationLoss.png"),transparent=False,dpi =300)
            
        self.model.plot(self.OutFile)
        
    
    def Reinitialize(self,file,initial=False,device='cpu'):
        '''
        Reinitialize model parameters from saved model 

        Parameters
        ----------
        file : str
            file name from which to reload params
        initial : boolean
            if True reinitialize initial state before training instead of final state
        Returns
        -------
        None.

        '''
        saved = torch.load(file,map_location=device)
        self.seed = saved['Seed']
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if initial == False:
            self.model.Reinitialize(saved['Parameters'])
        else:
            self.model.Reinitialize(saved['InitialState'])
        
        
        
    def initialize(self,ConnType,nNN = None, p = .1,reverse=False,Neighborhood = None,suffix="",gain = 1.0):
        '''
        Initialize network model
        Must be run before training and before Re-initializing 

        Parameters
        ----------
        ConnType : str
            DESCRIPTION.
        nNN : int
            number of nearest neighbors defaults to ninputs
        p : float
            probability of rewiring edges. The default is .1.
        reverse : boolean, optional - currently not used
            For Connection-type based on anatomical distance from mouse model can reverse sort distances. The default is False.
        Neighborhood : float, optional -- currently not used
            If Connection-type based on distances, can restrict to regions within Neighborhood distance. The default is None.
        suffix : str, optional
            Descriptor for training run. suffix is appended to the output file name. The default is "".

        Returns
        -------
        None.

        '''
        self.ConnType = {"ConnType":ConnType, 'nNN':nNN,'p':p,'Neighbrhood':Neighborhood}
        self.OutFile = "_".join((ConnType,str(self.nhidden),"ninputs",str(self.ninputs)))
        self.gain = gain
        if ConnType=='Random': #Erdos Remyi
            if nNN is None:
                nNN = self.ninputs
            CM = 1 - np.eye(self.nhidden)
            nz = np.nonzero(CM.flatten())[0]
            std = self.gain*1/np.sqrt(np.sqrt(len(nz)))
            CM = np.zeros(CM.shape).flatten()
            Num = int(p*len(nz))
            idxs = self.rng.choice(nz,size = Num,replace=False)
            vals = self.rng.uniform(-std,std,len(nz))
            CM[idxs] = vals
            CM = torch.from_numpy(CM.reshape((self.nhidden,self.nhidden))).type(torch.float32)
            self.OutFile = self.OutFile + "_".join(("nNN"+str(nNN),"p",str(p)))
        elif ConnType == 'WattsStrogatz':
            CM = nx.watts_strogatz_graph(self.nhidden,nNN,p,seed = self.seed)
            CM = nx.to_numpy_matrix(CM).flatten()
            nz = np.nonzero(CM)
            std = self.gain*1/np.sqrt(np.sqrt(len(nz[0])))
            #vals = self.rng.uniform(-std,std,len(nz[0]))
            vals = self.rng.normal(0,std,len(nz[0]))
            CM[nz] = vals
            CM = torch.from_numpy(CM.reshape((self.nhidden,self.nhidden))).type(torch.float32)
            self.OutputDir = os.path.join(self.OutputDir, "Gaussian")
            if not os.path.exists(self.OutputDir):
                os.makedirs(self.OutputDir)
            if self.noise:
                self.OutputDir = os.path.join(self.OutputDir, "noise")
            self.OutFile = self.OutFile + "_".join(("nNN"+str(nNN),"p",str(p),"gain",str(gain)))
            
        elif ConnType == 'Density':
            f = open('../MesoScope/Cortex_Thal_Cla_HippN198.sav', 'rb')
            df = pickle.load(f)
            f.close()
            CM= df['Density']
            if 'thresh' in suffix:
               weights = CM.flatten() 
               order = np.argsort(weights)
               if "to" in suffix:
                   thresh_to = int(suffix.split("to")[1].split("_")[0])
                   thresh_to = (198*198) -  thresh_to  #matches sparsity of nNN 128
               else:
                   thresh_to = (198*198)  -  25344  #matches sparsity of nNN 128
               weights[order[:thresh_to]] = 0.0
               nz = np.nonzero(weights)[0].size
               CM =  weights.reshape((self.nhidden, self.nhidden))
            elif 'sparsify' in suffix:
                weights = CM.flatten() 
                nz = np.nonzero(weights)[0]
                sparsify = self.rng.choice(nz,len(nz)-25344,replace = False)
                weights[sparsify] =0.0
                nz = np.nonzero(weights)[0].size
                CM =  weights.reshape((self.nhidden, self.nhidden))
            else:
                nz = 198*198 - 198
            std = 1/np.sqrt(np.sqrt(nz))
            nz = np.nonzero(CM)
            sign =[-1,1]
            signs = self.rng.choice(sign,len(nz[0]),replace=True)
            vals = CM[nz].flatten()*signs
            CM[nz] = vals
            CM[nz] = (CM[nz] - np.mean(CM[nz]))/np.std(CM[nz])
            CM = self.gain*std*CM
            CM = torch.from_numpy(CM).type(torch.float32)
            self.OutputDir = os.path.join(self.OutputDir, "Density")
            self.OutFile = self.OutFile + "_Density_gain_"+str(gain)
        elif ConnType =="microns":
            connectome_dir = "/allen/programs/mindscope/workgroups/tiny-blue-dot/EM/microns"
            CM = np.load(os.path.join(connectome_dir,"full_ground_truth_summed_weights.npy"))
            if 'sampled' in suffix:
                #column_idxs = np.load(os.path.join(connectome_dir,"column_indexes_within_full_square_array.npy"))
                column_idxs = np.load(os.path.join(connectome_dir,"column_indexes_in_sorted_full.npy"))
                other_idxs = [i for i in np.arange(198) if i not in column_idxs]
                num_sample = 198-len(column_idxs)
                sample = self.rng.choice(other_idxs,num_sample, replace=False)
                idxs = np.append(column_idxs,sample)
            else:    
                column_idxs = np.load(os.path.join(connectome_dir,"column_indexes_198_in_sorted_full.npy"))
                idxs = column_idxs
            CM = CM[:,idxs]
            CM = CM[idxs,:]
            CM = CM/np.max(np.abs(CM))
            nz = np.nonzero(CM)
            if 'block_permuted' in suffix:
                CM = block_permutation(CM)
            elif 'permuted' in suffix:
                weights = CM.copy()
                nzw = np.nonzero(weights)
                shuffled_weights = self.rng.choice(weights[nzw].flatten(), nzw[0].size, replace = False)
                di = np.diag_indices(self.nhidden)
                di = np.ravel_multi_index(di, (self.nhidden, self.nhidden))
                od = [o for o in np.arange(self.nhidden*self.nhidden) if o not in di]
                locations = self.rng.choice(od,nzw[0].size, replace = False)
                CM = np.zeros_like(CM)
                locations = np.unravel_index(locations, (self.nhidden,self.nhidden))
                CM[locations] = shuffled_weights
            if 'norm_198' in suffix:
                WS_fc_nz = 198*198 - 198
                std = 1/np.sqrt(np.sqrt(WS_fc_nz))
            else:    
                std=1/np.sqrt(np.sqrt(len(nz[0])))
            CM[nz] = (CM[nz] - np.mean(CM[nz]))/np.std(CM[nz])
            CM[nz] = self.gain*CM[nz]*std 
            CM = torch.from_numpy(CM).type(torch.float32)
            self.OutputDir = os.path.join(self.OutputDir, "EM_column","microns")
            self.OutFile = self.OutFile + "_gain_"+str(gain)            
        elif ConnType == "v1dd":
            if 'zuul' in hostname:
                connectome_dir = "/home/dana/allen/programs/mindscope/workgroups/tiny-blue-dot/EM/v1dd"
            else:
                connectome_dir = "/allen/programs/mindscope/workgroups/tiny-blue-dot/EM/v1dd"
            if "23_4" in suffix:
                CM = np.load(os.path.join(connectome_dir,"23_4.npy"))
                CM = CM/np.max(np.abs(CM))
            elif "23" in suffix:
                CM = np.load(os.path.join(connectome_dir,"23.npy"))
                CM = CM/np.max(np.abs(CM))
            elif "4_" in suffix:
                CM = np.load(os.path.join(connectome_dir,"4.npy"))
                CM = CM/np.max(np.abs(CM))    
            if 'block_permuted' in suffix:
                CM = block_permutation(CM)
            elif 'permuted' in suffix:
                weights = CM.copy()
                nzw = np.nonzero(weights)
                shuffled_weights = self.rng.choice(weights[nzw].flatten(), nzw[0].size, replace = False)
                di = np.diag_indices(self.nhidden)
                di = np.ravel_multi_index(di, (self.nhidden, self.nhidden))
                od = [o for o in np.arange(self.nhidden*self.nhidden) if o not in di]
                locations = self.rng.choice(od,nzw[0].size, replace = False)
                CM = np.zeros_like(CM)
                locations = np.unravel_index(locations, (self.nhidden,self.nhidden))
                CM[locations] = shuffled_weights
            elif 'normed' in suffix:
                nz = np.nonzero(CM)
                std = 1/np.sqrt(np.sqrt(nz[0].size))
                CM[nz]  = (CM[nz] - np.mean(CM[nz]))/np.std(CM[nz])
                CM = CM*std
            elif 'transformed' in suffix:
                nz = np.nonzero(CM)
                std  = np.std(CM[nz])
                transformed = CM.copy()
                shift = np.abs(np.min(transformed)) 
                shifted = transformed + shift + 1
                CMt, _ = stats.boxcox(shifted[nz].flatten())
                CMt = CMt -shift
                CMt = CMt/np.max(CMt)
                transformed[nz] = CMt
                transformed[nz] = (transformed[nz] - np.mean(transformed[nz]))/np.std(transformed[nz])
                transformed = transformed*std
                CM[nz] = transformed[nz]
            CM = torch.from_numpy(CM*self.gain).type(torch.float32)
            self.OutputDir = os.path.join(self.OutputDir, "EM_column","v1dd")  
            self.OutFile = self.OutFile + "_gain_"+str(gain)  
        elif ConnType == 'NearestNeighbors' :
            self.OutFile = self.OutFile + "_".join(("Neighborhood"+Neighborhood))
            
            '''
            Dists = get_ConnMatrix(Conn = ConnParams[3])
            if ConnParams[3] == 'Dists':
                reverse = False
            else:
                reverse= True
            Neighborhood = ConnParams[4]
            BinaryConnMatrix = BinaryConnectedNeighbors(nOsc,nNN,p,Dists,Neighborhood = Neighborhood,reverse=reverse)
            density = get_ConnMatrix()
            ConnMatrix = np.multiply(BinaryConnMatrix.numpy(),density)
            ConnMatrix = torch.tensor(Sum_normalization_as(ConnMatrix,density,by='row'),dConnType = torch.float64)
            '''
        self.ConnType.update({'CM':CM})
        self.model.initialize(CM)
        
    
    def save(self,gradients = None):
        '''
        Saves model parameters

        Returns
        -------
        None.

        '''
        params = list(self.model.named_parameters())
        out = {"Parameters":params,"TrainingLoss":self.TrainingLoss,"ValidationLoss":self.ValidationLoss,\
               "config":self.config,'ConnType':self.ConnType,'TrainingAccuracy':self.TrainingAccuracy,\
                   'ValidationAccuracy':self.ValidationAccuracy,'InitialState':self.model.InitialState,'Seed':self.seed,'Gain':self.gain}
        if gradients is not None:
            out['gradients'] = gradients
        modelOut = self.model.save()
        if modelOut is not None: 
            out.update(modelOut)
        torch.save(out,os.path.join(self.OutputDir,self.OutFile))
    
    def test(self,pixel_by_pixel=False):
       '''
        Model Validation

        Returns
        -------
        None.

       '''
       celoss = nn.CrossEntropyLoss()
       valacc = []
       with torch.no_grad():
           for images,labels in self.dl.test_loader:
               if not pixel_by_pixel:
                   images = images.reshape(self.batch_size, self.ninputs, int(images.shape[1]/self.ninputs))
                   images = images.permute(2,0,1)
               else:
                   images = images.permute(1,0,2)
               images, labels = images.to(device),labels.to(device)
               output = self.model(images)
               loss = celoss(output, labels)
               self.ValidationLoss.append(loss.cpu().numpy())
               pred = torch.argmax(output, dim=1)
               correct_digit = pred.eq(labels)
               accuracy = 100.*torch.sum(correct_digit)/len(labels)
               valacc.append(accuracy)
               self.ValidationAccuracy.append(accuracy.cpu().numpy())
       return torch.mean(torch.stack(valacc))
    
    def train(self,save_gradients = False, pixel_by_pixel= False,lr=0.00001):
        '''
        Model Training Function

        Returns
        -------
        None.

        '''
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        celoss = nn.CrossEntropyLoss()
        if save_gradients:
            gradients = []
            sample = list(self.rng.choice(len(self.dl.train_loader),100,replace= False))
        else:
            gradients = None
        
        for i, (images, labels) in enumerate(self.dl.train_loader):
            optimizer.zero_grad()
            if not pixel_by_pixel:
                images = images.reshape(self.batch_size, self.ninputs, int(images.shape[1]/self.ninputs))
                images = images.permute(2,0,1).to(device)
            else:
                images = images.permute(1,0,2).to(device)
            labels = labels.to(device)
            output = self.model(images)
            loss = celoss(output, labels)
            loss.backward()
            #params = list(self.model.parameters())
            #for p in params:
            #    print(p.grad)
            self.model.Clamp_Grads()
            if self.posW:
                self.model.Clamp_Weights()
            optimizer.step()
            if save_gradients:
                if i in sample:
                    grads = {}
                    for param in net.named_parameters():
                        if param[1].grad is not None:
                            grads[param[0]] = param[1].grad.cpu().detach().numpy()
                    gradients.append(grads) 
            self.TrainingLoss.append(loss.detach().cpu().numpy())
            pred = torch.argmax(output, dim=1)
            correct_digit = pred.eq(labels)
            accuracy = 100.*torch.sum(correct_digit)/len(labels)
            #print(accuracy)
            self.TrainingAccuracy.append(accuracy.cpu().numpy())
        valacc = self.test(pixel_by_pixel = pixel_by_pixel)
        self.plot()
        self.save(gradients = gradients)
        return valacc

        

class sMNIST():
    def __init__(self,batch_size,digits = None):
        '''
        Sequential MNIST data loader 

        Parameters
        ----------
        batch_size : int
            creates training and testing arrays of size batch_size

        Returns
        -------
        None.

        '''
        
        #Returns a Dataset
        
        
        transform = torchvision.transforms.Compose(
             [torchvision.transforms.ToTensor(),
              torchvision.transforms.Lambda(lambda x: x.view(-1,1))
             ])
        train_dataset = torchvision.datasets.MNIST(root='MNIST/',
                                               train=True,
                                               transform=transform,
                                               download=True)
        

        test_dataset = torchvision.datasets.MNIST(root='MNIST/',
                                              train=False,
                                              transform=transform)
        
        
        if digits is not None:
            train_indices = [i for i,target in enumerate(train_dataset.targets) if target in digits ]
            train_dataset.data, train_dataset.targets = train_dataset.data[train_indices], train_dataset.targets[train_indices]
            test_indices = [i for i,target in enumerate(test_dataset.targets) if target in digits ]
            test_dataset.data, test_dataset.targets = test_dataset.data[test_indices], test_dataset.targets[test_indices]
            ntrain = len(train_dataset.targets)
            ntest = len(test_dataset.targets)
            '''
            train_mod = ntrain % batch_size
            test_mod = ntest % batch_size
            if train_mod !=0:
                train_dataset.data,train_dataset.targets = train_dataset.data[:-train_mod,...],train_dataset.targets[:-train_mod,...]
            if test_mod !=0:
                test_dataset.data,test_dataset.targets = test_dataset.data[:-test_mod,...],test_dataset.targets[:-test_mod,...]
            '''
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,drop_last = True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,drop_last = True)

class psMNIST():
    def __init__(self):

        pixel_permutation = torch.randperm(28*28)
        transform = torchvision.transforms.Compose(
             [torchvision.transforms.ToTensor(),
              torchvision.transforms.Lambda(lambda x: x.view(-1,1)[pixel_permutation])
             ])
        
        

#python Network.py --ConnType Random --ModelType Kuramoto --nhidden 10 --ninputs 28 --nlayers 1 --batch_size 5 --nNN 10 --p .1

if __name__ == "__main__":
    
   
    suffix = ""
    #required
    parser = argparse.ArgumentParser()
    parser.add_argument("--ConnType",type = str, required = True)
    parser.add_argument('--ModelType',type = str, required = True)
    parser.add_argument('--nhidden',type = int, required = True)
    parser.add_argument('--ninputs',type = int, required = True)
    parser.add_argument('--nlayers',type = int, required = True)
    parser.add_argument('--batch_size',type = int, required = True)
    
    #optional
    parser.add_argument('--nepochs',type = str, required = False)
    parser.add_argument('--nNN',type = int, required = False)
    parser.add_argument('--p',type = float, required = False)
    parser.add_argument('--reverse',type = bool, required = False)
    parser.add_argument('--Neighborhood',type = str, required = False) #Can be Target or Source
    parser.add_argument('--suffix',type = str,required = False)#Add more description to OutFile
    parser.add_argument('--input_bias',type=bool, required = False)
    parser.add_argument('--positive_weights_only',type=bool, required = False)
    parser.add_argument('--init_gain',type = float, required = False)
    parser.add_argument('--add_noise',required=False,action='store_true')
    parser.add_argument("--save_gradients",required=False, action ='store_true')
    parser.add_argument("--pixel_by_pixel",required = False, action ="store_true")
    parser.add_argument("--lr", type=float,required= False)
    parser.add_argument("--nonlinearity", type=str,required= False)
    parser.add_argument("--digits",type=int,nargs='+',action = "extend",default = None,required = False)
    parser.add_argument("--reload",type= str, required = False)
    parser.add_argument("--reload_initial",type= str, required = False)
    parser.add_argument("--transform",type= str, required = False)# log-normal, power-law, gaussian etc
    parser.add_argument("--permuteIO", required = False, action = 'store_true')
    
    args = parser.parse_args()
    if args.nNN is not None:
        nNN = args.nNN
    
    if args.p is not None:
        p = args.p
        
    
    if args.input_bias is not None:
        input_bias = args.input_bias
    else:
        input_bias = True
    if args.positive_weights_only is not None:
        posW = args.positive_weights_only
    else:
        posW = False
     
    if args.init_gain is not None:
        init_gain = args.init_gain
        assert args.ModelType == 'RNN' or args.ModelType == "GLIFR"
    else:
        init_gain = 1.0
    
    if args.permuteIO is not None:
        permuteIO  = args.permuteIO
    else:
        permuteIO = False
    
    print("permuteIO",permuteIO)
        
    if args.add_noise is not None:
        noise = args.add_noise
        if noise:
            assert args.ModelType == 'RNN'
    else:
        noise = False
       
    if args.pixel_by_pixel is not None:
        pixel_by_pixel = args.pixel_by_pixel
    else:
        pixel_by_pixel = False
    
    if args.save_gradients is not None:
        save_gradients = args.save_gradients
    else:
        save_gradients = False
    
        
    if args.suffix is not None:
        suffix = suffix + args.suffix
    else:
        suffix = suffix
        
    if args.lr is not None:
        lr = args.lr
    else:
        lr = 0.00001
        
    if args.nepochs is not None:   
        if  args.nepochs != 'inf':
            nepochs = int(args.nepochs)
        else:
            nepochs = args.nepochs
    else:
        nepochs = 100    
    
    if args.nonlinearity is not None:
        nonlinearity = args.nonlinearity
    else:
        nonlinearity = 'tanh'
    
    if args.reload is not None:
        reload = True
        reload_file = args.reload
    else:
        reload = False
        
        
    if args.reload_initial is not None:
        reload_initial = True
        reload_file = args.reload_initial
    else:
        reload_initial = False
    
    run = suffix.split("Run_")
    if len(run) > 0:
        run = int(run[1])
    else:
        run = None

    digits = args.digits
    print(digits)
    print(lr) 
    print(nepochs)
    net = Net(args.ModelType,args.ninputs,args.nhidden,args.batch_size,input_bias=input_bias,posW = posW,digits = digits,noise=noise,nonlinearity=nonlinearity,pixel_by_pixel=pixel_by_pixel,run = run)
    net.initialize(args.ConnType,nNN=nNN,p=p,suffix=suffix,gain=init_gain)
    if reload:
        net.Reinitialize(reload_file)
        if digits:
            if nepochs =='inf':
                net.OutputDir = os.path.join(net.OutputDir, "generalize")
            elif nepochs == 1:
                net.OutputDir = os.path.join(net.OutputDir, "oneshot")
        else:
            if permuteIO:
                net.OutputDir = os.path.join(net.OutputDir,"permuteIO")
            else:    
                net.OutputDir = os.path.join(net.OutputDir,"fully_trained")
        if not os.path.exists(net.OutputDir):
            os.mkdir(net.OutputDir)
    elif reload_initial:
        net.Reinitialize(reload_file,iniital = True)
    
    if permuteIO:
        net.model.PermuteInputOutput()
        
    net.to(device)
    
    
    if suffix is not None:
        net.OutFile = "_".join((net.OutFile,suffix))
    print(net.OutputDir)
    
    epoch = 0
  
    if nepochs == 'inf':
        print('creating early stopping class')
        es = EarlyStopping(patience=10,mode='max')

        num_epochs = 1000
        for epoch in range(num_epochs):
            valacc = net.train(save_gradients = save_gradients,pixel_by_pixel = pixel_by_pixel,lr = lr)
            print(epoch, valacc)
            if es.step(valacc):
                break  # early stop criterion is met, we can stop now
                
        '''
        valacc =torch.from_numpy(np.array(0.0)).type(torch.float32).to(device)
        last_valacc = []
        for i in range(5):
            last_valacc.append(valacc -.2)
        while valacc > torch.mean(torch.stack(last_valacc)):
            last_valacc.append(valacc)
            last_valacc.pop(0)
            valacc = net.train(save_gradients = save_gradients,pixel_by_pixel = pixel_by_pixel,lr = lr)
            print(epoch,valacc)
            epoch += 1
        '''        
    else:
        for epoch in range(nepochs):
            valacc = net.train(save_gradients = save_gradients,pixel_by_pixel = pixel_by_pixel,lr = lr)
            print(epoch, valacc)
            epoch += 1
            
            
            
            
            
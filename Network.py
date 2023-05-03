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
import torch.jit as jit
import networkx as nx
import numpy as np
import pickle
import Parameters
import argparse
import sys
import socket
hostname = socket.gethostname()
if 'zuul' in hostname:
    sys.path.append("/home/dana/allen/programs/braintv/workgroups/tiny-blue-dot/GLIFS_ASC/main")
else:
    sys.path.append("/allen/programs/braintv/workgroups/tiny-blue-dot/GLIFS_ASC/main")
from models.networks import BNNFC

rng = np.random.default_rng(seed = 8)
nout = 10    #For sMNIST this should not change
nepochs = 100
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# if batch is size 5
#Valid numbers for inputs include: 8,14,16,28,49,56,98,112,196,392



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
    def __init__(self, n_inp, n_hid, n_out,OutputDir, batch_size,input_bias,noise,device):
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
    def __init__(self,ModelType,ninputs, nhidden,batch_size,input_bias=True,posW = False,digits = None,noise=False,device=device):
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
        self.OutputDir = os.path.join(ModelType,"corrected_test_batch")
        if noise:
            self.OutputDir = os.path.join(self.OutputDir,"noise")
            
        self.ModelType = ModelType
        self.batch_size= batch_size
        self.ninputs = ninputs
        self.nhidden = nhidden
        self.posW = posW
        seed = int(np.random.randint(100,size=1)[0])
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        
        if ModelType =="RNN":
            self.model = RNN(ninputs,nhidden,nout,self.OutputDir,batch_size,input_bias,noise,device)
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
        
    
    def Reinitialize(self,file,initial=False,device='cuda'):
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
            self.OutFile = self.OutFile + "_".join(("nNN"+str(nNN),"p",str(p),"gain",str(gain)))
        elif ConnType == 'Density':
            f = open('../MesoScope/Cortex_Thal_Cla_HippN198.sav', 'rb')
            df = pickle.load(f)
            f.close()
            CM= df['Density']
            WS_fc_nz = 198*198 - 198
            std = 1/np.sqrt(np.sqrt(WS_fc_nz))
            nz = np.nonzero(CM)
            sign =[-1,1]
            signs = self.rng.choice(sign,len(nz[0]),replace=True)
            vals = CM[nz].flatten()*signs
            CM[nz] = vals
            CM[nz] = (CM[nz] - np.mean(CM[nz]))/np.std(CM[nz])
            CM = self.gain*std*CM
            CM = torch.from_numpy(CM).type(torch.float32)
            self.OutputDir = os.path.join("RNN","Density")
            self.OutFile = self.OutFile + "_Density_gain_"+str(gain)
        elif ConnType =="EM_column":
            connectome_dir = "/allen/programs/mindscope/workgroups/tiny-blue-dot/EM/connectomes"
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
            if 'norm_198' in suffix:
                WS_fc_nz = 198*198 - 198
                std = 1/np.sqrt(np.sqrt(WS_fc_nz))
            else:    
                std=1/np.sqrt(np.sqrt(len(nz[0])))
            CM[nz] = (CM[nz] - np.mean(CM[nz]))/np.std(CM[nz])
            CM[nz] = self.gain*CM[nz]*std 
            CM = torch.from_numpy(CM).type(torch.float32)
            self.OutputDir = os.path.join("RNN","EM_column")
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
       accuracy = []
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
               self.ValidationAccuracy.append(accuracy.cpu().numpy())
           
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
            sample = list(rng.choice(len(self.dl.train_loader),100,replace= False))
        else:
            gradients = None
        for epoch in range(nepochs):
            print(epoch)
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
                self.TrainingAccuracy.append(accuracy.cpu().numpy())
            self.test(pixel_by_pixel = pixel_by_pixel)
            self.plot()
            self.save(gradients = gradients)

        

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
            train_mod = ntrain % batch_size
            test_mod = ntest % batch_size
            if train_mod !=0:
                train_dataset.data,train_dataset.targets = train_dataset.data[:-train_mod,...],train_dataset.targets[:-train_mod,...]
            if test_mod !=0:
                test_dataset.data,test_dataset.targets = test_dataset.data[:-test_mod,...],test_dataset.targets[:-test_mod,...]
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
    parser.add_argument('--nNN',type = int, required = False)
    parser.add_argument('--p',type = float, required = False)
    parser.add_argument('--reverse',type = bool, required = False)
    parser.add_argument('--Neighborhood',type = str, required = False) #Can be Target or Source
    parser.add_argument('--suffix',type = str,required = False)#Add more description to OutFile
    parser.add_argument('--input_bias',type=bool, required = False)
    parser.add_argument('--positive_weights_only',type=bool, required = False)
    parser.add_argument('--digits',nargs='+',type = int,required = False)
    parser.add_argument('--init_gain',type = float, required = False)
    parser.add_argument('--add_noise',required=False,action='store_true')
    parser.add_argument("--save_gradients",required=False, action ='store_true')
    parser.add_argument("--pixel_by_pixel",required = False, action ="store_true")
    parser.add_argument("--lr", type=float,required= False)
    
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
    if args.digits is not None:
        digits = args.digits
    else:
        digits = None
    
    if args.init_gain is not None:
        init_gain = args.init_gain
        assert args.ModelType == 'RNN' or args.ModelType == "GLIFR"
    else:
        init_gain = 1.0
    
    
    if args.add_noise is not None:
        noise = args.add_noise
        if noise:
            assert args.ModelType == 'RNN'
    else:
        noise = False
       
    if args.pixel_by_pixel is not None:
        pixel_by_pixel = args.pixel_by_pixel
        if pixel_by_pixel:
            suffix = suffix + "pixel_by_pixel"
            assert args.ninputs == 1
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
    print(lr)   
    net = Net(args.ModelType,args.ninputs,args.nhidden,args.batch_size,input_bias=input_bias,posW = posW,digits = digits,noise=noise)
    net.initialize(args.ConnType,nNN=nNN,p=p,suffix=suffix,gain=init_gain)
    net.to(device)
    
    
    if suffix is not None:
        net.OutFile = "_".join((net.OutFile,suffix))
    print(net.OutputDir)
        
    net.train(save_gradients = save_gradients,pixel_by_pixel = pixel_by_pixel,lr = lr)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:05:39 2022

@author: danamastrovito
"""
#exec(open("example.py").read())
from utils import *
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#Find out what trained models are available
#returns a dict with nNN = number of nearest neighbors and p = rewiring probability gain gain runs
params = GetModelFileParameters()
print(params)




#Find files with selected run params
#There are generally 10 runs of each with different initializations
nNN = params['nNN'][0]
prob = params['p'][0]
gain = params['gain'][0]
suffix = ""
files = GetModelFiles(nNN,prob,gain = gain)


#To find paramters for v1dd data
params = GetModelFileParameters(v1dd = True, suffix="23_4_")

#For v1dd files
nNN = params['nNN'][0]
prob = params['p'][0]
gain = params['gain'][0]
suffix = "23_4"
files = GetModelFiles(nNN,prob,gain = gain,v1dd = True, suffix= suffix)

    
    
    
#Get Hidden Weights from trained model reloaded from the first file
#Returns model weights at initialziaton if init = True
file = files[0]
Weights = GetModelWeights(file,initial=False)

#Get Shuffled Version of weights
ShuffledWeights = Shuffle_Weights(Weights)

#Get Network instance re-initialized with trained model params
#May generate a warning, but if so, this can be ignored.
batch_size = 100
net = GetInitializedModel(file,initial = False,batch_size = batch_size,noise = False,device = device,suffix=suffix)

#Get a training batch for this network
batch = GetMNIST_TrainData(net)
images, label =  next(batch)
images = image.to(device)



#Get a testing batch for this network
batch = GetMNIST_TestData(net)
images, label =  next(batch)
images = image.to(device)

#Run this network forward on an MNIST batch return predictions [batch_size,ndigits = 10]
out = net.model.forward(images)

#hidden state values [batch_size=5,nhidden=198,nsteps=28]
state = net.model.state.detach().numpy()


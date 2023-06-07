#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:05:39 2022

@author: danamastrovito, Helena-Yuhan-Liu

"""
#exec(open("example.py").read())
from utils import *
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--nNN_idx', default=0, type=int, help='nNN list index')

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dir = "/allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity"
#Find out what trained models are available
#returns a dict with nNN = number of nearest neighbors and p = rewiring probability gain gain runs
params = GetModelFileParameters(Density = False,em = False,dir = dir)
print(params)

#Find files with selected run params
#There are generally 10 runs of each with different initializations
nNN = params['nNN'][args.nNN_idx]

### Modified ### 
gain_list = []
all_rep_sim_list = []
all_delta_Wh_norm_list = []

for pp in range(len(params['p'])):
    prob = params['p'][pp]
    rep_sim_list = []
    delta_Wh_norm_list = []
    
    for ii in range(len(params['gain'])-1): # exclude gain=25
        
        gain = params['gain'][ii]
        ### Modified ###
        files = GetModelFiles(nNN,prob,gain = gain,\
                              noise = False,density=False,em = False,dir = dir)
            
        #Get Hidden Weights from trained model reloaded from the first file
        #Returns model weights at initialziaton if init = True
        file = files[0]
        Weights = GetModelWeights(file,initial=False)
        
        #Get Shuffled Version of weights
        ShuffledWeights = Shuffle_Weights(Weights)
        
        #Get Network instance re-initialized with trained model params
        #May generate a warning, but if so, this can be ignored.
        batch_size = 100
        net = GetInitializedModel(file,initial = False,batch_size = batch_size,noise = False,device = device)
        
        ### Modified ###
        # Get a training batch for this network
        batch = GetMNIST_TrainData(net)
        # images, label =  next(batch)
        images, label =  batch
        images = images.to(device)
        
        # #Get a testing batch for this network
        # batch = GetMNIST_TestData(net)
        # images, label =  next(batch)
        # images = images.to(device)
        
        #Run this network forward on an MNIST batch return predictions [batch_size,ndigits = 10]
        out = net.model.forward(images)
        
        #hidden state values [batch_size=5,nhidden=198,nsteps=28]
        state = net.model.state.detach().numpy()
        
        # print('state size = ' + str(state.size))
        
        print('###')
        
        # Get initial network and weights 
        Weights0 = GetModelWeights(file,initial=True)
        net0 = GetInitializedModel(file,initial=True,batch_size=batch_size,noise = False,device = device)
        out0 = net0.model.forward(images)
        
        # get weight change norm
        delta_Wh_norm = np.linalg.norm(Weights - Weights0)
        # print('Hidden weight change norm: ' + str(delta_Wh_norm))
        
        # Get representation similarity based on hidden state at the last step 
        activity_last = net.model.state[:,:,-1]
        activity0_last = net0.model.state[:,:,-1]
        KR0 = torch.mm(activity0_last, activity0_last.T) # (b,j) @ (j,b) -> (b,b)
        KR = torch.mm(activity_last, activity_last.T) # (b,j) @ (j,b) -> (b,b)
        rep_sim = (torch.sum(KR*KR0) / torch.norm(KR0) / torch.norm(KR)).detach().numpy()
        # print('Representation alignment: ' + str( rep_sim ))    
        
        rep_sim_list.append(rep_sim)
        delta_Wh_norm_list.append(delta_Wh_norm)
        
        if pp==0:
            gain_list.append(gain)
    
    all_rep_sim_list.append(rep_sim_list)
    all_delta_Wh_norm_list.append(delta_Wh_norm_list)
    # save    
    np.save('rep_align_nNN' + str(nNN) + '_p' + str(prob) + '.npy', rep_sim_list)


# get transition point
transitions_list = [3, 3, 2, 2, 1.5, 1.5, 1.5] # Dana's email
transition = transitions_list[args.nNN_idx]
transition_idx = gain_list.index(transition)

# plot
fig = plt.figure(figsize=(6, 3))
cmap = plt.cm.cool
plt.title('nNN = ' + str(nNN))
for pp, rep_sims in enumerate(all_rep_sim_list):
    plt.plot(gain_list, rep_sims, marker='s', linestyle='-', color=cmap(pp * 80), label=str(params['p'][pp]))
    if pp==0 and ((nNN==4) or (nNN==16)):
        plt.plot(gain_list[transition_idx+1], rep_sims[transition_idx+1], 'r*', markersize=12)
    else:
        plt.plot(transition, rep_sims[transition_idx], 'r*', markersize=12) # mark the transition point 
if nNN==4: 
    plt.legend(title="Rewiring prob")
    plt.ylabel('Representation alignment')
  
# save
fig.savefig('rep_align_nNN' + str(nNN) + '.pdf')

### Modified ### 


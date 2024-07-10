#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:05:39 2022

@author: danamastrovito, Helena-Yuhan-Liu

"""
#exec(open("KA_RA.py").read())
import sys
sys.path.append("/allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity")
from utils import *
import numpy as np
import pandas as pd
import networkx
import torch
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import argparse
import pickle

import os


#parser = argparse.ArgumentParser(description='')
#parser.add_argument('--batch_size', default=20, type=int, help='batch size, larger takes longer to run')

#args = parser.parse_args()

batch_size = 20

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

V1DD = False
DENSITY = False
DEGREE_DISTRIBUTION = False

if V1DD:
    suffix = "23_4_topology_"
    add_dir = ""#"Dales"
    Dales = False
elif DENSITY:
    suffix="_threshto5573"
    add_dir = ""
    Dales = False
elif DEGREE_DISTRIBUTION:
    suffix = ""
    add_dir = ""
    Dales = False
else:
    add_dir = ""#"Dales"
    suffix = ""
    Dales = False#True

    
def plot_per_prob(alignment,filename):
    plt.clf()
    cmap = plt.cm.cool
    
    for pp, rep_sims in enumerate(all_rep_sim_list):
        # plt.plot(gain_list, rep_sims, marker='s', linestyle='-', color=cmap(pp * 80), label=str(params['p'][pp]))
        rep_sims_std = all_rep_sim_list_std[pp]
        plt.errorbar(gain_list, rep_sims, yerr=rep_sims_std, fmt='-s', color=cmap(pp * 80), \
                      label=str(params['p'][pp]), zorder=1)
        # label transition
        transition_val = transition_df.iloc[nNN_idx * len(params['p']) + pp, 3]
        transition_idx = gain_list.index(transition_val)
        # plt.plot(transition_val, rep_sims[transition_idx], 'r*', markersize=20, zorder=2) # mark the transition point 
        plt.axvline(x=transition_val, linestyle='dotted', color=cmap(pp * 80), zorder=2)
    #if (nNN==4) or (nNN==8): 
    #    plt.legend(title="Rewiring prob")
    #    plt.ylabel('Representation alignment')    
    #ax = plt.gca()
    fig.savefig('v1dd_rep_align_nNN' + str(nNN) + '.pdf',dpi = 600)


params = GetModelFileParameters(v1dd = V1DD,Density = DENSITY, degree_distribution = DEGREE_DISTRIBUTION,add_dir = add_dir,suffix=suffix)


'''
nNN28p1_files = [GetModelFiles(nNN = 28, p=1.0,gain =g) for g in params['gain']]
#nNN28p1D_files = [GetModelFiles(nNN = 28, p=1.0,gain =g,add_dir = "Dales") for g in params['gain']]
nNN198p0_files = [GetModelFiles(nNN = 198, p=0.0,gain =g) for g in params['gain']]



#EM_23_4_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix = "23_4_",add_dir="Dales") for g in v1params['gain']]
#EM_23_4_permuted_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_permuted_") for g in params['gain']]
#EM_23_4_permuted_topology_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_permutedT_") for g in params['gain']]

#EM_23_4_block_permuted_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_block_permuted_",add_dir="Dales") for g in params['gain']]
#EM_23_4_block_permuted_topology_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_block_permuted_",add_dir="Dales/topology") for g in params['gain']]
#EM_23_4_flipped_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_flipped_",add_dir="Dales") for g in v1params['gain']]

#EM_23_4_topology_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,v1dd=True,suffix="23_4_topology_") for g in params['gain']]

#dd_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,degree_distribution=True) for g in params['gain']]


#density_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,density = True) for g in params['gain']]
density_thresh_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,density = True,suffix="_thresh") for g in params['gain']]
density_sparse_files = [GetModelFiles(nNN = 198, p=0.0,gain =g,density = True,suffix="_sparsify") for g in params['gain']]
#density_threshto_files = [GetModelFiles(nNN = 198, p=0.0, gain = g, density = True, suffix="_threshto5573") for g in params['gain']]
'''

nnNN = len(params['nNN'])
nprob = len(params['p'])
ng = len(params['gain'])
nf = 10

### Modified ### 
gain_list = []
all_rep_sim = np.zeros((nnNN, nprob, ng, nf))
all_kernel_alignment = np.zeros((nnNN, nprob, ng, nf))
all_rep_sim_std = np.zeros((nnNN, nprob, ng, nf))
all_kernel_alignment_std = np.zeros((nnNN, nprob, ng, nf))


all_rep_sim.fill(np.nan)
all_kernel_alignment.fill(np.nan)
all_rep_sim_std.fill(np.nan)
all_kernel_alignment_std.fill(np.nan)

files = GetModelFiles(params['nNN'][0],params['p'][0],gain=params['gain'][0],v1dd =V1DD,density=DENSITY, degree_distribution = DEGREE_DISTRIBUTION, suffix = suffix,add_dir = add_dir)
outfile = os.path.join(os.path.dirname(files[0]),"RA",suffix+'Alignment.pkl')

if os.path.exists(outfile):
    print('reloading ',outfile)
    with open(outfile, 'rb') as of:
        #out = {'NTK':all_kernel_alignment, 'RA':all_rep_sim}
        align = pickle.load(of)
        all_rep_sim = align['RA']
        all_kernel_alignment = align['NTK']
         
        
        
for n,nNN in enumerate(params['nNN']):
        for pp, prob in enumerate(params['p']): # iterate over probability
            for ii,gain in enumerate(params['gain']): # iterate over gain
                if np.any(np.isnan(all_rep_sim[n,pp,ii,:]) == True):
                    print(nNN,prob, gain)
                    ### Modified ###
                    files = GetModelFiles(nNN,prob,gain=gain,v1dd =V1DD,density=DENSITY, degree_distribution = DEGREE_DISTRIBUTION, suffix = suffix,add_dir = add_dir)
                        
                    #Get Hidden Weights from trained model reloaded from the first file
                    #Returns model weights at initialziaton if init = True
                    
                    for f,file in enumerate(files): # iterate over files
                        Weights = GetModelWeights(file,initial=False)
                        
                        #Get Shuffled Version of weights
                        ShuffledWeights = Shuffle_Weights(Weights)
                        
                        #Get Network instance re-initialized with trained model params
                        #May generate a warning, but if so, this can be ignored.
                        net = GetInitializedModel(file,initial = False,batch_size = batch_size,noise = False,device = device,suffix=suffix)
                        ### Modified ###
                        # Get a training batch for this network
                        batch = GetMNIST_TestData(net)
                        # images, label =  next(batch)
                        images, label =  next(batch)
                        images = images.reshape(batch_size, 28, int(images.shape[1]/28))
                        images = images.permute(2,0,1).to(device)
                        
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
                        net0 = GetInitializedModel(file,initial = True,batch_size = batch_size,noise = False,device = device,suffix=suffix,Dales = Dales)
                        out0 = net0.model.forward(images)
                        
                        # # get weight change norm
                        # delta_Wh_norm = np.linalg.norm(Weights - Weights0)
                        # print('Hidden weight change norm: ' + str(delta_Wh_norm))
                        
                        ### Rep alignment ###
                        
                        # Get representation similarity based on hidden state at the last step 
                        activity_last = net.model.state[:,:,-1]
                        activity0_last = net0.model.state[:,:,-1]
                        KR0 = torch.mm(activity0_last, activity0_last.T) # (b,j) @ (j,b) -> (b,b)
                        KR = torch.mm(activity_last, activity_last.T) # (b,j) @ (j,b) -> (b,b)
                        rep_sim = (torch.sum(KR*KR0) / torch.norm(KR0) / torch.norm(KR)).detach().numpy()
                        # print('Representation alignment: ' + str( rep_sim ))  
                       
                        
                        ### Rep alignment ###
                        
                        ### Tangent Kernel Alignment ###
                        
                        # # Method 1: using torch.autograd
                        # output_size = 10 
                        # for b in range(batch_size):
                        #     for k in range(output_size): 
                        #         # Compute dy/dw for each datapoint and output unit 
                        #         # df_1 = torch.autograd.grad(out[b,k], net.model.cell.weight_hh, retain_graph=True)[0]
                        #         # df_1 = torch.autograd.grad(out[b,k], net.model.readout.weight, retain_graph=True)[0]
                        #         # df_1 = torch.autograd.grad(out[b,k], net.model.cell.weight_ih, retain_graph=True)[0]
                        #         df_1 = torch.cat((torch.autograd.grad(out[b,k], net.model.cell.weight_hh, retain_graph=True)[0],\
                        #                           torch.autograd.grad(out[b,k], net.model.cell.weight_ih, retain_graph=True)[0],\
                        #                           torch.autograd.grad(out[b,k], net.model.readout.weight, retain_graph=True)[0].t()),\
                        #                           dim=1)
                                        
                        #         df_1 = torch.unsqueeze(df_1, dim=0)
                        #         # Then, concatenate
                        #         if (b==0) and (k==0):
                        #             df = df_1
                        #         else:
                        #             df = torch.cat((df, df_1), dim=0) 
                        
                        # Method 2: manually by tensor multiplication
                        num_steps = 28
                        num_hidden = state.shape[1]
                        output_size = num_hidden # no wout
                        df = torch.zeros(batch_size, output_size, num_hidden, num_hidden)    # Wrec        
                        # df = torch.zeros(batch_size, output_size, net.model.cell.weight_ih.shape[0], net.model.cell.weight_ih.shape[1]) # dWin
                        for step in range(num_steps-1, -1, -1):
                            psi = 1 - net.model.state[:,:,step] ** 2
                            if step == (num_steps-1):
                                # dout_dh = torch.einsum('ki,bi->bki', net.model.readout.weight, psi)
                                dout_dh = torch.einsum('ki,bi->bki', torch.eye(num_hidden), psi) # no wout
                            else:
                                dout_dh = torch.einsum('bki,ij->bkj', dout_dh, net.model.cell.weight_hh.cpu()) 
                                dout_dh = torch.einsum('bki,bi->bki', dout_dh, psi)
                            
                            if step > 0:
                                dh_dw = net.model.state[:, :, step-1].unsqueeze(1).expand(-1, num_hidden, -1) # bj->bij
                                # dh_dw = images[:, :, step-1].T.unsqueeze(1).expand(-1, num_hidden, -1) # dWin
                            else:
                                dh_dw = torch.zeros_like(dh_dw)
                            
                            # Compute and add current Jacobian to total Jacobian
                            curr_Jacobian = torch.einsum('bki,bij->bkij', dout_dh, dh_dw)                
                            df += curr_Jacobian
                        df = df.reshape(batch_size*output_size, df.shape[2], df.shape[3])            
                            
                        Kf = torch.einsum('bij,aij->ba', df, df)             
                        print('Kf norm =' + str(torch.norm(Kf).detach().numpy()))
                        
                        # Repeat for initial NTK
                        # # Method 1: using torch.autograd
                        # for b in range(batch_size):
                        #     for k in range(output_size):
                        #         # df_1 = torch.autograd.grad(out0[b,k], net0.model.cell.weight_hh, retain_graph=True)[0]
                        #         # df_1 = torch.autograd.grad(out0[b,k], net0.model.readout.weight, retain_graph=True)[0]
                        #         # df_1 = torch.autograd.grad(out0[b,k], net0.model.cell.weight_ih, retain_graph=True)[0]
                        #         df_1 = torch.cat((torch.autograd.grad(out0[b,k], net0.model.cell.weight_hh, retain_graph=True)[0],\
                        #                           torch.autograd.grad(out0[b,k], net0.model.cell.weight_ih, retain_graph=True)[0],\
                        #                           torch.autograd.grad(out0[b,k], net0.model.readout.weight, retain_graph=True)[0].t()),\
                        #                          dim=1)
                        #         df_1 = torch.unsqueeze(df_1, dim=0)
                        #         if (b==0) and (k==0):
                        #             df = df_1
                        #         else:
                        #             df = torch.cat((df, df_1), dim=0)
                        
                        # Method 2: manually by tensor multiplication
                        df = torch.zeros(batch_size, output_size, num_hidden, num_hidden)
                        # df = torch.zeros(batch_size, output_size, net.model.cell.weight_ih.shape[0], net.model.cell.weight_ih.shape[1]) #d Win
                        for step in range(num_steps-1, -1, -1):
                            psi = 1 - net0.model.state[:,:,step] ** 2
                            if step == (num_steps-1):
                                # dout_dh = torch.einsum('ki,bi->bki', net0.model.readout.weight, psi)
                                dout_dh = torch.einsum('ki,bi->bki', torch.eye(num_hidden), psi) # no wout
                            else:
                                dout_dh = torch.einsum('bki,ij->bkj', dout_dh, net0.model.cell.weight_hh.cpu()) 
                                dout_dh = torch.einsum('bki,bi->bki', dout_dh, psi)
                            
                            if step > 0:
                                dh_dw = net0.model.state[:, :, step-1].unsqueeze(1).expand(-1, num_hidden, -1) # bj->bij                    
                                # dh_dw = images[:, :, step-1].T.unsqueeze(1).expand(-1, num_hidden, -1) # dWin
                            else:
                                dh_dw = torch.zeros_like(dh_dw)
                            
                            # Compute and add current Jacobian to total Jacobian
                            curr_Jacobian = torch.einsum('bki,bij->bkij', dout_dh, dh_dw)
                            df += curr_Jacobian
                        df = df.reshape(batch_size*output_size, df.shape[2], df.shape[3])            
                        
                        K0 = torch.einsum('bij,aij->ba', df, df)            
                        kernel_alignment = (torch.sum(Kf*K0) / torch.norm(Kf) / torch.norm(K0)).detach().numpy()
                        
                        all_rep_sim[n,pp,ii,f] = rep_sim
                        all_kernel_alignment[n,pp,ii,f] = kernel_alignment

print("writing to outfile",outfile)
if not os.path.exists(os.path.dirname(outfile)):
    os.makedirs(os.path.dirname(outfile))
    
    
out = {'NTK':all_kernel_alignment, 'RA':all_rep_sim}

with open(outfile,'wb') as outf:
    pickle.dump(out, outf)

'''        
# save    
np.save('./saved_rep_align/v1dd_rep_align_mean_nNN' + str(nNN) + '_p' + str(prob) + '.npy', rep_sim_list)
np.save('./saved_rep_align/v1dd_rep_align_std_nNN' + str(nNN) + '_p' + str(prob) + '.npy', rep_sim_list_std)
np.save('./saved_NTK_align/v1dd_NTK_align_mean_nNN' + str(nNN) + '_p' + str(prob) + '.npy', kernel_alignment_list)
np.save('./saved_NTK_align/v1dd_NTK_align_std_nNN' + str(nNN) + '_p' + str(prob) + '.npy', kernel_alignment_list_std)

'''
'''
# get transition point
transition_df = pd.read_csv('gain_transitions.csv')
nNN_idx = 0

# plot rep align
fig = plt.figure(figsize=(6, 3))
cmap = plt.cm.cool
plt.title('nNN = ' + str(nNN))
for pp, rep_sims in enumerate(all_rep_sim_list):
    # plt.plot(gain_list, rep_sims, marker='s', linestyle='-', color=cmap(pp * 80), label=str(params['p'][pp]))
    rep_sims_std = all_rep_sim_list_std[pp]
    plt.errorbar(gain_list, rep_sims, yerr=rep_sims_std, fmt='-s', color=cmap(pp * 80), \
                  label=str(params['p'][pp]), zorder=1)
    # label transition
    transition_val = transition_df.iloc[nNN_idx * len(params['p']) + pp, 3]
    transition_idx = gain_list.index(transition_val)
    # plt.plot(transition_val, rep_sims[transition_idx], 'r*', markersize=20, zorder=2) # mark the transition point 
    plt.axvline(x=transition_val, linestyle='dotted', color=cmap(pp * 80), zorder=2)
if (nNN==4) or (nNN==8): 
    plt.legend(title="Rewiring prob")
    plt.ylabel('Representation alignment')    
fig.savefig('v1dd_rep_align_nNN' + str(nNN) + '.pdf')

# plot NTK align
fig2 = plt.figure(figsize=(6, 3))
cmap = plt.cm.cool
plt.title('nNN = ' + str(nNN))
for pp, kernel_align in enumerate(all_kernel_alignment_list):
    # plt.plot(gain_list, kernel_align, marker='s', linestyle='-', color=cmap(pp * 80), label=str(params['p'][pp]))
    kernel_align_std = all_kernel_alignment_list_std[pp]    
    transition_val = transition_df.iloc[nNN_idx * len(params['p']) + pp, 3]
    transition_idx = gain_list.index(transition_val)
    
    # Plot the points before the transition th element with full opacity
    plt.errorbar(gain_list[:transition_idx+1], kernel_align[:transition_idx+1], yerr=kernel_align_std[:transition_idx+1], fmt='-s', \
                  color=cmap(pp * 80), label=str(params['p'][pp]), zorder=1)        
    # Plot the points from the  transition th element onwards with lower opacity
    plt.errorbar(gain_list[transition_idx:], kernel_align[transition_idx:], yerr=kernel_align_std[transition_idx:], fmt='-s', \
                  color=cmap(pp * 80), alpha=0.2, zorder=1)
    
    # plt.plot(transition_val, kernel_align[transition_idx], 'r*', markersize=20, zorder=2) # mark the transition point  
    plt.axvline(x=transition_val, linestyle='dotted', color=cmap(pp * 80), zorder=2) 
    
if (nNN==4) or (nNN==8): 
    plt.legend(title="Rewiring prob")
    plt.ylabel('Tangent kernel alignment')    
fig2.savefig('v1dd_NTK_align_nNN' + str(nNN) + '.pdf')

### Modified ### 

'''
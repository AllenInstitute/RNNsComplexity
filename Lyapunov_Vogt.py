#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:01:36 2023

@author: dana.mastrovito
"""
#exec(open("Lyapunov_Vogt.py").read())
#this code makes use of published code for computation of Lyapunov
#Exponents in RNN from https://github.com/lyapunov-hyperopt/lyapunov_hyperopt
import sys
import Network 
from utils import GetInitializedModel,GetModelFileParameters,GetModelFiles
import torch
import pickle
import os
import socket


hostname = socket.gethostname()
if 'zuul' in hostname:
    home = "/home/dana"
else:
    home = "/"

dir = os.path.join(home,"allen/programs/braintv/workgroups/tiny-blue-dot/RNN/Complexity/")

sys.path.append(os.path.join(home,dir, "lyapunov-hyperopt"))
from lyapunov import *

#run
#mean_0.05_std_0.05_ee_0.075_0.075
#mean_0.05_std_0.05_ei_0.075
#_mean_0.05_std_0.05_ii_0.075_0.075
#mean_0.05_std_0.05_ei_0.025
#mean_0.05_std_0.05_ei_0.075_0.075
#mean_0.05_std_0.05_ie_0.025

#['',
# '_',
# '',
# '_',
# '_', 
#'_', 
#'_mean_0.05_std_0.05_ee_0.025', 
#'_mean_0.05_std_0.05_ie_0.075_0.075',
# '_mean_0.05_std_0.075',
# '_mean_0.05_std_0.05_ee_0.075',
# '_mean_0.05_std_0.025',
# '_mean_0.05_std_.075', 
#'_mean_0.05_std_0.05_ie_0.075', 
#'_mean_0.05_std_0.05_ii_0.075', 
#'_mean_0.05_std_0.05',
# '_mean_0.05_std_0.05_ii_0.025']

NARROW = False
PIXEL = False
DIGITS = False
DENSITY = False
EM  = False
V1DD = False
DEGREE_DISTRIBUTION = False
CV  = False

INITIAL = True



NOISE = False
#suffix = "_over_trained_10k_"
suffix = ""
ninputs = 28
nonlinearity = 'tanh'

if DENSITY:
    outdir = os.path.join("RNN","Density","jacobian_product_trajectory")
    suffix = "_threshto5573"
    #suffix = "_sparsify"
elif EM:
    outdir = os.path.join("RNN","EM_column","microns","jacobian_product_trajectory")
    suffix  = "permuted_"
elif V1DD:
    add_dir = "Dales"
    Dales = True
    outdir = os.path.join("RNN","EM_column","v1dd",add_dir, "jacobian_product_trajectory")
    suffix = "23_4_"
    #nonlinearity = 'relu'
elif DIGITS:
    outdir = os.path.join("RNN","digits","Gaussian","jacobian_product_trajectory")
    suffix = ""
elif PIXEL:
    outdir = os.path.join("RNN","pixel_by_pixel","Gaussian","jacobian_product_trajectory")
    ninputs = 14
    suffix = ""
elif NARROW:
    outdir = os.path.join("RNN","narrow","Gaussian","jacobian_product_trajectory")
    suffix = ""
elif DEGREE_DISTRIBUTION:
    outdir = os.path.join("RNN","degree_dist","jacobian_product_trajectory")
    add_dir = ""
elif CV:
    Dales = False
    add_dir = ""
    suffix = "mean_0.05_std_0.05_ii_0.075_0.05_"
    outdir = os.path.join("RNN","coefficient_variation","jacobian_product_trajectory")
else:
    add_dir = "fully_trained"#"coefficient_variation"
    Dales = False
    outdir = os.path.join("RNN","Gaussian",add_dir, "jacobian_product_trajectory")
    suffix = ""


if NOISE:
    outdir = outdir = os.path.join("RNN","Gaussian","noise","jacobian_product_trajectory")
                      
if not os.path.exists(outdir):
    os.mkdir(outdir)

    

if INITIAL:
    outfile_suffix=suffix + "_init"
else:
    outfile_suffix=suffix 


def calc_lyap(data, model):
		model.eval()
		model.lyapunov = True
		h = torch.zeros(batch_size,model.cell.hidden_size)
		i = torch.randint(low = 0, high = data.shape[0], size =  (1,)).item()
		LEs, rvals, qvect, Js = calc_LEs_an(torch.swapaxes(data[i],0,1), h, model = model, k_LE = 10000, rec_layer = 'rnn', warmup = 0, T_ons = 1)
		LE_mean, LE_std = LE_stats(LEs)
		model.lyapunov = False
		return LE_mean, LE_std, LEs,rvals, qvect, Js

batch_size = 100

params = GetModelFileParameters(em = EM,v1dd = V1DD, Density = DENSITY,digits = DIGITS,narrow = NARROW, coeff_var = CV,
                                pixel = PIXEL,degree_distribution = DEGREE_DISTRIBUTION,suffix=suffix,add_dir  = add_dir,home = home)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


dl = Network.sMNIST(batch_size)
images = []
for batch in dl.test_loader:
    batch_images = batch[0]
    batch_images = batch_images.reshape(batch_size, ninputs, int(batch_images.shape[1]/ninputs))
    batch_images = batch_images.permute(2,0,1).to(device)
    images.append(batch_images)

data  = torch.stack(images)



params['nNN'].reverse()

for n, nNN in enumerate(params['nNN']):
    for p, prob in enumerate(params['p']):
        for g, gain in enumerate(params['gain']):
            files = GetModelFiles(nNN, prob, gain=gain, NetworkType='RNN', noise = NOISE, density=DENSITY,v1dd = V1DD,coeff_var = CV,
                                  em = EM,digits = DIGITS,narrow = NARROW, pixel = PIXEL,degree_distribution = DEGREE_DISTRIBUTION, suffix=suffix,add_dir = add_dir,home = home)

            for file in files:
                print(file)
                #if True:
                if not os.path.exists(os.path.join(outdir,os.path.basename(file)+outfile_suffix)):
                    net = GetInitializedModel(file,initial = INITIAL, batch_size = batch_size,suffix = suffix, Dales = Dales, device=device,nonlinearity = nonlinearity)
                    net.model.rnn_layer = net.model.cell
                    model_params = list(net.model.rnn_layer.named_parameters())
                    net.model.rnn_layer.all_weights = [[p[1] for p in model_params]]
                    LE_mean, LE_std, LEs,rvals, qvect, Js = calc_lyap(data, net.model)     
                    LE = {}
                    LE['mean'] = LE_mean
                    LE['std'] = LE_std
                    #LE['all'] = LEs
                    #LE['rvals'] = rvals
                    #LE['qvect'] = qvect
                    #LE['Js'] = Js
                    with open(os.path.join(outdir,os.path.basename(file)+outfile_suffix),'wb') as f:
                        pickle.dump(LE,f) 
                    del net


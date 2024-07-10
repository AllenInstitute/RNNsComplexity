#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:24:26 2023

@author: dana.mastrovito
"""
#exec(open("calc_time_to_max.py").read())
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import pickle
import numpy as np
from utils import  GetModelFiles,GetModelFileParameters
from utils import GetModelWeights,GetInitializedModel
import torch

device = 'cpu'

nfiles = 10

dir = os.path.join("RNN")

DENSITY = False
V1DD = True
NOISE = False
EM = False
DIGITS = False
DEGREE_DISTRIBUTION = False
FULLY_TRAINED = False


NORM = False
suffix  = ""


#criteria = 'acc'
#criteria = 'num_epochs'


if EM:
    dir = os.path.join(dir, "EM_column","microns")
    if NORM:
        suffix="permuted"
elif V1DD:
    dir = os.path.join(dir, "EM_column","v1dd")
    add_dir  ="Dales/relu"
    suffix="23_4_"
    if add_dir is not None:
        dir = os.path.join(dir, add_dir)
elif DENSITY:
    dir = os.path.join(dir, "Density")
    suffix = "_threshto5573"
elif DIGITS:
    dir = os.path.join(dir, 'digits','Gaussian')
elif DEGREE_DISTRIBUTION:
    dir = os.path.join(dir, "degree_dist")
else:
    dir = os.path.join(dir, "Gaussian")
    if FULLY_TRAINED:
        add_dir= "fully_trained"
    #add_dir = "Dales"
    if add_dir is not None:
        dir = os.path.join(dir, add_dir)

if NOISE:
    dir = os.path.join(dir, "noise")

dir = os.path.join(dir,"pcist")

params = GetModelFileParameters(em= EM,v1dd = V1DD,Density = DENSITY,digits = DIGITS,degree_distribution = DEGREE_DISTRIBUTION, suffix = suffix,add_dir = add_dir)


acc = 90

if not FULLY_TRAINED:
    outfile = os.path.join(dir,"time_to_max_acc_nNN" ) 
else:
    outfile = os.path.join(dir,"time_to_acc_"+str(acc)+"_nNN")



for nNN in params['nNN']:
    if True:
    #if not os.path.exists(outfile+str(nNN)+suffix+".pkl"):
        print("nNN",nNN)
        time_to_max_acc = np.zeros((len(params['gain']),len(params['p']),nfiles))
        max_acc = np.zeros((len(params['gain']),len(params['p']),nfiles))
        training_acc = np.zeros((len(params['gain']),len(params['p']),nfiles))
        time_to_max_acc.fill(np.nan)
        max_acc.fill(np.nan)
        training_acc.fill(np.nan)
        mfiles = np.ndarray((len(params['gain']),len(params['p']),nfiles),dtype = object)
        for g, gain in enumerate(params['gain']):
            print("gain",gain)
            for p, prob in enumerate(params['p']):
                print("prob",prob)
                files = GetModelFiles(nNN, prob, gain = gain,noise = NOISE,density = DENSITY,v1dd = V1DD,em = EM,digits = DIGITS,degree_distribution = DEGREE_DISTRIBUTION,suffix= suffix,add_dir = add_dir)
                if len(files) >0:
                    mfiles[g,p,:len(files)] = files
                    for f,file in enumerate(files):
                        try:
                            model = torch.load(file,map_location=device)
                        except:
                            print("couldn't read file ",file)
                        else:
                            valacc = np.array(model['ValidationAccuracy'])
                            if FULLY_TRAINED:
                                if valacc[0] >= acc:
                                    model = torch.load(os.path.join(os.path.dirname(file),'..',os.path.basename(file)),map_location=device)
                                    time_to_max_acc[g,p,f] = (np.where(np.array(model['ValidationAccuracy']) >= acc)[0][0])
                                else:
                                    ttm = np.where(np.array(model['ValidationAccuracy']) >= acc)
                                    if ttm[0].size >0:
                                        time_to_max_acc[g,p,f] = (ttm[0][0]) + 3900.
                            else:
                                assert  (len(valacc)/100. >=  39.0) #(valacc[-1] > 90) or
                                if valacc[-1] < acc:
                                    print(file, valacc[-1])
                                    trainacc = np.array(model['TrainingAccuracy'])
                                    training_acc[g,p,f] = np.max(trainacc)
                                    max_acc [g,p,f] = np.max(valacc)
                                    time_to_max_acc[g,p,f] = np.where(valacc == np.max(valacc))[0][0]
                            
                                    #/39.0 # 39 is number of batches per epoch
                            
        timetomax = {}
        timetomax['train_acc'] = training_acc
        timetomax['max_acc'] = max_acc
        timetomax['nepochs']  = time_to_max_acc
        timetomax['files'] = mfiles
        if not FULLY_TRAINED:
            with open(os.path.join(dir,"time_to_max_acc_nNN"+str(nNN)+suffix+".pkl"),'wb') as wf:
                pickle.dump(timetomax,wf)
        else:
            with open(os.path.join(dir,"time_to_acc_"+str(acc)+"_nNN"+str(nNN)+suffix+".pkl"),'wb') as wf:
                pickle.dump(timetomax,wf)
    
        
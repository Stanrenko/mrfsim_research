#%load_ext autoreload
#%autoreload 2

#import matplotlib
#matplotlib.u<se("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,BruteDictSearch
from utils_mrf import *

import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat
import twixtools
from mutools import io
import cv2
import scipy
import numpy as np


base_folder = "./3D"


suffix=""
nb_filled_slices = 8
nb_empty_slices=2
repeat_slice=1
nb_slices = nb_filled_slices+2*nb_empty_slices

nb_allspokes = 1400
nspoke=8


undersampling_factor=1


name = "SquareSimu3D"

localfile="/"+name
filename = base_folder+localfile

folder = "/".join(str.split(filename,"/")[:-1])


filename_paramMap=filename+"_paramMap_sl{}_rp{}{}.pkl".format(nb_slices,repeat_slice,"")
filename_mask= filename+"_mask_sl{}_rp{}_us{}{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)

filename_volume = filename+"_volumes_sl{}_rp{}_us{}_{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)
dictfile='./mrf175_SimReco2_light.dict'

mask=np.load(filename_mask)
volumes_all=np.load(filename_volume)
with open(filename_paramMap, "rb") as file:
    paramMap=pickle.load(file)

all_signals=volumes_all[:,mask>0]
all_signals/=np.linalg.norm(all_signals,axis=0)

nb_signals=100

total_nb_signals=all_signals.shape[1]

nb_signals=np.minimum(total_nb_signals,nb_signals)
ind=np.random.choice(total_nb_signals,size=nb_signals,replace=False)
ind=np.arange(nb_signals)
ff_gt=paramMap["ff"][ind]
df_gt=paramMap["df"][ind]

all_signals=all_signals[:,ind]

pca = True
threshold_pca_bc = 20
threshold_pca_brute=20

nb_signals=all_signals.shape[1]
split=nb_signals+1

return_matched_signals=True

useGPU=True
dict_optim_brute=BruteDictSearch(FF_list=np.arange(0,1.01,0.01),mask=mask,split=split,pca=pca,threshold_pca=threshold_pca_brute,log=False,useGPU_dictsearch=useGPU,ntimesteps=None,log_phase=False,return_matched_signals=return_matched_signals)
dict_optim_bc =SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=useGPU, useGPU_simulation=False,
                                gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,return_matched_signals=return_matched_signals)



#all_maps_brute,matched_signals_brute=dict_optim_brute.search_patterns(dictfile,all_signals)
all_maps_bc_cf,matched_signals_bc_cf=dict_optim_bc.search_patterns_test(dictfile,all_signals)
#all_maps_bc_matrix,matched_signals_bc_matrix=dict_optim_bc.search_patterns_matrix(dictfile,all_signals)


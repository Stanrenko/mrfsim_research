




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


suffix="_fullReco"
nb_filled_slices = 8
nb_empty_slices=2
repeat_slice=1
nb_slices = nb_filled_slices+2*nb_empty_slices

nb_allspokes = 1400
nspoke=8


undersampling_factor=1


name = "SquareSimu3D_SS_SimReco2"

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

pca = True
threshold_pca_bc = 20
threshold_pca_brute=20

nb_signals=volumes_all[:,mask>0].shape[1]

splits = [10]
return_matched_signals = False

useGPU = False

all_times_matrix = []

for split in splits:
    # dict_optim_brute=BruteDictSearch(FF_list=np.arange(0,1.01,0.01),mask=mask,split=split,pca=pca,threshold_pca=threshold_pca_brute,log=False,useGPU_dictsearch=useGPU,ntimesteps=None,log_phase=False,return_matched_signals=return_matched_signals)
    dict_optim_bc = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                     threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=useGPU,
                                     useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                     return_matched_signals=return_matched_signals)
    # dict_optim_bc_matrix = dict_optim_bc_cf
    start_time = time.time()
    all_maps_bc_matrix = dict_optim_bc.search_patterns_matrix(dictfile, volumes_all)
    end_time = time.time()

    all_times_matrix.append((end_time - start_time) / nb_signals)
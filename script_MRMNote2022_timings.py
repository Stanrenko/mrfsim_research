
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
dictfile='./mrf175_Dico2_Invivo.dict'
dictfile_light='./mrf175_Dico2_Invivo_light_for_matching.dict'

mask=np.load(filename_mask)
volumes_all=np.load(filename_volume)
with open(filename_paramMap, "rb") as file:
    paramMap=pickle.load(file)

pca = True
threshold_pca_bc = 20
threshold_pca_brute=20


nb_signals=2000

all_signals=volumes_all[:,mask>0]
total_nb_signals=all_signals.shape[1]

nb_signals=np.minimum(total_nb_signals,nb_signals)
ind=np.random.choice(total_nb_signals,size=nb_signals,replace=False)


all_signals=all_signals[:,ind]

splits=[1,10,100,200,300,400,500]
#splits = [1, 10]
return_matched_signals = False

### GPU #####
useGPU = True

all_times_cf_gpu = []
for split in splits:

    # dict_optim_brute=BruteDictSearch(FF_list=np.arange(0,1.01,0.01),mask=mask,split=split,pca=pca,threshold_pca=threshold_pca_brute,log=False,useGPU_dictsearch=useGPU,ntimesteps=None,log_phase=False,return_matched_signals=return_matched_signals)
    dict_optim_bc = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                     threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=useGPU,
                                     useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                     return_matched_signals=return_matched_signals)
    # dict_optim_bc_matrix = dict_optim_bc_cf

    try:
        start_time = time.time()
        all_maps_bc_cf = dict_optim_bc.search_patterns_test_multi(dictfile, all_signals)
        end_time = time.time()
    except:
        continue
    all_times_cf_gpu.append((end_time - start_time) / nb_signals * 1000)


fig=plt.figure(figsize=(15,10))
ax  = fig.add_subplot(111)
ax.bar(np.array(splits[:len(all_times_cf_gpu)]).astype(str),all_times_cf_gpu,color="gray",width=0.25)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_ylabel("Matching time per pixel (ms)",fontsize=18)
ax.set_xlabel("Batch size",fontsize=18)
fig.patch.set_facecolor("white")

fig.savefig("Proposed method - batch size analysis.png",dpi=600)

all_times_matrix_gpu = []

for split in splits:

    dict_optim_bc = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                     threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=useGPU,
                                     useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                     return_matched_signals=return_matched_signals, dictfile_light=dictfile_light,
                                     threshold_ff=0.9)
    # dict_optim_bc_matrix = dict_optim_bc_cf


    try:
        start_time = time.time()
        all_maps_bc_cf = dict_optim_bc.search_patterns_test_multi_2_steps_dico(dictfile, all_signals)
        end_time = time.time()
    except:
        continue

    all_times_matrix_gpu.append((end_time - start_time) / nb_signals * 1000)
#
#
fig=plt.figure(figsize=(15,10))
ax  = fig.add_subplot(111)
ax.bar(np.array(splits).astype(str),all_times_matrix_gpu,color="gray")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_ylabel("Matching time per pixel (ms)",fontsize=18)
ax.set_xlabel("Batch size",fontsize=18)
fig.patch.set_facecolor("white")

fig.savefig("Proposed method with clustering - batch size analysis.png",dpi=600)

all_times_brute_gpu = []

for split in splits:

    dict_optim_brute=BruteDictSearch(FF_list=np.arange(0,1.01,0.05),mask=mask,split=split,pca=pca,threshold_pca=threshold_pca_brute,log=False,useGPU_dictsearch=useGPU,ntimesteps=None,log_phase=False,return_matched_signals=return_matched_signals,n_clusters_dico=1000,pruning=0.05)

    # dict_optim_bc =SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
    #                                threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=useGPU, useGPU_simulation=False,
    #                                gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,return_matched_signals=return_matched_signals)
    # dict_optim_bc_matrix = dict_optim_bc_cf
    try:
        start_time = time.time()
        all_maps_brute = dict_optim_brute.search_patterns(dictfile, all_signals)
        end_time = time.time()
    except:
        continue

    all_times_brute_gpu.append((end_time - start_time) / nb_signals * 1000)




splits = [10, 100, 1000, nb_signals+1]
#splits = [1,10]
return_matched_signals = False

### GPU #####
useGPU = False

all_times_cf = []
for split in splits:

    # dict_optim_brute=BruteDictSearch(FF_list=np.arange(0,1.01,0.01),mask=mask,split=split,pca=pca,threshold_pca=threshold_pca_brute,log=False,useGPU_dictsearch=useGPU,ntimesteps=None,log_phase=False,return_matched_signals=return_matched_signals)
    dict_optim_bc = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                     threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=useGPU,
                                     useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                     return_matched_signals=return_matched_signals)
    # dict_optim_bc_matrix = dict_optim_bc_cf

    try:
        start_time = time.time()
        all_maps_bc_cf = dict_optim_bc.search_patterns_test(dictfile, all_signals)
        end_time = time.time()
    except:
        continue
    all_times_cf.append((end_time - start_time) / nb_signals * 1000)


#splits = [1,10]
all_times_matrix = []

for split in splits:

    # dict_optim_brute=BruteDictSearch(FF_list=np.arange(0,1.01,0.01),mask=mask,split=split,pca=pca,threshold_pca=threshold_pca_brute,log=False,useGPU_dictsearch=useGPU,ntimesteps=None,log_phase=False,return_matched_signals=return_matched_signals)
    dict_optim_bc = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                     threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=useGPU,
                                     useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                     return_matched_signals=return_matched_signals)
    # dict_optim_bc_matrix = dict_optim_bc_cf

    try:
        start_time = time.time()
        all_maps_bc_matrix = dict_optim_bc.search_patterns_matrix(dictfile, all_signals)
        end_time = time.time()
    except:
        continue

    all_times_matrix.append((end_time - start_time) / nb_signals * 1000)

all_times_brute = []

for split in splits:

    dict_optim_brute = BruteDictSearch(FF_list=np.arange(0, 1.01, 0.01), mask=mask, split=split, pca=pca,
                                       threshold_pca=threshold_pca_brute, log=False, useGPU_dictsearch=useGPU,
                                       ntimesteps=None, log_phase=False, return_matched_signals=return_matched_signals)
    # dict_optim_bc =SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
    #                                threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=useGPU, useGPU_simulation=False,
    #                                gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,return_matched_signals=return_matched_signals)
    # dict_optim_bc_matrix = dict_optim_bc_cf
    try:
        start_time = time.time()
        all_maps_brute = dict_optim_brute.search_patterns(dictfile, all_signals)
        end_time = time.time()
    except:
        continue

    all_times_brute.append((end_time - start_time) / nb_signals * 1000)









#import matplotlib
#matplotlib.u<se("TkAgg")
import numpy as np
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,GaussianWeighting,BruteDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat
from mutools import io
from sklearn import linear_model
from scipy.optimize import minimize
from epgpy import epg

base_folder = "./data/InVivo/3D"


#Artefact simulation
localfile = "/patient.004.v1/meas_MID00022_FID03367_raFin_3D_tra_1x1x5mm_FULL_bw5_aggregated_volumes_MRF_map_model.pkl"

filename = base_folder+localfile
filename_kdata_label = str.split(filename,"volumes_MRF_map_model.pkl")[0]+"kdata_labels_model.npy"
filename_volume_label = str.split(filename,"volumes_MRF_map_model.pkl")[0]+"volumes_label{}_model.npy"



with open(filename,"rb") as file:
    all_maps_model=pickle.load(file)

mask_labels = all_maps_model[1]
maps = all_maps_model[0]
cluster_centers=all_maps_model[2]

unique_labels=np.unique(mask_labels.flatten())

nb_allspokes = 1400
undersampling_factor = 1
incoherent = True
mode = "old"

ntimesteps=175

kdata_mask_label=[]
volumes_mask_label=[]
unique_labels=[0]
for l in tqdm(unique_labels):
    current_mask=(mask_labels==l)*1


    nb_slices=current_mask.shape[0]
    npoint=2*current_mask.shape[1]


    radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                           nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    radial_traj.traj=radial_traj.get_traj().reshape(1,-1,3)
    current_mask=np.expand_dims(current_mask,axis=0)

    current_kdata=generate_kdata(current_mask,radial_traj,useGPU=True)[0]

    current_kdata=current_kdata.reshape(nb_allspokes,-1)
    current_volumes=simulate_radial_undersampled_images_multi(np.expand_dims(current_kdata,axis=0),radial_traj,mask_labels.shape,density_adj=True,b1=np.ones((1,)+mask_labels.shape),ntimesteps=ntimesteps,light_memory_usage=True)
    del current_kdata
    #kdata_mask_label.append(current_kdata)

    np.save(filename_volume_label.format(l),current_volumes)

np.save(filename_volume_label,np.array(volumes_mask_label))


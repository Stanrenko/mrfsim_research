
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from trajectory import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np


from mutools import io

folder="./data/KneePhantom/Phantom1/"
file_mask="maskFull_Control_multislice.npy"
file_dico="dicoMasks_Control_multislice_retained.pkl"

mask_full=np.load(folder+file_mask)


with open(folder+file_dico,"rb") as file:
    dico_retained=pickle.load(file)


#animate_images(mask_full)



keys=list(dico_retained.keys())
num=0
mask=dico_retained[keys[0]]
#animate_images(mask)

volumes=np.tile(mask,(95,1,1,1))
volumes=volumes.astype("complex64")
b1_all_slices=np.ones((1,)+mask.shape)

radial_traj=Radial3D(total_nspokes=760,undersampling_factor=2,npoint=512,nb_slices=24,incoherent=False,mode="old")


volumes_us=undersampling_operator(volumes,radial_traj,b1_all_slices,density_adj=True,light_memory_usage=True)

np.save("test.npy",volumes_us)

animate_images(volumes_us[:,12])
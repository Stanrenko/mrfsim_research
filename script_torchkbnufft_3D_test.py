
#import matplotlib
#matplotlib.u<se("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
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

import torchkbnufft as tkbn
import torch

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./data/InVivo/3D"



localfile="/patient.008.v1/meas_MID00199_FID26313_raFin_3D_tra_1x1x5mm_FULL_new_rw_11.dat"

filename = base_folder+localfile

filename_volume = str.split(filename,".dat") [0]+"_volumes{}.npy".format("")
filename_b1 = str.split(filename,".dat") [0]+"_b1{}.npy".format("")
#filename_kdata = str.split(filename,".dat") [0]+"_kdata{}.npy".format("")
filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"
filename_kdata = str.split(filename,".dat") [0]+"_kdata{}.npy".format("")
window=8

file = open(filename_seqParams, "rb")
dico_seqParams = pickle.load(file)
file.close()


wnidow=8

try:
    use_navigator_dll=dico_seqParams["use_navigator_dll"]
except:
    use_navigator_dll=False

if use_navigator_dll:
    meas_sampling_mode=dico_seqParams["alFree"][14]
    nb_gating_spokes = dico_seqParams["alFree"][6]
else:
    meas_sampling_mode = dico_seqParams["alFree"][12]
    nb_gating_spokes = 0

if nb_gating_spokes>0:
    meas_orientation =  dico_seqParams["alFree"][11]
    if meas_orientation==1:
        nav_direction = "READ"
    elif meas_orientation==2:
        nav_direction = "PHASE"
    elif meas_orientation==3:
        nav_direction = "SLICE"

nb_segments = dico_seqParams["alFree"][4]
dummy_echos = dico_seqParams["alFree"][5]

ntimesteps=int(nb_segments/window)


x_FOV = dico_seqParams["x_FOV"]
y_FOV = dico_seqParams["y_FOV"]
z_FOV = dico_seqParams["z_FOV"]
#z_FOV=64
nb_part = dico_seqParams["nb_part"]
undersampling_factor = dico_seqParams["alFree"][9]
#undersampling_factor=1

del dico_seqParams

if meas_sampling_mode==1:
    incoherent=False
    mode = None
elif meas_sampling_mode==2:
    incoherent = True
    mode = "old"
elif meas_sampling_mode==3:
    incoherent = True
    mode = "new"


#kdata=np.load(filename_kdata)
#data_shape=kdata.shape

nb_channels=30
nb_allspokes = 1400
npoint = 800
nb_slices = 48
image_size = (nb_slices, int(npoint/2), int(npoint/2))

density = np.abs(np.linspace(-1, 1, npoint))

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)

b1_all_slices=np.load(filename_b1)
volumes_all = np.load(filename_volume)




if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device=torch.device("cpu") #too big - crashes GPU

ts=0
ktraj=radial_traj.get_traj()[ts]
ktraj=np.moveaxis(ktraj,1,0)
ktraj=torch.from_numpy(ktraj).to(device)

dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=image_size)

dcomp_kernel = tkbn.calc_toeplitz_kernel(ktraj, image_size, weights=dcomp, norm="ortho")







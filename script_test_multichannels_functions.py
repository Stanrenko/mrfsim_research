
#import matplotlib
#matplotlib.use("TkAgg")
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

base_folder = "./data/InVivo/3D"
localfile ="/20211105_TestCS_MRF/meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"

filename = base_folder+localfile


filename_save=str.split(filename,".dat") [0]+".npy"
folder = "/".join(str.split(filename,"/")[:-1])


filename_b1 = str.split(filename,".dat") [0]+"_b1.npy"
filename_volume = str.split(filename,".dat") [0]+"_volumes.npy"
filename_kdata = str.split(filename,".dat") [0]+"_kdata.npy"
filename_mask= str.split(filename,".dat") [0]+"_mask.npy"

data = np.load(filename_save)
data = np.moveaxis(data, 0, -1)
# data=np.moveaxis(data,-2,-1)

data_shape = data.shape

nb_channels = data_shape[0]

ntimesteps = 175

nb_allspokes = data_shape[1]
npoint = data_shape[-1]
nb_slices = data_shape[2]
image_size = (nb_slices, int(npoint/2), int(npoint/2))
undersampling_factor=1
incoherent=False

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent)



#kdata_all_channels_all_slices=open_memmap(filename_kdata)

# Density adjustment all slices
print("Performing Density Adjustment....")
density = np.abs(np.linspace(-1, 1, npoint))
kdata_all_channels_all_slices_new_density_calc = data.reshape(-1, npoint)
kdata_all_channels_all_slices = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in data]
del data
kdata_all_channels_all_slices_new_density_calc = (kdata_all_channels_all_slices_new_density_calc*density).reshape(data_shape)
kdata_all_channels_all_slices=np.array(kdata_all_channels_all_slices).reshape(data_shape)



random_channel=np.random.choice(list(range(nb_channels)))
random_timestep = np.random.choice(list(range(nb_allspokes)))
print(np.max(np.abs(kdata_all_channels_all_slices[random_channel,random_timestep,:,:]-kdata_all_channels_all_slices_new_density_calc[random_channel,random_timestep,:,:])))
random_slice = np.random.choice(list(range(nb_slices)))
plt.figure()
plt.plot(np.abs(kdata_all_channels_all_slices_new_density_calc[random_channel,random_timestep,random_slice,:]))
plt.plot(np.abs(kdata_all_channels_all_slices[random_channel,random_timestep,random_slice,:]))


res = 16
#b1_all_slices=np.load(filename_b1)
b1_all_slices=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=False)
b1_all_slices_light_mem=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=True)

del kdata_all_channels_all_slices

sl=20
list_images = list(np.abs(b1_all_slices[sl]))
plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))

list_images = list(np.abs(b1_all_slices_light_mem[sl]))
plot_image_grid(list_images,(6,6),title="Light Mem - Sensitivity map for slice {}".format(sl))

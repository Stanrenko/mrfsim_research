import sys
#path = r"/Users/constantinslioussarenko/PythonGitRepositories/MyoMap"
path = r"/home/cslioussarenko/PythonRepositories"
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")
sys.path.append(path+"/mrf-sim")

#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
from utils_reco import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle

from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

from multiprocessing import Pool

from voxelmorph.tf.utils import transform
import neurite as ne


filename_volume="./data/InVivo/3D/patient.003.v21/meas_MID00184_FID76217_raFin_3D_tra_1_6x1_6x12mm_FULL_new_respi_shortTE_bart12_volumes_allbins_registered_allindex.npy"
file_model="./data/InVivo/3D/patient.003.v21/meas_MID00184_FID76217_raFin_3D_tra_1_6x1_6x12mm_FULL_new_respi_shortTE_bart12_volumes_allbins_registered_allindex_vxm_model_weights.h5"

file_config="./config_train_voxelmorph_shell.json"

with open(file_config,"r") as file:
    config_train=json.load(file)


all_volumes=np.load(filename_volume)

all_volumes=all_volumes.astype("float32")
nb_gr=all_volumes.shape[0]



#pad_amount=config_train["padding"]
#pad_amount=tuple(tuple(l) for l in pad_amount)
n = all_volumes.shape[-1]
pad_1 = 2 ** (int(np.log2(n)) + 1) - n
pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n
if pad_2 < 0:
    pad = int(pad_1 / 2)
else:
    pad = int(pad_2 / 2)
pad_amount = ((0, 0), (pad, pad), (pad, pad))


nb_features=config_train["nb_features"]

sl=20
gr=2

plt.imshow(np.abs(all_volumes[gr,sl]))



x_val_fixed,x_val_moving=format_input_voxelmorph(all_volumes[[gr,gr+1],sl:(sl+2)],pad_amount,sl_down=0,sl_top=2)
inshape=x_val_fixed.shape[1:]
vxm_model=vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
vxm_model.load_weights(file_model)

val_input=[x_val_moving[...,None],x_val_fixed[...,None]]
val_pred=vxm_model.predict(val_input)


field_array=np.zeros(shape=val_pred[1].shape[:-1]+(3,),dtype=val_pred[1].dtype)
field_array[:,:,:,:2]=val_pred[1]

import tensorflow as tf

volumes_moved=transform(tf.convert_to_tensor(np.moveaxis(x_val_moving,0,-1)),np.moveaxis(val_pred[1],0,-2))
plt.imshow(np.array(volumes_moved[:,:,0]).squeeze())
volumes_fixed=x_val_fixed
plt.imshow(np.array(volumes_fixed).squeeze())


animate_images([x_val_moving.squeeze(),x_val_fixed.squeeze()])
animate_images([np.array(volumes_moved[:,:,0]).squeeze(),x_val_fixed[0].squeeze()])
animate_images([np.array(volumes_moved[:,:,0]).squeeze(),x_val_moving[0].squeeze()])

ne.utils.volshape_to_meshgrid(x_val_fixed[0].shape, indexing="ij")


from scipy.interpolate import interpn

image_size=x_val_moving.shape[1:]

mesh=np.array(ne.utils.volshape_to_meshgrid(image_size, indexing="ij"))

loc=val_pred[None,:]+mesh


file_deformation="./data/InVivo/3D/patient.003.v21/meas_MID00184_FID76217_raFin_3D_tra_1_6x1_6x12mm_FULL_new_respi_shortTE_bart12_volumes_allbins_registered_allindex_deformation_map.npy"
filename_volume="./data/InVivo/3D/patient.003.v21/meas_MID00184_FID76217_raFin_3D_tra_1_6x1_6x12mm_FULL_new_respi_shortTE_bart12_volumes_allbins.npy"

file_deformation="./data/InVivo/3D/patient.003.v21/meas_MID00186_FID76219_raFin_3D_tra_0_8x0_8x12mm_FULL_new_respi_bart12_volumes_allbins_registered_allindex_deformation_map.npy"
filename_volume="./data/InVivo/3D/patient.003.v21/meas_MID00186_FID76219_raFin_3D_tra_0_8x0_8x12mm_FULL_new_respi_bart12_volumes_allbins_registered_allindex.npy"
#filename_volume="./data/InVivo/3D/patient.003.v21/meas_MID00186_FID76219_raFin_3D_tra_0_8x0_8x12mm_FULL_new_respi_bart12_volumes_allbins.npy"


file_deformation="./data/InVivo/3D/patient.003.v21/meas_MID00186_FID76219_raFin_3D_tra_0_8x0_8x12mm_FULL_new_respi_bart26_volumes_allbins_denoised_gamma_0_8_deformation_map.npy"
filename_volume="./data/InVivo/3D/patient.003.v21/meas_MID00186_FID76219_raFin_3D_tra_0_8x0_8x12mm_FULL_new_respi_bart26_volumes_allbins_registered_allindex.npy"
#filename_volume="./data/InVivo/3D/patient.003.v21/meas_MID00186_FID76219_raFin_3D_tra_0_8x0_8x12mm_FULL_new_respi_bart26_volumes_allbins_denoised_gamma_0_8.npy"
#filename_volume="./data/InVivo/3D/patient.003.v21/meas_MID00186_FID76219_raFin_3D_tra_0_8x0_8x12mm_FULL_new_respi_bart26_volumes_allbins.npy"


all_volumes=np.real(np.load(filename_volume))

deformation_map=np.load(file_deformation)

deformed_volumes = np.zeros_like(all_volumes)
nb_gr=all_volumes.shape[0]

for gr in range(nb_gr):
    deformed_volumes[gr] = apply_deformation_to_complex_volume(all_volumes[gr], deformation_map[:, gr],interp=cv2.INTER_LINEAR)

sl=20

plt.close("all")
moving_image=np.concatenate([deformed_volumes[:,sl],deformed_volumes[1:-1,sl][::-1]],axis=0)
animate_images((moving_image-np.min(moving_image,axis=(1,2),keepdims=True))/(np.max(moving_image,axis=(1,2),keepdims=True)-np.min(moving_image,axis=(1,2),keepdims=True)),interval=10)
plt.close("all")
moving_image=np.concatenate([all_volumes[:,sl],all_volumes[1:-1,sl][::-1]],axis=0)
animate_images((moving_image-np.min(moving_image,axis=(1,2),keepdims=True))/(np.max(moving_image,axis=(1,2),keepdims=True)-np.min(moving_image,axis=(1,2),keepdims=True)),interval=10)



deformation_map.dtype
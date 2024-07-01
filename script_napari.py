

import numpy as np
import napari
from tqdm import tqdm


#volumes=np.load("./data/InVivo/3D/DMD/meas_MID00059_FID81502_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
#volumes=np.load("./data/InVivo/3D/patient.011.v2/meas_MID00030_FID81817_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins.npy")

#volumes=np.load("./data/InVivo/3D/patient.011.v2/meas_MID00030_FID81817_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
#volumes=np.load("./data/InVivo/3D/patient.003.v24/meas_MID00287_FID82176_raFin_3D_tra_1x1x5mm_FULL_new_respi_coro_bart30_volumes_allbins.npy")

#volumes=np.load("./data/InVivo/3D/patient.003.v24/meas_MID00287_FID82176_raFin_3D_tra_1x1x5mm_FULL_new_respi_coro_bart30_volumes_allbins.npy")
#volumes=np.load("./data/InVivo/3D/patient.003.v24/meas_MID00287_FID82176_raFin_3D_tra_1x1x5mm_FULL_new_respi_coro_bart30_volumes_allbins.npy")
#volumes=np.load("./data/InVivo/3D/patient.003.v24/meas_MID00287_FID82176_raFin_3D_tra_1x1x5mm_FULL_new_respi_coro_bart30_volumes_allbins.npy")
#volumes=np.load("./data/InVivo/3D/patient.003.v24/fixedalpha/meas_MID00287_FID82176_raFin_3D_tra_1x1x5mm_FULL_new_respi_coro_bart30_volumes_allbins.npy")
#volumes=np.load("./data/InVivo/3D/patient.003.v24/movingalpha/meas_MID00287_FID82176_raFin_3D_tra_1x1x5mm_FULL_new_respi_coro_bart30_volumes_allbins.npy")
# volumes=np.load("./data/InVivo/3D/patient.003.v24/wholebin/meas_MID00287_FID82176_raFin_3D_tra_1x1x5mm_FULL_new_respi_coro_bart30_volumes_allbins.npy")
#
#volumes=np.load("./data/InVivo/3D/patient.008.v14/meas_MID00020_FID82644_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
#
#volumes=np.load("./data/InVivo/3D/patient.013.v1/meas_MID00034_FID82658_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")

#volumes=np.load("./data/InVivo/3D/patient.017.v1/meas_MID00083_FID83140_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
#volumes=np.load("./data/InVivo/3D/patient.016.v1/meas_MID00154_FID82778_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
#volumes=np.load("./data/InVivo/3D/patient.013.v1/meas_MID00034_FID82658_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_denoised_lrpatches.npy")
#volumes=np.load("./data/InVivo/3D/patient.013.v1/meas_MID00034_FID82658_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins.npy")

#volumes=np.load("/home/cslioussarenko/PythonRepositories/mrf-sim/data/InVivo/3D/patient.010.v8/meas_MID00069_FID83126_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")


#volumes=np.load("./data/InVivo/3D/patient.017.v1/meas_MID00083_FID83140_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
#volumes=np.load("./data/InVivo/3D/patient.010.v8/meas_MID00069_FID83126_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
volumes=np.load("./data/InVivo/3D/patient.017.v1/meas_MID00084_FID83141_raFin_3D_tra_1x1x5mm_FULL_new_mrf_bart30_volumes_singular_allbins.npy")
volumes=np.load("./data/InVivo/3D/patient.010.v9/meas_MID00139_FID84920_raFin_3D_tra_1x1x5mm_FULL_new_mrf_volumes_singular.npy")
volumes=np.load("./data/InVivo/3D/patient.010.v9/meas_MID00139_FID84920_raFin_3D_tra_1x1x5mm_FULL_new_mrf_volumes_singular_denoised.npy")

offset=0



flip = False
flip_z=True

from skimage.transform import resize

offset=0
if volumes.ndim==5:
    offset+=1
    volumes_resized_all=np.zeros(shape=(volumes.shape[0],volumes.shape[1],volumes.shape[2]*5,volumes.shape[3],volumes.shape[4]))

    for l in range(volumes.shape[1]):
        volumes_resized = np.zeros(shape=(volumes.shape[0], volumes.shape[2] * 5, volumes.shape[3], volumes.shape[4]))
        for ts in tqdm(range(volumes_resized.shape[0])):
            if flip:
                volumes_resized[ts]=resize(np.abs(np.flip(volumes[ts,l],axis=(1+offset,2+offset))),volumes_resized.shape[1:])
            else:
                volumes_resized[ts] = resize(np.abs(volumes[ts,l]), volumes_resized.shape[1:])
        volumes_resized_all[:,l]=volumes_resized

    volumes_resized=volumes_resized_all

else:
    volumes_resized = np.zeros(shape=(volumes.shape[0], volumes.shape[1] * 5, volumes.shape[2], volumes.shape[3]))

    for ts in tqdm(range(volumes_resized.shape[0])):
        if flip:
            volumes_resized[ts]=resize(np.abs(np.flip(volumes[ts],axis=(1+offset,2+offset))),volumes_resized.shape[1:])
        else:
            volumes_resized[ts] = resize(np.abs(volumes[ts]), volumes_resized.shape[1:])

if flip_z:
    volumes_resized=np.flip(volumes_resized,axis=(1+offset))

napari.imshow(volumes_resized)

napari.imshow(np.concatenate([volumes_resized[:-1],volumes_resized[::-1][:-1]],axis=0))










import numpy as np
import napari
from tqdm import tqdm
from mutools import io


curr_file="./data/InVivo/3D/patient.016.v1/meas_MID00154_FID82778_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
curr_file="./data/InVivo/3D/patient.017.v1/meas_MID00083_FID83140_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
curr_file="./data/InVivo/3D/patient.010.v8/meas_MID00069_FID83126_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
curr_file="./data/InVivo/3D/patient.018.v1/meas_MID00157_FID83292_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
curr_file="./data/InVivo/3D/patient.008.v14/meas_MID00020_FID82644_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
curr_file="./data/InVivo/3D/patient.009.v4/meas_MID00020_FID67345_raFin_3D_tra_1x1x5mm_FULL_new_respi_volumes_allbins_registered_allindex.npy"
curr_file="./data/InVivo/3D/patient.002.v14/meas_MID00020_FID73259_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"



volumes=np.load(curr_file)

gr=0


dz,dx,dy=5,1,1
file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_gr{}.mha".format(gr)
io.write(file_mha,np.abs(volumes[gr]),tags={"spacing":[dz,dx,dy]})





import numpy as np
import napari
from tqdm import tqdm
from mrfsim import *
from utils_mrf import *
from mutools import io
import matplotlib
import matplotlib.pyplot as plt


curr_file=r"/home/cslioussarenko/PythonRepositories/mrf-sim/data/InVivo/3D/patient.002.v17/meas_MID00020_FID88706_raFin_3D_tra_1x1x5mm_FULL_new_mrf_thighs_volumes_singular_denoised.npy"

volumes=np.real(np.load(curr_file))
dz,dx,dy=5,1,1
for sing in range(volumes.shape[0]):


    file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_real_l{}.mha".format(sing)
    io.write(file_mha,np.abs(volumes[sing]),tags={"spacing":[dz,dx,dy]})





import numpy as np
import napari
from skimage.transform import resize
from tqdm import tqdm
import glob
import pickle

def makevol(values, mask):
    """ fill volume """
    values = np.asarray(values)
    new = np.zeros(mask.shape, dtype=values.dtype)
    new[mask] = values
    return new



flip_axis = (2,)
flip_z=True

#filename_volume="meas_MID00083_FID83140_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
#filename_maps="meas_MID00084_FID83141_raFin_3D_tra_1x1x5mm_FULL_new_mrf_bart30_volumes_singular_allbins_volumes_allbins_registered_ref*_it1_CF_iterative_2Dplus1_MRF_map.pkl"




filename_volume="./data/InVivo/3D/patient.016.v1/meas_MID00154_FID82778_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
filename_maps="./data/InVivo/3D/patient.016.v1/meas_MID00155_FID82779_raFin_3D_tra_1x1x5mm_FULL_new_mrf_bart30_volumes_singular_allbins_volumes_allbins_registered_ref*_it1_CF_iterative_2Dplus1_MRF_map_matched_distrib.pkl"

filename_volume="./data/InVivo/3D/patient.017.v1/meas_MID00083_FID83140_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
filename_maps="./data/InVivo/3D/patient.017.v1/meas_MID00084_FID83141_raFin_3D_tra_1x1x5mm_FULL_new_mrf_bart30_volumes_singular_allbins_volumes_allbins_registered_ref*_it1_CF_iterative_2Dplus1_MRF_map.pkl"


#filename_volume="meas_MID00069_FID83126_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
#filename_maps="meas_MID00070_FID83127_raFin_3D_tra_1x1x5mm_FULL_new_mrf_bart30_volumes_singular_allbins_volumes_allbins_registered_ref*_it1_CF_iterative_2Dplus1_MRF_map.pkl"

#filename_volume="meas_MID00059_FID81502_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
#filename_maps="meas_MID00060_FID81503_raFin_3D_tra_1x1x5mm_FULL_new_mrf_bart30_volumes_singular_allbins_volumes_allbins_registered_ref*_it1_CF_iterative_2Dplus1_MRF_map.pkl"



#volumes=np.load("./data/InVivo/3D/DMD/meas_MID00059_FID81502_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
#volumes=np.load("./data/InVivo/3D/patient.012.v1/meas_MID00132_FID81741_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
#volumes=np.load("./data/InVivo/3D/patient.003.v24/meas_MID00287_FID82176_raFin_3D_tra_1x1x5mm_FULL_new_respi_coro_bart30_volumes_allbins_registered_allindex.npy")
file_maps=sorted(glob.glob(filename_maps))

maps_list=[]
k="wT1"
for file in file_maps:
    with open(file,"rb") as f:
        all_maps=pickle.load(f)
    map_rebuilt = all_maps[0][0]
    mask = all_maps[0][1]
    if k=="wT1":
        map_rebuilt[k][map_rebuilt["ff"]>0.7]=0
    curr_map=makevol(map_rebuilt[k],mask>0)
    maps_list.append(curr_map)

volumes=np.array(maps_list)

volumes_resized=np.zeros(shape=(volumes.shape[0],5*volumes.shape[1],volumes.shape[2],volumes.shape[3]))


for ts in tqdm(range(volumes_resized.shape[0])):
    if flip_axis is not None:
        volumes_resized[ts] = resize(np.flip(np.abs(volumes[ts]),axis=flip_axis), volumes_resized.shape[1:])
    else:
        volumes_resized[ts]=resize(np.abs(volumes[ts]),volumes_resized.shape[1:])

if flip_z:
    volumes_resized=np.flip(volumes_resized,axis=1)

if volumes_resized.shape[0]==1:
    volumes_resized_wT1=volumes_resized
else:
    volumes_resized_wT1=np.concatenate([volumes_resized[:-1],volumes_resized[::-1][:-1]],axis=0)
viewer=napari.Viewer()
viewer.add_image(volumes_resized_wT1)

maps_list=[]
k="ff"
for file in file_maps:
    with open(file,"rb") as f:
        all_maps=pickle.load(f)
    map_rebuilt = all_maps[0][0]
    mask = all_maps[0][1]
    curr_map=makevol(map_rebuilt[k],mask>0)
    maps_list.append(curr_map)

volumes=np.array(maps_list)

volumes_resized=np.zeros(shape=(volumes.shape[0],5*volumes.shape[1],volumes.shape[2],volumes.shape[3]))


for ts in tqdm(range(volumes_resized.shape[0])):
    if flip_axis is not None:
        volumes_resized[ts] = resize(np.flip(np.abs(volumes[ts]),axis=flip_axis), volumes_resized.shape[1:])
    else:
        volumes_resized[ts]=resize(np.abs(volumes[ts]),volumes_resized.shape[1:])

if flip_z:
    volumes_resized=np.flip(volumes_resized,axis=1)


if volumes_resized.shape[0]==1:
    volumes_resized_ff=volumes_resized
else:
    volumes_resized_ff=np.concatenate([volumes_resized[:-1],volumes_resized[::-1][:-1]],axis=0)
viewer.add_image(volumes_resized_ff)




nbins=volumes_resized.shape[0]
volumes=np.load(filename_volume)
volumes=volumes[:nbins]
volumes_resized=np.zeros(shape=(volumes.shape[0],5*volumes.shape[1],volumes.shape[2],volumes.shape[3]))

for ts in tqdm(range(nbins)):
    if flip_axis is not None:
        volumes_resized[ts] = resize(np.flip(np.abs(volumes[ts]),axis=flip_axis), volumes_resized.shape[1:])
    else:
        volumes_resized[ts]=resize(np.abs(volumes[ts]),volumes_resized.shape[1:])

if flip_z:
    volumes_resized=np.flip(volumes_resized,axis=1)

if volumes_resized.shape[0] == 1:
    volumes_resized = volumes_resized
else:
    volumes_resized=np.concatenate([volumes_resized[:-1],volumes_resized[::-1][:-1]],axis=0)

viewer.add_image(volumes_resized)










import numpy as np
import napari
from skimage.transform import resize
from tqdm import tqdm
import glob
import pickle

def makevol(values, mask):
    """ fill volume """
    values = np.asarray(values)
    new = np.zeros(mask.shape, dtype=values.dtype)
    new[mask] = values
    return new



flip_axis = (1,2)
flip_z=False

#filename_volume="meas_MID00083_FID83140_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
#filename_maps="meas_MID00084_FID83141_raFin_3D_tra_1x1x5mm_FULL_new_mrf_bart30_volumes_singular_allbins_volumes_allbins_registered_ref*_it1_CF_iterative_2Dplus1_MRF_map.pkl"





#filename_volume="meas_MID00069_FID83126_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
#filename_maps="meas_MID00070_FID83127_raFin_3D_tra_1x1x5mm_FULL_new_mrf_bart30_volumes_singular_allbins_volumes_allbins_registered_ref*_it1_CF_iterative_2Dplus1_MRF_map.pkl"

#filename_volume="meas_MID00059_FID81502_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy"
#filename_maps="meas_MID00060_FID81503_raFin_3D_tra_1x1x5mm_FULL_new_mrf_bart30_volumes_singular_allbins_volumes_allbins_registered_ref*_it1_CF_iterative_2Dplus1_MRF_map.pkl"



#volumes=np.load("./data/InVivo/3D/DMD/meas_MID00059_FID81502_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
#volumes=np.load("./data/InVivo/3D/patient.012.v1/meas_MID00132_FID81741_raFin_3D_tra_1x1x5mm_FULL_new_respi_bart30_volumes_allbins_registered_allindex.npy")
#volumes=np.load("./data/InVivo/3D/patient.003.v24/meas_MID00287_FID82176_raFin_3D_tra_1x1x5mm_FULL_new_respi_coro_bart30_volumes_allbins_registered_allindex.npy")

file_maps=[
    "./data/InVivo/3D/patient.009.v6/meas_MID00114_FID84523_raFin_3D_tra_1x1x5mm_FULL_new_mrf_legs_volumes_singular_denoised_CF_iterative_2Dplus1_MRF_map.pkl",
    "./data/InVivo/3D/patient.009.v6/meas_MID00101_FID84510_raFin_3D_tra_1x1x5mm_FULL_new_mrf_thighs_volumes_singular_denoised_CF_iterative_2Dplus1_MRF_map.pkl",
    "./data/InVivo/3D/patient.009.v6/meas_MID00083_FID84492_raFin_3D_tra_1x1x5mm_FULL_new_mrf_bart30_volumes_singular_allbins_volumes_allbins_registered_ref4_it1_CF_iterative_2Dplus1_MRF_map.pkl"

           ]
k="wT1"
maps_list=[]
for i,file in enumerate(file_maps):
    with open(file,"rb") as f:
        all_maps=pickle.load(f)
    map_rebuilt = all_maps[0][0]
    mask = all_maps[0][1]
    if k=="wT1":
        map_rebuilt[k][map_rebuilt["ff"]>0.7]=0
    curr_map=makevol(map_rebuilt[k],mask>0)
    maps_list.append(curr_map)

volumes=np.array(maps_list)
volumes=np.concatenate(volumes,axis=1)
if volumes.ndim==3:
    volumes=np.expand_dims(volumes,axis=0)


volumes_resized=np.zeros(shape=(volumes.shape[0],5*volumes.shape[1],volumes.shape[2],volumes.shape[3]))


for ts in tqdm(range(volumes_resized.shape[0])):
    if flip_axis is not None:
        volumes_resized[ts] = resize(np.flip(np.abs(volumes[ts]),axis=flip_axis), volumes_resized.shape[1:])
    else:
        volumes_resized[ts]=resize(np.abs(volumes[ts]),volumes_resized.shape[1:])

if flip_z:
    volumes_resized=np.flip(volumes_resized,axis=1)

if volumes_resized.shape[0]==1:
    volumes_resized_wT1=volumes_resized
else:
    volumes_resized_wT1=np.concatenate([volumes_resized[:-1],volumes_resized[::-1][:-1]],axis=0)
viewer=napari.Viewer()
viewer.add_image(volumes_resized_wT1)

maps_list=[]
k="ff"
maps_list=[]
for file in file_maps:
    with open(file,"rb") as f:
        all_maps=pickle.load(f)
    map_rebuilt = all_maps[0][0]
    mask = all_maps[0][1]
    if k=="wT1":
        map_rebuilt[k][map_rebuilt["ff"]>0.7]=0
    curr_map=makevol(map_rebuilt[k],mask>0)
    maps_list.append(curr_map)

volumes=np.array(maps_list)
volumes=np.concatenate(volumes,axis=1)
if volumes.ndim==3:
    volumes=np.expand_dims(volumes,axis=0)

volumes_resized=np.zeros(shape=(volumes.shape[0],5*volumes.shape[1],volumes.shape[2],volumes.shape[3]))


for ts in tqdm(range(volumes_resized.shape[0])):
    if flip_axis is not None:
        volumes_resized[ts] = resize(np.flip(np.abs(volumes[ts]),axis=flip_axis), volumes_resized.shape[1:])
    else:
        volumes_resized[ts]=resize(np.abs(volumes[ts]),volumes_resized.shape[1:])

if flip_z:
    volumes_resized=np.flip(volumes_resized,axis=1)

if volumes_resized.shape[0]==1:
    volumes_resized_ff=volumes_resized
else:
    volumes_resized_ff=np.concatenate([volumes_resized[:-1],volumes_resized[::-1][:-1]],axis=0)
viewer.add_image(volumes_resized_ff)

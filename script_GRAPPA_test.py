
#import matplotlib
#matplotlib.u<se("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,GaussianWeighting
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

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./2D"

dictfile = "mrf175_SimReco2_light.dict"
dictjson="mrf_dictconf_SimReco2_light.json"
#dictfile = "mrf175_SimReco2.dict"
#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
#dictfile = "mrf175_Dico2_Invivo.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)

seq = T1MRF(**sequence_config)

name = "SquareSimu3D"

use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

folder = "/".join(str.split(filename,"/")[:-1])

suffix="_cartesian"

filename_paramMap=filename+"_paramMap.pkl"
filename_volume = filename+"_volumes{}.npy".format(suffix)
filename_groundtruth = filename+"_groundtruth_volumes{}.npy".format("")

filename_kdata = filename+"_kdata{}.npy".format(suffix)
filename_mask= filename+"_mask{}.npy".format(suffix)
file_map = filename + "{}_MRF_map.pkl".format(suffix)


#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint_x = 256


undersampling_factor=1

incoherent=True
mode="old"



with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
mask_reduction_factor=1/4



image_size = (int(npoint_x), int(npoint_x))
nb_segments=nb_allspokes
nspoke=int(nb_segments/ntimesteps)


m = RandomMap(name,dict_config,resting_time=4000,image_size=image_size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)



if str.split(filename_paramMap,"/")[-1] not in os.listdir(folder):
    m.buildParamMap()
    with open(filename_paramMap, "wb" ) as file:
        pickle.dump(m.paramMap, file)

    map_rebuilt = m.paramMap
    mask = m.mask

    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    for key in ["ff", "wT1", "df", "attB1"]:
        file_mha = "/".join(["/".join(str.split(filename_paramMap, "/")[:-1]),
                             "_".join(str.split(str.split(filename_paramMap, "/")[-1], ".")[:-1])]) + "_{}.mha".format(
                                                                                                                   key)
        io.write(file_mha, map_for_sim[key], tags={"spacing": [5, 1, 1]})

else:
    with open(filename_paramMap, "rb") as file:
        m.paramMap=pickle.load(file)


# file_map_matlab = filename+"_paramMap_sl{}_rp{}.mat".format(nb_slices,repeat_slice)
# file_mask_matlab= filename+"_mask_sl{}_rp{}.mat".format(nb_slices,repeat_slice)
#
# savemat(file_map_matlab,m.paramMap)
# savemat(file_mask_matlab,{"Mask":m.mask})


m.build_ref_images(seq)

if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
    np.save(filename_groundtruth,m.images_series[::nspoke])

i=0
image=m.images_series[i]

npoint_y = 128

curr_traj=cartesian_traj_2D(npoint_x,npoint_y)

kdata = finufft.nufft2d2(curr_traj[:,0], curr_traj[:,1], image)



rebuilt_image=finufft.nufft2d1(curr_traj[:,0], curr_traj[:,1], kdata, image_size)

plt.imshow(np.abs(rebuilt_image))

nb_channels=4
sig=50**2

nb_means=int(np.sqrt(nb_channels))

means_x=np.arange(1,nb_means+1)*(1/(nb_means+1))*image_size[0]
means_y=np.arange(1,nb_means+1)*(1/(nb_means+1))*image_size[1]

x = np.arange(image_size[0])
y = np.arange(image_size[1])

X,Y = np.meshgrid(x,y)
pixels=np.stack([X.flatten(), Y.flatten()], axis=-1)

from scipy.stats import multivariate_normal
b1_maps=[]
for mu_x in means_x:
    for mu_y in means_y:

        b1_maps.append(multivariate_normal.pdf(pixels, mean=[mu_x,mu_y], cov=sig*np.eye(2)))


b1_maps = np.array(b1_maps)
b1_maps=b1_maps/np.expand_dims(np.max(b1_maps,axis=-1),axis=-1)
b1_maps=b1_maps.reshape((nb_channels,)+image_size)


for ch in range(nb_channels):
    plt.figure()
    plt.imshow(b1_maps[ch])
    plt.colorbar()


image_all_channels=np.array([image*b1 for b1 in b1_maps])

kdata_all_channels=[finufft.nufft2d2(curr_traj[:,0], curr_traj[:,1], p) for p in image_all_channels]

image_rebuilt_all_channels=[finufft.nufft2d1(curr_traj[:,0], curr_traj[:,1], k, image_size) for k in kdata_all_channels]

for ch in range(nb_channels):
    plt.figure()
    plt.imshow(np.abs(image_rebuilt_all_channels[ch]))



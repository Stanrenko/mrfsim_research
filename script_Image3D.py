
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from utils_mrf import radial_golden_angle_traj,animate_images,animate_multiple_images,compare_patterns,translation_breathing,find_klargest_freq,SearchMrf,basicDictSearch,compare_paramMaps,regression_paramMaps,dictSearchMemoryOptim,voronoi_volumes,transform_py_map,radial_golden_angle_traj_3D
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np


## Random map simulation

dictfile = "mrf175.dict"
#dictfile = "mrf175_CS.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

window = 8 #corresponds to nspoke by image
size=(256,256)

file_matlab_paramMap = "./data/paramMap.mat"

###### Building Map
m = MapFromFile3D("TestPhantomV1",nb_slices=4,nb_empty_slices=4,file=file_matlab_paramMap,rounding=True)
m.buildParamMap()

#m.plotParamMap("wT1")

##### Simulating Ref Images
m.build_ref_images(seq,window)


undersampling_factor=4
npoint=512
nspoke=8
total_nspoke=nspoke*175
nb_slices=m.paramDict["nb_slices"]+2*m.paramDict["nb_empty_slices"]
size=m.image_size
images_series=m.images_series
nb_rep=int(nb_slices/undersampling_factor)

#all_spokes_one_rep=np.reshape(all_spokes,(175,undersampling_factor,-1))
#traj=np.repeat(all_spokes_one_rep,nb_rep,axis=1)



traj_3D=radial_golden_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4)

images_series_rebuilt=m.simulate_radial_undersampled_images_3D(traj_3D,nb_rep,nspoke,density_adj=True,npoint=npoint)


for i in [0,4,8,11]:
    plt.figure()
    plt.imshow(np.abs(images_series_rebuilt[0][i,:,:]))
    plt.colorbar()

for i in [0,4,8,11]:
    plt.figure()
    plt.imshow(np.abs(images_series[0][i,:,:]))
    plt.colorbar()

plt.figure()
plt.imshow(np.abs(images_series_rebuilt[0][5,:,:]-images_series_rebuilt[0][0,:,:]))
plt.colorbar()

slice=0
image_slice=list(np.array(images_series)[:,slice,:,:])
image_slice_rebuilt=list(np.array(images_series_rebuilt)[:,slice,:,:])
ani1,ani2=animate_multiple_images(image_slice,image_slice_rebuilt)


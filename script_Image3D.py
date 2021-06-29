
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from utils_mrf import radial_golden_angle_traj,animate_images,animate_multiple_images,compare_patterns,translation_breathing,find_klargest_freq,SearchMrf,basicDictSearch,compare_paramMaps,compare_paramMaps_3D,dictSearchMemoryOptim,voronoi_volumes,transform_py_map,radial_golden_angle_traj_3D
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np


## Random map simulation

dictfile = "mrf175.dict"
dictfile = "mrf175_CS.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

window = 8 #corresponds to nspoke by image
size=(256,256)

file_matlab_paramMap = "./data/paramMap.mat"

###### Building Map
#m = MapFromFile3D("TestPhantomV1",nb_slices=2,nb_empty_slices=3,file=file_matlab_paramMap,rounding=True)

with open("mrf_dictconf_CS.json") as f:
    dict_config = json.load(f)

dict_config["ff"]=np.arange(0.,1.05,0.05)
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
size=(256,256)
mask_reduction_factor=1/4
m = RandomMap3D("TestRandom3D",dict_config,nb_slices=4,nb_empty_slices=6,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor)

m.buildParamMap()

##### Simulating Ref Images
m.build_ref_images(seq,window)


undersampling_factor=4
npoint=512
nspoke=8
ntimesteps=175

nb_slices=m.paramDict["nb_total_slices"]

radial_traj_3D=Radial3D(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint,nb_slices=nb_slices,undersampling_factor=undersampling_factor)

all_maps_adj=m.dictSearchMemoryOptimIterative(dictfile,seq,radial_traj_3D,niter=1,split=500,threshold_pca=15,log=False,useAdjPred=True,true_mask=False)
compare_paramMaps_3D(m.paramMap,all_maps_adj[1][0],m.mask>0,all_maps_adj[1][1]>0,slice=5,title1="Orig",title2="Slice 5",proj_on_mask1=False)
compare_paramMaps_3D(m.paramMap,all_maps_adj[1][0],m.mask>0,all_maps_adj[1][1]>0,slice=6,title1="Orig",title2="Slice 6",proj_on_mask1=False)
compare_paramMaps_3D(m.paramMap,all_maps_adj[1][0],m.mask>0,all_maps_adj[1][1]>0,slice=8,title1="Orig",title2="Slice 8",proj_on_mask1=False)

plt.close("all")


direction = np.array([0,4,0])
shifts_t = lambda t:translation_breathing(t,direction)
m.translate_images(shifts_t,round=True)

#ani = animate_images(m.images_series[:,m.paramDict["nb_empty_slices"],:,:])

all_maps_adj_mvt=m.dictSearchMemoryOptimIterative(dictfile,seq,radial_traj_3D,niter=1,split=500,threshold_pca=15,log=False,useAdjPred=True,true_mask=False)

compare_paramMaps_3D(m.paramMap,all_maps_adj_mvt[1][0],m.mask>0,all_maps_adj_mvt[1][1]>0,slice=5,title1="Orig",title2="OutsideMvt",proj_on_mask1=False,save=True)
compare_paramMaps_3D(m.paramMap,all_maps_adj_mvt[1][0],m.mask>0,all_maps_adj_mvt[1][1]>0,slice=6,title1="Orig",title2="InsideMvt",proj_on_mask1=False,save=True)
compare_paramMaps_3D(m.paramMap,all_maps_adj_mvt[1][0],m.mask>0,all_maps_adj_mvt[1][1]>0,slice=8,title1="Orig",title2="CenterMvt",proj_on_mask1=False,save=True)


# for i in [0,m.paramDict["nb_empty_slices"]]:
#     plt.figure()
#     plt.imshow(np.abs(images_series_rebuilt[0][i,:,:]))
#     plt.colorbar()
#
# for i in [0,m.paramDict["nb_empty_slices"]]:
#     plt.figure()
#     plt.imshow(np.abs(images_series[0][i,:,:]))
#     plt.colorbar()
#
# plt.figure()
# plt.imshow(np.abs(images_series_rebuilt[0][5,:,:]-images_series_rebuilt[0][0,:,:]))
# plt.colorbar()
#
# slice=7
# image_slice=list(np.array(images_series)[:,slice,:,:])
# image_slice_rebuilt=list(np.array(images_series_rebuilt)[:,slice,:,:])
# image_slice_rebuilt_border=list(np.array(images_series_rebuilt)[:,0,:,:])
# ani1,ani2=animate_multiple_images(image_slice,image_slice_rebuilt)
#
# ani1,ani2=animate_multiple_images(image_slice_rebuilt_border,image_slice_rebuilt)
#


#
# pixel=(125,125)
# time=0
#
# rebuilt_pixel_z_profile =  np.abs(images_series_rebuilt[time][:,pixel[0],pixel[1]])
# orig_pixel_z_profile =  np.abs(images_series[time][:,pixel[0],pixel[1]])
# rebuilt_pixel_z_profile =rebuilt_pixel_z_profile /np.max(rebuilt_pixel_z_profile)
# orig_pixel_z_profile =orig_pixel_z_profile /np.max(orig_pixel_z_profile)
# ymin=0
# ymax=np.maximum(np.max(rebuilt_pixel_z_profile),np.max(rebuilt_pixel_z_profile))*(1+0.05)
#
# plt.figure()
# plt.plot(rebuilt_pixel_z_profile,label="Rebuilt along z axis pixel {}".format(pixel),color="blue")
# plt.plot(orig_pixel_z_profile,label="Original along z axis pixel {}".format(pixel),color="green")
# plt.vlines(x=m.paramDict["nb_empty_slices"]-1,ymin=ymin,ymax=ymax,color="red")
# plt.vlines(x=m.paramDict["nb_total_slices"]-m.paramDict["nb_empty_slices"],ymin=ymin,ymax=ymax,color="red")
# plt.legend()
#

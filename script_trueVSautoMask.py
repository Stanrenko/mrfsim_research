
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from utils_mrf import radial_golden_angle_traj,animate_images,animate_multiple_images,compare_patterns,translation_breathing,find_klargest_freq,SearchMrf,basicDictSearch,compare_paramMaps,regression_paramMaps,dictSearchMemoryOptim,voronoi_volumes,transform_py_map
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

with open("mrf_dictconf.json") as f:
    dict_config = json.load(f)

dict_config["ff"]=np.arange(0.,1.05,0.05)

window = 8 #corresponds to nspoke by image
size=(256,256)

file_matlab_paramMap = "./data/paramMap.mat"

###### Building Map
m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)
m.buildParamMap()

##### Simulating Ref Images
m.build_ref_images(seq,window)

npoint = 2*m.images_series.shape[1]
total_nspoke=8*175
nspoke=8

all_spokes=radial_golden_angle_traj(total_nspoke,npoint)
traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))

all_maps_auto_mask=m.dictSearchMemoryOptimIterative(dictfile,seq,traj,npoint,niter=1,split=500,threshold_pca=15,log=False,useAdjPred=False,true_mask=False)
all_maps_true_mask=m.dictSearchMemoryOptimIterative(dictfile,seq,traj,npoint,niter=1,split=500,threshold_pca=15,log=False,useAdjPred=False,true_mask=True)



regression_paramMaps(m.paramMap,all_maps_true_mask[1][0],m.mask>0,all_maps_true_mask[1][1]>0,title="Orig vs True Mask",adj_wT1=True,fat_threshold=0.7)
regression_paramMaps(m.paramMap,all_maps_auto_mask[1][0],m.mask>0,all_maps_auto_mask[1][1]>0,title="Orig vs Auto Mask",adj_wT1=True,fat_threshold=0.7)

compare_paramMaps(m.paramMap,all_maps_true_mask[1][0],m.mask>0,all_maps_true_mask[1][1]>0,title1="Orig",title2="True Mask",adj_wT1=True,fat_threshold=0.7,proj_on_mask1=True)
compare_paramMaps(m.paramMap,all_maps_auto_mask[1][0],m.mask>0,all_maps_auto_mask[1][1]>0,title1="Orig",title2="Auto Mask",adj_wT1=True,fat_threshold=0.7,proj_on_mask1=True)


regression_paramMaps(m.paramMap,all_maps_auto_mask[1][0],m.mask>0,all_maps_auto_mask[1][1]>0,title="Orig vs Auto Mask",adj_wT1=True,fat_threshold=0.7,mode="Boxplot")

regression_paramMaps(m.paramMap,all_maps_auto_mask[1][0],m.mask>0,all_maps_auto_mask[1][1]>0,title="Orig vs Auto Mask",adj_wT1=True,fat_threshold=0.7,mode="Boxplot",proj_on_mask1=True)


#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from trajectory import *
from utils_mrf import radial_golden_angle_traj,animate_images,animate_multiple_images,compare_patterns,translation_breathing,find_klargest_freq,SearchMrf,basicDictSearch,compare_paramMaps,regression_paramMaps,dictSearchMemoryOptim,voronoi_volumes,transform_py_map
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np




ntimesteps=175
nspoke=8
npoint=512

radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint)
radial_traj.get_traj()


us=4

radial_traj_3D=Radial3D(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint,nb_slices=16,undersampling_factor=us)
radial_traj_3D.get_traj()

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
m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)
m.buildParamMap()

##### Simulating Ref Images
m.build_ref_images(seq,window)

all_maps_adj=m.dictSearchMemoryOptimIterative(dictfile,seq,radial_traj,niter=1,split=500,threshold_pca=15,log=False,useAdjPred=True,true_mask=False)
regression_paramMaps(m.paramMap,all_maps_adj[1][0],m.mask>0,all_maps_adj[1][1]>0,title="Orig vs Adjusted Iterative",proj_on_mask1=True)





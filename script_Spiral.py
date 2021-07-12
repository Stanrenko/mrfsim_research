
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
from movements import *
from dictoptimizers import *


## Random map simulation

useGPU=True

dictfile = "mrf175.dict"
#dictfile = "mrf175_CS.dict"
dictfile = "mrf175_SimReco2.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

window = 8 #corresponds to nspoke by image
size=(256,256)

file_matlab_paramMap = "./data/Phantom1/paramMap.mat"

###### Building Map
#m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)

with open("mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

dict_config["ff"]=np.arange(0.,1.05,0.05)

window = 8 #corresponds to nspoke by image
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
size=(256,256)

###### Building Map
m = RandomMap("TestRandom",dict_config,image_size=size,region_size=region_size,mask_reduction_factor=1/4)

m.buildParamMap()

#m.plotParamMap(save=True)

##### Simulating Ref Images
m.build_ref_images(seq,window)

ntimesteps=175
nspoke=8
npoint = 2*m.images_series.shape[1]

spiral_traj=VariableSpiral(ntimesteps=ntimesteps,nspiral=nspoke,npoint=256,alpha=4,spatial_us=7,temporal_us=2)


kdata = m.generate_kdata(spiral_traj,useGPU=useGPU)
volumes = simulate_undersampled_images(kdata,spiral_traj,m.image_size,useGPU=useGPU,density_adj=True)

ani = animate_images(volumes)

radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=512)
kdata_radial = m.generate_kdata(radial_traj,useGPU=useGPU)
volumes_radial = simulate_radial_undersampled_images(kdata_radial,radial_traj,m.image_size,useGPU=useGPU,density_adj=True)
#
ani = animate_images(volumes_radial)

#TO DO - mask for generic traj
#mask = build_mask_single_image(kdata,radial_traj,m.image_size)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream

optimizer = SimpleDictSearch(mask=m.mask,niter=0,seq=seq,trajectory=spiral_traj,split=500,pca=True,threshold_pca=15,log=False,useAdjPred=False)
all_maps_adj_radial=optimizer.search_patterns(dictfile,volumes_radial)

plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_adj_radial.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_adj_radial[iter][0], m.mask > 0, all_maps_adj_radial[iter][1] > 0,maskROI=maskROI,
                             title="ROI Orig vs Iteration {}".format(iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7)

plot_evolution_params(m.paramMap,m.mask>0,all_maps_adj)
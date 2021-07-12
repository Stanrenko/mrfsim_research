
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
import pickle
from dictoptimizers import SimpleDictSearch


## Random map simulation

start = datetime.now()

dictfile = "mrf175.dict"
dictfile = "mrf175_CS.dict"
#dictfile = "mrf175_SimReco2.dict"


with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

window = 8 #corresponds to nspoke by image
size=(256,256)

file_matlab_paramMap = "./data/KneePhantom/Phantom1/paramMap.mat"

###### Building Map
#m = MapFromFile3D("TestPhantomV1",nb_slices=64,nb_empty_slices=8,file=file_matlab_paramMap,rounding=True)

with open("mrf_dictconf_CS.json") as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
size=(256,256)
mask_reduction_factor=1/4
m = RandomMap3D("TestRandom3DMovement",dict_config,nb_slices=64,nb_empty_slices=8,undersampling_factor=4,repeat_slice=8,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor)

m.buildParamMap()

m.plotParamMap("wT1")

##### Simulating Ref Images
m.build_ref_images(seq,window)

direction=np.array([0.0,4.0,0.0])
move = TranslationBreathing(direction,T=4000,frac_exp=0.7)

m.add_movements([move])

npoint=512
nspoke=8
ntimesteps=175

nb_slices=m.paramDict["nb_total_slices"]
undersampling_factor = m.paramDict["undersampling_factor"]

radial_traj_3D=Radial3D(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint,nb_slices=nb_slices,undersampling_factor=undersampling_factor)
kdata = m.generate_radial_kdata(radial_traj_3D,useGPU=True)
#kdata_noGPU=m.generate_radial_kdata(radial_traj_3D,useGPU=False)
volumes = simulate_radial_undersampled_images(kdata,radial_traj_3D,m.image_size,density_adj=True,useGPU=True)
#volumes_noGPU = simulate_radial_undersampled_images(kdata_noGPU,radial_traj_3D,m.image_size,density_adj=True,useGPU=True)
#ani,ani1=animate_multiple_images(volumes[:,4,:,:],volumes_noGPU[:,4,:,:])

#ani=animate_images(volumes[:,8,:,:])

mask = build_mask_single_image(kdata,radial_traj_3D,m.image_size,useGPU=True)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream
plt.imshow(mask[m.paramDict["nb_empty_slices"]+int(m.paramDict["nb_slices"]/2),:,:])

plt.imshow(mask[m.paramDict["nb_empty_slices"]-5,:,:])




optimizer = SimpleDictSearch(mask=mask,niter=4,seq=seq,trajectory=radial_traj_3D,split=1000,pca=True,threshold_pca=15,useGPU=True,log=False,useAdjPred=False)
all_maps_adj=optimizer.search_patterns(dictfile,volumes)

end=datetime.now()
print(end-start)

file = open( "all_maps_{}.pkl".format(m.name), "wb" )
# dump information to that file
pickle.dump(all_maps_adj, file)
# close the file
file.close()

plt.close("all")

maskROI=buildROImask(m.paramMap)

for iter in all_maps_adj.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_adj[iter][0], m.mask > 0, all_maps_adj[iter][1] > 0,maskROI=maskROI,
                             title="ROI Orig vs Iteration {}".format(iter), proj_on_mask1=False, adj_wT1=True, fat_threshold=0.7)



compare_paramMaps_3D(m.paramMap,all_maps_adj[2][0],m.mask>0,all_maps_adj[2][1]>0,slice=m.paramDict["nb_empty_slices"]-1,title1="Orig",title2="Outside",proj_on_mask1=True,save=False)
compare_paramMaps_3D(m.paramMap,all_maps_adj[2][0],m.mask>0,all_maps_adj[2][1]>0,slice=m.paramDict["nb_empty_slices"]+5,title1="Orig",title2="Inside",proj_on_mask1=True,save=False)
compare_paramMaps_3D(m.paramMap,all_maps_adj[2][0],m.mask>0,all_maps_adj[2][1]>0,slice=m.paramDict["nb_empty_slices"]+int(m.paramDict["nb_slices"]/2),title1="Orig",title2="Center",proj_on_mask1=True,save=False)

size_slice = int(m.paramDict["nb_slices"]/m.paramDict["repeat_slice"])

compare_paramMaps_3D(m.paramMap,all_maps_adj[2][0],m.mask>0,all_maps_adj[2][1]>0,slice=m.paramDict["nb_empty_slices"]+int(size_slice/2),title1="Orig",title2="Center",proj_on_mask1=True,save=False)
compare_paramMaps_3D(m.paramMap,all_maps_adj[2][0],m.mask>0,all_maps_adj[2][1]>0,slice=m.paramDict["nb_empty_slices"]+4*int(size_slice/2),title1="Orig",title2="Center",proj_on_mask1=True,save=False)

regression_paramMaps_ROI(m.paramMap, all_maps_adj[4][0], m.mask > 0, all_maps_adj[4][1] > 0,maskROI=maskROI,
                             title="ROI Orig vs Iteration {}".format(4), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7)

plt.close("all")


import pycuda.autoinit
import pycuda.gpuarray as cua
import pycuda.tools as cut
import numpy as np

m = cut.DeviceMemoryPool()

a= np.ones(2**30-1,dtype=np.float32)
b= cua.to_gpu(a, allocator=m.allocate)  # Passes

a= np.ones(2**30,dtype=np.float32)
b= cua.to_gpu(a, allocator=m.allocate)
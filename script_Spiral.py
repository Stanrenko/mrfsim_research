
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
dictfile = "mrf175_SimReco2_window_1.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

window = 1 #corresponds to nspoke by image
size=(256,256)

file_matlab_paramMap = "./data/KneePhantom/Phantom1/paramMap.mat"

###### Building Map
#m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)

with open("mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

dict_config["ff"]=np.arange(0.,1.05,0.05)

window = 1 #corresponds to nspoke by image
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
size=(256,256)

###### Building Map
m = RandomMap("TestRandom",dict_config,image_size=size,region_size=region_size,mask_reduction_factor=1/4)
m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)

m.buildParamMap()

#m.plotParamMap(save=True)

##### Simulating Ref Images
m.build_ref_images(seq,window)

ntimesteps=1400
nspoke=1
npoint = 2*m.images_series.shape[1]

spiral_traj=VariableSpiral(ntimesteps=ntimesteps,nspiral=nspoke,npoint=256,ninterleaves=1,alpha=8,spatial_us=6,temporal_us=1)


kdata = m.generate_kdata(spiral_traj,useGPU=useGPU)
volumes = simulate_undersampled_images(kdata,spiral_traj,m.image_size,useGPU=useGPU,density_adj=True)

# plt.scatter(spiral_traj.traj[0,:,0],spiral_traj.traj[0,:,1])
# plt.scatter(kdata[0].real,kdata[0].imag)
# tx,ty=np.meshgrid(spiral_traj.traj[0,:,0],spiral_traj.traj[0,:,1])
plt.figure()
plt.tricontourf(spiral_traj.traj[0,:,0],spiral_traj.traj[0,:,1],np.abs(kdata[0]),levels=100)
plt.colorbar()

ani = animate_images(volumes)

plt.figure()
plt.imshow(np.abs(volumes[0]))


radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=512)
kdata_radial = m.generate_kdata(radial_traj,useGPU=useGPU)
volumes_radial = simulate_radial_undersampled_images(kdata_radial,radial_traj,m.image_size,useGPU=useGPU,density_adj=True)
#
ani = animate_images(volumes_radial)

plt.figure()
plt.imshow(np.abs(volumes_radial[0]))

plt.scatter(radial_traj.traj[0,:,0],radial_traj.traj[0,:,1])

plt.figure()
plt.tricontourf(radial_traj.traj[0,:,0],radial_traj.traj[0,:,1],np.abs(kdata_radial[0]),levels=100)
plt.colorbar()

plt.figure()
plt.plot(np.abs(kdata_radial[0]))


from scipy.interpolate import griddata
# Create grid values first.
xi = np.linspace(-np.pi, np.pi, 1000)
yi = np.linspace(-np.pi, np.pi, 1000)

# Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
t = 0
x= spiral_traj.traj[t,:,0]
y= spiral_traj.traj[t,:,1]
z= kdata[t]

zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear',fill_value=0)


x= radial_traj.traj[t,:,0]
y= radial_traj.traj[t,:,1]
z= kdata_radial[t]

zi_radial = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear',fill_value=0)

plt.figure()
plt.plot(np.abs(zi[500,:]),label="spiral")
plt.plot(np.abs(zi_radial[500,:]),label="radial")
plt.legend()


plt.figure()
plt.plot(np.abs(zi[500,:])-np.abs(zi_radial[500,:]),label="diff")
plt.legend()


plt.figure()
plt.imshow(np.abs(zi)-np.abs(zi_radial))
plt.colorbar()



plt.figure()
plt.plot(zi[0,:])



#TO DO - mask for generic traj
#mask = build_mask_single_image(kdata,radial_traj,m.image_size)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream

optimizer = SimpleDictSearch(mask=m.mask,niter=0,seq=seq,trajectory=spiral_traj,split=500,pca=True,threshold_pca=15,log=False,useAdjPred=False)
all_maps_adj=optimizer.search_patterns(dictfile,volumes)

plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_adj_radial.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_adj[iter][0], m.mask > 0, all_maps_adj[iter][1] > 0,maskROI=maskROI,
                             title="Spiral ROI Orig vs Iteration {}".format(iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7)
it=0
compare_paramMaps(m.paramMap,all_maps_adj[it][0],m.mask>0,all_maps_adj[it][1]>0,adj_wT1=True,fat_threshold=0.7,title1="Orig",title2="Spiral Rebuilt It {}".format(it),figsize=(30,10),fontsize=15,save=True,proj_on_mask1=True)

plot_evolution_params(m.paramMap,m.mask>0,all_maps_adj)
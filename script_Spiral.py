
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

useGPU=False

dictfile = "mrf175.dict"
#dictfile = "mrf175_CS.dict"
dictfile = "mrf175_SimReco2.dict"
#dictfile = "mrf175_SimReco2_window_1.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

window = 1 #corresponds to nspoke by image
size=(256,256)

type="KneePhantom"
num =1

file_matlab_paramMap = "./data/{}/Phantom{}/paramMap.mat".format(type,num)

###### Building Map
#m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)

with open("mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

dict_config["ff"]=np.arange(0.,1.05,0.05)

region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
size=(256,256)

###### Building Map
#m = RandomMap("TestRandom",dict_config,image_size=size,region_size=region_size,mask_reduction_factor=1/4)
m = MapFromFile("Map{}{}".format(type,num),image_size=size,file=file_matlab_paramMap,rounding=True,gen_mode="other")

m.buildParamMap()
m.build_ref_images(seq)

ntimesteps=175
nspoke=8
npoint = 2*m.images_series.shape[1]



spatial_us_list = [8,16,32]
temporal_us_list = [0.1,0.25,0.5,1.0]

maskROI=buildROImask_unique(m.paramMap)
optimizer = SimpleDictSearch(mask=m.mask,niter=0,seq=None,trajectory=None,split=1000,pca=True,threshold_pca=15,log=False,useAdjPred=False)

# spiral_traj = VariableSpiral(ntimesteps=ntimesteps, nspiral=nspoke, npoint=256, ninterleaves=1, alpha=128,
#                                      spatial_us=8, temporal_us=1)
#
# kdata = m.generate_kdata(spiral_traj, useGPU=useGPU)
# np.array(kdata).shape

df_python=pd.DataFrame()
for sp in spatial_us_list:
    for tp in temporal_us_list:
        spiral_traj = VariableSpiral(ntimesteps=ntimesteps, nspiral=nspoke, npoint=256, ninterleaves=1, alpha=128,
                                     spatial_us=sp, temporal_us=tp)

        kdata = m.generate_kdata(spiral_traj, useGPU=useGPU)
        volumes = simulate_undersampled_images(kdata, spiral_traj, m.image_size, useGPU=useGPU, density_adj=True)
        all_maps_adj = optimizer.search_patterns(dictfile, volumes)

        df_current = metrics_paramMaps_ROI(m.paramMap, all_maps_adj[0][0], m.mask > 0, all_maps_adj[0][1] > 0,
                                           maskROI=maskROI, adj_wT1=True, fat_threshold=0.7, proj_on_mask1=True,
                                           name="Spiral spus{}_tpus{}".format(sp,tp))

        if df_python.empty:
            df_python = df_current
        else:
            df_python = pd.merge(df_python, df_current, left_index=True, right_index=True)

radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=512)
kdata_radial = m.generate_kdata(radial_traj,useGPU=useGPU)
volumes_radial = simulate_radial_undersampled_images(kdata_radial,radial_traj,m.image_size,useGPU=useGPU,density_adj=True)
all_maps_adj = optimizer.search_patterns(dictfile, volumes_radial)

df_current = metrics_paramMaps_ROI(m.paramMap, all_maps_adj[0][0], m.mask > 0, all_maps_adj[0][1] > 0,
                                           maskROI=maskROI, adj_wT1=True, fat_threshold=0.7, proj_on_mask1=True,
                                           name="Radial")


df_res = pd.merge(df_python,df_current,left_index=True,right_index=True)
df_res.to_csv("{} {} Spiral vs Radial.csv".format(type,num))
# plt.scatter(spiral_traj.traj[0,:,0],spiral_traj.traj[0,:,1])
# plt.scatter(kdata[0].real,kdata[0].imag)
# tx,ty=np.meshgrid(spiral_traj.traj[0,:,0],spiral_traj.traj[0,:,1])
# plt.figure()
# plt.tricontourf(spiral_traj.traj[0,:,0],spiral_traj.traj[0,:,1],np.abs(kdata[0]),levels=100)
# plt.colorbar()

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

optimizer = SimpleDictSearch(mask=m.mask,niter=0,seq=seq,trajectory=spiral_traj,split=1000,pca=True,threshold_pca=15,log=False,useAdjPred=False)
all_maps_adj=optimizer.search_patterns(dictfile,volumes)

plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_adj.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_adj[iter][0], m.mask > 0, all_maps_adj[iter][1] > 0,maskROI=maskROI,
                             title="Spiral ROI Orig vs Iteration {}".format(iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7)
it=0
compare_paramMaps(m.paramMap,all_maps_adj[it][0],m.mask>0,all_maps_adj[it][1]>0,adj_wT1=True,fat_threshold=0.7,title1="Orig",title2="Spiral Rebuilt It {}".format(it),figsize=(30,10),fontsize=15,save=True,proj_on_mask1=True)

plot_evolution_params(m.paramMap,m.mask>0,all_maps_adj)
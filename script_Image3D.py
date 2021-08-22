
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
from tqdm import tqdm
#import pycuda.autoinit
# from pycuda.gpuarray import GPUArray, to_gpu
#
# c_gpu = GPUArray((1, 10), dtype=np.complex64)
# c_gpu.fill(0)
# c = c_gpu.get()
# c_gpu.gpudata.free()
## Random map simulation

start = datetime.now()

dictfile = "mrf175.dict"
dictfile = "mrf175_CS.dict"
#dictfile = "mrf175_SimReco2.dict"

useGPU = False

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

nb_slices= 64
nb_empty_slices=8
undersampling_factor=4
repeat_slice=8

gen_mode ="loop"

m = RandomMap3D("TestRandom3DMovement",dict_config,nb_slices=nb_slices,nb_empty_slices=nb_empty_slices,undersampling_factor=undersampling_factor,repeat_slice=repeat_slice,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)

m.buildParamMap()

m.plotParamMap("wT1")

##### Simulating Ref Images
m.build_ref_images(seq)

#direction=np.array([0.0,4.0,0.0])
#move = TranslationBreathing(direction,T=4000,frac_exp=0.7)

#m.add_movements([move])

npoint=512
nspoke=8
ntimesteps=175

nb_total_slices=m.paramDict["nb_total_slices"]
undersampling_factor = m.paramDict["undersampling_factor"]

radial_traj_3D=Radial3D(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint,nb_slices=nb_total_slices,undersampling_factor=undersampling_factor)


#kdata = m.generate_kdata(radial_traj_3D,useGPU=useGPU)

# with open("kdata_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
#     pickle.dump(kdata, file)

kdata = pickle.load( open( "kdata_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" ) )



#kdata_noGPU=m.generate_radial_kdata(radial_traj_3D,useGPU=False)
# volumes = simulate_radial_undersampled_images(kdata,radial_traj_3D,m.image_size,density_adj=True,useGPU=useGPU)
#
# with open("volumes_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
#     pickle.dump(volumes, file)

volumes = pickle.load( open( "volumes_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" ) )


#volumes_noGPU = simulate_radial_undersampled_images(kdata_noGPU,radial_traj_3D,m.image_size,density_adj=True,useGPU=True)
#ani,ani1=animate_multiple_images(volumes[:,4,:,:],volumes_noGPU[:,4,:,:])

#ani=animate_images(volumes[:,6,:,:])

mask = build_mask_single_image(kdata,radial_traj_3D,m.image_size,useGPU=False)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream
#plt.imshow(mask[m.paramDict["nb_empty_slices"]+int(m.paramDict["nb_slices"]/2),:,:])

#plt.imshow(mask[m.paramDict["nb_empty_slices"]-5,:,:])

optimizer = SimpleDictSearch(mask=mask,niter=0,seq=seq,trajectory=radial_traj_3D,split=500,pca=True,threshold_pca=15,useGPU=True,log=False,useAdjPred=False,verbose=False)
all_maps_adj=optimizer.search_patterns(dictfile,volumes)

end=datetime.now()
print(end-start)

file = open( "all_maps_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" )
# dump information to that file
pickle.dump(all_maps_adj, file)
# close the file
file.close()

plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_adj.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_adj[iter][0], m.mask > 0, all_maps_adj[iter][1] > 0,maskROI=maskROI,
                             title="Slices{}_US{} No movements ROI Orig vs Iteration {}".format(nb_total_slices,undersampling_factor,iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=True)

# compare_paramMaps_3D(m.paramMap,all_maps_adj[0][0],m.mask>0,all_maps_adj[0][1]>0,slice=m.paramDict["nb_empty_slices"]-1,title1="Orig",title2="Outside",proj_on_mask1=True,save=False,adj_wT1=True,fat_threshold=0.7)
# compare_paramMaps_3D(m.paramMap,all_maps_adj[0][0],m.mask>0,all_maps_adj[0][1]>0,slice=m.paramDict["nb_empty_slices"]+5,title1="Orig",title2="Inside",proj_on_mask1=True,save=False,adj_wT1=True,fat_threshold=0.7)
# compare_paramMaps_3D(m.paramMap,all_maps_adj[0][0],m.mask>0,all_maps_adj[0][1]>0,slice=m.paramDict["nb_empty_slices"]+int(m.paramDict["nb_slices"]/2),title1="Orig",title2="Center",proj_on_mask1=True,save=False,adj_wT1=True,fat_threshold=0.7)

size_slice = int(m.paramDict["nb_slices"]/m.paramDict["repeat_slice"])

iter =0
sl = 1

compare_paramMaps_3D(m.paramMap,all_maps_adj[iter][0],m.mask>0,all_maps_adj[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl-1)*size_slice+int(size_slice/2),title1="Slices{}_US{} No movements Orig".format(nb_total_slices,undersampling_factor),title2="Mid Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)
compare_paramMaps_3D(m.paramMap,all_maps_adj[iter][0],m.mask>0,all_maps_adj[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl)*size_slice,title1="Slices{}_US{} No movements Orig".format(nb_total_slices,undersampling_factor),title2="Border Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)

plt.close("all")


##### ADDING MOVEMENT
direction=np.array([0.0,4.0,0.0])
move = TranslationBreathing(direction,T=4000,frac_exp=0.7)

m.add_movements([move])

kdata = m.generate_kdata(radial_traj_3D,useGPU=useGPU)
#kdata_noGPU=m.generate_radial_kdata(radial_traj_3D,useGPU=False)

with open("kdata_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
    pickle.dump(kdata, file)

#kdata_noGPU=m.generate_radial_kdata(radial_traj_3D,useGPU=False)
volumes = simulate_radial_undersampled_images(kdata,radial_traj_3D,m.image_size,density_adj=True,useGPU=useGPU)

with open("volumes_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
    pickle.dump(volumes, file)

#volumes_noGPU = simulate_radial_undersampled_images(kdata_noGPU,radial_traj_3D,m.image_size,density_adj=True,useGPU=True)
#ani,ani1=animate_multiple_images(volumes[:,4,:,:],volumes_noGPU[:,4,:,:])

#ani=animate_images(volumes[:,6,:,:])

mask = build_mask_single_image(kdata,radial_traj_3D,m.image_size,useGPU=False)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream

all_maps_mvt=optimizer.search_patterns(dictfile,volumes)

#
# regression_paramMaps_ROI(m.paramMap, all_maps_adj[4][0], m.mask > 0, all_maps_adj[4][1] > 0,maskROI=maskROI,
#                              title="ROI Orig vs Iteration {}".format(4), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7)

plt.close("all")

file = open( "all_maps_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" )
# dump information to that file
pickle.dump(all_maps_mvt, file)
# close the file
file.close()

plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_mvt.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_mvt[iter][0], m.mask > 0, all_maps_mvt[iter][1] > 0,maskROI=maskROI,
                             title="Slices{}_US{} movement ROI Orig vs Iteration {}".format(nb_total_slices,undersampling_factor,iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=True)

# compare_paramMaps_3D(m.paramMap,all_maps_adj[0][0],m.mask>0,all_maps_adj[0][1]>0,slice=m.paramDict["nb_empty_slices"]-1,title1="Orig",title2="Outside",proj_on_mask1=True,save=False,adj_wT1=True,fat_threshold=0.7)
# compare_paramMaps_3D(m.paramMap,all_maps_adj[0][0],m.mask>0,all_maps_adj[0][1]>0,slice=m.paramDict["nb_empty_slices"]+5,title1="Orig",title2="Inside",proj_on_mask1=True,save=False,adj_wT1=True,fat_threshold=0.7)
# compare_paramMaps_3D(m.paramMap,all_maps_adj[0][0],m.mask>0,all_maps_adj[0][1]>0,slice=m.paramDict["nb_empty_slices"]+int(m.paramDict["nb_slices"]/2),title1="Orig",title2="Center",proj_on_mask1=True,save=False,adj_wT1=True,fat_threshold=0.7)

size_slice = int(m.paramDict["nb_slices"]/m.paramDict["repeat_slice"])

iter =0
sl = 1

compare_paramMaps_3D(m.paramMap,all_maps_mvt[iter][0],m.mask>0,all_maps_mvt[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl-1)*size_slice+int(size_slice/2),title1="Slices{}_US{} movement Orig".format(nb_total_slices,undersampling_factor),title2="Mid Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)
compare_paramMaps_3D(m.paramMap,all_maps_mvt[iter][0],m.mask>0,all_maps_mvt[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl)*size_slice,title1="Slices{}_US{} movement Orig".format(nb_total_slices,undersampling_factor),title2="Border Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)

#####CORRECTION

kdata = pickle.load( open( "kdata_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" ) )
kdata = np.array(kdata)

transf=m.list_movements[0].paramDict["transformation"]
t = m.t
shifts = transf(t.flatten().reshape(-1,1))[:,1]


traj=radial_traj_3D.get_traj()
traj_for_reconstruction=radial_traj_3D.get_traj_for_reconstruction()

traj_for_selection = np.array(groupby(traj,npoint,axis=1))
kdata_for_selection = np.array(groupby(kdata,npoint,axis=1))

traj_for_selection=traj_for_selection.reshape(shifts.shape[0],-1,3)
kdata_for_selection=kdata_for_selection.reshape(shifts.shape[0],-1)

shifts_reshaped=shifts.reshape(t.shape)
shifts_reshaped=shifts.reshape(m.paramDict["nb_rep"],ntimesteps,-1)

# traj_for_selection=traj_for_selection.reshape(t.shape+traj_for_selection.shape[-2:])
# traj_for_selection=traj_for_selection.reshape((m.paramDict["nb_rep"],ntimesteps,-1)+traj_for_selection.shape[-2:])
#
# kdata_for_selection=kdata_for_selection.reshape(t.shape+kdata_for_selection.shape[-1:])
# kdata_for_selection=kdata_for_selection.reshape((m.paramDict["nb_rep"],ntimesteps,-1)+kdata_for_selection.shape[-1:])


perc=80
threshold = np.percentile(shifts,perc)
cond = (shifts>np.percentile(shifts,perc))
indices=np.where(shifts_reshaped>np.percentile(shifts,perc))

traj_retained=traj_for_selection[cond,:,:]
kdata_retained=kdata_for_selection[cond,:]


dico_traj={}
dico_kdata={}
for i,index in enumerate(np.array(indices).T):
    curr_slice=index[0]
    ts=index[1]
    curr_spoke=index[2]
    if ts not in dico_traj:
        dico_traj[ts]= []
        dico_kdata[ts] =[]

    #dico_traj[ts]=[*dico_traj[ts],*traj_retained[i]]
    #dico_kdata[ts]=[*dico_kdata[ts],*kdata_retained[i]]

    dico_traj[ts].append(traj_retained[i])
    dico_kdata[ts].append(kdata_retained[i])

retained_timesteps = list(dico_traj.keys())
retained_timesteps.sort()

traj_retained_final=[]
kdata_retained_final=[]

# for ts in tqdm(range(len(retained_timesteps))):
#     traj_retained_final.append(np.array(dico_traj[ts]))
#     kdata_retained_final.append(np.array(dico_kdata[ts]))

for ts in tqdm(retained_timesteps):
    traj_retained_final.append(np.array(dico_traj[ts]).flatten().reshape(-1,3))
    kdata_retained_final.append(np.array(dico_kdata[ts]).flatten())

radial_traj_3D_corrected=Radial3D(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint,nb_slices=nb_total_slices,undersampling_factor=undersampling_factor)
radial_traj_3D_corrected.traj_for_reconstruction=traj_retained_final

volumes_corrected = simulate_radial_undersampled_images(kdata_retained_final,radial_traj_3D_corrected,m.image_size,density_adj=True,useGPU=useGPU)

with open("volumes_mvt_corrected_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
    pickle.dump(volumes_corrected, file)

#mask = build_mask_single_image(kdata_retained_final,radial_traj_3D_corrected,m.image_size,useGPU=False)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream

mask = build_mask_single_image(kdata,radial_traj_3D,m.image_size,useGPU=False)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream


optimizer = SimpleDictSearch(mask=mask,niter=0,seq=seq,trajectory=radial_traj_3D_corrected,split=1000,pca=True,threshold_pca=15,useGPU=True,log=False,useAdjPred=False,verbose=False)
all_maps_mvt_corrected=optimizer.search_patterns(dictfile,volumes_corrected,retained_timesteps=retained_timesteps)

file = open( "all_maps_mvt_corrected_perc{}_sl{}us{}_{}.pkl".format(perc,nb_total_slices,undersampling_factor,m.name), "wb" )
# dump information to that file
pickle.dump(all_maps_mvt_corrected, file)
# close the file
file.close()

plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_mvt_corrected.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_mvt_corrected[iter][0], m.mask > 0, all_maps_mvt_corrected[iter][1] > 0,maskROI=maskROI,
                             title="Slices{}_US{} movement corrected perc {} ROI Orig vs Iteration {}".format(nb_total_slices,undersampling_factor,perc,iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=True)

# compare_paramMaps_3D(m.paramMap,all_maps_adj[0][0],m.mask>0,all_maps_adj[0][1]>0,slice=m.paramDict["nb_empty_slices"]-1,title1="Orig",title2="Outside",proj_on_mask1=True,save=False,adj_wT1=True,fat_threshold=0.7)
# compare_paramMaps_3D(m.paramMap,all_maps_adj[0][0],m.mask>0,all_maps_adj[0][1]>0,slice=m.paramDict["nb_empty_slices"]+5,title1="Orig",title2="Inside",proj_on_mask1=True,save=False,adj_wT1=True,fat_threshold=0.7)
# compare_paramMaps_3D(m.paramMap,all_maps_adj[0][0],m.mask>0,all_maps_adj[0][1]>0,slice=m.paramDict["nb_empty_slices"]+int(m.paramDict["nb_slices"]/2),title1="Orig",title2="Center",proj_on_mask1=True,save=False,adj_wT1=True,fat_threshold=0.7)

size_slice = int(m.paramDict["nb_slices"]/m.paramDict["repeat_slice"])

iter =0
sl = 1

compare_paramMaps_3D(m.paramMap,all_maps_mvt_corrected[iter][0],m.mask>0,all_maps_mvt_corrected[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl-1)*size_slice+int(size_slice/2),title1="Slices{}_US{} movement corrected perc{} Orig".format(nb_total_slices,undersampling_factor,perc),title2="Mid Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)
compare_paramMaps_3D(m.paramMap,all_maps_mvt_corrected[iter][0],m.mask>0,all_maps_mvt_corrected[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl)*size_slice,title1="Slices{}_US{} movement corrected perc{} Orig".format(nb_total_slices,undersampling_factor,perc),title2="Border Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)






#
# import pycuda.autoinit
# import pycuda.gpuarray as cua
# import pycuda.tools as cut
# import numpy as np
#
# m = cut.DeviceMemoryPool()
#
# a= np.ones(2**30-1,dtype=np.float32)
# b= cua.to_gpu(a, allocator=m.allocate)  # Passes
#
# a= np.ones(2**30,dtype=np.float32)
# b= cua.to_gpu(a, allocator=m.allocate)
#
#
# import torch
# import sys
# print('__Python VERSION:', sys.version)
# print('__pyTorch VERSION:', torch.__version__)
# print('__CUDA VERSION')
# from subprocess import call
# # call(["nvcc", "--version"]) does not work
#
# print('__CUDNN VERSION:', torch.backends.cudnn.version())
# print('__Number CUDA Devices:', torch.cuda.device_count())
# print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
# print('Active CUDA Device: GPU', torch.cuda.current_device())
#
# print ('Available devices ', torch.cuda.device_count())
# print ('Current cuda device ', torch.cuda.current_device())
#
#
#
# import cupy as cp
#
# n = 500
# a = cp.ones([n,256,100], dtype=cp.float32)
# b = cp.ones([n,100,100], dtype=cp.float32)
#
# a = cp.reshape(a, [n,256,100,1])
# b = cp.reshape(b, [n,1,100,100])
# c = cp.sum(a*b, axis=-2)
#
# print('Complete.')
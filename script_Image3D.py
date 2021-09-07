
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


load_paramMap=True
build_ref_images=True
load=True

if not(load_paramMap):
    load=False

load_maps=True

is_random=False


dictfile = "mrf175.dict"
dictfile = "mrf175_CS.dict"
#dictfile = "mrf175_SimReco2.dict"

useGPU_simulation=False
useGPU_dictsearch=False

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

nb_slices= 16
nb_empty_slices=4
undersampling_factor=4
repeat_slice=8

gen_mode ="other"

m = RandomMap3D("TestRandom3DMovement",dict_config,nb_slices=nb_slices,nb_empty_slices=nb_empty_slices,undersampling_factor=undersampling_factor,repeat_slice=repeat_slice,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)

if not(load_paramMap):
    m.buildParamMap()
    with open("paramMap_sl{}_{}.pkl".format(nb_slices+2*nb_empty_slices,m.name), "wb" ) as file:
        pickle.dump(m.paramMap, file)

else:
    m.paramMap=pickle.load(open("paramMap_sl{}_{}.pkl".format(nb_slices+2*nb_empty_slices,m.name), "rb"))
#m.plotParamMap("wT1")

##### Simulating Ref Images
if build_ref_images:
    m.build_ref_images(seq)
else:#still need to build the timeline for applying movement / movement correction
    m.build_timeline(seq)


npoint=512
nspoke=8
ntimesteps=175

nb_total_slices=m.paramDict["nb_total_slices"]
undersampling_factor = m.paramDict["undersampling_factor"]

radial_traj_3D=Radial3D(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint,nb_slices=nb_total_slices,undersampling_factor=undersampling_factor,is_random=is_random)

if not(load):

    kdata = m.generate_kdata(radial_traj_3D,useGPU=useGPU_simulation)

    if not(is_random):
        with open("kdata_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
            pickle.dump(kdata, file)
    else:
        with open("kdata_random_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
            pickle.dump(kdata, file)

else:
    if not(is_random):
        kdata = pickle.load( open( "kdata_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" ) )
    else:
        kdata = pickle.load(
            open("kdata_random_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices, undersampling_factor, m.name), "rb"))


#kdata_noGPU=m.generate_radial_kdata(radial_traj_3D,useGPU=False)

if not(load):
    volumes = simulate_radial_undersampled_images(kdata,radial_traj_3D,m.image_size,density_adj=True,useGPU=useGPU_simulation)

    if not(is_random):
        with open("volumes_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
            pickle.dump(volumes, file)
    else:
        with open("volumes_random_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
            pickle.dump(volumes, file)
else:
    if not(is_random):
        volumes = pickle.load( open( "volumes_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" ) )
    else:
        volumes = pickle.load(
            open("volumes_random_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices, undersampling_factor, m.name), "rb"))
#volumes_noGPU = simulate_radial_undersampled_images(kdata_noGPU,radial_traj_3D,m.image_size,density_adj=True,useGPU=True)
#ani,ani1=animate_multiple_images(volumes[:,4,:,:],volumes_noGPU[:,4,:,:])

#ani=animate_images(volumes[:,6,:,:])

mask = build_mask_single_image(kdata,radial_traj_3D,m.image_size,useGPU=useGPU_simulation)
#plt.imshow(mask[m.paramDict["nb_empty_slices"]+int(m.paramDict["nb_slices"]/2),:,:])

#plt.imshow(mask[m.paramDict["nb_empty_slices"]-5,:,:])

niter=2

optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=radial_traj_3D,split=2000,pca=True,threshold_pca=20,useGPU_simulation=useGPU_simulation,useGPU_dictsearch=useGPU_dictsearch,log=True,useAdjPred=False,verbose=False,gen_mode=gen_mode)


if not(load_maps):

    all_maps_adj=optimizer.search_patterns(dictfile,volumes)

    if not(is_random):
        file = open( "all_maps_no_mvt_sl{}us{}_iter{}_{}.pkl".format(nb_total_slices,undersampling_factor,niter,m.name), "wb" )
    # dump information to that file
    else:
        file = open(
            "all_maps_random_no_mvt_sl{}us{}_iter{}_{}.pkl".format(nb_total_slices, undersampling_factor, niter, m.name), "wb")
    pickle.dump(all_maps_adj, file)
    # close the file
    file.close()
else:
    if not(is_random):
        all_maps_adj = pickle.load( open("all_maps_no_mvt_sl{}us{}_iter{}_{}.pkl".format(nb_total_slices,undersampling_factor,niter,m.name), "rb" ))
    else:
        all_maps_adj = pickle.load(
            open("all_maps_random_no_mvt_sl{}us{}_iter{}_{}.pkl".format(nb_total_slices, undersampling_factor, niter, m.name),
                 "rb"))

end=datetime.now()
print(end-start)

plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_adj.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_adj[iter][0], m.mask > 0, all_maps_adj[iter][1] > 0,maskROI=maskROI,
                             title="Slices{}_US{} No movements ROI Orig vs Iteration {}".format(nb_total_slices,undersampling_factor,iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=True)


size_slice = int(m.paramDict["nb_slices"]/m.paramDict["repeat_slice"])

plot_evolution_params(m.paramMap,m.mask>0,all_maps_adj,maskROI,save=True)

# iter =0
# sl = 1
#
# compare_paramMaps_3D(m.paramMap,all_maps_adj[iter][0],m.mask>0,all_maps_adj[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl-1)*size_slice+int(size_slice/2),title1="Slices{}_US{} No movements Orig".format(nb_total_slices,undersampling_factor),title2="Mid Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)
# compare_paramMaps_3D(m.paramMap,all_maps_adj[iter][0],m.mask>0,all_maps_adj[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl)*size_slice,title1="Slices{}_US{} No movements Orig".format(nb_total_slices,undersampling_factor),title2="Border Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)

#plt.close("all")


##### ADDING MOVEMENT
direction=np.array([0.0,4.0,0.0])
move = TranslationBreathing(direction,T=4000,frac_exp=0.7)

m.add_movements([move])

load=False
load_maps=False

if not(load):
    kdata = m.generate_kdata(radial_traj_3D,useGPU=useGPU_simulation)
    #kdata_noGPU=m.generate_radial_kdata(radial_traj_3D,useGPU=False)
    with open("kdata_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
        pickle.dump(kdata, file)

else:
    kdata = pickle.load(
        open("kdata_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb"))

if not(load):
    #kdata_noGPU=m.generate_radial_kdata(radial_traj_3D,useGPU=False)
    volumes = simulate_radial_undersampled_images(kdata,radial_traj_3D,m.image_size,density_adj=True,useGPU=useGPU_simulation)

    with open("volumes_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
        pickle.dump(volumes, file)

else:
    volumes = pickle.load(open("volumes_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb"))
#volumes_noGPU = simulate_radial_undersampled_images(kdata_noGPU,radial_traj_3D,m.image_size,density_adj=True,useGPU=True)
#ani,ani1=animate_multiple_images(volumes[:,4,:,:],volumes_noGPU[:,4,:,:])

#ani=animate_images(volumes[:,6,:,:])

mask = build_mask_single_image(kdata,radial_traj_3D,m.image_size,useGPU=useGPU_simulation)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream

if not(load_maps):
    all_maps_mvt=optimizer.search_patterns(dictfile,volumes)
    file = open( "all_maps_mvt_sl{}us{}_iter{}_{}.pkl".format(nb_total_slices,undersampling_factor,niter,m.name), "wb" )
    # dump information to that file
    pickle.dump(all_maps_mvt, file)
    # close the file
    file.close()

else:
    all_maps_mvt = pickle.load(  open("all_maps_mvt_sl{}us{}_iter{}_{}.pkl".format(nb_total_slices,undersampling_factor,niter,m.name), "rb" ))

#
# regression_paramMaps_ROI(m.paramMap, all_maps_adj[4][0], m.mask > 0, all_maps_adj[4][1] > 0,maskROI=maskROI,
#                              title="ROI Orig vs Iteration {}".format(4), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7)



#plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_mvt.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_mvt[iter][0], m.mask > 0, all_maps_mvt[iter][1] > 0,maskROI=maskROI,
                             title="Slices{}_US{} movement ROI Orig vs Iteration {}".format(nb_total_slices,undersampling_factor,iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=True)


# size_slice = int(m.paramDict["nb_slices"]/m.paramDict["repeat_slice"])
#
# iter =0
# sl = 1
#
# compare_paramMaps_3D(m.paramMap,all_maps_mvt[iter][0],m.mask>0,all_maps_mvt[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl-1)*size_slice+int(size_slice/2),title1="Slices{}_US{} movement Orig".format(nb_total_slices,undersampling_factor),title2="Mid Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)
# compare_paramMaps_3D(m.paramMap,all_maps_mvt[iter][0],m.mask>0,all_maps_mvt[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl)*size_slice,title1="Slices{}_US{} movement Orig".format(nb_total_slices,undersampling_factor),title2="Border Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)

#####CORRECTION

load=False
load_maps=False

kdata = pickle.load( open( "kdata_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" ) )
kdata = np.array(kdata)

if not(load):
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


    perc=50
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

    traj_retained_final=np.array(traj_retained_final)
    kdata_retained_final=np.array(kdata_retained_final)

    size_initial = traj_for_reconstruction.size / 3
    size_retained_final = traj_retained_final.size / 3
    ratio=size_retained_final/size_initial
    print("Compression factor : {}%".format((1-ratio)*100))


    radial_traj_3D_corrected=Radial3D(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint,nb_slices=nb_total_slices,undersampling_factor=undersampling_factor)
    radial_traj_3D_corrected.traj_for_reconstruction=traj_retained_final

    volumes_corrected = simulate_radial_undersampled_images(kdata_retained_final,radial_traj_3D_corrected,m.image_size,density_adj=True,useGPU=useGPU_simulation)

    with open("volumes_mvt_corrected__perc{}sl{}us{}_{}.pkl".format(perc,nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
        pickle.dump(volumes_corrected, file)

else:
    volumes_corrected=pickle.load(
        open("volumes_mvt_corrected_perc{}sl{}us{}_{}.pkl".format(perc,nb_total_slices,undersampling_factor,m.name), "rb"))

#mask = build_mask_single_image(kdata_retained_final,radial_traj_3D_corrected,m.image_size,useGPU=False)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream


if not(load_maps):

    mask = build_mask_single_image(kdata, radial_traj_3D, m.image_size,
                                   useGPU=useGPU_simulation)  # Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream
    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj_3D_corrected, split=1000,
                                 pca=True, threshold_pca=15, useGPU=True, log=False, useAdjPred=False, verbose=False)

    all_maps_mvt_corrected=optimizer.search_patterns(dictfile,volumes_corrected,retained_timesteps=retained_timesteps)
    file = open( "all_maps_mvt_corrected_perc{}_sl{}us{}_iter{}_{}.pkl".format(perc,nb_total_slices,undersampling_factor,niter,m.name), "wb" )
    # dump information to that file
    pickle.dump(all_maps_mvt_corrected, file)
    # close the file
    file.close()

else:
    all_maps_mvt = pickle.load(
        open("all_maps_mvt_corrected_perc{}_sl{}us{}_iter{}_{}.pkl".format(perc,nb_total_slices,undersampling_factor,niter,m.name), "rb"))

#plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_mvt_corrected.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_mvt_corrected[iter][0], m.mask > 0, all_maps_mvt_corrected[iter][1] > 0,maskROI=maskROI,
                             title="Slices{}_US{} movement corrected perc {} ROI Orig vs Iteration {}".format(nb_total_slices,undersampling_factor,perc,iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=True)


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


#test kdatai


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

load=True
load_maps=False
load_paramMap=True
build_ref_images=False

is_random=False


dictfile = "mrf175.dict"
dictfile = "mrf175_CS.dict"
#dictfile = "mrf175_SimReco2.dict"

useGPU_simulation=True
useGPU_dictsearch=True

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

if not(load_paramMap):
    m.buildParamMap()
    with open("paramMap_sl{}_{}.pkl".format(nb_slices+2*nb_empty_slices,m.name), "wb" ) as file:
        pickle.dump(m.paramMap, file)

else:
    m.paramMap=pickle.load(open("paramMap_sl{}_{}.pkl".format(nb_slices+2*nb_empty_slices,m.name), "rb"))
#m.plotParamMap("wT1")

##### Simulating Ref Images
if build_ref_images:
    m.build_ref_images(seq)
else:#still need to build the timeline for applying movement / movement correction
    m.build_timeline(seq)


npoint=512
nspoke=8
ntimesteps=175

nb_total_slices=m.paramDict["nb_total_slices"]
undersampling_factor = m.paramDict["undersampling_factor"]

radial_traj_3D=Radial3D(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint,nb_slices=nb_total_slices,undersampling_factor=undersampling_factor,is_random=is_random)
volumes = pickle.load( open( "volumes_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" ) )
volumes = volumes / np.linalg.norm(volumes, 2, axis=0)
volumes0=volumes


kdatai=np.load("./log/kdatai.npy")

kdatai = np.array(kdatai)
nans = np.nonzero(np.isnan(kdatai))
if len(nans)>0:
    print("Warning : Nan Values replaced by zeros in rebuilt kdata")
    kdatai[nans]=0.0


kdatai[np.abs(kdatai)>1e5]=1e5

volumesi=simulate_radial_undersampled_images(kdatai,radial_traj_3D,m.image_size,useGPU=useGPU_simulation,density_adj=True)

nans_volumes=np.argwhere(np.isnan(volumesi))
if len(nans_volumes)>0:
    raise ValueError("Error : Nan Values in volumes")

sl=40
ani = animate_images((volumesi[:,sl,:,:]))

plt.figure()
plt.imshow(np.abs(volumesi[1,sl,:,:]))

volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

new_volumes = [vol + (vol0 - voli) for vol, vol0, voli in zip(volumes, volumes0, volumesi)]
new_volumes=np.array(new_volumes)

kdata = pickle.load( open( "kdata_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" ) )

mask = build_mask_single_image(kdata,radial_traj_3D,m.image_size,useGPU=useGPU_simulation)

niter=0
optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=radial_traj_3D,split=2000,pca=True,threshold_pca=20,useGPU_simulation=useGPU_simulation,useGPU_dictsearch=useGPU_dictsearch,log=True,useAdjPred=False,verbose=False,gen_mode=gen_mode)

all_maps_adj = optimizer.search_patterns(dictfile,new_volumes)


save=True
maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_adj.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_adj[iter][0], m.mask > 0, all_maps_adj[iter][1] > 0,maskROI=maskROI,
                             title="Slices{}_US{} No movements ROI Orig vs Iteration {}".format(nb_total_slices,undersampling_factor,1), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=save)


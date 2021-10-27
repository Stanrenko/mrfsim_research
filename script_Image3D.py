
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


folder_3D = "./3D/"

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

nb_slices= 8
nb_empty_slices=2
undersampling_factor=1
repeat_slice=4

gen_mode ="other"

m = RandomMap3D("TestRandom3DMovement",dict_config,nb_slices=nb_slices,nb_empty_slices=nb_empty_slices,undersampling_factor=undersampling_factor,repeat_slice=repeat_slice,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)

if not(load_paramMap):
    m.buildParamMap()
    with open(folder_3D+"paramMap_sl{}_rp{}_{}.pkl".format(nb_slices+2*nb_empty_slices,repeat_slice,m.name), "wb" ) as file:
        pickle.dump(m.paramMap, file)

else:
    file=open(folder_3D+"paramMap_sl{}_rp{}_{}.pkl".format(nb_slices+2*nb_empty_slices,repeat_slice,m.name), "rb")
    m.paramMap=pickle.load(file)
    file.close()
#m.plotParamMap("wT1")

##### Simulating Ref Images
if build_ref_images:
    m.build_ref_images(seq)
else:#still need to build the timeline for applying movement / movement correction
    m.build_timeline(seq)


npoint=512
nspoke=8 #nspokes per z encoding
all_spokes=1400 #spokes for each partition
ntimesteps=int(all_spokes/nspoke)


nb_total_slices=m.paramDict["nb_total_slices"]
undersampling_factor = m.paramDict["undersampling_factor"]

radial_traj_3D=Radial3D(total_nspokes=all_spokes,nspoke_per_z_encoding=nspoke,npoint=npoint,nb_slices=nb_total_slices,undersampling_factor=undersampling_factor,is_random=is_random)

if not(load):

    kdata = m.generate_kdata(radial_traj_3D,useGPU=useGPU_simulation)

    if not(is_random):
        with open(folder_3D+"kdata_no_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,m.name), "wb" ) as file:
            pickle.dump(kdata, file)
    else:
        with open(folder_3D+"kdata_random_no_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,m.name), "wb" ) as file:
            pickle.dump(kdata, file)

else:
    if not(is_random):
        file=open( folder_3D+"kdata_no_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,m.name), "rb" )
        kdata = pickle.load(file)
        file.close()
    else:
        file=open(folder_3D+"kdata_random_no_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices, repeat_slice,undersampling_factor, m.name), "rb")
        kdata = pickle.load(file)
        file.close()

# nav_y=Navigator3D(direction=[0.0,1.0,0.0])
# nav_x = Navigator3D(direction=[1.0,0.0,0.0])
#
# kdata_nav_y = m.generate_kdata(nav_y,useGPU=useGPU_simulation)
# kdata_nav_x= m.generate_kdata(nav_x,useGPU=useGPU_simulation)
#
# with open(folder_3D + "kdata_nav_x_sl{}us{}_{}.pkl".format(nb_total_slices, undersampling_factor, m.name), "wb") as file:
#     pickle.dump(kdata_nav_x, file)
#
# with open(folder_3D + "kdata_nav_y_sl{}us{}_{}.pkl".format(nb_total_slices, undersampling_factor, m.name), "wb") as file:
#     pickle.dump(kdata_nav_y, file)
#
#
# with open(folder_3D + "kdata_nav_x_sl{}us{}_{}.pkl".format(nb_total_slices, undersampling_factor, m.name), "rb") as file:
#     kdata_nav_x=pickle.load(file)
#
# with open(folder_3D + "kdata_nav_y_sl{}us{}_{}.pkl".format(nb_total_slices, undersampling_factor, m.name), "rb") as file:
#     kdata_nav_y=pickle.load(file)

#kdata_noGPU=m.generate_radial_kdata(radial_traj_3D,useGPU=False)

if not(load):
    volumes = simulate_radial_undersampled_images(kdata,radial_traj_3D,m.image_size,density_adj=True,useGPU=useGPU_simulation)

    if not(is_random):
        with open(folder_3D+"volumes_no_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,m.name), "wb" ) as file:
            pickle.dump(volumes, file)
    else:
        with open(folder_3D+"volumes_random_no_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,m.name), "wb" ) as file:
            pickle.dump(volumes, file)
else:
    if not(is_random):
        volumes = pickle.load( open( folder_3D+"volumes_no_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,m.name), "rb" ) )
        file= open( "volumes_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" )
        volumes = pickle.load(file)
        file.close()
    else:
        file=open(folder_3D+"volumes_random_no_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices, repeat_slice,undersampling_factor, m.name), "rb")
        volumes = pickle.load(file)
        file.close()
#volumes_noGPU = simulate_radial_undersampled_images(kdata_noGPU,radial_traj_3D,m.image_size,density_adj=True,useGPU=True)
#ani,ani1=animate_multiple_images(volumes[:,4,:,:],volumes_noGPU[:,4,:,:])

#ani=animate_images(volumes[:,6,:,:])

mask = build_mask_single_image(kdata,radial_traj_3D,m.image_size,useGPU=useGPU_simulation)
#plt.imshow(mask[m.paramDict["nb_empty_slices"]+int(m.paramDict["nb_slices"]/2),:,:])

#plt.imshow(mask[m.paramDict["nb_empty_slices"]-5,:,:])

niter=0

optimizer = SimpleDictSearch(mask=m.mask,niter=niter,seq=seq,trajectory=radial_traj_3D,split=250,pca=True,threshold_pca=20,useGPU_simulation=useGPU_simulation,useGPU_dictsearch=useGPU_dictsearch,log=False,useAdjPred=False,verbose=False,gen_mode=gen_mode,adj_phase=True)


if not(load_maps):

    all_maps_adj=optimizer.search_patterns(dictfile,volumes)

    if not(is_random):
        file = open( folder_3D+"all_maps_no_mvt_sl{}rp{}us{}_iter{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,niter,m.name), "wb" )
    # dump information to that file
    else:
        file = open(
            folder_3D+"all_maps_random_no_mvt_sl{}rp{}us{}_iter{}_{}.pkl".format(nb_total_slices, repeat_slice,undersampling_factor, niter, m.name), "wb")
    pickle.dump(all_maps_adj, file)
    # close the file
    file.close()
else:
    if not(is_random):
        file = open(folder_3D+"all_maps_no_mvt_sl{}rp{}us{}_iter{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,niter,m.name), "rb" )
        all_maps_adj = pickle.load(file )
        file.close()
    else:
        file=open(folder_3D+"all_maps_random_no_mvt_sl{}rp{}us{}_iter{}_{}.pkl".format(nb_total_slices, repeat_slice,undersampling_factor, niter, m.name),
                 "rb")
        all_maps_adj = pickle.load(file)
        file.close()

end=datetime.now()
print(end-start)

plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_adj.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_adj[iter][0], m.mask > 0, all_maps_adj[iter][1] > 0,maskROI=maskROI,
                             title="Slices{}_US{} No movements ROI Orig vs Iteration {}".format(nb_total_slices,undersampling_factor,iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=True)



size_slice = int(m.paramDict["repeat_slice"])

#plot_evolution_params(m.paramMap,m.mask>0,all_maps_adj,maskROI,save=True)

iter =0
sl = 1
#
#compare_paramMaps_3D(m.paramMap,all_maps_adj[iter][0],m.mask>0,all_maps_adj[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl-1)*size_slice+int(size_slice/2),title1="Slices{}_US{} No movements Orig".format(nb_total_slices,undersampling_factor),title2="Mid Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)
#compare_paramMaps_3D(m.paramMap,all_maps_adj[iter][0],m.mask>0,all_maps_adj[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl)*size_slice,title1="Slices{}_US{} No movements Orig".format(nb_total_slices,undersampling_factor),title2="Border Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)

#plt.close("all")


##### ADDING MOVEMENT
direction=np.array([1.0,0.0,0.0])
move = TranslationBreathing(direction,T=4000,frac_exp=0.7)

m.add_movements([move])

load=True
load_maps=True
useGPU_simulation=False
#
# kdata_nav_y_mvt = m.generate_kdata(nav_y,useGPU=useGPU_simulation)
# kdata_nav_x_mvt= m.generate_kdata(nav_x,useGPU=useGPU_simulation)
#
# with open(folder_3D + "kdata_nav_x_mvt_sl{}us{}_{}.pkl".format(nb_total_slices, undersampling_factor, m.name), "wb") as file:
#     pickle.dump(kdata_nav_x_mvt, file)
#
# with open(folder_3D + "kdata_nav_y_mvt_sl{}us{}_{}.pkl".format(nb_total_slices, undersampling_factor, m.name), "wb") as file:
#     pickle.dump(kdata_nav_y_mvt, file)
#
#
# with open(folder_3D + "kdata_nav_x_mvt_sl{}us{}_{}.pkl".format(nb_total_slices, undersampling_factor, m.name), "rb") as file:
#     kdata_nav_x_mvt=pickle.load(file)
#
# with open(folder_3D + "kdata_nav_y_mvt_sl{}us{}_{}.pkl".format(nb_total_slices, undersampling_factor, m.name), "rb") as file:
#     kdata_nav_y_mvt=pickle.load(file)

if not(load):
    kdata_mvt = m.generate_kdata(radial_traj_3D,useGPU=useGPU_simulation)
    #kdata_noGPU=m.generate_radial_kdata(radial_traj_3D,useGPU=False)
    with open(folder_3D+"kdata_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,m.name), "wb" ) as file:
        pickle.dump(kdata_mvt, file)

else:
    file=open(folder_3D+"kdata_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,m.name), "rb")
    kdata_mvt = pickle.load(file)
    file.close()


useGPU_simulation=False

nav_z=Navigator3D(direction=[0.0,0.0,1.0],applied_timesteps=[1399])
kdata_nav = m.generate_kdata(nav_z,useGPU=useGPU_simulation)

kdata_nav = np.array(kdata_nav[0])
plt.close("all")

ind=-2
#np.sort(np.mod(np.angle(kdata_nav[2,:]),np.pi))
dz=-np.angle(kdata_nav[:,ind])/nav_z.get_traj()[0,ind,2]
phase_correction=np.exp(1j*np.unique(radial_traj_3D.get_traj()[:,:,2],axis=1)*dz[np.newaxis,:])

kdata_mvt_corrected=np.array(kdata_mvt)
kdata_mvt_corrected=kdata_mvt_corrected.reshape(kdata_mvt_corrected.shape[0],m.paramDict["nb_rep"],-1)
kdata_mvt_corrected = kdata_mvt_corrected*phase_correction[:,:,np.newaxis]


volumes_corrected = simulate_radial_undersampled_images(kdata_mvt_corrected,radial_traj_3D,m.image_size,density_adj=True,useGPU=useGPU_simulation)

all_maps_mvt=optimizer.search_patterns(dictfile,volumes_corrected)
file = open( folder_3D+"all_maps_mvt_corrected_sl{}rp{}us{}_iter{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,niter,m.name), "wb" )
# dump information to that file
pickle.dump(all_maps_mvt, file)
    # close the file
file.close()


plt.plot(-np.angle(kdata_nav[:,0]).T/np.pi)



plt.figure()
plt.plot()

dz =

# volumes_mvt_nav_x = simulate_radial_undersampled_images(kdata_nav_x_mvt,nav_x,m.image_size,density_adj=False,useGPU=useGPU_simulation,is_theta_z_adjusted=True)
# volumes_mvt_nav_y = simulate_radial_undersampled_images(kdata_nav_y_mvt,nav_y,m.image_size,density_adj=False,useGPU=useGPU_simulation)


if not(load):
    #kdata_noGPU=m.generate_radial_kdata(radial_traj_3D,useGPU=False)
    volumes_mvt = simulate_radial_undersampled_images(kdata_mvt,radial_traj_3D,m.image_size,density_adj=True,useGPU=useGPU_simulation)

    with open(folder_3D+"volumes_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,m.name), "wb" ) as file:
        pickle.dump(volumes_mvt, file)

else:
    volumes_mvt = pickle.load(open(folder_3D+"volumes_mvt_sl{}rp{}us{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,m.name), "rb"))
#volumes_noGPU = simulate_radial_undersampled_images(kdata_noGPU,radial_traj_3D,m.image_size,density_adj=True,useGPU=True)
#ani,ani1=animate_multiple_images(volumes[:,4,:,:],volumes_noGPU[:,4,:,:])

#ani=animate_images(volumes[:,6,:,:])

useGPU_simulation=False

mask = build_mask_single_image(kdata_mvt,radial_traj_3D,m.image_size,useGPU=useGPU_simulation)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream

niter=0

optimizer = SimpleDictSearch(mask=m.mask,niter=niter,seq=seq,trajectory=radial_traj_3D,split=250,pca=True,threshold_pca=20,useGPU_simulation=useGPU_simulation,useGPU_dictsearch=useGPU_dictsearch,log=False,useAdjPred=False,verbose=False,gen_mode=gen_mode,adj_phase=True)


if not(load_maps):
    all_maps_mvt=optimizer.search_patterns(dictfile,volumes_mvt)
    file = open( folder_3D+"all_maps_mvt_sl{}rp{}us{}_iter{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,niter,m.name), "wb" )
    # dump information to that file
    pickle.dump(all_maps_mvt, file)
    # close the file
    file.close()

else:
    file =open(folder_3D+"all_maps_mvt_sl{}rp{}us{}_iter{}_{}.pkl".format(nb_total_slices,repeat_slice,undersampling_factor,niter,m.name), "rb" )
    all_maps_mvt = pickle.load( file )
    file.close()

#
# regression_paramMaps_ROI(m.paramMap, all_maps_adj[4][0], m.mask > 0, all_maps_adj[4][1] > 0,maskROI=maskROI,
#                              title="ROI Orig vs Iteration {}".format(4), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7)



#plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_mvt.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_mvt[iter][0], m.mask > 0, all_maps_mvt[iter][1] > 0,maskROI=maskROI,
                             title="Slices{}_US{} movement corrected ROI Orig vs Iteration {}".format(nb_total_slices,undersampling_factor,iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=True)


plt.close("all")

size_slice = int(m.paramDict["repeat_slice"])

iter =0
sl = 2

compare_paramMaps_3D(m.paramMap,all_maps_adj[iter][0],m.mask>0,all_maps_adj[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl-1)*size_slice+int(size_slice/2),title1="Slices{}_US{} No movements Orig".format(nb_total_slices,undersampling_factor),title2="Mid Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)
compare_paramMaps_3D(m.paramMap,all_maps_mvt[iter][0],m.mask>0,all_maps_mvt[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl-1)*size_slice+int(size_slice/2),title1="Slices{}_US{} movement Orig".format(nb_total_slices,undersampling_factor),title2="Mid Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)

compare_paramMaps_3D(m.paramMap,all_maps_mvt[iter][0],m.mask>0,all_maps_mvt[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl)*size_slice,title1="Slices{}_US{} movement Orig".format(nb_total_slices,undersampling_factor),title2="Border Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)

iter=0
images_rebuilt_no_mvt,_=simulate_image_series_from_maps(all_maps_adj[iter][0],all_maps_adj[iter][1])
images_rebuilt_mvt,_=simulate_image_series_from_maps(all_maps_mvt[iter][0],all_maps_mvt[iter][1])

bad_pixel=(4,140,75)
good_pixel=(4,155,155)

pixel=bad_pixel
plt.figure()
metric=np.abs
signal = metric(volumes[:,pixel[0],pixel[1],pixel[2]])
signal = (signal-signal.mean())/signal.std()
plt.plot(signal,"g",label="No movement",alpha=0.5)
signal = metric(images_rebuilt_no_mvt[:,pixel[0],pixel[1],pixel[2]])
signal = (signal-signal.mean())/signal.std()
plt.plot(signal,"gx",label="No movement matched pattern")
signal = metric(volumes_mvt[:,pixel[0],pixel[1],pixel[2]])
signal = (signal-signal.mean())/signal.std()
plt.plot(signal,"b",label="Movement",alpha=0.5)
signal = metric(images_rebuilt_mvt[:,pixel[0],pixel[1],pixel[2]])
signal = (signal-signal.mean())/signal.std()
plt.plot(signal,"k--",label="Current slice pattern")
signal = metric(images_rebuilt_no_mvt[:,pixel[0]+size_slice,pixel[1],pixel[2]])
signal = (signal-signal.mean())/signal.std()
plt.plot(signal,"k--",label="Next slice pattern")
plt.legend()




kdata_for_mvt_analysis_no_mvt=np.array(groupby(np.array(kdata),512,axis=1))
kdata_for_mvt_analysis_no_mvt=kdata_for_mvt_analysis_no_mvt[5]
kdata_for_mvt_analysis_no_mvt=kdata_for_mvt_analysis_no_mvt.reshape(-1,512)

kdata_for_mvt_analysis=np.array(groupby(np.array(kdata_mvt),512,axis=1))
kdata_for_mvt_analysis=kdata_for_mvt_analysis[5]
kdata_for_mvt_analysis=kdata_for_mvt_analysis.reshape(-1,512)


plt.close("all")
plt.figure()
plt.plot(np.angle(kdata_for_mvt_analysis_no_mvt[200:250,255]),label="No mvt")
plt.plot(np.angle(kdata_for_mvt_analysis[200:250,255]),label="mvt")
plt.legend()

plt.figure()
plt.plot(np.angle(kdata_for_mvt_analysis[:,255])-np.angle(kdata_for_mvt_analysis_no_mvt[:,255]))



plt.figure()
plt.plot(np.abs(kdata_for_mvt_analysis_no_mvt[:,255]))
plt.figure()
plt.plot(np.abs(kdata_for_mvt_analysis_no_mvt[:,256]))
plt.figure()
plt.plot(np.abs(kdata_for_mvt_analysis_no_mvt[:,255]+kdata_for_mvt_analysis_no_mvt[:,256]))


plt.figure()
plt.plot(np.abs(kdata_for_mvt_analysis[:,255]))
plt.figure()
plt.plot(np.abs(kdata_for_mvt_analysis[:,256]))
plt.figure()
plt.plot(np.abs(kdata_for_mvt_analysis[:,255]+kdata_for_mvt_analysis[:,256]))

ind=255
wind=3
metric=np.angle

plt.close("all")

kdata_centrum=kdata_for_mvt_analysis[:,ind-wind:ind+wind]
pca=PCAComplex()
pca.fit(kdata_centrum.T)

kdata_centrum_no_mvt=kdata_for_mvt_analysis_no_mvt[:,ind-wind:ind+wind]
pca_no_mvt=PCAComplex()
pca_no_mvt.fit(kdata_centrum_no_mvt.T)


comp=0
fig, ax1 = plt.subplots()
ax2=ax1.twinx()

ax1.plot(metric(pca.components_[:,comp]),label="Kdata Movement comp {}".format(comp))
ax1.plot(metric(pca_no_mvt.components_[:,comp]),label="Kdata No movement comp {}".format(comp))
ax1.legend()
ax2.plot(move.paramDict["transformation"](m.t[5,:].reshape(-1,1))[:,1],label="Movement shape")
ax2.legend()

comp=1
fig, ax1 = plt.subplots()
ax2=ax1.twinx()
ax1.plot(metric(pca.components_[:,comp]),label="Kdata Movement comp {}".format(comp))
ax1.plot(metric(pca_no_mvt.components_[:,comp]),label="Kdata No movement comp {}".format(comp))
ax1.legend()
ax2.plot(move.paramDict["transformation"](m.t[5,:].reshape(-1,1))[:,1],label="Movement shape")
ax2.legend()

wind=8

##############################################
kdata_for_mvt_analysis_no_mvt=np.array(groupby(np.array(kdata),512,axis=1))
kdata_for_mvt_analysis=np.array(groupby(np.array(kdata_mvt),512,axis=1))
traj=radial_traj_3D.get_traj()
traj_for_mvt_analysis =np.array(groupby(traj,512,axis=1))


dtheta = np.pi / nspoke
dz = 1/1

kdata_current = kdata_for_mvt_analysis_no_mvt
kdata_normalized = kdata_current / (npoint)*dz * dtheta

density = np.abs(np.linspace(-1, 1, npoint))[np.newaxis,np.newaxis,:]
kdata_normalized =kdata_normalized*density

images_reps=np.zeros((kdata_normalized.shape[0],175,m.image_size[0],m.image_size[1],m.image_size[2]),dtype=np.complex128)

for j in tqdm(range(kdata_normalized.shape[0])):
    traj_rep = traj_for_mvt_analysis[j]
    kdata_rep = kdata_normalized[j]
    traj_rep=traj_rep.reshape(175,-1,3)
    kdata_rep=kdata_rep.reshape(175,-1)
    images=np.zeros((175,m.image_size[0],m.image_size[1],m.image_size[2]),dtype=np.complex128)
    for i,x in tqdm(enumerate(zip(traj_rep, kdata_rep))):
        t=x[0]
        s=x[1]
        images[i] = (finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], s, (m.image_size[0],m.image_size[1],m.image_size[2])))
    images_reps[j]=images


##############################################
kdata_for_mvt_analysis_no_mvt=np.array(groupby(np.array(kdata),512,axis=1))
kdata_for_mvt_analysis_no_mvt=kdata_for_mvt_analysis_no_mvt[5]

kdata_for_mvt_analysis=np.array(groupby(np.array(kdata_mvt),512,axis=1))
kdata_for_mvt_analysis=kdata_for_mvt_analysis[5]

traj=radial_traj_3D.get_traj()
traj_for_mvt_analysis =np.array(groupby(traj,512,axis=1))
traj_for_mvt_analysis=traj_for_mvt_analysis[5]

kdata_for_mvt_analysis_filtered=np.zeros(kdata_for_mvt_analysis.shape,dtype=np.complex128)
kdata_for_mvt_analysis_no_mvt_filtered=np.zeros(kdata_for_mvt_analysis_no_mvt.shape,dtype=np.complex128)

kdata_for_mvt_analysis_filtered[:,255-wind:255+wind]=kdata_for_mvt_analysis[:,255-wind:255+wind]
kdata_for_mvt_analysis_no_mvt_filtered[:,255-wind:255+wind]=kdata_for_mvt_analysis_no_mvt[:,255-wind:255+wind]

dtheta = np.pi / nspoke
dz = 1/1

kdata_ = [k / (npoint)*dz * dtheta for k in kdata_for_mvt_analysis_filtered]

density = np.abs(np.linspace(-1, 1, npoint))
kdata_ = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata_]

images_series_rebuilt=[]
for t,s in tqdm(zip(traj_for_mvt_analysis,kdata_)):
    images_series_rebuilt.append(finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, m.image_size))
#####CORRECTION

load=False
load_maps=False

useGPU_simulation=False
useGPU_dictsearch=True

kdata = pickle.load( open( folder_3D+"kdata_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" ) )
kdata = np.array(kdata)

if not(load):
    transf=m.list_movements[0].paramDict["transformation"]
    t = m.t
    shifts = transf(t.flatten().reshape(-1,1))[:,1]

    traj=radial_traj_3D.get_traj()
    traj_for_reconstruction=radial_traj_3D.get_traj_for_reconstruction()

    perc=60
    cond=calculate_condition_mvt_correction(t,transf,perc)

    kdata_retained_final,traj_retained_final,retained_timesteps=correct_mvt_kdata(kdata,traj,cond,ntimesteps,density_adj=True)

    kdata_retained_final=kdata_retained_final.astype(np.complex128)

    size_initial = traj_for_reconstruction.size / 3
    size_retained_final = np.concatenate(traj_retained_final).shape[0]
    ratio=size_retained_final/size_initial
    print("Compression factor : {}%".format((1-ratio)*100))


    radial_traj_3D_corrected=Radial3D(total_nspokes=all_spokes,nspoke_per_z_encoding=nspoke,npoint=npoint,nb_slices=nb_total_slices,undersampling_factor=undersampling_factor)
    radial_traj_3D_corrected.traj_for_reconstruction=traj_retained_final

    volumes_corrected = simulate_radial_undersampled_images(kdata_retained_final,radial_traj_3D_corrected,m.image_size,density_adj=False,useGPU=False,is_theta_z_adjusted=True)

    with open(folder_3D+"volumes_mvt_corrected_perc{}sl{}us{}_{}.pkl".format(perc,nb_total_slices,undersampling_factor,m.name), "wb" ) as file:
        pickle.dump(volumes_corrected, file)

else:
    perc=60
    volumes_corrected=pickle.load(
        open(folder_3D+"volumes_mvt_corrected_perc{}sl{}us{}_{}.pkl".format(perc,nb_total_slices,undersampling_factor,m.name), "rb"))

#mask = build_mask_single_image(kdata_retained_final,radial_traj_3D_corrected,m.image_size,useGPU=False)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream


if not(load_maps):

    niter=0
    mask = build_mask_single_image(kdata_retained_final, radial_traj_3D_corrected, m.image_size,
                                   useGPU=useGPU_simulation)  # Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream

    mask = build_mask_single_image(kdata, radial_traj_3D, m.image_size,
                                   useGPU=useGPU_simulation)

    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj_3D, split=1000,
                                 pca=True, threshold_pca=15, useGPU_simulation=useGPU_simulation,useGPU_dictsearch=useGPU_dictsearch, log=False, useAdjPred=False, verbose=False,gen_mode="other",movement_correction=True,cond=cond)

    all_maps_mvt_corrected=optimizer.search_patterns(dictfile,volumes_corrected,retained_timesteps=retained_timesteps)
    file = open( folder_3D+"all_maps_mvt_corrected_perc{}_sl{}us{}_iter{}_{}.pkl".format(perc,nb_total_slices,undersampling_factor,niter,m.name), "wb" )
    # dump information to that file
    pickle.dump(all_maps_mvt_corrected, file)
    # close the file
    file.close()

else:
    all_maps_mvt_corrected = pickle.load(
        open(folder_3D+"all_maps_mvt_corrected_perc{}_sl{}us{}_iter{}_{}.pkl".format(perc,nb_total_slices,undersampling_factor,niter,m.name), "rb"))

#plt.close("all")

maskROI=buildROImask_unique(m.paramMap)

for iter in all_maps_mvt_corrected.keys():
    regression_paramMaps_ROI(m.paramMap, all_maps_mvt_corrected[iter][0], m.mask > 0, all_maps_mvt_corrected[iter][1] > 0,maskROI=maskROI,
                             title="Slices{}_US{} movement corrected perc {} ROI Orig vs Iteration {}".format(nb_total_slices,undersampling_factor,perc,iter), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=True)

size_slice = int(nb_slices/repeat_slice)

plot_evolution_params(m.paramMap,m.mask>0,all_maps_mvt_corrected,maskROI,save=False)


iter =2
sl = 1

compare_paramMaps_3D(m.paramMap,all_maps_mvt_corrected[iter][0],m.mask>0,all_maps_mvt_corrected[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl-1)*size_slice+int(size_slice/2),title1="Slices{}_US{} movement corrected perc{} Orig".format(nb_total_slices,undersampling_factor,perc),title2="Mid Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)
compare_paramMaps_3D(m.paramMap,all_maps_mvt_corrected[iter][0],m.mask>0,all_maps_mvt_corrected[iter][1]>0,slice=m.paramDict["nb_empty_slices"]+(sl)*size_slice,title1="Slices{}_US{} movement corrected perc{} Orig".format(nb_total_slices,undersampling_factor,perc),title2="Border Slice {} Iter {}".format(sl,iter),proj_on_mask1=True,save=True,adj_wT1=True,fat_threshold=0.7)

ani1,ani2=animate_multiple_images(volumes[:,28,:,:],volumes_corrected[:,28,:,:])





nb_total_slices=56

volumes = pickle.load(open("volumes_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb"))
volumes_mvt = pickle.load(open("volumes_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb"))
perc=60
volumes_corrected=pickle.load(
        open("volumes_mvt_corrected_perc{}sl{}us{}_{}.pkl".format(perc,nb_total_slices,undersampling_factor,m.name), "rb"))

ts=1
sl=28
plt.close("all")
plt.figure()
plt.imshow(np.abs(volumes[ts,sl,:,:]))
plt.title("No movement")
plt.figure()
plt.imshow(np.abs(volumes_mvt[ts,sl,:,:]))
plt.title("movement ")
plt.figure()
plt.imshow(np.abs(volumes_corrected[ts,sl,:,:]))
plt.title("movement corrected")


mrfdict = dictsearch.Dictionary()
mrfdict.load(dictfile, force=True)

pixel=(125,125)


niter=2
all_maps_adj = pickle.load( open("all_maps_no_mvt_sl{}us{}_iter{}_{}.pkl".format(nb_total_slices,undersampling_factor,niter,m.name), "rb" ))

it=0
maps_retrieved = all_maps_adj[it][0]
mask_retrieved = all_maps_adj[it][1]
maps_retrieved_volume_on_region = {}

for k in maps_retrieved.keys():
    maps_retrieved_volume_on_region[k]=makevol(maps_retrieved[k],mask_retrieved>0)[sl,pixel[0],pixel[1]]

map_all_on_mask = np.stack(list(maps_retrieved_volume_on_region.values())[:-1], axis=-1)
map_ff_on_mask = maps_retrieved_volume_on_region["ff"]

images_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0] * (1 - [map_ff_on_mask][i]) + mrfdict[tuple(
            pixel_params)][:, 1] * ([map_ff_on_mask][i]) for (i, pixel_params) in enumerate([map_all_on_mask])])


plt.close("all")
plt.plot(np.abs(volumes[:,sl,pixel[0],pixel[1]]),label="No movement")
plt.plot(np.abs(volumes_mvt[:,sl,pixel[0],pixel[1]]),label="Movement")
plt.plot(np.abs(volumes_corrected[:,sl,pixel[0],pixel[1]]),label="Movement corrected")
plt.plot(np.abs(images_in_mask),label="Matched Pattern no movement")
plt.legend()












nb_rep = m.paramDict["nb_rep"]

df=pd.DataFrame(columns=["rep","ts","spoke","kz","theta"],index=range(nb_rep*ntimesteps*nspoke))
df["rep"]=np.repeat(list(range(nb_rep)),ntimesteps*nspoke)
df["ts"]=list(np.repeat(list(range(ntimesteps)),(nspoke)))*nb_rep
df["spoke"]=list(range(nspoke))*nb_rep*ntimesteps

np.repeat(np.repeat(list(range(ntimesteps)),(nspoke)),nb_rep).shape
traj_for_selection = np.array(groupby(traj, npoint, axis=1))
kdata_for_selection = np.array(groupby(kdata, npoint, axis=1))
traj_for_selection = traj_for_selection.reshape(cond.shape[0], -1, 3)

df["kz"]=traj_for_selection[:,:,2][:,0]
golden_angle = 111.246*np.pi/180
df["theta"]=np.array(list(np.mod(np.arange(0,int(df.shape[0]/nb_rep))*golden_angle,np.pi))*nb_rep)

indices = np.unravel_index(np.argwhere(cond).T,(nb_rep,ntimesteps,nspoke))
retained_indices = np.squeeze(np.array(indices).T)

df_retained=df.iloc[np.nonzero(cond)]
kz_by_timestep=df_retained.groupby("ts")["kz"].unique()
theta_by_rep_timestep=df_retained.groupby(["ts","rep"])["theta"].unique()

df_retained=df_retained.join(kz_by_timestep,on="ts",rsuffix="_s")
df_retained=df_retained.join(theta_by_rep_timestep,on=["ts","rep"],rsuffix="_s")

#Theta weighting
df_retained["theta_s"]=df_retained["theta_s"].apply(lambda x:np.sort(x))
df_retained["theta_s"]=df_retained["theta_s"].apply(lambda x:np.concatenate([[x[-1]-np.pi],x,[x[0]+np.pi]]))
df_retained["theta_weight"]=(df_retained.theta-df_retained["theta_s"]).abs().apply(lambda x:np.sort(x)[1:3].mean())
df_retained.loc[df_retained["theta_weight"].isna(),"theta_weight"]=1.0
sum_weights=df_retained.groupby(["ts","rep"])["theta_weight"].sum()
df_retained=df_retained.join(sum_weights,on=["ts","rep"],rsuffix="_sum")
df_retained["theta_weight"]=df_retained["theta_weight"]/df_retained["theta_weight_sum"]


#KZ weighting
df_retained["kz_s"]=df_retained["kz_s"].apply(lambda x:np.concatenate([[-np.pi],x,[np.pi]]))
df_retained["kz_weight"]=(df_retained.kz-df_retained["kz_s"]).abs().apply(lambda x:np.sort(x)[1:3].mean())
df_retained.loc[df_retained["kz_weight"].isna(),"kz_weight"]=1.0
sum_weights=df_retained.groupby(["ts"])["kz_weight"].unique().apply(lambda x:x.sum())
df_retained=df_retained.join(sum_weights,on=["ts"],rsuffix="_sum")
df_retained["kz_weight"]=df_retained["kz_weight"]/df_retained["kz_weight_sum"]
#
# df_retained[df_retained["theta_s"].apply(lambda x:len(x)).sort_values()>5]["theta_s"].apply(lambda x:len(x)).sort_values()
# df_retained.loc[3277]
#
#
# k_max=np.pi
# #base_spoke = np.arange(-k_max, k_max, 2 * k_max / npoint, dtype=np.complex_)
# base_spoke = -k_max+np.arange(npoint)*2*k_max/(npoint-1)
# total_nspoke=1400
# all_rotations = np.exp(1j * np.mod(np.arange(total_nspoke*nb_rep) * golden_angle,2*np.pi))[3276:3280]
# all_rotations
#
#
# ts=59
# print(df_retained[df_retained.ts==ts][["ts","rep","spoke","theta","kz","theta_weight","kz_weight"]])
# rep=2
# print(df_retained[(df_retained.ts==ts)&(df_retained.rep==rep)][["theta"]]/np.pi)
# rep=2
# traj_to_plot =traj_retained_final[ts][traj_retained_final[ts][:,2]==np.unique(traj_retained_final[ts][:,2])[rep-1]]
# plt.scatter(x=traj_to_plot[:,0],y=traj_to_plot[:,1])

kdata=np.array(kdata)
traj=np.array(traj)

nb_rep = cond.shape[0]/traj.shape[0]
npoint = int(traj.shape[1] / nb_rep)
nspoke = int(traj.shape[0] / ntimesteps)

traj_for_selection = np.array(groupby(traj, npoint, axis=1))
kdata_for_selection = np.array(groupby(kdata, npoint, axis=1))

traj_for_selection = traj_for_selection.reshape(cond.shape[0], -1, 3)
kdata_for_selection = kdata_for_selection.reshape(cond.shape[0], -1)

indices = np.unravel_index(np.argwhere(cond).T,(nb_rep,ntimesteps,nspoke))
retained_indices = np.squeeze(np.array(indices).T)


print(retained_indices.shape)

traj_retained = traj_for_selection[cond, :, :]
kdata_retained = kdata_for_selection[cond, :]


golden_angle = 111.246*np.pi/180

for i, index in enumerate(indices):
    curr_slice = index[0]
    ts = index[1]
    curr_spoke = index[2]

points=traj_retained_final[0][:,:2]
vor = Voronoi(points)


new_regions = []
new_vertices = vor.vertices.tolist()

center = vor.points.mean(axis=0)
radius = vor.points.ptp().max()*2
all_ridges = {}
for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
    all_ridges.setdefault(p1, []).append((p2, v1, v2))
    all_ridges.setdefault(p2, []).append((p1, v1, v2))
all_ridges = {}

points = np.array([[0, 0], [0, 1], [0, 2],
                   [1, 0], [1, 1], [1, 2],
                   [2, 0], [2, 1], [2, 2]])


vor = Voronoi(points)

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



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
import glob
from tqdm import tqdm
import pickle

## Random map simulation

dictfile = "./mrf175_SimReco2_mid_point.dict"
dictfile = "./mrf175_SimReco2.dict"
#dictfile = "mrf175_CS.dict"

with open("./mrf_sequence.json") as f:
    sequence_config = json.load(f)

seq = T1MRF(**sequence_config)


size=(256,256)
useGPU=True

load_maps=False
save_maps = False

type="SquarePhantom"

for ph_num in tqdm([1]):
    print("##################### {} : PHANTOM {} #########################".format(type,ph_num))
    file_matlab_paramMap = "./data/{}/Phantom{}/paramMap.mat".format(type,ph_num)

    ###### Building Map
    m = MapFromFile("{}{}".format(type,ph_num), image_size=size, file=file_matlab_paramMap, rounding=True,gen_mode="loop")
    m.buildParamMap()

    if not(load_maps):

        ##### Simulating Ref Images
        m.build_ref_images(seq)

        #### Rebuilding the map from undersampled images
        ntimesteps=175
        nspoke=8
        npoint = 2*m.images_series.shape[1]

        radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint)
        kdata = m.generate_kdata(radial_traj,useGPU=True)

        volumes = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=True)
        mask = build_mask_single_image(kdata,radial_traj,m.image_size)

        # kdata_noGPU = m.generate_kdata(radial_traj, useGPU=False)
        # volumes_noGPU = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=False)

        optimizer = SimpleDictSearch(mask=mask,niter=10,seq=seq,trajectory=radial_traj,split=500,pca=True,threshold_pca=15,log=False,useAdjPred=False,useGPU_dictsearch=False,useGPU_simulation=True)
        all_maps_adj=optimizer.search_patterns(dictfile,volumes)

        if save_maps:
            file = open("all_maps_{}.pkl".format(m.name), "wb")
            # dump information to that file
            pickle.dump(all_maps_adj, file)
            # close the file
            file.close()

    else:
        with open("all_maps_{}{}.pkl".format(type,ph_num), 'rb') as f:
            all_maps_adj = pickle.load(f)
    #### Finding Matlab files
    file_names=glob.glob("./data/{}/Phantom{}/MRFmap8SpokesSVD15*".format(type,ph_num))

    all_maps_matlab = {}
    for file in file_names :
        it = int(str.split(file,"_")[1][-1])
        map_rebuilt_Matlab=MapFromFile("MapRebuiltMatlab",image_size=size,file=file,rounding=False,file_type="Result")
        map_rebuilt_Matlab.buildParamMap()
        all_maps_matlab[it]=(map_rebuilt_Matlab.paramMap,map_rebuilt_Matlab.mask)


    maskROI= buildROImask_unique(m.paramMap)
    #maskROI=buildROImask(m.paramMap)
    # plt.close("all")
    for it in [0]:#all_maps_adj.keys():
        regression_paramMaps_ROI(m.paramMap, all_maps_adj[it][0], m.mask > 0, all_maps_adj[it][1] > 0,maskROI=maskROI,
                                 title="{} {} : ROI Orig vs Python Iteration {}".format(type,ph_num,it), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,figsize=(30,15),fontsize=8,save=False)

    # plt.close("all")
    for it in [0]:#all_maps_matlab.keys():
        regression_paramMaps_ROI(m.paramMap, all_maps_matlab[it][0], m.mask > 0, all_maps_matlab[it][1] > 0,maskROI=maskROI,
                                 title="{} {} : ROI Orig vs Matlab Iteration {}".format(type,ph_num,it), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,figsize=(30,15),fontsize=8,save=False)
    # plt.close("all")

    plot_evolution_params(m.paramMap,m.mask>0,all_maps_adj,maskROI=maskROI,metric="R2",title="{} {} : Python Evolution v2".format(type,ph_num),fontsize=10,adj_wT1=True,save=True)
    plot_evolution_params(m.paramMap,m.mask>0,all_maps_matlab,maskROI=maskROI,metric="R2",title="{} {} : Matlab Evolution v2".format(type,ph_num),fontsize=10,adj_wT1=True,save=True)
    #plt.close("all")

    # for it in all_maps_adj.keys():
    #     compare_paramMaps(m.paramMap,all_maps_adj[it][0],m.mask>0,all_maps_adj[it][1]>0,adj_wT1=True,fat_threshold=0.7,title1="{} {} Orig".format(type,ph_num),title2="Python Rebuilt It {}".format(it),figsize=(30,10),fontsize=15,save=True,proj_on_mask1=True)
    #     plt.close("all")
    #
    # for it in all_maps_matlab.keys():
    #     compare_paramMaps(m.paramMap,all_maps_matlab[it][0],m.mask>0,all_maps_matlab[it][1]>0,adj_wT1=True,fat_threshold=0.7,title1="{} {} Orig".format(type,ph_num),title2="Matlab Rebuilt It {}".format(it),figsize=(30,10),fontsize=15,save=True,proj_on_mask1=True)
    #     plt.close("all")

    df_python = pd.DataFrame()
    for it in [0,5,10]:
        df_current = metrics_paramMaps_ROI(m.paramMap,all_maps_adj[it][0],m.mask>0,all_maps_adj[it][1]>0,maskROI=maskROI,adj_wT1=True, fat_threshold=0.7, proj_on_mask1=True,name="Phantom {} Python Iteration {}".format(ph_num,it))
        if df_python.empty:
            df_python = df_current
        else:
            df_python = pd.merge(df_python,df_current,left_index=True,right_index=True)

    df_matlab = pd.DataFrame()
    for it in [0,4]:
        df_current = metrics_paramMaps_ROI(m.paramMap,all_maps_matlab[it][0],m.mask>0,all_maps_matlab[it][1]>0,maskROI=maskROI,adj_wT1=True, fat_threshold=0.7, proj_on_mask1=True,name="Phantom {} Matlab Iteration {}".format(ph_num,it))
        if df_matlab.empty:
            df_matlab = df_current
        else:
            df_matlab = pd.merge(df_matlab,df_current,left_index=True,right_index=True)


    df_res = pd.merge(df_python,df_matlab,left_index=True,right_index=True)
    df_res.to_csv("{} {} : Results Summary Python vs Matlab v2.csv".format(type,ph_num))


#### COMPARISON GPU vs NO GPU FFT Python

ani=animate_images(volumes-volumes_noGPU)

diff_GPU = np.abs(volumes-volumes_noGPU)
index_max = np.unravel_index(np.argmax(diff_GPU),diff_GPU.shape)

plt.figure()
plt.plot(np.abs(volumes[:,index_max[1],index_max[2]]),label="volume undersampled GPU max diff path")
plt.plot(np.abs(volumes_noGPU[:,index_max[1],index_max[2]]),label="volume undersampled no GPU max diff path")
plt.plot(diff_GPU[:,index_max[1],index_max[2]],label="diff - max diff {}".format(np.max(diff_GPU[:,index_max[1],index_max[2]])))
plt.legend()



map_ff = makevol(m.paramMap["ff"],m.mask>0)
#mask_single_region = (np.round(map_ff,3)==0.109)
mask_single_region = np.zeros(m.image_size)
mask_single_region[133:142,138:147]=1
mask_single_region=mask_single_region>0

volume_original = m.images_series[:,mask_single_region]
volume_original = [np.mean(gp, axis=0) for gp in groupby(volume_original, 8)]
volume_original=np.array(volume_original)

undersampled_normalized = (volumes[:,mask_single_region] - np.mean(volumes[:,mask_single_region],axis=0))/np.std(volumes[:,mask_single_region],axis=0)

#pd.DataFrame(volumes[:,mask_single_region]).to_csv("Test_UndersampledSignal_Python.csv")

mean_region=np.mean((volumes[:,mask_single_region]),axis=1)
mean_region = (mean_region -np.mean(mean_region))/np.std(mean_region)

it=0
maps_retrieved = all_maps_adj[it][0]
mask_retrieved = all_maps_adj[it][1]
maps_retrieved_volume_on_region = {}

for k in maps_retrieved.keys():
    maps_retrieved_volume_on_region[k]=makevol(maps_retrieved[k],mask_retrieved>0)[mask_single_region]

map_all_on_mask = np.stack(list(maps_retrieved_volume_on_region.values())[:-1], axis=-1)
map_ff_on_mask = maps_retrieved_volume_on_region["ff"]

mrfdict = dictsearch.Dictionary()
mrfdict.load(dictfile, force=True)

images_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0] * (1 - map_ff_on_mask[i]) + mrfdict[tuple(
            pixel_params)][:, 1] * (map_ff_on_mask[i]) for (i, pixel_params) in enumerate(map_all_on_mask)])

images_in_mask = np.transpose(images_in_mask)



mean_retrieved = np.mean(images_in_mask,axis=1)
mean_retrieved=(mean_retrieved -np.mean(mean_retrieved))/np.std(mean_retrieved)

original_normalized = (volume_original - np.mean(volume_original,axis=0))/np.std(volume_original,axis=0)
mean_original = np.mean(volume_original,axis=1)
mean_original=(mean_original -np.mean(mean_original))/np.std(mean_original)


plt.figure()
plt.plot(np.real(mean_region),label='Mean signal undersampled on region')
plt.plot(np.real(mean_retrieved),label='Mean pattern retrieved on region')
plt.plot(np.real(mean_original),"x",label='Mean signal original on region')
plt.title("Mean Signals comparison in region with ff {} vs {}".format(np.round(map_ff[mask_single_region][0],4),np.round(np.mean(map_ff_on_mask),3)))
plt.legend()

###### Compare mid_point vs mean

mrfdict = dictsearch.Dictionary()
mrfdict.load("./mrf175_SimReco2_mid_point.dict", force=True)

images_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0] * (1 - map_ff_on_mask[i]) + mrfdict[tuple(
            pixel_params)][:, 1] * (map_ff_on_mask[i]) for (i, pixel_params) in enumerate(map_all_on_mask)])

images_in_mask = np.transpose(images_in_mask)

mrfdict_2 = dictsearch.Dictionary()
mrfdict_2.load("./mrf175_SimReco2.dict", force=True)

images_in_mask_2 = np.array([mrfdict_2[tuple(pixel_params)][:, 0] * (1 - map_ff_on_mask[i]) + mrfdict_2[tuple(
            pixel_params)][:, 1] * (map_ff_on_mask[i]) for (i, pixel_params) in enumerate(map_all_on_mask)])

images_in_mask_2 = np.transpose(images_in_mask_2)

num = np.unravel_index(np.argmax(np.abs(images_in_mask - images_in_mask_2)),images_in_mask.shape)[1]
plt.figure()
plt.plot(images_in_mask[:,num],label="mid_point")
plt.plot(images_in_mask_2[:,num],label="mean")
plt.legend()
#############################################


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
import glob
from tqdm import tqdm
import pickle

## Random map simulation

dictfile = "./mrf175_SimReco2.dict"
#dictfile = "mrf175_CS.dict"

with open("./mrf_sequence.json") as f:
    sequence_config = json.load(f)


sig_matlab = loadmat(r"./data/SquarePhantom/Phantom1/Sig2.mat")["Sig2"]
sig_matlab=sig_matlab.T
volumes = np.expand_dims(sig_matlab,axis=-1)
mask = np.array([[1]])


ntimesteps=175
nspoke=8
npoint = 512
radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint)

seq = T1MRF(**sequence_config)

optimizer = SimpleDictSearch(mask=mask,niter=0,seq=seq,trajectory=radial_traj,split=500,pca=True,threshold_pca=15,log=False,useAdjPred=False,useGPU=False)
all_maps_adj=optimizer.search_patterns(dictfile,volumes)


mask_single_region = mask>0

map_ff = makevol(m.paramMap["ff"],m.mask>0)
mask_single_region_all = np.zeros(m.image_size)
mask_single_region_all[133:142,138:147]=1
mask_single_region_all=mask_single_region_all>0
volume_original = m.images_series[:,mask_single_region_all]


undersampled_normalized = (volumes[:,mask_single_region] - np.mean(volumes[:,mask_single_region],axis=0))/np.std(volumes[:,mask_single_region],axis=0)
mean_region=np.mean((volumes[:,mask_single_region]),axis=1)
mean_region = (mean_region -np.mean(mean_region))/np.std(mean_region)

it=0
maps_retrieved = all_maps_adj[it][0]
mask_retrieved = all_maps_adj[it][1]
maps_retrieved_volume_on_region = {}

for k in maps_retrieved.keys():
    maps_retrieved_volume_on_region[k]=makevol(maps_retrieved[k],mask_retrieved>0)[mask_single_region]

map_all_on_mask = np.stack(list(maps_retrieved_volume_on_region.values())[:-1], axis=-1)
map_ff_on_mask = maps_retrieved_volume_on_region["ff"]

mrfdict = dictsearch.Dictionary()
mrfdict.load(dictfile, force=True)

images_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0] * (1 - map_ff_on_mask[i]) + mrfdict[tuple(
            pixel_params)][:, 1] * (map_ff_on_mask[i]) for (i, pixel_params) in enumerate(map_all_on_mask)])

images_in_mask = np.transpose(images_in_mask)
retrieved_normalized = (images_in_mask - np.mean(images_in_mask,axis=0))/np.std(images_in_mask,axis=0)

mean_retrieved = np.mean(images_in_mask,axis=1)
mean_retrieved=(mean_retrieved -np.mean(mean_retrieved))/np.std(mean_retrieved)

original_normalized = (volume_original - np.mean(volume_original,axis=0))/np.std(volume_original,axis=0)
mean_original = np.mean(volume_original,axis=1)
mean_original=(mean_original -np.mean(mean_original))/np.std(mean_original)




plt.figure()
plt.plot(np.real(mean_region),label='Mean signal undersampled on region')
plt.plot(np.real(mean_retrieved),label='Mean pattern retrieved on region')
plt.plot(np.real(mean_original),"x",label='Mean signal original on region')
plt.title("Mean Signals comparison in region with ff {} vs retrieved {}".format(np.round(map_ff[mask_single_region_all][0],4),np.round(map_ff_on_mask[0],3)))
plt.legend()


##### CHECK IDEAL IMAGE #################################



image_series_matlab = loadmat(r"./data/SquarePhantom/Phantom1/ImgSeries_ideal_iter_TestCS0.mat")["ImgSeries"]
image_series_matlab = np.moveaxis(image_series_matlab,-1,0)
image_series_matlab = np.squeeze(image_series_matlab)

image_series_python = m.images_series

metric=np.abs

python_on_mask = image_series_python[:,m.mask>0]
matlab_on_mask=image_series_matlab[:,m.mask>0]

error=np.linalg.norm(metric(python_on_mask-matlab_on_mask),axis=0)

index_max = error.argmax()
index_min = error.argmin()

plt.figure()
plt.title("Signal with max error")
plt.plot(metric(matlab_on_mask[:,index_max]),label="Matlab")
plt.plot(metric(python_on_mask[:,index_max]),label="Python")
plt.plot(metric(matlab_on_mask[:,index_max]-python_on_mask[:,index_max]),label="Diff")
plt.legend()

plt.figure()
plt.title("Signal with max error")
plt.plot(metric(matlab_on_mask[790:810,index_max]),label="Matlab")
plt.plot(metric(python_on_mask[790:810,index_max]),label="Python")
plt.legend()

print(metric(matlab_on_mask[800,index_max]))
print(metric(python_on_mask[800,index_max]))



print(m.paramMap["ff"][index_max])

plt.figure()
plt.title("Signal with min error")
plt.plot(metric(matlab_on_mask[:,index_min]),label="Matlab")
plt.plot(metric(python_on_mask[:,index_min]),label="Python")
plt.legend()


print(m.paramMap["ff"][index_min])

##########CHECK KSpace Traj############################
traj_matlab=loadmat(r"./data/SquarePhantom/Phantom1/Traj_CS.mat")["paramTraj"]
kx_matlab=traj_matlab[0][0][0]
ky_matlab = traj_matlab[0][0][1]

kx_matlab = np.moveaxis(kx_matlab,-1,0)
ky_matlab = np.moveaxis(ky_matlab,-1,0)

traj_python=radial_traj.get_traj()
traj_python=traj_python.reshape(kx_matlab.shape[0],-1,traj_python.shape[-1])
kx_python=traj_python[:,:,0]
ky_python=traj_python[:,:,1]

plt.figure()
plt.scatter(kx_python[-8:,:].flatten(),ky_python[-8:,:].flatten(),label="Python Traj")
plt.scatter(kx_matlab[-8:,:].flatten(),ky_matlab[-8:,:].flatten(),label="Matlab Traj")
plt.legend()

plt.figure()
plt.plot(np.abs(kx_python[0,:]-kx_matlab[0,:]),label="Error on base spoke")



############ CHECK KDATA###############################
kdata_matlab=loadmat(r"./data/SquarePhantom/Phantom1/KSpaceData_TestCS4.mat")["KSpaceData"]
#kdata_matlab=loadmat(r"./data/KneePhantom/Phantom1/KSpaceData_TestCS4.mat")["KSpaceData"]
kdata_matlab=np.moveaxis(kdata_matlab,-1,0)

kdata_python=np.array(kdata)

plt.close("all")

diff = np.linalg.norm(np.abs(kdata_python-kdata_matlab),axis=1)

plt.figure()
plt.plot(np.mean(np.abs(kdata_python),axis=1),label="mean kdata Python")
plt.plot(np.mean(np.abs(kdata_matlab),axis=1),label="mean kdata Matlab")
plt.legend()

plt.figure()
plt.plot(diff,label="diff python vs matlab kdata over timesteps")
plt.legend()



index_max = np.unravel_index(diff.argmax(),diff.shape)

diff_list=list(diff)
del diff_list[index_max[0]]
index_2nd = np.unravel_index(np.array(diff_list).argmax(),np.array(diff_list).shape)
del diff_list[index_2nd[0]]
diff_adj = np.array(diff_list)

plt.figure()
plt.plot(diff_adj,label="diff without outlier python vs matlab kdata over timesteps")
plt.legend()

plt.figure()
plt.plot(np.abs(kdata_python[index_max[0],:]),label="kdata Python max diff")
plt.plot(np.abs(kdata_matlab[index_max[0],:]),label="kdata Matlab max diff")
plt.legend()

timestep=0
plt.figure()
plt.plot(np.abs(kdata_matlab[timestep,:]),label="Kdata Matlab")
plt.plot(np.abs(kdata_python[timestep,:]),label="Kdata Python")
plt.legend()



kdata[0]
kdata_matlab[0]
############ CHECK UNDERSAMPLED IMAGE###############################
import h5py

f = h5py.File(r"./data/SquarePhantom/Phantom1/ImgSeries8Spokes_block_iter0.mat","r")
f = h5py.File(r"./data/KneePhantom/Phantom1/ImgSeries8Spokes_block_iter0_2.mat","r")
volumes_matlab = f.get('ImgSeries')
volumes_matlab = np.squeeze(np.array(volumes_matlab).view("complex"))

volumes_matlab=loadmat(r"./data/KneePhantom/Phantom1/ImgSeries8Spokes_block.mat")["ImgSeries"]
volumes_matlab=np.moveaxis(volumes_matlab,-1,0)

timestep = 0

image_python=volumes[timestep,:,:]
image_matlab=volumes_matlab[timestep,:,:]
image_original=m.images_series[timestep,:,:]

plt.figure()
plt.imshow(np.abs(image_python))

plt.figure()
plt.imshow(np.abs(image_original))


plt.figure()
plt.imshow(np.abs(image_matlab))
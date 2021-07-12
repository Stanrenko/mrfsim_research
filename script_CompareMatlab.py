
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

seq = T1MRF(**sequence_config)

window = 8 #corresponds to nspoke by image
size=(256,256)
useGPU=True

load_maps=True
for ph_num in tqdm([1,2,3,4,5]):
    print("##################### PHANTOM {} #########################".format(ph_num))
    file_matlab_paramMap = "./data/SquarePhantom/Phantom{}/paramMap.mat".format(ph_num)

    ###### Building Map
    m = MapFromFile("PythonPhantom{}".format(ph_num), image_size=size, file=file_matlab_paramMap, rounding=False)
    m.buildParamMap()

    if not(load_maps):

        ##### Simulating Ref Images
        m.build_ref_images(seq,window)

        #### Rebuilding the map from undersampled images
        ntimesteps=175
        nspoke=8
        npoint = 2*m.images_series.shape[1]

        radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint)
        kdata = m.generate_radial_kdata(radial_traj,useGPU=useGPU)

        volumes = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=useGPU)
        mask = build_mask_single_image(kdata,radial_traj,m.image_size)

        optimizer = SimpleDictSearch(mask=mask,niter=10,seq=seq,trajectory=radial_traj,split=500,pca=True,threshold_pca=15,log=False,useAdjPred=False,useGPU=useGPU)
        all_maps_adj=optimizer.search_patterns(dictfile,volumes)

        file = open("all_maps_{}.pkl".format(m.name), "wb")
        # dump information to that file
        pickle.dump(all_maps_adj, file)
        # close the file
        file.close()

    else:
        with open("all_maps_PythonPhantom{}.pkl".format(ph_num), 'rb') as f:
            all_maps_adj = pickle.load(f)
    #### Finding Matlab files
    file_names=glob.glob("./data/SquarePhantom/Phantom{}/MRFmap8SpokesSVD15*".format(ph_num))

    all_maps_matlab = {}
    for file in file_names :
        it = int(str.split(file,"_")[1][-1])
        map_rebuilt_Matlab=MapFromFile("MapRebuiltMatlab",image_size=size,file=file,rounding=False,file_type="Result")
        map_rebuilt_Matlab.buildParamMap()
        all_maps_matlab[it]=(map_rebuilt_Matlab.paramMap,map_rebuilt_Matlab.mask)


    maskROI= buildROImask_unique(m.paramMap)

    # plt.close("all")
    # for it in all_maps_adj.keys():
    #     regression_paramMaps_ROI(m.paramMap, all_maps_adj[it][0], m.mask > 0, all_maps_adj[it][1] > 0,maskROI=maskROI,
    #                              title="Phantom {} : ROI Orig vs Python Iteration {}".format(ph_num,it), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,figsize=(30,15),fontsize=8,save=True)
    #
    # plt.close("all")
    # for it in all_maps_matlab.keys():
    #     regression_paramMaps_ROI(m.paramMap, all_maps_matlab[it][0], m.mask > 0, all_maps_matlab[it][1] > 0,maskROI=maskROI,
    #                              title="Phantom {} : ROI Orig vs Matlab Iteration {}".format(ph_num,it), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,figsize=(30,15),fontsize=8,save=True)
    # plt.close("all")

    plot_evolution_params(m.paramMap,m.mask>0,all_maps_adj,maskROI=maskROI,metric="R2",title="Phantom {} : Python Evolution".format(ph_num),fontsize=10,adj_wT1=True,save=True)
    plot_evolution_params(m.paramMap,m.mask>0,all_maps_matlab,maskROI=maskROI,metric="R2",title="Phantom {} : Matlab Evolution".format(ph_num),fontsize=10,adj_wT1=True,save=True)
    #plt.close("all")

    # for it in all_maps_adj.keys():
    #     compare_paramMaps(m.paramMap,all_maps_adj[it][0],m.mask>0,all_maps_adj[it][1]>0,adj_wT1=True,fat_threshold=0.7,title1="{} Orig".format(ph_num),title2="Python Rebuilt It {}".format(it),figsize=(30,10),fontsize=15,save=True,proj_on_mask1=True)
    #     plt.close("all")
    #
    # for it in all_maps_matlab.keys():
    #     compare_paramMaps(m.paramMap,all_maps_matlab[it][0],m.mask>0,all_maps_matlab[it][1]>0,adj_wT1=True,fat_threshold=0.7,title1="{} Orig".format(ph_num),title2="Matlab Rebuilt It {}".format(it),figsize=(30,10),fontsize=15,save=True,proj_on_mask1=True)
    #     plt.close("all")

    # df_python = pd.DataFrame()
    # for it in [0,5,10]:
    #     df_current = metrics_paramMaps_ROI(m.paramMap,all_maps_adj[it][0],m.mask>0,all_maps_adj[it][1]>0,maskROI=maskROI,adj_wT1=True, fat_threshold=0.7, proj_on_mask1=True,name="Phantom {} Python Iteration {}".format(ph_num,it))
    #     if df_python.empty:
    #         df_python = df_current
    #     else:
    #         df_python = pd.merge(df_python,df_current,left_index=True,right_index=True)
    #
    # df_matlab = pd.DataFrame()
    # for it in [0,4]:
    #     df_current = metrics_paramMaps_ROI(m.paramMap,all_maps_matlab[it][0],m.mask>0,all_maps_matlab[it][1]>0,maskROI=maskROI,adj_wT1=True, fat_threshold=0.7, proj_on_mask1=True,name="Phantom {} Matlab Iteration {}".format(ph_num,it))
    #     if df_matlab.empty:
    #         df_matlab = df_current
    #     else:
    #         df_matlab = pd.merge(df_matlab,df_current,left_index=True,right_index=True)
    #
    #
    # df_res = pd.merge(df_python,df_matlab,left_index=True,right_index=True)
    # df_res.to_csv("Phantom {} : Results Summary Python vs Matlab.csv".format(ph_num))

from sigpy.mri import spiral
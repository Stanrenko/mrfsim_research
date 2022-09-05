
### Movements simulation

#import matplotlib
#matplotlib.u<se("TkAgg")
import numpy as np
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,GaussianWeighting,BruteDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat
from mutools import io
from sklearn import linear_model
from scipy.optimize import minimize
from movements import TranslationBreathing

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./2D"

dictfile = "mrf175_SimReco2_light.dict"
dictjson="mrf_dictconf_SimReco2_light.json"
#dictfile = "mrf175_SimReco2.dict"
#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
#dictfile = "mrf175_Dico2_Invivo.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)

seq = T1MRF(**sequence_config)


nb_phantom = 5

ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 512

image_size = (int(npoint/2), int(npoint/2))
nspoke=int(nb_allspokes/ntimesteps)

with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.01,0.01)

region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
mask_reduction_factor=1/4


name = "SquareSimu2D"


use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

folder = "/".join(str.split(filename,"/")[:-1])

all_results_brute={}
all_results_cf={}
all_results_matrix={}


for num in range(nb_phantom):
    print("################## PROCESSING Phantom {} ##################".format(num))
    filename_paramMap=filename+"_paramMap_{}.pkl".format(num)
    filename_paramMask=filename+"_paramMask_{}.npy".format(num)
    filename_groundtruth = filename+"_groundtruth_{}.npy".format(num)

    filename_kdata = filename+"_kdata_{}.npy".format(num)

    filename_volume = filename+"_volumes_{}.npy".format(num)
    filename_mask= filename+"_mask_{}.npy".format(num)
    file_map_brute = filename + "_{}_brute_MRF_map.pkl".format(num)
    file_map_matrix = filename + "_{}_matrix_MRF_map.pkl".format(num)
    file_map_cf = filename + "_{}_cf_MRF_map.pkl".format(num)



    #filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"




    m_ = RandomMap(name,dict_config,resting_time=4000,image_size=image_size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)



    if str.split(filename_paramMap,"/")[-1] not in os.listdir(folder):
        m_.buildParamMap()
        with open(filename_paramMap, "wb" ) as file:
            pickle.dump(m_.paramMap, file)

        map_rebuilt = m_.paramMap
        mask = m_.mask

        keys_simu = list(map_rebuilt.keys())
        values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
        map_for_sim = dict(zip(keys_simu, values_simu))

        np.save(filename_paramMask,mask)

        for key in ["ff", "wT1", "df", "attB1"]:
            file_mha = "/".join(["/".join(str.split(filename_paramMap, "/")[:-1]),
                                 "_".join(str.split(str.split(filename_paramMap, "/")[-1], ".")[:-1])]) + "_{}.mha".format(
                                                                                                                       key)
            io.write(file_mha, map_for_sim[key], tags={"spacing": [5, 1, 1]})

    else:
        with open(filename_paramMap, "rb") as file:
            m_.paramMap=pickle.load(file)
        m_.mask=np.load(filename_paramMask)



    m_.build_ref_images(seq)

    if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
        np.save(filename_groundtruth,m_.images_series[::nspoke])

    radial_traj=Radial(total_nspokes=nb_allspokes,npoint=npoint)

    if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):

        #images = copy(m.images_series)
        data=m_.generate_kdata(radial_traj,useGPU=use_GPU)

        data=np.array(data)
        np.save(filename_kdata, data)

    else:
        #kdata_all_channels_all_slices = open_memmap(filename_kdata)
        data = np.load(filename_kdata)


    data=data.reshape(nb_allspokes,-1,npoint)
    b1_all_slices = np.ones(image_size)
    b1_all_slices=np.expand_dims(b1_all_slices,axis=0)

    ##volumes for slice taking into account coil sensi
    print("Building Volumes....")
    if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
        volumes_all=simulate_radial_undersampled_images(data,radial_traj,image_size,density_adj=True,useGPU=use_GPU)
        np.save(filename_volume,volumes_all)
        # sl=20
        # ani = animate_images(volumes_all[:,sl,:,:])
        del volumes_all

    # volumes_all=np.load(filename_volume)
    # ani = animate_images(volumes_all[:,:,:])

    del data
    del b1_all_slices



    ########################## Dict mapping ########################################

    with open("mrf_sequence.json") as f:
        sequence_config = json.load(f)

    seq = T1MRF(**sequence_config)

    load_map=False
    save_map=True

    dictfile = "mrf175_SimReco2_light.dict"
    #dictfile = "mrf175_SimReco2_window_1.dict"
    #dictfile = "mrf175_SimReco2_window_21.dict"
    #dictfile = "mrf175_SimReco2_window_55.dict"
    #dictfile = "mrf175_Dico2_Invivo.dict"
    #filename_volume_corrected_final = str.split(filename,".dat") [0]+"_volumes_corrected_final_old{}.npy".format("")

    mask=m_.mask
    volumes_all = np.load(filename_volume)

    ntimesteps=175


    if str.split(file_map_cf,"/")[-1] not in os.listdir(folder):
        niter = 0
        optimizer = BruteDictSearch(FF_list=np.arange(0,1.05,0.05),mask=mask,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=False,ntimesteps=ntimesteps,log_phase=True)
        all_maps_brute = optimizer.search_patterns(dictfile, volumes_all, retained_timesteps=None)


        optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=100, pca=True,
                                     threshold_pca=20, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps)
        all_maps_cf = optimizer.search_patterns_test(dictfile, volumes_all, retained_timesteps=None)
        all_maps_matrix = optimizer.search_patterns_matrix(dictfile, volumes_all, retained_timesteps=None)

        with open(file_map_brute,"wb") as file:
            pickle.dump(all_maps_brute, file)
        with open(file_map_cf,"wb") as file:
            pickle.dump(all_maps_cf, file)
        with open(file_map_matrix,"wb") as file:
            pickle.dump(all_maps_matrix, file)

    else:
        with open(file_map_brute,"rb") as file:
            all_maps_brute=pickle.load(file)
        with open(file_map_cf,"rb") as file:
            all_maps_cf=pickle.load(file)
        with open(file_map_matrix,"rb") as file:
            all_maps_matrix=pickle.load(file)


    maskROI=buildROImask_unique(m_.paramMap)

    regression_paramMaps_ROI(m_.paramMap,all_maps_brute[0][0],mask>0,all_maps_brute[0][1]>0,maskROI,adj_wT1=False,title="Brute_regROI_"+str.split(str.split(filename_volume,"/")[-1],".npy")[0],save=True)
    regression_paramMaps_ROI(m_.paramMap,all_maps_cf[0][0],mask>0,all_maps_cf[0][1]>0,maskROI,adj_wT1=False,title="CF_regROI_"+str.split(str.split(filename_volume,"/")[-1],".npy")[0],save=True)
    regression_paramMaps_ROI(m_.paramMap,all_maps_matrix[0][0],mask>0,all_maps_matrix[0][1]>0,maskROI,adj_wT1=False,title="Matrix_regROI_"+str.split(str.split(filename_volume,"/")[-1],".npy")[0],save=True)

    results = get_ROI_values(m_.paramMap,all_maps_brute[0][0],mask>0,all_maps_brute[0][1]>0,maskROI=maskROI,kept_keys=["attB1","df","wT1","ff"],adj_wT1=False,fat_threshold=0.7)

    if all_results_brute=={}:
        all_results_brute=results
    else:
        for k in all_results_brute.keys():
            all_results_brute[k]=np.concatenate([results[k],all_results_brute[k]],axis=0)

    results = get_ROI_values(m_.paramMap, all_maps_cf[0][0], mask > 0, all_maps_cf[0][1] > 0, maskROI=maskROI,
                             kept_keys=["attB1", "df", "wT1", "ff"], adj_wT1=False, fat_threshold=0.7)

    if all_results_cf == {}:
        all_results_cf = results
    else:
        for k in all_results_cf.keys():
            all_results_cf[k] = np.concatenate([results[k], all_results_cf[k]], axis=0)

    results = get_ROI_values(m_.paramMap, all_maps_matrix[0][0], mask > 0, all_maps_matrix[0][1] > 0, maskROI=maskROI,
                             kept_keys=["attB1", "df", "wT1", "ff"], adj_wT1=False, fat_threshold=0.7)
    if all_results_matrix == {}:
        all_results_matrix = results
    else:
        for k in all_results_matrix.keys():
            all_results_matrix[k] = np.concatenate([results[k], all_results_matrix[k]], axis=0)


df_comp=pd.DataFrame(columns=["FF Ground Truth","FF reference","FF matrix","FF proposed"])
df_comp["FF Ground Truth"]=all_results_brute["ff"][all_results_brute["ff"][:,0]<0.7,0]
df_comp["FF reference"]=all_results_brute["ff"][all_results_brute["ff"][:,0]<0.7,1]
df_comp["FF matrix"]=all_results_matrix["ff"][all_results_matrix["ff"][:,0]<0.7,1]
df_comp["FF proposed"]=all_results_cf["ff"][all_results_cf["ff"][:,0]<0.7,1]



import seaborn as sns
g=sns.pairplot(df_comp,diag_kind="kde",kind="reg",plot_kws={'line_kws':{'color':'red',"alpha":0.5},"scatter_kws":{"s":3}},corner=True)
g.fig.suptitle("FF methods comparison")


df_comp=pd.DataFrame(columns=["$T1_{H2O}$ Ground Truth","$T1_{H2O}$ reference","$T1_{H2O}$ matrix","$T1_{H2O}$ proposed"])
df_comp["$T1_{H2O}$ Ground Truth"]=all_results_brute["wT1"][all_results_brute["ff"][:,0]<0.7,0]
df_comp["$T1_{H2O}$ reference"]=all_results_brute["wT1"][all_results_brute["ff"][:,0]<0.7,1]
df_comp["$T1_{H2O}$ matrix"]=all_results_matrix["wT1"][all_results_matrix["ff"][:,0]<0.7,1]
df_comp["$T1_{H2O}$ proposed"]=all_results_cf["wT1"][all_results_cf["ff"][:,0]<0.7,1]



import seaborn as sns
g=sns.pairplot(df_comp,diag_kind="kde",kind="reg",plot_kws={'line_kws':{'color':'red',"alpha":0.5},"scatter_kws":{"s":3}},corner=True)
g.fig.suptitle("$T1_{H2O}$ methods comparison")




#
#
# curr_file=file_map
# file = open(curr_file, "rb")
# all_maps = pickle.load(file)
# file.close()
# for iter in list(all_maps.keys()):
#
#     map_rebuilt=all_maps[iter][0]
#     mask=all_maps[iter][1]
#
#     keys_simu = list(map_rebuilt.keys())
#     values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
#     map_for_sim = dict(zip(keys_simu, values_simu))
#
#     #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
#     #map_Python.buildParamMap()
#
#
#     for key in ["ff","wT1","df","attB1"]:
#         file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_it{}_{}.mha".format(iter,key)
#         io.write(file_mha,map_for_sim[key],tags={"spacing":[5,1,1]})
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


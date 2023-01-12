

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

dictfile = "mrf175_SimReco2_adjusted.dict"
dictjson="mrf_dictconf_SimReco2.json"
dictfile_light='mrf175_SimReco2_light_matching_adjusted.dict'
#dictfile = "mrf175_SimReco2.dict"
#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
dictfile = "mrf175_Dico2_Invivo.dict"
dictfile_light="mrf175_Dico2_Invivo_light_for_matching.dict"
dictjson="mrf_dictconf_Dico2_Invivo.json"


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

snr=None
name = "SquareSimu2D_SimReco2"
name = "SquareSimu2D_Dico2Invivo"


use_GPU = True
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
    filename_mask = filename + "_mask_{}.npy".format(num)

    if snr is not None:
        filename_volume = filename+"_volumes_{}_snr{}.npy".format(num,snr)

        file_map_brute = filename + "_{}_snr{}_brute_MRF_map.pkl".format(num,snr)
        file_map_matrix = filename + "_{}_snr{}_matrix_MRF_map.pkl".format(num,snr)
        file_map_cf = filename + "_{}_snr{}_cf_MRF_map.pkl".format(num,snr)
    else:
        filename_volume = filename + "_volumes_{}.npy".format(num)

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
        data=m_.generate_kdata(radial_traj,useGPU=False)

        data=np.array(data)
        np.save(filename_kdata, data)

    else:
        #kdata_all_channels_all_slices = open_memmap(filename_kdata)
        data = np.load(filename_kdata)

    data=data.reshape(nb_allspokes,npoint)
    if snr is not None:
        center_point = int(npoint / 2)
        res = int(npoint / 16)
        mean_data = np.mean(
            np.abs(data[:, (center_point - res):(center_point + res)]))
        noise = mean_data / snr * (np.random.normal(size=data.shape) + 1j * np.random.normal(size=data.shape))
        data+=noise
    data=np.expand_dims(data,axis=0)
    b1_all_slices = np.ones(image_size)
    b1_all_slices=np.expand_dims(b1_all_slices,axis=0)

    ##volumes for slice taking into account coil sensi
    print("Building Volumes....")
    if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
        volumes_all=simulate_radial_undersampled_images_multi(data,radial_traj,image_size,density_adj=True,useGPU=False,b1=b1_all_slices,normalize_iterative=True)
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

    #dictfile = "mrf175_SimReco2_light.dict"
    #dictfile = "mrf175_Dico2_Invivo_adjusted.dict"
    #dictfile = "mrf175_SimReco2_window_1.dict"
    #dictfile = "mrf175_SimReco2_window_21.dict"
    #dictfile = "mrf175_SimReco2_window_55.dict"
    #dictfile = "mrf175_Dico2_Invivo.dict"
    #filename_volume_corrected_final = str.split(filename,".dat") [0]+"_volumes_corrected_final_old{}.npy".format("")

    mask=m_.mask
    volumes_all = np.load(filename_volume)

    ntimesteps=175
    niter = 0

    optimizer_brute = BruteDictSearch(FF_list=np.arange(0,1.01,0.05),mask=mask,split=1,pca=True,threshold_pca=15,log=False,useGPU_dictsearch=use_GPU,ntimesteps=ntimesteps,log_phase=True,n_clusters_dico=1000,pruning=0.05)
    # all_maps = optimizer.search_patterns(dictfile, volumes_all, retained_timesteps=None)

    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=100, pca=True,
                                 threshold_pca=15, log=True, useGPU_dictsearch=use_GPU, useGPU_simulation=False,
                                 gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                 threshold_ff=0.9, dictfile_light=dictfile_light)

    optimizer_clustering = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=100, pca=True,
                                 threshold_pca=15, log=True, useGPU_dictsearch=use_GPU, useGPU_simulation=False,
                                 gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                 threshold_ff=0.9, dictfile_light=dictfile_light)
    # all_maps = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all, retained_timesteps=None)

    # optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=100, pca=True,
    #                              threshold_pca=20, log=False, useGPU_dictsearch=use_GPU, useGPU_simulation=False,
    #                              gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps)
    #
    # optimizer_brute = BruteDictSearch(FF_list=np.arange(0, 1.05, 0.05), mask=mask, split=100, pca=True, threshold_pca=20,
    #                             log=False, useGPU_dictsearch=False, ntimesteps=ntimesteps, log_phase=True)

    if str.split(file_map_cf,"/")[-1] not in os.listdir(folder):
        all_maps_cf = optimizer.search_patterns_test_multi(dictfile, volumes_all, retained_timesteps=None)
        with open(file_map_cf,"wb") as file:
            pickle.dump(all_maps_cf, file)

    else:
        with open(file_map_cf,"rb") as file:
            all_maps_cf=pickle.load(file)

    if str.split(file_map_brute, "/")[-1] not in os.listdir(folder):
        all_maps_brute = optimizer_brute.search_patterns(dictfile, volumes_all, retained_timesteps=None)

        with open(file_map_brute,"wb") as file:
            pickle.dump(all_maps_brute, file)

    else:
        with open(file_map_brute,"rb") as file:
            all_maps_brute=pickle.load(file)

    if str.split(file_map_matrix, "/")[-1] not in os.listdir(folder):
        all_maps_matrix = optimizer_clustering.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all, retained_timesteps=None)

        with open(file_map_matrix,"wb") as file:
            pickle.dump(all_maps_matrix, file)

    else:

        with open(file_map_matrix,"rb") as file:
            all_maps_matrix=pickle.load(file)


    maskROI=m_.buildROImask()

    regression_paramMaps_ROI(m_.paramMap,all_maps_brute[0][0],mask>0,all_maps_brute[0][1]>0,maskROI,adj_wT1=True,title="Brute_regROI_"+str.split(str.split(filename_volume,"/")[-1],".npy")[0],save=True)
    regression_paramMaps_ROI(m_.paramMap,all_maps_cf[0][0],mask>0,all_maps_cf[0][1]>0,maskROI,adj_wT1=True,title="CF_regROI_"+str.split(str.split(filename_volume,"/")[-1],".npy")[0],save=True)
    regression_paramMaps_ROI(m_.paramMap,all_maps_matrix[0][0],mask>0,all_maps_matrix[0][1]>0,maskROI,adj_wT1=True,title="CF_Clustering_regROI_"+str.split(str.split(filename_volume,"/")[-1],".npy")[0],save=True)

    results = get_ROI_values(m_.paramMap,all_maps_brute[0][0],mask>0,all_maps_brute[0][1]>0,maskROI=maskROI,kept_keys=["attB1","df","wT1","ff"],adj_wT1=False,fat_threshold=0.7,return_std=True)

    if all_results_brute=={}:
        all_results_brute=results
    else:
        for k in all_results_brute.keys():
            all_results_brute[k]=np.concatenate([results[k],all_results_brute[k]],axis=0)

    results = get_ROI_values(m_.paramMap, all_maps_cf[0][0], mask > 0, all_maps_cf[0][1] > 0, maskROI=maskROI,
                             kept_keys=["attB1", "df", "wT1", "ff"], adj_wT1=False, fat_threshold=0.7,return_std=True)

    if all_results_cf == {}:
        all_results_cf = results
    else:
        for k in all_results_cf.keys():
            all_results_cf[k] = np.concatenate([results[k], all_results_cf[k]], axis=0)

    results = get_ROI_values(m_.paramMap, all_maps_matrix[0][0], mask > 0, all_maps_matrix[0][1] > 0, maskROI=maskROI,
                             kept_keys=["attB1", "df", "wT1", "ff"], adj_wT1=False, fat_threshold=0.7,return_std=True)
    if all_results_matrix == {}:
        all_results_matrix = results
    else:
        for k in all_results_matrix.keys():
            all_results_matrix[k] = np.concatenate([results[k], all_results_matrix[k]], axis=0)

df_comp=pd.DataFrame(columns=["Reference","Proposed","Proposed with clustering"])
df_comp["Reference"]=all_results_brute["ff"][:,2]-all_results_brute["ff"][:,0]
df_comp["Proposed with clustering"]=all_results_matrix["ff"][:,2]-all_results_brute["ff"][:,0]
df_comp["Proposed"]=all_results_cf["ff"][:,2]-all_results_brute["ff"][:,0]

fig=plt.figure()
ax=df_comp.boxplot(grid=False,showfliers=False,showmeans=True,medianprops=dict(color="red"),boxprops=dict(color="black"),whiskerprops=dict(color="black"),meanprops=dict(marker="x",markeredgecolor="gray"),whis=[5,95])
ax.set_ylabel("FF Error vs ground truth",fontsize=14)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
fig.patch.set_facecolor("white")
fig.set_tight_layout(True)


fig.savefig("boxplot ff numerical errors.png".format(k),dpi=600)


plt.close("all")


df_comp=pd.DataFrame(columns=["Reference","Proposed","Proposed with clustering"])
df_comp["Reference"]=all_results_brute["wT1"][all_results_brute["ff"][:,0]<0.7,2]-all_results_brute["wT1"][all_results_brute["ff"][:,0]<0.7,0]
df_comp["Proposed with clustering"]=all_results_matrix["wT1"][all_results_matrix["ff"][:,0]<0.7,2]-all_results_brute["wT1"][all_results_brute["ff"][:,0]<0.7,0]
df_comp["Proposed"]=all_results_cf["wT1"][all_results_cf["ff"][:,0]<0.7,2]-all_results_brute["wT1"][all_results_brute["ff"][:,0]<0.7,0]

fig=plt.figure()
ax=df_comp.boxplot(grid=False,showfliers=False,showmeans=True,medianprops=dict(color="red"),boxprops=dict(color="black"),whiskerprops=dict(color="black"),meanprops=dict(marker="x",markeredgecolor="gray"),whis=[5,95])
ax.set_ylabel("$T1_{H2O}$ Error vs ground truth (ms)",fontsize=14)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
fig.patch.set_facecolor("white")
fig.set_tight_layout(True)


fig.savefig("boxplot T1 numerical errors.png".format(k),dpi=600)


df_comp=pd.DataFrame(columns=["Reference","Proposed","Proposed with clustering"])
df_comp["Reference"]=all_results_brute["df"][:,2]-all_results_brute["df"][:,0]
df_comp["Proposed with clustering"]=all_results_matrix["df"][:,2]-all_results_brute["df"][:,0]
df_comp["Proposed"]=all_results_cf["df"][:,2]-all_results_brute["df"][:,0]

df_comp=df_comp*1000

fig=plt.figure()
ax=df_comp.boxplot(grid=False,showfliers=False,showmeans=True,medianprops=dict(color="red"),boxprops=dict(color="black"),whiskerprops=dict(color="black"),meanprops=dict(marker="x",markeredgecolor="gray"),whis=[5,95])
ax.set_ylabel("Df Error vs ground truth (Hz)",fontsize=14)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
fig.patch.set_facecolor("white")
fig.set_tight_layout(True)


fig.savefig("boxplot df numerical errors.png".format(k),dpi=600)


df_comp=pd.DataFrame(columns=["Reference","Proposed","Proposed with clustering"])
df_comp["Reference"]=all_results_brute["attB1"][:,2]-all_results_brute["attB1"][:,0]
df_comp["Proposed with clustering"]=all_results_matrix["attB1"][:,2]-all_results_brute["attB1"][:,0]
df_comp["Proposed"]=all_results_cf["attB1"][:,2]-all_results_brute["attB1"][:,0]

fig=plt.figure()
ax=df_comp.boxplot(grid=False,showfliers=False,showmeans=True,medianprops=dict(color="red"),boxprops=dict(color="black"),whiskerprops=dict(color="black"),meanprops=dict(marker="x",markeredgecolor="gray"),whis=[5,95])
ax.set_ylabel("B1 Error vs ground truth",fontsize=14)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
fig.patch.set_facecolor("white")
fig.set_tight_layout(True)

fig.savefig("boxplot attB1 numerical errors.png".format(k),dpi=600)


df_std=pd.DataFrame(columns=["Reference","Proposed","Proposed with clustering"])
df_std["Reference"]=all_results_brute["wT1"][all_results_matrix["ff"][:,0]<0.7,3]
df_std["Proposed with clustering"]=all_results_matrix["wT1"][all_results_matrix["ff"][:,0]<0.7,3]
df_std["Proposed"]=all_results_cf["wT1"][all_results_matrix["ff"][:,0]<0.7,3]


fig=plt.figure()
ax=df_std.boxplot(grid=False,showfliers=False,showmeans=True,medianprops=dict(color="red"),boxprops=dict(color="black"),whiskerprops=dict(color="black"),meanprops=dict(marker="x",markeredgecolor="gray"),whis=[5,95])
ax.set_ylabel("$T1_{H2O}$ Std per ROI (ms)",fontsize=14)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
fig.patch.set_facecolor("white")
fig.set_tight_layout(True)

fig.savefig("boxplot std T1 numerical phantoms.png".format(k),dpi=600)


df_std=pd.DataFrame(columns=["Reference","Proposed","Proposed with clustering"])
df_std["Reference"]=all_results_brute["ff"][:,3]
df_std["Proposed with clustering"]=all_results_matrix["ff"][:,3]
df_std["Proposed"]=all_results_cf["ff"][:,3]


fig=plt.figure()
ax=df_std.boxplot(grid=False,showfliers=False,showmeans=True,medianprops=dict(color="red"),boxprops=dict(color="black"),whiskerprops=dict(color="black"),meanprops=dict(marker="x",markeredgecolor="gray"),whis=[5,95])
ax.set_ylabel("FF Std per ROI",fontsize=14)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
fig.patch.set_facecolor("white")
fig.set_tight_layout(True)

fig.savefig("boxplot std ff numerical phantoms.png".format(k),dpi=600)


import statsmodels.api as sm
plt.close("all")
sm.graphics.mean_diff_plot(df_comp["$T1_{H2O}$ reference"], df_comp["$T1_{H2O}$ matrix"])
plt.title("$T1_{H2O}$ : Comparison reference vs matrix method",fontsize=13)

sm.graphics.mean_diff_plot(df_comp["$T1_{H2O}$ reference"], df_comp["$T1_{H2O}$ proposed"])
plt.title("$T1_{H2O}$ : Comparison reference vs proposed method",fontsize=13)

sm.graphics.mean_diff_plot(df_comp["$T1_{H2O}$ proposed"], df_comp["$T1_{H2O}$ matrix"])
plt.title("$T1_{H2O}$ : Comparison proposed vs matrix method",fontsize=13)


plt.close("all")
sm.graphics.mean_diff_plot(df_comp["$T1_{H2O}$ Ground Truth"], df_comp["$T1_{H2O}$ matrix"])
plt.title("$T1_{H2O}$ : Comparison ground truth vs BC Clustering method",fontsize=13)
plt.show()

sm.graphics.mean_diff_plot(df_comp["$T1_{H2O}$ Ground Truth"], df_comp["$T1_{H2O}$ proposed"])
plt.title("$T1_{H2O}$ : Comparison ground truth vs BC method",fontsize=13)
plt.show()
sm.graphics.mean_diff_plot(df_comp["$T1_{H2O}$ Ground Truth"], df_comp["$T1_{H2O}$ reference"])
plt.title("$T1_{H2O}$ : Comparison ground truth vs Brute method",fontsize=13)
plt.show()

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


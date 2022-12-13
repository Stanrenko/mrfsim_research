
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF

from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time

import pickle

from mutools import io



filename="./data/InVivo/3D/20211105_TestCS_MRF/meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"
filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00044_FID42066_raFin_3D_tra_1x1x5mm_us4_vivo.dat"
filename="./data/InVivo/3D/20211122_EV_MRF/meas_MID00146_FID42269_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"
filename="./data/InVivo/3D/20211122_EV_MRF/meas_MID00149_FID42272_raFin_3D_tra_1x1x5mm_USx2.dat"
filename="./data/InVivo/3D/20211122_EV_MRF/meas_MID00147_FID42270_raFin_3D_tra_1x1x5mm_FULL_incoherent.dat"
filename="./data/InVivo/3D/20211122_EV_MRF/meas_MID00148_FID42271_raFin_3D_tra_1x1x5mm_FULL_high_res.dat"
filename="./data/InVivo/3D/20211129_BM/meas_MID00085_FID43316_raFin_3D_FULL_highRES_incoh.dat"
filename="./data/InVivo/3D/20211105_TestCS_MRF/meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"
filename="./data/InVivo/3D/20211105_TestCS_MRF/meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro_volumes_norm_vol.dat"
filename="./data/InVivo/3D/20211122_EV_MRF/meas_MID00146_FID42269_raFin_3D_tra_1x1x5mm_FULL_vitro_test.dat"
#filename="./data/InVivo/3D/20211209_AL_Tongue/meas_MID00258_FID45162_raFin_3D_tra_1x1x5mm_FULl.dat"
filename="./data/InVivo/3D/20211220_Phantom_MRF/meas_MID00026_FID47383_raFin_3D_tra_1x1x5mm_FULl.dat"
filename="./data/InVivo/3D/20220106/meas_MID00167_FID48477_raFin_3D_tra_1x1x5mm_FULL_new.dat"
filename="./data/InVivo/3D/20220113_CS/meas_MID00163_FID49558_raFin_3D_tra_1x1x5mm_FULL_50GS_read_volumes_corrected.dat"
filename="./data/InVivo/3D/20220113_CS/meas_MID00162_FID49557_raFin_3D_tra_1x1x5mm_FULL_noGS_volumes.dat"

filename="./data/InVivo/3D/20220113_CS/meas_MID00163_FID49558_raFin_3D_tra_1x1x5mm_FULL_50GS_read_volumes.dat"
filename="./data/InVivo/3D/phantom.001.v1/phantom.001.v1_corrected.dat"

filename="./data/InVivo/3D/phantom.001.v1/meas_MID00030_FID51057_raFin_3D_phantom_mvt_0_corrected_dens_adj_disp8nob1.dat"

#filename="./3D/SquareSimu3D_sl8_rp2.dat"
#filename="./3D/SquareSimu3D_sl8_rp2_th1.dat"
#filename="./3D/SquareSimu3D_sl8_rp2_GW2.dat"

filename="./3D/SquareSimu3D_sl8_rp8.dat"
filename='./3D/SquareSimu3D_sl8_rp8_tv1.dat'
filename='./data/InVivo/3D/phantom.001.v1/phantom.001.v1_5iter.dat'
filename='./data/InVivo/3D/phantom.001.v1/phantom.001.v1_allspokes8.dat'
#filename='./data/InVivo/3D/phantom.001.v1/phantom.001.v1_allspokes4.dat'
filename='./data/InVivo/3D/20220113_CS/meas_MID00163_FID49558_raFin_3D_tra_1x1x5mm_FULL_50GS_read_allspokes8.dat'
filename='./data/InVivo/3D/patient.001.v1/meas_MID00215_FID60605_raFin_3D_tra_FULl_Tikhonov_0_01_us8ref10_volumes_grappa.dat'
#meas_MID00163_FID49558_raFin_3D_tra_1x1x5mm_FULL_50GS_read_volumes_corrected_final_MRF_map.pkl
filename='./data/InVivo/3D/20220113_CS/meas_MID00163_FID49558_raFin_3D_tra_1x1x5mm_FULL_50GS_read_volumes_corrected_final.dat'
filename='./data/InVivo/3D/patient.002.v2/meas_MID00038_FID01901_raFin_3D_tra_1x1x5mm_FULL_FOV90_Sl160_volumes_modif.dat'
filename='./data/InVivo/3D/patient.003.v1/aggregated_volumes_sl_16_17_18.dat'
#filename='./data/InVivo/3D/patient.003.v1/meas_MID00125_FID02111_raFin_3D_tra_1x1x5mm_FULL_1_volumes_s16_17_18.dat'
filename='./data/InVivo/KB/meas_MID02754_FID765278_JAMBES_raFin_CLI_BILAT_volumes_0.dat'
#filename="./3D/SquareSimu3D_sl8_rp2fullysampled.dat"
#filename="./3D/SquareSimu3D_sl8_rp2_fullysampled.dat"

#filename='./3D/SquareSimu3D_sl8_rp2_nophaseadj.dat'



file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
#
#file_map="./log/maps_it_0_20220211_101810.pkl"

# file_map="./log/maps_it_2_20220208_132700.pkl"

file = open(file_map, "rb")
all_maps = pickle.load(file)

for iter in list(all_maps.keys()):

    map_rebuilt=all_maps[iter][0]
    mask=all_maps[iter][1]

    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
    #map_Python.buildParamMap()


    for key in ["ff","wT1","df","attB1"]:
        file_mha = "/".join(["/".join(str.split(file_map,"/")[:-1]),"_".join(str.split(str.split(file_map,"/")[-1],".")[:-1])]) + "_it{}_{}.mha".format(iter,key)
        io.write(file_mha,map_for_sim[key],tags={"spacing":[5,1,1]})


folder =r"\\192.168.0.1\RMN_FILES"
file ="\meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro_mask_norm_vol.npy"

file_path = folder+file

mask = np.load(file_path)
animate_images(mask)

folder=r"\\192.168.0.1\RMN_FILES\0_Wip\New\1_Methodological_Developments\1_Methodologie_3T\&0_2021_MR_MyoMaps\3_Data\4_3D\Invivo\20211105_TestCS_MRF"
file="\meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro_mask.npy"
file_path = folder+file

mask_base = np.load(file_path)

animate_images(mask_base)





#compare_maps

import pickle
from utils_mrf import *
from mutools import io



folder ="./data/InVivo/3D/patient.001.v1/"

file_maps=[
            "meas_MID00215_FID60605_raFin_3D_tra_FULl_volumes_cf_MRF_map.pkl",
            "meas_MID00215_FID60605_raFin_3D_tra_FULl_volumes_Matrix_MRF_map.pkl"
           #"meas_MID00028_FID07406_raFin_3D_tra_1x1x5mm_FULL_optimCorrelShorten_allspokes8_MRF_map.pkl",
           #"meas_MID00028_FID07406_raFin_3D_tra_1x1x5mm_FULL_optimCorrelShorten_allspokes8_DicoInvivo_MRF_map.pkl"
           ]

dico_maps={}

for map_file in file_maps:
    with open(folder+map_file, "rb") as file:
        all_maps = pickle.load(file)
    dico_maps[map_file]=all_maps


all_maps_1=dico_maps[file_maps[0]]
all_maps_2 = dico_maps[file_maps[1]]
#maskROI=np.array(ROI_data)[all_maps_1[0][1]>0]
#maskROI = buildROImask(all_maps_1[0][0],max_clusters=10)


for key in list(dico_maps.keys())[1:]:
    all_maps_2 = dico_maps[key]
    regression_paramMaps(all_maps_1[0][0],all_maps_2[0][0],all_maps_1[0][1]>0,all_maps_2[0][1]>0,proj_on_mask1=True,fontsize=5,mode="Standard",adj_wT1=True,save=True)

m1=all_maps_1[0][0]
m2=all_maps_2[0][0]


import statsmodels.api as sm
sm.graphics.mean_diff_plot(m1["wT1"][m1["ff"]<0.7], m2["wT1"][m1["ff"]<0.7])

sm.graphics.mean_diff_plot(m1["ff"], m2["ff"])

sm.graphics.mean_diff_plot(m2["ff"], m1["ff"])

#compare_maps ROI




import pickle
from utils_mrf import *
from mutools import io


folder ="./data/InVivo/3D/patient.003.v7/"

#meas_MID00360_FID09597_raFin_3D_tra_1x1x5mm_FULL_1400_DE_FF_reco4_allspokes8_MRF_map
file_maps=[
            #"meas_MID00033_FID09694_raFin_3D_tra_1x1x5mm_FULL_new_volumes_MRF_map.pkl",
            "meas_MID00034_FID09695_raFin_3D_tra_1x1x5mm_FULL_DE_reco3_allspokes8_MRF_map.pkl",
"meas_MID00034_FID09695_raFin_3D_tra_1x1x5mm_FULL_DE_reco3CS_MRF_map.pkl"

           #"meas_MID00028_FID07406_raFin_3D_tra_1x1x5mm_FULL_optimCorrelShorten_allspokes8_MRF_map.pkl",
           #"meas_MID00028_FID07406_raFin_3D_tra_1x1x5mm_FULL_optimCorrelShorten_allspokes8_DicoInvivo_MRF_map.pkl"
           ]

file_maps=[
            #"meas_MID00033_FID09694_raFin_3D_tra_1x1x5mm_FULL_new_volumes_MRF_map.pkl",
            "meas_MID00021_FID13878_raFin_3D_tra_1x1x5mm_FULL_1400_old_full_MRF_map.pkl",
            "meas_MID00024_FID13881_raFin_3D_tra_1x1x5mm_FULL_760_old_reco3_volumes_CF_iterative_MRF_map.pkl",
"meas_MID00023_FID13880_raFin_3D_tra_1x1x5mm_FULL_760_DE_reco3_volumes_CF_iterative_MRF_map.pkl"

           #"meas_MID00028_FID07406_raFin_3D_tra_1x1x5mm_FULL_optimCorrelShorten_allspokes8_MRF_map.pkl",
           #"meas_MID00028_FID07406_raFin_3D_tra_1x1x5mm_FULL_optimCorrelShorten_allspokes8_DicoInvivo_MRF_map.pkl"
           ]


file_maps=[
    "meas_MID00021_FID13878_raFin_3D_tra_1x1x5mm_FULL_1400_old_full_lightDFB1_MRF_map.pkl",
    "meas_MID00021_FID13878_raFin_3D_tra_1x1x5mm_FULL_1400_old_full_lightDFB1_us_MRF_map.pkl"
    ]

file_maps=[
    "meas_MID00060_FID14882_raFin_3D_tra_1x1x5mm_FULL_1400_old_lightDFB1_MRF_map.pkl",
    "meas_MID00061_FID14883_raFin_3D_tra_1x1x5mm_FULL_760_DE_reco3_lightDFB1_MRF_map.pkl",
    "meas_MID00062_FID14884_raFin_3D_tra_1x1x5mm_FULL_760_random_v4_reco3_9_lightDFB1_MRF_map.pkl",
    "meas_MID00063_FID14885_raFin_3D_tra_1x1x5mm_FULL_760_random_v5_reco4_lightDFB1_MRF_map.pkl"
    ]

file_maps=[
    "meas_MID00215_FID60605_raFin_3D_tra_FULl_volumes_CF_iterative_MRF_map.pkl",
    "meas_MID00215_FID60605_raFin_3D_tra_FULl_us4_volumes_CF_iterative_MRF_map.pkl"
    ]

file_maps=[
    "meas_MID00019_FID17057_raFin_3D_tra_1x1x5mm_FULL_new_MRF_map.pkl",
    "meas_MID00022_FID17060_raFin_3D_tra_1x1x5mm_FULL_randomv1_3_95_bis_dummy_echo_MRF_map.pkl",
    "meas_MID00022_FID17060_raFin_3D_tra_1x1x5mm_FULL_randomv1_3_95_bis_dummy_echo_SS_MRF_map.pkl"
    ]

file_maps=[
    "meas_MID00159_FID17675_raFin_3D_tra_1x1x5mm_FULL_1400_reco4_2StepsDico_MRF_map.pkl",
    "meas_MID00159_FID17675_raFin_3D_tra_1x1x5mm_FULL_1400_reco4_SS_MRF_map.pkl"
    ]


file_maps=[
    "meas_MID00021_FID18400_raFin_3D_tra_1x1x5mm_FULL_optim_reco_395_2StepsDico_MRF_map.pkl",
    "meas_MID00025_FID18404_raFin_3D_tra_1x1x5mm_FULL_new_2StepsDico_MRF_map.pkl",
"meas_MID00022_FID18401_raFin_3D_tra_1x1x5mm_FULL_optim_reco_395_US2_2StepsDico_MRF_map.pkl",
"meas_MID00022_FID18401_raFin_3D_tra_1x1x5mm_FULL_optim_reco_395_US2_MRF_map.pkl",
"meas_MID00026_FID18405_raFin_3D_tra_1x1x5mm_FULL_new_US2_2StepsDico_MRF_map.pkl",
"meas_MID00026_FID18405_raFin_3D_tra_1x1x5mm_FULL_new_US2_MRF_map.pkl"

    ]

file_ROI=folder+"roi.mha"

dico_maps={}

for map_file in file_maps:
    with open(folder+map_file, "rb") as file:
        all_maps = pickle.load(file)
    dico_maps[map_file]=all_maps


ROI_data=io.read(file_ROI)

all_maps_1=dico_maps[file_maps[0]]
maskROI=np.array(ROI_data)[all_maps_1[0][1]>0]
#maskROI = buildROImask(all_maps_1[0][0],max_clusters=10)



#regression_paramMaps_ROI(all_maps_1[0][0],all_maps_2[0][0],all_maps_1[0][1]>0,all_maps_2[0][1]>0,save=True,fontsize=5,mode="Standard",adj_wT1=False,maskROI=maskROI,title="No Reco vs Full Reco")
#regression_paramMaps_ROI(all_maps_1[0][0],all_maps_2_optim[0][0],all_maps_1[0][1]>0,all_maps_2_optim[0][1]>0,save=True,fontsize=5,mode="Standard",adj_wT1=False,maskROI=maskROI,title="No Reco Optim vs Full Reco"e)

dico_values={}
for k in dico_maps.keys():
    all_maps_2=dico_maps[k]
    for iter in all_maps_2.keys():

        dico_values[k+"_it{}".format(iter)]=get_ROI_values(all_maps_1[0][0],all_maps_2[iter][0],all_maps_1[0][1]>0,all_maps_2[iter][1]>0,maskROI=maskROI,return_std=True)


#volume_ROI=makevol(maskROI,all_maps_1[0][1]>0)

plt.close("all")


import statsmodels.api as sm


k = "wT1"
for key in list(dico_values.keys())[1:]:
    values=dico_values[key][k].sort_values(by=["Obs Mean"])
    sm.graphics.mean_diff_plot(np.array(values[["Obs Mean"]]).flatten(), np.array(values[["Pred Mean"]]).flatten());
    #plt.title("{} : {} vs Reference Sequence".format(k,key))

k = "ff"
for key in list(dico_values.keys())[1:]:
    values=dico_values[key][k].sort_values(by=["Obs Mean"])
    sm.graphics.mean_diff_plot(np.array(values[["Obs Mean"]]).flatten(), np.array(values[["Pred Mean"]]).flatten());
    #plt.title("{} : {} vs Reference Sequence".format(k,key))


labels=["Optim","Current","Optim US 2","Optim US 2 Traditional Matching","Current US 2","Current US 2 Traditional Matching"]

plt.figure()
k="df"
plt.title(k +" roi mean")
for i,key in enumerate(dico_values.keys()):
    values=dico_values[key][k].sort_values(by=["Obs Mean"])
    if key=="meas_MID00215_FID60605_raFin_3D_tra_FULl_volumes_CF_iterative_MRF_map.pkl_it0":
        plt.plot(values[["Pred Mean"]],'x',label=labels[i],color="k")#,marker="x")

    else:
        plt.plot(values[["Pred Mean"]],'o',label=labels[i])#,marker="x")
plt.legend()



plt.figure()
plt.title(k + " roi std")
for i,key in enumerate(dico_values.keys()):
    values=dico_values[key][k].sort_values(by=["Obs Mean"])
    if key=="meas_MID00215_FID60605_raFin_3D_tra_FULl_volumes_CF_iterative_MRF_map.pkl_it0":
        plt.plot(values[["Obs Mean"]], values[["Pred Std"]], label=labels[i], linestyle="dashed",color="k")
    else:
        plt.plot(values[["Obs Mean"]],values[["Pred Std"]],label=labels[i],marker="x")
plt.legend()


labels=["MRF T1-FF 1400 spokes","Optimized MRF T1-FF 760 spokes"]
plt.figure()
k="ff"
plt.title(k +" roi mean")
for i,key in enumerate(dico_values.keys()):
    values=dico_values[key][k].sort_values(by=["Obs Mean"])
    plt.plot(values[["Obs Mean"]],values[["Pred Mean"]],"o",label=labels[i])
plt.legend()

plt.figure()
plt.title(k + " roi std")
for i,key in enumerate(dico_values.keys()):
    values=dico_values[key][k].sort_values(by=["Obs Mean"])
    plt.plot(values[["Obs Mean"]],values[["Pred Std"]],label=labels[i],marker="x")
plt.legend()

plt.figure()
k="wT1"
plt.title(k)
plt.plot(values_ROI[k][:,0],label='mean per ROI full reco')
plt.plot(values_ROI[k][:,2],label='mean per ROI no reco')
plt.plot(values_optim_ROI[k][:,2],label='mean per ROI no reco optimized')
plt.legend()

plt.figure()
k="ff"
plt.title(k)
plt.plot(values_ROI[k][:,1],label='std per ROI full reco')
plt.plot(values_ROI[k][:,3],label='std per ROI no reco')
plt.plot(values_optim_ROI[k][:,3],label='std per ROI no reco optimized')
plt.legend()

plt.figure()
k="ff"
plt.title(k)
plt.plot(values_ROI[k][:,0],label='mean per ROI full reco')
plt.plot(values_ROI[k][:,2],label='mean per ROI no reco')
plt.plot(values_optim_ROI[k][:,2],label='mean per ROI no reco optimized')
plt.legend()

plt.close("all")



#compare_maps

import pickle
from utils_mrf import *
from mutools import io



folder ="./data/InVivo/3D/patient.001.v1/"

#meas_MID00360_FID09597_raFin_3D_tra_1x1x5mm_FULL_1400_DE_FF_reco4_allspokes8_MRF_map
file_maps=[
            "meas_MID00215_FID60605_raFin_3D_tra_FULl_volumes_cf_MRF_map.pkl",
            "meas_MID00215_FID60605_raFin_3D_tra_FULl_volumes_Matrix_MRF_map.pkl"
           ]



list_maps=[]

for map_file in file_maps:
    with open(folder+map_file, "rb") as file:
        all_maps = pickle.load(file)
    list_maps.append(all_maps)


regression_paramMaps(list_maps[0][0][0],list_maps[1][0][0],list_maps[0][0][1]>0,list_maps[1][0][1]>0,proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,save=True)

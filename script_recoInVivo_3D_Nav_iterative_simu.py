

### Movements simulation

#import matplotlib
#matplotlib.u<se("TkAgg")
import numpy as np
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,GaussianWeighting
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
base_folder = "./3D"

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

nb_filled_slices = 16
nb_empty_slices=2
repeat_slice=16
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

name = "SquareSimu3DMT"


use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

folder = "/".join(str.split(filename,"/")[:-1])

suffix=""

filename_paramMap=filename+"_paramMap_sl{}_rp{}.pkl".format(nb_slices,repeat_slice)
filename_paramMask=filename+"_paramMask_sl{}_rp{}.npy".format(nb_slices,repeat_slice)
filename_groundtruth = filename+"_groundtruth_volumes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")

filename_kdata = filename+"_kdata_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_kdata_gt = filename+"_kdata_mvt_gt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_nav = filename+"_nav_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_volume = filename+"_volumes_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_volume_corrected = filename+"_volumes_corrected_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_mask= filename+"_mask_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask_corrected= filename+"_mask_corrected_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

file_map = filename + "_mvt_sl{}_rp{}_us{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,suffix)



#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 128



incoherent=True
mode="old"

image_size = (nb_slices, int(npoint/2), int(npoint/2))
nb_segments=nb_allspokes
nspoke=int(nb_segments/ntimesteps)
size = image_size[1:]

with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)

if "Square" in name:
    region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
    mask_reduction_factor=1/4


    m_ = RandomMap3D(name,dict_config,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,undersampling_factor=undersampling_factor,repeat_slice=repeat_slice,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)

# elif name=="KneePhantom":
#     num =1
#     file_matlab_paramMap = "./data/{}/Phantom{}/paramMap.mat".format(name,num)
#
#     m = MapFromFile(name,image_size=image_size,file=file_matlab_paramMap,rounding=True,gen_mode="other")

else:
    raise ValueError("Unknown Name")



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

# i=0
# image=m.images_series[i]


# file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
#                              "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_ideal_ts{}.mha".format(
#                                                                                                                    i)
# io.write(file_mha, np.abs(image), tags={"spacing": [5, 1, 1]})


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,is_random=False)

nb_channels=1


direction=np.array([0.0,8.0,0.0])
move = TranslationBreathing(direction,T=4000,frac_exp=0.7)

m_.add_movements([move])


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):

    #images = copy(m.images_series)
    data=m_.generate_kdata(radial_traj,useGPU=use_GPU)
    data=np.array(data)
    data=np.expand_dims(data,axis=0)
    np.save(filename_kdata, data)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    data = np.load(filename_kdata)

nb_segments=radial_traj.get_traj().shape[0]

b1_full = np.ones(image_size)
b1_all_slices=np.expand_dims(b1_full,axis=0)



print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    kdata_all_channels_all_slices=np.load(filename_kdata)
    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_iterative=True)
    np.save(filename_volume,volumes_all)
    # sl=20
    ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
    del volumes_all


if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    np.save(filename_mask,m_.mask)



print("Processing Nav Data...")

nb_gating_spokes=50
ts_between_spokes=int(nb_allspokes/50)
timesteps = list(np.arange(1400)[::ts_between_spokes])

if str.split(filename_nav,"/")[-1] not in os.listdir(folder):

    nav_z=Navigator3D(direction=[1,0,0.0],applied_timesteps=timesteps,npoint=npoint)
    kdata_nav = m_.generate_kdata(nav_z,useGPU=use_GPU)

    kdata_nav=np.array(kdata_nav)

    data_for_nav = np.expand_dims(kdata_nav,axis=0)
    data_for_nav =  np.moveaxis(data_for_nav,1,2)
    data_for_nav = data_for_nav.astype("complex64")
    np.save(filename_nav,data_for_nav)

else:
    data_for_nav=np.load(filename_nav)

npoint_nav=data_for_nav.shape[-1]
nb_slices_nav=data_for_nav.shape[1]
nb_gating_spokes=data_for_nav.shape[-2]

all_timesteps = np.arange(nb_allspokes)
nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices_nav,
                       applied_timesteps=list(nav_timesteps))

nav_image_size = (int(npoint_nav / 2),)

print("Calculating Sensitivity Maps for Nav Images...")
b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
b1_nav_mean = np.mean(b1_nav, axis=(1, 2))

print("Rebuilding Nav Images...")
images_nav_mean = np.abs(simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, b1_nav_mean))


print("Estimating Movement...")
shifts = list(range(-10, 10))
bottom = 10
top = 54
displacements = calculate_displacement(images_nav_mean, bottom, top, shifts)

displacement_for_binning = displacements
bin_width = 2
max_bin = np.max(displacement_for_binning)
min_bin = np.min(displacement_for_binning)
plt.figure();plt.plot(displacements)

maxi = 0
for j in range(bin_width):
    min_bin = np.min(displacement_for_binning) + j
    bins = np.arange(min_bin, max_bin + bin_width, bin_width)
    #print(bins)
    categories = np.digitize(displacement_for_binning, bins)
    df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
    df_groups = df_cat.groupby("cat").count()
    curr_max = df_groups.displacement.max()
    if curr_max > maxi:
        maxi = curr_max
        df_groups_max=copy(df_groups)
        min_bin_max=min_bin
        max_bin_max=max_bin
        idx_cat = df_groups.displacement.idxmax()
        retained_nav_spokes = (categories == idx_cat)

retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
spoke_groups = np.argmin(np.abs(np.arange(0, nb_segments * nb_slices, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_slices,nb_segments / nb_gating_spokes).reshape(1,-1)),axis=-1)

if not (nb_segments == nb_gating_spokes):
    spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
        nb_segments / nb_gating_spokes / 2) + 1:] - 1
    spoke_groups = spoke_groups.flatten()

included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
displacements_real=m_.list_movements[0].paramDict["transformation"](m_.t.reshape(-1,1))[:,1]
#plt.figure();plt.plot(displacements_real[::(int(nb_allspokes/nb_gating_spokes))]);plt.plot(displacements+5,marker='x')
#plt.figure();plt.plot(displacements_real);plt.plot(included_spokes)
included_spokes=displacements_real>6

included_spokes[::int(nb_segments/nb_gating_spokes)]=False

weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, ntimesteps)






print("Rebuilding Images With Corrected volumes...")

if str.split(filename_volume_corrected,"/")[-1] not in os.listdir(folder):
    kdata_all_channels_all_slices=np.load(filename_kdata)
    volumes_corrected = simulate_radial_undersampled_images_multi_new(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,ntimesteps=ntimesteps,density_adj=True,useGPU=False,light_memory_usage=True,retained_timesteps=retained_timesteps,weights=weights)
    animate_images(volumes_corrected[:,int(nb_slices/2),:,:])
    np.save(filename_volume_corrected,volumes_corrected)
else:
    volumes_corrected=np.load(filename_volume_corrected)





if str.split(filename_mask_corrected,"/")[-1] not in os.listdir(folder):
    kdata_all_channels_all_slices = np.load(filename_kdata)
    volumes_full_corrected = simulate_radial_undersampled_images_multi_new(kdata_all_channels_all_slices, radial_traj,
                                                                           image_size, b1=b1_full,
                                                                           ntimesteps=ntimesteps,
                                                                           density_adj=True, useGPU=False,
                                                                           light_memory_usage=True,
                                                                           retained_timesteps=retained_timesteps,
                                                                           weights=weights, ntimesteps_final=1)[0]
    mask = False
    unique = np.histogram(np.abs(volumes_full_corrected), 100)[1]
    mask = mask | (np.abs(volumes_full_corrected) > unique[int(len(unique) * 0.07)])
    # mask = ndimage.binary_closing(mask, iterations=3)
    animate_images(mask)
    np.save(filename_mask_corrected,mask)



# animate_images(volumes_full_corrected)
#
# kdata_all_channels_all_slices=np.load(filename_kdata)
# volumes_full= simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,normalize_iterative=True)[0]
# animate_multiple_images(volumes_full,volumes_full_corrected)

# kdata_all_channels_all_slices=np.load(filename_kdata)
# weights_one=np.ones(weights.shape)
# volumes_full_not_corrected = simulate_radial_undersampled_images_multi_new(kdata_all_channels_all_slices, radial_traj,
#                                                                   image_size, b1=b1_full, ntimesteps=ntimesteps,
#                                                                   density_adj=False, useGPU=False,
#                                                                   light_memory_usage=True,
#                                                                   retained_timesteps=retained_timesteps,
#                                                                   weights=weights_one,ntimesteps_final=1)[0]
#
# animate_multiple_images(volumes_full_not_corrected,volumes_full_corrected)



#del kdata_all_channels_all_slices
#del b1_all_slices



########################## Dict mapping ########################################


seq = None

load_map=False
save_map=True

dictfile = "mrf175_SimReco2_light.dict"

mask = np.load(filename_mask_corrected)
volumes_all = np.load(filename_volume_corrected)
animate_images(volumes_all[:,int(nb_slices/2),:,:])

if not(load_map):
    niter = 0
    niter = 100
    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=10, pca=True,
                                 threshold_pca=20, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                 gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                 b1=b1_all_slices, mu="Adaptative",weights=weights,mu_TV=1,weights_TV=[1.,0.,0.])
    all_maps = optimizer.search_patterns_test_multi_mvt(dictfile, volumes_all, retained_timesteps=retained_timesteps)

if(save_map):
        import pickle

        #file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format(suffix)
        #file_map = filename.split(".dat")[0] + "_corrected_dens_adj{}_MRF_map.pkl".format(suffix)
        file_map = filename.split(".dat")[0] + "{}_new_iter_MRF_map.pkl".format(suffix)
        file = open(file_map, "wb")
        # dump information to that file
        pickle.dump(all_maps, file)
        # close the file
        file.close()

else:
    import pickle
    file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
    file = open(file_map, "rb")
    all_maps = pickle.load(file)

#maskROI=buildROImask_unique(m_.paramMap)
#regression_paramMaps_ROI(m_.paramMap,all_maps[0][0],m_.mask>0,all_maps[0][1]>0,maskROI,adj_wT1=True,title="regROI_"+str.split(str.split(filename_volume_corrected,"/")[-1],".npy")[0],save=True)


plt.close("all")
maskROI = buildROImask_unique(m_.paramMap)
for it in range(niter+1):

    regression_paramMaps_ROI(m_.paramMap, all_maps[it][0], m_.mask > 0, all_maps[it][1] > 0, maskROI, adj_wT1=True,
                             title="it{}_regROI_".format(it) + str.split(str.split(filename_volume, "/")[-1], ".npy")[0], save=True)



from mutools import io
curr_file=file_map
# file = open(curr_file, "rb")
# all_maps = pickle.load(file)
#file.close()
for iter in list(all_maps.keys()):

    map_rebuilt=all_maps[iter][0]
    mask=all_maps[iter][1]

    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
    #map_Python.buildParamMap()


    for key in ["ff","wT1","df","attB1"]:
        file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_it{}_{}.mha".format(iter,key)
        io.write(file_mha,map_for_sim[key],tags={"spacing":[5,1,1]})


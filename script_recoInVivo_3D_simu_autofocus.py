######SIMU########
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
from utils_simu import *

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"

#dictfile = "mrf175_SimReco2_light.dict"
dictjson="mrf_dictconf_SimReco2_light.json"
dictfile="mrf_dictconf_SimReco2_adjusted_1_87_reco4_w8_simmean.dict"
dictfile_light='./mrf_dictconf_SimReco2_light_matching_adjusted_1_87_reco4_w8_simmean.dict'
suffix="_fullReco"
with open("./mrf_sequence_adjusted_1_87.json") as f:
   sequence_config = json.load(f)
Treco=4000

nrep=2
rep=nrep-1
TR_total = np.sum(sequence_config["TR"])

#Treco = TR_total-np.sum(sequence_config["TR"])

##other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=nrep
sequence_config["rep"]=rep

seq=T1MRFSS(**sequence_config)

nb_filled_slices = 16
nb_empty_slices=2
repeat_slice=16
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

name = "SquareSimu3DMT"
name = "Knee3DMT"

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
filename_volume_corrected_autofocus = filename+"_volumes_corrected_autofocus_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")


filename_mask= filename+"_mask_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask_corrected= filename+"_mask_corrected_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask_corrected_autofocus= filename+"_mask_corrected_autofocus_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_displacements_real= filename+"_displacements_real_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")


file_map = filename + "_mvt_sl{}_rp{}_us{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,suffix)



#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 256

if "Knee" in name:
    npoint_backup=npoint
    npoint=512



incoherent=False
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

elif "Knee" in name:
    num =1
    file_matlab_paramMap = "./data/KneePhantom/Phantom{}/paramMap_Control.mat".format(num)
    m_ = MapFromFile3D(name,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,image_size=image_size,file=file_matlab_paramMap,rounding=True,gen_mode="other",undersampling_factor=undersampling_factor,resting_time=4000)

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


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    m_.build_ref_images(seq)
else:
    m_.build_timeline(list(seq.TR))

if "Knee" in name:
    m_.change_resolution(int(npoint/npoint_backup))
    npoint=npoint_backup

image_size = (nb_slices, int(npoint/2), int(npoint/2))
size = image_size[1:]

#if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
#    np.save(filename_groundtruth,m_.images_series[::nspoke])

# i=0
# image=m.images_series[i]


# file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
#                              "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_ideal_ts{}.mha".format(
#                                                                                                                    i)
# io.write(file_mha, np.abs(image), tags={"spacing": [5, 1, 1]})


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,is_random=False)

nb_channels=1


direction=np.array([0.0,10,0.0])
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
shifts = list(range(-15, 15))
bottom = -shifts[0]
top = int(npoint_nav/2)-shifts[-1]
displacements = calculate_displacement(images_nav_mean, bottom, top, shifts)

displacement_for_binning = displacements_real[::28]
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
else:
    mask=np.load(filename_mask_corrected)


print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    #del kdata_all_channels_all_slices
    kdata_all_channels_all_slices = np.load(filename_kdata)

    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,m_.image_size,b1=b1_all_slices,density_adj=True,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_iterative=True)
    np.save(filename_volume,volumes_all)
    # sl=10
    # ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
else:
    volumes_all=np.load(filename_volume)

##volumes for slice taking into account coil sensi
print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    mask=m_.mask
    np.save(filename_mask,mask)
else:
    mask=np.load(filename_mask)


displacements_extrapolated = np.array([displacements[j] for j in spoke_groups])




displacements_real=(m_.list_movements[0].paramDict["transformation"])(m_.t.reshape(-1,1))[:,1]
#plt.figure()
#plt.plot(displacements_real)
plt.figure()
plt.plot(displacements)
plt.plot(displacements_real[::28])


np.save(filename_displacements_real,displacements_real[::28])

x = np.arange(-1, 1.1, 0.5)
y = np.arange(-1, 1.1, 0.5)

entropy_all = []
ent_min = np.inf

displacements_entropy=displacements_real
all_volumes_modif=[]
for dx in tqdm(x):
    entropy_x = []
    for dy in y:
        alpha = np.array([dx, dy, 0])
        dr = np.expand_dims(alpha, axis=(0, 1)) * np.expand_dims(displacements_entropy.reshape(nb_slices, 1400).T, axis=(2))
        modif = np.exp(
            1j * np.sum((radial_traj.get_traj().reshape(1400, -1, npoint, 3) * np.expand_dims(dr, axis=2)), axis=-1))
        modif=modif.reshape(data.shape)
        data_modif = data * modif
        volume_full_modif = \
        simulate_radial_undersampled_images_multi(data_modif, radial_traj,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_iterative=True)[0]
        ent = calc_grad_entropy(volume_full_modif)
        entropy_x.append(ent)
        all_volumes_modif.append(volume_full_modif)
        if ent < ent_min:
            modif_final = modif
            alpha_min = alpha
            ent_min = ent

    entropy_all.append(entropy_x)

sl=int(nb_slices/2)
ind=0
plt.figure()
plot_image_grid(list(np.abs(np.array(all_volumes_modif)[:,sl])),nb_row_col=(5,5))

X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(15,15))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, np.array(entropy_all), rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)



#volumes for slice taking into account coil sensi
print("Building Volumes Corrected....")
if str.split(filename_volume_corrected_autofocus,"/")[-1] not in os.listdir(folder):
    data_modif = kdata_all_channels_all_slices*modif_final
    volumes_all_modif=simulate_radial_undersampled_images_multi(data_modif,radial_traj,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_iterative=True)
    np.save(filename_volume_corrected_autofocus,volumes_all_modif)
    # sl=20
    ani = animate_images(volumes_all_modif[:,int(nb_slices/2),:,:])
    #del volumes_all_modif

print("Building Mask Corrected....")
if str.split(filename_mask_corrected_autofocus,"/")[-1] not in os.listdir(folder):
    selected_spokes = np.r_[10:400]
    selected_spokes=None
    data_modif = kdata_all_channels_all_slices * modif_final
    mask_modif=build_mask_single_image_multichannel(data_modif,radial_traj,image_size,b1=b1_all_slices,density_adj=True,threshold_factor=1/25, normalize_kdata=False,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask_corrected_autofocus,mask_modif)
    animate_images(mask_modif)
    del mask_modif


optimizer = SimpleDictSearch(mask=mask_modif, niter=0, seq=None, trajectory=radial_traj, split=100, pca=True,
                                     threshold_pca=15, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                     b1=b1_all_slices, threshold_ff=0.9, dictfile_light=dictfile_light,
                                     return_matched_signals=True)



all_maps_usual, matched_signals = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all_modif,
                                                                                        retained_timesteps=retained_timesteps)


curr_file=file_map
return_cost=True
dx=1
dy=1
dz=5
all_maps=all_maps_usual
suffix="_autofocus_dispreal"
for iter in list(all_maps.keys()):

    map_rebuilt=all_maps[iter][0]
    mask=all_maps[iter][1]

    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
    #map_Python.buildParamMap()


    for key in ["ff","wT1","df","attB1"]:
        file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,iter,key)
        io.write(file_mha,map_for_sim[key],tags={"spacing":[dz,dx,dy]})

    if return_cost:
        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,
            iter, "correlation")
        io.write(file_mha, makevol(all_maps[iter][2],mask>0), tags={"spacing": [dz, dx, dy]})

        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,
            iter, "phase")
        io.write(file_mha, makevol(all_maps[iter][3],mask>0), tags={"spacing": [dz, dx, dy]})









######SIMU########
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
from utils_simu import *

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"

#dictfile = "mrf175_SimReco2_light.dict"
dictjson="mrf_dictconf_SimReco2_light.json"
dictfile="mrf_dictconf_SimReco2_adjusted_1_87_reco4_w8_simmean.dict"
dictfile_light='./mrf_dictconf_SimReco2_light_matching_adjusted_1_87_reco4_w8_simmean.dict'
suffix="_fullReco"
with open("./mrf_sequence_adjusted_1_87.json") as f:
   sequence_config = json.load(f)
Treco=4000

nrep=2
rep=nrep-1
TR_total = np.sum(sequence_config["TR"])

#Treco = TR_total-np.sum(sequence_config["TR"])

##other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=nrep
sequence_config["rep"]=rep

seq=T1MRFSS(**sequence_config)

nb_filled_slices = 16
nb_empty_slices=2
repeat_slice=16
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

name = "SquareSimu3DMT"
name = "Knee3DMT"

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
filename_volume_corrected_autofocus = filename+"_volumes_corrected_autofocus_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")


filename_mask= filename+"_mask_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask_corrected= filename+"_mask_corrected_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask_corrected_autofocus= filename+"_mask_corrected_autofocus_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_displacements_real= filename+"_displacements_real_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")


file_map = filename + "_mvt_sl{}_rp{}_us{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,suffix)



#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 256

if "Knee" in name:
    npoint_backup=npoint
    npoint=512



incoherent=False
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

elif "Knee" in name:
    num =1
    file_matlab_paramMap = "./data/KneePhantom/Phantom{}/paramMap_Control.mat".format(num)
    m_ = MapFromFile3D(name,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,image_size=image_size,file=file_matlab_paramMap,rounding=True,gen_mode="other",undersampling_factor=undersampling_factor,resting_time=4000)

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


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    m_.build_ref_images(seq)
else:
    m_.build_timeline(list(seq.TR))

if "Knee" in name:
    m_.change_resolution(int(npoint/npoint_backup))
    npoint=npoint_backup

image_size = (nb_slices, int(npoint/2), int(npoint/2))
size = image_size[1:]

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,is_random=False)

nb_channels=1


direction=np.array([0.0,10,0.0])
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

nb_gating_spokes=data_for_nav.shape[-2]
displacements_real=(m_.list_movements[0].paramDict["transformation"])(m_.t.reshape(-1,1))[:,1]

displacement_for_binning = displacements_real[::28]
bin_width = 2
max_bin = np.max(displacement_for_binning)
min_bin = np.min(displacement_for_binning)
plt.figure();plt.plot(displacement_for_binning)

min_bin = np.min(displacement_for_binning)
bins = np.arange(min_bin, max_bin + bin_width, bin_width)
    #print(bins)
categories = np.digitize(displacement_for_binning, bins)
df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
df_groups = df_cat.groupby("cat").count()


group_1=(categories==1)
#group_2=(categories==2)|(categories==3)|(categories==4)
group_3=(categories==5)

groups=[group_1,group_3]

nb_part=nb_slices
dico_traj_retained = {}
for j, g in tqdm(enumerate(groups)):
    print("######################  BUILDING FULL VOLUME AND MASK FOR GROUP {} ##########################".format(j))
    retained_nav_spokes_index = np.argwhere(g).flatten()
    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
        axis=-1)

    spoke_groups=spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:]=spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:]-1 #adjustment for change of partition
    spoke_groups=spoke_groups.flatten()

    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
    #included_spokes[::int(nb_segments / nb_gating_spokes)] = False
    #print(np.sum(included_spokes))

    weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, ntimesteps)
    print(len(retained_timesteps))

    #print(traj_retained_final_volume.shape[1]/800)

    dico_traj_retained[j] = weights

L0=10
filename_phi=str.split(dictfile,".dict") [0]+"_phi_L0_{}.npy".format(L0)


if filename_phi not in os.listdir():
    mrfdict = dictsearch.Dictionary()
    keys,values=read_mrf_dict(dictfile,np.arange(0.,1.01,0.1))

    import dask.array as da
    u,s,vh = da.linalg.svd(da.asarray(values))

    vh=np.array(vh)
    s=np.array(s)

    L0=10
    phi=vh[:L0]
    np.save(filename_phi,phi)
else:
    phi=np.load(filename_phi)

all_volumes_singular=[]

for gr in tqdm(dico_traj_retained.keys()):
    data=np.load(filename_kdata)
    weights = np.expand_dims((dico_traj_retained[gr]>0)*1,axis=(0,-1))
    traj=radial_traj.get_traj().reshape(ntimesteps,-1,3)
    kdata_all_channels_all_slices=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
    kdata_all_channels_all_slices*=weights
    kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,-1)
    kdata_singular=np.zeros((nb_channels,)+traj.shape[:-1]+(L0,),dtype=kdata_all_channels_all_slices.dtype)
    for ts in tqdm(range(ntimesteps)):
        kdata_singular[:,ts,:,:]=kdata_all_channels_all_slices[:,ts,:,None]@(phi.conj().T[ts][None,:])

    kdata_singular=np.moveaxis(kdata_singular,-1,1)

    kdata_singular=kdata_singular.reshape(nb_channels,L0,-1)
    volumes_singular = simulate_radial_undersampled_singular_images_multi(kdata_singular, radial_traj, image_size,
                                                                          density_adj=True, b1=b1_all_slices, light_memory_usage=True)
    all_volumes_singular.append(volumes_singular)

all_volumes_singular=np.array(all_volumes_singular)

sl=int(nb_slices/2)

plot_image_grid(np.abs(all_volumes_singular[:,:,sl]).reshape((-1,)+image_size[1:]),nb_row_col=(len(groups),L0))




import SimpleITK as sitk

def command_iteration(method):
    if method.GetOptimizerIteration() == 0:
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():7.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

comp=0
fixed = sitk.GetImageFromArray(np.abs(all_volumes_singular[1,comp,sl]))
#fixed.SetOrigin((0, 0, 0))
#fixed.SetSpacing(spacing)

moving = sitk.GetImageFromArray(np.abs(all_volumes_singular[0,comp,sl]))
#moving.SetOrigin((0, 0, 0))
#moving.SetSpacing(spacing)



R = sitk.ImageRegistrationMethod()

R.SetMetricAsCorrelation()

R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
R.SetOptimizerScalesFromIndexShift()



dimension = 2
offset = [2]*dimension # use a Python trick to create the offset list based on the dimension
transform = sitk.TranslationTransform(dimension, offset)

tx = transform

R.SetInitialTransform(tx)

R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

outTx = R.Execute(fixed, moving)


print("-------")
print(outTx)
print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
print(f" Iteration: {R.GetOptimizerIteration()}")
print(f" Metric value: {R.GetMetricValue()}")

#sitk.WriteTransform(outTx, args[3])

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)

resampler.SetTransform(outTx)

out = resampler.Execute(moving)

simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)


animate_images([sitk.GetArrayFromImage(fixed),sitk.GetArrayFromImage(moving)])
animate_images([sitk.GetArrayFromImage(fixed),sitk.GetArrayFromImage(out)])

resampler_inverse = sitk.ResampleImageFilter()
resampler_inverse.SetReferenceImage(fixed)
resampler_inverse.SetInterpolator(sitk.sitkLinear)

inverseTx=outTx.GetInverse()
resampler_inverse.SetTransform(inverseTx)
animate_images([sitk.GetArrayFromImage(resampler.Execute(fixed)),sitk.GetArrayFromImage(moving)])

Mn=inverseTx
Mn_H = outTx






dico_transform={}
dico_transform[0]=(Mn,Mn_H)
dico_transform[1]=(sitk.sitkIdentity,sitk.sitkIdentity)


def applyTransform(array,transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)

    resampler.SetTransform(transform)

    return sitk.GetArrayFromImage(resampler.Execute(sitk.GetImageFromArray(array)))





if str.split(filename_volume_singular,"/")[-1] not in os.listdir(folder):
    traj=radial_traj.get_traj().reshape(ntimesteps,-1,3)
    kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,-1)

    kdata_singular=np.zeros((nb_channels,)+traj.shape[:-1]+(L0,),dtype=kdata_all_channels_all_slices.dtype)
    for ts in tqdm(range(ntimesteps)):
        kdata_singular[:,ts,:,:]=kdata_all_channels_all_slices[:,ts,:,None]@(phi.conj().T[ts][None,:])

    kdata_singular=np.moveaxis(kdata_singular,-1,1)

    kdata_singular=kdata_singular.reshape(nb_channels,L0,-1)









optimizer = SimpleDictSearch(mask=mask_modif, niter=0, seq=None, trajectory=radial_traj, split=100, pca=True,
                                     threshold_pca=15, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                     b1=b1_all_slices, threshold_ff=0.9, dictfile_light=dictfile_light,
                                     return_matched_signals=True)



all_maps_usual, matched_signals = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all_modif,
                                                                                        retained_timesteps=retained_timesteps)


curr_file=file_map
return_cost=True
dx=1
dy=1
dz=5
all_maps=all_maps_usual
suffix="_autofocus_dispreal"
for iter in list(all_maps.keys()):

    map_rebuilt=all_maps[iter][0]
    mask=all_maps[iter][1]

    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
    #map_Python.buildParamMap()


    for key in ["ff","wT1","df","attB1"]:
        file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,iter,key)
        io.write(file_mha,map_for_sim[key],tags={"spacing":[dz,dx,dy]})

    if return_cost:
        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,
            iter, "correlation")
        io.write(file_mha, makevol(all_maps[iter][2],mask>0), tags={"spacing": [dz, dx, dy]})

        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,
            iter, "phase")
        io.write(file_mha, makevol(all_maps[iter][3],mask>0), tags={"spacing": [dz, dx, dy]})

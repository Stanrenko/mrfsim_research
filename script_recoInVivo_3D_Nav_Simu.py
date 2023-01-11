
#import matplotlib
#matplotlib.u<se("TkAgg")
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

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"
nav_folder = "./data/InVivo/3D"

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

nb_filled_slices = 8
nb_empty_slices=0
repeat_slice=8
nb_slices = nb_filled_slices+2*nb_empty_slices
name = "SquareSimu3D"

use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

filename_nav_save=str.split(nav_folder+"/phantom.001.v1/phantom.001.v1.dat",".dat") [0]+"_nav.npy"
nb_gating_spokes=50
#filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"

folder = "/".join(str.split(filename,"/")[:-1])

suffix="_tv1"

filename_paramMap=filename+"_paramMap_sl{}_rp{}.pkl".format(nb_slices,repeat_slice)
filename_volume = filename+"_volumes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_groundtruth = filename+"_groundtruth_volumes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")

filename_volume_corrected = filename+"_volumes_corrected_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_kdata = filename+"_kdata_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_mask= filename+"_mask_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
file_map = filename + "_sl{}_rp{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,suffix)


#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



data_for_nav=np.load(filename_nav_save)

ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 512

undersampling_factor=1

incoherent=True
mode="old"



with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
mask_reduction_factor=1/4



image_size = (nb_slices, int(npoint/2), int(npoint/2))
size=image_size[1:]
nb_segments=nb_allspokes
nspoke=int(nb_segments/ntimesteps)


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
m = RandomMap3D(name,dict_config,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,undersampling_factor=undersampling_factor,repeat_slice=repeat_slice,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)



if str.split(filename_paramMap,"/")[-1] not in os.listdir(folder):
    m.buildParamMap()
    with open(filename_paramMap, "wb" ) as file:
        pickle.dump(m.paramMap, file)

    map_rebuilt = m.paramMap
    mask = m.mask

    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    for key in ["ff", "wT1", "df", "attB1"]:
        file_mha = "/".join(["/".join(str.split(filename_paramMap, "/")[:-1]),
                             "_".join(str.split(str.split(filename_paramMap, "/")[-1], ".")[:-1])]) + "_{}.mha".format(
                                                                                                                   key)
        io.write(file_mha, map_for_sim[key], tags={"spacing": [5, 1, 1]})

else:
    with open(filename_paramMap, "rb") as file:
        m.paramMap=pickle.load(file)


# file_map_matlab = filename+"_paramMap_sl{}_rp{}.mat".format(nb_slices,repeat_slice)
# file_mask_matlab= filename+"_mask_sl{}_rp{}.mat".format(nb_slices,repeat_slice)
#
# savemat(file_map_matlab,m.paramMap)
# savemat(file_mask_matlab,{"Mask":m.mask})


m.build_ref_images(seq)

if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
    np.save(filename_groundtruth,m.images_series[::nspoke])



if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    # Density adjustment all slices
    data=m.generate_kdata(radial_traj)
    data=np.expand_dims(data,axis=0)

    density = np.abs(np.linspace(-1, 1, npoint))
    density = np.expand_dims(density, tuple(range(data.ndim - 1)))

    #kdata_all_channels_all_slices = data.reshape(-1, npoint)
    #del data
    print("Performing Density Adjustment....")

    data = data.reshape(1, nb_allspokes, -1, npoint)
    data *= density

    np.save(filename_kdata, data)
    del data

    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)


b1_full = np.ones(image_size)
b1_full=np.expand_dims(b1_full,axis=0)


print("Processing Nav Data...")
data_for_nav=np.load(filename_nav_save)

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
shifts = list(range(-20, 20))
bottom = 50
top = 150
displacements = calculate_displacement(images_nav_mean, bottom, top, shifts)

displacement_for_binning = displacements
bin_width = 8
max_bin = np.max(displacement_for_binning)
min_bin = np.min(displacement_for_binning)

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
        idx_cat = df_groups.displacement.idxmax()
        retained_nav_spokes = (categories == idx_cat)

retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
spoke_groups = np.argmin(np.abs(np.arange(0, nb_segments * nb_slices, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_slices,nb_segments / nb_gating_spokes).reshape(1,-1)),axis=-1)
included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
included_spokes[::int(nb_segments/nb_gating_spokes)]=False
#included_spokes[:]=True

# perc_retained=0.4
# import random
# indices_included_random=random.sample(range(spoke_groups.shape[0]),int(perc_retained*spoke_groups.shape[0]))
# included_spokes=np.zeros(spoke_groups.shape[0])
# included_spokes[indices_included_random]=1.0
# included_spokes=included_spokes.astype(bool)

#traj = radial_traj.get_traj()

print("Filtering KData for movement...")
kdata_retained_final_list = []
for i in tqdm(range(nb_channels)):
    kdata_retained_final, traj_retained_final, retained_timesteps = correct_mvt_kdata(
        kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, ntimesteps, density_adj=True,log=False)
    kdata_retained_final_list.append(kdata_retained_final)

print("Rebuilding Images With Corrected volumes...")

radial_traj_3D_corrected=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
radial_traj_3D_corrected.traj_for_reconstruction=traj_retained_final

if str.split(filename_volume_corrected,"/")[-1] not in os.listdir(folder):
    volumes_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list,radial_traj_3D_corrected,image_size,b1=b1_full,ntimesteps=len(retained_timesteps),density_adj=False,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,is_theta_z_adjusted=True,normalize_volumes=False)
    #animate_images(volumes_corrected[:,0,:,:])
    np.save(filename_volume_corrected,volumes_corrected)
else:
    volumes_corrected=np.load(filename_volume_corrected)

animate_images(volumes_corrected[:,int(nb_slices/2),:,:])

#volumes for slice taking into account coil sensi
print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_full,density_adj=False,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=False)
    np.save(filename_volume,volumes_all)
    # sl=20
    # ani = animate_images(volumes_all[:,sl,:,:])
    del volumes_all

print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    #selected_spokes = np.r_[10:400]
    selected_spokes=None
    mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_full,density_adj=False,threshold_factor=1/25, normalize_kdata=True,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask,mask)
    #animate_images(mask)
    del mask






del kdata_all_channels_all_slices


########################## Dict mapping ########################################


load_map=False
save_map=True


mask = np.load(filename_mask)
mask=m.mask
volumes_all = np.load(filename_volume)

animate_images(mask)


#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])



if not(load_map):
    niter = 10
    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=radial_traj,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=False,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps,threshold=None,tv_denoising_weight=0.1)
    all_maps=optimizer.search_patterns_test(dictfile,volumes_all,retained_timesteps=None)

    if(save_map):
        import pickle
        file = open(file_map, "wb")
        # dump information to that file
        pickle.dump(all_maps, file)
        # close the file
        file.close()

else:
    import pickle

    file = open(file_map, "rb")
    all_maps = pickle.load(file)

plot_evolution_params(m.paramMap,m.mask>0,all_maps,maskROI=buildROImask_unique(m.paramMap))

iter=0
regression_paramMaps_ROI(m.paramMap,all_maps[iter][0],m.mask>0,all_maps[iter][1]>0,buildROImask_unique(m.paramMap))

iter=2
regression_paramMaps_ROI(m.paramMap,all_maps[iter][0],m.mask>0,all_maps[iter][1]>0,buildROImask_unique(m.paramMap))

iter=3
regression_paramMaps_ROI(m.paramMap,all_maps[iter][0],m.mask>0,all_maps[iter][1]>0,buildROImask_unique(m.paramMap))


iter=10
regression_paramMaps_ROI(m.paramMap,all_maps[iter][0],m.mask>0,all_maps[iter][1]>0,buildROImask_unique(m.paramMap))











#########################@

#GAUSSIAN WEIGHTING




#import matplotlib
#matplotlib.u<se("TkAgg")
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

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"
nav_folder = "./data/InVivo/3D"

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

sig_0=3*np.pi/4
gw=GaussianWeighting(sig=sig_0)


nb_filled_slices = 8
nb_empty_slices=0
repeat_slice=2
nb_slices = nb_filled_slices+2*nb_empty_slices
name = "SquareSimu3D"

use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

filename_nav_save=str.split(nav_folder+"/phantom.001.v1/phantom.001.v1.dat",".dat") [0]+"_nav.npy"
nb_gating_spokes=50
#filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"

folder = "/".join(str.split(filename,"/")[:-1])

suffix="_GW2"

filename_paramMap=filename+"_paramMap_sl{}_rp{}.pkl".format(nb_slices,repeat_slice)
filename_volume = filename+"_volumes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,suffix)
filename_groundtruth = filename+"_groundtruth_volumes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")

filename_volume_corrected = filename+"_volumes_corrected_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,suffix)
filename_kdata = filename+"_kdata_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_mask= filename+"_mask_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
file_map = filename + "_sl{}_rp{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,suffix)


#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



data_for_nav=np.load(filename_nav_save)

ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 512

undersampling_factor=1

incoherent=True
mode="old"



with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
mask_reduction_factor=1/4



image_size = (nb_slices, int(npoint/2), int(npoint/2))
size=image_size[1:]
nb_segments=nb_allspokes
nspoke=int(nb_segments/ntimesteps)


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
m = RandomMap3D(name,dict_config,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,undersampling_factor=undersampling_factor,repeat_slice=repeat_slice,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)



if str.split(filename_paramMap,"/")[-1] not in os.listdir(folder):
    m.buildParamMap()
    with open(filename_paramMap, "wb" ) as file:
        pickle.dump(m.paramMap, file)

    map_rebuilt = m.paramMap
    mask = m.mask

    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    for key in ["ff", "wT1", "df", "attB1"]:
        file_mha = "/".join(["/".join(str.split(filename_paramMap, "/")[:-1]),
                             "_".join(str.split(str.split(filename_paramMap, "/")[-1], ".")[:-1])]) + "_{}.mha".format(
                                                                                                                   key)
        io.write(file_mha, map_for_sim[key], tags={"spacing": [5, 1, 1]})

else:
    with open(filename_paramMap, "rb") as file:
        m.paramMap=pickle.load(file)




m.build_ref_images(seq)

if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
    np.save(filename_groundtruth,m.images_series[::nspoke])



if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    # Density adjustment all slices
    data=m.generate_kdata(radial_traj)
    data=np.expand_dims(data,axis=0)

    density = np.abs(np.linspace(-1, 1, npoint))
    density = np.expand_dims(density, tuple(range(data.ndim - 1)))

    #kdata_all_channels_all_slices = data.reshape(-1, npoint)
    #del data
    print("Performing Density Adjustment....")

    data = data.reshape(1, nb_allspokes, -1, npoint)
    data *= density

    np.save(filename_kdata, data)
    del data

    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)


b1_full = np.ones(image_size)
b1_full=np.expand_dims(b1_full,axis=0)

traj=radial_traj.get_traj()
kdata_all_channels_all_slices=[gw.apply(traj[j])*k.reshape(nb_segments,-1) for j,k in enumerate(kdata_all_channels_all_slices)]
kdata_all_channels_all_slices=np.array(kdata_all_channels_all_slices)

print("Processing Nav Data...")
data_for_nav=np.load(filename_nav_save)

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
shifts = list(range(-20, 20))
bottom = 50
top = 150
displacements, _ = calculate_displacement(images_nav_mean, bottom, top, shifts)

displacement_for_binning = displacements
bin_width = 8
max_bin = np.max(displacement_for_binning)
min_bin = np.min(displacement_for_binning)

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
        idx_cat = df_groups.displacement.idxmax()
        retained_nav_spokes = (categories == idx_cat)

retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
spoke_groups = np.argmin(np.abs(np.arange(0, nb_segments * nb_slices, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_slices,nb_segments / nb_gating_spokes).reshape(1,-1)),axis=-1)
included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
included_spokes[::int(nb_segments/nb_gating_spokes)]=False
#included_spokes[:]=True

# perc_retained=0.4
# import random
# indices_included_random=random.sample(range(spoke_groups.shape[0]),int(perc_retained*spoke_groups.shape[0]))
# included_spokes=np.zeros(spoke_groups.shape[0])
# included_spokes[indices_included_random]=1.0
# included_spokes=included_spokes.astype(bool)

#traj = radial_traj.get_traj()

print("Filtering KData for movement...")
kdata_retained_final_list = []
for i in tqdm(range(nb_channels)):
    kdata_retained_final, traj_retained_final, retained_timesteps = correct_mvt_kdata(
        kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, ntimesteps, density_adj=True,log=False)
    kdata_retained_final_list.append(kdata_retained_final)

print("Rebuilding Images With Corrected volumes...")

radial_traj_3D_corrected=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
radial_traj_3D_corrected.traj_for_reconstruction=traj_retained_final

if str.split(filename_volume_corrected,"/")[-1] not in os.listdir(folder):
    volumes_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list,radial_traj_3D_corrected,image_size,b1=b1_full,ntimesteps=len(retained_timesteps),density_adj=False,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,is_theta_z_adjusted=True,normalize_volumes=False)
    #animate_images(volumes_corrected[:,0,:,:])
    np.save(filename_volume_corrected,volumes_corrected)
else:
    volumes_corrected=np.load(filename_volume_corrected)

animate_images(volumes_corrected[:,int(nb_slices/2),:,:])

#volumes for slice taking into account coil sensi
print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):

    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_full,density_adj=False,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=False)
    np.save(filename_volume,volumes_all)
    # sl=20
    ani = animate_images(volumes_all[:,4,:,:])
    del volumes_all

print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    #selected_spokes = np.r_[10:400]
    selected_spokes=None
    mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_full,density_adj=False,threshold_factor=1/25, normalize_kdata=True,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask,mask)
    #animate_images(mask)
    del mask






del kdata_all_channels_all_slices


########################## Dict mapping ########################################


load_map=False
save_map=True


mask = np.load(filename_mask)
mask=m.mask
volumes_all = np.load(filename_volume)

animate_images(mask)


#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])



if not(load_map):
    niter = 10
    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=100, pca=True,
                                 threshold_pca=20, log=True, useGPU_dictsearch=False, useGPU_simulation=False,
                                 gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                 Weighting=gw)

    all_maps=optimizer.search_patterns_MRF_denoising(dictfile,volumes_all,retained_timesteps=None)

    if(save_map):
        import pickle
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


plot_evolution_params(m.paramMap,m.mask>0,all_maps,maskROI=buildROImask_unique(m.paramMap),adj_wT1=True)

iter=0
regression_paramMaps_ROI(m.paramMap,all_maps[iter][0],m.mask>0,all_maps[iter][1]>0,buildROImask_unique(m.paramMap))

iter=2
regression_paramMaps_ROI(m.paramMap,all_maps[iter][0],m.mask>0,all_maps[iter][1]>0,buildROImask_unique(m.paramMap))

iter=3
regression_paramMaps_ROI(m.paramMap,all_maps[iter][0],m.mask>0,all_maps[iter][1]>0,buildROImask_unique(m.paramMap))


iter=10
regression_paramMaps_ROI(m.paramMap,all_maps[iter][0],m.mask>0,all_maps[iter][1]>0,buildROImask_unique(m.paramMap))







#import matplotlib
#matplotlib.u<se("TkAgg")
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

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"
nav_folder = "./data/InVivo/3D"

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

nb_filled_slices = 8
nb_empty_slices=0
repeat_slice=2
nb_slices = nb_filled_slices+2*nb_empty_slices
name = "SquareSimu3D"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

filename_nav_save=str.split(nav_folder+"/phantom.001.v1/phantom.001.v1.dat",".dat") [0]+"_nav.npy"
nb_gating_spokes=50
#filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"

folder = "/".join(str.split(filename,"/")[:-1])

suffix=""



filename_paramMap=filename+"_paramMap_sl{}_rp{}{}.pkl".format(nb_slices,repeat_slice,name)
filename_volume = filename+"_volumes_sl{}_rp{}_{}{}.npy".format(nb_slices,repeat_slice,name,suffix)
filename_volume_corrected = filename+"_volumes_corrected_sl{}_rp{}_{}{}.npy".format(nb_slices,repeat_slice,name,suffix)
filename_kdata = filename+"_kdata_sl{}_rp{}_{}{}.npy".format(nb_slices,repeat_slice,name,"")
filename_mask= filename+"_mask_sl{}_rp{}_{}{}.npy".format(nb_slices,repeat_slice,name,"")
file_map = filename + "_sl{}_rp{}_{}_GW{}_MRF_map.pkl".format(nb_slices,repeat_slice,name,suffix)


#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"

gw=GaussianWeighting(sig=np.pi/2)

use_GPU = False
light_memory_usage=True

data_for_nav=np.load(filename_nav_save)

ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 512

undersampling_factor=1

incoherent=True
mode="old"



with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
mask_reduction_factor=1/4



image_size = (nb_slices, int(npoint/2), int(npoint/2))
size=image_size[1:]


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
m = RandomMap3D(name,dict_config,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,undersampling_factor=undersampling_factor,repeat_slice=repeat_slice,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode="other")



if str.split(filename_paramMap,"/")[-1] not in os.listdir(folder):
    m.buildParamMap()
    with open(filename_paramMap, "wb" ) as file:
        pickle.dump(m.paramMap, filename_paramMap)

else:
    m.paramMap=pickle.load(filename_paramMap)
    file.close()

if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    # Density adjustment all slices
    data=m.generate_kdata(radial_traj)
    data=np.expand_dims(data,axis=0)

    if density_adj_radial:
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density,tuple(range(data.ndim-1)))
    else:
        density=1
    #kdata_all_channels_all_slices = data.reshape(-1, npoint)
    #del data
    print("Performing Density Adjustment....")
    data *= density
    np.save(filename_kdata, data)
    del data

    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)

traj=radial_traj.get_traj()
kdata_all_channels_all_slices=[gw.apply(traj[j])*k for j,k in enumerate(kdata_all_channels_all_slices)]

b1_full = np.ones(image_size)
b1_full=np.expand_dims(b1_full,axis=0)


print("Processing Nav Data...")
data_for_nav=np.load(filename_nav_save)

all_timesteps = np.arange(nb_allspokes)
nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint, nb_slices=nb_slices,
                       applied_timesteps=list(nav_timesteps))

nav_image_size = (int(npoint / 2),)

print("Calculating Sensitivity Maps for Nav Images...")
b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
b1_nav_mean = np.mean(b1_nav, axis=(1, 2))


print("Rebuilding Nav Images...")
images_nav_mean = np.abs(simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, b1_nav_mean))


print("Estimating Movement...")
shifts = list(range(-20, 20))
bottom = 50
top = 150
displacements, _ = calculate_displacement(images_nav_mean, bottom, top, shifts)

displacement_for_binning = displacements
bin_width = 8
max_bin = np.max(displacement_for_binning)
min_bin = np.min(displacement_for_binning)

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
        idx_cat = df_groups.displacement.idxmax()
        retained_nav_spokes = (categories == idx_cat)

retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
spoke_groups = np.argmin(np.abs(np.arange(0, nb_segments * nb_slices, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_slices,nb_segments / nb_gating_spokes).reshape(1,-1)),axis=-1)
included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
included_spokes[::int(nb_segments/nb_gating_spokes)]=False
#included_spokes[:]=True

# perc_retained=0.4
# import random
# indices_included_random=random.sample(range(spoke_groups.shape[0]),int(perc_retained*spoke_groups.shape[0]))
# included_spokes=np.zeros(spoke_groups.shape[0])
# included_spokes[indices_included_random]=1.0
# included_spokes=included_spokes.astype(bool)

#traj = radial_traj.get_traj()

print("Filtering KData for movement...")
kdata_retained_final_list = []
for i in tqdm(range(nb_channels)):
    kdata_retained_final, traj_retained_final, retained_timesteps = correct_mvt_kdata(
        kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, ntimesteps, density_adj=True,log=False)
    kdata_retained_final_list.append(kdata_retained_final)

print("Rebuilding Images With Corrected volumes...")

radial_traj_3D_corrected=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
radial_traj_3D_corrected.traj_for_reconstruction=traj_retained_final

if str.split(filename_volume_corrected,"/")[-1] not in os.listdir(folder):
    volumes_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list,radial_traj_3D_corrected,image_size,b1=b1_full,ntimesteps=len(retained_timesteps),density_adj=False,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,is_theta_z_adjusted=True,normalize_volumes=False)
    #animate_images(volumes_corrected[:,0,:,:])
    np.save(filename_volume_corrected,volumes_corrected)
else:
    volumes_corrected=np.load(filename_volume_corrected)

##volumes for slice taking into account coil sensi
# print("Building Volumes....")
# if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
#     volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=True,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
#     np.save(filename_volume,volumes_all)
#     # sl=20
#     # ani = animate_images(volumes_all[:,sl,:,:])
#     del volumes_all
#
print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    #selected_spokes = np.r_[10:400]
    selected_spokes=None
    mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/25, normalize_kdata=True,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask,mask)
    #animate_images(mask)
    del mask






del kdata_all_channels_all_slices


########################## Dict mapping ########################################

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)


load_map=False
save_map=True


mask = np.load(filename_mask)
#volumes_all = np.load(filename_volume)


#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])



if not(load_map):
    niter = 10
    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=radial_traj,split=100,pca=True,threshold_pca=20,log=True,useGPU_dictsearch=False,useGPU_simulation=False,gen_mode="other",movement_correction=True,cond=included_spokes,ntimesteps=ntimesteps,Weighting=gw)
    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=100, pca=True,
                                 threshold_pca=20, log=True, useGPU_dictsearch=False, useGPU_simulation=False,
                                 gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                Weighting=gw)

    all_maps=optimizer.search_patterns_MRF_denoising(dictfile,volumes_all,retained_timesteps=None)

    if(save_map):
        import pickle
        file_map='./3D/SquareSimu3D_sl8_rp2_GW_MRF_map'
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



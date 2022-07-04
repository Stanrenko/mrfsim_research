
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

name = "SquareSimu3DMTRandom"


use_GPU = True
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

filename_volume = filename+"_volumes_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_volume_corrected_final = filename+"_volumes_corrected_final_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask= filename+"_mask_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
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

if name=="SquareSimu3DMTRandom":
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
    np.save(filename_kdata, data)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    data = np.load(filename_kdata)


data=data.reshape(nb_segments,-1,npoint)
density = np.abs(np.linspace(-1, 1, npoint))
density = np.expand_dims(density,tuple(range(data.ndim-1)))
kdata_all_channels_all_slices =data* density
kdata_all_channels_all_slices=np.expand_dims(kdata_all_channels_all_slices,axis=0)

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

volumes_all=np.load(filename_volume)
sl=int(nb_slices/2)
ani = animate_images(volumes_all[:,sl,:,:])

nb_gating_spokes=1400
ts_between_spokes=int(nb_allspokes/nb_gating_spokes)
timesteps = list(np.arange(1400)[::ts_between_spokes])



nav_z=Navigator3D(direction=[1,0,0.0],applied_timesteps=timesteps,npoint=npoint)
kdata_nav = m_.generate_kdata(nav_z,useGPU=use_GPU)

kdata_nav=np.array(kdata_nav)

data_for_nav = np.expand_dims(kdata_nav,axis=0)
data_for_nav =  np.moveaxis(data_for_nav,1,2)
data_for_nav = data_for_nav.astype("complex64")


all_timesteps = np.arange(nb_allspokes)
nav_timesteps = timesteps

nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint, nb_slices=nb_slices,
                       applied_timesteps=list(nav_timesteps))

nav_image_size = (int(npoint / 2),)

print("Rebuilding Nav Images...")
images_nav_mean = np.abs(simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, None))
plt.figure()
plt.imshow(np.abs(np.repeat(images_nav_mean,20).reshape(-1,20*int(npoint/2))).T)



nav_z_all=Navigator3D(direction=[1,0,0.0],applied_timesteps=np.arange(nb_segments),npoint=npoint)
kdata_nav_all = m_.generate_kdata(nav_z_all,useGPU=use_GPU)

kdata_nav_all=np.array(kdata_nav_all)

data_for_nav_all = np.expand_dims(kdata_nav_all,axis=0)
data_for_nav_all =  np.moveaxis(data_for_nav_all,1,2)
data_for_nav_all = data_for_nav_all.astype("complex64")
nav_traj_all = Navigator3D(direction=[0, 0, 1], npoint=npoint, nb_slices=nb_slices,
                       applied_timesteps=np.arange(nb_segments))
images_nav_mean_all = np.abs(simulate_nav_images_multi(data_for_nav_all, nav_traj_all, nav_image_size, None))

plt.figure()
plt.imshow(np.abs(np.repeat(images_nav_mean_all,50).reshape(-1,50*int(npoint/2))).T)



print("Estimating Movement...")
shifts = list(range(-10, 10))
bottom = 20
top = 50

displacements, _ = calculate_displacement(images_nav_mean, bottom, top, shifts)

displacement_for_binning = displacements
bin_width = 2
max_bin = np.max(displacement_for_binning)
min_bin = np.min(displacement_for_binning)

bins = np.arange(min_bin, max_bin + bin_width, bin_width)
#print(bins)
categories = np.digitize(displacement_for_binning, bins)
df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
df_groups = df_cat.groupby("cat").count()

group_1=(categories==1)|(categories==2)
group_2=(categories==3)|(categories==4)
group_3=(categories==5)|(categories==6)

groups=[group_1,group_2,group_3]

plt.figure()
gr=0
plt.imshow(np.abs(np.repeat(images_nav_mean,20).reshape(-1,20*int(npoint/2))).T)
for i,is_spoke_in_group in enumerate(groups[gr]):
    if is_spoke_in_group:
        plt.axvline(x=i,alpha=0.005,c="r")

plt.figure()
gr=0
plt.plot(displacement_for_binning)
for i,is_spoke_in_group in enumerate(groups[gr]):
    if is_spoke_in_group:
        plt.axvline(x=i,alpha=0.01,c="r")



nb_part=nb_slices
plt.figure()
gr=0
g=groups[gr]

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
for i,is_spoke_in_group in enumerate(included_spokes):
    if is_spoke_in_group:
        plt.axvline(x=i,alpha=0.005,c="r")
plt.imshow(np.abs(np.repeat(images_nav_mean_all,50).reshape(-1,50*int(npoint/2))).T)



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

    kdata_retained_final, traj_retained_final_volume, retained_timesteps = correct_mvt_kdata(
        kdata_all_channels_all_slices[0].reshape(nb_segments, -1), radial_traj, included_spokes, 1,
        density_adj=True, log=False)
    #print(traj_retained_final_volume.shape[1]/800)

    dico_traj_retained[j] = traj_retained_final_volume

nyquist_total_points = np.prod(image_size)*np.pi/2
for j,gr in enumerate(groups):
    print("Group {} : {} % of Nyquist criteria".format(j,np.round(dico_traj_retained[j].shape[1]/nyquist_total_points*100,2)))

for j,gr in enumerate(groups):
    print("Group {} : {} % of Kz Nyquist criteria".format(j,len(np.unique(dico_traj_retained[j][:,:,2]))/nb_slices*100,2))

all_kz = np.unique(radial_traj.get_traj()[:,:,-1])

df_sampling = pd.DataFrame(columns=range(len(groups)),index=all_kz,data=0)

for j,g in tqdm(enumerate(groups)):
    traj_retained = dico_traj_retained[j]
    for kz in all_kz:
        curr_traj = traj_retained[traj_retained[:,:,2]==kz]
        df_sampling.loc[kz,j]=curr_traj.shape[0]/(npoint**2*np.pi/8)

df_sampling.loc["Total"]=[len(np.unique(dico_traj_retained[j][:,:,2]))/nb_slices for j,g in enumerate(groups)]

dico_volume = {}
dico_mask = {}

#j=2
#g=groups[j]


resol_factor=1.0
center_point=int(npoint/2)





for j, g in tqdm(enumerate(groups)):
    print("######################  BUILDING FULL VOLUME AND MASK FOR GROUP {} ##########################".format(j))
    retained_nav_spokes_index = np.argwhere(g).flatten()
    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
                             axis=-1)
    spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
        nb_segments / nb_gating_spokes / 2) + 1:] - 1  # adjustment for change of partition
    spoke_groups = spoke_groups.flatten()

    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
    #included_spokes[::int(nb_segments / nb_gating_spokes)] = False
    print("Filtering KData for movement...")
    kdata_retained_final_list_volume = []
    for i in tqdm(range(nb_channels)):
        kdata_retained_final, traj_retained_final_volume, retained_timesteps = correct_mvt_kdata(
            kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, 1,
            density_adj=True, log=False)
        kdata_retained_final_list_volume.append(kdata_retained_final)

    radial_traj_3D_corrected_single_volume = Radial3D(total_nspokes=nb_allspokes,
                                                      undersampling_factor=undersampling_factor, npoint=npoint,
                                                      nb_slices=nb_slices, incoherent=incoherent, mode=mode)

    if resol_factor<1:
        traj_retained_final_volume=traj_retained_final_volume.reshape(1,-1,npoint,3)
        traj_retained_final_volume=traj_retained_final_volume[:,:,(center_point-int(resol_factor*npoint/2)):(center_point+int(resol_factor*npoint/2)),:]
        kdata_retained_final_list_volume=np.array(kdata_retained_final_list_volume)
        kdata_retained_final_list_volume=kdata_retained_final_list_volume.reshape(nb_channels,1,-1,npoint)
        kdata_retained_final_list_volume = kdata_retained_final_list_volume[:, :, :,(center_point - int(resol_factor * npoint / 2)):(
                    center_point + int(resol_factor * npoint / 2))]

    radial_traj_3D_corrected_single_volume.traj_for_reconstruction = traj_retained_final_volume

    volume_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list_volume,
                                                                 radial_traj_3D_corrected_single_volume, image_size,
                                                                 b1=b1_all_slices, density_adj=False, ntimesteps=1,
                                                                 useGPU=False, normalize_kdata=False, memmap_file=None,
                                                                 light_memory_usage=True, normalize_volumes=True,
                                                                 is_theta_z_adjusted=True)

    file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
                         "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[
                                  :-1])]) + "_fullvol_gr{}_resol_{}_b1.mha".format(j, "_".join(str.split(str(resol_factor), ".")))

    io.write(file_mha, np.abs(volume_corrected[0]), tags={"spacing": [5, 1, 1]})

    dico_volume[j] = copy(volume_corrected[0])
    mask = build_mask_single_image_multichannel(kdata_retained_final_list_volume,
                                                radial_traj_3D_corrected_single_volume, image_size, b1=b1_all_slices,
                                                density_adj=False, threshold_factor=1 / 15, normalize_kdata=False,
                                                light_memory_usage=True, selected_spokes=None, is_theta_z_adjusted=True,
                                                normalize_volumes=True)
    dico_mask[j] = copy(mask)

volume_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,
                                                                 radial_traj, image_size,
                                                                 b1=None, density_adj=False, ntimesteps=1,
                                                                 useGPU=False, normalize_kdata=False, memmap_file=None,
                                                                 light_memory_usage=True, normalize_volumes=True,
                                                                 is_theta_z_adjusted=False)

file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
                         "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_fullvol_all_groups.mha"

io.write(file_mha,np.abs(volume_all[0]),tags={"spacing":[5,1,1]})

animate_images(mask)

del volume_corrected
del kdata_retained_final_list_volume
del radial_traj_3D_corrected_single_volume
del mask

import cv2
import numpy as np




for j in dico_volume.keys():
    plt.figure(figsize=(10,10))
    #plt.imshow(denoise_tv_chambolle(np.abs((dico_mask[j]*dico_volume[j])[int(nb_slices/2),:,:]),weight=0.00001))
    plt.imshow(np.abs((dico_mask[j][int(nb_slices/2),:,:]*dico_volume[j][int(nb_slices/2),:,:])))
    #plt.imshow(np.abs((dico_mask[j][int(nb_slices / 2), :, :])))


plt.figure(figsize=(10,10))
for j in dico_volume.keys():

    #plt.imshow(denoise_tv_chambolle(np.abs((dico_mask[j]*dico_volume[j])[int(nb_slices/2),:,:]),weight=0.00001))
    image = np.abs((dico_mask[j][int(nb_slices/2),int(image_size[1]/2),:]*dico_volume[j][int(nb_slices/2),int(image_size[1]/2),:]))
    plt.plot(image)
    print("y-width group {} : {}".format(j,len(image[image>0])))
    #plt.imshow(np.abs((dico_mask[j][int(nb_slices / 2), :, :])))


plt.figure(figsize=(10,10))
for j in dico_volume.keys():
    image = np.abs((dico_mask[j][int(nb_slices/2),:,int(image_size[1]/2)]*dico_volume[j][int(nb_slices/2),:,int(image_size[1]/2)]))
    #plt.imshow(denoise_tv_chambolle(np.abs((dico_mask[j]*dico_volume[j])[int(nb_slices/2),:,:]),weight=0.00001))
    plt.plot(image)
    #plt.imshow(np.abs((dico_mask[j][int(nb_slices / 2), :, :])))
    print("x-width group {} : {}".format(j, len(image[image > 0])))



#TEST FABIAN CODE
from pymia.filtering.registration import MultiModalRegistration,MultiModalRegistrationParams,RegistrationType
import SimpleITK as sitk



index_ref = 1

dico_homographies = {}
registration = MultiModalRegistration(registration_type=RegistrationType.RIGID)

for index_to_align in tqdm(dico_volume.keys()):
    dico_homographies[index_to_align] = {}
    for sl in range(nb_slices):
        if index_to_align==index_ref:
            dico_homographies[index_to_align][sl] = sitk.Transform(3, sitk.sitkIdentity)
        else:
            array_to_align = np.abs(dico_mask[index_to_align][sl] * dico_volume[index_to_align][sl])
            array_ref = np.abs(dico_mask[index_ref][sl] * dico_volume[index_ref][sl])

            fixed_image = sitk.GetImageFromArray(array_ref)
            moving_image = sitk.GetImageFromArray(array_to_align)

              # specify parameters to your needs
            params = MultiModalRegistrationParams(fixed_image)
            registration.execute(moving_image, params)

            dico_homographies[index_to_align][sl] = registration.transform


# sl = int(nb_slices / 2)
# test_index=2
#
# array_ref = np.abs(dico_mask[index_ref][sl] * dico_volume[index_ref][sl])
#
# array_to_align = np.abs(dico_mask[test_index][sl] * dico_volume[test_index][sl])
# moving_image = sitk.GetImageFromArray(array_to_align)
#
# registered_image = registration.execute(moving_image, dico_homographies[test_index][sl])
# registered_array = sitk.GetArrayFromImage(registered_image)
#
# animate_images([array_ref,registered_array])

#
from sklearn.metrics import mutual_info_score

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

test_index=0
for sl in range(nb_slices):
    print("############### SLICE {} ################".format(sl))
    array_ref = np.abs(dico_mask[index_ref][sl] * dico_volume[index_ref][sl])
    fixed_image = sitk.GetImageFromArray(array_ref)

    array_to_align = np.abs(dico_mask[test_index][sl] * dico_volume[test_index][sl])
    moving_image = sitk.GetImageFromArray(array_to_align)

    registered_image = sitk.Resample(moving_image, transform=dico_homographies[test_index][sl],
                                     interpolator=registration.resampling_interpolator,
                                     outputPixelType=moving_image.GetPixelIDValue())
    registered_array = sitk.GetArrayFromImage(registered_image)

    #animate_images([array_ref, registered_array])
    #animate_images([array_ref, array_to_align])

    print("MI score for unregistered image : {}".format(np.round(calc_MI(array_ref.flatten(),array_to_align.flatten(),bins=100),3)))
    print("MI score for registered image : {}".format(
        np.round(calc_MI(array_ref.flatten(),registered_array.flatten(),bins=100), 3)))



sl = int(nb_slices / 2)
#sl=1
#sl=41
#sl=30

array_ref = np.abs(dico_mask[index_ref][sl] * dico_volume[index_ref][sl])
fixed_image = sitk.GetImageFromArray(array_ref)

array_to_align = np.abs(dico_mask[test_index][sl] * dico_volume[test_index][sl])
moving_image = sitk.GetImageFromArray(array_to_align)

registered_image = sitk.Resample(moving_image,transform=dico_homographies[test_index][sl],interpolator=registration.resampling_interpolator,outputPixelType=moving_image.GetPixelIDValue())
registered_array = sitk.GetArrayFromImage(registered_image)

animate_images([array_ref,registered_array])
animate_images([array_ref,array_to_align])

animate_images([array_to_align,registered_array])

fig=plt.figure(frameon=False)
im1=plt.imshow(array_ref)
im2=plt.imshow(array_to_align,alpha=0.5)
plt.show()

fig=plt.figure(frameon=False)
im1=plt.imshow(array_ref)
im2=plt.imshow(registered_array,alpha=0.5)
plt.show()


volumes_corrected_final=np.zeros((ntimesteps,nb_slices,int(npoint/2),int(npoint/2)),dtype="complex64")
ts_indices=np.zeros(len(groups)).astype(int)
count=np.zeros(ntimesteps).astype(int)
#total_weight=np.zeros(ntimesteps).astype(int)

del dico_mask
del dico_volume
#
# dico_kdata_retained_registration = {}
# dico_volumes_corrected = {}
#
# J=0
# g=groups[0]

dico_volumes_corrected={}
dico_volumes_corrected_registered={}
for j, g in tqdm(enumerate(groups)):
    retained_nav_spokes_index = np.argwhere(g).flatten()
    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
                             axis=-1)
    spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
        nb_segments / nb_gating_spokes / 2) + 1:] - 1  # adjustment for change of partition
    spoke_groups = spoke_groups.flatten()

    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
    #included_spokes[::int(nb_segments / nb_gating_spokes)] = False
    print("Filtering KData for movement...")
    kdata_retained_final_list = []
    for i in (range(nb_channels)):
        kdata_retained_final, traj_retained_final, retained_timesteps = correct_mvt_kdata(
            kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, ntimesteps,
            density_adj=True, log=False)
        kdata_retained_final_list.append(kdata_retained_final)

    #dico_kdata_retained_registration[j] = retained_timesteps



    radial_traj_3D_corrected = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor,
                                        npoint=npoint, nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    radial_traj_3D_corrected.traj_for_reconstruction = traj_retained_final

    volumes_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list, radial_traj_3D_corrected,
                                                                  image_size, b1=b1_all_slices,
                                                                  ntimesteps=len(retained_timesteps), density_adj=False,
                                                                  useGPU=False, normalize_kdata=False, memmap_file=None,
                                                                  light_memory_usage=True, is_theta_z_adjusted=True,
                                                                  normalize_volumes=True)

    dico_volumes_corrected[j]=copy(volumes_corrected)
    print("Re-registering corrected volumes")
    for ts in tqdm(range(volumes_corrected.shape[0])):
        for sl in range(volumes_corrected.shape[1]):

            moving_image_real = sitk.GetImageFromArray(volumes_corrected[ts, sl, :, :].real)
            moving_image_imag = sitk.GetImageFromArray(volumes_corrected[ts, sl, :, :].imag)

            volumes_corrected[ts, sl, :, :] = sitk.GetArrayFromImage( sitk.Resample(moving_image_real, transform=dico_homographies[j][sl],
                                             interpolator=registration.resampling_interpolator,
                                             outputPixelType=moving_image_real.GetPixelIDValue())) + 1j * sitk.GetArrayFromImage( sitk.Resample(moving_image_imag, transform=dico_homographies[j][sl],
                                             interpolator=registration.resampling_interpolator,
                                             outputPixelType=moving_image_imag.GetPixelIDValue()))



            #registration.execute(sitk.GetImageFromArray(volumes_corrected[ts, sl, :, :].real),
            #                                                                 dico_homographies[j][sl]))+1j*sitk.GetArrayFromImage(registration.execute(sitk.GetImageFromArray(volumes_corrected[ts, sl, :, :].imag),
            #                                                                 dico_homographies[j][sl]))
    #dico_volumes_corrected[j] = copy(volumes_corrected)
    dico_volumes_corrected_registered[j] = volumes_corrected

    print("Forming final volumes with contribution from group {}".format(j))
    for ts in tqdm(range(ntimesteps)):
        if ts in retained_timesteps:
            volumes_corrected_final[ts]+=volumes_corrected[ts_indices[j]]*traj_retained_final[ts_indices[j]].shape[0]
            #count[ts]+=1
            count[ts] += traj_retained_final[ts_indices[j]].shape[0]
            ts_indices[j] += 1
            #total_weight[ts]+=traj_retained_final[ts_indices[j]].shape[0]


count=np.expand_dims(count,axis=tuple(range(1,volumes_corrected_final.ndim)))
volumes_corrected_final/=count
np.save(filename_volume_corrected_final,volumes_corrected_final)

sl=int(nb_slices/2)
ts = np.random.choice(ntimesteps)
gr = 0

animate_images([dico_volumes_corrected[gr][ts,sl,:,:],dico_volumes_corrected_registered[gr][ts,sl,:,:]])
animate_images([dico_volumes_corrected[index_ref][ts,sl,:,:],dico_volumes_corrected_registered[gr][ts,sl,:,:]])
animate_images([dico_volumes_corrected[0][ts,sl,:,:],dico_volumes_corrected[1][ts,sl,:,:],dico_volumes_corrected[2][ts,sl,:,:]])
animate_images([dico_volumes_corrected_registered[0][ts,sl,:,:],dico_volumes_corrected_registered[1][ts,sl,:,:],dico_volumes_corrected_registered[2][ts,sl,:,:]])
plt.figure()
plt.imshow(np.abs(volumes_corrected_final[ts,sl,:,:]))

plt.figure()
plt.imshow(np.abs(volumes_all[ts,sl,:,:]))

animate_multiple_images(dico_volumes_corrected[gr][:,sl,:,:],dico_volumes_corrected_registered[gr][:,sl,:,:])
animate_multiple_images(dico_volumes_corrected[gr][:,sl,:,:],dico_volumes_corrected_registered[gr][:,sl,:,:])



del volumes_corrected
del radial_traj_3D_corrected
del dico_homographies
del kdata_retained_final_list




#
animate_images(volumes_corrected_final[:,int(nb_slices/2),:,:])

######################################################################################""








#
# index_ref = 1
#
# dico_homographies = {}
#
# for index_to_align in dico_volume.keys():
#     dico_homographies[index_to_align] = {}
#     for sl in range(nb_slices):
#
#         # Open the image files.
#         array_to_align = np.abs(dico_mask[index_to_align][sl] * dico_volume[index_to_align][sl])
#         array_ref = np.abs(dico_mask[index_ref][sl] * dico_volume[index_ref][sl])
#
#         array_to_align = array_to_align / (array_to_align).max()
#         array_ref = array_ref / (array_ref).max()
#
#         uint_img_to_align = np.array(array_to_align * 255).astype('uint8')
#         uint_img_ref = np.array(array_ref * 255).astype('uint8')
#
#         # img1_color = np.abs() # Image to be aligned.
#         # img2_color = np.abs(dico_volume[1][int(nb_slices/2)])
#
#         # Convert to grayscale.
#         img1_color = cv2.cvtColor(uint_img_to_align, cv2.COLOR_GRAY2BGR)
#         img2_color = cv2.cvtColor(uint_img_ref, cv2.COLOR_GRAY2BGR)
#
#         img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
#         img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
#
#         # Find size of image1
#         sz = img1.shape
#
#         # Define the motion model
#         warp_mode = cv2.MOTION_TRANSLATION
#
#         # Define 2x3 or 3x3 matrices and initialize the matrix to identity
#         if warp_mode == cv2.MOTION_HOMOGRAPHY:
#             warp_matrix = np.eye(3, 3, dtype=np.float32)
#         else:
#             warp_matrix = np.eye(2, 3, dtype=np.float32)
#
#         # Specify the number of iterations.
#         number_of_iterations = 5000;
#
#         # Specify the threshold of the increment
#         # in the correlation coefficient between two iterations
#         termination_eps = 1e-10;
#
#         # Define termination criteria
#         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
#
#         # Run the ECC algorithm. The results are stored in warp_matrix.
#         (cc, warp_matrix) = cv2.findTransformECC(img2, img1, warp_matrix, warp_mode, criteria)
#
#         # if warp_mode == cv2.MOTION_HOMOGRAPHY :
#         # Use warpPerspective for Homography
#         #    transformed_img = cv2.warpPerspective (img1, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#         # else :
#         # Use warpAffine for Translation, Euclidean and Affine
#         #    transformed_img = cv2.warpAffine(img1, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
#
#         dico_homographies[index_to_align][sl] = warp_matrix
#
# sl = int(nb_slices / 2)
# test_index=0
#
# animate_multiple_images(np.abs(dico_volume[index_ref]),np.abs(dico_volume[test_index]))
#
# registered_volume=[]
# for sl in range(nb_slices):
#     registered_volume.append(scipy.ndimage.affine_transform(np.abs(dico_volume[test_index][sl].T), dico_homographies[test_index][sl]).T)
# registered_volume=np.array(registered_volume)
#
# animate_multiple_images(np.abs(dico_volume[index_ref]),registered_volume)
#
#
# animate_images([np.abs(dico_volume[test_index][sl]),
#                 np.abs(dico_volume[index_ref][sl])])
#
# animate_images([scipy.ndimage.affine_transform(np.abs(dico_volume[test_index][sl].T), dico_homographies[test_index][sl]).T,
#                 np.abs(dico_volume[index_ref][sl])])
#
# volumes_corrected_final=np.zeros((ntimesteps,nb_slices,int(npoint/2),int(npoint/2)),dtype="complex64")
# ts_indices=np.zeros(len(groups)).astype(int)
# count=np.zeros(ntimesteps).astype(int)
# #total_weight=np.zeros(ntimesteps).astype(int)
#
# del dico_mask
# del dico_volume
# #
# # dico_kdata_retained_registration = {}
# # dico_volumes_corrected = {}
# #
# # J=0
# # g=groups[0]
# for j, g in tqdm(enumerate(groups)):
#     retained_nav_spokes_index = np.argwhere(g).flatten()
#     spoke_groups = np.argmin(np.abs(
#         np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
#                                                                           nb_segments / nb_gating_spokes).reshape(1,
#                                                                                                                   -1)),
#                              axis=-1)
#     included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
#     included_spokes[::int(nb_segments / nb_gating_spokes)] = False
#     print("Filtering KData for movement...")
#     kdata_retained_final_list = []
#     for i in (range(nb_channels)):
#         kdata_retained_final, traj_retained_final, retained_timesteps = correct_mvt_kdata(
#             kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, ntimesteps,
#             density_adj=True, log=False)
#         kdata_retained_final_list.append(kdata_retained_final)
#
#     #dico_kdata_retained_registration[j] = retained_timesteps
#
#
#
#     radial_traj_3D_corrected = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor,
#                                         npoint=npoint, nb_slices=nb_slices, incoherent=incoherent, mode=mode)
#     radial_traj_3D_corrected.traj_for_reconstruction = traj_retained_final
#
#     volumes_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list, radial_traj_3D_corrected,
#                                                                   image_size, b1=b1_all_slices,
#                                                                   ntimesteps=len(retained_timesteps), density_adj=False,
#                                                                   useGPU=False, normalize_kdata=False, memmap_file=None,
#                                                                   light_memory_usage=True, is_theta_z_adjusted=True,
#                                                                   normalize_volumes=True)
#
#     print("Re-registering corrected volumes")
#     for ts in tqdm(range(volumes_corrected.shape[0])):
#         for sl in range(volumes_corrected.shape[1]):
#             volumes_corrected[ts, sl, :, :] = scipy.ndimage.affine_transform(volumes_corrected[ts, sl, :, :].T,
#                                                                              dico_homographies[j][sl]).T
#     #dico_volumes_corrected[j] = copy(volumes_corrected)
#     print("Forming final volumes with contribution from group {}".format(j))
#     for ts in tqdm(range(ntimesteps)):
#         if ts in retained_timesteps:
#             volumes_corrected_final[ts]+=volumes_corrected[ts_indices[j]]*traj_retained_final[ts_indices[j]].shape[0]
#             #count[ts]+=1
#             count[ts] += traj_retained_final[ts_indices[j]].shape[0]
#             ts_indices[j] += 1
#             #total_weight[ts]+=traj_retained_final[ts_indices[j]].shape[0]
#

# del volumes_corrected
# del radial_traj_3D_corrected
# del dico_homographies
# del kdata_retained_final_list

# volumes_corrected_final=np.zeros((ntimesteps,nb_slices,int(npoint/2),int(npoint/2)),dtype=dico_volumes_corrected[0].dtype)
# ts_indices=np.zeros(len(dico_kdata_retained_registration.keys())).astype(int)
# count=np.zeros(ntimesteps).astype(int)


# for ts in tqdm(range(ntimesteps)):
#     count=0
#     for j in dico_kdata_retained_registration.keys():
#         if ts in dico_kdata_retained_registration[j]:
#             volumes_corrected_final[ts]+=dico_volumes_corrected[j][ts_indices[j]]
#             ts_indices[j]+=1
#             count+=1
#     volumes_corrected_final[ts]/=count

# for j in tqdm(dico_kdata_retained_registration.keys()):
#     for ts in tqdm(range(ntimesteps)):
#         if ts in dico_kdata_retained_registration[j]:
#             volumes_corrected_final[ts]+=dico_volumes_corrected[j][ts_indices[j]]
#             ts_indices[j]+=1
#             count[ts]+=1
#     del dico_kdata_retained_registration[j]
#     del dico_volumes_corrected[j]
#
# count=np.expand_dims(count,axis=tuple(range(1,volumes_corrected_final.ndim)))
# volumes_corrected_final/=count
#
# np.save(filename_volume_corrected_final,volumes_corrected_final)
#
# animate_images(volumes_corrected_final[:,int(nb_slices/2),:,:])

#volumes for slice taking into account coil sensi
print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
    np.save(filename_volume,volumes_all)
    # sl=20
    # ani = animate_images(volumes_all[:,sl,:,:])
    del volumes_all

print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    selected_spokes = np.r_[10:400]
    selected_spokes=None
    mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/25, normalize_kdata=True,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask,mask)
    animate_images(mask)
    del mask

#animate_images(np.abs(volumes_all[:,int(nb_slices/2),:,:]))

# #Check modulation of nav signal by MRF
# plt.figure()
# rep=0
# signal_MRF = np.abs(volumes_all[:,int(nb_slices/2),int(npoint/4),int(npoint/4)])
# signal_MRF = signal_MRF/np.max(signal_MRF)
# signal_nav =image_nav[rep,:,int(npoint/4)]
# signal_nav = signal_nav/np.max(signal_nav)
# plt.plot(signal_MRF,label="MRF signal at centre pixel")
# plt.scatter(x=(nav_timesteps/8).astype(int),y=signal_nav,c="r",label="Nav image at centre pixel for rep {}".format(rep))
# rep=4
# plt.scatter(x=(nav_timesteps/8).astype(int),y=signal_nav,c="g",label="Nav image at centre pixel for rep {}".format(rep))
# plt.legend()

##MASK





del kdata_all_channels_all_slices
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


mask = np.load(filename_mask)
mask=dico_mask[index_ref]
volumes_all = np.load(filename_volume)
volumes_corrected_final=np.load(filename_volume_corrected_final)
#
# filename_volume_corrected_disp5= "./data/InVivo/3D/20220113_CS/meas_MID00163_FID49558_raFin_3D_tra_1x1x5mm_FULL_50GS_read_volumes_corrected_disp5.npy"
# volumes_corrected_disp5 = np.load(filename_volume_corrected_disp5)
#
# center_z = int(nb_slices/2)
# center_pixel=int(image_size[1]/2)
#
# dplane = image_size[1]/8
# dz = image_size[0]/8
#
# x = int(np.random.normal(center_pixel,dplane))
# y = int(np.random.normal(center_pixel,dplane))
# z = int(np.random.normal(center_z,dz))
#
#
#
# plt.close("all")
# metric=np.real
# plt.figure()
# plt.plot(metric(volumes_all[:,z,x,y]),label="All spokes")
# plt.plot(metric(volumes_corrected_final[:,z,x,y]),label="Corrected spokes")
# plt.plot(metric(volumes_corrected_disp5[:,z,x,y]),label="Only one cycle")
# plt.title("Volume series comparison {}".format((z,x,y)))
# plt.legend()
#



#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])

suffix="_corrected"
ntimesteps=175
if not(load_map):
    niter = 0
    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=None,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps)
    all_maps=optimizer.search_patterns_test(dictfile,volumes_corrected_final,retained_timesteps=None)

    if(save_map):
        import pickle

        file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format(suffix)
        #file_map = filename.split(".dat")[0] + "_corrected_dens_adj{}_MRF_map.pkl".format(suffix)
        #file_map = filename.split(".dat")[0] + "_5iter_MRF_map.pkl".format("")
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


curr_file=file_map
file = open(curr_file, "rb")
all_maps = pickle.load(file)
file.close()
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






























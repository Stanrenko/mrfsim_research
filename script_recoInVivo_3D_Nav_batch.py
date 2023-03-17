
#import matplotlib
#matplotlib.u<se("TkAgg")
import pandas as pd
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
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
base_folder = "./data/InVivo/3D"


files=["/patient.003.v1/meas_MID00125_FID02111_raFin_3D_tra_1x1x5mm_FULL_1.dat",
"/patient.003.v1/meas_MID00129_FID02115_raFin_3D_tra_1x1x5mm_FULL_2.dat",
"/patient.003.v1/meas_MID00127_FID02113_raFin_3D_tra_1x1x5mm_FULL_3.dat",
"/patient.003.v1/meas_MID00128_FID02114_raFin_3D_tra_1x1x5mm_FULL_4.dat"]

# files=["/patient.004.v1/meas_MID00022_FID03367_raFin_3D_tra_1x1x5mm_FULL_1.dat",
# "/patient.004.v1/meas_MID00023_FID03368_raFin_3D_tra_1x1x5mm_FULL_2.dat",
# "/patient.004.v1/meas_MID00024_FID03369_raFin_3D_tra_1x1x5mm_FULL_3.dat",
# "/patient.004.v1/meas_MID00025_FID03370_raFin_3D_tra_1x1x5mm_FULL_4.dat"]

folder = base_folder+"/".join(str.split(files[0],"/")[:-1])


df_groups_global=pd.DataFrame()
density_adj_radial=True
use_GPU = True
light_memory_usage=True
ntimesteps=175
undersampling_factor=1
window=8
categories_global=[]

bin_width = 3


filename_kdata_final = base_folder + str.split(files[0],"_1.dat")[0]+"_bw{}_aggregated_kdata.npy".format(bin_width)
filename_categories_global = folder +"/categories_global_bw{}.npy".format(bin_width)
filename_df_groups_global= folder+"/df_groups_global_bw{}.pkl".format(bin_width)

filename_kdata_final_corrected = str.split(filename_kdata_final,".npy")[0]+"_corrected.npy"
filename_b1=str.split(filename_kdata_final,"kdata.npy")[0]+"b1.npy"

filename_b1_corrected = str.split(filename_kdata_final,"kdata.npy")[0]+"b1_corrected.npy"
filename_volumes_corrected = str.split(filename_kdata_final,"kdata.npy")[0]+"volumes_corrected.npy"
filename_mask_corrected = str.split(filename_kdata_final,"kdata.npy")[0]+"mask_corrected.npy"



#kdata_all_channels_all_slices=np.load(filename_kdata_final)


for localfile in files:

    filename = base_folder+localfile

    #filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
    #filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
    #filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

    filename_save=str.split(filename,".dat") [0]+".npy"
    #filename_nav_save=str.split(base_folder+"/phantom.001.v1/phantom.001.v1.dat",".dat") [0]+"_nav.npy"
    filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"

    folder = "/".join(str.split(filename,"/")[:-1])



    suffix=""

    filename_b1 = str.split(filename,".dat") [0]+"_b1{}.npy".format("")
    filename_kdata = str.split(filename, ".dat")[0] + "_kdata{}.npy".format("")

    filename_volume = str.split(filename,".dat") [0]+"_volumes{}.npy".format(suffix)

    filename_mask= str.split(filename,".dat") [0]+"_mask{}.npy".format("")
    filename_oop=str.split(filename,".dat") [0]+"_volumes_oop{}.npy".format(suffix)
    filename_oop_corrected=str.split(filename,".dat") [0]+"_volumes_oop_corrected{}.npy".format(suffix)


    #filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"


    #Parsed_File = rT.map_VBVD(filename)
    #idx_ok = rT.detect_TwixImg(Parsed_File)
    #RawData = Parsed_File[str(idx_ok)]["image"].readImage()

    dico_seqParams=build_dico_seqParams(filename,folder)


    use_navigator_dll=dico_seqParams["use_navigator_dll"]

    if use_navigator_dll:
        meas_sampling_mode=dico_seqParams["alFree"][14]
        nb_gating_spokes = dico_seqParams["alFree"][6]
    else:
        meas_sampling_mode = dico_seqParams["alFree"][12]
        nb_gating_spokes = 0

    if nb_gating_spokes>0:
        meas_orientation =  dico_seqParams["alFree"][11]
        if meas_orientation==1:
            nav_direction = "READ"
        elif meas_orientation==2:
            nav_direction = "PHASE"
        elif meas_orientation==3:
            nav_direction = "SLICE"

    nb_segments = dico_seqParams["alFree"][4]

    x_FOV = dico_seqParams["x_FOV"]
    y_FOV = dico_seqParams["y_FOV"]
    z_FOV = dico_seqParams["z_FOV"]
    nb_part = dico_seqParams["nb_part"]

    del dico_seqParams

    if meas_sampling_mode==1:
        incoherent=False
        mode = None
    elif meas_sampling_mode==2:
        incoherent = True
        mode = "old"
    elif meas_sampling_mode==3:
        incoherent = True
        mode = "new"

    data,data_for_nav=build_data(filename,folder,nb_segments,nb_gating_spokes)


    data_shape = data.shape

    nb_channels=data_shape[0]
    nb_allspokes = data_shape[-3]
    npoint = data_shape[-1]


    nb_slices = data_shape[-2]
    image_size = (nb_slices, int(npoint/2), int(npoint/2))


    npoint_nav=data_for_nav.shape[-1]



    dx = x_FOV/(npoint/2)
    dy = y_FOV/(npoint/2)
    dz = z_FOV/nb_slices

    if str.split(filename_kdata,"/")[-1] in os.listdir(folder):
        del data


    if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
        # Density adjustment all slices

        if density_adj_radial:
            density = np.abs(np.linspace(-1, 1, npoint))
            density = np.expand_dims(density,tuple(range(data.ndim-1)))
        else:
            density=1
        print("Performing Density Adjustment....")
        data *= density
        np.save(filename_kdata, data)
        del data

        kdata_all_channels_all_slices = np.load(filename_kdata)

    else:
        kdata_all_channels_all_slices = np.load(filename_kdata)

    kdata_shape=kdata_all_channels_all_slices.shape

    print("Calculating Coil Sensitivity....")

    radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
    #radial_traj.adjust_traj_for_window(window)

    nb_segments=radial_traj.get_traj().shape[0]

    if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
        res = 16
        b1_all_slices=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage)
        np.save(filename_b1,b1_all_slices)
    else:
        b1_all_slices=np.load(filename_b1)

    if nb_gating_spokes>0:
        print("Processing Nav Data...")
        data_for_nav=np.load(filename_nav_save)

        nb_allspokes=nb_segments
        nb_slices=data_for_nav.shape[1]
        nb_channels=data_for_nav.shape[0]
        npoint_nav=data_for_nav.shape[-1]

        all_timesteps = np.arange(nb_allspokes)
        nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

        nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                               applied_timesteps=list(nav_timesteps))

        nav_image_size = (int(npoint_nav / 2),)

        print("Calculating Sensitivity Maps for Nav Images...")
        #b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
        #b1_nav_mean = np.mean(b1_nav, axis=(1, 2))


        ch=2
        image_nav_ch =simulate_nav_images_multi(np.expand_dims(data_for_nav[ch],axis=0),nav_traj, nav_image_size)
        #plt.imshow(np.abs(b1_nav[ch].reshape(-1, int(npoint/2))))


        plt.figure()
        plt.plot(np.abs(image_nav_ch.reshape(-1, int(npoint/2)))[10,:])


        print("Estimating Movement...")
        shifts = list(range(-20, 20))
        bottom = 20
        top = 80
        displacements = calculate_displacement(image_nav_ch, bottom, top, shifts,0.001)

        plt.figure()
        plt.plot(displacements)

        displacement_for_binning = displacements

        max_bin = np.max(displacement_for_binning)
        min_bin = -20

        bins = np.arange(min_bin, max_bin + bin_width, bin_width)
        # print(bins)
        categories = np.digitize(displacement_for_binning, bins)
        df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
        df_groups = df_cat.groupby("cat").count()
        curr_max = df_groups.displacement.max()

        if df_groups_global.empty:
            df_groups_global=df_groups
        else:
            df_groups_global+=df_groups

        categories_global.append(categories)


#################################################################################################################################"

categories_global=np.array(categories_global)


np.save(filename_categories_global,categories_global)
df_groups_global.to_pickle(filename_df_groups_global)

categories_global=np.load(filename_categories_global)
df_groups_global=pd.read_pickle(filename_df_groups_global)

idx_cat = df_groups_global.displacement.idxmax()

nb_segments=kdata_all_channels_all_slices.shape[1]
nb_slices=kdata_all_channels_all_slices.shape[2]
nb_channels=kdata_all_channels_all_slices.shape[0]
npoint=kdata_all_channels_all_slices.shape[-1]
nb_allspokes=nb_segments

nb_part=nb_slices
nb_gating_spokes=50

kdata_final=np.zeros(kdata_all_channels_all_slices.shape,dtype=kdata_all_channels_all_slices.dtype)
count = np.zeros((nb_segments,nb_slices))

for i,localfile in tqdm(enumerate(files)):

    filename = base_folder + localfile

    retained_nav_spokes = (categories_global[i] == idx_cat)

    retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
    spoke_groups = np.argmin(np.abs(np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,nb_segments / nb_gating_spokes).reshape(1,-1)),axis=-1)

    if not (nb_segments == nb_gating_spokes):
        spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
        spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
            nb_segments / nb_gating_spokes / 2) + 1:] - 1
        spoke_groups = spoke_groups.flatten()

    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])


    included_spokes[::int(nb_segments/nb_gating_spokes)]=False
    included_spokes=included_spokes.reshape(nb_slices,nb_segments)
    included_spokes = included_spokes.T

    filename_kdata = str.split(filename, ".dat")[0] + "_kdata{}.npy".format("")

    kdata_all_channels_all_slices = np.load(filename_kdata)
    kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(nb_channels,nb_segments,nb_slices,npoint)

    kdata_final+=(np.expand_dims(1*included_spokes,axis=(0,-1)))*kdata_all_channels_all_slices
    count += (1*included_spokes)

included_spokes_global=count>0
count[count==0]=1
kdata_final/=np.expand_dims(count,axis=(0,-1))

np.save(filename_kdata_final,kdata_final)



kdata_final=np.load(filename_kdata_final)

print("Correcting KData for missing spokes...")
incoherent=True
mode="old"
radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                       nb_slices=nb_slices, incoherent=incoherent, mode=mode)

included_spokes_global=included_spokes_global.T.flatten()


kdata_retained_final_list = []
for i in tqdm(range(nb_channels)):
    kdata_retained_final, traj_retained_final, retained_timesteps = correct_mvt_kdata(kdata_final[i].reshape(nb_segments, -1), radial_traj, included_spokes_global, ntimesteps, density_adj=True,log=False)
    kdata_retained_final_list.append(kdata_retained_final)



radial_traj_3D_corrected=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
radial_traj_3D_corrected.traj_for_reconstruction=traj_retained_final

image_size=(nb_slices,int(npoint/2),int(npoint/2))

#if str.split(filename_b1_corrected,"/")[-1] not in os.listdir(folder):
#    res = 16
#    b1_all_slices_corrected=calculate_sensitivity_map_3D(np.array(kdata_retained_final_list),radial_traj_3D_corrected,res,image_size,useGPU=False,light_memory_usage=True)
#    np.save(filename_b1_corrected,b1_all_slices_corrected)
#else:
#    b1_all_slices_corrected=np.load(filename_b1_corrected)


if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices=calculate_sensitivity_map_3D(kdata_final,radial_traj,res,image_size,useGPU=False,light_memory_usage=True)
    np.save(filename_b1,b1_all_slices)
else:
    b1_all_slices=np.load(filename_b1)


if str.split(filename_volumes_corrected,"/")[-1] not in os.listdir(folder):
    volumes_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list,radial_traj_3D_corrected,image_size,b1=b1_all_slices,ntimesteps=len(retained_timesteps),density_adj=False,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,is_theta_z_adjusted=True,normalize_volumes=True)
    animate_images(volumes_corrected[:,int(nb_slices/2),:,:])
    np.save(filename_volumes_corrected,volumes_corrected)
else:
    volumes_corrected=np.load(filename_volumes_corrected)



res = 16
b1_all_slices_final=calculate_sensitivity_map_3D(kdata_final,radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage)
sl=int(b1_all_slices_final.shape[1]/2)
list_images = list(np.abs(b1_all_slices_final[:,sl,:,:]))
plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))

del kdata_final
kdata_final=np.load(filename_kdata_final)

radial_traj_anatomy=Radial3D(total_nspokes=400,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
radial_traj_anatomy.traj = radial_traj.get_traj()[800:1200]
volume_outofphase_final=simulate_radial_undersampled_images_multi(kdata_final[:,800:1200,:,:],radial_traj_anatomy,image_size,b1=b1_all_slices_final,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=True,memmap_file=None,light_memory_usage=True)
animate_images(volume_outofphase_final[0],cmap="gray")

from mutools import io
file_mha = filename.split(".dat")[0] + "_volume_out_of_phase_final.mha"
io.write(file_mha,np.abs(volume_outofphase_final[0]),tags={"spacing":[dz,dx,dy]})

volume_outofphase=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices[:,800:1200,:,:],radial_traj_anatomy,image_size,b1=b1_all_slices_final,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=True,memmap_file=None,light_memory_usage=True)
animate_images(volume_outofphase[0],cmap="gray")

from mutools import io
file_mha = filename.split(".dat")[0] + "_volume_out_of_phase.mha"
io.write(file_mha,np.abs(volume_outofphase[0]),tags={"spacing":[dz,dx,dy]})


included_spokes_global=np.array(included_spokes_global)
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
            kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, ntimesteps, density_adj=False,log=False)
        kdata_retained_final_list.append(kdata_retained_final)

#
# i=np.random.choice(ntimesteps)
# #i=6 #argmax
# #i=38 #argmin
# curr_traj=traj_retained_final[i]
# dk = kdata_retained_final_list[0][i]
#
# print(dk.shape[0]/(nb_part*nb_segments/ntimesteps*npoint))
#
# Pt=convolution_kernel_radial_single_channel(curr_traj,dk,npoint,image_size)
#
# #animate_images(Pt)
#
# kx=-np.pi+np.arange(npoint)*2*np.pi/(npoint-1)
# ky=-np.pi+np.arange(npoint)*2*np.pi/(npoint-1)
# kz=-np.pi+np.arange(nb_slices)*2*np.pi/(nb_slices-1)
#
# KX,KY,KZ=np.meshgrid(kx,ky,kz)
#
# traj_full=np.stack((KX,KY,KZ),axis=-1)
# traj_full=traj_full.reshape(-1,3)
# traj_full=traj_full.astype("float32")
#
# F_Pt=finufft.nufft3d2(traj_full[:, 2],traj_full[:, 0], traj_full[:, 1], Pt)
#
# F_Pt=F_Pt.reshape((nb_slices,npoint,npoint))
# #
# sl=int(nb_slices/2)
# pow=3
#
# plt.figure()
# plt.imshow(np.abs(1-F_Pt[sl]**pow))
# plt.colorbar()
#
# plt.figure()
# plt.plot(np.abs(1-F_Pt[sl,int(npoint/2),:]**pow))
#
# print(np.linalg.norm(1-F_Pt**pow))
#
# animate_images(1-F_Pt**pow)


# plt.plot(displacements)
#
# #del kdata_all_channels_all_slices
#
ch=0
plt.figure()
reduction_factors=[]
for el in kdata_retained_final_list[ch]:
   reduction_factors.append(el.shape[0]/(nb_part*nb_segments/ntimesteps*npoint))
plt.plot(reduction_factors)
np.argmax(np.array(reduction_factors))

print("Rebuilding Images With Corrected volumes...")

radial_traj_3D_corrected=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
radial_traj_3D_corrected.traj_for_reconstruction=traj_retained_final

b1_full = np.ones(image_size)
b1_full=np.expand_dims(b1_full,axis=0)

if str.split(filename_volume_corrected,"/")[-1] not in os.listdir(folder):
    volumes_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list,radial_traj_3D_corrected,image_size,b1=b1_all_slices,ntimesteps=len(retained_timesteps),density_adj=False,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,is_theta_z_adjusted=True,normalize_volumes=True)
    animate_images(volumes_corrected[:,int(nb_slices/2),:,:])
    np.save(filename_volume_corrected,volumes_corrected)
else:
    volumes_corrected=np.load(filename_volume_corrected)
# #
# norm_base=np.linalg.norm(base_images)
#
# sl=int(nb_slices/2)
# ts = np.random.choice(volumes_corrected.shape[0])
# norm_ts = np.linalg.norm(volumes_corrected[ts,:,:,:])
# plt.figure()
# plt.imshow(np.abs(volumes_corrected[ts,sl,:,:]))
# plt.title("timestep {} norm {} base_norm {} ".format(ts,np.round(norm_ts,1),np.round(norm_base,1)))
# plt.colorbar()
#
# all_pixels_norm=np.linalg.norm(volumes_corrected,axis=0)
# max_ts_pixel=np.argmax(np.abs(volumes_corrected),axis=0)
# max_pixel=np.argmax(np.abs(volumes_corrected))
# max_pixel_unrav=np.unravel_index(max_pixel,volumes_corrected.shape)
#
# plt.plot(np.abs(volumes_corrected[:,max_pixel_unrav[1],max_pixel_unrav[2],max_pixel_unrav[3]]))
#
# norm_ratio=all_pixels_norm/norm_base
#
# np.linalg.norm(volumes_corrected[ts,:,:,:])

# animate_images(volumes_corrected[:,sl,:,:])

# if nav_direction=="SLICE":
#     coil_sensitivity_nav = np.sum(b1_all_slices,axis=(-2,-1))
# elif nav_direction=="PHASE":
#     coil_sensitivity_nav = np.sum(b1_all_slices, axis=(0, -2))
# elif nav_direction == "READ":
#     coil_sensitivity_nav = np.sum(b1_all_slices, axis=(0, -1))
#
# coil_sensitivity_nav /= np.linalg.norm(coil_sensitivity_nav, axis=0)
# coil_sensitivity_nav /= np.max(np.abs(coil_sensitivity_nav.flatten()))
#
# sl=int(b1_all_slices.shape[1]/2)
# list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
# plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))

#
# volume_rebuilt = build_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,density_adj=False,eps=1e-6,b1=b1_all_slices,useGPU=True,normalize_kdata=True,light_memory_usage=True,is_theta_z_adjusted=False)
# np.save(str.split(filename,".dat") [0]+"_volume_allspokes.npy",volume_rebuilt)
#
# from mutools import io
# file_mha = filename.split(".dat")[0] + "_volume_allspokes.mha"
# io.write(file_mha,np.abs(volume_rebuilt),tags={"spacing":[dz,dx,dy]})
# animate_images(volume_rebuilt,cmap="gray")
#
#
#
#
# #build out of phase spokes image
# if str.split(filename_oop,"/")[-1] not in os.listdir(folder):
#     radial_traj_anatomy=Radial3D(total_nspokes=400,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
#     radial_traj_anatomy.traj = radial_traj.get_traj()[800:1200]
#     volume_outofphase=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices[:,800:1200,:,:],radial_traj_anatomy,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=True,memmap_file=None,light_memory_usage=True)
#     np.save(filename_oop, volume_outofphase)
# else:
#     volume_outofphase=np.load(filename_oop)
#
# animate_images(volume_outofphase[0],cmap="gray")

#build out of phase spokes image
#
# volume=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_full,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,normalize_volumes=False)
# # animate_images(volume[0],cmap="gray")
# #
# #
# #
#
#
# kdata_retained_final_list_volume = []
# for i in tqdm(range(nb_channels)):
#     kdata_retained_final, traj_retained_final_volume, _ = correct_mvt_kdata(
#             kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, 1, density_adj=True,log=False)
#     kdata_retained_final_list_volume.append(kdata_retained_final)
#
# radial_traj_3D_corrected_single_volume=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
# radial_traj_3D_corrected_single_volume.traj_for_reconstruction=traj_retained_final_volume
#
#
# volume_corrected=simulate_radial_undersampled_images_multi(kdata_retained_final_list_volume,radial_traj_3D_corrected_single_volume,image_size,b1=b1_full,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,normalize_volumes=False,is_theta_z_adjusted=True)
# animate_images(volume_corrected[0],cmap="gray")
# plt.figure()
# plt.imshow(np.abs(volume_corrected[0])[15,:,:])
# plt.colorbar()
#
# plt.figure()
# plt.imshow(np.abs(volume[0])[8,:,:])
# plt.colorbar()
#
#
# np.linalg.norm(volume_corrected[0])
# np.linalg.norm(base_images)
# np.linalg.norm(volume[0])
#
# animate_multiple_images(volume[0],volume_corrected[0])


# volume_oop_2=np.load("./data/InVivo/3D/20220113_CS/meas_MID00163_FID49558_raFin_3D_tra_1x1x5mm_FULL_50GS_read_volumes_oop.npy")
# animate_multiple_images(volume_outofphase[0],volume_oop_2[0],cmap="gray")
#
# from PIL import Image
# gif=[]
# volume_for_gif = np.abs(volume_corrected[0])
# for i in range(volume_for_gif.shape[0]):
#     img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
#     img=img.convert("P")
#     gif.append(img)
#
# img.show()
#
# filename_gif = str.split(filename,".dat") [0]+"_volume_corrected.gif"
# gif[0].save(filename_gif,
#                save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)


# # list_images = list(np.abs(volume_outofphase)[0][:])
# # plot_image_grid(list_images,(8,8),title="Anatomic Image Out Of Phase Spokes",cmap="gray")
# #
# #
# # path = r"/Users/constantinslioussarenko/PythonGitRepositories"
# # sys.path.append(path+"/epgpy")
# # sys.path.append(path+"/machines")
# # sys.path.append(path+"/mutools")
# # sys.path.append(path+"/dicomstack")
# #
#
#from mutools import io
#file_mha = filename.split(".dat")[0] + "_volumesoutofphase.mha"
#io.write(file_mha,np.abs(volume_outofphase)[0],tags={"spacing":[dz,dx,dy]})
#




# volumes_all_spokes=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1)
# sl=10
# plt.figure()
# plt.title("Approximation : rebuilt image all data")
# plt.imshow(np.abs(np.squeeze(volumes_all_spokes)[sl,:,:]),cmap="gray")
#
# animate_images((np.squeeze(volumes_all_spokes)),interval=1000)

##volumes for slice taking into account coil sensi
# print("Building Volumes....")
# if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
#     volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=True,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
#     np.save(filename_volume,volumes_all)
#     # sl=20
#     # ani = animate_images(volumes_all[:,sl,:,:])
#     del volumes_all
#
# print("Building Mask....")
# if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
#     selected_spokes = np.r_[10:400]
#     selected_spokes=None
#     mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/25, normalize_kdata=True,light_memory_usage=True,selected_spokes=selected_spokes)
#     np.save(filename_mask,mask)
#     animate_images(mask)
#     del mask

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

mask = np.load(filename_mask)
#volumes_all = np.load(filename_volume)
volumes_corrected=np.load(filename_volume_corrected)


#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])



if not(load_map):
    niter = 0
    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=radial_traj,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=False,useGPU_simulation=False,gen_mode="other",movement_correction=True,cond=included_spokes,ntimesteps=ntimesteps)
    all_maps=optimizer.search_patterns_test(dictfile,volumes_corrected,retained_timesteps=retained_timesteps)

    if(save_map):
        import pickle

        #file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format(suffix)
        #file_map = filename.split(".dat")[0] + "_corrected_dens_adj{}_MRF_map.pkl".format(suffix)
        file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format(suffix)
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

from mutools import io
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


map_rebuilt=all_maps[0][0]
keys_simu = list(map_rebuilt.keys())
values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
map_for_sim = dict(zip(keys_simu, values_simu))

images_pred = MapFromDict3D("RebuiltMapFromParams", paramMap=map_for_sim, rounding=True,gen_mode="other")
images_pred.buildParamMap()

images_pred.build_ref_images(seq)

kdatai = images_pred.generate_kdata(radial_traj,useGPU=False)

kdatai_retained, traji_retained, retained_timesteps = correct_mvt_kdata(
            kdatai, radial_traj, included_spokes, ntimesteps, density_adj=True)


def compare_signals(dictfile,volumes,maps,mask,volumes_1,maps_1,mask_1=None,pixel=None,pixel_1=None,figsize=(10,15)):
    if pixel is None:
        raise ValueError("pixel should be a tuple")
    if pixel_1 is None:
        pixel_1=pixel
    if mask_1 is None:
        mask_1=mask

    signal=volumes[:,pixel[0],pixel[1],pixel[2]]
    signal_1=volumes_1[:,pixel_1[0],pixel_1[1],pixel_1[2]]

    signal=signal/np.linalg.norm(signal)
    signal_1=signal_1/np.linalg.norm(signal_1)

    for k in maps.keys():
        maps_retrieved_volume[k] = makevol(maps[k], mask > 0)[pixel[0],pixel[1],pixel[2]]

    for k in maps.keys():
        maps_retrieved_volume_1[k] = makevol(maps_1[k], mask_1 > 0)[pixel_1[0], pixel_1[1], pixel_1[2]]

    params = list(maps_retrieved_volume.values())[:-1]
    ff = maps_retrieved_volume["ff"]

    params_1 = list(maps_retrieved_volume_1.values())[:-1]
    ff_1 = maps_retrieved_volume_1["ff"]

    mrfdict = dictsearch.Dictionary()
    mrfdict.load(dictfile, force=True)

    mapped_signal = mrfdict[tuple(params)][:, 0] * (1 - ff) + mrfdict[tuple(
        params)][:, 1] * (ff)
    mapped_signal_1 = mrfdict[tuple(params_1)][:, 0] * (1 - ff_1) + mrfdict[tuple(
        params_1)][:, 1] * (ff_1)

    mapped_signal=mapped_signal/np.linalg.norm(mapped_signal)
    mapped_signal_1 = mapped_signal_1 / np.linalg.norm(mapped_signal_1)

    plt.figure(figsize=figsize)
    metric=np.real
    plt.title("Real Part {}".format(pixel))
    plt.plot(metric(signal),label="Original Signal")
    plt.plot(metric(signal_1), label="Original Signal 1")
    plt.plot(metric(mapped_signal), label="Mapped Signal {}".format(maps_retrieved_volume))
    plt.plot(metric(mapped_signal_1), label="Mapped Signal 1 {}".format(maps_retrieved_volume_1))

    plt.figure(figsize=figsize)
    metric = np.imag
    plt.title("Imaginary Part")
    plt.plot(metric(signal), label="Original Signal")
    plt.plot(metric(signal_1), label="Original Signal 1")
    plt.plot(metric(mapped_signal), label="Mapped Signal {}".format(maps_retrieved_volume))
    plt.plot(metric(mapped_signal_1), label="Mapped Signal 1 {}".format(maps_retrieved_volume_1))

    plt.figure(figsize=figsize)
    metric = np.abs
    plt.title("Module")
    plt.plot(metric(signal), label="Original Signal")
    plt.plot(metric(signal_1), label="Original Signal 1")
    plt.plot(metric(mapped_signal), label="Mapped Signal {}".format(maps_retrieved_volume))
    plt.plot(metric(mapped_signal_1), label="Mapped Signal 1 {}".format(maps_retrieved_volume_1))

iter=0
map_rebuilt=all_maps[iter][0]
mask=all_maps[iter][1]

keys_simu = list(map_rebuilt.keys())
values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
map_for_sim = dict(zip(keys_simu, values_simu))

map_Python = MapFromDict("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
map_Python.buildParamMap()

map_Python.plotParamMap()
map_Python.plotParamMap("ff")
map_Python.plotParamMap("df")
map_Python.plotParamMap("attB1")
map_Python.plotParamMap("wT1")

map_Python.build_ref_images(seq=seq)
rebuilt_image_series = map_Python.images_series
rebuilt_image_series= [np.mean(gp, axis=0) for gp in groupby(rebuilt_image_series, 8)]
rebuilt_image_series=np.array(rebuilt_image_series)


#ani=animate_images(rebuilt_image_series)
load_volume=True
if load_volume:
    volumes_all=np.load(filename.split(".dat")[0] + "_volumes.npy")

plt.figure()
pixel=(103,76)
metric=np.angle
signal_orig = metric(volumes_all[:,pixel[0],pixel[1]])
signal_orig=signal_orig/np.std(signal_orig)
signal_rebuilt = metric(rebuilt_image_series[:,pixel[0],pixel[1]])
signal_rebuilt=signal_rebuilt/np.std(signal_rebuilt)
plt.plot(signal_orig,label="Original")
plt.plot(signal_rebuilt,label="Python")
plt.legend()

######################################################################################################################################################
#Comp volume Matlab vs Python
import numpy as np
volumes_python=np.load(filename.split(".dat")[0] + "_volumes.npy")

import h5py
folder_matlab="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/3_Comp_Matlab/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI/Reco8"

f = h5py.File(folder_matlab+"/ImgSeries.mat","r")
volumes_matlab=np.array(f.get("Img"))

slice=2
volumes_matlab_slice=volumes_matlab[slice,:,:,:].view("complex")
volumes_matlab_slice=np.rot90(volumes_matlab_slice,axes=(2,1))

plt.figure()
plt.imshow(np.abs(volumes_python[25,:,:]))
plt.title("Python volume")

plt.figure()
plt.imshow(np.abs(volumes_matlab_slice[25,:,:]))
plt.title("Matlab volume")



plt.close("all")
metric=np.imag
error_volumes=np.linalg.norm(metric(volumes_matlab_slice-volumes_python),axis=0)
max_index_error = np.unravel_index(np.argmax(error_volumes),(256,256))
plt.figure()
plt.imshow(error_volumes)
plt.colorbar()



pixel=(75,75)
pixel=max_index_error

plt.figure()

signal_orig = metric(volumes_python[:,pixel[0],pixel[1]])
signal_orig=signal_orig/np.std(signal_orig)
signal_rebuilt = metric(volumes_matlab_slice[:,pixel[0],pixel[1]])
signal_rebuilt=signal_rebuilt/np.std(signal_rebuilt)
plt.plot(signal_orig,label="Python signal")
plt.plot(signal_rebuilt,label="Matlab signal")
plt.title("Max error for rebuilt images series on pixel {}".format(pixel))
plt.legend()


#######################################################################################################################################################
#Comp matlab vs Python
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import pickle


image_size=(256,256)

filename="./data/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI.dat"
file_map = filename.split(".dat")[0]+"_MRF_map_3.pkl"
file = open( file_map, "rb" )
all_maps=pickle.load(file)

filename="./data/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI.dat"
file_map = filename.split(".dat")[0]+"_MRF_map_matlab_volumes_3.pkl"
file = open( file_map, "rb" )
all_maps_matlab_volumes=pickle.load(file)

slice=2


#Matlab
file_matlab = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/3_Comp_Matlab/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI/Reco8/MRFmaps0.mat"
map_Matlab=MapFromFile("MapRebuiltMatlab",image_size=(5,448,224),file=file_matlab,rounding=False,file_type="Result")
map_Matlab.buildParamMap()

#matobj = loadmat(map_Matlab.paramDict["file"])["MRFmaps"]
#map_wT1 = matobj["T1water_map"][0][0]

#map_Matlab.plotParamMap("ff",sl=slice)

all_maps_matlab_current_slice={}
all_maps_matlab_current_slice[0]={}
all_maps_matlab_current_slice[1]=map_Matlab.mask[slice,:,:]

for k in map_Matlab.paramMap.keys():
    current_volume = makevol(map_Matlab.paramMap[k],map_Matlab.mask>0)[slice,:,:]
    all_maps_matlab_current_slice[0][k]=current_volume[all_maps_matlab_current_slice[1]>0]

maps_python_current_slice=all_maps[0][0]
mask_python_current_slice=all_maps[0][1]
mask_python_current_slice=np.flip(np.rot90(mask_python_current_slice),axis=1)

for k in maps_python_current_slice.keys():
    current_volume = makevol(maps_python_current_slice[k],all_maps[0][1]>0)
    current_volume = np.flip(np.rot90(current_volume),axis=1)
    maps_python_current_slice[k]=current_volume[mask_python_current_slice>0]

all_maps_python_current_slice=(maps_python_current_slice,mask_python_current_slice)

maps_python_matlab_volumes_current_slice=all_maps_matlab_volumes[0][0]
mask_python_matlab_volumes_current_slice=all_maps_matlab_volumes[0][1]
mask_python_matlab_volumes_current_slice=np.flip(np.rot90(mask_python_matlab_volumes_current_slice),axis=1)

for k in maps_python_matlab_volumes_current_slice.keys():
    current_volume = makevol(maps_python_matlab_volumes_current_slice[k],all_maps_matlab_volumes[0][1]>0)
    current_volume = np.flip(np.rot90(current_volume),axis=1)
    maps_python_matlab_volumes_current_slice[k]=current_volume[mask_python_matlab_volumes_current_slice>0]


all_maps_python_matlab_volumes_current_slice=(maps_python_matlab_volumes_current_slice,mask_python_matlab_volumes_current_slice)

####################################################################################################
map_df_python =  makevol(maps_python_current_slice["df"],mask_python_current_slice>0)
map_df_matlab = makevol(all_maps_matlab_current_slice[0]["df"],all_maps_matlab_current_slice[1]>0)

map_df_python=np.rot90(np.flip(map_df_python,axis=0))
map_df_matlab=np.rot90(np.flip(map_df_matlab,axis=0))

map_df_python_matlab_volumes=np.rot90(np.flip(map_df_python_matlab_volumes,axis=0))
map_df_python_matlab_volumes =  makevol(maps_python_matlab_volumes_current_slice["df"],mask_python_matlab_volumes_current_slice>0)

error_df = map_df_python-map_df_matlab
max_diff_df = np.unravel_index(np.argmax(error_df),image_size)
plt.figure()
plt.imshow(error_df)
plt.colorbar()

#############################################################################################################################""
compare_paramMaps(all_maps_matlab_current_slice[0],all_maps_python_current_slice[0],all_maps_matlab_current_slice[1]>0,all_maps_python_current_slice[1]>0,title1="Matlab",title2="Python",proj_on_mask1=True,adj_wT1=True,save=True)

compare_paramMaps(all_maps_matlab_current_slice[0],all_maps_python_matlab_volumes_current_slice[0],all_maps_matlab_current_slice[1]>0,all_maps_python_matlab_volumes_current_slice[1]>0,title1="Matlab",title2="Python Matlab volumes",proj_on_mask1=True,adj_wT1=True,save=True)

compare_paramMaps(all_maps_python_matlab_volumes_current_slice[0],all_maps_python_current_slice[0],all_maps_python_matlab_volumes_current_slice[1]>0,all_maps_python_current_slice[1]>0,title1="Python Matlab volumes",title2="Python",proj_on_mask1=True,adj_wT1=True,save=True)



with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)


def simulate_image_series_from_maps(map_rebuilt,mask_rebuilt,window=8):
    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask_rebuilt > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    map_ = MapFromDict("RebuiltMapFromParam", paramMap=map_for_sim)
    map_.buildParamMap()

    map_.build_ref_images(seq=seq)
    rebuilt_image_series = map_.images_series
    rebuilt_image_series= [np.mean(gp, axis=0) for gp in groupby(rebuilt_image_series, window)]
    rebuilt_image_series=np.array(rebuilt_image_series)
    return rebuilt_image_series,map_for_sim

rebuilt_image_series_python,map_for_sim_python=simulate_image_series_from_maps(all_maps_python_current_slice[0],all_maps_python_current_slice[1])
rebuilt_image_series_python_matlab_volumes,map_for_sim_python_matlab_volumes=simulate_image_series_from_maps(all_maps_python_matlab_volumes_current_slice[0],all_maps_python_matlab_volumes_current_slice[1])
rebuilt_image_series_matlab,map_for_sim_matlab=simulate_image_series_from_maps(all_maps_matlab_current_slice[0],all_maps_matlab_current_slice[1])

volumes_python_transformed = np.rot90(np.flip(volumes_python,axis=1),axes=(1,2))
volumes_matlab_transformed = np.rot90(np.flip(volumes_matlab_slice,axis=1),axes=(1,2))

plt.close("all")
ts=1
metric=np.abs
plt.figure()
plt.imshow(metric(rebuilt_image_series_python[ts]))
plt.title("Python")
plt.figure()
plt.imshow(metric(rebuilt_image_series_python_matlab_volumes[ts]))
plt.title("Python Matlab volumes")
plt.figure()
plt.imshow(metric(rebuilt_image_series_matlab[ts]))
plt.title("Matlab")
plt.figure()
plt.imshow(metric(volumes_python_transformed[ts]))
plt.title("Orig Python")
plt.figure()
plt.imshow(metric(volumes_matlab_transformed[ts]))
plt.title("Orig Matlab")


map_df_python=makevol(maps_python_current_slice["df"],mask_python_current_slice>0)
map_df_matlab=makevol(all_maps_matlab_current_slice[0]["df"],all_maps_matlab_current_slice[1]>0)

map_df_python_matlab_volumes=makevol(maps_python_matlab_volumes_current_slice["df"],mask_python_matlab_volumes_current_slice>0)

error_df =  map_df_python-map_df_matlab

map_ff_python=makevol(maps_python_current_slice["ff"],mask_python_current_slice>0)
map_ff_python_matlab_volumes=makevol(maps_python_matlab_volumes_current_slice["ff"],mask_python_matlab_volumes_current_slice>0)
map_ff_matlab=makevol(all_maps_matlab_current_slice[0]["ff"],all_maps_matlab_current_slice[1]>0)

map_df_python_matlab_volumes_filtered=map_df_python_matlab_volumes[all_maps_matlab_current_slice[1]>0]
map_df_matlab_filtered=map_df_matlab[all_maps_matlab_current_slice[1]>0]
map_ff_python_matlab_volumes_filtered=map_ff_python_matlab_volumes[all_maps_matlab_current_slice[1]>0]
map_ff_matlab_filtered=map_ff_matlab[all_maps_matlab_current_slice[1]>0]

map_df_python_matlab_volumes_filtered[np.nonzero(1-(map_ff_python_matlab_volumes_filtered==map_ff_matlab_filtered))]=0.0
map_df_matlab_filtered[np.nonzero(1-(map_ff_python_matlab_volumes_filtered==map_ff_matlab_filtered))]=0.0

error_df_filtered = map_df_python_matlab_volumes_filtered-map_df_matlab_filtered
error_df_filtered = makevol(error_df_filtered,all_maps_matlab_current_slice[1]>0)
idx_max_diff_filtered = np.unravel_index(np.argmax(error_df_filtered),error_df_filtered.shape)

plt.figure()
plt.imshow(error_df_filtered)

print(pd.DataFrame(np.abs(error_df[all_maps_matlab_current_slice[1]>0][np.nonzero((map_ff_python_matlab_volumes_filtered==map_ff_matlab_filtered))]),columns=["Errors df on pixels where FF match"]).describe())
print(pd.DataFrame(np.abs(error_df[all_maps_matlab_current_slice[1]>0][np.nonzero(1-(map_ff_python_matlab_volumes_filtered==map_ff_matlab_filtered))]),columns=["Errors df on pixels where FF don't match"]).describe())
print(pd.DataFrame(np.abs(error_df[all_maps_matlab_current_slice[1]>0]),columns=["Errors df"]).describe())

plt.figure()
plt.hist(np.abs(error_df[all_maps_matlab_current_slice[1]>0]))
plt.figure()
plt.hist(np.abs(error_df[all_maps_matlab_current_slice[1]>0][np.nonzero((map_ff_python_matlab_volumes_filtered==map_ff_matlab_filtered))]))
plt.figure()
plt.hist(np.abs(error_df[all_maps_matlab_current_slice[1]>0][np.nonzero(1-(map_ff_python_matlab_volumes_filtered==map_ff_matlab_filtered))]))



error_df = map_df_python-map_df_matlab
max_diff_df = np.unravel_index(np.argmax(error_df),image_size)
plt.figure()
plt.imshow(error_df)
plt.colorbar()

error_df_filtered=error_df.copy()
error_df_filtered[np.abs(error_df)>0.03]=0.

max_diff_df_filtered = np.unravel_index(np.argmax(error_df_filtered),image_size)
plt.figure()
plt.imshow(error_df_filtered)
plt.colorbar()

plt.figure();plt.plot(np.sort(error_df.flatten())[::-1])

plt.close("all")

pixel=(160,65)
pixel=(69,51)
pixel=(170,80)
pixel=(154,188)
pixel=(155,205)
pixel=(154,62)

param_retrieved_python = [map_for_sim_python[k][pixel[0],pixel[1]] for k in map_for_sim_python.keys()]
param_retrieved_python_matlab_volumes = [map_for_sim_python_matlab_volumes[k][pixel[0],pixel[1]] for k in map_for_sim_python_matlab_volumes.keys()]
param_retrieved_matlab= [map_for_sim_matlab[k][pixel[0],pixel[1]] for k in map_for_sim_matlab.keys()]

param_retrieved_python=dict(zip(map_for_sim_python.keys(),param_retrieved_python))
param_retrieved_python_matlab_volumes=dict(zip(map_for_sim_python_matlab_volumes.keys(),param_retrieved_python_matlab_volumes))
param_retrieved_matlab=dict(zip(map_for_sim_matlab.keys(),param_retrieved_matlab))
param_retrieved_matlab.pop("wT2")
param_retrieved_matlab.pop("fT2")
param_retrieved_matlab["attB1"]=np.round(param_retrieved_matlab["attB1"],2)



dictfile = "mrf175_SimReco2.dict"
dictfile = "mrf175_Dico2_Invivo.dict"

mrfdict = dictsearch.Dictionary()
mrfdict.load(dictfile, force=True)

metric=np.real
python_retrieved=metric(mrfdict[tuple(list(param_retrieved_python.values())[:-1])][:, 0] * (1 - param_retrieved_python["ff"]) + mrfdict[tuple(list(param_retrieved_python.values())[:-1]
                                                                                                                )][:, 1] * (param_retrieved_python["ff"]))
python_matlab_volumes_retrieved=metric(mrfdict[tuple(list(param_retrieved_python_matlab_volumes.values())[:-1])][:, 0] * (1 - param_retrieved_python_matlab_volumes["ff"]) + mrfdict[tuple(list(param_retrieved_python_matlab_volumes.values())[:-1]
                                                                                                                )][:, 1] * (param_retrieved_python_matlab_volumes["ff"]))
matlab_retrieved=metric(mrfdict[tuple(list(param_retrieved_matlab.values())[:-1])][:, 0] * (1 - param_retrieved_matlab["ff"]) + mrfdict[tuple(list(param_retrieved_matlab.values())[:-1]
                                                                                                                )][:, 1] * (param_retrieved_matlab["ff"]))

plt.figure()
plt.plot(python_retrieved/np.std(python_retrieved),label="Python pattern")
plt.plot(python_matlab_volumes_retrieved/np.std(python_matlab_volumes_retrieved),label="Python Matlab volumes pattern")
plt.plot(matlab_retrieved/np.std(matlab_retrieved),label="Matlab pattern")
plt.legend()


#metric = np.real
plt.figure()
signal_orig_python = metric(volumes_python_transformed[:,pixel[0],pixel[1]])
signal_orig_python=signal_orig_python/np.std(signal_orig_python)
signal_rebuilt_python = metric(rebuilt_image_series_python[:,pixel[0],pixel[1]])
signal_rebuilt_python=signal_rebuilt_python/np.std(signal_rebuilt_python)
plt.plot(signal_orig_python,label="Original Python")
error_rebuilt_python = np.linalg.norm(signal_rebuilt_python-signal_orig_python)
plt.plot(signal_rebuilt_python,label="Rebuilt Python {}; Params : {}".format(round(error_rebuilt_python,2),param_retrieved_python))
plt.legend()


plt.figure()
signal_orig_matlab = metric(volumes_matlab_transformed[:,pixel[0],pixel[1]])
signal_orig_matlab=signal_orig_matlab/np.std(signal_orig_matlab)
signal_rebuilt_python_matlab_volumes = metric(rebuilt_image_series_python_matlab_volumes[:,pixel[0],pixel[1]])
signal_rebuilt_python_matlab_volumes=signal_rebuilt_python_matlab_volumes/np.std(signal_rebuilt_python_matlab_volumes)
signal_rebuilt_matlab = metric(rebuilt_image_series_matlab[:,pixel[0],pixel[1]])
signal_rebuilt_matlab=signal_rebuilt_matlab/np.std(signal_rebuilt_matlab)
plt.plot(signal_orig_matlab,label="Original Matlab pixel ")
error_rebuilt_matlab = np.linalg.norm(signal_rebuilt_matlab-signal_orig_matlab)
plt.plot(signal_rebuilt_matlab,label="Rebuilt Matlab {}; Params : {}".format(round(error_rebuilt_matlab,2),param_retrieved_matlab))
error_rebuilt_python_matlab_volumes = np.linalg.norm(signal_rebuilt_python_matlab_volumes-signal_orig_matlab)
plt.plot(signal_rebuilt_python_matlab_volumes,label="Rebuilt Python on Matlab volumes {}; Params : {}".format(round(error_rebuilt_python_matlab_volumes,2),param_retrieved_python_matlab_volumes))
plt.legend()

error_python = np.linalg.norm(metric(volumes_python_transformed/np.std(volumes_python_transformed,axis=0) - rebuilt_image_series_python/np.std(rebuilt_image_series_python,axis=0)),axis=0)
error_matlab = np.linalg.norm(metric(volumes_matlab_transformed/np.std(volumes_matlab_transformed,axis=0) - rebuilt_image_series_matlab/np.std(rebuilt_image_series_matlab,axis=0)),axis=0)
error_python_matlab_volumes = np.linalg.norm(metric(volumes_matlab_transformed/np.std(volumes_matlab_transformed,axis=0) - rebuilt_image_series_python_matlab_volumes/np.std(rebuilt_image_series_python_matlab_volumes,axis=0)),axis=0)

error_python[np.isnan(error_python)]=0.0
error_matlab[np.isnan(error_matlab)]=0.0
error_python_matlab_volumes[np.isnan(error_python_matlab_volumes)]=0.0
#plt.imshow(error_python)

plt.figure()
plt.hist(error_python)
plt.figure()
plt.hist(error_matlab)
plt.figure()
plt.hist(error_python_matlab_volumes)

print(pd.DataFrame(error_python.flatten(),columns=["Python errors"]).describe())
print(pd.DataFrame(error_matlab.flatten(),columns=["Matlab errors"]).describe())
print(pd.DataFrame(error_python_matlab_volumes.flatten(),columns=["Python with Matlab volumes errors"]).describe())

maskROI=buildROImask(all_maps_python_current_slice[0],max_clusters=10)

########################################################################################################################
from mutools import io
file_ROI = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/3_Comp_Matlab/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI/MRF_CS_Cuisses_ROI.mha"
maskROI = io.read(file_ROI)
maskROI = np.moveaxis(maskROI,-1,0)
maskROI = np.moveaxis(maskROI,-1,1)
maskROI=np.array(maskROI)
for j in range(maskROI.shape[0]):
    maskROI[j]=np.flip((maskROI[j]),axis=1)

maskROI = maskROI[slice,:,:][all_maps_python_current_slice[1]>0]


df_python = metrics_paramMaps_ROI(all_maps_python_current_slice[0],all_maps_matlab_current_slice[0],all_maps_python_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,maskROI=maskROI,adj_wT1=True,proj_on_mask1=True)
df_python_matlab_volumes = metrics_paramMaps_ROI(all_maps_python_matlab_volumes_current_slice[0],all_maps_matlab_current_slice[0],all_maps_python_matlab_volumes_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,maskROI=maskROI,adj_wT1=True,proj_on_mask1=True)

#df.to_csv("Results_Comparison_Invivo")
regression_paramMaps_ROI(all_maps_python_current_slice[0],all_maps_matlab_current_slice[0],all_maps_python_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,maskROI=maskROI,save=True,title="Python vs Matlab Invivo slice {}".format(slice),kept_keys=["attB1","df","wT1","ff"],adj_wT1=True,fat_threshold=0.7)
regression_paramMaps_ROI(all_maps_python_matlab_volumes_current_slice[0],all_maps_matlab_current_slice[0],all_maps_python_matlab_volumes_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,maskROI=maskROI,save=True,title="Python Matlab Volumes vs Matlab Invivo slice {}".format(slice),kept_keys=["attB1","df","wT1","ff"],adj_wT1=True,fat_threshold=0.7)

from copy import deepcopy

map_python_rounded_ff=deepcopy(all_maps_python_current_slice[0])
map_matlab_rounded_ff=deepcopy(all_maps_matlab_current_slice[0])
map_python_rounded_ff["ff"]=np.round(map_python_rounded_ff["ff"],2)
map_matlab_rounded_ff["ff"]=np.round(map_matlab_rounded_ff["ff"],2)


regression_paramMaps(map_python_rounded_ff,map_matlab_rounded_ff,all_maps_python_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,save=True,mode="Boxplot",fontsize=5)


#Check Dixon

dixon_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/3_Comp_Matlab/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI/Dixon"
dixonmap_file = "b0map.mha"
dixonmask_file = "mask.mha"

from mutools import io
dixonmap = io.read(dixon_folder+"//"+dixonmap_file)
dixonmask = io.read(dixon_folder+"//"+dixonmask_file)

plt.close("all")




sl_dix=32
image_dixon = np.flip(np.rot90(np.array(dixonmap)[:,:,sl_dix]))
image_python =  makevol(maps_python_current_slice["df"],mask_python_current_slice>0)
image_matlab = makevol(all_maps_matlab_current_slice[0]["df"],all_maps_matlab_current_slice[1]>0)

center_image_y = int(image_dixon.shape[1]/2)
resol_y = int(3/4*image_python.shape[1])
image_dixon = image_dixon[:,center_image_y-resol_y:center_image_y+resol_y]


center_image_x =  int(image_python.shape[0]/2)
resol_x = int(image_dixon.shape[0]/2)
image_python = image_python[center_image_x-resol_x:center_image_x+resol_x,:]
image_matlab = image_matlab[center_image_x-resol_x:center_image_x+resol_x,:]

fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(10,30))
ax1.imshow(image_python)
ax1.set_title("Python df")
ax2.imshow(image_dixon)
ax2.set_title("Dixon df")
ax3.imshow(image_matlab)
ax3.set_title("Matlab df")

plt.figure()
plt.imshow()
plt.colorbar()

# Check Dict
dict_conf = "mrf_dictconf_Dico2_Invivo.json"
import h5py
f = h5py.File(r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/2_Codes_info/Matlab/MRF_reco_linux/Dictionaries/Config_Dico_2D.mat","r")
paramsH5py_Matlab=f.get("paramDico")

paramMatlab = {}
for k in paramsH5py_Matlab.keys():
    paramMatlab[k]=np.array(paramsH5py_Matlab.get(k))

paramDico_Matlab ={}

paramDico_Matlab["water_T1"] =list(np.array(f[paramMatlab["T1"][0,0]]).flatten())
paramDico_Matlab["water_T2"] =list(np.array(f[paramMatlab["T2"][0,0]]).flatten())
paramDico_Matlab["fat_T1"] =list(np.array(f[paramMatlab["T1"][1,0]]).flatten())
paramDico_Matlab["fat_T2"] =list(np.array(f[paramMatlab["T2"][1,0]]).flatten())
paramDico_Matlab["ff"] = list(paramMatlab["FF"].flatten())
paramDico_Matlab["B1_att"] = list(paramMatlab["FA"].flatten())
paramDico_Matlab["delta_freqs"] = list(paramMatlab["Df"].flatten())
paramDico_Matlab["fat_amp"] = list(paramMatlab["FatAmp"].flatten())
paramDico_Matlab["fat_cshift"] = list(paramMatlab["FatShift"].flatten())

with open(dict_conf) as file:
    paramDico_Python = json.load(file)

k = "fat_cshift"

print(np.max(np.abs(np.array(paramDico_Matlab[k])-np.array(paramDico_Python[k]))))

print(paramDico_Matlab[k])
print(paramDico_Python[k])


print(len(paramDico_Matlab[k]))
print(len(paramDico_Python[k]))

dict_conf = "mrf_dictconf_SimReco2.json"
dict_conf = "mrf_dictconf_Dico2_Invivo.json"
with open(dict_conf,"rb") as file:
    dico_conf=json.load(file)

dico_count={}
for k in dico_conf.keys():
    try:
        dico_count[k]=(len(dico_conf[k]),np.min(dico_conf[k]),np.max(dico_conf[k]))
    except:
        dico_count[k] = (1,dico_conf[k],dico_conf[k])

with open("mrf_sequence.json") as file:
    seq_conf=json.load(file)

seq_count={}
for k in seq_conf.keys():
    try:
        seq_count[k]=(len(seq_conf[k]),np.min(seq_conf[k]),np.max(seq_conf[k]))
    except:
        seq_count[k] = (1,seq_conf[k],seq_conf[k])
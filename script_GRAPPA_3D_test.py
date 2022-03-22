
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

nb_filled_slices = 20
nb_empty_slices=4
repeat_slice=4
nb_slices = nb_filled_slices+2*nb_empty_slices
name = "SquareSimu3DGrappa"


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
filename_volume = filename+"_volumes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_volume_all_spokes = filename+"_volumes_all_spokes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_b1_all_spokes = filename+"_b1_all_spokes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_groundtruth = filename+"_groundtruth_volumes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")

filename_kdata = filename+"_kdata_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_kdata_all_spokes = filename+"_kdata_all_spokes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_mask= filename+"_mask_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
file_map = filename + "_sl{}_rp{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,suffix)
filename_b1 = filename+ "_b1_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")

filename_b1_grappa = filename+ "_b1_grappa_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")


filename_kdata_grappa = filename+"_kdata_grappa_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_currtraj_grappa = filename+"_currtraj_grappa_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_mask_grappa= filename+"_mask_grappa_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_volume_grappa = filename+"_volumes_grappa_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")

file_map_grappa = filename + "_grappa_sl{}_rp{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,suffix)
file_map_all_spokes = filename + "_all_spokes_sl{}_rp{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,suffix)

#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 512

undersampling_factor=2

incoherent=False
mode="old"

image_size = (nb_slices, int(npoint/2), int(npoint/2))
nb_segments=nb_allspokes
nspoke=int(nb_segments/ntimesteps)
size = image_size[1:]

with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)

if name=="SquareSimu3DGrappa":
    region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
    mask_reduction_factor=1/4


    m = RandomMap3D(name,dict_config,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,undersampling_factor=undersampling_factor,repeat_slice=repeat_slice,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)

# elif name=="KneePhantom":
#     num =1
#     file_matlab_paramMap = "./data/{}/Phantom{}/paramMap.mat".format(name,num)
#
#     m = MapFromFile(name,image_size=image_size,file=file_matlab_paramMap,rounding=True,gen_mode="other")

else:
    raise ValueError("Unknown Name")

if str.split(filename_paramMap,"/")[-1] not in os.listdir(folder):
    m.buildParamMap()
    with open(filename_paramMap, "wb" ) as file:
        pickle.dump(m.paramMap, file)

    map_rebuilt = m.paramMap
    mask = m.mask

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
        m.paramMap=pickle.load(file)
    m.mask=np.load(filename_paramMask)

# file_map_matlab = filename+"_paramMap_sl{}_rp{}.mat".format(nb_slices,repeat_slice)
# file_mask_matlab= filename+"_mask_sl{}_rp{}.mat".format(nb_slices,repeat_slice)
#
# savemat(file_map_matlab,m.paramMap)
# savemat(file_mask_matlab,{"Mask":m.mask})


m.build_ref_images(seq)

if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
    np.save(filename_groundtruth,m.images_series[::nspoke])

# i=0
# image=m.images_series[i]


# file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
#                              "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_ideal_ts{}.mha".format(
#                                                                                                                    i)
# io.write(file_mha, np.abs(image), tags={"spacing": [5, 1, 1]})


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)

nb_channels=8



nb_means=int(nb_channels**(1/3))


means_z=np.arange(1,nb_means+1)*(1/(nb_means+1))*image_size[0]
means_x=np.arange(1,nb_means+1)*(1/(nb_means+1))*image_size[1]
means_y=np.arange(1,nb_means+1)*(1/(nb_means+1))*image_size[2]


sig_z=(image_size[0]/(2*(nb_means+1)))**2
sig_x=(image_size[1]/(2*(nb_means+1)))**2
sig_y=(image_size[2]/(2*(nb_means+1)))**2

z = np.arange(image_size[0])
x = np.arange(image_size[1])
y = np.arange(image_size[2])


X,Y,Z = np.meshgrid(x,y,z)
pixels=np.stack([Z,X,Y], axis=-1)
pixels=np.moveaxis(pixels,-2,0)

pixels=pixels.reshape(-1,3)

from scipy.stats import multivariate_normal
b1_maps=[]

for mu_z in means_z:
    for mu_x in means_x:
        for mu_y in means_y:
            b1_maps.append(multivariate_normal.pdf(pixels, mean=[mu_z,mu_x,mu_y], cov=np.diag([sig_z,sig_x,sig_y])))

b1_maps = np.array(b1_maps)
b1_maps=b1_maps/np.expand_dims(np.max(b1_maps,axis=-1),axis=-1)
b1_maps=b1_maps.reshape((nb_channels,)+image_size)

sl=int(nb_slices/2)
sl=3
list_images = list(np.abs(b1_maps[:,sl,:,:]))
plot_image_grid(list_images,(3,3),title="Ground Truth : Sensitivity map for slice {}".format(sl))


ch=0
mu_x=means_x[ch]
mu_y=means_y[ch]





animate_images(b1_maps[0,:,:,:])

b1_prev = np.ones(b1_maps[0].shape,dtype=b1_maps[0].dtype)
b1_all = np.concatenate([np.expand_dims(b1_prev, axis=0), b1_maps], axis=0)

if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):

    data=[]

    #images = copy(m.images_series)

    for i in tqdm(range(1,b1_all.shape[0])):
        m.images_series*=np.expand_dims(b1_all[i]/b1_all[i-1],axis=0)
        data.append(np.array(m.generate_kdata(radial_traj,useGPU=False)))

    m.images_series/=np.expand_dims(b1_all[-1],axis=0)
    #del images

    data=np.array(data)


    #density = np.abs(np.linspace(-1, 1, npoint))
    #density = np.expand_dims(density, tuple(range(data.ndim - 1)))

    #kdata_all_channels_all_slices = data.reshape(-1, npoint)
    #del data
    #print("Performing Density Adjustment....")

    data = data.reshape(nb_channels, nb_allspokes, -1, npoint)
    #data *= density

    np.save(filename_kdata, data)
    del data
    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)



if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage,density_adj=True)
    np.save(filename_b1,b1_all_slices)
else:
    b1_all_slices=np.load(filename_b1)

sl=int(b1_all_slices.shape[1]/2)
list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
plot_image_grid(list_images,(3,3),title="Sensitivity map for slice {}".format(sl))

radial_traj_all=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)


if str.split(filename_kdata_all_spokes,"/")[-1] not in os.listdir(folder):


    data=[]

    for i in range(1,b1_all.shape[0]):
        m.images_series*=np.expand_dims(b1_all[i]/b1_all[i-1],axis=0)
        data.append(np.array(m.generate_kdata(radial_traj_all,useGPU=False)))


    #del images

    data=np.array(data)


    #density = np.abs(np.linspace(-1, 1, npoint))
    #density = np.expand_dims(density, tuple(range(data.ndim - 1)))

    #kdata_all_channels_all_slices = data.reshape(-1, npoint)
    #del data
    #print("Performing Density Adjustment....")

    data = data.reshape(nb_channels, nb_allspokes, -1, npoint)
    #data *= density

    np.save(filename_kdata_all_spokes, data)
    del data
    kdata_all_channels_all_slices_all_spokes = np.load(filename_kdata_all_spokes)

    m.build_ref_images(seq)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices_all_spokes = np.load(filename_kdata_all_spokes)


if str.split(filename_b1_all_spokes, "/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices_all_spokes = calculate_sensitivity_map_3D(kdata_all_channels_all_slices_all_spokes, radial_traj_all, res, image_size, useGPU=False,
                                                 light_memory_usage=light_memory_usage, density_adj=True)
    np.save(filename_b1_all_spokes, b1_all_slices_all_spokes)
else:
    b1_all_slices_all_spokes = np.load(filename_b1_all_spokes)


ch=0
file_mha = "/".join(["/".join(str.split(filename_b1,"/")[:-1]),"_".join(str.split(str.split(filename_b1,"/")[-1],".")[:-1])]) + "_ch{}.mha".format(ch)
io.write(file_mha,np.abs(b1_all_slices[ch]),tags={"spacing":[5,1,1]})

file_mha = "/".join(["/".join(str.split(filename_b1_all_spokes,"/")[:-1]),"_".join(str.split(str.split(filename_b1_all_spokes,"/")[-1],".")[:-1])]) + "_ch{}.mha".format(ch)
io.write(file_mha,np.abs(b1_all_slices_all_spokes[ch]),tags={"spacing":[5,1,1]})

print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    del kdata_all_channels_all_slices
    kdata_all_channels_all_slices = np.load(filename_kdata)

    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
    np.save(filename_volume,volumes_all)
    # sl=20
    # ani = animate_images(volumes_all[:,sl,:,:])
    del volumes_all



print("Building Volumes All Spokes....")
if str.split(filename_volume_all_spokes,"/")[-1] not in os.listdir(folder):
    del kdata_all_channels_all_slices_all_spokes
    kdata_all_channels_all_slices_all_spokes = np.load(filename_kdata_all_spokes)

    volumes_all_spokes=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices_all_spokes,radial_traj_all,image_size,b1=b1_maps,density_adj=True,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
    np.save(filename_volume_all_spokes,volumes_all_spokes)
    sl=int(nb_slices/2)
    ani = animate_images(volumes_all_spokes[:,sl,:,:])
    del volumes_all_spokes




print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    del kdata_all_channels_all_slices
    kdata_all_channels_all_slices = np.load(filename_kdata)

    selected_spokes=None
    mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=True,threshold_factor=1/25, normalize_kdata=False,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask,mask)
    animate_images(mask)
    del mask


#del kdata_all_channels_all_slices
#kdata_all_channels_all_slices = np.load(filename_kdata)

#volume=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,normalize_volumes=True)
#animate_images(volume[0],interval=1000)

del kdata_all_channels_all_slices
kdata_all_channels_all_slices = np.load(filename_kdata)

del kdata_all_channels_all_slices_all_spokes
kdata_all_channels_all_slices_all_spokes = np.load(filename_kdata_all_spokes)


#
#
# for ch in range(nb_channels):
#     plt.figure()
#     plt.imshow(b1_maps[ch])
#     plt.colorbar()


# npoint_y = 128
# curr_traj=cartesian_traj_2D(npoint_x,npoint_y)



traj_all=radial_traj_all.get_traj()
traj_all=traj_all.reshape(nb_allspokes,-1,npoint,3)

# plt.imshow(np.abs(rebuilt_image))


#Calib data



kdata_all_channels_completed_all_ts = []
curr_traj_completed_all_ts =[]

calibration_mode="Standard"
lambd = 0.
over_calibrate=3
#over_calibrate=1

#kernel_x = list(range(-31,32))
kernel_y = [0,1,2]
#kernel_y=[0,1]

Neq = int(undersampling_factor*np.maximum(over_calibrate-len(kernel_y)+1,0)+1)

len_kerx = int(npoint*Neq/(nb_channels*len(kernel_y)))
kernel_x=np.arange(-(int(len_kerx/2)-1),int(len_kerx/2))

#ts=8
ts=1

plot_fit=False

for ts in tqdm(range(nb_allspokes)):
    center_line = int(nb_slices / 2) - 1
    image=m.images_series[ts]
    kz_all = np.unique(traj_all[ts, :, :, 2])
    kz_curr = np.unique(radial_traj.get_traj()[ts, :, 2])
    kz_to_estimate = np.array(sorted(list(set(kz_all) - set(kz_curr))))
    kz_lines_to_estimate = np.where(np.isin(kz_all, kz_to_estimate))[0]
    kz_lines_measured = np.where(1-np.isin(kz_all, kz_to_estimate))[0]

    if kz_all[center_line] in kz_curr:
        center_line=center_line+1

    calib_lines=np.zeros((undersampling_factor-1,Neq)).astype(int)

    for i in range(undersampling_factor-1):
        calib_line = center_line
        for j in range(Neq):
            calib_lines[i, j]=calib_line
            calib_line+=1

        center_line+=1

    traj_calib_for_grappa=traj_all[ts,calib_lines,:,:]
    traj_calib_for_grappa=traj_calib_for_grappa.reshape(traj_calib_for_grappa.shape[0],-1,3)

    traj_calib=traj_calib_for_grappa.reshape(-1,3)

    kdata_calib =finufft.nufft3d2(traj_calib[:,2],traj_calib[:,0], traj_calib[:,1], m.images_series[0])
    kdata_calib_all_channels=[finufft.nufft3d2(traj_calib[:,2],traj_calib[:,0], traj_calib[:,1], b1*image) for b1 in b1_maps]


    #
    # for ch in range(nb_channels):
    #     plt.figure()
    #     plt.imshow(np.abs(image_rebuilt_all_channels[ch]))




    ###GRAPPA RECONSTRUCTION#####





    kdata_all_channels=kdata_all_channels_all_slices[:,ts,:,:]

    curr_traj_for_grappa=radial_traj.get_traj()[ts].reshape(-1,npoint,3)

    kdata_calib_all_channels=np.array(kdata_calib_all_channels).reshape((nb_channels,)+traj_calib_for_grappa.shape[:-1])

    kdata_all_channels_for_grappa=kdata_all_channels_all_slices[:,ts,:,:]

    curr_traj_for_grappa_calib=traj_all[ts,np.unique(np.concatenate([calib_lines.flatten(),kz_lines_measured])),:,:].reshape(-1,3)
    #kdata_for_grappa_calib = finufft.nufft3d2(traj_calib[:, 2], traj_calib[:, 0], traj_calib[:, 1], m.images_series[0])
    kdata_all_channels_for_grappa_calib = np.array([finufft.nufft3d2(curr_traj_for_grappa_calib[:, 2], curr_traj_for_grappa_calib[:, 0], curr_traj_for_grappa_calib[:, 1], b1 * image) for b1 in b1_maps])

    curr_traj_for_grappa_calib = curr_traj_for_grappa_calib.reshape(-1,npoint,3)
    kdata_all_channels_for_grappa_calib = kdata_all_channels_for_grappa_calib.reshape(nb_channels,-1,npoint)


    # CALIBRATION

    F_source_calib=np.zeros((nb_channels*len(kernel_x)*len(kernel_y),traj_calib_for_grappa.shape[1]),dtype=kdata_calib.dtype)

    pad_y=((np.array(kernel_y)<0).sum(),(np.array(kernel_y)>0).sum())
    pad_x=((np.array(kernel_x)<0).sum(),(np.array(kernel_x)>0).sum())

    kdata_all_channels_for_grappa_padded=np.pad(kdata_all_channels_for_grappa,((0,0),pad_y,pad_x),mode="edge")
    curr_traj_for_grappa_padded = np.pad(curr_traj_for_grappa, (pad_y, pad_x,(0,0)), mode="edge")

    kdata_all_channels_for_grappa_calib_padded = np.pad(kdata_all_channels_for_grappa_calib, ((0, 0), pad_y, pad_x), mode="edge")
    curr_traj_for_grappa_calib_padded = np.pad(curr_traj_for_grappa_calib, (pad_y, pad_x, (0, 0)), mode="edge")

    weights=np.zeros((len(calib_lines),nb_channels,nb_channels*len(kernel_x)*len(kernel_y)),dtype=kdata_calib.dtype)

    index_ky_target=0
    j=250

    for index_ky_target in range(len(calib_lines)):
        print("Calibration for line {}".format(index_ky_target))
        F_target_calib = kdata_calib_all_channels[:, index_ky_target, :]
        for j in tqdm(range(traj_calib_for_grappa.shape[1])):
            kx=traj_calib_for_grappa[index_ky_target,j,0]
            ky = traj_calib_for_grappa[index_ky_target,j,1]
            kz = traj_calib_for_grappa[index_ky_target, j, 2]

            curr_traj_for_grappa_calib_no_kz=copy(curr_traj_for_grappa_calib)
            curr_traj_for_grappa_calib_no_kz[curr_traj_for_grappa_calib_no_kz[:, :, 2] == kz] = np.inf

            kx_comp = (kx-curr_traj_for_grappa_calib_no_kz[0,:,0])
            kx_comp[kx_comp < 0] = np.inf

            kx_ref=curr_traj_for_grappa_calib_no_kz[0,np.argmin(kx_comp),0]

            ky_comp = (ky - curr_traj_for_grappa_calib_no_kz[0,:,1])
            ky_comp[ky_comp < 0] = np.inf

            ky_ref = curr_traj_for_grappa_calib_no_kz[0,np.argmin(ky_comp),1]

            kz_comp = (kz - curr_traj_for_grappa_calib_no_kz[:, 0, 2])
            kz_comp[kz_comp < 0] = np.inf

            kz_ref = curr_traj_for_grappa_calib_no_kz[np.argmin(kz_comp), 0, 2]

            j0, i0 = np.unravel_index(
                np.argmin(np.linalg.norm(curr_traj_for_grappa_calib - np.expand_dims(np.array([kx_ref, ky_ref,kz_ref]), axis=(0, 1)), axis=-1)),
                curr_traj_for_grappa_calib.shape[:-1])
            local_indices = np.stack(np.meshgrid(pad_y[0]+j0 + undersampling_factor*np.array(kernel_y), pad_x[0]+i0 + np.array(kernel_x)), axis=0).T.reshape(-1, 2)

            local_kdata_for_grappa = kdata_all_channels_for_grappa_calib_padded[:, local_indices[:, 0], local_indices[:, 1]]
            local_kdata_for_grappa = np.array(local_kdata_for_grappa)
            local_kdata_for_grappa = local_kdata_for_grappa.flatten()
            F_source_calib[:,j]=local_kdata_for_grappa

            local_traj_for_grappa = curr_traj_for_grappa_calib_padded[local_indices[:, 0], local_indices[:, 1], :]

            local_traj_for_grappa = np.tile(local_traj_for_grappa,
                                            (nb_channels,) + tuple(
                                                np.ones(local_traj_for_grappa.ndim).astype(int))).reshape(-1, 3)

        if calibration_mode=="Tikhonov":
                weights[index_ky_target]=F_target_calib@F_source_calib.conj().T@np.linalg.inv(F_source_calib@F_source_calib.conj().T+lambd*np.eye(F_source_calib.shape[0]))

        elif calibration_mode=="Lasso":
            def function_to_optimize(w):
                global lambd
                Y=F_target_calib
                X=F_source_calib

                w = w.reshape((2,nb_channels,nb_channels*len(kernel_x)*len(kernel_y)))

                wR = w[0]
                wI = w[1]

                #print(wR.shape)

                wRX=wR @ X
                wIX = wI @ X

                # print(X.shape)
                # print(Y.shape)
                # print(wR.shape)
                # print(wI.shape)
                # print(wRX.shape)
                # print(wIX.shape)
                #
                # print(((wRX.conj().T)@wRX).shape)

                return np.trace((wRX.conj().T)@wRX) + np.trace((wIX.conj().T)@wIX) - 2*np.trace(np.real(Y.conj().T @ wR @ X)) + 2*np.trace(np.imag(Y.conj().T @ wI @ X)) + lambd*(np.linalg.norm(wR.flatten(),ord=1)+np.linalg.norm(wI.flatten(),ord=1))

            w0=np.zeros((2,nb_channels,nb_channels*len(kernel_x)*len(kernel_y)),dtype=kdata_calib.dtype).flatten()

            res=minimize(function_to_optimize,w0)

            w_sol = res.x.reshape((2,nb_channels,nb_channels*len(kernel_x)*len(kernel_y)))

            weights[index_ky_target]=np.apply_along_axis(lambda args: [complex(*args)], 0, w_sol)


        else :
            weights[index_ky_target]=F_target_calib@np.linalg.pinv(F_source_calib)

        if plot_fit:
            F_estimate =weights[index_ky_target]@F_source_calib

            plt.figure()
            plt.plot(np.linalg.norm(F_target_calib-F_estimate,axis=0)/np.sqrt(nb_channels))
            plt.title("Calibration Error for line {} aggregated accross channels".format(index_ky_target))

            nplot_x = int(np.sqrt(nb_channels))
            nplot_y = int(nb_channels/nplot_x)+1
            #
            # # metric=np.real
            # # fig, axs = plt.subplots(nplot_x, nplot_y)
            # # fig.suptitle("Calibration Performance for line {} : Real Part".format(index_ky_target))
            # # for i in range(nplot_x):
            # #     for j in range(nplot_y):
            # #         ch=i*nplot_y+j
            # #         if ch >= nb_channels:
            # #             break
            # #         axs[i, j].plot(metric(F_target_calib[ch,:]),label="Target")
            # #         axs[i, j].plot(metric(F_estimate[ch, :]), label="Estimate")
            # #         axs[i,j].set_title('Channel {}'.format(ch))
            # #         axs[i,j].legend(loc="upper right")
            # #
            # # metric = np.imag
            # # fig, axs = plt.subplots(nplot_x, nplot_y)
            # # fig.suptitle("Calibration Performance for line {} : Imaginary Part".format(index_ky_target))
            # # for i in range(nplot_x):
            # #     for j in range(nplot_y):
            # #         ch = i * nplot_y + j
            # #         if ch >= nb_channels:
            # #             break
            # #         axs[i, j].plot(metric(F_target_calib[ch, :]), label="Target")
            # #         axs[i, j].plot(metric(F_estimate[ch, :]), label="Estimate")
            # #         axs[i, j].set_title('Channel {}'.format(ch))
            # #         axs[i, j].legend(loc="upper right")
            #
            metric = np.abs
            fig, axs = plt.subplots(nplot_x, nplot_y)
            fig.suptitle("Calibration Performance for line {} : Amplitude".format(index_ky_target))
            for i in range(nplot_x):
                for j in range(nplot_y):
                    ch = i * nplot_y + j
                    if ch >= nb_channels:
                        break
                    axs[i, j].plot(metric(F_target_calib[ch, :]), label="Target")
                    axs[i, j].plot(metric(F_estimate[ch, :]), label="Estimate")
                    axs[i, j].set_title('Channel {}'.format(ch))
                    axs[i, j].legend(loc="upper right")

            metric=np.abs
            fig, axs = plt.subplots(nplot_x, nplot_y)
            kdata_source_lines = kdata_all_channels_for_grappa[:, pad_y[0] + j0 + np.array(kernel_y), :]
            fig.suptitle("Calibration Lines used for {} : Amplitude".format(index_ky_target))
            for i in range(nplot_x):
                for j in range(nplot_y):
                    ch = i * nplot_y + j
                    if ch >= nb_channels:
                        break
                    axs[i, j].plot(metric(F_target_calib[ch, :]), label="Target")
                    axs[i, j].plot(metric(F_estimate[ch, :]), label="Estimate")
                    axs[i, j].plot(metric(kdata_source_lines[ch, :,:]).T,linestyle='dashed',alpha=0.5)
                    axs[i, j].set_title('Channel {}'.format(ch))
                    axs[i, j].legend(loc="upper right")

            plt.figure()
            kdata_for_correl=np.moveaxis(kdata_source_lines,0,1).reshape(-1,kdata_source_lines.shape[-1])
            sns.heatmap(np.real(np.corrcoef(kdata_for_correl)))

            plt.figure()
            sns.distplot(np.real(np.corrcoef(kdata_for_correl)),bins=10)

            print(np.linalg.cond(F_source_calib))

            def J(w):
                global F_source_calib
                global F_target
                F_estimate = w @ F_source_calib
                return np.linalg.norm(F_target_calib-F_estimate)

            N=10
            w = weights[index_ky_target].flatten()
            num=np.random.choice(len(w))
            w0_real = np.real(w[num])
            w0_imag = np.imag(w[num])
            #w_real_range = np.arange(w0_real/2,2*w0_real,(2*w0_real-w0_real/2)/N)
            #w_imag_range = np.arange(w0_imag / 2, 2 * w0_imag, (2 * w0_imag - w0_imag / 2) / N)
            w_real_max = np.maximum(1, 2 * np.abs(w0_real))
            w_imag_max = np.maximum(1, 2 * np.abs(w0_imag))

            w_real_range = np.arange(-w_real_max, w_real_max, (2 * w_real_max) / N)
            w_imag_range = np.arange(-w_imag_max, w_imag_max, (2 * w_imag_max) / N)

            W_R,W_I=np.meshgrid(w_real_range,w_imag_range)

            Z=np.zeros((N,N))
            for iR in tqdm(range(N)):
                for iI in range(N):
                    curr_w = w
                    curr_w[num] = w_real_range[iR]+1j* w_imag_range[iI]
                    Z[iR,iI]=J(curr_w.reshape(weights[index_ky_target].shape))

            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(W_R, W_I, Z.T,alpha=0.5)

            ax.scatter(w0_real, w0_imag, J(weights[index_ky_target]),marker="x", c="red")

            ax.set_xlabel('W_R')
            ax.set_ylabel('W_I')
            ax.set_zlabel('J')
            ax.set_title("Index {}".format(num))
            plt.show()


            #Plot result on sensi

            b1_maps_start = int(mask_reduction_factor*b1_maps.shape[2])
            b1_maps_end=int(b1_maps.shape[2]-mask_reduction_factor*b1_maps.shape[2])
            x = np.random.choice(range(b1_maps_start,b1_maps_end))
            y = np.random.choice(range(b1_maps_start,b1_maps_end))
            z = np.random.choice(range(nb_empty_slices,nb_slices-nb_empty_slices))

            s_target = b1_maps[:,z,x,y] * np.exp(-1j*(kx_ref*x+ky_ref*y+kz_ref*z))*m.mask[z,x,y]
            s_source = np.tile(b1_maps[:,z,x,y],(len(kernel_x)*len(kernel_y)))*np.exp(-1j*(local_traj_for_grappa[:,0]*x+local_traj_for_grappa[:,1]*y+local_traj_for_grappa[:,2]*z))*m.mask[z,x,y]

            def J_sensi(w_):
                global x
                global y
                global z

                global s_source
                global s_target
                s_estimate = w_ @ s_source
                return np.linalg.norm(s_target-s_estimate)



            #w = weights[index_ky_target].flatten()
            #num = np.random.choice(len(w))
            #w0_real = np.real(w[num])
            #w0_imag = np.imag(w[num])

            #N = 100
            #w_real_range = np.arange(w0_real / 2, 2 * w0_real, (2 * w0_real - w0_real / 2) / N)
            #w_imag_range = np.arange(w0_imag / 2, 2 * w0_imag, (2 * w0_imag - w0_imag / 2) / N)



            #W_R, W_I = np.meshgrid(w_real_range, w_imag_range)

            Z = np.zeros((N, N))
            for iR in tqdm(range(N)):
                for iI in range(N):
                    curr_w = copy(w)
                    curr_w[num] = w_real_range[iR] + 1j * w_imag_range[iI]
                    Z[iR, iI] = J_sensi(curr_w.reshape(weights[index_ky_target].shape))

            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(W_R, W_I, Z.T,alpha=0.5)

            ax.scatter(w0_real, w0_imag, J_sensi(weights[index_ky_target]), marker="x", c="red")

            # iR=np.argmin(np.abs(w0_real-w_real_range))
            # iI=np.argmin(np.abs(w0_imag - w_imag_range))
            #
            # ax.scatter(w_real_range[iR], w_imag_range[iI], Z[iR, iI], marker="x", c="green")
            #
            # curr_w = copy(w)
            # curr_w[num] = w_real_range[iR] + 1j * w_imag_range[iI]
            # J_sensi(curr_w.reshape(weights[index_ky_target].shape))

            ax.set_xlabel('W_R')
            ax.set_ylabel('W_I')
            ax.set_zlabel('J_sensi')
            ax.set_title("Index {} Position {}".format(num,(x,y,z)))
            plt.show()


            #ESTIMATION





    print("Estimating missing kz lines for timestep {}".format(ts))

    curr_traj_for_grappa_ts = radial_traj.get_traj()[ts].reshape(-1, npoint, 3)

    F_target_estimate = np.zeros((nb_channels,len(kz_lines_to_estimate),npoint),dtype=kdata_calib.dtype)

    i=6
    l=kz_lines_to_estimate[i]

    j=250

    for i,l in (enumerate(kz_lines_to_estimate)):
        F_source_estimate = np.zeros((nb_channels * len(kernel_x) * len(kernel_y), npoint),
                                  dtype=kdata_calib.dtype)
        for j in range(npoint):

            kx=traj_all[ts,l,j,0]
            ky = traj_all[ts, l, j,1]
            kz = traj_all[ts, l, j,2 ]


            points = np.unique(curr_traj_for_grappa_ts[:, :, 2].flatten())
            kz_comp = (kz - points)
            kz_comp[kz_comp < 0] = np.inf
            slice_ref = np.argmin(kz_comp)
            kz_ref = points[slice_ref]

            traj_slice = curr_traj_for_grappa_ts[slice_ref,:,:]

            points = traj_slice[:, 0].flatten()
            kx_comp = (kx - points)
            kx_comp[kx_comp < 0] = np.inf
            kx_ref = points[np.argmin(kx_comp)]

            points = traj_slice[:, 1].flatten()
            ky_comp = (ky - points)
            ky_comp[ky_comp < 0] = np.inf
            ky_ref = points[np.argmin(ky_comp)]


            j0, i0 = np.unravel_index(
                    np.argmin(np.linalg.norm(curr_traj_for_grappa_ts - np.expand_dims(np.array([kx_ref, ky_ref,kz_ref]), axis=(0, 1)), axis=-1)),
                    curr_traj_for_grappa_ts.shape[:-1])
            local_indices = np.stack(np.meshgrid(pad_y[0]+j0 + np.array(kernel_y), pad_x[0]+i0 + np.array(kernel_x)), axis=0).T.reshape(-1, 2)

            local_kdata_for_grappa = kdata_all_channels_for_grappa_padded[:, local_indices[:, 0], local_indices[:, 1]]
            local_kdata_for_grappa = np.array(local_kdata_for_grappa)
            local_kdata_for_grappa = local_kdata_for_grappa.flatten()
            F_source_estimate[:,j]=local_kdata_for_grappa

        F_target_estimate[:,i,:]=weights[(l-1)%(undersampling_factor-1)]@F_source_estimate




    kdata_all_channels_completed=np.concatenate([kdata_all_channels_for_grappa,F_target_estimate],axis=1)
    curr_traj_estimate=traj_all[ts,kz_lines_to_estimate,:,:]
    curr_traj_completed=np.concatenate([curr_traj_for_grappa,curr_traj_estimate],axis=0)

    # plt.figure()
    # plt.plot(np.abs(kdata_all_channels_for_grappa[0, 6, :]))
    # plt.plot(np.abs(kdata_all_channels_completed.reshape(nb_channels, nb_slices, -1)[0, 6, :]))

    #Replace calib lines
    for i, l in (enumerate(kz_lines_to_estimate)):
        if l in calib_lines.flatten():
            ind_line= np.argwhere(np.unique(np.concatenate([calib_lines.flatten(),kz_lines_measured]))==l).flatten()[0]
            kdata_all_channels_completed[:,kdata_all_channels_for_grappa.shape[1]+i,:]=kdata_all_channels_for_grappa_calib[:,ind_line,:]


    kdata_all_channels_completed=kdata_all_channels_completed.reshape((nb_channels,-1))
    curr_traj_completed=curr_traj_completed.reshape((-1,3))

    # plt.figure()
    # plt.plot(np.abs(kdata_all_channels_for_grappa[0, 6, :]))
    # plt.plot(np.abs(kdata_all_channels_completed.reshape(nb_channels, nb_slices, -1)[0, 6, :]))

    kdata_all_channels_completed_all_ts.append(kdata_all_channels_completed)
    curr_traj_completed_all_ts.append(curr_traj_completed)

    #Replace calib lines



kdata_all_channels_completed_all_ts=np.array(kdata_all_channels_completed_all_ts)
kdata_all_channels_completed_all_ts=np.moveaxis(kdata_all_channels_completed_all_ts,0,1)
curr_traj_completed_all_ts=np.array(curr_traj_completed_all_ts)

for ts in tqdm(range(curr_traj_completed_all_ts.shape[0])):
    ind=np.lexsort((curr_traj_completed_all_ts[ts][:, 0], curr_traj_completed_all_ts[ts][:, 2]))
    curr_traj_completed_all_ts[ts] = curr_traj_completed_all_ts[ts,ind,:]
    kdata_all_channels_completed_all_ts[:,ts]=kdata_all_channels_completed_all_ts[:,ts,ind]

#plt.figure()
#plt.plot(np.abs(kdata_all_channels_for_grappa[0, 6, :]))
#plt.plot(np.abs(kdata_all_channels_completed_all_ts[0,0].reshape(nb_slices, -1)[12, :]))

np.save(filename_kdata_grappa,kdata_all_channels_completed_all_ts)
np.save(filename_currtraj_grappa,curr_traj_completed_all_ts)

kdata_all_channels_completed_all_ts=np.load(filename_kdata_grappa)
curr_traj_completed_all_ts=np.load(filename_currtraj_grappa)



ch=0
metric=np.abs
plt.figure()
plt.title("Ts {} Sl {}".format(ts,sl))
plt.plot(metric(kdata_all_channels_all_slices_all_spokes[ch].reshape(nb_segments,nb_slices,npoint)[ts,sl]),label='Kdata fully sampled')
plt.plot(metric(kdata_all_channels_completed_all_ts[ch].reshape(nb_segments,nb_slices,npoint)[ts,sl]),label='Kdata grappa estimate')
plt.legend()



ts=0
#sl=np.random.choice(nb_slices)
#ts=np.random.choice(nb_segments)
ch=3
metric=np.abs

plt.figure()
plt.title("Fully sampled Ch {} Ts {}".format(ch,ts,sl))
curr_traj = radial_traj_all.get_traj()[ts]
ind = np.lexsort((curr_traj[:,0], curr_traj[:,2]))
curr_kdata = kdata_all_channels_all_slices_all_spokes[ch,ts].flatten()
curr_kdata=curr_kdata[ind]
for sl in range(nb_slices):
    plt.plot(metric(curr_kdata.reshape(kdata_all_channels_all_slices_all_spokes.shape[2], npoint)[sl]),
             label='Kdata fully sampled sl {}'.format(sl))

plt.legend()


plot_next_slice=False
ts=1
ch=0
sl =17
metric=np.abs
plt.figure()
plt.title("Ch {} Ts {} Sl {}".format(ch,ts,sl))
curr_traj = radial_traj_all.get_traj()[ts]
ind = np.lexsort((curr_traj[:,0], curr_traj[:,2]))
curr_traj=curr_traj[ind]
curr_kdata = kdata_all_channels_all_slices_all_spokes[ch,ts].flatten()
curr_kdata=curr_kdata[ind]
plt.plot(metric(curr_kdata.reshape(kdata_all_channels_all_slices_all_spokes.shape[2],npoint)[sl]),label='Kdata fully sampled')
plt.plot(metric(kdata_all_channels_completed_all_ts[ch, 0].reshape(nb_slices,npoint)[sl]),label='Kdata grappa estimate')

if plot_next_slice:
    if sl>0:
        plt.plot(metric(curr_kdata.reshape(kdata_all_channels_all_slices_all_spokes.shape[2],npoint)[sl-1]),label='Kdata fully sampled previous slice',linestyle='dashed')
    else:
        plt.plot(metric(curr_kdata.reshape(kdata_all_channels_all_slices_all_spokes.shape[2], npoint)[0]),
                 label='Kdata fully sampled previous slice',linestyle='dashed')
    if sl<(nb_slices-2):
        plt.plot(metric(curr_kdata.reshape(kdata_all_channels_all_slices_all_spokes.shape[2],npoint)[sl+1]),label='Kdata fully sampled next slice',linestyle='dashed')
    else:
        plt.plot(metric(curr_kdata.reshape(kdata_all_channels_all_slices_all_spokes.shape[2], npoint)[nb_slices-1]),
                 label='Kdata fully sampled next slice',linestyle='dashed')
plt.legend()


plt.figure()
plt.title("Error Ch {} Ts {} Sl {}".format(ch,ts,sl))
plt.plot(metric(curr_kdata.reshape(kdata_all_channels_all_slices_all_spokes.shape[2],npoint)[sl]-kdata_all_channels_completed_all_ts[ch,ts].reshape(nb_slices,npoint)[sl]))

ts=10
sl=3
ch=0
metric=np.real
plt.figure()
plt.title("Ts {} Sl {}".format(ts,sl))
curr_traj = radial_traj.get_traj()[ts]
ind = np.lexsort((curr_traj[:,0], curr_traj[:,2]))
curr_kdata = kdata_all_channels_all_slices[ch,ts].flatten()
curr_kdata=curr_kdata[ind]
plt.plot(metric(curr_kdata.reshape(kdata_all_channels_all_slices.shape[2],npoint)[sl]),label='Kdata fully sampled')
plt.plot(metric(kdata_all_channels_completed_all_ts[ch,ts].reshape(nb_slices,npoint)[sl]),label='Kdata grappa estimate')

plt.legend()



radial_traj_grappa = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                                  nb_slices=nb_slices, incoherent=incoherent, mode=mode)
radial_traj_grappa.traj = curr_traj_completed_all_ts


print("Building B1 Map Gappa....")
if str.split(filename_b1_grappa,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices_grappa=calculate_sensitivity_map_3D(kdata_all_channels_completed_all_ts.reshape(nb_channels,nb_segments,-1,npoint),radial_traj_grappa,res,image_size,useGPU=False,light_memory_usage=light_memory_usage,density_adj=True)
    np.save(filename_b1_grappa,b1_all_slices_grappa)
else:
    b1_all_slices_grappa=np.load(filename_b1_grappa)


sl=int(b1_all_slices_all_spokes.shape[1]/2)
sl=2

list_images = list(np.abs(b1_all_slices_grappa[:,sl,:,:]))
plot_image_grid(list_images,(3,3),title="Grappa : Sensitivity map for slice {}".format(sl))

list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
plot_image_grid(list_images,(3,3),title="Undersampled : Sensitivity map for slice {}".format(sl))


list_images = list(np.abs(b1_all_slices_all_spokes[:,sl,:,:]))
plot_image_grid(list_images,(3,3),title="All Spokes : Sensitivity map for slice {}".format(sl))

list_images = list(np.abs(b1_maps[:,sl,:,:]))
plot_image_grid(list_images,(3,3),title="Ground Truth : Sensitivity map for slice {}".format(sl))

print("Building Volumes Grappa....")
if str.split(filename_volume_grappa, "/")[-1] not in os.listdir(folder):
        #kdata_all_channels_all_slices = np.load(filename_kdata)
    del kdata_all_channels_completed_all_ts
    kdata_all_channels_completed_all_ts=np.load(filename_kdata_grappa)
    volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels_completed_all_ts, radial_traj_grappa, image_size,
                                                                b1=b1_all_slices_grappa, density_adj=True,
                                                                ntimesteps=ntimesteps, useGPU=False,
                                                                normalize_kdata=False, memmap_file=None,
                                                                light_memory_usage=light_memory_usage,
                                                                normalize_volumes=True)
    np.save(filename_volume_grappa, volumes_all)
    sl=int(nb_slices/2)
    ani = animate_images(volumes_all[:,sl,:,:])
    del volumes_all

print("Building Mask Grappa....")
if str.split(filename_mask_grappa, "/")[-1] not in os.listdir(folder):
    del kdata_all_channels_completed_all_ts
    np.load(filename_kdata_grappa)
    kdata_all_channels_completed_all_ts = np.load(filename_kdata_grappa)

    selected_spokes = None
    mask = build_mask_single_image_multichannel(kdata_all_channels_completed_all_ts, radial_traj_grappa, image_size,
                                                    b1=b1_all_slices, density_adj=True, threshold_factor=1 / 25,
                                                    normalize_kdata=True, light_memory_usage=True,
                                                    selected_spokes=selected_spokes)
    np.save(filename_mask_grappa, mask)
    #animate_images(mask)
    del mask




    #Replace calibration lines

    # curr_traj_completed_replace_calib=np.concatenate([curr_traj_for_grappa,curr_traj_estimate],axis=0)
    # kdata_all_channels_completed_replace_calib=np.concatenate([kdata_all_channels_for_grappa,F_target_estimate],axis=1)
    #
    # for index_ky_target in range(len(calib_lines)):
    #     for j in range(npoint_x):
    #         kx=traj_calib_for_grappa[index_ky_target,j,0]
    #         ky = traj_calib_for_grappa[index_ky_target,j,1]
    #         j0, i0 = np.unravel_index(
    #             np.argmin(np.linalg.norm(curr_traj_completed_replace_calib - np.expand_dims(np.array([kx, ky]), axis=(0, 1)), axis=-1)),
    #             curr_traj_completed_replace_calib.shape[:-1])
    #
    #         kdata_all_channels_completed_replace_calib[:,j0,i0] =kdata_calib_all_channels[:,index_ky_target,j]
    #         curr_traj_completed_replace_calib[j0,i0]=np.array([kx,ky])
    #
    # kdata_all_channels_completed_replace_calib=kdata_all_channels_completed_replace_calib.reshape((nb_channels,-1))
    # curr_traj_completed_replace_calib=curr_traj_completed_replace_calib.reshape((-1,2))
    #
    # image_rebuilt_all_channels_completed_replace_calib=[finufft.nufft2d1(curr_traj_completed_replace_calib[:,0], curr_traj_completed_replace_calib[:,1], k, image_size) for k in kdata_all_channels_completed_replace_calib]
    # image_rebuilt_completed_replace_calib=np.sqrt(np.sum(np.abs(image_rebuilt_all_channels_completed_replace_calib) ** 2, axis=0))
    #
    # plt.figure()
    # plt.imshow(np.abs(image_rebuilt_completed_replace_calib))
    # plt.title("With GRAPPA replace calib kernel_x {} kernel_y {}".format(kernel_x,kernel_y))












########################## Dict mapping ########################################


load_map=False
save_map=True


mask = np.load(filename_mask)
mask=m.mask
volumes_all = np.load(filename_volume)
volumes_all_grappa=np.load(filename_volume_grappa)
volumes_all_spokes=np.load(filename_volume_all_spokes)


ts=0
file_mha = "/".join(["/".join(str.split(filename_volume_grappa,"/")[:-1]),"_".join(str.split(str.split(filename_volume_grappa,"/")[-1],".")[:-1])]) + "_ts{}.mha".format(ts)
io.write(file_mha,np.abs(volumes_all_grappa[ts]),tags={"spacing":[5,1,1]})
file_mha = "/".join(["/".join(str.split(filename_volume,"/")[:-1]),"_".join(str.split(str.split(filename_volume,"/")[-1],".")[:-1])]) + "_ts{}.mha".format(ts)
io.write(file_mha,np.abs(volumes_all[ts]),tags={"spacing":[5,1,1]})
file_mha = "/".join(["/".join(str.split(filename_volume_all_spokes,"/")[:-1]),"_".join(str.split(str.split(filename_volume_all_spokes,"/")[-1],".")[:-1])]) + "_ts{}.mha".format(ts)
io.write(file_mha,np.abs(volumes_all_spokes[ts]),tags={"spacing":[5,1,1]})


#animate_images(mask)


#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])



if not(load_map):
    niter = 0
    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=radial_traj,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=False,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps,threshold=None)
    all_maps=optimizer.search_patterns(dictfile,volumes_all,retained_timesteps=None)

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
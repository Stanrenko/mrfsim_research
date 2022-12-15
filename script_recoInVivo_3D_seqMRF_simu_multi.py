

#import matplotlib
#matplotlib.u<se("TkAgg")
import numpy as np
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,GaussianWeighting,BruteDictSearch
from utils_mrf import *
from utils_simu import *
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
from epgpy import epg
from scipy.signal import medfilt
base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"

suffix_simu=""
#dictfile = "mrf175_SimReco2_light.dict"
#dictjson="mrf_dictconf_SimReco2_light_df0.json"
dictjson="mrf_dictconf_SimReco2.json".format(suffix_simu)

dictfile_light='./mrf175_SimReco2_light_matching_adjusted.dict'

#dictfile = "mrf175_SimReco2.dict"
#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
#dictfile = "mrf175_Dico2_Invivo.dict"

suffix=""
#suffix="_FFDf"
#suffix="_Cohen"
#suffix="_CohenWeighted"
#suffix="_CohenCSWeighted"
#suffix="_CohenBS"
#suffix="_PW"
#suffix="_PWCR"
#suffix="_PWMag"

#suffix="_PWWeighted"
#suffix="_"

use_GPU = True
light_memory_usage=True
gen_mode="other"
medfilter=False
#suffix="_plateau600"
#suffix="_constantTE_last"
#suffix=""

# with open("mrf{}_SeqFF{}_config.json".format(nb_allspokes,suffix)) as f:
#     sequence_config = json.load(f)
#with open("./mrf_sequence_adjusted.json") as f:
#    sequence_config = json.load(f)

#name = "SquareSimu3D_SS_FF0_1"
name = "SquareSimu3D_SS_Multicoil"
name = "Knee3D_Control_SS_SimReco2_MultiCoil"
snr=None
gauss_filter=False

#dictfile="mrf_dictconf_SimReco2_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_reco3.dict"

#dictfile="./mrf_dictconf_SimReco2_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_reco3.95_w8_simmean.dict"
dictfile="./mrf_dictconf_SimReco2_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v3_reco4_w8_simmean.dict"
suffix="_DE_Simu_FF_random_v3_reco4"
with open("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v3.json") as f:
    sequence_config = json.load(f)

suffix_file_map=""


#dictfile="./mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v5_reco4_w8_simmean.dict"
#suffix="_DE_Simu_random_v5_reco4"
#with open("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v5.json") as f:
#    sequence_config = json.load(f)
#generate_epg_dico_T1MRFSS_from_sequence_file("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF.json","./mrf_dictconf_SimReco2_lightDFB1.json",3)
# dictfile="mrf_dictconf_SimReco2_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_v3_reco3.6.dict"
# dictfile="mrf_dictconf_SimReco2_lightDFB1_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_v3_reco3.6.dict"
# suffix="_DE_Simu_FF_v3_reco3.6"
# with open("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_v3.json") as f:
#    sequence_config = json.load(f)


#dictfile="mrf_dictconf_SimReco2_adjusted_760_reco3_w8_simmean.dict"
# dictfile="mrf_dictconf_SimReco2_lightDFB1_adjusted_760_reco3_w8_simmean.dict"
#suffix="_old_760_reco3"
#with open("./mrf_sequence_adjusted_760.json") as f:
#    sequence_config = json.load(f)


dictfile="mrf_SimReco2_light_adjusted.dict"
suffix=""
with open("./mrf_sequence_adjusted.json") as f:
    sequence_config = json.load(f)
#
# dictfile="mrf175_SimReco2_adjusted.dict"
# suffix="_fullReco"
# with open("./mrf_sequence_adjusted.json") as f:
#     sequence_config = json.load(f)


if gauss_filter:
    suffix += "_gaussfilter"

if snr is not None:
    suffix+="_SNR_{}".format(snr)

nb_allspokes = len(sequence_config["TE"])
nspoke=8
nb_segments=nb_allspokes
ntimesteps=int(nb_segments/nspoke)
# with open("./mrf_sequence_adjusted.json") as f:
#     sequence_config_base = json.load(f)
#
# plt.close("all")
# plt.figure()
# plt.plot(sequence_config["TE"])
# plt.plot(sequence_config_base["TE"])
# plt.plot(sequence_config["TR"])
# plt.plot(sequence_config_base["TR"])
#
# plt.figure()
# plt.plot(sequence_config["B1"])
# plt.plot(sequence_config_base["B1"])
#
# print(sequence_config["FA"])
# print(sequence_config_base["FA"])


#with open("mrf_dictconf_SimReco2_light.json") as f:
#    dict_config = json.load(f)
#

# unique_TEs=np.unique(sequence_config["TE"])
# unique_TRs=np.unique(sequence_config["TR"])
# FA=20
#
#
# nTR=60
# nTR=int(nTR/3)*3
#
#
# params_0=int(nTR/3)*[unique_TEs[0]]+int(nTR/3)*[unique_TEs[1]]+int(nTR/3)*[unique_TEs[2]]
# nTR=len(params_0)
# params_0=[FA]+params_0


#seq = FFMRF(**sequence_config)

L0 = 2
nrep=2
rep=nrep-1
TR_total = np.sum(sequence_config["TR"])

Treco = TR_total-np.sum(sequence_config["TR"])
Treco=4000
##other options
#sequence_config["T_recovery"]=Treco
#sequence_config["nrep"]=nrep
#sequence_config["rep"]=rep

#seq=T1MRFSS(**sequence_config)

seq=T1MRF(**sequence_config)

nb_filled_slices = 16
nb_empty_slices = 2
repeat_slice = 1

if "Knee3D" in name:
    repeat_slice=1
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

is_random=False
frac_center=1.0


dens_adj=True
use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

folder = "/".join(str.split(filename,"/")[:-1])

nb_channels=8

filename_paramMap=filename+"_paramMap_sl{}_rp{}{}.pkl".format(nb_slices,repeat_slice,"")

filename_paramMask=filename+"_paramMask_sl{}_rp{}.npy".format(nb_slices,repeat_slice)

filename_volume = filename+"_volumes_sl{}_rp{}_us{}_{}w{}_ch{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,nb_channels,suffix)
filename_volume_no_dens_adj = filename+"_volumes_no_dens_adj_sl{}_rp{}_us{}_{}w{}_ch{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,nb_channels,suffix)

filename_volume_singular = filename+"_volumes_singular_sl{}_rp{}_us{}_{}w{}_ch{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,nb_channels,suffix)

filename_groundtruth = filename+"_groundtruth_volumes_sl{}_rp{}_{}w{}{}.npy".format(nb_slices,repeat_slice,nb_allspokes,nspoke,suffix)

filename_kdata = filename+"_kdata_sl{}_rp{}_us{}{}w{}_ch{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,nb_channels,suffix)
filename_kdata_no_noise = filename+"_kdata_no_noise_sl{}_rp{}_us{}{}w{}_ch{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,nb_channels,suffix)

filename_b1 = filename+"_b1_sl{}_rp{}_us{}{}w{}_ch{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,nb_channels,suffix)

filename_mask= filename+"_mask_sl{}_rp{}_us{}{}w{}_ch{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,nb_channels,suffix)
file_map = filename + "_sl{}_rp{}_us{}{}w{}_ch{}{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,nb_channels,suffix,suffix_file_map)
filename_phi=str.split(dictfile,".dict") [0]+"_phi_L0_{}.npy".format(L0)

#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"

nb_channels=8
npoint = 512



incoherent=True
mode="old"

image_size = (nb_slices, int(npoint/2), int(npoint/2))

size = image_size[1:]

with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)
#dict_config["ff"]=np.array([0.1])
#dict_config["delta_freqs"]=[0.0]

if "SquareSimu3D" in name:

    region_size=4 #size of the regions with uniform values for params in pixel number (square regions)
    mask_reduction_factor=1/4
    m = RandomMap3D(name,dict_config,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,undersampling_factor=undersampling_factor,repeat_slice=repeat_slice,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)

elif "Knee" in name:
     num =1
     file_matlab_paramMap = "./data/KneePhantom/Phantom{}/paramMap_Control.mat".format(num)
     m = MapFromFile3D(name,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,image_size=image_size,file=file_matlab_paramMap,rounding=True,gen_mode="other",undersampling_factor=undersampling_factor,resting_time=4000)

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

if str.split(filename_kdata, "/")[-1] not in os.listdir(folder):
    m.build_ref_images(seq)

#if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
#    np.save(filename_groundtruth,m.images_series[::nspoke])



#animate_images(m.images_series[::nspoke,int(nb_slices/2)])

#plt.figure()
#plt.plot(m.images_series[::nspoke,int(nb_slices/2),int(npoint/4),int(npoint/4)])

#animate_images(m.images_series[::nspoke,int(nb_slices/2)])
# i=0
# image=m.images_series[i]


# file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
#                              "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_ideal_ts{}.mha".format(
#                                                                                                                    i)
# io.write(file_mha, np.abs(image), tags={"spacing": [5, 1, 1]})


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,is_random=is_random,frac_center=frac_center,nspoke_per_z_encoding=nspoke)


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

#animate_images(b1_maps[0,:,:,:])

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

    if snr is not None:
        center_point=int(npoint/2)
        center_sl = int(nb_slices/2/undersampling_factor)
        res=int(npoint/8)
        res_sl=np.maximum(int(nb_slices/undersampling_factor/8),1)
        mean_data=np.mean(np.abs(data[:,:,(center_sl-res_sl):(center_sl+res_sl),(center_point-res):(center_point+res)]))
        noise = mean_data/snr*(np.random.normal(size=data.shape)+1j*np.random.normal(size=data.shape))
        #noise=0
        data_no_noise=copy(data)
        np.save(filename_kdata_no_noise,data_no_noise)
        data+=noise
    np.save(filename_kdata, data)
#del data
    kdata_all_channels_all_slices = np.load(filename_kdata)


else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)
    if snr is not None:
        data_no_noise = np.load(filename_kdata_no_noise)


#
# center_sl=int(nb_slices/2)
# center_point=int(npoint/2)
# res=4
# res_sl=2
# border=4
# border_sl=2
#
#
# mean_signal=np.mean(np.abs(kdata_all_channels_all_slices[0,:,(center_sl-res_sl):(center_sl+res_sl),(center_point-res):(center_point+res)]))
#
# std_signal=np.std(np.abs(kdata_all_channels_all_slices[0,:,:,np.r_[0:border,(npoint-border):npoint]][:,:,np.r_[0:border_sl,(nb_slices-border_sl):nb_slices]]))
#
# mean_signal/std_signal
#
# kdata_reshaped=kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
# mean_signal_by_ts=np.mean(np.abs(kdata_reshaped[0,:,:,(center_sl-res_sl):(center_sl+res_sl),(center_point-res):(center_point+res)]).reshape(ntimesteps,-1),axis=-1)
#
# std_signal_by_ts=np.std(np.abs(kdata_reshaped[0,:,:,:,np.r_[0:border,(npoint-border):npoint]][:,:,:,np.r_[0:border_sl,(nb_slices-border_sl):nb_slices]]),axis=(0,2,3))
#
# plt.figure();plt.plot(mean_signal_by_ts/std_signal_by_ts),plt.title("SNR by timestep")

# traj=radial_traj.get_traj()
#
#
# kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,nspoke,nb_slices,npoint)
# data_no_noise=data_no_noise.reshape(nb_channels,ntimesteps,nspoke,nb_slices,npoint)
#
# ch=4
# ts=0
# ch=np.random.choice(range(nb_channels))
# sl=8
# curr_kdata=kdata_all_channels_all_slices[:,ts,:,sl,:]
# curr_kdata_no_noise=data_no_noise[ch,ts,:,sl,:]
#
# plt.figure()
# plt.plot(curr_kdata.T)
#
#
# plt.figure()
# plt.plot(curr_kdata_no_noise.T)
#
#
# sp=4
# ts=0
# ch=np.random.choice(range(nb_channels))
# sl=8
# curr_kdata=kdata_all_channels_all_slices[:,ts,sp,sl,:]
# curr_kdata_no_noise=data_no_noise[ch,ts,:,sl,:]
#
# plt.figure()
# plt.plot(curr_kdata.T)
#
#
# plt.figure()
# plt.plot(curr_kdata_no_noise.T)
#
# kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(nb_channels,nb_segments,-1)
# data_no_noise=data_no_noise.reshape(nb_channels,nb_segments,-1)
#
# ts=0
# curr_data=kdata_all_channels_all_slices[:,ts]
# curr_data_no_noise=data_no_noise[:,ts]
#
# u,s,vh=np.linalg.svd(curr_data)
# u_no_noise,s_no_noise,vh_no_noise=np.linalg.svd(curr_data_no_noise)
#
# plt.figure();plt.plot(np.cumsum(s)/np.sum(s));plt.plot(np.cumsum(s_no_noise)/np.sum(s_no_noise))
#
#
# keys,D=read_mrf_dict(dictfile,FF_list=np.arange(0.,1.01,0.05),aggregate_components=True)
# D_plus=np.linalg.pinv(D)
#
# ch=0
# curr_kdata=kdata_all_channels_all_slices[0].reshape(ntimesteps,-1)
# curr_kdata_no_noise=data_no_noise[0].reshape(ntimesteps,-1)
#
#
#
# def filter_in_k_space(curr_kdata_sp,res=20):
#     center_point=int(curr_kdata.shape[0]/2)
#     F_curr_kdata = finufft.nufft1d2(np.arange(-np.pi, np.pi, 2 * np.pi / npoint) + np.pi / (npoint), curr_kdata[sp])
#     F_curr_kdata[:center_point - res] = 0
#     F_curr_kdata[center_point + res:] = 0
#
#     curr_kdata_corrected = finufft.nufft1d2(np.arange(-np.pi, np.pi, 2 * np.pi / npoint) + np.pi / (npoint),
#                                             F_curr_kdata, isign=1) / npoint
#
#     return curr_kdata_corrected
#
#
#
# if filename_phi not in os.listdir():
#     mrfdict = dictsearch.Dictionary()
#     keys,values=read_mrf_dict(dictfile,np.arange(0.,1.01,0.05))
#
#     import dask.array as da
#     u,s,vh = da.linalg.svd(da.asarray(values))
#
#     vh=np.array(vh)
#     s=np.array(s)
#
#     phi=vh[:L0]
#     np.save(filename_phi,phi)
# else:
#     phi=np.load(filename_phi)
#
# traj=radial_traj.get_traj().reshape(ntimesteps,-1,3)
# kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,-1)
#
# kdata_singular=np.zeros((nb_channels,)+traj.shape[:-1]+(L0,),dtype=kdata_all_channels_all_slices.dtype)
# for ts in tqdm(range(ntimesteps)):
#     kdata_singular[:,ts,:,:]=kdata_all_channels_all_slices[:,ts,:,None]@(phi.conj().T[ts][None,:])
#
# kdata_singular=np.moveaxis(kdata_singular,-1,1)
#
# kdata_singular=kdata_singular.reshape(nb_channels,L0,nb_segments,nb_slices,npoint)
#
#
# data_no_noise=data_no_noise.reshape(nb_channels,ntimesteps,-1)
#
# kdata_no_noise_singular=np.zeros((nb_channels,)+traj.shape[:-1]+(L0,),dtype=kdata_all_channels_all_slices.dtype)
# for ts in tqdm(range(ntimesteps)):
#     kdata_no_noise_singular[:,ts,:,:]=data_no_noise[:,ts,:,None]@(phi.conj().T[ts][None,:])
#
# kdata_no_noise_singular=np.moveaxis(kdata_no_noise_singular,-1,1)
#
# kdata_no_noise_singular=kdata_no_noise_singular.reshape(nb_channels,L0,nb_segments,nb_slices,npoint)
#
#
# ch=0
# seg=np.random.choice(range(nb_segments))
# sl=8
#
# plt.figure()
# plt.plot(kdata_singular[ch,0,seg,sl,:])
# plt.plot(kdata_no_noise_singular[ch,0,seg,sl,:])
#
#
# print("Building Volumes....")
# if str.split(filename_volume_singular,"/")[-1] not in os.listdir(folder):
#     volumes_singular=simulate_radial_undersampled_singular_images_multi(kdata_singular,radial_traj,image_size,density_adj=True,b1=b1_all_slices,ntimesteps=L0,light_memory_usage=True)
#     np.save(filename_volume_singular,volumes_singular)
#     # sl=20
#     # ani = animate_images(volumes_singular[0,:,:,:])
#     #del volumes_singular
#
# volumes_singular_no_noise = simulate_radial_undersampled_singular_images_multi(kdata_no_noise_singular, radial_traj, image_size,
#                                                                       density_adj=True, b1=b1_all_slices, ntimesteps=L0,
#                                                                       light_memory_usage=True)
#
#
# animate_multiple_images(volumes_singular_no_noise[0],volumes_singular[0])
#
# sigma=2*np.pi/(npoint)
# sigma_z=2*np.pi/(2*nb_slices)
# sigmas=np.array([sigma]*2+[sigma_z])
#
# ts=0
# curr_kdata_all_channels_all_slices=kdata_all_channels_all_slices[:,ts]
# curr_kdata_all_channels_all_slices=curr_kdata_all_channels_all_slices.reshape(nb_channels,-1)
# curr_traj=traj[ts]
#
# d_ki_kj=np.linalg.norm((np.expand_dims(curr_traj,axis=0)-np.expand_dims(curr_traj,axis=1))/(np.sqrt(2)*(np.expand_dims(sigmas,axis=(0,1)))),axis=-1)**2
# curr_kdata_all_channels_all_slices=curr_kdata_all_channels_all_slices.reshape(-1,npoint)
# density = np.abs(np.linspace(-1, 1, npoint))
# density = np.expand_dims(density, tuple(range(curr_kdata_all_channels_all_slices.ndim - 1)))
# curr_kdata_all_channels_all_slices *=density
# curr_kdata_all_channels_all_slices=curr_kdata_all_channels_all_slices.reshape(nb_channels,-1)
# sigma=1
#
# gaussian_kernel_matrix=np.exp(-d_ki_kj)
# filtered_curr_kdata=np.einsum("ij,lj->li",gaussian_kernel_matrix,curr_kdata_all_channels_all_slices)
# filtered_curr_kdata=filtered_curr_kdata.reshape(nb_channels,-1,npoint)
# curr_data_no_noise=data_no_noise[:,ts,:,:]
# curr_data_no_noise*=np.expand_dims(density,axis=0)
#
# sl=10
# ch=np.random.choice(range(nb_channels))
# plt.figure()
# plt.plot(curr_data_no_noise[ch,sl,:])
# plt.plot(filtered_curr_kdata[ch,sl,:])
#
#
# from scipy.ndimage import gaussian_filter1d
# ts=np.random.choice(range(nb_segments))
# sl=0
# ch=np.random.choice(range(nb_channels))
# input=kdata_all_channels_all_slices[ch,ts,sl,:]
# output=gaussian_filter1d(input,1.5)
# output[center_point-16:center_point+16]=input[center_point-16:center_point+16]
# input_ref=data_no_noise[ch,ts,sl,:]
# #
# plt.figure()
# plt.plot(input)
# plt.plot(output)
# plt.plot(input_ref)
#
#
# plt.figure()
# plt.plot(kdata_all_channels_all_slices[0,0,int(nb_slices/2),:])




# kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(-1,npoint)
# from scipy.ndimage import gaussian_filter1d
# input=kdata_all_channels_all_slices[0,0,int(nb_slices/2),:]
# output=gaussian_filter1d(input,1)
#
# plt.figure()
# plt.plot(np.imag(input))
# plt.plot(np.imag(output))



#
# if gauss_filter:
#     from scipy.ndimage import gaussian_filter1d
#     kdata_all_channels_all_slices = kdata_all_channels_all_slices.reshape(-1, npoint)
#     data_filtered = np.apply_along_axis(lambda x: filter_in_k_space(x, 20), -1,kdata_all_channels_all_slices)
#
#     data_filtered[:,(center_point-16):(center_point+16)]=kdata_all_channels_all_slices[:,center_point-16:center_point+16]
#     data_filtered = data_filtered.reshape(nb_channels, nb_segments, nb_slices, -1)
#
#     kdata_all_channels_all_slices=data_filtered
#     np.save(filename_kdata, kdata_all_channels_all_slices)
#
# plt.figure()
# plt.plot(kdata_all_channels_all_slices[0,0,int(nb_slices/2),:])
#

if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage,density_adj=True)
    np.save(filename_b1,b1_all_slices)
else:
    b1_all_slices=np.load(filename_b1)

sl=int(b1_all_slices.shape[1]/2)
list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
plot_image_grid(list_images,(3,3),title="Sensitivity map for slice {}".format(sl))

# #del kdata_all_channels_all_slices
# kdata_all_channels_all_slices=np.load(filename_kdata)
# volume_full=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_iterative=True)
#
# animate_images(volume_full[0])


print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    del kdata_all_channels_all_slices
    kdata_all_channels_all_slices = np.load(filename_kdata)

    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_iterative=True)
    np.save(filename_volume,volumes_all)
    # sl=10
    # ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
    del volumes_all



##volumes for slice taking into account coil sensi
print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    np.save(filename_mask,m.mask)

#volumes_all=np.load(filename_volume)
#ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])


########################## Dict mapping ########################################
#
# with open("mrf_sequence{}.json".format(suffix)) as f:
#      sequence_config = json.load(f)


#seq = None


load_map=False
save_map=True

#dictfile="mrf175_SimReco2_light_adjusted_NoReco.dict"
#dictfile="mrf175_SimReco2_light.dict"

#dictfile="mrf175_SimReco2_light_adjusted_M0_T1_filter_DFFFTR_2.dict"
#dictfile="mrf175_SimReco2_light.dict"
#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
#dictfile = "mrf175_Dico2_Invivo.dict"

#mask = m.mask

mask=np.load(filename_mask)
if dens_adj:
    volumes_all = np.load(filename_volume)
else:
    volumes_all = np.load(filename_volume_no_dens_adj)

#file_map=str.split(file_map,".pkl")[0]+"_test.pkl"

if not(load_map):
    niter = 0
    #optimizer = BruteDictSearch(FF_list=np.arange(0,1.01,0.05),mask=mask,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=False,ntimesteps=ntimesteps,log_phase=True)
    #all_maps = optimizer.search_patterns(dictfile, volumes_all, retained_timesteps=None)


    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=10, pca=True,threshold_pca=10, log=False, useGPU_dictsearch=False, useGPU_simulation=False,gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,b1=b1_all_slices,mu="Adaptative",dens_adj=dens_adj,dictfile_light=dictfile_light,threshold_ff=0.9)#,kdata_init=kdata_all_channels_all_slices)#,mu_TV=0.5)#,kdata_init=data_no_noise)
    all_maps=optimizer.search_patterns_test_multi_2_steps_dico(dictfile,volumes_all,retained_timesteps=None)

    if(save_map):
        import pickle

        #file_map = filename.split(".dat")[0] + "_corrected_dens_adj{}_MRF_map.pkl".format(suffix)
        #file_map = filename.split(".dat")[0] + "_5iter_MRF_map.pkl".format("")
        file = open(file_map, "wb")
        # dump information to that file
        pickle.dump(all_maps, file)
        # close the file
        file.close()

else:
    import pickle
    #file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
    file = open(file_map, "rb")
    all_maps = pickle.load(file)
    file.close()

niter=len(all_maps.keys())
plt.close("all")
maskROI = buildROImask_unique(m.paramMap)
niter=0
for it in range(niter+1):

    regression_paramMaps_ROI(m.paramMap, all_maps[it][0], m.mask > 0, all_maps[it][1] > 0, maskROI, adj_wT1=True,
                             title="it{}_regROI_".format(it) + "_".join(str.split(str.split(str.split(filename_volume, "/")[-1], ".npy")[0],".")), save=True,fontsize_axis=10,kept_keys=["wT1","ff"],marker_size=2)

#regression_paramMaps(m.paramMap,all_maps[0][0],mode="Boxplot")


curr_file=file_map
file = open(curr_file, "rb")
all_maps = pickle.load(file)
file.close()
for iter in list(range(len(all_maps.keys()))):

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


plt.close("all")



name="SquareSimu3D_SS_Multicoil"
name="Knee3D_Control_SS_SimReco2_MultiCoil"
#name="SquareSimu3D_SS_FF0_1"

list_suffix=["fullReco_T1MRF_adjusted","fullReco_Brute","DE_Simu_FF_reco3","DE_Simu_FF_v2_reco3"]
list_suffix=["DE_Simu_FF_reco3_SNR_{}".format(snr)]
list_suffix=["DE_Simu_FF_reco3.9".format(snr)]
list_suffix=["fullReco_SNR_{}".format(snr),"DE_Simu_FF_reco3.9_SNR_{}".format(snr)]
list_suffix=["fullReco_SNR_{}".format(snr),"DE_Simu_FF_reco3.9_SNR_{}".format(snr),"DE_Simu_FF_v2_reco1.55_SNR_{}".format(snr)]
list_suffix=["fullReco".format(snr),"DE_Simu_FF_reco3.9".format(snr),"DE_Simu_FF_v2_reco1.55".format(snr),"DE_Simu_FF_v2_reco4".format(snr)]

list_suffix=["DE_Simu_FF_reco3_SNR_{}".format(snr)]

if snr is not None:
    list_suffix=["DE_Simu_FF_reco3_SNR_{}".format(snr)]
else:
    list_suffix=["DE_Simu_FF_reco3".format(snr)]
list_suffix=["fullReco".format(snr),"DE_Simu_FF_reco3".format(snr),"DE_Simu_FF_v6_reco3.5".format(snr)]
list_suffix=["fullReco".format(snr),"DE_Simu_FF_reco3","old_760_reco3"]
list_suffix=["fullReco".format(snr),"DE_Simu_FF_reco3","old_760_reco3","DE_Simu_FF_v6_reco3.8"]
list_suffix=["fullReco".format(snr),"DE_Simu_FF_reco3","DE_Simu_FF_v6_reco3.8","DE_Simu_FF_random_v4_reco3.9","DE_Simu_FF_random_v5_reco4","DE_Simu_FF_random_v2_reco4","DE_Simu_FF_random_FA_v2_reco4"]
list_suffix=["fullReco","old_760_reco3","DE_Simu_random_v5_reco4","DE_Simu_FF_random_v1_reco3_95"]
list_suffix=["DE_Simu_FF_random_v1_reco3_95","DE_Simu_FF_random_v3_reco4"]

#list_suffix=["fullReco","DE_Simu_FF_v3_reco3.6".format(snr)]
undersampling_factor=1
#list_suffix=["fullReco_SNR_{}".format(snr)]
dic_maps={}
for suffix in list_suffix:

    if ("random" in suffix)or("old" in suffix):
        file_map = "/{}_sl{}_rp{}_us{}{}w{}_ch{}_{}_MRF_map.pkl".format(name, nb_slices, repeat_slice, undersampling_factor,
                                                                   760, nspoke,nb_channels, suffix)

    elif ("DE_Simu_FF" in suffix) or ("old" in suffix):
        file_map = "/{}_sl{}_rp{}_us{}{}w{}_ch{}_{}_MRF_map.pkl".format(name, nb_slices, repeat_slice, undersampling_factor,
                                                                   760, nspoke,nb_channels, suffix)


    else:
        file_map="/{}_sl{}_rp{}_us{}{}w{}_ch{}_{}_MRF_map.pkl".format(name,nb_slices,repeat_slice,undersampling_factor,1400,nspoke,nb_channels,suffix)

    print(file_map)
    with open(base_folder + file_map, "rb") as file:
        dic_maps[file_map] = pickle.load(file)
    #dic_maps[file_map] = all_maps
file_map="/{}_sl{}_rp{}_us{}{}w{}_ch{}_{}_MRF_map.pkl".format(name,nb_slices,repeat_slice,1,1400,nspoke,nb_channels,suffix)
with open(base_folder + file_map, "rb") as file:
    dic_maps["fullReco_noUS"] = pickle.load(file)
min_iter=0
max_iter=min_iter+1
k="wT1"
maskROI=buildROImask_unique(m.paramMap,key=k)
fig,ax=plt.subplots(1,2)
plt.title(k)



for i,key in enumerate(dic_maps.keys()):
    for it in (list(range(min_iter,np.minimum(len(dic_maps[key].keys()),max_iter),3))):
        if key=="fullReco_noUS" and it>0:
            continue
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)
        #dic_roi_values[key]=roi_values

        ax[0].plot(roi_values["Obs Mean"], roi_values["Pred Mean"].values,
                       label=key + " Iteration {}".format(it))
        ax[1].plot(roi_values["Obs Mean"], roi_values["Pred Std"].values,
                       label=key + " Iteration {} value {}".format(it,np.mean(roi_values["Pred Std"].values)))

ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],linestyle="--")
plt.legend()

k="ff"
maskROI=buildROImask_unique(m.paramMap,key=k)
fig,ax=plt.subplots(1,2)
plt.title(k)
for i,key in enumerate(dic_maps.keys()):
    for it in (list(range(min_iter,np.minimum(len(dic_maps[key].keys()),max_iter),3))):
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=False,maskROI=maskROI)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)

        ax[0].plot(roi_values["Obs Mean"], roi_values["Pred Mean"].values,
                       label=key + " Iteration {}".format(it))
        ax[1].plot(roi_values["Obs Mean"], roi_values["Pred Std"].values,
                       label=key + " Iteration {} value {}".format(it,np.mean(roi_values["Pred Std"].values)))


ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],linestyle="--")

plt.legend()






min_iter=0
max_iter=1
labels=["MRF T1-FF","MRF T1-FF 760 spokes","Optim Gaussian Noise","Optim Undersampling Simu"]
labels=["v1","v3"]
maskROI=buildROImask_unique(m.paramMap)
#maskROI=m.buildROImask()

df_result=pd.DataFrame()

plt.figure()
k="ff"
for i,key in enumerate(dic_maps.keys()):
    for it in (list(range(min_iter,np.minimum(len(dic_maps[key].keys()),max_iter),3))):
        if key=="fullReco_noUS" and it>0:
            continue
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI,fat_threshold=0.7)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        #roi_values.sort_values(by=["Obs Mean"],inplace=True)
        error=list((roi_values["Pred Mean"]-roi_values["Obs Mean"]))
        if df_result.empty:
            df_result=pd.DataFrame(data=error,columns=[labels[i] + " Iteration {}".format(it)])
        else:
            df_result[labels[i] + " Iteration {}".format(it)]=error

columns=[df_result.columns[-1]]+list(df_result.columns[:-1])
df_result=df_result[columns]
df_result.boxplot(grid=False, rot=45, fontsize=10,showfliers=False)
plt.axhline(y=0,linestyle="dashed",color="k",linewidth=0.5)

k="ff"
#maskROI=buildROImask_unique(m.paramMap,key=k)
fig,ax=plt.subplots(1,2)
plt.title(k)
for i,key in enumerate(dic_maps.keys()):
    for it in (list(range(min_iter,np.minimum(len(dic_maps[key].keys()),max_iter),3))):
        if key=="fullReco_noUS" and it>0:
            continue
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=False,maskROI=maskROI)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)
        #dic_roi_values[key]=roi_values
        if key=="fullReco_noUS":
            ax[0].plot(roi_values["Obs Mean"],roi_values["Pred Mean"].values,label=labels[i]+" Iteration {}".format(it),linestyle="dashed",linewidth=2.0,color="k")
            ax[1].plot(roi_values["Obs Mean"],roi_values["Pred Std"].values,label=labels[i]+" Iteration {}".format(it),linestyle="dashed",linewidth=2.0,color="k")

        else:
            ax[0].plot(roi_values["Obs Mean"], roi_values["Pred Mean"].values,
                       label=labels[i] + " Iteration {}".format(it))
            ax[1].plot(roi_values["Obs Mean"], roi_values["Pred Std"].values,
                       label=labels[i] + " Iteration {}".format(it))


ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],linestyle="dotted")

plt.legend()


min_iter=0
max_iter=1
labels=["fullReco","Optim Gaussian Noise","Optim Undersampling Simu"]
maskROI=buildROImask_unique(m.paramMap)



k="wT1"
fig,ax=plt.subplots(1,2)
for i,key in enumerate(dic_maps.keys()):
    for it in (list(range(min_iter,np.minimum(len(dic_maps[key].keys()),max_iter),3))):
        if key=="fullReco_noUS" and it>0:
            continue
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI,fat_threshold=0.7)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)
        #dic_roi_values[key]=roi_values
        if key == "fullReco_noUS":
            ax[0].plot(roi_values["Obs Mean"], roi_values["Pred Mean"].values,
                       label=labels[i] + " Iteration {}".format(it),linestyle="dashed",linewidth=2.0,color="k")
            ax[1].plot(roi_values["Obs Mean"], roi_values["Pred Std"].values,
                       label=labels[i] + " Iteration {}".format(it),linestyle="dashed",linewidth=2.0,color="k")

        else:
            ax[0].plot(roi_values["Obs Mean"], roi_values["Pred Mean"].values,
                       label=labels[i] + " Iteration {}".format(it))
            ax[1].plot(roi_values["Obs Mean"], roi_values["Pred Std"].values,
                       label=labels[i] + " Iteration {}".format(it))

ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],linestyle="dotted")
plt.legend()

k="ff"
#maskROI=buildROImask_unique(m.paramMap,key=k)
fig,ax=plt.subplots(1,2)
plt.title(k)
for i,key in enumerate(dic_maps.keys()):
    for it in (list(range(min_iter,np.minimum(len(dic_maps[key].keys()),max_iter),3))):
        if key=="fullReco_noUS" and it>0:
            continue
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=False,maskROI=maskROI)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)
        #dic_roi_values[key]=roi_values
        if key=="fullReco_noUS":
            ax[0].plot(roi_values["Obs Mean"],roi_values["Pred Mean"].values,label=labels[i]+" Iteration {}".format(it),linestyle="dashed",linewidth=2.0,color="k")
            ax[1].plot(roi_values["Obs Mean"],roi_values["Pred Std"].values,label=labels[i]+" Iteration {}".format(it),linestyle="dashed",linewidth=2.0,color="k")

        else:
            ax[0].plot(roi_values["Obs Mean"], roi_values["Pred Mean"].values,
                       label=labels[i] + " Iteration {}".format(it))
            ax[1].plot(roi_values["Obs Mean"], roi_values["Pred Std"].values,
                       label=labels[i] + " Iteration {}".format(it))


ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],linestyle="dotted")

plt.legend()



min_iter=0
max_iter=10
labels=["Undersampling kz x 4","No Undersampling"]
#maskROI=buildROImask_unique(m.paramMap,key=k)
maskROI=m.buildROImask()

k="wT1"
fig,ax=plt.subplots(1,2)
for i,key in enumerate(dic_maps.keys()):
    for it in (list(range(min_iter,np.minimum(len(dic_maps[key].keys()),max_iter),3))):
        if key=="fullReco_noUS" and it>0:
            continue
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI,fat_threshold=0.7)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)
        #dic_roi_values[key]=roi_values
        if key == "fullReco_noUS":
            ax[0].scatter(roi_values["Obs Mean"], roi_values["Pred Mean"].values,
                       label=labels[i] + " Iteration {}".format(it),marker="x",s=20,color="k",alpha=0.5)
            ax[1].scatter(roi_values["Obs Mean"], roi_values["Pred Std"].values,
                       label=labels[i] + " Iteration {}".format(it),marker="x",s=20,color="k",alpha=0.5)

        else:
            ax[0].scatter(roi_values["Obs Mean"], roi_values["Pred Mean"].values,
                       label=labels[i] + " Iteration {}".format(it),s=6)
            ax[1].scatter(roi_values["Obs Mean"], roi_values["Pred Std"].values,
                       label=labels[i] + " Iteration {}".format(it),s=6)

ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],color="k")
plt.legend()


k="ff"
fig,ax=plt.subplots(1,2)
for i,key in enumerate(dic_maps.keys()):
    for it in (list(range(min_iter,np.minimum(len(dic_maps[key].keys()),max_iter),3))):
        if key=="fullReco_noUS" and it>0:
            continue
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI,fat_threshold=0.7)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)
        #dic_roi_values[key]=roi_values
        if key == "fullReco_noUS":
            ax[0].scatter(roi_values["Obs Mean"], roi_values["Pred Mean"].values,
                       label=labels[i] + " Iteration {}".format(it),marker="x",s=10,color="k",alpha=0.5)
            ax[1].scatter(roi_values["Obs Mean"], roi_values["Pred Std"].values,
                       label=labels[i] + " Iteration {}".format(it),marker="x",s=10,color="k",alpha=0.5)

        else:
            ax[0].scatter(roi_values["Obs Mean"], roi_values["Pred Mean"].values,
                       label=labels[i] + " Iteration {}".format(it),s=3)
            ax[1].scatter(roi_values["Obs Mean"], roi_values["Pred Std"].values,
                       label=labels[i] + " Iteration {}".format(it),s=3)

ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],color="k")
plt.legend()


min_iter=0
max_iter=30
k="wT1"
maskROI=buildROImask_unique(m.paramMap,key=k)
fig,ax=plt.subplots(1,2)
plt.title(k)

for key in dic_maps.keys():
    for it in np.arange(0,30,5):
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)
        #dic_roi_values[key]=roi_values
        ax[0].plot(roi_values["Obs Mean"],roi_values["Pred Mean"].values,label=key+"_it{}".format(it))
        ax[1].plot(roi_values["Obs Mean"],roi_values["Pred Std"].values,label=key+"_it{}".format(it))


ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],linestyle="--")
plt.legend()

k="ff"
maskROI=buildROImask_unique(m.paramMap,key=k)
fig,ax=plt.subplots(1,2)
plt.title(k)
for key in dic_maps.keys():
    for it in np.arange(0,30,5):
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=False,maskROI=maskROI)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)
        #dic_roi_values[key]=roi_values
        ax[0].plot(roi_values["Obs Mean"],roi_values["Pred Mean"].values,label=key+"_it{}".format(it))
        ax[1].plot(roi_values["Obs Mean"],roi_values["Pred Std"].values,label=key+"_it{}".format(it))


ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],linestyle="--")

plt.legend()

k="attB1"
maskROI=buildROImask_unique(m.paramMap,key=k)
fig,ax=plt.subplots(1,2)
plt.title(k)
for key in dic_maps.keys():
    for it in (range(len(dic_maps[key].keys()))):
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=False,maskROI=maskROI)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)
        #dic_roi_values[key]=roi_values
        ax[0].plot(roi_values["Obs Mean"],roi_values["Pred Mean"].values,label=key+"_it{}".format(it))
        ax[1].plot(roi_values["Obs Mean"],roi_values["Pred Std"].values,label=key+"_it{}".format(it))


ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],linestyle="--")

plt.legend()

k="df"
maskROI=buildROImask_unique(m.paramMap,key=k)
fig,ax=plt.subplots(1,2)
plt.title(k)
for key in dic_maps.keys():
    for it in (range(len(dic_maps[key].keys()))):
        roi_values=get_ROI_values(m.paramMap,dic_maps[key][it][0],m.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=False,maskROI=maskROI)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
        roi_values.sort_values(by=["Obs Mean"],inplace=True)
        #dic_roi_values[key]=roi_values
        ax[0].plot(roi_values["Obs Mean"],roi_values["Pred Mean"].values,label=key+"_it{}".format(it))
        ax[1].plot(roi_values["Obs Mean"],roi_values["Pred Std"].values,label=key+"_it{}".format(it))


ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],linestyle="--")

plt.legend()

plt.close("all")



















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
from epgpy import epg

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"

suffix_simu=""
#dictfile = "mrf175_SimReco2_light.dict"
#dictjson="mrf_dictconf_SimReco2_light_df0.json"
dictjson="mrf_dictconf_SimReco2_light{}.json".format(suffix_simu)
#dictfile = "mrf175_SimReco2.dict"
#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
#dictfile = "mrf175_Dico2_Invivo.dict"

suffix=""
#suffix="_FFDf"
#suffix="_Cohen"
#suffix="_CohenWeighted"
#suffix="_CohenCSWeighted"
#suffix="_CohenBS"
#suffix="_PW"
#suffix="_PWCR"
#suffix="_PWMag"

#suffix="_PWWeighted"
#suffix="_"

nb_allspokes = 760
nspoke=8
nb_segments=nb_allspokes
ntimesteps=int(nb_segments/nspoke)

#suffix="_plateau600"
#suffix="_constantTE_last"
#suffix=""

# with open("mrf{}_SeqFF{}_config.json".format(nb_allspokes,suffix)) as f:
#     sequence_config = json.load(f)
#with open("./mrf_sequence_adjusted.json") as f:
#    sequence_config = json.load(f)


#name = "SquareSimu3D_SS_FF0_1"
name = "SquareSimu3D_SS_noise1"

dictfile="mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v1_reco3.4.dict"
suffix="_DE_Simu_FF_random_v1_reco3.4"
with open("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v1.json") as f:
    sequence_config = json.load(f)

# with open("./mrf_sequence_adjusted.json") as f:
#     sequence_config_base = json.load(f)
#
# plt.close("all")
# plt.figure()
# plt.plot(sequence_config["TE"])
# plt.plot(sequence_config_base["TE"])
# plt.plot(sequence_config["TR"])
# plt.plot(sequence_config_base["TR"])
#
# plt.figure()
# plt.plot(sequence_config["B1"])
# plt.plot(sequence_config_base["B1"])
#
# print(sequence_config["FA"])
# print(sequence_config_base["FA"])


#with open("mrf_dictconf_SimReco2_light.json") as f:
#    dict_config = json.load(f)
#

# unique_TEs=np.unique(sequence_config["TE"])
# unique_TRs=np.unique(sequence_config["TR"])
# FA=20
#
#
# nTR=60
# nTR=int(nTR/3)*3
#
#
# params_0=int(nTR/3)*[unique_TEs[0]]+int(nTR/3)*[unique_TEs[1]]+int(nTR/3)*[unique_TEs[2]]
# nTR=len(params_0)
# params_0=[FA]+params_0


#seq = FFMRF(**sequence_config)

nrep=2
rep=nrep-1
TR_total = np.sum(sequence_config["TR"])

Treco = TR_total-np.sum(sequence_config["TR"])
Treco=4000
##other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=nrep
sequence_config["rep"]=rep

seq=T1MRFSS(**sequence_config)

#seq=T1MRF(**sequence_config)




nb_filled_slices = 16
nb_empty_slices=2
repeat_slice=1
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

is_random=False
frac_center=1.0



use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

folder = "/".join(str.split(filename,"/")[:-1])



filename_paramMap=filename+"_paramMap_sl{}_rp{}{}.pkl".format(nb_slices,repeat_slice,"")

filename_paramMask=filename+"_paramMask_sl{}_rp{}.npy".format(nb_slices,repeat_slice)

filename_volume = filename+"_volumes_sl{}_rp{}_us{}_{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)
filename_groundtruth = filename+"_groundtruth_volumes_sl{}_rp{}_{}w{}{}.npy".format(nb_slices,repeat_slice,nb_allspokes,nspoke,suffix)

filename_kdata = filename+"_kdata_sl{}_rp{}_us{}{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)
filename_mask= filename+"_mask_sl{}_rp{}_us{}{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)
file_map = filename + "_sl{}_rp{}_us{}{}w{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)

#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"

nb_channels=1
npoint = 128



incoherent=True
mode="old"

image_size = (nb_slices, int(npoint/2), int(npoint/2))

size = image_size[1:]

with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)
#dict_config["ff"]=np.array([0.1])
#dict_config["delta_freqs"]=[0.0]

if "SquareSimu3D" in name:
    region_size=4 #size of the regions with uniform values for params in pixel number (square regions)
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



m.build_ref_images(seq)

if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
    np.save(filename_groundtruth,m.images_series[::nspoke])

animate_images(m.images_series[::nspoke,int(nb_slices/2)])
# i=0
# image=m.images_series[i]


# file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
#                              "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_ideal_ts{}.mha".format(
#                                                                                                                    i)
# io.write(file_mha, np.abs(image), tags={"spacing": [5, 1, 1]})


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,is_random=is_random,frac_center=frac_center,nspoke_per_z_encoding=nspoke)

nb_channels=1


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):

    #images = copy(m.images_series)
    data=m.generate_kdata(radial_traj,useGPU=use_GPU)

    data=np.array(data)
    np.save(filename_kdata, data)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    data = np.load(filename_kdata)

#plt.plot(np.unique(radial_traj.get_traj()[:,:,2],axis=-1),"s")

##volumes for slice taking into account coil sensi
print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    volumes_all=simulate_radial_undersampled_images(data,radial_traj,image_size,density_adj=True,useGPU=use_GPU,ntimesteps=ntimesteps)
    np.save(filename_volume,volumes_all)
    # sl=20
    # ani = animate_images(volumes_all[:,sl,:,:])
    del volumes_all

##volumes for slice taking into account coil sensi
print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    np.save(filename_mask,m.mask)


volumes_all=np.load(filename_volume)

volumes_ideal=[np.mean(gp, axis=0) for gp in groupby(m.images_series[:,m.mask>0], nspoke)]
volumes_ideal=np.array(volumes_ideal)

volumes_all=volumes_all[:,m.mask>0]


j=np.random.choice(volumes_all.shape[-1])
plt.figure()
plt.plot(volumes_ideal[:,j],label="Original")
plt.plot(volumes_all[:,j],label="With Artefact")
plt.legend()

all_errors=volumes_all-volumes_ideal
#
# plt.close("all")
# metric=np.real
# plt.figure()
# errors=metric(all_errors)
# plt.hist(errors,density=True)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# mu=np.mean(errors)
# std=np.std(errors)
# from scipy.stats import norm
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)

plt.close("all")

t=np.random.choice(all_errors.shape[0])

metric=np.real
plt.figure()
plt.title("Histogram of artefact errors for timestep {}".format(t))
errors=metric(all_errors[t])
plt.hist(errors,density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
mu=np.mean(errors)
std=np.std(errors)
from scipy.stats import norm
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)


means=[]
stds=[]
p_s=[]
from scipy.stats import normaltest
for t in tqdm(range(all_errors.shape[0])):
    errors_real = np.real(all_errors[t])
    errors_imag = np.imag(all_errors[t])
    mu_real = np.mean(errors_real)
    std_real = np.std(errors_real)
    mu_imag = np.mean(errors_imag)
    std_imag = np.std(errors_imag)

    means.append([mu_real,mu_imag])
    stds.append([std_real,std_imag])

    k2, p_real = normaltest(errors_real)
    k2, p_imag = normaltest(errors_imag)
    p_s.append([p_real, p_imag])


stds=np.array(stds)
means=np.array(means)
p_s=np.array(p_s)

plt.figure()
plt.plot(means)

plt.figure()
plt.plot(stds)
plt.plot(np.mean(np.abs(volumes_all),axis=-1))


plt.figure()
plt.plot(stds/(np.mean(np.abs(volumes_all),axis=-1)[:,None]))

plt.figure()
plt.plot(p_s)

print(np.sum(p_s[:,0]>0.05))
print(np.sum(p_s[:,1]>0.05))


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
from epgpy import epg

#base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"

suffix_simu=""
#dictfile = "mrf175_SimReco2_light.dict"
#dictjson="mrf_dictconf_SimReco2_light_df0.json"
dictjson="./mrf_dictconf_SimReco2_light{}.json".format(suffix_simu)
#dictfile = "mrf175_SimReco2.dict"
#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
#dictfile = "mrf175_Dico2_Invivo.dict"

suffix=""

nb_allspokes = 1400
nspoke=8
nb_segments=nb_allspokes
ntimesteps=int(nb_segments/nspoke)

#suffix="_plateau600"
#suffix="_constantTE_last"
#suffix=""

# with open("mrf{}_SeqFF{}_config.json".format(nb_allspokes,suffix)) as f:
#     sequence_config = json.load(f)
#with open("./mrf_sequence_adjusted.json") as f:
#    sequence_config = json.load(f)


#name = "SquareSimu3D_SS_FF0_1"
name = "SquareSimu3D_Patches"

dictfile="./mrf175_SimReco2_light_adjusted.dict"
suffix=""
with open("./mrf_sequence_adjusted.json") as f:
    sequence_config = json.load(f)


#nrep=2
#rep=nrep-1
#TR_total = np.sum(sequence_config["TR"])

#Treco = TR_total-np.sum(sequence_config["TR"])
#Treco=0
##other options
#sequence_config["T_recovery"]=Treco
#sequence_config["nrep"]=nrep
#sequence_config["rep"]=rep

#seq=T1MRFSS(**sequence_config)

seq=T1MRF(**sequence_config)




nb_filled_slices = 8
nb_empty_slices=2
repeat_slice=1
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

is_random=False
frac_center=1.0



use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile


folder = "/".join(str.split(filename,"/")[:-1])



filename_paramMap=filename+"_paramMap_sl{}_rp{}{}.pkl".format(nb_slices,repeat_slice,"")

filename_paramMask=filename+"_paramMask_sl{}_rp{}.npy".format(nb_slices,repeat_slice)

filename_volume = filename+"_volumes_sl{}_rp{}_us{}_{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)
filename_volume_oop = filename+"_volume_oop_sl{}_rp{}_us{}_{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)


filename_groundtruth = filename+"_groundtruth_volumes_sl{}_rp{}_{}w{}{}.npy".format(nb_slices,repeat_slice,nb_allspokes,nspoke,suffix)

filename_kdata = filename+"_kdata_sl{}_rp{}_us{}{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)
filename_mask= filename+"_mask_sl{}_rp{}_us{}{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)
file_map = filename + "_sl{}_rp{}_us{}{}w{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)

#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"

nb_channels=1
npoint = 128



incoherent=True
mode="old"

image_size = (nb_slices, int(npoint/2), int(npoint/2))

size = image_size[1:]

with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)
#dict_config["ff"]=np.array([0.1])
#dict_config["delta_freqs"]=[0.0]

if "SquareSimu3D" in name:
    region_size=4 #size of the regions with uniform values for params in pixel number (square regions)
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



m.build_ref_images(seq)

if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
    np.save(filename_groundtruth,m.images_series[::nspoke])

animate_images(m.images_series[::nspoke,int(nb_slices/2)])
# i=0
# image=m.images_series[i]


# file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
#                              "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_ideal_ts{}.mha".format(
#                                                                                                                    i)
# io.write(file_mha, np.abs(image), tags={"spacing": [5, 1, 1]})


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,is_random=is_random,frac_center=frac_center,nspoke_per_z_encoding=nspoke)

nb_channels=1


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):

    #images = copy(m.images_series)
    data=m.generate_kdata(radial_traj,useGPU=use_GPU)

    data=np.array(data)
    np.save(filename_kdata, data)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    data = np.load(filename_kdata)

#plt.plot(np.unique(radial_traj.get_traj()[:,:,2],axis=-1),"s")

##volumes for slice taking into account coil sensi
print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    volumes_all=simulate_radial_undersampled_images(data,radial_traj,image_size,density_adj=True,useGPU=use_GPU,ntimesteps=ntimesteps)
    np.save(filename_volume,volumes_all)
    # sl=20
    # ani = animate_images(volumes_all[:,sl,:,:])
    #del volumes_all

# if str.split(filename_oop,"/")[-1] not in os.listdir(folder):
#     radial_traj_anatomy=Radial3D(total_nspokes=400,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
#     radial_traj_anatomy.traj = radial_traj.get_traj()[800:1200]
#     volume_outofphase=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices[:,800:1200,:,:],radial_traj_anatomy,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=True,memmap_file=None,light_memory_usage=True)
#     np.save(filename_oop, volume_outofphase)
# else:
#     volume_outofphase=np.load(filename_oop)

print("Building Volume OOP....")
if str.split(filename_volume_oop,"/")[-1] not in os.listdir(folder):
    data=np.load(filename_kdata)
    radial_traj_anatomy = Radial3D(total_nspokes=400, undersampling_factor=undersampling_factor, npoint=npoint,
                                   nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    radial_traj_anatomy.traj = radial_traj.get_traj()[800:1200]
    volume_oop=simulate_radial_undersampled_images(data[800:1200],radial_traj_anatomy,image_size,density_adj=True,useGPU=use_GPU,ntimesteps=1)[0]
    np.save(filename_volume_oop,volume_oop)
    # sl=20
    # ani = animate_images(volumes_all[:,sl,:,:])
    #del volumes_all

#animate_images(volume_oop)

##volumes for slice taking into account coil sensi
#print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    np.save(filename_mask,m.mask)

volumes_all=np.load(filename_volume)
volume_oop=np.load(filename_volume_oop)
mask=np.load(filename_mask)
masked_volume_oop=volume_oop[mask>0]
masked_volumes_all=volumes_all[:,mask>0]

variance_explained=0.7
all_pixels=np.argwhere(mask>0)

pixels_group=[]
patches_group=[]

dico_pixel_group={}

for j,pixel in tqdm(enumerate(all_pixels[:])):
    #print(pixel)
    all_patches_retained, pixels = select_similar_patches(tuple(pixel), volumes_all, volume_oop, window=(1, 2, 2),
                                                          quantile=10)
    shape=all_patches_retained.shape
    all_patches_retained = all_patches_retained.reshape(shape[0], shape[1],
                                                        -1)
    all_patches_retained = np.moveaxis(all_patches_retained, 0, -1)
    res=compute_low_rank_tensor(all_patches_retained,variance_explained)
    res=np.moveaxis(all_patches_retained,-1,0)
    res=res.reshape(res.shape[0],res.shape[1],-1)
    res = np.moveaxis(res, 1, -1)
    res=res.reshape(-1,res.shape[-1])

    #print(res.shape)
    #res=res.reshape(shape)

    patches_group.append(res)

    pixels=np.moveaxis(pixels,1,-1)
    pixels = pixels.reshape(-1, pixels.shape[-1])
    #print(pixels.shape)



    for i,pixel_patches in enumerate(pixels):
        if tuple(pixel_patches) not in dico_pixel_group.keys():
            dico_pixel_group[tuple(pixel_patches)]=np.zeros(len(all_pixels))

        dico_pixel_group[tuple(pixel_patches)][j]=i+1




patches_group=np.array(patches_group)
pixels_group=np.array(pixels_group)

pixel=all_pixels[0]
dico_pixel_group[tuple(pixel)]


plt.figure()

plt.plot(Sk_cur[0,0,:],label="Original")
plt.plot(res[0,0,:],label="Denoised")



cov_signal = np.real(masked_volumes_all.T@masked_volumes_all.conj())
inverse_std_signal = np.diag(np.sqrt(1/np.diag(cov_signal)))
corr_signal = inverse_std_signal@cov_signal@inverse_std_signal

import seaborn as sns
plt.figure()
sns.histplot(corr_signal,stat="probability")









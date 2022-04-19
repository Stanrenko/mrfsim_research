
#import matplotlib
#matplotlib.use('Qt5Agg')
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
import twixtools
import cupy as cp

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"
base_folder = "./data/InVivo/3D"

localfile ="/patient.001.v1/meas_MID00215_FID60605_raFin_3D_tra_FULl.dat"




with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)

seq = T1MRF(**sequence_config)

use_GPU = False
light_memory_usage=True

nb_ref_lines = 24
undersampling_factor=8

nspokes_per_z_encoding = 8
ntimesteps=175

filename = base_folder+localfile

filename_save=str.split(filename,".dat") [0]+".npy"
filename_save_us=str.split(filename,".dat") [0]+"_us{}ref{}_us.npy".format(undersampling_factor,nb_ref_lines)
filename_save_calib=str.split(filename,".dat") [0]+"_us{}ref{}_calib.npy".format(undersampling_factor,nb_ref_lines)
folder = "/".join(str.split(filename,"/")[:-1])


suffix_grappa = "_Tikhonov_0_01"
suffix=""
filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"

filename_b1 = str.split(filename,".dat") [0]+"_us{}_b1.npy".format(undersampling_factor)
filename_volume = str.split(filename,".dat") [0]+"_us{}_volumes.npy".format(undersampling_factor)
filename_kdata = str.split(filename,".dat") [0]+"_us{}_kdata.npy".format(undersampling_factor)
filename_mask= str.split(filename,".dat") [0]+"_us{}_mask.npy".format(undersampling_factor)
file_map = filename.split(".dat")[0] + "{}_us{}_MRF_map.pkl".format(suffix,undersampling_factor)

filename_volume_measured_and_calib = str.split(filename,".dat") [0]+"_volumes_allacqlines.npy"

filename_b1_grappa = str.split(filename,".dat") [0]+"{}_us{}ref{}_b1_grappa.npy".format(suffix_grappa,undersampling_factor,nb_ref_lines)
filename_volume_grappa = str.split(filename,".dat") [0]+"{}_us{}ref{}_volumes_grappa.npy".format(suffix_grappa,undersampling_factor,nb_ref_lines)
filename_kdata_grappa = str.split(filename,".dat") [0]+"{}_us{}ref{}_kdata_grappa.npy".format(suffix_grappa,undersampling_factor,nb_ref_lines)
filename_kdata_grappa_no_replace = str.split(filename,".dat") [0]+"{}_us{}ref{}_kdata_grappa_no_replace.npy".format(suffix_grappa,undersampling_factor,nb_ref_lines)
filename_mask_grappa= str.split(filename,".dat") [0]+"{}_us{}ref{}_mask_grappa.npy".format(suffix_grappa,undersampling_factor,nb_ref_lines)
file_map_grappa = filename.split(".dat")[0] + "{}_us{}ref{}_MRF_map_grappa.pkl".format(suffix_grappa,undersampling_factor,nb_ref_lines)

filename_currtraj_grappa = str.split(filename,".dat") [0]+"{}_us{}ref{}_currtraj_grappa.npy".format(suffix_grappa,undersampling_factor,nb_ref_lines)




if str.split(filename_seqParams,"/")[-1] not in os.listdir(folder):

    twix = twixtools.read_twix(filename,optional_additional_maps=["sWipMemBlock","sKSpace"],optional_additional_arrays=["SliceThickness"])

    alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
    x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
    y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
    z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]

    nb_slices=twix[-1]["hdr"]["Meas"]["lPartitions"]

    dico_seqParams = {"alFree":alFree,"x_FOV":x_FOV,"y_FOV":y_FOV,"z_FOV":z_FOV,"nb_slices":nb_slices}

    del alFree

    file = open(filename_seqParams, "wb")
    pickle.dump(dico_seqParams, file)
    file.close()

else:
    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)




meas_sampling_mode=dico_seqParams["alFree"][12]
nb_segments = dico_seqParams["alFree"][4]
nb_slices=dico_seqParams["nb_slices"]
x_FOV = dico_seqParams["x_FOV"]
y_FOV = dico_seqParams["y_FOV"]
z_FOV = dico_seqParams["z_FOV"]


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




if str.split(filename_save_us,"/")[-1] not in os.listdir(folder):

    if str.split(filename_save, "/")[-1] not in os.listdir(folder):
        if 'twix' not in locals():
            print("Re-loading raw data")
            twix = twixtools.read_twix(filename)
        mdb_list = twix[-1]['mdb']
        data = []

        for i, mdb in enumerate(mdb_list):
            if mdb.is_image_scan():
                data.append(mdb)

        data = np.array([mdb.data for mdb in data])
        data = data.reshape((-1, int(nb_segments)) + data.shape[1:])
        data = np.moveaxis(data, 2, 0)
        data = np.moveaxis(data, 2, 1)

        try:
            del twix
        except:
            pass

        np.save(filename_save, data)

    else:
        data = np.load(filename_save)

    data = groupby(data, nspokes_per_z_encoding, axis=1)
    #data=np.array(data)

    all_lines = np.arange(nb_slices).astype(int)

    shift=0
    data_calib=[]

    for ts in tqdm(range(len(data))):
        lines_measured = all_lines[shift::undersampling_factor]
        lines_to_estimate = list(set(all_lines) - set(lines_measured))
        center_line = int(nb_slices / 2)
        lines_ref = all_lines[(center_line - int(nb_ref_lines / 2)):(center_line + int(nb_ref_lines / 2))]

        data_calib.append(data[ts][:, :, lines_ref, :])
        data[ts] = data[ts][:, :, lines_measured, :]
        shift += 1
        shift = shift % (undersampling_factor)


    data_calib=np.array(data_calib)
    data=np.array(data)

    data=np.moveaxis(data,0,1)
    data_calib = np.moveaxis(data_calib, 0, 1)

    data=data.reshape(data.shape[0],-1,data.shape[-2],data.shape[-1])
    data_calib=data_calib.reshape(data_calib.shape[0],-1,data_calib.shape[-2],data_calib.shape[-1])

    np.save(filename_save_us, data)
    np.save(filename_save_calib, data_calib)

else:
    data = np.load(filename_save_us)
    data_calib = np.load(filename_save_calib)

try:
    del twix
except:
    pass

data_shape = data.shape

nb_channels = data_shape[0]

#nb_allspokes = data_shape[1]
npoint = data_shape[-1]
nb_part= nb_slices
image_size = (nb_slices, int(npoint/2), int(npoint/2))

dx = x_FOV/image_size[1]
dy = y_FOV/image_size[2]
dz = z_FOV/image_size[0]

nb_allspokes=nb_segments








radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    # Density adjustment all slices

    density = np.abs(np.linspace(-1, 1, npoint))
    density = np.expand_dims(density,tuple(range(data.ndim-1)))
    print("Performing Density Adjustment....")
    data *= density
    np.save(filename_kdata, data)

    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)


# Coil sensi estimation for all slices
print("Calculating Coil Sensitivity....")

if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage)
    np.save(filename_b1,b1_all_slices)
else:
    b1_all_slices=np.load(filename_b1)





sl=int(b1_all_slices.shape[1]/2)
list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
plot_image_grid(list_images,(5,4),title="Sensitivity map for slice {}".format(sl))


##volumes for slice taking into account coil sensi
print("Building Volume....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    del kdata_all_channels_all_slices
    kdata_all_channels_all_slices = np.load(filename_kdata)

    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=use_GPU,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
    np.save(filename_volume,volumes_all)
    # sl=20
    ani = animate_images(volumes_all[0,:,:,:],cmap="gray")

    #file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
    #                     "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_volume.mha"
    #io.write(file_mha, np.abs(volumes_all[0]), tags={"spacing": [dz, dx, dy]})

    del volumes_all

if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    del kdata_all_channels_all_slices
    kdata_all_channels_all_slices = np.load(filename_kdata)
    selected_spokes = np.r_[10:400]
    selected_spokes=None
    mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/25, normalize_kdata=False,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask,mask)
    del mask

#animate_images(mask)

del kdata_all_channels_all_slices
del b1_all_slices


########################## Dict mapping ########################################

#dictfile = "mrf175_SimReco2.dict"
dictfile = "mrf175_Dico2_Invivo.dict"

volumes_all = np.load(filename_volume)
mask = np.load(filename_mask)

#ani = animate_images(volumes_all[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])


with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)


load_map=False
save_map=True

if not(load_map):
    niter = 0

    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=None,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other")
    all_maps=optimizer.search_patterns(dictfile,volumes_all)

    if(save_map):
        import pickle

        #file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
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


###############################################################################

del volumes_all
del mask
del all_maps



kdata_measured_and_calib=np.concatenate([data,data_calib],axis=2)
radial_traj_measured_and_calib=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,nspoke_per_z_encoding=nb_allspokes)
radial_traj_measured_and_calib.traj=np.concatenate([traj_all[:,lines_measured,:,:],traj_all[:,lines_ref,:,:]],axis=1)


print("Building Volume....")
if str.split(filename_volume_measured_and_calib,"/")[-1] not in os.listdir(folder):

    volumes_all=simulate_radial_undersampled_images_multi(kdata_measured_and_calib,radial_traj_measured_and_calib,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=1,useGPU=use_GPU,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
    np.save(filename_volume_measured_and_calib,volumes_all)
    # sl=20
    #ani = animate_images(volumes_all[0,:,:,:],cmap="gray")

    file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
                        "_".join(str.split(str.split(filename_volume_measured_and_calib, "/")[-1], ".")[:-1])]) + "_volume.mha"
    io.write(file_mha, np.abs(volumes_all[0]), tags={"spacing": [dz, dx, dy]})
    #del volumes_all



#radial_traj_calib=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,nspoke_per_z_encoding=nb_allspokes)
#radial_traj_calib.traj=radial_traj_all.get_traj().reshape(nb_allspokes,nb_slices,npoint,3)[:,lines_ref,:,:].reshape(nb_allspokes,-1,3)

kernel_y=[0,1]
len_kerx = 7

indices_kz = list(range(1,(undersampling_factor)))


kernel_x=np.arange(-(int(len_kerx/2)-1),int(len_kerx/2))

pad_y=((np.array(kernel_y)<0).sum(),(np.array(kernel_y)>0).sum())
pad_x=((np.array(kernel_x)<0).sum(),(np.array(kernel_x)>0).sum())


data_calib_padded = np.pad(data_calib,((0, 0),(0, 0), pad_y, pad_x), mode="constant")
data_padded = np.pad(data,((0, 0),(0, 0), pad_y, pad_x), mode="constant")

all_lines = np.arange(nb_slices).astype(int)



ts=0
index_ky_target=0

kdata_all_channels_completed_all_ts=[]
curr_traj_completed_all_ts=[]

plot_fit=False
replace_calib_lines=True
useGPU=True

radial_traj_all=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,nspoke_per_z_encoding=nspokes_per_z_encoding)
traj_all = radial_traj_all.get_traj().reshape(nb_allspokes,nb_slices,npoint,3)

shift=0

lambd=0.01
calibration_mode = "Tikhonov"

ts=0
index_ky_target=0

for ts in tqdm(range(nb_allspokes)):

    if ((ts)%nspokes_per_z_encoding==0)and(not(ts==0)):
        shift += 1
        shift = shift % (undersampling_factor)

    lines_measured = all_lines[shift::undersampling_factor]
    lines_to_estimate = list(set(all_lines) - set(lines_measured))
    center_line = int(nb_slices / 2)
    lines_ref = all_lines[(center_line - int(nb_ref_lines / 2)):(center_line + int(nb_ref_lines / 2))]


    calib_lines = []
    calib_lines_ref_indices = []
    for i in indices_kz:
        current_calib_lines = []
        current_calib_lines_ref_indices = []

        for j in range(undersampling_factor * (kernel_y[0]) + i,
                       nb_ref_lines - (kernel_y[-1] * undersampling_factor - i)):
            current_calib_lines.append(lines_ref[j])
            current_calib_lines_ref_indices.append(j)
        calib_lines.append(current_calib_lines)
        calib_lines_ref_indices.append(current_calib_lines_ref_indices)

    calib_lines = np.array(calib_lines).astype(int)
    calib_lines_ref_indices = np.array(calib_lines_ref_indices).astype(int)

    weights = np.zeros((len(calib_lines), nb_channels, nb_channels * len(kernel_x) * len(kernel_y)),
                       dtype=data_calib.dtype)

    for index_ky_target in range(len(calib_lines_ref_indices)):
        calib_lines_ref_index = calib_lines_ref_indices[index_ky_target]
        F_target_calib=data_calib[:,ts,calib_lines_ref_indices[index_ky_target],:]#.reshape(nb_channels,-1)
        F_source_calib = np.zeros((nb_channels * len(kernel_x) * len(kernel_y),) +F_target_calib.shape[1:],dtype=F_target_calib.dtype)

        # l=calib_lines_ref_indices[index_ky_target][0]
        # j=0
        # i=0
        #i=8
        #l=calib_lines_ref_indices[index_ky_target][i]
        for i,l in enumerate(calib_lines_ref_indices[index_ky_target]):
            l0 = l - (index_ky_target + 1)
            for j in range(npoint):
                local_indices=np.stack(np.meshgrid(pad_y[0]+l0 + undersampling_factor*np.array(kernel_y), pad_x[0]+j + np.array(kernel_x)), axis=0).T.reshape(-1, 2)
                F_source_calib[:,i,j]=data_calib_padded[:,ts,local_indices[:,0],local_indices[:,1]].flatten()

        F_source_calib=F_source_calib.reshape(nb_channels * len(kernel_x) * len(kernel_y),-1)
        F_target_calib = F_target_calib.reshape(nb_channels,-1)

        if calibration_mode == "Tikhonov":
            if useGPU:
                F_source_calib_cupy = cp.asarray(F_source_calib)
                F_target_calib_cupy = cp.asarray(F_target_calib)
                weights[index_ky_target] = (F_target_calib_cupy @ F_source_calib_cupy.conj().T @ cp.linalg.inv(
                    F_source_calib_cupy @ F_source_calib_cupy.conj().T + lambd * cp.eye(F_source_calib_cupy.shape[0]))).get()

            else:
                weights[index_ky_target] = F_target_calib @ F_source_calib.conj().T @ np.linalg.inv(
                    F_source_calib @ F_source_calib.conj().T + lambd * np.eye(F_source_calib.shape[0]))
        else:
            if useGPU:
                weights[index_ky_target] = (
                            cp.asarray(F_target_calib) @ cp.linalg.pinv(cp.asarray(F_source_calib))).get()
            else:
                weights[index_ky_target] = F_target_calib @ np.linalg.pinv(F_source_calib)

    #i=0

    #l=lines_to_estimate[0]
    if plot_fit:
        F_estimate = weights[index_ky_target] @ F_source_calib

        plt.figure()
        plt.plot(np.linalg.norm(F_target_calib - F_estimate, axis=0) / np.sqrt(nb_channels))
        plt.title("Calibration Error for line {} aggregated accross channels".format(index_ky_target))

        nplot_x = int(np.sqrt(nb_channels))
        nplot_y = int(nb_channels / nplot_x) + 1
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



    F_target_estimate = np.zeros((nb_channels,len(lines_to_estimate),npoint),dtype=data.dtype)

    #i=10
    #l=lines_to_estimate[i]
    for i,l in (enumerate(lines_to_estimate)):
        F_source_estimate = np.zeros((nb_channels * len(kernel_x) * len(kernel_y), npoint),
                                      dtype=data_calib.dtype)
        index_ky_target=(l ) % (undersampling_factor)-1
        try:
            l0=np.argwhere(np.array(lines_measured-shift)==l - (index_ky_target + 1))[0][0]
        except:
            print("Line {} for Ts {} not estimated".format(l,ts))
            continue
        for j in range(npoint):
            local_indices = np.stack(
                np.meshgrid(pad_y[0] + l0 +np.array(kernel_y), pad_x[0] + j + np.array(kernel_x)),
                axis=0).T.reshape(-1, 2)
            F_source_estimate[:, j] = data_padded[:, ts, local_indices[:, 0], local_indices[:, 1]].flatten()
        F_target_estimate[:, i, :] = weights[index_ky_target] @ F_source_estimate

    #Replace calib lines
    #i=10
    #l=lines_to_estimate[i]
    F_target_estimate[:, :, :pad_x[0]] = 0
    F_target_estimate[:, :, -pad_x[1]:] = 0

    max_measured_line = np.max(lines_measured)
    min_measured_line = np.min(lines_measured)

    for i, l in (enumerate(lines_to_estimate)):
        if (l < min_measured_line) or (l > max_measured_line):
            F_target_estimate[:, i, :] = 0

    if replace_calib_lines:
        for i, l in (enumerate(lines_to_estimate)):
            if l in lines_ref.flatten():
                ind_line= np.argwhere(lines_ref.flatten()==l).flatten()[0]
                F_target_estimate[:,i,:]=data_calib[:,ts,ind_line,:]

    kdata_all_channels_completed=np.concatenate([data[:,ts],F_target_estimate],axis=1)
    curr_traj_estimate=traj_all[ts,lines_to_estimate,:,:]
    curr_traj_completed=np.concatenate([traj_all[ts,lines_measured,:,:],curr_traj_estimate],axis=0)

    kdata_all_channels_completed=kdata_all_channels_completed.reshape((nb_channels,-1))
    curr_traj_completed=curr_traj_completed.reshape((-1,3))

    kdata_all_channels_completed_all_ts.append(kdata_all_channels_completed)
    curr_traj_completed_all_ts.append(curr_traj_completed)


kdata_all_channels_completed_all_ts=np.array(kdata_all_channels_completed_all_ts)
kdata_all_channels_completed_all_ts=np.moveaxis(kdata_all_channels_completed_all_ts,0,1)
curr_traj_completed_all_ts=np.array(curr_traj_completed_all_ts)


for ts in tqdm(range(curr_traj_completed_all_ts.shape[0])):
    ind=np.lexsort((curr_traj_completed_all_ts[ts][:, 0], curr_traj_completed_all_ts[ts][:, 2]))
    curr_traj_completed_all_ts[ts] = curr_traj_completed_all_ts[ts,ind,:]
    kdata_all_channels_completed_all_ts[:,ts]=kdata_all_channels_completed_all_ts[:,ts,ind]

kdata_all_channels_completed_all_ts=kdata_all_channels_completed_all_ts.reshape(nb_channels,nb_allspokes,nb_slices,npoint)

if replace_calib_lines:
    np.save(filename_kdata_grappa,kdata_all_channels_completed_all_ts)
    np.save(filename_currtraj_grappa, curr_traj_completed_all_ts)

else:
    np.save(filename_kdata_grappa_no_replace,kdata_all_channels_completed_all_ts)
    del kdata_all_channels_completed_all_ts
    #kdata_all_channels_completed_all_ts_no_replace=np.load(filename_kdata_grappa_no_replace)


del data_calib
del data
del data_padded
del data_calib_padded
del F_source_calib
del F_source_estimate
del F_target_calib
del F_target_estimate
del weights


kdata_all_channels_completed_all_ts=np.load(filename_kdata_grappa)
curr_traj_completed_all_ts=np.load(filename_currtraj_grappa)
kdata_all_channels_completed_all_ts_no_replace=np.load(filename_kdata_grappa_no_replace)


plot_next_slice=False
ts=0
ch=0
sl =np.random.choice(list(set(lines_ref)-set(lines_measured)))
metric=np.abs
plt.figure()
plt.title("Ch {} Ts {} Sl {}".format(ch,ts,sl))
curr_traj = radial_traj_all.get_traj()[ts]
ind = np.lexsort((curr_traj[:,0], curr_traj[:,2]))
curr_traj=curr_traj[ind]
curr_kdata = data_allspokes[ch,ts].flatten()
curr_kdata=curr_kdata[ind]
plt.plot(metric(curr_kdata.reshape(data_allspokes.shape[2],npoint)[sl]),label='Kdata fully sampled')
plt.plot(metric(kdata_all_channels_completed_all_ts[ch, ts].reshape(nb_slices,npoint)[sl]),label='Kdata grappa estimate')
plt.plot(metric(kdata_all_channels_completed_all_ts_no_replace[ch, ts].reshape(nb_slices,npoint)[sl]),label='Kdata grappa estimate no replace')

if plot_next_slice:
    if sl>0:
        plt.plot(metric(curr_kdata.reshape(data_allspokes.shape[2],npoint)[sl-1]),label='Kdata fully sampled previous slice',linestyle='dashed')
    else:
        plt.plot(metric(curr_kdata.reshape(data_allspokes.shape[2], npoint)[0]),
                 label='Kdata fully sampled previous slice',linestyle='dashed')
    if sl<(nb_slices-2):
        plt.plot(metric(curr_kdata.reshape(data_allspokes.shape[2],npoint)[sl+1]),label='Kdata fully sampled next slice',linestyle='dashed')
    else:
        plt.plot(metric(curr_kdata.reshape(data_allspokes.shape[2], npoint)[nb_slices-1]),
                 label='Kdata fully sampled next slice',linestyle='dashed')
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


sl=int(b1_all_slices_grappa.shape[1]/2)
#sl=2

list_images = list(np.abs(b1_all_slices_grappa[:,sl,:,:]))
plot_image_grid(list_images,(3,3),title="Grappa : Sensitivity map for slice {}".format(sl))

# list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
# plot_image_grid(list_images,(3,3),title="Undersampled : Sensitivity map for slice {}".format(sl))
#
#
# list_images = list(np.abs(b1_all_slices_all_spokes[:,sl,:,:]))
# plot_image_grid(list_images,(3,3),title="All Spokes : Sensitivity map for slice {}".format(sl))
#
# list_images = list(np.abs(b1_maps[:,sl,:,:]))
# plot_image_grid(list_images,(3,3),title="Ground Truth : Sensitivity map for slice {}".format(sl))

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
    kdata_all_channels_completed_all_ts = np.load(filename_kdata_grappa)
    selected_spokes = np.r_[10:400]
    selected_spokes = None
    mask = build_mask_single_image_multichannel(kdata_all_channels_completed_all_ts, radial_traj_grappa, image_size,
                                                b1=b1_all_slices, density_adj=False, threshold_factor=None,
                                                normalize_kdata=False, light_memory_usage=True,
                                                selected_spokes=selected_spokes)
    np.save(filename_mask_grappa, mask)
    del mask




animate_images(mask)

########################## Dict mapping ########################################

#dictfile = "mrf175_SimReco2.dict"
dictfile = "mrf175_Dico2_Invivo.dict"

volumes_all = np.load(filename_volume_grappa)
mask = np.load(filename_mask_grappa)

#ani = animate_images(volumes_all[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])


with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)


load_map=False
save_map=True

if not(load_map):
    niter = 0

    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=None,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other")
    all_maps=optimizer.search_patterns(dictfile,volumes_all)

    if(save_map):
        import pickle

        #file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
        file = open(file_map_grappa, "wb")
        # dump information to that file
        pickle.dump(all_maps, file)
        # close the file
        file.close()

else:
    import pickle
    #file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
    file = open(file_map_grappa, "rb")
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
        io.write(file_mha,map_for_sim[key],tags={"spacing":[dz,dx,dy]})
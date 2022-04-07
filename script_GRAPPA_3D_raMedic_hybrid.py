
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


localfile = "/phantom.002.v1/meas_MID00273_FID59700_raMedic_3Dgated.dat"
localfile = "/phantom.002.v1/meas_MID00274_FID59701_raMedic_3Dgated_GRAPPA2.dat"
localfile = "/phantom.002.v1/meas_MID00275_FID59702_raMedic_3Dgated_GRAPPA4.dat"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)

seq = T1MRF(**sequence_config)

use_GPU = False
light_memory_usage=True
gen_mode="other"



filename = base_folder+localfile

filename_save=str.split(filename,".dat") [0]+".npy"
filename_save_calib=str.split(filename,".dat") [0]+"_calib.npy"
folder = "/".join(str.split(filename,"/")[:-1])


suffix = ""
filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"

filename_b1 = str.split(filename,".dat") [0]+"_b1.npy"
filename_volume = str.split(filename,".dat") [0]+"_volumes.npy"
filename_kdata = str.split(filename,".dat") [0]+"_kdata.npy"
filename_mask= str.split(filename,".dat") [0]+"_mask.npy"
file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format(suffix)

filename_volume_measured_and_calib = str.split(filename,".dat") [0]+"_volumes_allacqlines.npy"

filename_b1_grappa = str.split(filename,".dat") [0]+"_b1_grappa.npy"
filename_volume_grappa = str.split(filename,".dat") [0]+"_volumes_grappa.npy"
filename_kdata_grappa = str.split(filename,".dat") [0]+"_kdata_grappa.npy"
filename_kdata_grappa_no_replace = str.split(filename,".dat") [0]+"_kdata_grappa_no_replace.npy"
filename_mask_grappa= str.split(filename,".dat") [0]+"_mask_grappa.npy"
file_map_grappa = filename.split(".dat")[0] + "{}_MRF_map_grappa.pkl".format(suffix)

filename_currtraj_grappa = str.split(filename,".dat") [0]+"_currtraj_grappa.npy"

filename_save_allspokes = str.split(base_folder + "/phantom.002.v1/meas_MID00273_FID59700_raMedic_3Dgated.dat",".dat")[0]+".npy"

data_allspokes = np.load(filename_save_allspokes)

#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"


use_GPU = False
light_memory_usage=True
#Parsed_File = rT.map_VBVD(filename)
#idx_ok = rT.detect_TwixImg(Parsed_File)
#RawData = Parsed_File[str(idx_ok)]["image"].readImage()

if str.split(filename_seqParams,"/")[-1] not in os.listdir(folder):

    twix = twixtools.read_twix(filename,optional_additional_maps=["sWipMemBlock","sKSpace"],optional_additional_arrays=["SliceThickness"])

    alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
    x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
    y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
    z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]
    undersampling_factor=twix[-1]["hdr"]["Meas"]["lAccelFact3D"]
    nb_ref_lines = twix[-1]["hdr"]["Meas"]["lRefLines3D"]
    nb_slices=twix[-1]["hdr"]["Meas"]["lPartitions"]

    dico_seqParams = {"alFree":alFree,"x_FOV":x_FOV,"y_FOV":y_FOV,"z_FOV":z_FOV,"US":undersampling_factor,"nb_slices":nb_slices,"nb_ref_lines":nb_ref_lines}

    del alFree

    file = open(filename_seqParams, "wb")
    pickle.dump(dico_seqParams, file)
    file.close()

else:
    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)




meas_sampling_mode=dico_seqParams["alFree"][12]
nb_allspokes=dico_seqParams["alFree"][5]
undersampling_factor=dico_seqParams["US"]
nb_slices=dico_seqParams["nb_slices"]
nb_ref_lines=dico_seqParams["nb_ref_lines"]
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

if str.split(filename_save,"/")[-1] not in os.listdir(folder):
    if 'twix' not in locals():
        print("Re-loading raw data")
        twix = twixtools.read_twix(filename)
    mdb_list = twix[-1]['mdb']
    data=[]

    data_calib = []
    for i, mdb in enumerate(mdb_list):
        if mdb.is_image_scan():
            data.append(mdb)
        if mdb.is_flag_set("PATREFSCAN"):
            data_calib.append(mdb)

    data = np.array([mdb.data for mdb in data])
    data = data.reshape((-1, int(nb_allspokes)) + data.shape[1:])
    data=np.moveaxis(data,2,0)
    data = np.moveaxis(data, 2, 1)

    data_calib = np.array([mdb.data for mdb in data_calib])
    data_calib = data_calib.reshape((-1, int(nb_allspokes)) + data_calib.shape[1:])
    if undersampling_factor>1:
        data_calib = np.moveaxis(data_calib, 2, 0)
        data_calib = np.moveaxis(data_calib, 2, 1)

    try:
        del twix
    except:
        pass


    np.save(filename_save,data)
    if undersampling_factor > 1:
        np.save(filename_save_calib, data_calib)

else :
    data = np.load(filename_save)
    data_calib = np.load(filename_save_calib)

try:
    del twix
except:
    pass

data_shape = data.shape

nb_channels = data_shape[0]

#nb_allspokes = data_shape[1]
npoint = data_shape[-1]
nb_part= data_shape[2]
image_size = (nb_slices, npoint, npoint)

dx = x_FOV/image_size[1]
dy = y_FOV/image_size[2]
dz = z_FOV/image_size[0]

#if str.split(filename_kdata,"/")[-1] in os.listdir(folder):
#    del data

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,nspoke_per_z_encoding=nb_allspokes)


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    # Density adjustment all slices

    density = np.abs(np.linspace(-1, 1, npoint))
    density = np.expand_dims(density,tuple(range(data.ndim-1)))
    #kdata_all_channels_all_slices = data.reshape(-1, npoint)
    #del data
    print("Performing Density Adjustment....")
    data *= density
    np.save(filename_kdata, data)
    #del data

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

    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=use_GPU,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
    np.save(filename_volume,volumes_all)
    # sl=20
    ani = animate_images(volumes_all[0,:,:,:],cmap="gray")

    file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
                         "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_volume.mha"
    io.write(file_mha, np.abs(volumes_all[0]), tags={"spacing": [dz, dx, dy]})

    del volumes_all







radial_traj_all=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,nspoke_per_z_encoding=nb_allspokes)
traj_all = radial_traj_all.get_traj().reshape(nb_allspokes,nb_slices,npoint,3)

all_lines=np.arange(nb_slices).astype(int)
lines_measured=all_lines[::undersampling_factor]
lines_to_estimate=list(set(all_lines)-set(lines_measured))

nb_ref_lines=24

center_line=int(nb_slices/2)
lines_ref = all_lines[(center_line-int(nb_ref_lines/2)):(center_line+int(nb_ref_lines/2))]

if not(nb_ref_lines==data_calib.shape[2]):
    skipped_lines = int((data_calib.shape[-2]-nb_ref_lines)/2)
    data_calib=data_calib[:,:,skipped_lines:-skipped_lines,:]

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

density = np.abs(np.linspace(-1, 1, npoint))
density = np.expand_dims(density,tuple(range(data.ndim-1)))

data*=density
data_calib*=density
data_allspokes*=density

t = traj_all[:,lines_measured,:,:]
if data.dtype == "complex64":
    t = t.astype("float32")



data_hybrid = [finufft.nufft2d1(t[:,s,:, 0].flatten(), t[:,s,:, 1].flatten(), data[:, :, s,:].reshape(nb_channels,-1), image_size[1:]) for s in range(data.shape[2])]
data_hybrid=np.array(data_hybrid)

t = traj_all[:,lines_ref,:,:]
if data_calib.dtype == "complex64":
    t = t.astype("float32")

data_calib_hybrid = [finufft.nufft2d1(t[:,s,:, 0].flatten(), t[:,s,:, 1].flatten(), data_calib[:, :, s,:].reshape(nb_channels,-1), image_size[1:]) for s in range(data_calib.shape[2])]
data_calib_hybrid=np.array(data_calib_hybrid)

t = traj_all[:, :, :, :]
if data_allspokes.dtype == "complex64":
    t = t.astype("float32")

data_allspokes_hybrid = [
    finufft.nufft2d1(t[:, s, :, 0].flatten(), t[:, s, :, 1].flatten(), data_allspokes[:, :, s, :].reshape(nb_channels, -1),
                     image_size[1:]) for s in range(data_allspokes.shape[2])]
data_allspokes_hybrid = np.array(data_allspokes_hybrid)


indices_kz = list(range(1,(undersampling_factor)))

calib_lines=[]
calib_lines_ref_indices = []
for i in indices_kz:
    current_calib_lines=[]
    current_calib_lines_ref_indices = []

    for j in range(undersampling_factor*(kernel_y[0])+i,nb_ref_lines-(kernel_y[-1]*undersampling_factor-i)):
        current_calib_lines.append(lines_ref[j])
        current_calib_lines_ref_indices.append(j)
    calib_lines.append(current_calib_lines)
    calib_lines_ref_indices.append(current_calib_lines_ref_indices)

calib_lines=np.array(calib_lines).astype(int)
calib_lines_ref_indices=np.array(calib_lines_ref_indices).astype(int)

pad_y=((np.array(kernel_y)<0).sum(),(np.array(kernel_y)>0).sum())


data_calib_hybrid_padded = np.pad(data_calib_hybrid,(pad_y,(0,0),(0,0),(0,0)), mode="constant")
data_hybrid_padded = np.pad(data_hybrid,(pad_y,(0,0),(0,0),(0,0)), mode="constant")



weights=np.zeros((image_size[1],image_size[2],len(calib_lines),nb_channels,nb_channels*len(kernel_y)),dtype=data_calib.dtype)


ts=0
index_ky_target=0

kdata_all_channels_completed_all_ts=[]
curr_traj_completed_all_ts=[]

plot_fit=False
replace_calib_lines=True
useGPU=True

x=128
y=128
index_ky_target=0

kdata_all_channels_completed_all_pixels=np.zeros((len(lines_to_estimate),nb_channels,npoint,npoint),dtype=data.dtype)

for x in tqdm(range(image_size[1])):
    for y in range((image_size[2])):
        for index_ky_target in range(len(calib_lines_ref_indices)):
            calib_lines_ref_index = calib_lines_ref_indices[index_ky_target]
            F_target_calib = data_calib_hybrid[calib_lines_ref_indices[index_ky_target], :,x,y]  # .reshape(nb_channels,-1)
            F_source_calib = np.zeros((nb_channels * len(kernel_y),) + F_target_calib.shape[1:],
                                      dtype=F_target_calib.dtype)

            # l=calib_lines_ref_indices[index_ky_target][0]
            # j=0
            # i=0
            # i=8
            # l=calib_lines_ref_indices[index_ky_target][i]
            #i=0
            #l=calib_lines_ref_indices[index_ky_target][i]

            i=0
            l=calib_lines_ref_indices[index_ky_target][i]

            for i, l in enumerate(calib_lines_ref_indices[index_ky_target]):
                l0 = l - (index_ky_target + 1)
                local_indices = np.stack(np.meshgrid(pad_y[0] + l0 + undersampling_factor * np.array(kernel_y)),
                                             axis=0)
                F_source_calib[:, i] = data_calib_hybrid_padded[local_indices[0], :, x,y].flatten()

            F_source_calib = F_source_calib.reshape(nb_channels * len(kernel_y), -1)
            F_target_calib = F_target_calib.reshape(-1, nb_channels)
            F_target_calib = np.moveaxis(F_target_calib,0,1)

            if useGPU:
                weights[x,y,index_ky_target] = (
                            cp.asarray(F_target_calib) @ cp.linalg.pinv(cp.asarray(F_source_calib))).get()
            else:
                weights[x,y,index_ky_target] = F_target_calib @ np.linalg.pinv(F_source_calib)

            if plot_fit:
                F_estimate = weights[x,y,index_ky_target] @ F_source_calib

                plt.figure()
                plt.plot(np.linalg.norm(F_target_calib - F_estimate, axis=0) / np.sqrt(nb_channels))
                plt.title("Calibration Error for line {} aggregated accross channels".format(index_ky_target))

                i=0
                plt.figure()
                plt.plot(np.abs(F_estimate[:,i]),label="Estimate")
                plt.plot(np.abs(F_target_calib[:, i]), label="Origin")
                plt.legend()

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

        F_target_estimate = np.zeros((nb_channels, len(lines_to_estimate)), dtype=data.dtype)



        i=9
        l=lines_to_estimate[i]
        for i, l in (enumerate(lines_to_estimate)):
            #F_source_estimate = np.zeros((nb_channels * len(kernel_y)),
             #                            dtype=data_calib.dtype)
            index_ky_target = (l) % (undersampling_factor) - 1
            l0 = np.argwhere(np.array(lines_measured) == l - (index_ky_target + 1))[0][0]
            local_indices = np.stack(
                    np.meshgrid(pad_y[0] + l0 + np.array(kernel_y)),
                    axis=0)
            F_source_estimate = data_hybrid_padded[local_indices[0], :, x,y].flatten()
            F_target_estimate[:, i] = weights[x,y,index_ky_target] @ F_source_estimate

        # Replace calib lines
        # i=10
        # l=lines_to_estimate[i]

        if replace_calib_lines:
            i=9
            l=lines_to_estimate[i]
            for i, l in (enumerate(lines_to_estimate)):
                if l in lines_ref.flatten():
                    ind_line = np.argwhere(lines_ref.flatten() == l).flatten()[0]
                    #plt.figure();plt.plot(np.abs(F_target_estimate[:, i]));plt.plot(np.abs(data_calib_hybrid[ind_line, :, x,y]));plt.plot(np.abs(F_estimate[:,0]));
                    F_target_estimate[:, i] = data_calib_hybrid[ind_line, :, x,y]

        kdata_all_channels_completed_all_pixels[:, :, x, y] =F_target_estimate.T




data_hybrid_completed = np.concatenate([kdata_all_channels_completed_all_pixels,data_hybrid],axis=0)

traj_kz_measured = np.unique(traj_all[:,lines_measured,:,2]).flatten()
traj_kz_estimate = np.unique(traj_all[:,lines_to_estimate,:,2]).flatten()

traj_kz = np.concatenate([traj_kz_estimate,traj_kz_measured])

ind=np.argsort(traj_kz)
traj_kz = traj_kz[ind]
data_hybrid_completed_sorted = data_hybrid_completed[ind,:,:,:]

data_hybrid_completed_sorted[:((undersampling_factor-1)*pad_y[0]),:,:,:]=0
data_hybrid_completed_sorted[(-(undersampling_factor-1)*pad_y[1]):,:,:]=0


x=np.random.choice(npoint)
y=np.random.choice(npoint)
ch=0

metric=np.abs
plt.figure()
plt.plot(metric(data_allspokes_hybrid[:,ch,x,y]),label='Kdata fully sampled')
plt.plot(metric(data_hybrid_completed_sorted[:,ch,x,y]),label='Kdata grappa estimate')
plt.legend()

plt.figure()
plt.title("Error channel {}".format(ch))
plt.plot(metric(data_allspokes_hybrid[:,ch,x,y]-data_hybrid_completed_sorted[:,ch,x,y]))




if data_hybrid_completed.dtype == "complex64":
    traj_kz = traj_kz.astype("float32")

kdata_hybrid_for_image = np.moveaxis(data_hybrid_completed_sorted.reshape(nb_slices,-1),-1,0)
images_all_channels = finufft.nufft1d1(traj_kz,kdata_hybrid_for_image,48)

images_all_channels=np.moveaxis(images_all_channels,-1,0)
images_all_channels=images_all_channels.reshape(nb_slices,nb_channels,image_size[1],image_size[2])
images_all_channels=np.moveaxis(images_all_channels,0,1)

plt.figure()
plt.imshow(np.abs(images_all_channels[5,24,:,:]))

image_rebuilt=np.sqrt(np.sum(np.abs(images_all_channels),axis=0))

ch=0
file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
                        "_".join(str.split(str.split(filename_volume_grappa, "/")[-1], ".")[:-1])]) + "_volume_hybridmethod_ch_{}.mha".format(ch)
io.write(file_mha, np.abs(images_all_channels[ch,:,:,:]), tags={"spacing": [dz, dx, dy]})


file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
                        "_".join(str.split(str.split(filename_volume_grappa, "/")[-1], ".")[:-1])]) + "_volume_hybridmethod_sos.mha"
io.write(file_mha, np.abs(image_rebuilt), tags={"spacing": [dz, dx, dy]})











for ts in tqdm(range(nb_allspokes)):
    for index_ky_target in range(len(calib_lines_ref_indices)):
        calib_lines_ref_index = calib_lines_ref_indices[index_ky_target]
        F_target_calib=data_calib[:,ts,calib_lines_ref_indices[index_ky_target],:]#.reshape(nb_channels,-1)
        F_source_calib = np.zeros((nb_channels * len(kernel_y),) +F_target_calib.shape[1:],dtype=F_target_calib.dtype)

        # l=calib_lines_ref_indices[index_ky_target][0]
        # j=0
        # i=0
        #i=8
        #l=calib_lines_ref_indices[index_ky_target][i]
        for i,l in enumerate(calib_lines_ref_indices[index_ky_target]):
            l0 = l - (index_ky_target + 1)
            for j in range(npoint):
                local_indices=np.stack(np.meshgrid(pad_y[0]+l0 + undersampling_factor*np.array(kernel_y)), axis=0).T.reshape(-1, 2)
                F_source_calib[:,i,j]=data_calib_padded[:,ts,local_indices[:,0],local_indices[:,1]].flatten()

        F_source_calib=F_source_calib.reshape(nb_channels * len(kernel_y),-1)
        F_target_calib = F_target_calib.reshape(nb_channels,-1)

        if useGPU:
            weights[index_ky_target] = (cp.asarray(F_target_calib) @ cp.linalg.pinv(cp.asarray(F_source_calib))).get()
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
        F_source_estimate = np.zeros((nb_channels * len(kernel_y), npoint),
                                      dtype=data_calib.dtype)
        index_ky_target=(l ) % (undersampling_factor)-1
        l0=np.argwhere(np.array(lines_measured)==l - (index_ky_target + 1))[0][0]
        for j in range(npoint):
            local_indices = np.stack(
                np.meshgrid(pad_y[0] + l0 +np.array(kernel_y)),
                axis=0).T.reshape(-1, 2)
            F_source_estimate[:, j] = data_padded[:, ts, local_indices[:, 0], local_indices[:, 1]].flatten()
        F_target_estimate[:, i, :] = weights[index_ky_target] @ F_source_estimate

    #Replace calib lines
    #i=10
    #l=lines_to_estimate[i]

    if replace_calib_lines:
        for i, l in (enumerate(lines_to_estimate)):
            if l in lines_ref.flatten():
                ind_line= np.argwhere(lines_ref.flatten()==l).flatten()[0]
                F_target_estimate[:,i,:]=data_calib[:,ts,ind_line,:]

    kdata_all_channels_completed=np.concatenate([data[:,ts],F_target_estimate],axis=1)
    curr_traj_estimate=traj_all[ts,lines_to_estimate,:,:]
    curr_traj_completed=np.concatenate([radial_traj.get_traj()[ts].reshape(-1,npoint,3),curr_traj_estimate],axis=0)

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
kdata_all_channels_completed_all_ts[:,:,:,:pad_x[0]]=0
kdata_all_channels_completed_all_ts[:,:,:,-pad_x[1]:]=0
kdata_all_channels_completed_all_ts[:,:,:((undersampling_factor-1)*pad_y[0]),:]=0
kdata_all_channels_completed_all_ts[:,:,(-(undersampling_factor-1)*pad_y[1]):,:]=0

if replace_calib_lines:
    np.save(filename_kdata_grappa,kdata_all_channels_completed_all_ts)
    np.save(filename_currtraj_grappa, curr_traj_completed_all_ts)

else:
    np.save(filename_kdata_grappa_no_replace,kdata_all_channels_completed_all_ts)
    del kdata_all_channels_completed_all_ts
    #kdata_all_channels_completed_all_ts_no_replace=np.load(filename_kdata_grappa_no_replace)

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



radial_traj_grappa = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,nspoke_per_z_encoding=nb_allspokes,
                                  nb_slices=nb_slices, incoherent=incoherent, mode=mode)
radial_traj_grappa.traj = curr_traj_completed_all_ts


print("Building B1 Map Gappa....")
if str.split(filename_b1_grappa,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices_grappa=calculate_sensitivity_map_3D(kdata_all_channels_completed_all_ts.reshape(nb_channels,nb_allspokes,-1,npoint),radial_traj_grappa,res,image_size,useGPU=False,light_memory_usage=light_memory_usage,density_adj=True)
    np.save(filename_b1_grappa,b1_all_slices_grappa)
else:
    b1_all_slices_grappa=np.load(filename_b1_grappa)

print("Building Volume....")
if str.split(filename_volume_grappa,"/")[-1] not in os.listdir(folder):
    del kdata_all_channels_completed_all_ts
    kdata_all_channels_completed_all_ts = np.load(filename_kdata_grappa)

    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_completed_all_ts,radial_traj_grappa,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=1,useGPU=use_GPU,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
    np.save(filename_volume_grappa,volumes_all)
    # sl=20
    ani = animate_images(volumes_all[0,:,:,:],cmap="gray")

    file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
                        "_".join(str.split(str.split(filename_volume_grappa, "/")[-1], ".")[:-1])]) + "_volume.mha"
    io.write(file_mha, np.abs(volumes_all[0]), tags={"spacing": [dz, dx, dy]})
    #del volumes_all





























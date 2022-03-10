
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
base_folder = "./2D"

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

name = "SquareSimu"
name = "KneePhantom"

use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

folder = "/".join(str.split(filename,"/")[:-1])

suffix="_cartesian"

filename_paramMap=filename+"_paramMap.pkl"
filename_paramMask=filename+"_paramMask.npy"
filename_volume = filename+"_volumes{}.npy".format(suffix)
filename_groundtruth = filename+"_groundtruth_volumes{}.npy".format("")

filename_kdata = filename+"_kdata{}.npy".format(suffix)
filename_mask= filename+"_mask{}.npy".format(suffix)
file_map = filename + "{}_MRF_map.pkl".format(suffix)


#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint_x = 256



incoherent=True
mode="old"

image_size = (int(npoint_x), int(npoint_x))
nb_segments=nb_allspokes
nspoke=int(nb_segments/ntimesteps)

with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)

if name=="SquareSimu":
    region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
    mask_reduction_factor=1/4

    m = RandomMap(name,dict_config,resting_time=4000,image_size=image_size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)

elif name=="KneePhantom":
    num =1
    file_matlab_paramMap = "./data/{}/Phantom{}/paramMap.mat".format(name,num)

    m = MapFromFile(name,image_size=image_size,file=file_matlab_paramMap,rounding=True,gen_mode="other")

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

i=0
image=m.images_series[i]

# npoint_y = 128
# curr_traj=cartesian_traj_2D(npoint_x,npoint_y)


undersampling =2

traj_all=cartesian_traj_2D(npoint_x,npoint_x)

traj_all=traj_all.reshape(npoint_x,npoint_x,-1)

curr_traj=traj_all[::undersampling,:,:]
curr_traj=curr_traj.reshape(-1,npoint_x,2)

npoint_y = curr_traj.shape[0]

curr_traj=curr_traj.reshape(-1,2)

kdata = finufft.nufft2d2(curr_traj[:,0], curr_traj[:,1], image)

rebuilt_image=finufft.nufft2d1(curr_traj[:,0], curr_traj[:,1], kdata, image_size)

# plt.imshow(np.abs(rebuilt_image))

nb_channels=4
sig=50**2

nb_means=int(np.sqrt(nb_channels))

means_x=np.arange(1,nb_means+1)*(1/(nb_means+1))*image_size[0]
means_y=np.arange(1,nb_means+1)*(1/(nb_means+1))*image_size[1]

x = np.arange(image_size[0])
y = np.arange(image_size[1])

X,Y = np.meshgrid(x,y)
pixels=np.stack([X.flatten(), Y.flatten()], axis=-1)

from scipy.stats import multivariate_normal
b1_maps=[]
for mu_x in means_x:
    for mu_y in means_y:
        b1_maps.append(multivariate_normal.pdf(pixels, mean=[mu_x,mu_y], cov=sig*np.eye(2)))

b1_maps = np.array(b1_maps)
b1_maps=b1_maps/np.expand_dims(np.max(b1_maps,axis=-1),axis=-1)
b1_maps=b1_maps.reshape((nb_channels,)+image_size)
#
#
# for ch in range(nb_channels):
#     plt.figure()
#     plt.imshow(b1_maps[ch])
#     plt.colorbar()


image_all_channels=np.array([image*b1 for b1 in b1_maps])

traj_all_spokes = traj_all.reshape(-1,2)

kdata_all_channels=[finufft.nufft2d2(curr_traj[:,0], curr_traj[:,1], p) for p in image_all_channels]
kdata_all_channels_all_spokes=[finufft.nufft2d2(traj_all_spokes[:,0], traj_all_spokes[:,1], p) for p in image_all_channels]


image_rebuilt_all_channels=[finufft.nufft2d1(curr_traj[:,0], curr_traj[:,1], k, image_size) for k in kdata_all_channels]

#Calib data

# traj_calib = np.array(np.broadcast_arrays(np.expand_dims(kx_line,axis=-1),np.expand_dims([0],axis=0)))
#
# traj_calib_for_grappa=np.moveaxis(traj_calib,-1,0)
# traj_calib_for_grappa=np.moveaxis(traj_calib_for_grappa,-1,-2)
#
# traj_calib=np.squeeze(np.moveaxis(traj_calib,0,1))

center_line=int(npoint_x/2)-1

over_calibrate=1

if center_line in np.arange(npoint_x)[::undersampling]:
    center_line=center_line+1

center_lines=[]
current_center_line=center_line
for i in range(over_calibrate):
    center_lines.append(current_center_line)
    current_center_line += undersampling


calib_lines=[]
calib_line=center_line
calib_lines = []
for j in range(len(center_lines)):
    calib_line = center_lines[j]
    for i in range(undersampling-1):
        calib_lines.append(calib_line)
        calib_line=calib_line+1



traj_calib_for_grappa=traj_all[calib_lines,:,:]

traj_calib=traj_calib_for_grappa.reshape(-1,2)

kdata_calib =finufft.nufft2d2(traj_calib[:,0], traj_calib[:,1], image)
kdata_calib_all_channels=[finufft.nufft2d2(traj_calib[:,0], traj_calib[:,1], p) for p in image_all_channels]


#
# for ch in range(nb_channels):
#     plt.figure()
#     plt.imshow(np.abs(image_rebuilt_all_channels[ch]))




image_rebuilt=np.sqrt(np.sum(np.abs(image_rebuilt_all_channels) ** 2, axis=0))

plt.figure()
plt.imshow(np.abs(image_rebuilt))
plt.title("Without GRAPPA")



###GRAPPA RECONSTRUCTION#####

kernel_x=[-2,-1,0,1,2]
kernel_y=[-1,0,1]

kernel_x=[-1,0,1]
kernel_y=[0,1]

lambd = 0.

list_kernels=[([-1,0,1],[0,1]),(np.arange(-2,3).astype(int),[0,1])]
list_kernels=[([-1,0,1],[0,1])]
#list_kernels=[(np.arange(-10,11).astype(int),[0,1])]

curr_traj_for_grappa=curr_traj.reshape(npoint_y,npoint_x,2)
kdata_all_channels_for_grappa=np.array(kdata_all_channels).reshape(nb_channels,npoint_y,npoint_x)

kdata_calib_all_channels=np.array(kdata_calib_all_channels).reshape((nb_channels,)+traj_calib_for_grappa.shape[:-1])

calibration_mode="Standard"

for (kernel_x,kernel_y) in list_kernels:
    # CALIBRATION

    F_source_calib=np.zeros((nb_channels*len(kernel_x)*len(kernel_y),npoint_x),dtype=kdata_calib.dtype)

    pad_y=((np.array(kernel_y)<0).sum(),(np.array(kernel_y)>0).sum())
    pad_x=((np.array(kernel_x)<0).sum(),(np.array(kernel_x)>0).sum())

    kdata_all_channels_for_grappa_padded=np.pad(kdata_all_channels_for_grappa,((0,0),pad_y,pad_x),mode="edge")
    weights=np.zeros((len(calib_lines),nb_channels,nb_channels*len(kernel_x)*len(kernel_y)),dtype=kdata_calib.dtype)

    for index_ky_target in range(len(calib_lines)):
        print("Calibration for line {}".format(index_ky_target))
        F_target_calib = kdata_calib_all_channels[:, index_ky_target, :]
        for j in tqdm(range(traj_calib_for_grappa.shape[1])):
            kx=traj_calib_for_grappa[index_ky_target,j,0]
            ky = traj_calib_for_grappa[index_ky_target,j,1]

            kx_comp = (kx-curr_traj_for_grappa[0,:,0])
            kx_comp[kx_comp < 0] = np.inf

            kx_ref=curr_traj_for_grappa[0,np.argmin(kx_comp),0]

            ky_comp = (ky - curr_traj_for_grappa[:, 0, 1])
            ky_comp[ky_comp < 0] = np.inf

            ky_ref = curr_traj_for_grappa[np.argmin(ky_comp), 0,1]

            j0, i0 = np.unravel_index(
                np.argmin(np.linalg.norm(curr_traj_for_grappa - np.expand_dims(np.array([kx_ref, ky_ref]), axis=(0, 1)), axis=-1)),
                curr_traj_for_grappa.shape[:-1])
            local_indices = np.stack(np.meshgrid(pad_y[0]+j0 + np.array(kernel_y), pad_x[0]+i0 + np.array(kernel_x)), axis=0).T.reshape(-1, 2)

            local_kdata_for_grappa = kdata_all_channels_for_grappa_padded[:, local_indices[:, 0], local_indices[:, 1]]
            local_kdata_for_grappa = np.array(local_kdata_for_grappa)
            local_kdata_for_grappa = local_kdata_for_grappa.flatten()
            F_source_calib[:,j]=local_kdata_for_grappa


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


                return np.trace((wRX.conj().T)@wRX) + np.trace((wIX.conj().T)@wIX) - 2*np.trace(np.real(Y.conj().T @ wR @ X)) + 2*np.trace(np.imag(Y.conj().T @ wI @ X)) + lambd*(np.linalg.norm(wR.flatten(),ord=1)+np.linalg.norm(wI.flatten(),ord=1))

            w0=np.zeros((2,nb_channels,nb_channels*len(kernel_x)*len(kernel_y)),dtype=kdata_calib.dtype).flatten()

            res=minimize(function_to_optimize,w0)

            w_sol = res.x.reshape((2,nb_channels,nb_channels*len(kernel_x)*len(kernel_y)))

            weights[index_ky_target]=np.apply_along_axis(lambda args: [complex(*args)], 0, w_sol)


        else :
            weights[index_ky_target]=F_target_calib@np.linalg.pinv(F_source_calib)


        F_estimate =weights[index_ky_target]@F_source_calib

        plt.figure()
        plt.plot(np.linalg.norm(F_target_calib-F_estimate,axis=0)/np.sqrt(nb_channels))
        plt.title("Calibration Error for line {} aggregated accross channels".format(index_ky_target))

        nplot_x = int(np.sqrt(nb_channels))
        nplot_y = int(nb_channels/nplot_x)+1

        # metric=np.real
        # fig, axs = plt.subplots(nplot_x, nplot_y)
        # fig.suptitle("Calibration Performance for line {} : Real Part".format(index_ky_target))
        # for i in range(nplot_x):
        #     for j in range(nplot_y):
        #         ch=i*nplot_y+j
        #         if ch >= nb_channels:
        #             break
        #         axs[i, j].plot(metric(F_target_calib[ch,:]),label="Target")
        #         axs[i, j].plot(metric(F_estimate[ch, :]), label="Estimate")
        #         axs[i,j].set_title('Channel {}'.format(ch))
        #         axs[i,j].legend(loc="upper right")
        #
        # metric = np.imag
        # fig, axs = plt.subplots(nplot_x, nplot_y)
        # fig.suptitle("Calibration Performance for line {} : Imaginary Part".format(index_ky_target))
        # for i in range(nplot_x):
        #     for j in range(nplot_y):
        #         ch = i * nplot_y + j
        #         if ch >= nb_channels:
        #             break
        #         axs[i, j].plot(metric(F_target_calib[ch, :]), label="Target")
        #         axs[i, j].plot(metric(F_estimate[ch, :]), label="Estimate")
        #         axs[i, j].set_title('Channel {}'.format(ch))
        #         axs[i, j].legend(loc="upper right")

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





    #ESTIMATION
    ky_lines_to_estimate=np.array(list(set(np.arange(npoint_x))-set(np.arange(npoint_x)[::undersampling])))
    ky_to_estimate=np.unique(traj_all[ky_lines_to_estimate,:,1])
    kx_line=traj_all[0,:,0]
    npoint_y_estimate=len(ky_lines_to_estimate)

    F_target_estimate = np.zeros((nb_channels,npoint_y_estimate,npoint_x),dtype=kdata_calib.dtype)

    for i,l in tqdm(enumerate(ky_lines_to_estimate)):
        for j in range(npoint_x):
            kx=kx_line[j]
            ky = ky_to_estimate[i]

            kx_comp = (kx - curr_traj_for_grappa[0, :, 0])
            kx_comp[kx_comp < 0] = np.inf

            kx_ref = curr_traj_for_grappa[0, np.argmin(kx_comp), 0]

            ky_comp = (ky - curr_traj_for_grappa[:, 0, 1])
            ky_comp[ky_comp < 0] = np.inf

            ky_ref = curr_traj_for_grappa[np.argmin(ky_comp), 0, 1]

            j0, i0 = np.unravel_index(
                np.argmin(np.linalg.norm(curr_traj_for_grappa - np.expand_dims(np.array([kx_ref, ky_ref]), axis=(0, 1)), axis=-1)),
                (npoint_y, npoint_x))
            local_indices = np.stack(np.meshgrid(pad_y[0]+j0 + np.array(kernel_y), pad_x[0]+i0 + np.array(kernel_x)), axis=0).T.reshape(-1, 2)

            local_kdata_for_grappa = kdata_all_channels_for_grappa_padded[:, local_indices[:, 0], local_indices[:, 1]]
            local_kdata_for_grappa = np.array(local_kdata_for_grappa)
            local_kdata_for_grappa = local_kdata_for_grappa.flatten()
            F_target_estimate[:,i,j]=weights[(l-1)%(undersampling-1)]@local_kdata_for_grappa


    kdata_all_channels_completed=np.concatenate([kdata_all_channels_for_grappa,F_target_estimate],axis=1)

    KX_, KY_ = np.meshgrid(kx_line, ky_to_estimate)
    curr_traj_estimate=np.stack([KX_, KY_], axis=-1)

    curr_traj_completed=np.concatenate([curr_traj_for_grappa,curr_traj_estimate],axis=0)

    kdata_all_channels_completed=kdata_all_channels_completed.reshape((nb_channels,-1))
    curr_traj_completed=curr_traj_completed.reshape((-1,2))


    image_rebuilt_all_channels_completed=[finufft.nufft2d1(curr_traj_completed[:,0], curr_traj_completed[:,1], k, image_size) for k in kdata_all_channels_completed]
    # for ch in range(nb_channels):
    #     plt.figure()
    #     plt.imshow(np.abs(image_rebuilt_all_channels_completed[ch]))



    image_rebuilt_completed=np.sqrt(np.sum(np.abs(image_rebuilt_all_channels_completed) ** 2, axis=0))



    plt.figure()
    plt.imshow(np.abs(image_rebuilt_completed))
    plt.title("With GRAPPA kernel_x {} kernel_y {} ".format(kernel_x,kernel_y))

    #Replace calibration lines

    curr_traj_completed_replace_calib=np.concatenate([curr_traj_for_grappa,curr_traj_estimate],axis=0)
    kdata_all_channels_completed_replace_calib=np.concatenate([kdata_all_channels_for_grappa,F_target_estimate],axis=1)

    for index_ky_target in range(len(calib_lines)):
        for j in range(npoint_x):
            kx=traj_calib_for_grappa[index_ky_target,j,0]
            ky = traj_calib_for_grappa[index_ky_target,j,1]
            j0, i0 = np.unravel_index(
                np.argmin(np.linalg.norm(curr_traj_completed_replace_calib - np.expand_dims(np.array([kx, ky]), axis=(0, 1)), axis=-1)),
                curr_traj_completed_replace_calib.shape[:-1])

            kdata_all_channels_completed_replace_calib[:,j0,i0] =kdata_calib_all_channels[:,index_ky_target,j]
            curr_traj_completed_replace_calib[j0,i0]=np.array([kx,ky])

    kdata_all_channels_completed_replace_calib=kdata_all_channels_completed_replace_calib.reshape((nb_channels,-1))
    curr_traj_completed_replace_calib=curr_traj_completed_replace_calib.reshape((-1,2))

    image_rebuilt_all_channels_completed_replace_calib=[finufft.nufft2d1(curr_traj_completed_replace_calib[:,0], curr_traj_completed_replace_calib[:,1], k, image_size) for k in kdata_all_channels_completed_replace_calib]
    image_rebuilt_completed_replace_calib=np.sqrt(np.sum(np.abs(image_rebuilt_all_channels_completed_replace_calib) ** 2, axis=0))

    plt.figure()
    plt.imshow(np.abs(image_rebuilt_completed_replace_calib))
    plt.title("With GRAPPA replace calib kernel_x {} kernel_y {}".format(kernel_x,kernel_y))


curr_traj_completed_replace_calib
kdata_all_channels_completed_replace_calib

ind=np.lexsort((curr_traj_completed_replace_calib[:, 0], curr_traj_completed_replace_calib[:, 1]))
curr_traj_completed_replace_calib= curr_traj_completed_replace_calib[ind,:]
kdata_all_channels_completed_replace_calib[:]=kdata_all_channels_completed_replace_calib[:,ind]


plot_next_slice=False
ch=3
sl = 129
metric=np.abs
plt.figure()
plt.title("Ch {} Sl {}".format(ch,sl))
curr_traj = traj_all_spokes
ind = np.lexsort((curr_traj[:,0], curr_traj[:,1]))
curr_kdata = kdata_all_channels_all_spokes[ch].flatten()
curr_kdata=curr_kdata[ind]
plt.plot(metric(curr_kdata.reshape(npoint_x,npoint_x)[sl]),label='Kdata fully sampled')
if sl in calib_lines:
    label="Calibration line"
elif sl%undersampling==0:
    label="Measured line"
else:
    label="Estimated line"
plt.plot(metric(kdata_all_channels_completed_replace_calib[ch].reshape(npoint_x,npoint_x)[sl]),label=label)

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






















#Check calib line vs estimated line

curr_traj_completed.reshape((255,256,-1))[191,:]
curr_traj_completed_replace_calib.reshape((255,256,-1))[191,:]

for ch in range(nb_channels):
    plt.figure()
    plt.title("Calib line replacement for channel {}".format(ch))
    plt.plot(np.abs(kdata_all_channels_completed.reshape((-1,255,256))[ch,191,:]),label="estimate")
    plt.plot(np.abs(kdata_all_channels_completed_replace_calib.reshape((-1,255,256))[ch,191,:]),label="measured calib line")
    plt.legend()

#Compare adjacent lines

estimated_line_index=100
ky_estimated=ky_to_estimate[estimated_line_index]


j0, i0 = np.unravel_index(
        np.argmin(np.linalg.norm(curr_traj_for_grappa - np.expand_dims(np.array([np.pi, ky_estimated]), axis=(0, 1)), axis=-1)),
        curr_traj_for_grappa.shape[:-1])

measured_line=kdata_all_channels_for_grappa[:,j0,:]
measured_line_next=kdata_all_channels_for_grappa[:,j0+1,:]

for ch in range(nb_channels):
    plt.figure()
    plt.title("Estimated line comparison for line {} and channel {}".format(estimated_line_index,ch))
    plt.plot(np.abs(measured_line[ch]),label="Previous measured line")
    plt.plot(np.abs(measured_line[ch]), label="Next measured line")
    plt.plot(np.abs(F_target_estimate[ch,estimated_line_index,:]), label="Estimated line")
    plt.legend()
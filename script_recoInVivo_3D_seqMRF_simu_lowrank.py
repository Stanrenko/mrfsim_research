

#import matplotlib
#matplotlib.u<se("TkAgg")
import matplotlib.pyplot as plt
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
name = "Knee3D_SimReco2"

dictfile="mrf_dictconf_SimReco2_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_reco3.dict"
dictfile_for_proj="mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_reco3.dict"

suffix="_DE_Simu_FF_reco3"
with open("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF.json") as f:
    sequence_config = json.load(f)

# dictfile="mrf_SimReco2_light_adjusted.dict"
# suffix=""
# with open("./mrf_sequence_adjusted.json") as f:
#     sequence_config = json.load(f)

#dictfile="mrf175_SimReco2_adjusted.dict"
#suffix="_fullReco"
#with open("./mrf_sequence_adjusted.json") as f:
#    sequence_config = json.load(f)

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

nrep=2
rep=nrep-1
TR_total = np.sum(sequence_config["TR"])

Treco = TR_total-np.sum(sequence_config["TR"])
Treco=3000
##other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=nrep
sequence_config["rep"]=rep

seq=T1MRFSS(**sequence_config)

#seq=T1MRFSS_NoInv(**sequence_config)
#seq=T1MRF(**sequence_config)




nb_filled_slices = 4
nb_empty_slices=1
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
filename_volume_optim = filename+"_volumes_optim_sl{}_rp{}_us{}_{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)


filename_kdata = filename+"_kdata_sl{}_rp{}_us{}{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)
filename_mask= filename+"_mask_sl{}_rp{}_us{}{}w{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)
file_map = filename + "_sl{}_rp{}_us{}{}w{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,nb_allspokes,nspoke,suffix)

#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"

nb_channels=1
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
     file_matlab_paramMap = "./data/KneePhantom/Phantom{}/paramMap.mat".format(num)

     m = MapFromFile3D(name,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,image_size=image_size,file=file_matlab_paramMap,rounding=True,gen_mode="other",undersampling_factor=undersampling_factor,resting_time=4000)

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



if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    m.build_ref_images(seq)

if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
    volumes_ideal=m.images_series[:,mask>0]
    volumes_ideal=[makevol(np.mean(arr,axis=0),mask>0) for arr in groupby(volumes_ideal,nspoke)]
    volumes_ideal=np.array(volumes_ideal)
    np.save(filename_groundtruth,volumes_ideal)

volumes_ideal = np.load(filename_groundtruth)

#animate_images(m.images_series[::nspoke,int(nb_slices/2)])
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



##volumes for slice taking into account coil sensi
print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):

    density_adj=True
    volumes_all=simulate_radial_undersampled_images(data,radial_traj,image_size,density_adj=density_adj,useGPU=use_GPU,ntimesteps=ntimesteps)
    np.save(filename_volume,volumes_all)
    # sl=20
    # ani = animate_images(volumes_all[:,sl,:,:])
    del volumes_all

##volumes for slice taking into account coil sensi
print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    np.save(filename_mask,m.mask)

volumes_all=np.load(filename_volume)
#ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])


def J(m):
    global traj
    global data
    global ntimesteps
    global curr_ntimesteps
    global npoint
    global density_adj
    global useGPU

    FU=[]
    if m.dtype=="complex64":
        traj_kdata=traj.astype("float32")
    else:
        traj_kdata = copy(traj)

    if not(useGPU):
        for j in tqdm(range(curr_ntimesteps)):
            FU.append(finufft.nufft3d2(traj_kdata[j,:, 2],traj_kdata[j,:, 0], traj_kdata[j,:, 1],m[j]))

    else:
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64
        N1, N2, N3 = m.shape[1], m.shape[2], m.shape[3]
        #M = traj.shape[0]
        #c_gpu = GPUArray((M), dtype=complex_dtype)
        FU = []
        for i in tqdm(range(curr_ntimesteps)):
            fk = m[i, :, :, :]
            kx = traj[i,:, 0]
            ky = traj[i,:, 1]
            kz = traj[i,:, 2]

            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            fk = fk.astype(complex_dtype)
            c_gpu = GPUArray((kx.shape[0]), dtype=complex_dtype)

            plan = cufinufft(2, (N1, N2, N3), 1, eps=1e-6, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))
            plan.execute(c_gpu, to_gpu(fk))
            c = np.squeeze(c_gpu.get())
            c_gpu.gpudata.free()
            FU.append(c)
            plan.__del__()

    FU=np.array(FU)

    kdata_error = FU - data.reshape(ntimesteps,-1)[:curr_ntimesteps]
    #kdata_ratio = FU/data.reshape(ntimesteps,-1)[:curr_ntimesteps]

    # return np.linalg.norm(kdata_error)**2
    kdata_error/=np.prod(m.shape[1:])
    #kdata_error /= np.sqrt(np.prod(data.shape[1:]))
    if density_adj:
        kdata_error = kdata_error.reshape(-1, npoint)
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_error.ndim - 1)))
        kdata_error *= np.sqrt(density)


    return np.linalg.norm(kdata_error) ** 2


def grad_J(m):
    global traj
    global data
    global ntimesteps
    global npoint
    global curr_ntimesteps
    global nspoke
    global nb_slices
    global density_adj
    global useGPU

    image_size=m.shape[1:]
    FU = []
    if m.dtype == "complex64":
        traj_kdata = traj.astype("float32")
    else:
        traj_kdata=copy(traj)

    if not (useGPU):
        for j in tqdm(range(curr_ntimesteps)):
            FU.append(finufft.nufft3d2(traj_kdata[j, :, 2], traj_kdata[j, :, 0], traj_kdata[j, :, 1], m[j]))
    else:
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64
        N1, N2, N3 = m.shape[1], m.shape[2], m.shape[3]
        #M = traj[0].shape[0]
        #c_gpu = GPUArray((M), dtype=complex_dtype)
        FU = []
        for i in tqdm(range(curr_ntimesteps)):
            fk = m[i, :, :, :]
            kx = traj_kdata[i,:, 0]
            ky = traj_kdata[i,:, 1]
            kz = traj_kdata[i,:, 2]

            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            fk = fk.astype(complex_dtype)

            c_gpu = GPUArray((kx.shape[0]), dtype=complex_dtype)

            plan = cufinufft(2, (N1, N2, N3), 1, eps=1e-6, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))
            plan.execute(c_gpu, to_gpu(fk))
            c = np.squeeze(c_gpu.get())
            c_gpu.gpudata.free()
            FU.append(c)
            plan.__del__()

    FU = np.array(FU)

    kdata_error = FU - data.reshape(ntimesteps,-1)[:curr_ntimesteps]

    kdata_error/=np.prod(m.shape[1:])**2
    #kdata_error /= data.shape[1:]
    # return np.linalg.norm(kdata_error)**2
    if density_adj:
        kdata_error = kdata_error.reshape(-1, npoint)
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_error.ndim - 1)))
        kdata_error*=density
        kdata_error=kdata_error.reshape(curr_ntimesteps,-1)


    dm=[]

    if not(useGPU):
        for j in tqdm(range(curr_ntimesteps)):
            dm.append(finufft.nufft3d1(traj[j,:, 2], traj[j,:, 0], traj[j,:, 1], kdata_error[j], image_size))

    else:
        N1, N2, N3 = image_size[0], image_size[1], image_size[2]
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64

        for i in tqdm(range(curr_ntimesteps)):
            fk_gpu = GPUArray(( N1, N2, N3), dtype=complex_dtype)
            c_retrieved = kdata_error[i]
            kx = traj[i,:, 0]
            ky = traj[i,:, 1]
            kz = traj[i,:, 2]
            #
            # print(fk_gpu.shape)
            # print(kx.shape)
            # print(c_retrieved.shape)

            # Cast to desired datatype.
            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            c_retrieved = c_retrieved.astype(complex_dtype)

            # Allocate memory for the uniform grid on the GPU.
            c_retrieved_gpu = to_gpu(c_retrieved)

            # Initialize the plan and set the points.
            plan = cufinufft(1, (N1, N2, N3),1, eps=1e-6, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

            # Execute the plan, reading from the strengths array c and storing the
            # result in fk_gpu.
            plan.execute(c_retrieved_gpu, fk_gpu)

            dm.append(np.squeeze(fk_gpu.get()))

            fk_gpu.gpudata.free()
            c_retrieved_gpu.gpudata.free()

            plan.__del__()


    return 2*np.array(dm)

def psi(t,alpha=1e-8):
    return np.sqrt(t+alpha**2)

def delta(m,axis):
    return np.diff(m,axis=1+axis,append=0)

def delta_adjoint(m,axis):
    return -np.diff(m, axis=1 + axis, prepend=0)

def J_TV(m):
    global axis
    global alpha
    global mask
    global is_weighted
    global weights
    #global delta_m
    delta_m = delta(m, axis)



    if is_weighted:
        bound_inf = np.min(np.argwhere(mask>0)[:,axis])
        bound_sup = np.max(np.argwhere(mask > 0)[:, axis])
        weights=np.ones(delta_m.shape,dtype=delta_m.dtype)
        idx=[np.s_[:], np.s_[:],np.s_[:],np.s_[:]]
        idx[axis+1]=np.s_[:bound_inf]
        weights[tuple(idx)]=0

        idx = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        idx[axis + 1] = np.s_[bound_sup:]
        weights[tuple(idx)] = 0

    else:
        weights=1


    return np.sum(weights*psi(np.abs(delta_m)**2,alpha))

def grad_J_TV(m):
    global axis
    global alpha
    global is_weighted
    #global W
    #global delta_m

    delta_m = delta(m,axis)

    if is_weighted:
        bound_inf = np.min(np.argwhere(mask > 0)[:, axis])
        bound_sup = np.max(np.argwhere(mask > 0)[:, axis])
        weights = np.ones(delta_m.shape, dtype=delta_m.dtype)
        idx = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        idx[axis + 1] = np.s_[:bound_inf]
        weights[tuple(idx)] = 0

        idx = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        idx[axis + 1] = np.s_[bound_sup:]
        weights[tuple(idx)] = 0

    else:
        weights = 1

    W = psi(np.abs(delta_m)**2,alpha)

    return delta_adjoint(weights*delta_m/W,axis)


mask=m.mask
curr_ntimesteps=ntimesteps
traj=radial_traj.get_traj_for_reconstruction(ntimesteps)
X0=np.zeros(volumes_all.shape,dtype=volumes_all.dtype)
density_adj=True

useGPU=False
J(volumes_all)

useGPU=True
J(volumes_all)

useGPU=True
grad_gpu = grad_J(volumes_all)

useGPU=False
grad = grad_J(volumes_all)

is_weighted=True
eps=np.zeros(volumes_all.shape,dtype=volumes_all.dtype)
sl=np.random.choice(range(image_size[0]))

point1=np.random.choice(range(image_size[-1]))
point2=np.random.choice(range(image_size[-1]))

eps_value=0.0001*1j
eps[10,sl,point1,point2]=eps_value
alpha=1e-13

print(volumes_all[10,sl,point1,point2]-volumes_all[10,sl-1,point1,point2])
print(volumes_all[10,sl+1,point1,point2]-volumes_all[10,sl,point1,point2])

dJ=(J_TV(volumes_all+eps)-J_TV(volumes_all))
grad_TV = np.real(np.dot(grad_J_TV(volumes_all).flatten().conj(),eps.flatten()))
print(dJ)
print(grad_TV)

grad_TV = grad_J_TV(np.abs(volumes_all))*

animate_images((delta_m/W)[10,:,:,:])

np.dot(grad_TV.flatten(),eps.flatten())
(delta_m/W)[10,:,:,:][(delta_m/W)[10,:,:,:]<0]

animate_images(grad_TV[10,:,:,:])
J_TV(volumes_ideal)




import dask.array as da

mask=m.mask

traj=radial_traj.get_traj_for_reconstruction(ntimesteps)
X0=np.zeros(volumes_all.shape,dtype=volumes_all.dtype)

mu=1
lambd=1
curr_ntimesteps=ntimesteps
keys,values=read_mrf_dict(dictfile,np.arange(0.0,1.01,0.1))
t0=1

X=X0
M_prev=X0[:,mask>0]
t=t0
Z =X[:,m.mask>0]- lambd*grad_J(X)[:,m.mask>0]

Z_proj=(values.T@values.conj())@Z


u, s, vh =da.linalg.svd(da.from_array(Z_proj))
u=np.array(u)
s=np.array(s)
vh=np.array(vh)

s[s<lambd*mu]=0
s[s>lambd*mu]=s[s>lambd*mu]-lambd*mu

M = u@np.diag(s)@vh

t_prev=t
t=(1+np.sqrt(1+4*t**2))/2

X = M + (t_prev-1)/(t+1) * (M - M_prev)
X = np.array([makevol(im,mask>0) for im in X ])
M_prev=M








import dask.array as da


mask=m.mask

traj=radial_traj.get_traj_for_reconstruction(ntimesteps)
X0=np.zeros(volumes_all.shape,dtype=volumes_all.dtype)
useGPU=False


mu=1
lambd=0.05
mu_TV=0
curr_ntimesteps=ntimesteps
keys,values=read_mrf_dict(dictfile_for_proj,FF_list=np.arange(0.,1.01,0.05),aggregate_components=True)
t0=1

is_weighted=True
alpha=1e-13
axis=0

D = values

X=X0
M_prev=X0[:,mask>0]
t=t0
D_plus=np.linalg.pinv(D.T)
i=0
density_adj=True

Z_costs = []
X_grads = []
X_grads_TV=[]
X_list=[]
niter_lowrank = 100


for i in tqdm(range(niter_lowrank)):
    print("############### ITER {} ################".format(i))
    grad=grad_J(X)
    grad_reg = grad_J_TV(X)
    norm_grad=np.linalg.norm(grad)
    norm_grad_reg = np.linalg.norm(grad_reg)
    #if norm_grad_reg==0:
    #    Z = X - mu * grad / norm_grad
    #else:
    #    Z = X - mu * grad/norm_grad-mu_TV*grad_reg/norm_grad_reg

    if norm_grad_reg==0:
        Z = X - mu * grad/norm_grad
    else:
        Z = X - mu * grad/norm_grad-mu_TV*grad_reg/norm_grad_reg
    #print(np.max(np.abs(grad)/norm_grad));print(np.max(np.abs(grad_reg)/norm_grad_reg));

    J_current=J(Z)

    Z_costs.append(J_current)
    X_grads.append(norm_grad)
    X_grads_TV.append(norm_grad_reg)

    #animate_images(Z[:,int(nb_slices/2),:,:])
    print("COST : {}".format(J_current))
    Z = Z[:,mask>0]

    Z_proj = (D.T@ D_plus) @ Z

    #Z_proj_ridge=D.T@(+0.5*np.eye(D.shape))

    #j=np.random.choice(Z.shape[1]);plt.figure();plt.plot(Z[:,j]);plt.plot(Z_proj[:,j])
    #

    u, s, vh = da.linalg.svd(da.from_array(Z_proj))
    u = np.array(u)
    s = np.array(s)
    vh = np.array(vh)

    lambd=s[10]/mu
    print("Lambd: {}".format(lambd))
    s[s < lambd * mu] = 0
    s[s > lambd * mu] = s[s > lambd * mu] - lambd * mu

    M = u @ np.diag(s) @ vh

    #j = np.random.choice(Z.shape[1]);plt.figure();plt.plot(Z[:, j]/np.linalg.norm(Z[:, j]));plt.plot(Z_proj[:, j]/np.linalg.norm(Z_proj[:, j]));plt.plot(M[:, j]/np.linalg.norm(M[:, j]));plt.plot(X_target[:,j]/np.linalg.norm(X_target[:,j]));
    #j = 9814;plt.figure();plt.plot(Z[:, j]/np.linalg.norm(Z[:, j]));plt.plot(Z_proj[:, j]/np.linalg.norm(Z_proj[:, j]));plt.plot(M[:, j]/np.linalg.norm(M[:, j]));plt.plot(X_target[:,j]/np.linalg.norm(X_target[:,j]));
    # j = 31325;plt.figure();plt.plot(Z[:, j]/np.linalg.norm(Z[:, j]));plt.plot(Z_proj[:, j]/np.linalg.norm(Z_proj[:, j]));plt.plot(M[:, j]/np.linalg.norm(M[:, j]));plt.plot(X_target[:,j]/np.linalg.norm(X_target[:,j]));

    t_prev = t
    t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2

    X = M + (t_prev - 1) / (t + 1) * (M - M_prev)
    X = np.array([makevol(im, mask > 0) for im in X])
    M_prev = M

    if i%10==0:
        X_list.append(X)

X_list.append(X)

np.save(filename_volume_optim,X_list[-1])

plt.figure()
plt.plot(Z_costs)

plt.figure()
plt.plot(X_grads)

plt.figure()
plt.plot(X_grads_TV)

plt.figure();plt.plot(np.abs(grad_reg[15,:,128,128]))

plt.figure();plt.plot([J_TV(x) for x in X_list])


X_list=np.array(X_list)
animate_images(X_list[:,10,3,:,:])

plt.figure()
plt.imshow(np.abs(X_list[4,10,1,:,:]))


X_rebuilt=volumes_all[:,mask>0]
X_rebuilt -= np.mean(X_rebuilt,axis=0)

X_final=X_list[5][:,mask>0]
X_final-= np.mean(X_final,axis=0)
cov_final=np.real(np.sum(X_final.conj()*X_rebuilt,axis=0))
corr_final=cov_final/np.linalg.norm(X_final,axis=0)**2
X_final*=corr_final.reshape(1,-1)
#X_final/=np.linalg.norm(X_final,axis=0)
X_target=volumes_ideal[:,mask>0]
X_target-= np.mean(X_target,axis=0)
cov_target=np.real(np.sum(X_target.conj()*X_rebuilt,axis=0))
corr_target=cov_target/np.linalg.norm(X_target,axis=0)**2
X_target*=corr_target.reshape(1,-1)
#X_target/=np.linalg.norm(X_target,axis=0)


metric=np.imag
plt.figure()
j=np.random.choice(range(X_final.shape[1]))
plt.plot(metric(X_final[:,j]),label="MFLOR")
plt.plot(metric(X_target[:,j]),label="Target")
plt.plot(metric(X_rebuilt[:,j]),label="NUFFT")
plt.axvline(x=62)
#plt.plot(metric(D[0,:]),label="Projection Dico")
plt.title("Fingerprints {}".format(j))
plt.legend()

error_fingerprints=np.linalg.norm(X_final-X_target,axis=0)
ind_max_error_fingerprint = np.argsort(error_fingerprints)[-1]
mask_error = np.zeros(mask.shape)
mask_error=mask_error[mask>0]
mask_error[ind_max_error_fingerprint]=1
mask_error=makevol(mask_error,mask>0)
sl_max,pt1_max,pt2_max=np.unravel_index(np.argmax(mask_error),mask.shape)

#plt.close("all")
metric=np.imag
plt.figure()
plt.plot(metric(X_final[:,ind_max_error_fingerprint]),label="MFLOR")
plt.plot(metric(X_target[:,ind_max_error_fingerprint]),label="Target")
plt.plot(metric(X_rebuilt[:,ind_max_error_fingerprint]),label="NUFFT")
plt.title("Fingerprint with max MFLOR error {}".format((sl_max,pt1_max,pt2_max)))
plt.legend()



plt.figure()
plt.plot(np.sort(error_fingerprints))

ind_max_errors_fingerprint=np.argwhere(error_fingerprints>0.015)
mask_max_errors_fingerprint=np.zeros(mask.shape)
mask_max_errors_fingerprint=mask_max_errors_fingerprint[mask>0]
mask_max_errors_fingerprint[ind_max_errors_fingerprint]=1
mask_max_errors_fingerprint=makevol(mask_max_errors_fingerprint,mask>0)

animate_images(mask_max_errors_fingerprint)

wT1_on_errors=m.paramMap["wT1"][ind_max_errors_fingerprint]
print(np.unique(wT1_on_errors))
pd.DataFrame(wT1_on_errors).describe()


ff_on_errors=m.paramMap["ff"][ind_max_errors_fingerprint]
print(np.unique(ff_on_errors))
pd.DataFrame(ff_on_errors).describe()



ind_min_errors_fingerprint=np.argwhere(error_fingerprints<0.02)
mask_min_errors_fingerprint=np.zeros(mask.shape)
mask_min_errors_fingerprint=mask_min_errors_fingerprint[mask>0]
mask_min_errors_fingerprint[ind_min_errors_fingerprint]=1
mask_min_errors_fingerprint=makevol(mask_min_errors_fingerprint,mask>0)

animate_images(mask_min_errors_fingerprint)

animate_multiple_images(mask_min_errors_fingerprint,mask_max_errors_fingerprint)
plt.figure()
plt.imshow(makevol(m.paramMap["ff"],mask>0)[int(nb_slices/2)])
plt.colorbar()
plt.figure()
plt.imshow(makevol(m.paramMap["wT1"],mask>0)[int(nb_slices/2)])
plt.colorbar()

wT1_on_errors_min=m.paramMap["wT1"][ind_min_errors_fingerprint]
print(np.unique(wT1_on_errors_min))
pd.DataFrame(wT1_on_errors_min).describe()

ff_on_errors_min=m.paramMap["ff"][ind_min_errors_fingerprint]
print(np.unique(ff_on_errors_min))
pd.DataFrame(ff_on_errors_min).describe()

np.unique(np.array(keys)[:,0])

plt.close("all")
plt.figure()
n,bins,patches=plt.hist(m.paramMap["wT1"],stacked=True)
plt.hist(wT1_on_errors,stacked=True,alpha=0.5,bins=bins)
plt.hist(wT1_on_errors_min,alpha=0.5,stacked=True,bins=bins)


plt.figure()
n,bins,patches=plt.hist(m.paramMap["ff"],stacked=True)
plt.hist(ff_on_errors,stacked=True,alpha=0.5,bins=bins)
plt.hist(ff_on_errors_min,alpha=0.5,stacked=True,bins=bins)


wT1_ff_on_errors=np.squeeze(np.stack([wT1_on_errors,ff_on_errors],axis=-1))
print(pd.DataFrame(wT1_ff_on_errors,columns=["wT1","ff"]).groupby(["wT1","ff"]).size())

wT1_ff_on_errors_min=np.squeeze(np.stack([wT1_on_errors_min,ff_on_errors_min],axis=-1))
print(pd.DataFrame(wT1_ff_on_errors_min,columns=["wT1 min","ff min"]).groupby(["wT1 min","ff min"]).size())

wT1_ff=np.squeeze(np.stack([m.paramMap["wT1"],m.paramMap["ff"]],axis=-1))
print(pd.DataFrame(wT1_ff,columns=["wT1 map","ff map"]).groupby(["wT1 map","ff map"]).size())


error_ts=np.linalg.norm(X_final-X_target,axis=1)
plt.figure()
j=np.random.choice(range(X_final.shape[1]))
plt.plot(error_ts/np.linalg.norm(error_ts))
plt.plot(X_target[:,j]/np.linalg.norm(X_target[:,j]))
plt.plot(np.array(sequence_config["TE"])[::nspoke]/np.linalg.norm(np.array(sequence_config["TE"])[::nspoke]))
plt.plot(np.array(sequence_config["B1"])[::nspoke]/np.linalg.norm(np.array(sequence_config["B1"])[::nspoke]))

ind_max_error_ts=np.argmax(error_ts)

ts = np.random.choice(range(ntimesteps))
#ts=62
ts=ind_max_error_ts
metric=np.abs
point1 = int(image_size[1]/2)+np.random.choice(range(-30,30))
point2 = int(image_size[2]/2)+np.random.choice(range(-30,30))
#point1=71
#point2=150
plt.figure()
j=np.random.choice(range(X_final.shape[1]))
plt.plot(metric(X_list[-1][ts,:,point1,point2]/np.linalg.norm(X_list[-1][ts,:,point1,point2])),label="MFLOR")
plt.plot(metric(volumes_ideal[ts,:,point1,point2]/np.linalg.norm(volumes_ideal[ts,:,point1,point2])),label="Target")
plt.plot(metric(volumes_all[ts,:,point1,point2]/np.linalg.norm(volumes_all[ts,:,point1,point2])),label="NUFFT")
plt.title("Slice profile ts {} point {}".format(ts,(point1,point2)))
plt.legend()

animate_images(X_list[-1][ind_max_error_ts,:,:,:])
animate_images(volumes_ideal[ind_max_error_ts,:,:,:])
animate_images(volumes_all[ind_max_error_ts,:,:,:])

delta_m = delta(X_list[-1], axis)



if is_weighted:
    bound_inf = np.min(np.argwhere(mask>0)[:,axis])
    bound_sup = np.max(np.argwhere(mask > 0)[:, axis])
    weights=np.ones(delta_m.shape,dtype=delta_m.dtype)
    idx=[np.s_[:], np.s_[:],np.s_[:],np.s_[:]]
    idx[axis+1]=np.s_[:bound_inf]
    weights[tuple(idx)]=0

    idx = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
    idx[axis + 1] = np.s_[bound_sup:]
    weights[tuple(idx)] = 0

else:
    weights=1


test_J_TV=weights*psi(np.abs(delta_m)**2,alpha)
ind_max=np.unravel_index(np.argmax(test_J_TV),test_J_TV.shape)

mask[ind_max[1],ind_max[2],ind_max[3]]

J(X_list[-1])
J_TV(X_list[-1])

J(X_list[0])
J_TV(X_list[0])


animate_multiple_images(X_list[-1][:,6,:,:],volumes_ideal[:,6,:,:])




########################## Dict mapping ########################################
#
# with open("mrf_sequence{}.json".format(suffix)) as f:
#      sequence_config = json.load(f)


seq = None


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
volumes_all = np.load(filename_volume)




if not(load_map):
    niter = 0
    #optimizer = BruteDictSearch(FF_list=np.arange(0,1.01,0.05),mask=mask,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=False,ntimesteps=ntimesteps,log_phase=True)
    #all_maps = optimizer.search_patterns(dictfile, volumes_all, retained_timesteps=None)


    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=100, pca=True,threshold_pca=10, log=False, useGPU_dictsearch=False, useGPU_simulation=False,gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps)
    all_maps=optimizer.search_patterns_test(dictfile,X_list[5],retained_timesteps=None)

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
    file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
    file = open(file_map, "rb")
    all_maps = pickle.load(file)


maskROI=buildROImask_unique(m.paramMap)
regression_paramMaps_ROI(m.paramMap,all_maps[0][0],m.mask>0,all_maps[0][1]>0,maskROI,adj_wT1=False,title="optim_regROI_"+str.split(str.split(filename_volume,"/")[-1],".npy")[0],save=True)

#regression_paramMaps(m.paramMap,all_maps[0][0],mode="Boxplot")


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


plt.close("all")

name="SquareSimu3D_SS_SimReco2"
#name="SquareSimu3D_SS_FF0_1"
dic_maps={}
list_suffix=["fullReco_T1MRF_adjusted","fullReco_Brute","DE_Simu_FF_reco3","DE_Simu_FF_v2_reco3"]
list_suffix=["fullReco","DE_Simu_FF_reco3","DE_Simu_FF_reco3_medfilter_3","DE_Simu_FF_v2_reco3","DE_Simu_FF_NoInv"]

for suffix in list_suffix:
    if "NoInv" in suffix:
        file_map = "/{}_sl{}_rp{}_us{}{}w{}_{}_MRF_map.pkl".format(name, nb_slices, repeat_slice, undersampling_factor,
                                                                   680, nspoke, suffix)
    elif "DE_Simu" in suffix:
        file_map = "/{}_sl{}_rp{}_us{}{}w{}_{}_MRF_map.pkl".format(name, nb_slices, repeat_slice, undersampling_factor,
                                                                   760, nspoke, suffix)

    else:
        file_map="/{}_sl{}_rp{}_us{}{}w{}_{}_MRF_map.pkl".format(name,nb_slices,repeat_slice,undersampling_factor,1400,nspoke,suffix)
    with open( base_folder + file_map, "rb") as file:
        dic_maps[file_map] = pickle.load(file)

k="wT1"
maskROI=buildROImask_unique(m.paramMap,key=k)
fig,ax=plt.subplots(1,2)
plt.title(k)
for key in dic_maps.keys():
    roi_values=get_ROI_values(m.paramMap,dic_maps[key][0][0],m.mask>0,dic_maps[key][0][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
    roi_values.sort_values(by=["Obs Mean"],inplace=True)
    #dic_roi_values[key]=roi_values
    ax[0].plot(roi_values["Obs Mean"],roi_values["Pred Mean"].values,label=key)
    ax[1].plot(roi_values["Obs Mean"],roi_values["Pred Std"].values,label=key)

ax[0].plot(roi_values["Obs Mean"],roi_values["Obs Mean"],linestyle="--")
plt.legend()

k="ff"
maskROI=buildROImask_unique(m.paramMap,key=k)
fig,ax=plt.subplots(1,2)
plt.title(k)
for key in dic_maps.keys():
    roi_values=get_ROI_values(m.paramMap,dic_maps[key][0][0],m.mask>0,dic_maps[key][0][1]>0,return_std=True,adj_wT1=False,maskROI=maskROI)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
    roi_values.sort_values(by=["Obs Mean"],inplace=True)
    #dic_roi_values[key]=roi_values
    ax[0].plot(roi_values["Obs Mean"],roi_values["Pred Mean"].values,label=key)
    ax[1].plot(roi_values["Obs Mean"],roi_values["Pred Std"].values,label=key)


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

dictfile="mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_reco3.dict"
suffix="_DE_Simu_FF_reco3"
with open("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF.json") as f:
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









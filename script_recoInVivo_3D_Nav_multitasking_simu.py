

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

nb_filled_slices = 4
nb_empty_slices=0
repeat_slice=4
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

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
filename_volume = filename+"_volumes_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_volume_all_spokes = filename+"_volumes_all_spokes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_b1_all_spokes = filename+"_b1_all_spokes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_groundtruth = filename+"_groundtruth_volumes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")

filename_kdata = filename+"_kdata_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_kdata_all_spokes = filename+"_kdata_all_spokes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")
filename_mask= filename+"_mask_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
file_map = filename + "_sl{}_rp{}_us{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,suffix)
filename_b1 = filename+ "_b1_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_b1_grappa = filename+ "_b1_grappa_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")


filename_kdata_grappa = filename+"_kdata_grappa_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_currtraj_grappa = filename+"_currtraj_grappa_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask_grappa= filename+"_mask_grappa_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_volume_grappa = filename+"_volumes_grappa_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

file_map_grappa = filename + "_grappa_sl{}_rp{}_us{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,suffix)
file_map_all_spokes = filename + "_all_spokes_sl{}_rp{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,suffix)



#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 512



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

nb_channels=1


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):

    #images = copy(m.images_series)
    data=m.generate_kdata(radial_traj)

    data=np.array(data)
    np.save(filename_kdata, data)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    data = np.load(filename_kdata)


dictfile = "./mrf175_SimReco2.dict"
ind_dico = 8

filename_dico_comp = str.split(dictfile,".dict") [0]+"_phi_dico_{}comp.npy".format(ind_dico)

if str.split(filename_dico_comp,"/")[-1]  not in os.listdir():

    FF_list = list(np.arange(0., 1.05, 0.05))
    keys, signal = read_mrf_dict(dictfile, FF_list)

    import dask.array as da

    A_r=signal.real
    A_i=signal.imag

    X_1 = np.concatenate([A_r,-A_i],axis=-1)
    X_2 = np.concatenate([A_i,A_r],axis=-1)
    X=np.concatenate([X_1,X_2],axis=0)

    u_dico, s_dico, vh_dico = da.linalg.svd(da.from_array(X))

    vh_dico=np.array(vh_dico[::2,:])
    s_dico=np.array(s_dico[::2])

    # plt.figure()
    # plt.plot(np.cumsum(s_dico)/np.sum(s_dico))

    #ind_dico = ((np.cumsum(s_dico)/np.sum(s_dico))<0.99).sum()
    #ind_dico=20

    vh_dico_retained = vh_dico[:ind_dico,:]
    phi_dico = vh_dico_retained[:,:ntimesteps] - 1j * vh_dico_retained[:,ntimesteps:]

    del u_dico
    del s_dico
    del vh_dico

    del vh_dico_retained
    del X_1
    del X_2
    del X
    #del signal


    np.save(filename_dico_comp,phi_dico)
else:
    filename_dico_comp = str.split(dictfile,".dict") [0]+"_phi_dico_{}comp.npy".format(ind_dico)
    phi_dico=np.load(filename_dico_comp)


ngroups=1
#
# data_for_svd = data.reshape(ntimesteps,-1)
# data_for_svd = data_for_svd.T
#
# u, s, vh = np.linalg.svd(data_for_svd, full_matrices=False)

phi=phi_dico
L0 = phi.shape[0]

#data = np.load(filename_save)
m_shape = (L0,)+image_size
m0=np.zeros(m_shape,dtype=data.dtype)
traj=radial_traj.get_traj_for_reconstruction()

if m0.dtype == "complex64":
    try:
        traj = traj.astype("float32")
    except:
        pass

traj=traj.reshape(-1,3)

data_mask = np.ones((nb_channels, 8, nb_slices, ngroups, ntimesteps))



def J(m):
    global L0
    global phi
    global traj
    global ntimesteps
    global data
    global nb_slices
    global nb_channels
    global npoint
    print(m.dtype)
    FU = finufft.nufft3d2(traj[:, 2],traj[:, 0], traj[:, 1], m)

    FU=FU.reshape(L0,ntimesteps,-1)
    FU=np.moveaxis(FU,0,-1)
    phi = phi.reshape(L0,-1,ntimesteps)
    ngroups=phi.shape[1]
    kdata_model=[]
    for ts in tqdm(range(ntimesteps)):
        kdata_model.append(FU[ts]@phi[:,:,ts])
    kdata_model=np.array(kdata_model)

    kdata_model=kdata_model.reshape(ntimesteps,8,nb_slices,npoint,ngroups)
    kdata_model=np.expand_dims(kdata_model,axis=0)
    kdata_model_retained = np.zeros(kdata_model.shape[:-1],dtype=data.dtype)

    for ts in tqdm(range(ntimesteps)):
        for sl in range(nb_slices):
            for sp in range(8):
                for g in range(ngroups):
                    if data_mask[0,sp,sl,g,ts]:
                        kdata_model_retained[:,ts,sp,sl,:]=kdata_model[:,ts,sp,sl,:,g]

    kdata_error = kdata_model_retained-data.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
    return np.linalg.norm(kdata_error)**2

def grad_J(m):
    global L0
    global phi
    global traj
    global ntimesteps
    global data
    global nb_slices
    global nb_channels
    global npoint
    global image_size



    FU = finufft.nufft3d2(traj[:, 2], traj[:, 0], traj[:, 1], m)

    FU = FU.reshape(L0, ntimesteps, -1)
    FU = np.moveaxis(FU, 0, -1)
    phi = phi.reshape(L0, -1, ntimesteps)
    ngroups = phi.shape[1]
    kdata_model = []
    for ts in tqdm(range(ntimesteps)):
        kdata_model.append(FU[ts] @ phi[:, :, ts])
    kdata_model = np.array(kdata_model)

    kdata_model = kdata_model.reshape(ntimesteps, 8, nb_slices, npoint, ngroups)
    kdata_model = np.expand_dims(kdata_model, axis=0)
    kdata_model_retained = np.zeros(kdata_model.shape[:-1], dtype=data.dtype)

    for ts in tqdm(range(ntimesteps)):
        for sl in range(nb_slices):
            for sp in range(8):
                for g in range(ngroups):
                    if data_mask[0, sp, sl, g, ts]:
                        kdata_model_retained[:, ts, sp, sl, :] = kdata_model[:, ts, sp, sl, :, g]

    kdata_error = kdata_model_retained - data.reshape(nb_channels, ntimesteps, -1, nb_slices, npoint)

    kdata_error_phiH = np.zeros(kdata_model.shape[:-1] + (L0,), dtype=kdata_error.dtype)
    # kdata_error_reshaped=np.zeros(kdata_model.shape+(ntimesteps,),dtype=kdata_model.dtype)

    # phi_H = phi.conj().reshape(L0,-1).T

    for ts in tqdm(range(ntimesteps)):
        for sl in range(nb_slices):
            for sp in range(8):
                for g in range(ngroups):
                    if data_mask[0, sp, sl, g, ts]:
                        # kdata_error_reshaped[:, ts, sp, sl, :,g,ts] = kdata_error[:, ts, sp, sl, :]
                        for l in range(L0):
                            kdata_error_phiH[:, ts, sp, sl, :, l] = kdata_error[:, ts, sp, sl, :] * phi.conj()[l, g, ts]

    # phi_H = phi.conj().reshape(L0,-1).T
    # kdata_error_reshaped=kdata_error_reshaped.reshape(-1,ngroups*ntimesteps)

    kdata_error_phiH = np.moveaxis(kdata_error_phiH, -1, 0)
    density = np.abs(np.linspace(-1, 1, npoint))
    density = np.expand_dims(density, tuple(range(kdata_error_phiH.ndim - 1)))
    kdata_error_phiH *= density

    dtheta = np.pi / (8*ntimesteps)
    dz = 1 / nb_slices

    kdata_error_phiH *= 1 / (2 * npoint) * dz * dtheta

    kdata_error_phiH = kdata_error_phiH.reshape(L0 * nb_channels, -1)

    dm = finufft.nufft3d1(traj[:, 2], traj[:, 0], traj[:, 1], kdata_error_phiH, image_size)
    #dm = dm/np.linalg.norm(dm)

    return 2*dm


m0 = np.zeros(m_shape,dtype=data.dtype)
grad_Jm= grad_J(m0)
#J_m= J(m0)
m0 = m0 - 100*grad_Jm

grad_Jm= grad_J(m0)
#J_m= J(m0)
m0 = m0 - 20*grad_Jm

grad_Jm= grad_J(m0)
#m0 = m0 - 10*grad_Jm

m0 = m0 - 40*grad_Jm
grad_Jm= grad_J(m0)

m0 = m0 - 10*grad_Jm
grad_Jm= grad_J(m0)

m0 = m0 - 70*grad_Jm
grad_Jm= grad_J(m0)

m0 = m0 - 10*grad_Jm
grad_Jm= grad_J(m0)


J_list=[]
num=10
max_t = 100
for t in tqdm(np.arange(0,max_t,max_t/num)):
    J_list.append(J(m0-t*grad_Jm))

plt.figure()
plt.plot(J_list)






from scipy.optimize import minimize,basinhopping,dual_annealing

def f(x):
    global m0
    x=x.reshape((2,)+m0.shape)
    return J((x[0]+1j*x[1]).astype("complex64"))

def  Jf(x):
    global m0
    x = x.reshape((2,) + m0.shape)
    grad = grad_J((x[0] + 1j * x[1]).astype("complex64"))
    grad=np.expand_dims(grad.flatten(),axis=0)
    grad = np.concatenate([grad.real, grad.imag], axis=0)
    grad = grad.flatten()
    return grad


x0=np.expand_dims(m0.flatten(),axis=0)
x0 = np.concatenate([x0.real,x0.imag],axis=0)
x0=x0.flatten()
#
#
x_opt=minimize(f,x0,method='CG',jac=Jf)
np.save("x_opt_CG_32.npy",x_opt)





eps=0.001
ind=(0,int(nb_slices/2),int(image_size[1]/2),int(image_size[1]/2))
h = np.zeros(m0.shape,dtype=m0.dtype)
h[ind[0],ind[1],ind[2],ind[3]]=eps

diff_Jm = J(m0+h)-J_m
diff_Jm_approx = grad_Jm[ind[0],ind[1],ind[2],ind[3]]*eps




m_opt=graddesc(J,grad_J,m0,alpha=0.1,log=True,tolgrad=1e-10)


m_opt=graddesc_linsearch(J,grad_J,m0,alpha=0.1,beta=0.6,log=True,tolgrad=1e-10,t0=300)


import time
start_time = time.time()
m_opt=conjgrad(J,grad_J,m0,alpha=0.1,beta=0.3,log=True,tolgrad=1e-10,t0=100)
filename_m_opt=str.split(filename,".dat") [0]+"_m_opt_L0{}.npy".format(L0)
np.save(filename_m_opt,m_opt)
filename_phi=str.split(filename,".dat") [0]+"_phi_L0{}.npy".format(L0)
np.save(filename_phi,phi)
print("--- %s seconds ---" % (time.time() - start_time))



sl=int(nb_slices/2)
l=np.random.choice(L0)
plt.figure()
plt.imshow(np.abs(m_opt[l,sl,:,:]))
plt.title("basis image for l={}".format(l))

gr=0
phi_gr=phi[:,gr,:]
sl=int(nb_slices/2)
volumes_rebuilt_gr=(m_opt[:,sl,:,:].reshape((L0,-1)).T@phi_gr).reshape(image_size[1],image_size[2],ntimesteps)
volumes_rebuilt_gr=np.moveaxis(volumes_rebuilt_gr,-1,0)
animate_images(volumes_rebuilt_gr)


volumes_all_rebuilt = (m_opt.reshape((L0,-1)).T@phi_gr).reshape(image_size[0],image_size[1],image_size[2],ntimesteps)
volumes_all_rebuilt=np.moveaxis(volumes_all_rebuilt,-1,0)

filename_volume_rebuilt_multitasking=str.split(filename,".dat") [0]+"_volumes_mt_L0{}_gr{}.npy".format(L0,gr)
np.save(filename_volume_rebuilt_multitasking,volumes_all_rebuilt)


v_error_final= grad_J(m0)



sl=int(nb_slices/2)
image_list = list(np.abs(dm[:,sl,:,:]))
plot_image_grid(image_list,(3,3))

v_error_final=np.moveaxis(v_error_final,0,1)

dm = v_error_final@(phi.reshape(L0,-1).T.conj())

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

mask = m.mask
#volumes_all = np.load(filename_volume)
#volumes_corrected_final=np.load(filename_volume_corrected_final)

gr=0
L0=8
filename_volume_rebuilt_multitasking=str.split(filename,".dat") [0]+"_volumes_mt_L0{}_gr{}.npy".format(L0,gr)
volumes_corrected_final=np.load(filename_volume_rebuilt_multitasking)

#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])

suffix="Multitasking_L0{}_gr{}".format(L0,gr)
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

nb_filled_slices = 4
nb_empty_slices=0
repeat_slice=4
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

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
filename_groundtruth = filename+"_groundtruth_volumes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")

filename_kdata = filename+"_kdata_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask= filename+"_mask_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
file_map = filename + "_mvt_sl{}_rp{}_us{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,suffix)



#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 512



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

nb_channels=1


direction=np.array([0.0,8.0,0.0])
move = TranslationBreathing(direction,T=4000,frac_exp=0.7)

m.add_movements([move])


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):

    #images = copy(m.images_series)
    data=m.generate_kdata(radial_traj)

    data=np.array(data)
    np.save(filename_kdata, data)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    data = np.load(filename_kdata)


timesteps = list(np.arange(1400)[::28])

nav_z=Navigator3D(direction=[1.0,0.0,0.0],applied_timesteps=timesteps)
kdata_nav = m.generate_kdata(nav_z)

dictfile = "./mrf175_SimReco2.dict"
ind_dico = 8

filename_dico_comp = str.split(dictfile,".dict") [0]+"_phi_dico_{}comp.npy".format(ind_dico)

if str.split(filename_dico_comp,"/")[-1]  not in os.listdir():

    FF_list = list(np.arange(0., 1.05, 0.05))
    keys, signal = read_mrf_dict(dictfile, FF_list)

    import dask.array as da

    A_r=signal.real
    A_i=signal.imag

    X_1 = np.concatenate([A_r,-A_i],axis=-1)
    X_2 = np.concatenate([A_i,A_r],axis=-1)
    X=np.concatenate([X_1,X_2],axis=0)

    u_dico, s_dico, vh_dico = da.linalg.svd(da.from_array(X))

    vh_dico=np.array(vh_dico[::2,:])
    s_dico=np.array(s_dico[::2])

    # plt.figure()
    # plt.plot(np.cumsum(s_dico)/np.sum(s_dico))

    #ind_dico = ((np.cumsum(s_dico)/np.sum(s_dico))<0.99).sum()
    #ind_dico=20

    vh_dico_retained = vh_dico[:ind_dico,:]
    phi_dico = vh_dico_retained[:,:ntimesteps] - 1j * vh_dico_retained[:,ntimesteps:]

    del u_dico
    del s_dico
    del vh_dico

    del vh_dico_retained
    del X_1
    del X_2
    del X
    #del signal


    np.save(filename_dico_comp,phi_dico)
else:
    filename_dico_comp = str.split(dictfile,".dict") [0]+"_phi_dico_{}comp.npy".format(ind_dico)
    phi_dico=np.load(filename_dico_comp)


ngroups=1
#
# data_for_svd = data.reshape(ntimesteps,-1)
# data_for_svd = data_for_svd.T
#
# u, s, vh = np.linalg.svd(data_for_svd, full_matrices=False)

phi=phi_dico
L0 = phi.shape[0]

#data = np.load(filename_save)
m_shape = (L0,)+image_size
m0=np.zeros(m_shape,dtype=data.dtype)
traj=radial_traj.get_traj_for_reconstruction()

if m0.dtype == "complex64":
    try:
        traj = traj.astype("float32")
    except:
        pass

traj=traj.reshape(-1,3)

data_mask = np.ones((nb_channels, 8, nb_slices, ngroups, ntimesteps))



def J(m):
    global L0
    global phi
    global traj
    global ntimesteps
    global data
    global nb_slices
    global nb_channels
    global npoint
    print(m.dtype)
    FU = finufft.nufft3d2(traj[:, 2],traj[:, 0], traj[:, 1], m)

    FU=FU.reshape(L0,ntimesteps,-1)
    FU=np.moveaxis(FU,0,-1)
    phi = phi.reshape(L0,-1,ntimesteps)
    ngroups=phi.shape[1]
    kdata_model=[]
    for ts in tqdm(range(ntimesteps)):
        kdata_model.append(FU[ts]@phi[:,:,ts])
    kdata_model=np.array(kdata_model)

    kdata_model=kdata_model.reshape(ntimesteps,8,nb_slices,npoint,ngroups)
    kdata_model=np.expand_dims(kdata_model,axis=0)
    kdata_model_retained = np.zeros(kdata_model.shape[:-1],dtype=data.dtype)

    for ts in tqdm(range(ntimesteps)):
        for sl in range(nb_slices):
            for sp in range(8):
                for g in range(ngroups):
                    if data_mask[0,sp,sl,g,ts]:
                        kdata_model_retained[:,ts,sp,sl,:]=kdata_model[:,ts,sp,sl,:,g]

    kdata_error = kdata_model_retained-data.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
    return np.linalg.norm(kdata_error)**2

def grad_J(m):
    global L0
    global phi
    global traj
    global ntimesteps
    global data
    global nb_slices
    global nb_channels
    global npoint
    global image_size



    FU = finufft.nufft3d2(traj[:, 2], traj[:, 0], traj[:, 1], m)

    FU = FU.reshape(L0, ntimesteps, -1)
    FU = np.moveaxis(FU, 0, -1)
    phi = phi.reshape(L0, -1, ntimesteps)
    ngroups = phi.shape[1]
    kdata_model = []
    for ts in tqdm(range(ntimesteps)):
        kdata_model.append(FU[ts] @ phi[:, :, ts])
    kdata_model = np.array(kdata_model)

    kdata_model = kdata_model.reshape(ntimesteps, 8, nb_slices, npoint, ngroups)
    kdata_model = np.expand_dims(kdata_model, axis=0)
    kdata_model_retained = np.zeros(kdata_model.shape[:-1], dtype=data.dtype)

    for ts in tqdm(range(ntimesteps)):
        for sl in range(nb_slices):
            for sp in range(8):
                for g in range(ngroups):
                    if data_mask[0, sp, sl, g, ts]:
                        kdata_model_retained[:, ts, sp, sl, :] = kdata_model[:, ts, sp, sl, :, g]

    kdata_error = kdata_model_retained - data.reshape(nb_channels, ntimesteps, -1, nb_slices, npoint)

    kdata_error_phiH = np.zeros(kdata_model.shape[:-1] + (L0,), dtype=kdata_error.dtype)
    # kdata_error_reshaped=np.zeros(kdata_model.shape+(ntimesteps,),dtype=kdata_model.dtype)

    # phi_H = phi.conj().reshape(L0,-1).T

    for ts in tqdm(range(ntimesteps)):
        for sl in range(nb_slices):
            for sp in range(8):
                for g in range(ngroups):
                    if data_mask[0, sp, sl, g, ts]:
                        # kdata_error_reshaped[:, ts, sp, sl, :,g,ts] = kdata_error[:, ts, sp, sl, :]
                        for l in range(L0):
                            kdata_error_phiH[:, ts, sp, sl, :, l] = kdata_error[:, ts, sp, sl, :] * phi.conj()[l, g, ts]

    # phi_H = phi.conj().reshape(L0,-1).T
    # kdata_error_reshaped=kdata_error_reshaped.reshape(-1,ngroups*ntimesteps)

    kdata_error_phiH = np.moveaxis(kdata_error_phiH, -1, 0)
    density = np.abs(np.linspace(-1, 1, npoint))
    density = np.expand_dims(density, tuple(range(kdata_error_phiH.ndim - 1)))
    kdata_error_phiH *= density

    dtheta = np.pi / (8*ntimesteps)
    dz = 1 / nb_slices

    kdata_error_phiH *= 1 / (2 * npoint) * dz * dtheta

    kdata_error_phiH = kdata_error_phiH.reshape(L0 * nb_channels, -1)

    dm = finufft.nufft3d1(traj[:, 2], traj[:, 0], traj[:, 1], kdata_error_phiH, image_size)
    #dm = dm/np.linalg.norm(dm)

    return 2*dm


m0 = np.zeros(m_shape,dtype=data.dtype)
grad_Jm= grad_J(m0)
#J_m= J(m0)
m0 = m0 - 100*grad_Jm

grad_Jm= grad_J(m0)
#J_m= J(m0)
m0 = m0 - 20*grad_Jm

grad_Jm= grad_J(m0)
#m0 = m0 - 10*grad_Jm

m0 = m0 - 40*grad_Jm
grad_Jm= grad_J(m0)

m0 = m0 - 10*grad_Jm
grad_Jm= grad_J(m0)

m0 = m0 - 70*grad_Jm
grad_Jm= grad_J(m0)

m0 = m0 - 10*grad_Jm
grad_Jm= grad_J(m0)


J_list=[]
num=10
max_t = 100
for t in tqdm(np.arange(0,max_t,max_t/num)):
    J_list.append(J(m0-t*grad_Jm))

plt.figure()
plt.plot(J_list)






from scipy.optimize import minimize,basinhopping,dual_annealing

def f(x):
    global m0
    x=x.reshape((2,)+m0.shape)
    return J((x[0]+1j*x[1]).astype("complex64"))

def  Jf(x):
    global m0
    x = x.reshape((2,) + m0.shape)
    grad = grad_J((x[0] + 1j * x[1]).astype("complex64"))
    grad=np.expand_dims(grad.flatten(),axis=0)
    grad = np.concatenate([grad.real, grad.imag], axis=0)
    grad = grad.flatten()
    return grad


x0=np.expand_dims(m0.flatten(),axis=0)
x0 = np.concatenate([x0.real,x0.imag],axis=0)
x0=x0.flatten()
#
#
x_opt=minimize(f,x0,method='CG',jac=Jf)
np.save("x_opt_CG_32.npy",x_opt)





eps=0.001
ind=(0,int(nb_slices/2),int(image_size[1]/2),int(image_size[1]/2))
h = np.zeros(m0.shape,dtype=m0.dtype)
h[ind[0],ind[1],ind[2],ind[3]]=eps

diff_Jm = J(m0+h)-J_m
diff_Jm_approx = grad_Jm[ind[0],ind[1],ind[2],ind[3]]*eps




m_opt=graddesc(J,grad_J,m0,alpha=0.1,log=True,tolgrad=1e-10)


m_opt=graddesc_linsearch(J,grad_J,m0,alpha=0.1,beta=0.6,log=True,tolgrad=1e-10,t0=300)


import time
start_time = time.time()
m_opt=conjgrad(J,grad_J,m0,alpha=0.1,beta=0.3,log=True,tolgrad=1e-10,t0=100)
filename_m_opt=str.split(filename,".dat") [0]+"_m_opt_L0{}.npy".format(L0)
np.save(filename_m_opt,m_opt)
filename_phi=str.split(filename,".dat") [0]+"_phi_L0{}.npy".format(L0)
np.save(filename_phi,phi)
print("--- %s seconds ---" % (time.time() - start_time))



sl=int(nb_slices/2)
l=np.random.choice(L0)
plt.figure()
plt.imshow(np.abs(m_opt[l,sl,:,:]))
plt.title("basis image for l={}".format(l))

gr=0
phi_gr=phi[:,gr,:]
sl=int(nb_slices/2)
volumes_rebuilt_gr=(m_opt[:,sl,:,:].reshape((L0,-1)).T@phi_gr).reshape(image_size[1],image_size[2],ntimesteps)
volumes_rebuilt_gr=np.moveaxis(volumes_rebuilt_gr,-1,0)
animate_images(volumes_rebuilt_gr)


volumes_all_rebuilt = (m_opt.reshape((L0,-1)).T@phi_gr).reshape(image_size[0],image_size[1],image_size[2],ntimesteps)
volumes_all_rebuilt=np.moveaxis(volumes_all_rebuilt,-1,0)

filename_volume_rebuilt_multitasking=str.split(filename,".dat") [0]+"_mvt_volumes_mt_L0{}_gr{}.npy".format(L0,gr)
np.save(filename_volume_rebuilt_multitasking,volumes_all_rebuilt)


v_error_final= grad_J(m0)



sl=int(nb_slices/2)
image_list = list(np.abs(dm[:,sl,:,:]))
plot_image_grid(image_list,(3,3))

v_error_final=np.moveaxis(v_error_final,0,1)

dm = v_error_final@(phi.reshape(L0,-1).T.conj())

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

mask = m.mask
#volumes_all = np.load(filename_volume)
#volumes_corrected_final=np.load(filename_volume_corrected_final)

gr=0
L0=8
filename_volume_rebuilt_multitasking=str.split(filename,".dat") [0]+"_volumes_mt_L0{}_gr{}.npy".format(L0,gr)
volumes_corrected_final=np.load(filename_volume_rebuilt_multitasking)

#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])

suffix="Multitasking_L0{}_gr{}".format(L0,gr)
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
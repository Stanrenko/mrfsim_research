

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
filename_volume = filename+"_volumes_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
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


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)

nb_channels=1


direction=np.array([0.0,8.0,0.0])
move = TranslationBreathing(direction,T=4000,frac_exp=0.7)

m_.add_movements([move])


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):

    #images = copy(m.images_series)
    data=m_.generate_kdata(radial_traj)

    data=np.array(data)
    np.save(filename_kdata, data)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    data = np.load(filename_kdata)


##volumes for slice taking into account coil sensi
print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    volumes_all=simulate_radial_undersampled_images(data,radial_traj,image_size,density_adj=True,useGPU=False)
    np.save(filename_volume,volumes_all)
    # sl=20
    # ani = animate_images(volumes_all[:,sl,:,:])
    del volumes_all

volumes_all=np.load(filename_volume)



nb_gating_spokes=50
ts_between_spokes=int(nb_allspokes/50)
timesteps = list(np.arange(1400)[::ts_between_spokes])

nav_z=Navigator3D(direction=[1,0,0.0],applied_timesteps=timesteps,npoint=512)
kdata_nav = m_.generate_kdata(nav_z)


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

plt.imshow(np.abs(images_nav_mean.reshape(-1,int(npoint/2))).T)



print("Estimating Movement...")
shifts = list(range(-20, 20))
bottom = 50
top = 150
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


group_1=(categories==1)
group_2=(categories==2)
group_3=(categories==5)|(categories==8)|(categories==9)

groups=[group_1,group_2,group_3]


nav_spoke_groups=np.argmin(np.abs(np.arange(0, ntimesteps, 1).reshape(-1, 1) - np.arange(0, ntimesteps,ntimesteps / nb_gating_spokes).reshape(1,-1)),axis=-1)
data_mt_training=copy(data_for_nav)
data_mt_training=np.squeeze(data_mt_training)
Sk=np.zeros((npoint,len(groups),ntimesteps),dtype=data_for_nav.dtype)
Sk_mask=np.ones((npoint,len(groups),ntimesteps),dtype=int)

data_mt_training_on_timesteps = np.zeros((nb_slices,ntimesteps,npoint),dtype=data_for_nav.dtype)

nb_part=nb_slices

for i in tqdm(range(len(groups))):
    for ts in range(ntimesteps):
        g=groups[i]
        gating_spoke_of_ts=nav_spoke_groups[ts]
        g_reshaped=copy(g).reshape(int(nb_part),int(nb_gating_spokes))
        g_reshaped[:,list(set(range(nb_gating_spokes))-set([gating_spoke_of_ts]))]=False
        retained_spokes = np.argwhere(g_reshaped)
        if len(retained_spokes)==0:
            Sk_mask[:,i,ts]=0
        else:
            Sk[:,i,ts]=data_mt_training[retained_spokes[:,0],retained_spokes[:,1],:].mean(axis=0)

        data_mt_training_on_timesteps[:,ts,:]=data_mt_training[:,gating_spoke_of_ts,:]


Sk_cur = copy(Sk)
niter=100
diffs = []
tol_diff = 1e-2
variance_explained=0.99
proj_on_fingerprints=False

for i in tqdm(range(niter)):
    Sk_1 = Sk_cur.reshape(Sk_cur.shape[0],-1)
    u_1, s_1, vh_1 = np.linalg.svd(Sk_1, full_matrices=False)

    Sk_2 = np.moveaxis(Sk_cur,1,0).reshape(Sk_cur.shape[1],-1)
    u_2, s_2, vh_2 = np.linalg.svd(Sk_2, full_matrices=False)

    Sk_3 = np.moveaxis(Sk_cur, 2, 0).reshape(Sk_cur.shape[2], -1)
    if proj_on_fingerprints:
        Sk_3 = phi_dico.T @ phi_dico.conj() @ Sk_final_3
    else:
        u_3, s_3, vh_3 = np.linalg.svd(Sk_3, full_matrices=False)
        cum_3 = np.cumsum(s_3) / np.sum(s_3)
        ind_3 = (cum_3 < variance_explained).sum()
        Sk_3 = u_3[:, :ind_3] @ (np.diag(s_3[:ind_3])) @ vh_3[:ind_3, :]


    cum_1=np.cumsum(s_1)/np.sum(s_1)
    cum_2=np.cumsum(s_2)/np.sum(s_2)


    ind_1 = (cum_1<variance_explained).sum()
    ind_2 = (cum_2<variance_explained).sum()

    Sk_1 = u_1[:,:ind_1]@(np.diag(s_1[:ind_1]))@vh_1[:ind_1,:]
    Sk_2 = u_2[:, :ind_2] @ (np.diag(s_2[:ind_2])) @ vh_2[:ind_2, :]


    Sk_1 = Sk_1.reshape(Sk_cur.shape[0],Sk_cur.shape[1],Sk_cur.shape[2])
    Sk_2 = Sk_2.reshape(Sk_cur.shape[1], Sk_cur.shape[0], Sk_cur.shape[2])
    Sk_3 = Sk_3.reshape(Sk_cur.shape[2], Sk_cur.shape[0], Sk_cur.shape[1])

    Sk_2=np.moveaxis(Sk_2,0,1)
    Sk_3 = np.moveaxis(Sk_3, 0, 2)

    Sk_cur_prev = copy(Sk_cur)
    Sk_cur=Sk*Sk_mask + np.mean(np.stack([Sk_1,Sk_2,Sk_3],axis=-1),axis=-1)*(1-Sk_mask)
    diff = np.linalg.norm((Sk_cur-Sk_cur_prev )/Sk_cur_prev/np.sqrt(np.sum(Sk_mask)))
    diffs.append(diff)

    if diff<tol_diff:
        break

# plt.figure()
# plt.plot(diffs)

Sk_mask.sum()/np.prod(Sk_mask.shape)

Sk_final = copy(Sk_cur)
del Sk_cur


D_non_proj=Sk_final.reshape(npoint,-1)
u_non_proj, s_non_proj, vh_non_proj = np.linalg.svd(D_non_proj, full_matrices=False)
L0 = 32
#phi_non_proj = (vh_non_proj)[:L0,:]
phi_non_proj=vh_non_proj[:L0,:]
phi=phi_non_proj

D_proj_on_phi = D_non_proj@phi_non_proj.T.conj()@phi_non_proj

k_num=256
k_num=np.random.choice(npoint)
metric=np.abs
plt.figure()
plt.plot(metric(D_non_proj[k_num,:]),label="Original k {}".format(k_num))
plt.plot(metric(D_proj_on_phi[k_num,:]),label="Projected")
plt.legend()


m0=np.zeros((L0,)+image_size,dtype=data.dtype)
traj=radial_traj.get_traj_for_reconstruction()

if m0.dtype == "complex64":
    try:
        traj = traj.astype("float32")
    except:
        pass

traj=traj.reshape(-1,3)

data_mask = np.zeros((nb_channels, 8, nb_slices, len(groups), ntimesteps))


for j, g in tqdm(enumerate(groups)):
    retained_nav_spokes_index = np.argwhere(g).flatten()
    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
                             axis=-1)
    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
    included_spokes[::int(nb_segments / nb_gating_spokes)] = False
    included_spokes_for_mask = included_spokes.astype(int).reshape(nb_slices, ntimesteps, 8)
    included_spokes_for_mask = np.moveaxis(included_spokes_for_mask, -1, 0)
    for i in range(nb_channels):
        data_mask[i, :, :, j, :] = included_spokes_for_mask


eps=1e-6
useGPU = False
b1=1


def J(m):
    global L0
    global phi
    global traj
    global ntimesteps
    global data
    global nb_slices
    global nb_channels
    global useGPU
    global eps
    print(m.dtype)

    if not(useGPU):
        FU = finufft.nufft3d2(traj[:, 2],traj[:, 0], traj[:, 1], m)
    else:
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64
        N1, N2, N3 = m.shape[1], m.shape[2], m.shape[3]
        M = traj.shape[0]
        c_gpu = GPUArray((M), dtype=complex_dtype)
        kdata = []
        for i in list(range(m.shape[0])):
            fk = m[i, :, :,:]
            kx = traj[:, 0]
            ky = traj[:, 1]
            kz = traj[:, 2]

            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            fk = fk.astype(complex_dtype)

            plan = cufinufft(2, (N1, N2, N3), 1, eps=eps, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))
            plan.execute(c_gpu, to_gpu(fk))
            c = np.squeeze(c_gpu.get())
            kdata.append(c)
            plan.__del__()
        FU = np.array(kdata)

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
    global groups
    global nb_part
    global nb_segments
    global nb_gating_spokes
    global nb_allspokes
    global undersampling_factor
    global mode
    global incoherent
    global image_size
    global useGPU
    global eps
    global b1


    if not(useGPU):
        FU = finufft.nufft3d2(traj[:, 2], traj[:, 0], traj[:, 1], m)
    else:
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64
        N1, N2, N3 = m.shape[1], m.shape[2], m.shape[3]
        M = traj.shape[0]
        c_gpu = GPUArray((M), dtype=complex_dtype)
        kdata = []
        for i in list(range(m.shape[0])):
            fk = m[i, :, :, :]
            kx = traj[:, 0]
            ky = traj[:, 1]
            kz = traj[:, 2]

            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            fk = fk.astype(complex_dtype)

            plan = cufinufft(2, (N1, N2, N3), 1, eps=eps, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))
            plan.execute(c_gpu, to_gpu(fk))
            c = np.squeeze(c_gpu.get())
            kdata.append(c)
            plan.__del__()
        FU = np.array(kdata)



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
    #density = np.abs(np.linspace(-1, 1, npoint))
    #density = np.expand_dims(density, tuple(range(kdata_error_phiH.ndim - 1)))
    #kdata_error_phiH *= density

    #dtheta = np.pi / (8*ntimesteps)
    #dz = 1 / nb_slices

    #kdata_error_phiH *= 1 / (2 * npoint) * dz * dtheta



    if not(useGPU):
        kdata_error_phiH = kdata_error_phiH.reshape(L0 * nb_channels, -1)
        dm = finufft.nufft3d1(traj[:, 2], traj[:, 0], traj[:, 1], kdata_error_phiH, image_size)
    else:
        dm = np.zeros(m.shape,dtype=m.dtype)
        kdata_error_phiH = kdata_error_phiH.reshape(L0,nb_channels, -1)
        N1, N2, N3 = image_size[0], image_size[1], image_size[2]
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64

        for i in tqdm(list(range(L0))):
            fk_gpu = GPUArray((nb_channels, N1, N2, N3), dtype=complex_dtype)
            c_retrieved = kdata_error_phiH[i, :,:]
            kx = traj[:, 0]
            ky = traj[:, 1]
            kz = traj[:, 2]

            # Cast to desired datatype.
            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            c_retrieved = c_retrieved.astype(complex_dtype)

            # Allocate memory for the uniform grid on the GPU.
            c_retrieved_gpu = to_gpu(c_retrieved)

            # Initialize the plan and set the points.
            plan = cufinufft(1, (N1, N2, N3), nb_channels, eps=eps, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

            # Execute the plan, reading from the strengths array c and storing the
            # result in fk_gpu.
            plan.execute(c_retrieved_gpu, fk_gpu)

            fk = np.squeeze(fk_gpu.get())

            fk_gpu.gpudata.free()
            c_retrieved_gpu.gpudata.free()

            if b1 is None:
                dm[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
            elif b1==1:
                dm[i] = fk
            else:
                dm[i] = np.sum(b1.conj() * fk, axis=0)

            plan.__del__()

        if (b1 is not None)and(not(b1==1)):
            dm /= np.expand_dims(np.sum(np.abs(b1) ** 2, axis=0), axis=0)

    #dm = dm/np.linalg.norm(dm)

    return 2*dm



J_list=[]
num=20
max_t = 0.0000005
m=m0
J_m=J(m)
g=grad_J(m)
d_m=-g
slope = np.real(np.dot(g.flatten(),d_m.conj().flatten()))
t_array=np.arange(0,max_t,max_t/num)

for t in tqdm(t_array):
     J_list.append(J(m0+t*d_m))
plt.figure()
plt.plot(J_list)
plt.plot(list(range(num)),J_m+slope*t_array)



import time
start_time = time.time()
filename_m_opt=str.split(filename,".dat") [0]+"_m_opt_L0{}.npy".format(L0)
filename_m_opt_figure=str.split(filename,".dat") [0]+"_m_opt_L0{}.jpg".format(L0)
#m_opt=conjgrad(J,grad_J,m0,alpha=0.1,beta=0.3,log=True,tolgrad=1e-10,t0=100,maxiter=1000,plot=True,filename_save=filename_m_opt)


log=True
plot=True
filename_save = filename_m_opt
t0=0.0000001
beta=0.6
alpha=0.05
tolgrad=1e-4
maxiter=30

k=0
m=m0
if log:
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    norm_g_list=[]

g=grad_J(m)
d_m=-g
#store = [m]

if plot:
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(30, 10))
    axs[0].set_title("Evolution of cost function")
while (np.linalg.norm(g)>tolgrad)and(k<maxiter):
    norm_g = np.linalg.norm(g)
    if log:
        print("################ Iter {} ##################".format(k))
        norm_g_list.append(norm_g)
    print("Grad norm for iter {}: {}".format(k,norm_g))
    if k%10==0:
        print(k)
        if filename_save is not None:
            np.save(filename_save,m)
    t = t0
    J_m = J(m)
    print("J for iter {}: {}".format(k,J_m))
    J_m_next = J(m+t*d_m)
    slope = np.real(np.dot(g.flatten(),d_m.conj().flatten()))
    if plot:
        axs[0].scatter(k,J_m,c="r",marker="+")
        axs[1].cla()
        axs[1].set_title("Line search for iteration {}".format(k))
        t_array = np.arange(0.,t0,t0/100)
        axs[1].plot(t_array,J_m+t_array*slope)
        axs[1].scatter(0,J_m,c="b",marker="x")
        plt.draw()

    while(J_m_next>J_m+alpha*t*slope):
        print(t)
        t = beta*t
        if plot:
            axs[1].scatter(t,J_m_next,c="b",marker="x")
        J_m_next=J(m+t*d_m)


    if plot:
        plt.savefig(filename_m_opt_figure)


    m = m + t*d_m
    g_prev = g
    g = grad_J(m)
    gamma = np.linalg.norm(g)**2/np.linalg.norm(g_prev)**2
    d_m = -g + gamma*d_m
    k=k+1
    #store.append(m)

if log:
    norm_g_list=np.array(norm_g_list)
    np.save('./logs/conjgrad_{}.npy'.format(date_time),norm_g_list)


m_opt=m
#
#
# #filename_m_opt=str.split(filename,".dat") [0]+"_m_opt_L0{}.npy".format(L0)
np.save(filename_m_opt,m_opt)

filename_phi=str.split(filename,".dat") [0]+"_phi_L0{}.npy".format(L0)
np.save(filename_phi,phi)


sl=int(nb_slices/2)
l=np.random.choice(L0)
plt.figure()
plt.imshow(np.abs(m_opt[l,sl,:,:]))
plt.title("basis image for l={}".format(l))

gr=2
phi_gr=phi[:,gr,:]
sl=int(nb_slices/2)
volumes_rebuilt_gr=(m_opt[:,sl,:,:].reshape((L0,-1)).T@phi_gr).reshape(image_size[1],image_size[2],ntimesteps)
volumes_rebuilt_gr=np.moveaxis(volumes_rebuilt_gr,-1,0)
animate_images(volumes_rebuilt_gr)


animate_multiple_images(volumes_rebuilt_gr,m_.images_series[::nspoke,sl,:,:])



gr=2
phi_gr=phi[:,gr,:]
volumes_all_rebuilt = (m_opt.reshape((L0,-1)).T@phi_gr).reshape(image_size[0],image_size[1],image_size[2],ntimesteps)
volumes_all_rebuilt=np.moveaxis(volumes_all_rebuilt,-1,0)
#
filename_volume_rebuilt_multitasking=str.split(filename,".dat") [0]+"_volumes_mt_L0{}_gr{}.npy".format(L0,gr)
np.save(filename_volume_rebuilt_multitasking,volumes_all_rebuilt)


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

mask = m_.mask
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
suffix=""
ntimesteps=175
if not(load_map):
    niter = 0
    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=None,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=False,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps)
    all_maps=optimizer.search_patterns_test(dictfile,volumes_all,retained_timesteps=None)

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
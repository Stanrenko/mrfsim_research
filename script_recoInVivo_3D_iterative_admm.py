######SIMU########
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
from utils_simu import *

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"

#dictfile = "mrf175_SimReco2_light.dict"
dictjson="mrf_dictconf_SimReco2_light.json"
dictfile="mrf_dictconf_SimReco2_adjusted_1_87_reco4_w8_simmean.dict"
dictfile_light='./mrf_dictconf_SimReco2_light_matching_adjusted_1_87_reco4_w8_simmean.dict'
suffix="_fullReco"
with open("./mrf_sequence_adjusted_1_87.json") as f:
   sequence_config = json.load(f)
Treco=4000

nrep=2
rep=nrep-1
TR_total = np.sum(sequence_config["TR"])

#Treco = TR_total-np.sum(sequence_config["TR"])

##other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=nrep
sequence_config["rep"]=rep

seq=T1MRFSS(**sequence_config)

nb_filled_slices = 16
nb_empty_slices=2
repeat_slice=16
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

name = "SquareSimu3DMT"


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
filename_kdata_gt = filename+"_kdata_mvt_gt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_nav = filename+"_nav_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_volume = filename+"_volumes_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_volume_corrected = filename+"_volumes_corrected_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_mask= filename+"_mask_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask_corrected= filename+"_mask_corrected_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

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

if "Square" in name:
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

#if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
#    np.save(filename_groundtruth,m_.images_series[::nspoke])

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
    data=np.expand_dims(data,axis=0)
    np.save(filename_kdata, data)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    data = np.load(filename_kdata)

nb_segments=radial_traj.get_traj().shape[0]

b1_full = np.ones(image_size)
b1_all_slices=np.expand_dims(b1_full,axis=0)


print("Rebuilding Images With Corrected volumes...")

if str.split(filename_volume_corrected,"/")[-1] not in os.listdir(folder):
    kdata_all_channels_all_slices=np.load(filename_kdata)
    volumes_corrected = simulate_radial_undersampled_images_multi_new(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,ntimesteps=ntimesteps,density_adj=True,useGPU=False,light_memory_usage=True,retained_timesteps=retained_timesteps,weights=weights)
    animate_images(volumes_corrected[:,int(nb_slices/2),:,:])
    np.save(filename_volume_corrected,volumes_corrected)
else:
    volumes_corrected=np.load(filename_volume_corrected)



if str.split(filename_mask_corrected,"/")[-1] not in os.listdir(folder):
    kdata_all_channels_all_slices = np.load(filename_kdata)
    volumes_full_corrected = simulate_radial_undersampled_images_multi_new(kdata_all_channels_all_slices, radial_traj,
                                                                           image_size, b1=b1_full,
                                                                           ntimesteps=ntimesteps,
                                                                           density_adj=True, useGPU=False,
                                                                           light_memory_usage=True,
                                                                           retained_timesteps=retained_timesteps,
                                                                           weights=weights, ntimesteps_final=1)[0]
    mask = False
    unique = np.histogram(np.abs(volumes_full_corrected), 100)[1]
    mask = mask | (np.abs(volumes_full_corrected) > unique[int(len(unique) * 0.07)])
    # mask = ndimage.binary_closing(mask, iterations=3)
    animate_images(mask)
    np.save(filename_mask_corrected,mask)
else:
    mask=np.load(filename_mask_corrected)

print("Processing Nav Data...")

nb_gating_spokes=50
ts_between_spokes=int(nb_allspokes/50)
timesteps = list(np.arange(1400)[::ts_between_spokes])

if str.split(filename_nav,"/")[-1] not in os.listdir(folder):

    nav_z=Navigator3D(direction=[1,0,0.0],applied_timesteps=timesteps,npoint=npoint)
    kdata_nav = m_.generate_kdata(nav_z,useGPU=use_GPU)

    kdata_nav=np.array(kdata_nav)

    data_for_nav = np.expand_dims(kdata_nav,axis=0)
    data_for_nav =  np.moveaxis(data_for_nav,1,2)
    data_for_nav = data_for_nav.astype("complex64")
    np.save(filename_nav,data_for_nav)

else:
    data_for_nav=np.load(filename_nav)

npoint_nav=data_for_nav.shape[-1]
nb_slices_nav=data_for_nav.shape[1]
nb_gating_spokes=data_for_nav.shape[-2]

all_timesteps = np.arange(nb_allspokes)
nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices_nav,
                       applied_timesteps=list(nav_timesteps))

nav_image_size = (int(npoint_nav / 2),)

print("Calculating Sensitivity Maps for Nav Images...")
b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
b1_nav_mean = np.mean(b1_nav, axis=(1, 2))

print("Rebuilding Nav Images...")
images_nav_mean = np.abs(simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, b1_nav_mean))


print("Estimating Movement...")
shifts = list(range(-10, 10))
bottom = 10
top = 54
displacements = calculate_displacement(images_nav_mean, bottom, top, shifts)

displacement_for_binning = displacements
bin_width = 2
max_bin = np.max(displacement_for_binning)
min_bin = np.min(displacement_for_binning)
plt.figure();plt.plot(displacements)

maxi = 0
for j in range(bin_width):
    min_bin = np.min(displacement_for_binning) + j
    bins = np.arange(min_bin, max_bin + bin_width, bin_width)
    #print(bins)
    categories = np.digitize(displacement_for_binning, bins)
    df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
    df_groups = df_cat.groupby("cat").count()
    curr_max = df_groups.displacement.max()
    if curr_max > maxi:
        maxi = curr_max
        df_groups_max=copy(df_groups)
        min_bin_max=min_bin
        max_bin_max=max_bin
        idx_cat = df_groups.displacement.idxmax()
        retained_nav_spokes = (categories == idx_cat)

retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
spoke_groups = np.argmin(np.abs(np.arange(0, nb_segments * nb_slices, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_slices,nb_segments / nb_gating_spokes).reshape(1,-1)),axis=-1)

if not (nb_segments == nb_gating_spokes):
    spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
        nb_segments / nb_gating_spokes / 2) + 1:] - 1
    spoke_groups = spoke_groups.flatten()

included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
displacements_real=m_.list_movements[0].paramDict["transformation"](m_.t.reshape(-1,1))[:,1]
#plt.figure();plt.plot(displacements_real[::(int(nb_allspokes/nb_gating_spokes))]);plt.plot(displacements+5,marker='x')
#plt.figure();plt.plot(displacements_real);plt.plot(included_spokes)
included_spokes=displacements_real>6

included_spokes[::int(nb_segments/nb_gating_spokes)]=False

weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, ntimesteps)





def J_admm(m, kdata_init, dens_adj, trajectory,mu,m_adj,b1_all_slices,weights=None):
    ntimesteps=m.shape[0]
    n_samples=trajectory.get_traj().reshape(ntimesteps,-1,3).shape[1]
    kdata = generate_kdata_multi(m, trajectory, b1_all_slices, ntimesteps=ntimesteps)
    kdata_error = kdata - kdata_init
    if dens_adj:
        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
        kdata_error = kdata_error.reshape(-1, trajectory.paramDict["npoint"])
        density = np.expand_dims(density, axis=0)
        kdata_error *= np.sqrt(density)

    if weights is not None:
        kdata_error = kdata_error.reshape(kdata.shape[0], kdata.shape[1], trajectory.paramDict["nspoke"], -1,
                                          trajectory.paramDict["npoint"])
        kdata_error*=np.sqrt(np.expand_dims(weights,axis=(0,-1)))

    return (np.linalg.norm(kdata_error) ** 2)/n_samples+mu*np.linalg.norm(m-m_adj) ** 2

def grad_J_admm(m,signals_0,dens_adj,trajectory,mu,m_adj,b1_all_slices,mask,weights=None,retained_timesteps=None):
    ntimesteps = m.shape[0]
    volumesi = undersampling_operator_new(m, trajectory, b1_all_slices,density_adj=dens_adj,ntimesteps=ntimesteps,retained_timesteps=retained_timesteps,weights=weights)

    signalsi = volumesi[:, mask > 0]
    signals_adj = m_adj[:, mask > 0]
    signals=m[:,mask>0]
    grad = 2*(signalsi - signals_0 + mu*(signals-signals_adj))

    return grad

def conjgrad(J,grad_J,m0,mask,tolgrad=1e-4,maxiter=100,alpha=0.05,beta=0.6,t0=1,log=False,plot=False,filename_save=None):
    '''
        J : function from W (domain of m) to R
        grad_J : function from W to W - gradient of J
        m0 : initial value of m
        '''
    k = 0
    m = m0
    if log or plot:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        norm_g_list = []

    m_vol = np.array([makevol(im, mask > 0) for im in m])

    g = grad_J(m_vol)
    d_m = -g

    # store = [m]

    if plot:
        plt.ion()
        fig, axs = plt.subplots(1, 2, figsize=(30, 10))
        axs[0].set_title("Evolution of cost function")
    while (np.linalg.norm(g) > tolgrad) and (k < maxiter):
        norm_g = np.linalg.norm(g)
        if log:
            print("################ Iter {} ##################".format(k))
            norm_g_list.append(norm_g)
        print("Grad norm for iter {}: {}".format(k, norm_g))
        if k % 10 == 0:
            print(k)
            if filename_save is not None:
                np.save(filename_save, m)
        t = t0
        J_m = J(m_vol)
        print("J for iter {}: {}".format(k, J_m))
        m_vol_t=np.array([makevol(im, mask > 0) for im in (m+t*d_m)])
        J_m_next = J(m_vol_t)
        slope = np.real(np.dot(g.flatten(), d_m.flatten()))
        if plot:
            axs[0].scatter(k, J_m, c="r", marker="+")
            axs[1].cla()
            axs[1].set_title("Line search for iteration {}".format(k))
            t_array = np.arange(0., t0, t0 / 100)
            axs[1].plot(t_array, J_m + t_array * slope)
            axs[1].scatter(0, J_m, c="b", marker="x")
            plt.savefig('./figures/conjgrad_plot_{}.png'.format(date_time))

        while (J_m_next > J_m + alpha * t * slope):
            print(t)
            t = beta * t
            if plot:
                axs[1].scatter(t, J_m_next, c="b", marker="x")
                plt.savefig('./figures/conjgrad_plot_{}.png'.format(date_time))
            m_vol_t = np.array([makevol(im, mask > 0) for im in (m + t * d_m)])
            J_m_next = J(m_vol_t)

        m = m + t * d_m
        m_vol=np.array([makevol(im, mask > 0) for im in (m)])
        g_prev = g
        g = grad_J(m_vol)
        gamma = np.linalg.norm(g) ** 2 / np.linalg.norm(g_prev) ** 2
        d_m = -g + gamma * d_m
        k = k + 1
        # store.append(m)

    if log:
        norm_g_list = np.array(norm_g_list)
        np.save('./logs/conjgrad_{}.npy'.format(date_time), norm_g_list)

    return m

v0=np.zeros(volumes_corrected.shape,dtype=volumes_corrected.dtype)
m_tilde_0=np.zeros(volumes_corrected.shape,dtype=volumes_corrected.dtype)
m_adj=m_tilde_0-v0

kdata_init=data.reshape(nb_channels,ntimesteps,-1)
dens_adj=True
trajectory=radial_traj
mu=0

#weights_test=np.ones_like(weights)
weights_test=weights
kdata_all_channels_all_slices=np.load(filename_kdata)
#volumes_corrected_test = simulate_radial_undersampled_images_multi_new(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,ntimesteps=ntimesteps,density_adj=dens_adj,useGPU=False,light_memory_usage=True,retained_timesteps=retained_timesteps,weights=weights_test)
signals_0=volumes_corrected[:,mask>0]


J=lambda m:J_admm(m, kdata_init, dens_adj, trajectory,mu,m_adj,b1_all_slices,weights=weights_test)
grad_J=lambda m:grad_J_admm(m,signals_0,dens_adj,trajectory,mu,m_adj,b1_all_slices,mask,weights=weights_test,retained_timesteps=retained_timesteps)



#
# J_list=[]
# num=5
# max_t = 0.1
# m=volumes_corrected
# n_samples=radial_traj.get_traj().reshape(ntimesteps,-1,3).shape[1]
#
# J_m=J(m)
# #n_samples=radial_traj.get_traj().reshape(ntimesteps,-1,3).shape[1]
# g=grad_J(m)
# d_m=-g
# d_m_vol=np.array([makevol(im,mask>0) for im in d_m])
# slope = np.real(np.dot(g.flatten(),d_m.conj().flatten()))
# slope=-np.linalg.norm(d_m.flatten())**2
# t_array=np.arange(0,max_t,max_t/num)
#
# for t in tqdm(t_array):
#      J_list.append(J(m+t*d_m_vol))
#
# plt.close("all")
# plt.figure()
# plt.plot(t_array,J_list)
# plt.plot(t_array,J_m+slope*t_array)
niter_admm=4
signals_0=volumes_corrected[:,mask>0]
niter_cg=4
mu=1


optimizer = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=radial_traj, split=100, pca=True,
                                     threshold_pca=15, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                     b1=b1_all_slices, threshold_ff=0.9, dictfile_light=dictfile_light,
                                     return_matched_signals=True)


signals_init = np.zeros_like(signals_0)
v_curr=np.zeros_like(signals_0)
matched_signals=np.zeros_like(signals_0)

all_maps_admm={}



for iter in range(niter_admm):




    m_adj=matched_signals-v_curr
    m_adj=np.array([makevol(im,mask>0) for im in m_adj])


    J=lambda m:J_admm(m, kdata_init, dens_adj, trajectory,mu,m_adj,b1_all_slices,weights=weights_test)
    grad_J=lambda m:grad_J_admm(m,signals_0,dens_adj,trajectory,mu,m_adj,b1_all_slices,mask,weights=weights_test,retained_timesteps=retained_timesteps)

    signals=conjgrad(J,grad_J,signals_init,mask,maxiter=niter_cg,plot=True)

    all_maps, matched_signals = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, signals+v_curr,
                                                                                       retained_timesteps=None)

    all_maps_admm[iter] = all_maps[0]

    v_curr+=signals-matched_signals
    signals_init=signals




curr_file=file_map
return_cost=True
dx=1
dy=1
dz=5
all_maps=all_maps_admm
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

    if return_cost:
        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "_it{}_{}.mha".format(
            iter, "correlation")
        io.write(file_mha, makevol(all_maps[iter][2],mask>0), tags={"spacing": [dz, dx, dy]})

        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "_it{}_{}.mha".format(
            iter, "phase")
        io.write(file_mha, makevol(all_maps[iter][3],mask>0), tags={"spacing": [dz, dx, dy]})













######SIMU########
### No motion

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
from utils_simu import *

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"

#dictfile = "mrf175_SimReco2_light.dict"
dictjson="mrf_dictconf_SimReco2_light.json"
dictfile="mrf_dictconf_SimReco2_adjusted_1_87_reco4_w8_simmean.dict"
dictfile_light='./mrf_dictconf_SimReco2_light_matching_adjusted_1_87_reco4_w8_simmean.dict'
suffix="_fullReco"
with open("./mrf_sequence_adjusted_1_87.json") as f:
   sequence_config = json.load(f)
Treco=4000

nrep=2
rep=nrep-1
TR_total = np.sum(sequence_config["TR"])

#Treco = TR_total-np.sum(sequence_config["TR"])

##other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=nrep
sequence_config["rep"]=rep

seq=T1MRFSS(**sequence_config)

nb_filled_slices = 16
nb_empty_slices=2
repeat_slice=16
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

name = "SquareSimu3DMT_nomotion"


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
filename_kdata_gt = filename+"_kdata_mvt_gt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_nav = filename+"_nav_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_volume = filename+"_volumes_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_volume_corrected = filename+"_volumes_corrected_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_mask= filename+"_mask_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask_corrected= filename+"_mask_corrected_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

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

if "Square" in name:
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

#if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
#    np.save(filename_groundtruth,m_.images_series[::nspoke])

# i=0
# image=m.images_series[i]


# file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
#                              "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_ideal_ts{}.mha".format(
#                                                                                                                    i)
# io.write(file_mha, np.abs(image), tags={"spacing": [5, 1, 1]})


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,is_random=False)

nb_channels=1


# direction=np.array([0.0,8.0,0.0])
# move = TranslationBreathing(direction,T=4000,frac_exp=0.7)
#
# m_.add_movements([move])


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):

    #images = copy(m.images_series)
    data=m_.generate_kdata(radial_traj,useGPU=use_GPU)
    data=np.array(data)
    data=np.expand_dims(data,axis=0)
    np.save(filename_kdata, data)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    data = np.load(filename_kdata)

nb_segments=radial_traj.get_traj().shape[0]

b1_full = np.ones(image_size)
b1_all_slices=np.expand_dims(b1_full,axis=0)


print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    #del kdata_all_channels_all_slices
    kdata_all_channels_all_slices = np.load(filename_kdata)

    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_iterative=True)
    np.save(filename_volume,volumes_all)
    # sl=10
    # ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
else:
    volumes_all=np.load(filename_volume)



##volumes for slice taking into account coil sensi
print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    mask=m_.mask
    np.save(filename_mask,mask)
else:
    mask=np.load(filename_mask)



def J_admm(m, kdata_init, dens_adj, trajectory,mu,m_adj,b1_all_slices,weights=None):
    ntimesteps=m.shape[0]
    n_samples=trajectory.get_traj().reshape(ntimesteps,-1,3).shape[1]
    kdata = generate_kdata_multi(m, trajectory, b1_all_slices, ntimesteps=ntimesteps)
    kdata_error = kdata - kdata_init
    if dens_adj:
        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
        kdata_error = kdata_error.reshape(-1, trajectory.paramDict["npoint"])
        density = np.expand_dims(density, axis=0)
        kdata_error *= np.sqrt(density)

    if weights is not None:
        kdata_error = kdata_error.reshape(kdata.shape[0], kdata.shape[1], trajectory.paramDict["nspoke"], -1,
                                          trajectory.paramDict["npoint"])
        kdata_error*=np.sqrt(np.expand_dims(weights,axis=(0,-1)))

    return (np.linalg.norm(kdata_error) ** 2)/n_samples+mu*np.linalg.norm(m-m_adj) ** 2

def grad_J_admm(m,signals_0,dens_adj,trajectory,mu,m_adj,b1_all_slices,mask,weights=None,retained_timesteps=None):
    ntimesteps = m.shape[0]
    volumesi = undersampling_operator_new(m, trajectory, b1_all_slices,density_adj=dens_adj,ntimesteps=ntimesteps,retained_timesteps=retained_timesteps,weights=weights)

    signalsi = volumesi[:, mask > 0]
    signals_adj = m_adj[:, mask > 0]
    signals=m[:,mask>0]
    grad = 2*(signalsi - signals_0 + mu*(signals-signals_adj))

    return grad

def conjgrad(J,grad_J,m0,mask,tolgrad=1e-4,maxiter=100,alpha=0.05,beta=0.6,t0=1,log=False,plot=False,filename_save=None):
    '''
        J : function from W (domain of m) to R
        grad_J : function from W to W - gradient of J
        m0 : initial value of m
        '''
    k = 0
    m = m0
    if log or plot:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        norm_g_list = []

    m_vol = np.array([makevol(im, mask > 0) for im in m])

    g = grad_J(m_vol)
    d_m = -g

    # store = [m]

    if plot:
        plt.ion()
        fig, axs = plt.subplots(1, 2, figsize=(30, 10))
        axs[0].set_title("Evolution of cost function")
    while (np.linalg.norm(g) > tolgrad) and (k < maxiter):
        norm_g = np.linalg.norm(g)
        if log:
            print("################ Iter {} ##################".format(k))
            norm_g_list.append(norm_g)
        print("Grad norm for iter {}: {}".format(k, norm_g))
        if k % 10 == 0:
            print(k)
            if filename_save is not None:
                np.save(filename_save, m)
        t = t0
        J_m = J(m_vol)
        print("J for iter {}: {}".format(k, J_m))
        m_vol_t=np.array([makevol(im, mask > 0) for im in (m+t*d_m)])
        J_m_next = J(m_vol_t)
        slope = np.real(np.dot(g.flatten(), d_m.flatten()))
        if plot:
            axs[0].scatter(k, J_m, c="r", marker="+")
            axs[1].cla()
            axs[1].set_title("Line search for iteration {}".format(k))
            t_array = np.arange(0., t0, t0 / 100)
            axs[1].plot(t_array, J_m + t_array * slope)
            axs[1].scatter(0, J_m, c="b", marker="x")
            plt.savefig('./figures/conjgrad_plot_{}.png'.format(date_time))

        while (J_m_next > J_m + alpha * t * slope):
            print(t)
            t = beta * t
            if plot:
                axs[1].scatter(t, J_m_next, c="b", marker="x")
                plt.savefig('./figures/conjgrad_plot_{}.png'.format(date_time))
            m_vol_t = np.array([makevol(im, mask > 0) for im in (m + t * d_m)])
            J_m_next = J(m_vol_t)

        m = m + t * d_m
        m_vol=np.array([makevol(im, mask > 0) for im in (m)])
        g_prev = g
        g = grad_J(m_vol)
        gamma = np.linalg.norm(g) ** 2 / np.linalg.norm(g_prev) ** 2
        d_m = -g + gamma * d_m
        k = k + 1
        # store.append(m)

    if log:
        norm_g_list = np.array(norm_g_list)
        np.save('./logs/conjgrad_{}.npy'.format(date_time), norm_g_list)

    return m

v0=np.zeros_like(volumes_all)
m_tilde_0=np.zeros_like(volumes_all)
m_adj=m_tilde_0-v0

kdata_init=data.reshape(nb_channels,ntimesteps,-1)
dens_adj=True
trajectory=radial_traj
mu=0

#weights_test=weights
kdata_all_channels_all_slices=np.load(filename_kdata)
#volumes_corrected_test = simulate_radial_undersampled_images_multi_new(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,ntimesteps=ntimesteps,density_adj=dens_adj,useGPU=False,light_memory_usage=True,retained_timesteps=retained_timesteps,weights=weights_test)
signals_0=volumes_all[:,mask>0]


J=lambda m:J_admm(m, kdata_init, dens_adj, trajectory,mu,m_adj,b1_all_slices,weights=None)
grad_J=lambda m:grad_J_admm(m,signals_0,dens_adj,trajectory,mu,m_adj,b1_all_slices,mask,weights=None,retained_timesteps=None)




J_list=[]
num=5
max_t = 0.1
m=volumes_all
n_samples=radial_traj.get_traj().reshape(ntimesteps,-1,3).shape[1]

J_m=J(m)
#n_samples=radial_traj.get_traj().reshape(ntimesteps,-1,3).shape[1]
g=grad_J(m)
d_m=-g
d_m_vol=np.array([makevol(im,mask>0) for im in d_m])
slope = np.real(np.dot(g.flatten(),d_m.conj().flatten()))
slope=-np.linalg.norm(d_m.flatten())**2
t_array=np.arange(0,max_t,max_t/num)

for t in tqdm(t_array):
     J_list.append(J(m+t*d_m_vol))

plt.close("all")
plt.figure()
plt.plot(t_array,J_list)
plt.plot(t_array,J_m+slope*t_array)



niter_admm=4
signals_0=volumes_all[:,mask>0]
niter_cg=4
mu=1


optimizer = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=radial_traj, split=100, pca=True,
                                     threshold_pca=15, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                     b1=b1_all_slices, threshold_ff=0.9, dictfile_light=dictfile_light,
                                     return_matched_signals=True)


signals_init = np.zeros_like(signals_0)
v_curr=np.zeros_like(signals_0)
matched_signals=np.zeros_like(signals_0)

all_maps_admm={}



for iter in range(niter_admm):




    m_adj=matched_signals-v_curr
    m_adj=np.array([makevol(im,mask>0) for im in m_adj])


    J=lambda m:J_admm(m, kdata_init, dens_adj, trajectory,mu,m_adj,b1_all_slices,weights=None)
    grad_J=lambda m:grad_J_admm(m,signals_0,dens_adj,trajectory,mu,m_adj,b1_all_slices,mask,weights=None,retained_timesteps=None)

    signals=conjgrad(J,grad_J,signals_init,mask,maxiter=niter_cg,plot=True)

    all_maps, matched_signals = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, signals+v_curr,
                                                                                       retained_timesteps=None)

    all_maps_admm[iter] = all_maps[0]

    v_curr+=signals-matched_signals
    signals_init=matched_signals.astype("complex64")




curr_file=file_map
return_cost=True
dx=1
dy=1
dz=5
all_maps=all_maps_admm
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

    if return_cost:
        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "_it{}_{}.mha".format(
            iter, "correlation")
        io.write(file_mha, makevol(all_maps[iter][2],mask>0), tags={"spacing": [dz, dx, dy]})

        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "_it{}_{}.mha".format(
            iter, "phase")
        io.write(file_mha, makevol(all_maps[iter][3],mask>0), tags={"spacing": [dz, dx, dy]})


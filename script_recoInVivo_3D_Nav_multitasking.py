
#import matplotlib
#matplotlib.u<se("TkAgg")
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
import twixtools
from mutools import io
import cv2
import scipy

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./data/InVivo/3D"




#localfile ="/20211105_TestCS_MRF/meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"
localfile = "/20211122_EV_MRF/meas_MID00146_FID42269_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"
#localfile = "/20211122_EV_MRF/meas_MID00147_FID42270_raFin_3D_tra_1x1x5mm_FULL_incoherent.dat"
#localfile = "/20211122_EV_MRF/meas_MID00148_FID42271_raFin_3D_tra_1x1x5mm_FULL_high_res.dat"
#localfile = "/20211122_EV_MRF/meas_MID00149_FID42272_raFin_3D_tra_1x1x5mm_USx2.dat"

# localfile = "/20211123_Phantom_MRF/meas_MID00317_FID42440_raFin_3D_tra_1x1x5mm_FULL_optimRG_vitro.dat"
# localfile = "/20211123_Phantom_MRF/meas_MID00318_FID42441_raFin_3D_tra_1x1x5mm_FULL_standardRG_vitro.dat"
# localfile = "/20211123_Phantom_MRF/meas_MID00319_FID42442_raFin_3D_tra_1x1x5mm_FULL_optimRNoG_vitro.dat"
# localfile = "/20211123_Phantom_MRF/meas_MID00320_FID42443_raFin_3D_tra_1x1x5mm_FULL_optimG_vitro.dat"
# localfile = "/20211123_Phantom_MRF/meas_MID00321_FID42444_raFin_3D_tra_1x1x5mm_FULL_standardRNoG_vitro.dat"

localfile = "/20211129_BM/meas_MID00085_FID43316_raFin_3D_FULL_highRES_incoh.dat"
#localfile = "/20211129_BM/meas_MID00086_FID43317_raFin_3D_FULL_new_highRES_inco.dat"
#localfile = "/20211129_BM/meas_MID00087_FID43318_raFin_3D_FULL_new_highRES_stack.dat"
#localfile = "/20211209_AL_Tongue/meas_MID00258_FID45162_raFin_3D_tra_1x1x5mm_FULl.dat"

#localfile = "/20211217_Phantom_MRF/meas_MID00252_FID47293_raFin_3D_tra_1x1x5mm_FULl.dat"
#localfile = "/20211220_Phantom_MRF/meas_MID00026_FID47383_raFin_3D_tra_1x1x5mm_FULl.dat"
#localfile = "/20211220_Phantom_MRF/meas_MID00027_FID47384_raFin_3D_tra_1x1x5mm_FULl_reduced_zFOV.dat"
#localfile = "/20211220_Phantom_MRF/meas_MID00028_FID47385_raFin_3D_tra_1x1x5mm_FULL_newpulse.dat"
# localfile = "/20211220_Phantom_MRF/meas_MID00029_FID47386_raFin_3D_tra_1x1x5mm_FULL_newpulse_reducedzFOV.dat"
# localfile = "/20211220_Phantom_MRF/meas_MID00038_FID47395_raFin_3D_tra_1x1x5mm_FULl_reducedzFOV.dat"
# localfile = "/20211220_Phantom_MRF/meas_MID00040_FID47397_raFin_3D_tra_1x1x5mm_FULL_newpulse_reducedzFOV.dat"
# localfile = "/20211220_Phantom_MRF/meas_MID00032_FID47389_raFin_3D_tra_1x1x5mm_FULL_newpulse_reducedxyzFOV.dat"

# localfile = "/20211221_Phantom_Flash/meas_MID00025_FID47488_ra_3D_tra_1x1x5mm_FULl.dat"
# localfile = "/20211221_Phantom_Flash/meas_MID00023_FID47486_ra_3D_tra_1x1x3mm_FULL_new.dat"
# localfile = "/20211221_Phantom_Flash/meas_MID00027_FID47490_ra_3D_tra_1x1x5mm_FULl_reducedFOV.dat"
# localfile = "/20211221_Phantom_Flash/meas_MID00026_FID47489_ra_3D_tra_1x1x3mm_FULL_new_reducedFOV.dat"

localfile = "/20211221_EV/meas_MID00044_FID47507_raFin_3D_FULL_new_highRES_inco_new.dat"
localfile = "/20211221_EV/meas_MID00045_FID47508_raFin_3D_FULL_new_highRES_inco.dat"
#localfile = "/20211221_EV_/meas_MID00046_FID47509_raFin_3D_FULL_new_highRES_stack.dat"

localfile = "/20220106/meas_MID00021_FID48331_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile = "/20220106/meas_MID00167_FID48477_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile = "/20220106_JM/meas_MID00180_FID48490_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile = "/20220113_CS/meas_MID00163_FID49558_raFin_3D_tra_1x1x5mm_FULL_50GS_read.dat"
#localfile = "/20220113_CS/meas_MID00164_FID49559_raFin_3D_tra_1x1x5mm_FULL_50GS_slice.dat"
#localfile = "/20220118_BM/meas_MID00151_FID49924_raFin_3D_tra_1x1x5mm_FULL_read_nav.dat"

localfile="/phantom.001.v1/phantom.001.v1.dat"
#localfile="/phantom.001.v1/meas_MID00030_FID51057_raFin_3D_phantom_mvt_0"


filename = base_folder+localfile



#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

filename_save=str.split(filename,".dat") [0]+".npy"
#filename_nav_save=str.split(base_folder+"/phantom.001.v1/phantom.001.v1.dat",".dat") [0]+"_nav.npy"
filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"

folder = "/".join(str.split(filename,"/")[:-1])

suffix="_allspokes8"

filename_b1 = str.split(filename,".dat") [0]+"_b1{}.npy".format("")
filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"

filename_volume = str.split(filename,".dat") [0]+"_volumes{}.npy".format(suffix)
filename_volume_corrected = str.split(filename,".dat") [0]+"_volumes_corrected{}.npy".format(suffix)
filename_volume_corrected_final = str.split(filename,".dat") [0]+"_volumes_corrected_final{}.npy".format("")
filename_kdata = str.split(filename,".dat") [0]+"_kdata{}.npy".format("")
filename_mask= str.split(filename,".dat") [0]+"_mask{}.npy".format("")
filename_oop=str.split(filename,".dat") [0]+"_volumes_oop{}.npy".format(suffix)
filename_oop_corrected=str.split(filename,".dat") [0]+"_volumes_oop_corrected{}.npy".format(suffix)

filename_dico_volumes_corrected=str.split(filename,".dat") [0]+"_dico_volumes_corrected{}.pkl".format(suffix)
filename_dico_kdata_retained=str.split(filename,".dat") [0]+"_dico_kdata_retained{}.pkl".format(suffix)
filename_m_opt=str.split(filename,".dat") [0]+"_m_opt{}.pkl".format(suffix)

dictfile = "./mrf_dictconf_SimReco2_adjusted_reco4.dict"
ind_dico = 200

filename_dico_comp = str.split(dictfile,".dict") [0]+"_phi_dico_{}comp.npy".format(ind_dico)
#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"

density_adj_radial=True
use_GPU = True
light_memory_usage=True
#Parsed_File = rT.map_VBVD(filename)
#idx_ok = rT.detect_TwixImg(Parsed_File)
#RawData = Parsed_File[str(idx_ok)]["image"].readImage()

if str.split(filename_seqParams,"/")[-1] not in os.listdir(folder):

    twix = twixtools.read_twix(filename,optional_additional_maps=["sWipMemBlock","sKSpace"],optional_additional_arrays=["SliceThickness"])

    if np.max(np.argwhere(np.array(twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"])>0))>=16:
        use_navigator_dll = True
    else:
        use_navigator_dll = False



    alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
    x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
    y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
    z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]

    nb_part = twix[-1]["hdr"]["Meas"]["Partitions"]

    dico_seqParams = {"alFree":alFree,"x_FOV":x_FOV,"y_FOV":y_FOV,"z_FOV":z_FOV,"use_navigator_dll":use_navigator_dll,"nb_part":nb_part}

    del alFree

    file = open(filename_seqParams, "wb")
    pickle.dump(dico_seqParams, file)
    file.close()

else:
    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()



try:
    del twix
except:
    pass


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




if str.split(filename_save,"/")[-1] not in os.listdir(folder):
    if 'twix' not in locals():
        print("Re-loading raw data")
        twix = twixtools.read_twix(filename)

    mdb_list = twix[-1]['mdb']
    if nb_gating_spokes > 0:
        print("Reading Navigator Data....")
        data_for_nav = []
        k = 0
        for i, mdb in enumerate(mdb_list):
            if mdb.is_image_scan() and mdb.mdh[14][9]:
                data_for_nav.append(mdb)

                #print("i : {} / k : {} / Line : {} / Part : {}".format(i, k, mdb.cLin, mdb.cPar))
                k += 1
        data_for_nav = np.array([mdb.data for mdb in data_for_nav])
        data_for_nav = data_for_nav.reshape((int(nb_part),int(nb_gating_spokes))+data_for_nav.shape[1:])

        if data_for_nav.ndim==3:
            data_for_nav=np.expand_dims(data_for_nav,axis=-2)

        data_for_nav = np.moveaxis(data_for_nav,-2,0)
        np.save(filename_nav_save, data_for_nav)

    del mdb_list

    ##################################################
    mapped = twixtools.map_twix(twix)
    try:
        del twix
    except:
        pass
    data = mapped[-1]['image']
    del mapped
    data = data[:].squeeze()
    if data.ndim == 3:
        data = np.expand_dims(data, axis=0)
        data=np.moveaxis(data,-2,-3)
    else:
        data = np.moveaxis(data, 0, -2)
        data = np.moveaxis(data, 1, 0)

    np.save(filename_save,data)



else :
    data = np.load(filename_save)
    if nb_gating_spokes>0:
        data_for_nav=np.load(filename_nav_save)

try:
    del twix
except:
    pass

data_shape = data.shape



#data_for_nav=data_for_nav[:,:nb_gating_spokes,:,:]
#data_for_nav = np.moveaxis(data_for_nav,-2,1)

#ntimesteps = 1400
window=8


nb_channels=data_shape[0]
nb_allspokes = data_shape[-3]
npoint = data_shape[-1]
nb_slices = data_shape[-2]
image_size = (nb_slices, int(npoint/2), int(npoint/2))
undersampling_factor=1


dx = x_FOV/(npoint/2)
dy = y_FOV/(npoint/2)
dz = z_FOV/nb_slices

#file_name_nav_mat=str.split(filename,".dat") [0]+"_nav.mat"
#savemat(file_name_nav_mat,{"Kdata":data_for_nav})

if str.split(filename_kdata,"/")[-1] in os.listdir(folder):
    del data



if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    # Density adjustment all slices

    if density_adj_radial:
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density,tuple(range(data.ndim-1)))
    else:
        density=1
    #kdata_all_channels_all_slices = data.reshape(-1, npoint)
    #del data
    print("Performing Density Adjustment....")
    data *= density
    np.save(filename_kdata, data)
    del data

    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)

#kdata_all_channels_all_slices=np.array(groupby(kdata_all_channels_all_slices,window,axis=1))
#ntimesteps=kdata_all_channels_all_slices.shape[0]
#kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(nb_channels,-1,nb_slices,npoint)
ntimesteps=175


#
# cond_gating_spokes=np.ones(nb_segments).astype(bool)
# cond_gating_spokes[::int(nb_segments/nb_gating_spokes)]=False
# kdata_retained_no_gating_spokes_list=[]
# for i in tqdm(range(nb_channels)):
#     kdata_retained_no_gating_spokes,traj_retained_no_gating_spokes,retained_timesteps=correct_mvt_kdata(kdata_all_channels_all_slices[i].reshape(nb_segments,-1),radial_traj.get_traj(),cond_gating_spokes,175,density_adj=False)
#     kdata_retained_no_gating_spokes_list.append(kdata_retained_no_gating_spokes)
#
# radial_traj.traj_for_reconstruction=traj_retained_no_gating_spokes
# Coil sensi estimation for all slices

print("Calculating Coil Sensitivity....")

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
#radial_traj.adjust_traj_for_window(window)

nb_segments=radial_traj.get_traj().shape[0]

if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage)
    np.save(filename_b1,b1_all_slices)
    del kdata_all_channels_all_slices
    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    b1_all_slices=np.load(filename_b1)


# sl=int(b1_all_slices.shape[1]/2)
# list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
# plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))
#

b1_full = np.ones(image_size)
b1_full=np.expand_dims(b1_full,axis=0)



#b1_all_slices=b1_full

print("Processing Nav Data...")
data_for_nav=np.load(filename_nav_save)

nb_allspokes=nb_segments
nb_slices=data_for_nav.shape[1]
nb_channels=data_for_nav.shape[0]
npoint=data_for_nav.shape[-1]

all_timesteps = np.arange(nb_allspokes)
nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint, nb_slices=nb_slices,
                       applied_timesteps=list(nav_timesteps))

nav_image_size = (int(npoint / 2),)

print("Calculating Sensitivity Maps for Nav Images...")
b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
b1_nav_mean = np.mean(b1_nav, axis=(1, 2))


print("Rebuilding Nav Images...")
images_nav_mean = np.abs(simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, b1_nav_mean))


print("Estimating Movement...")
shifts = list(range(-20, 20))
bottom = 50
top = 150
displacements = calculate_displacement(images_nav_mean, bottom, top, shifts)

displacement_for_binning = displacements
bin_width = 5
max_bin = np.max(displacement_for_binning)
min_bin = np.min(displacement_for_binning)

bins = np.arange(min_bin, max_bin + bin_width, bin_width)
#print(bins)
categories = np.digitize(displacement_for_binning, bins)
df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
df_groups = df_cat.groupby("cat").count()

group_1=(categories==2)
group_2=(categories==3)
group_3=(categories==4)
group_4=(categories==5)

groups=[group_1,group_2,group_3,group_4]

ntimesteps=nb_gating_spokes
nav_spoke_groups=np.argmin(np.abs(np.arange(0, ntimesteps, 1).reshape(-1, 1) - np.arange(0, ntimesteps,ntimesteps / nb_gating_spokes).reshape(1,-1)),axis=-1)
data_mt_training=copy(data_for_nav)
data_mt_training=np.squeeze(data_mt_training)
Sk=np.zeros((npoint,len(groups),ntimesteps),dtype=data_for_nav.dtype)
Sk_mask=np.ones((npoint,len(groups),ntimesteps),dtype=int)

data_mt_training_on_timesteps = np.zeros((nb_slices,ntimesteps,npoint),dtype=data_for_nav.dtype)

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



dictfile = "./mrf_dictconf_SimReco2_adjusted_reco4_w28_simmid_point_start0.dict"
ind_dico = 20

filename_dico_comp = str.split(dictfile,".dict") [0]+"_phi_dico_{}comp.npy".format(ind_dico)

if str.split(filename_dico_comp,"/")[-1]  not in os.listdir():

    FF_list = list(np.arange(0., 1.0, 0.05))
    keys, signal = read_mrf_dict(dictfile, FF_list)

    import dask.array as da



    u_dico, s_dico, vh_dico = da.linalg.svd(da.from_array(signal))
    s_dico=np.array(s_dico)
    vh_dico=np.array(vh_dico)



    # plt.figure()
    # plt.plot(np.cumsum(s_dico)/np.sum(s_dico))

    #ind_dico = ((np.cumsum(s_dico)/np.sum(s_dico))<0.99).sum()
    #ind_dico=20

    vh_dico_retained = vh_dico[:ind_dico,:]
    phi_dico = vh_dico_retained[:,:ntimesteps]

    del u_dico
    del s_dico
    del vh_dico

    del vh_dico_retained

    #del signal


    np.save(filename_dico_comp,phi_dico)
else:
    filename_dico_comp = str.split(dictfile,".dict") [0]+"_phi_dico_{}comp.npy".format(ind_dico)
    phi_dico=np.load(filename_dico_comp)


Sk_cur = copy(Sk)
niter=100
diffs = []
tol_diff = 1e-3
variance_explained=0.95
proj_on_fingerprints=True

for i in tqdm(range(niter)):
    Sk_1 = Sk_cur.reshape(Sk_cur.shape[0],-1)
    u_1, s_1, vh_1 = np.linalg.svd(Sk_1, full_matrices=False)

    Sk_2 = np.moveaxis(Sk_cur,1,0).reshape(Sk_cur.shape[1],-1)
    u_2, s_2, vh_2 = np.linalg.svd(Sk_2, full_matrices=False)

    Sk_3 = np.moveaxis(Sk_cur, 2, 0).reshape(Sk_cur.shape[2], -1)
    if proj_on_fingerprints:
        Sk_3 = phi_dico.T @ phi_dico.conj() @ Sk_3
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

Sk_final = copy(Sk_cur)
del Sk_cur
#
# plt.figure()
# plt.plot(np.abs(Sk_final[-1,-1,:]))
# plt.plot(np.abs(Sk_final[-1,-1,:])*Sk_mask[-1,-1,:],linestyle="dotted")
#
# # D=Sk_final.reshape(npoint,-1)
# # u, s, vh = np.linalg.svd(D, full_matrices=False)
#
#
#
# # plt.figure()
# # plt.plot(vh.reshape(-1,len(groups),ntimesteps)[0,:,-1])
# # plt.figure()
# # plt.plot(vh.reshape(-1,len(groups),ntimesteps)[0,1,:])
#
#
#
# i_sig=np.random.choice(signal.shape[0])
# proj_sig_i = phi_dico.T@phi_dico.conj()@signal[i_sig]
#
# plt.figure()
# plt.plot(np.abs(proj_sig_i),label="Projected signal")
# plt.plot(np.abs(signal[i_sig]),label="Original signal {}".format(i_sig),linestyle="dotted")
# plt.legend()
#

Sk_final_3 = np.moveaxis(Sk_final,2,0).reshape(Sk_final.shape[2],-1)
Sk_final_3_proj = phi_dico.T@phi_dico.conj()@Sk_final_3
Sk_mask_reshaped = np.moveaxis(Sk_mask,2,0).reshape(Sk_mask.shape[2],-1)


kdata=data_for_nav
nb_channels = kdata.shape[0]
npoint = kdata.shape[-1]
nb_slices = kdata.shape[1]
nb_gating_spokes = kdata.shape[2]

j = np.random.choice(Sk_final_3_proj.shape[-1])
#j=1917
plt.figure()
plt.plot(np.abs(Sk_final_3[:,j]*Sk_mask_reshaped[:,j]),label="Original fingerprint {}".format(""),linestyle="dotted")
plt.plot(np.abs(Sk_final_3[:,j]),label="Completed fingerprint {}".format(""))
plt.plot(np.abs(Sk_final_3_proj[:,j]),label="Projected fingerprint")
plt.legend()

test_signal=Sk[400,:,:]
plt.figure()
plt.plot(np.real(test_signal.T))



def calc_y(x):
    global A
    global intercept
    if intercept:
        w=x[:signal.shape[0]].reshape(1,-1)
        b=x[signal.shape[0]:]
        y_predict = b + w @ signal
    else:
        w=x.reshape(1,-1)
        y_predict = w @ signal

    return y_predict

def objective(x):
    global y
    return np.sum(np.abs((y-calc_y(x)))**2) + 10*np.sum(np.abs(x))



intercept=False

y=np.real(Sk[400,0,:])
A=np.real(signal)

if intercept:
    x0 = np.zeros(A.shape[0]+y.shape[0],dtype=A.dtype)
else:
    x0 = np.zeros(A.shape[0], dtype=A.dtype)
#no_bnds = (-1.0e10, 1.0e10)
#bnds = tuple([no_bnds] *x0.shape[0])

from scipy.optimize import minimize

solution = minimize(objective,x0)
x = solution.x




#
Sk_final_3_proj_for_nav_image=Sk_final_3_proj.reshape(ntimesteps,npoint,len(groups))
Sk_final_3_proj_for_nav_image=np.moveaxis(Sk_final_3_proj_for_nav_image,2,0).astype("complex64")

Sk_final_3_for_nav_image=Sk_final_3.reshape(ntimesteps,npoint,len(groups))
Sk_final_3_for_nav_image=np.moveaxis(Sk_final_3_for_nav_image,2,0).astype("complex64")

Sk_for_nav_image = np.moveaxis(Sk,1,0)
Sk_for_nav_image = np.moveaxis(Sk_for_nav_image,2,1)

#
#

nav_traj_completed_proj = Navigator3D(direction=[0, 0, 1], npoint=npoint, nb_slices=len(groups),
                       applied_timesteps=list(range(len(groups))))

images_nav_mean_original = np.abs(simulate_nav_images_multi(np.expand_dims(Sk_for_nav_image,axis=0), nav_traj_completed_proj, nav_image_size, b1_nav_mean))

gr=2
plt.figure()
plt.imshow(np.abs(images_nav_mean_original[gr,:,:]),cmap="gray")
plt.title("Original 0-filled Image Respiratory Bin {}".format(gr))

images_nav_mean_completed = np.abs(simulate_nav_images_multi(np.expand_dims(Sk_final_3_for_nav_image,axis=0), nav_traj_completed_proj, nav_image_size, b1_nav_mean))

plt.figure()
plt.imshow(np.abs(images_nav_mean_completed[0,:,:]),cmap="gray")
plt.title("Completed Image Respiratory Bin {}".format(gr))

images_nav_mean_proj = np.abs(simulate_nav_images_multi(np.expand_dims(Sk_final_3_proj_for_nav_image,axis=0), nav_traj_completed_proj, nav_image_size, b1_nav_mean))
plt.figure()
plt.imshow(np.abs(images_nav_mean_proj[1,:,:]),cmap="gray")
plt.title("Projected Image Respiratory Bin {}".format(gr))

images_nav_mean_timesteps = np.abs(simulate_nav_images_multi(np.expand_dims(data_mt_training_on_timesteps,axis=0), nav_traj_completed_proj, nav_image_size, b1_nav_mean))
sl=0
plt.figure()
plt.imshow(np.abs(images_nav_mean_timesteps[sl,:,:]),cmap="gray")
plt.title("Original Nav Image Slice {}".format(sl))






#
# Sk_final_proj = Sk_final_3_proj.reshape(Sk_final.shape[2], Sk_final.shape[0], Sk_final.shape[1])
# Sk_final_proj = np.moveaxis(Sk_final_proj, 0, 2)
#
# D=Sk_final_proj.reshape(npoint,-1)
# u, s, vh = np.linalg.svd(D, full_matrices=False)
#
#
# L0 = 8
#
# phi = (vh)[:L0,:]
# phi=vh[:L0,:]
#
#
# i_sig = np.random.choice(D_non_proj.shape[0])
#
# proj_sig_i = phi.T@phi.conj()@D_non_proj[i_sig]
# #proj_sig_i_non_proj = phi_non_proj.T@phi_non_proj.conj()@D_non_proj[i_sig]
#
#
# plt.figure()
# plt.plot(np.abs(proj_sig_i),label="Projected navigator signal - basis constrained by Fingerprints simulation")
# #plt.plot(np.abs(proj_sig_i_non_proj),label="Projected navigator signal - basis not constrained by Fingerprints simulation")
# plt.plot(np.abs(D_non_proj[i_sig]),label="Original signal",linestyle="dotted")
# plt.legend()
#
#
# Sk_final_non_proj_HOSVD = phi_non_proj.T@phi_non_proj.conj()@D_non_proj.T
# Sk_final_non_proj_HOSVD =Sk_final_non_proj_HOSVD.reshape(Sk_final.shape[1],Sk_final.shape[-1],Sk_final.shape[0]).astype("complex64")
# #Sk_final_non_proj_HOSVD =np.moveaxis(Sk_final_non_proj_HOSVD,2,0).astype("complex64")
#
# gr=2
# images_nav_mean_nonproj_HOSVD = np.abs(simulate_nav_images_multi(np.expand_dims(Sk_final_non_proj_HOSVD,axis=0), nav_traj_completed_proj, nav_image_size, b1_nav_mean))
# plt.figure()
# plt.imshow(np.abs(images_nav_mean_nonproj_HOSVD[gr,:,:]),cmap="gray")
# plt.title("Projected Image Respiratory Bin {}".format(gr))


# plt.figure()
# for g in range(len(groups)):
#     plt.plot(vh.reshape(-1,len(groups),ntimesteps)[g,:,-1],label="Movement group {}".format(g))
# plt.legend()
#
# plt.figure()
# for g in range(len(groups)):
#     plt.plot(vh.reshape(-1,len(groups),ntimesteps)[g,:,0],label="Movement group {}".format(g))
# plt.legend()
#
# plt.figure()
# plt.plot(vh.reshape(-1,len(groups),ntimesteps)[0,1,:])


D_non_proj=Sk_final.reshape(npoint,-1)
u_non_proj, s_non_proj, vh_non_proj = np.linalg.svd(D_non_proj, full_matrices=False)
L0 = 32
#phi_non_proj = (vh_non_proj)[:L0,:]
phi_non_proj=vh_non_proj[:L0,:]
phi=phi_non_proj



data = np.load(filename_save)
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
useGPU = True
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
    global dx
    global dy
    global dz
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
    # return np.linalg.norm(kdata_error)**2
    density = np.abs(np.linspace(-1, 1, npoint))
    density = np.expand_dims(density, tuple(range(kdata_error.ndim - 1)))
    kdata_error *= np.sqrt(density)

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

import time
start_time = time.time()
useGPU=True
J_m= J(m0)
print("GPU --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
useGPU=True
grad_Jm= grad_J(m0)
print("Grad GPU --- %s seconds ---" % (time.time() - start_time))



start_time = time.time()
useGPU=False
J_m= J(m0)
print("No GPU --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
useGPU=False
grad_Jm= grad_J(m0)
print("Grad No GPU --- %s seconds ---" % (time.time() - start_time))


sl = int(nb_slices/2)
l=np.random.choice(L0)

plt.imshow(np.abs(grad_Jm[l,sl]))



use_GPU=True
J_list=[]
num=10
max_t = 0.00001
t_array = np.arange(0,max_t,max_t/num)
for t in tqdm(t_array):
    J_list.append(J(m0-t*grad_Jm))


slope = -np.linalg.norm(grad_Jm)**2

plt.figure()
plt.plot(J_list)
plt.plot(np.arange(num),J_m+t_array*slope)


#Conj grad test
use_GPU = True
J_list = []

g=grad_Jm
d_m=-g
slope = np.real(np.dot(g.flatten(),d_m.flatten().conj()))

num = 10
max_t = 0.00001
t_array = np.arange(0, max_t, max_t / num)
for t in tqdm(t_array):
    J_list.append(J(m0 +t*d_m))

plt.figure()
plt.plot(J_list)
plt.plot(np.arange(num), J_m + t_array * slope)




#
#
#
#
#
#
# from scipy.optimize import minimize,basinhopping,dual_annealing
#
# def f(x):
#     global m0
#     x=x.reshape((2,)+m0.shape)
#     return J((x[0]+1j*x[1]).astype("complex64"))
#
# def  Jf(x):
#     global m0
#     x = x.reshape((2,) + m0.shape)
#     grad = grad_J((x[0] + 1j * x[1]).astype("complex64"))
#     grad=np.expand_dims(grad.flatten(),axis=0)
#     grad = np.concatenate([grad.real, grad.imag], axis=0)
#     grad = grad.flatten()
#     return grad
#
#
# x0=np.expand_dims(m0.flatten(),axis=0)
# x0 = np.concatenate([x0.real,x0.imag],axis=0)
# x0=x0.flatten()
# #
# #
# x_opt=minimize(f,x0,method='CG',jac=Jf)
# np.save("x_opt_CG_32.npy",x_opt)




#
useGPU=True
eps_grad=0.001
ind=(0,0,0,0)
h = np.zeros(m0.shape,dtype=m0.dtype)
h[ind[0],ind[1],ind[2],ind[3]]=eps_grad

diff_Jm = J(m0+h)-J_m
diff_Jm_approx = grad_Jm[ind[0],ind[1],ind[2],ind[3]]*eps_grad




# m_opt=graddesc(J,grad_J,m0,alpha=0.1,log=True,tolgrad=1e-10)
#
#
# m_opt=graddesc_linsearch(J,grad_J,m0,alpha=0.1,beta=0.6,log=True,tolgrad=1e-10,t0=300)



import time
start_time = time.time()
filename_m_opt=str.split(filename,".dat") [0]+"_m_opt_L0{}.npy".format(L0)
filename_m_opt_figure=str.split(filename,".dat") [0]+"_m_opt_L0{}.jpg".format(L0)
#m_opt=conjgrad(J,grad_J,m0,alpha=0.1,beta=0.3,log=True,tolgrad=1e-10,t0=100,maxiter=1000,plot=True,filename_save=filename_m_opt)


log=True
plot=True
useGPU=False

filename_save = filename_m_opt
# t0=100
# beta=0.3
# alpha=0.1
# tolgrad=1e-10

t0=0.00001
beta=0.6
alpha=0.05
tolgrad=1e-4

maxiter=200

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
    slope = np.real(np.dot(g.flatten(),d_m.flatten().conj()))
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
            plt.savefig(filename_m_opt_figure)
        J_m_next=J(m+t*d_m)



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

#
#
# #filename_m_opt=str.split(filename,".dat") [0]+"_m_opt_L0{}.npy".format(L0)
np.save(filename_m_opt,m_opt)
print("--- %s seconds ---" % (time.time() - start_time))


#filename_phi=str.split(filename,".dat") [0]+"_phi_L0{}.npy".format(L0)
#np.save(filename_phi,phi)
# print("--- %s seconds ---" % (time.time() - start_time))
#
#
#

L0 = 32
filename_phi=str.split(filename,".dat") [0]+"_phi_L0{}.npy".format(L0)
filename_m_opt=str.split(filename,".dat") [0]+"_m_opt_L0{}.npy".format(L0)

m_opt = np.load(filename_m_opt)
phi = np.load(filename_phi)

sl=int(nb_slices/2)
l=np.random.choice(L0)
plt.figure()
plt.imshow(np.abs(m_opt[l,sl,:,:]))
plt.title("basis image for l={}".format(l))

gr=3
phi_gr=phi[:,gr,:]
sl=int(nb_slices/2)
volumes_rebuilt_gr=(m_opt[:,sl,:,:].reshape((L0,-1)).T@phi_gr).reshape(image_size[1],image_size[2],ntimesteps)
volumes_rebuilt_gr=np.moveaxis(volumes_rebuilt_gr,-1,0)
animate_images(volumes_rebuilt_gr)
#
#
volumes_all_rebuilt = (m_opt.reshape((L0,-1)).T@phi_gr).reshape(image_size[0],image_size[1],image_size[2],ntimesteps)
volumes_all_rebuilt=np.moveaxis(volumes_all_rebuilt,-1,0)
#
filename_volume_rebuilt_multitasking=str.split(filename,".dat") [0]+"_volumes_mt_L0{}_gr{}.npy".format(L0,gr)
np.save(filename_volume_rebuilt_multitasking,volumes_all_rebuilt)
#
#
# v_error_final= grad_J(m0)
#
#
#
# sl=int(nb_slices/2)
# image_list = list(np.abs(dm[:,sl,:,:]))
# plot_image_grid(image_list,(3,3))
#
# v_error_final=np.moveaxis(v_error_final,0,1)
#
# dm = v_error_final@(phi.reshape(L0,-1).T.conj())
#
# ##volumes for slice taking into account coil sensi
# # print("Building Volumes....")
# # if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
# #     volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=True,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
# #     np.save(filename_volume,volumes_all)
# #     # sl=20
# #     # ani = animate_images(volumes_all[:,sl,:,:])
# #     del volumes_all
# #
# # print("Building Mask....")
# # if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
# #     selected_spokes = np.r_[10:400]
# #     selected_spokes=None
# #     mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/25, normalize_kdata=True,light_memory_usage=True,selected_spokes=selected_spokes)
# #     np.save(filename_mask,mask)
# #     animate_images(mask)
# #     del mask
#
# #animate_images(np.abs(volumes_all[:,int(nb_slices/2),:,:]))
#
# # #Check modulation of nav signal by MRF
# # plt.figure()
# # rep=0
# # signal_MRF = np.abs(volumes_all[:,int(nb_slices/2),int(npoint/4),int(npoint/4)])
# # signal_MRF = signal_MRF/np.max(signal_MRF)
# # signal_nav =image_nav[rep,:,int(npoint/4)]
# # signal_nav = signal_nav/np.max(signal_nav)
# # plt.plot(signal_MRF,label="MRF signal at centre pixel")
# # plt.scatter(x=(nav_timesteps/8).astype(int),y=signal_nav,c="r",label="Nav image at centre pixel for rep {}".format(rep))
# # rep=4
# # plt.scatter(x=(nav_timesteps/8).astype(int),y=signal_nav,c="g",label="Nav image at centre pixel for rep {}".format(rep))
# # plt.legend()
#
# ##MASK
#
#
#
#
#
# del kdata_all_channels_all_slices
# del b1_all_slices
#
#
#
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
#volumes_corrected_final=np.load(filename_volume_corrected_final)

gr=3
L0=32
filename_volume_rebuilt_multitasking=str.split(filename,".dat") [0]+"_volumes_mt_L0{}_gr{}.npy".format(L0,gr)
volumes_corrected_final=np.load(filename_volume_rebuilt_multitasking)

volumes_corrected_final=volumes_all_rebuilt
#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])

suffix="Multitasking_L0{}_gr{}".format(L0,gr)
ntimesteps=175
if not(load_map):
    niter = 0
    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=None,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=False,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps)
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
#
#
#
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
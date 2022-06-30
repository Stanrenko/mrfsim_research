
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

suffix="_autofocus"

filename_b1 = str.split(filename,".dat") [0]+"_b1{}.npy".format("")
filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"

filename_volume = str.split(filename,".dat") [0]+"_volumes{}.npy".format(suffix)
filename_volume_corrected = str.split(filename,".dat") [0]+"_volumes_corrected{}.npy".format(suffix)
filename_mask_corrected = str.split(filename,".dat") [0]+"_mask_corrected{}.npy".format(suffix)
filename_volume_corrected_final = str.split(filename,".dat") [0]+"_volumes_corrected_final{}.npy".format("")
filename_kdata = str.split(filename,".dat") [0]+"_kdata{}.npy".format("")
filename_mask= str.split(filename,".dat") [0]+"_mask{}.npy".format("")
filename_oop=str.split(filename,".dat") [0]+"_volumes_oop{}.npy".format(suffix)
filename_oop_corrected=str.split(filename,".dat") [0]+"_volumes_oop_corrected{}.npy".format(suffix)

filename_dico_volumes_corrected=str.split(filename,".dat") [0]+"_dico_volumes_corrected{}.pkl".format(suffix)
filename_dico_kdata_retained=str.split(filename,".dat") [0]+"_dico_kdata_retained{}.pkl".format(suffix)


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


sl=int(b1_all_slices.shape[1]/2)
list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))


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
displacements = calculate_displacement(images_nav_mean, bottom, top, shifts,lambda_tv=0)

plt.figure()
plt.plot(displacements)

# displacement_for_binning = displacements
# bin_width = 8
# max_bin = np.max(displacement_for_binning)
# min_bin = np.min(displacement_for_binning)
#
# bins = np.arange(min_bin, max_bin + bin_width, bin_width)
# #print(bins)
# categories = np.digitize(displacement_for_binning, bins)
# df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
# df_groups = df_cat.groupby("cat").count()

spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
                             axis=-1)

if not (nb_segments == nb_gating_spokes):
    spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
            nb_segments / nb_gating_spokes / 2) + 1:] - 1
    spoke_groups = spoke_groups.flatten()

displacements_extrapolated = np.array([displacements[j] for j in spoke_groups])

plt.figure()
plt.plot(displacements_extrapolated)


x = np.arange(0., 2.2, 0.2)
y = np.arange(-1, 1.2, 0.2)

entropy_all = []
ent_min = np.inf
for dx in tqdm(x):
    entropy_x = []
    for dy in y:
        alpha = np.array([dx, dy, 0])
        dr = np.expand_dims(alpha, axis=(0, 1)) * np.expand_dims(displacements_extrapolated.reshape(nb_slices, 1400).T, axis=(2))
        modif = np.exp(
            1j * np.sum((radial_traj.get_traj().reshape(1400, -1, npoint, 3) * np.expand_dims(dr, axis=2)), axis=-1))
        data_modif = kdata_all_channels_all_slices * modif
        volume_full_modif = \
        simulate_radial_undersampled_images_multi(data_modif, radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=True,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)[0]
        ent = calc_grad_entropy(volume_full_modif)
        entropy_x.append(ent)
        if ent < ent_min:
            modif_final = modif
            alpha_min = alpha
            ent_min = ent

    entropy_all.append(entropy_x)


X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(15,15))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, np.array(entropy_all), rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)


#volumes for slice taking into account coil sensi
print("Building Volumes Corrected....")
if str.split(filename_volume_corrected,"/")[-1] not in os.listdir(folder):
    data_modif = kdata_all_channels_all_slices*modif_final
    volumes_all_modif=simulate_radial_undersampled_images_multi(data_modif,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=True,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
    np.save(filename_volume_corrected,volumes_all_modif)
    # sl=20
    ani = animate_images(volumes_all_modif[:,int(nb_slices/2),:,:])
    del volumes_all_modif

print("Building Mask Corrected....")
if str.split(filename_mask_corrected,"/")[-1] not in os.listdir(folder):
    selected_spokes = np.r_[10:400]
    selected_spokes=None
    data_modif = kdata_all_channels_all_slices * modif_final
    mask_modif=build_mask_single_image_multichannel(data_modif,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/25, normalize_kdata=True,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask_corrected,mask_modif)
    animate_images(mask_modif)
    del mask_modif

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
#filename_volume_corrected_final = str.split(filename,".dat") [0]+"_volumes_corrected_final_old{}.npy".format("")


mask = np.load(filename_mask_corrected)
#volumes_all = np.load(filename_volume)
volumes_all=np.load(filename_volume_corrected)


#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])


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
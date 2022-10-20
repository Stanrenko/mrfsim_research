
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

#localfile="/patient.002.v1/meas_MID00099_FID01839_raFin_3D_tra_1x1x5mm_FULL_new.dat"
#localfile="/patient.002.v1/meas_MID00103_FID01843_raFin_3D_tra_1x1x5mm_FULL_TR7000.dat"

#localfile="/patient.002.v2/meas_MID00037_FID01900_raFin_3D_tra_1x1x5mm_FULL_new.dat"
#localfile="/patient.002.v2/meas_MID00038_FID01901_raFin_3D_tra_1x1x5mm_FULL_FOV90_Sl160.dat"


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

filename_volume = str.split(filename,".dat") [0]+"_volumes{}.npy".format("")
filename_volume_corrected_bestgroup = str.split(filename,".dat") [0]+"_volumes_corrected_bestgroup{}.npy".format("")
filename_volume_corrected_final = str.split(filename,".dat") [0]+"_volumes_corrected_final{}.npy".format("")
filename_mask_corrected_final = str.split(filename,".dat") [0]+"_mask_corrected_final{}.npy".format("")
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
    if nb_gating_spokes == 0:
        data = []

        for i, mdb in enumerate(mdb_list):
            if mdb.is_image_scan():
                data.append(mdb)



    else:
        print("Reading Navigator Data....")
        data_for_nav = []
        data = []
        nav_size_initialized = False
        # k = 0
        for i, mdb in enumerate(mdb_list):
            if mdb.is_image_scan():
                if not (mdb.mdh[14][9]):
                    mdb_data_shape = mdb.data.shape
                    mdb_dtype = mdb.data.dtype
                    nav_size_initialized = True
                    break

        for i, mdb in enumerate(mdb_list):
            if mdb.is_image_scan():
                if not (mdb.mdh[14][9]):
                    data.append(mdb)
                else:
                    data_for_nav.append(mdb)
                    data.append(np.zeros(mdb_data_shape, dtype=mdb_dtype))

                # print("i : {} / k : {} / Line : {} / Part : {}".format(i, k, mdb.cLin, mdb.cPar))
                # k += 1
        data_for_nav = np.array([mdb.data for mdb in data_for_nav])
        data_for_nav = data_for_nav.reshape((int(nb_part), int(nb_gating_spokes)) + data_for_nav.shape[1:])

        if data_for_nav.ndim == 3:
            data_for_nav = np.expand_dims(data_for_nav, axis=-2)

        data_for_nav = np.moveaxis(data_for_nav, -2, 0)
        np.save(filename_nav_save, data_for_nav)

    data = np.array([mdb.data for mdb in data])
    data = data.reshape((-1, int(nb_segments)) + data.shape[1:])
    data = np.moveaxis(data, 2, 0)
    data = np.moveaxis(data, 2, 1)

    del mdb_list

    ##################################################
    try:
        del twix
    except:
        pass

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

volume_rebuilt = build_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,density_adj=False,eps=1e-6,b1=b1_all_slices,useGPU=True,normalize_kdata=False,light_memory_usage=True,is_theta_z_adjusted=False,normalize_volumes=True)
del kdata_all_channels_all_slices
kdata_all_channels_all_slices = np.load(filename_kdata)

np.save(str.split(filename,".dat") [0]+"_volume_allspokes.npy",volume_rebuilt)

from mutools import io
file_mha = filename.split(".dat")[0] + "_volume_allspokes.mha"
io.write(file_mha,np.abs(volume_rebuilt),tags={"spacing":[dz,dx,dy]})
animate_images(volume_rebuilt,cmap="gray")
plt.imshow(np.abs(volume_rebuilt[10]))
#
# b1_full = np.ones(image_size)
# b1_full=np.expand_dims(b1_full,axis=0)



#b1_all_slices=b1_full

print("Processing Nav Data...")
data_for_nav=np.load(filename_nav_save)

nb_allspokes=nb_segments
nb_slices=data_for_nav.shape[1]
nb_channels=data_for_nav.shape[0]
npoint_nav=data_for_nav.shape[-1]

all_timesteps = np.arange(nb_allspokes)
nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                       applied_timesteps=list(nav_timesteps))

nav_image_size = (int(npoint_nav/2),)

print("Calculating Sensitivity Maps for Nav Images...")
b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
b1_nav_mean = np.mean(b1_nav, axis=(1, 2))


print("Rebuilding Nav Images...")
images_nav_mean = np.abs(simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, b1_nav_mean))
plt.figure()
nb_cycles=10
plt.imshow(np.abs(images_nav_mean.reshape(-1,int(npoint_nav/2))).T[:,:(nb_cycles*nb_gating_spokes)])


image_nav_all_channels=[]
for j in range(nb_channels):
    images_series_rebuilt_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[j],axis=0), nav_traj, nav_image_size, b1=None)
    image_nav_ch = np.abs(images_series_rebuilt_nav_ch)
    # plt.figure()
    #plt.imshow(image_nav_ch.reshape(-1, int(npoint / 2)).T, cmap="gray")
    #plt.title("Image channel {}".format(j))
    image_nav_all_channels.append(image_nav_ch)
#plt.close("all")
image_nav_all_channels=np.array(image_nav_all_channels)


nb_cycles=10
for ch in range(nb_channels):
    plt.figure()
    plt.imshow(np.abs(image_nav_all_channels[ch].reshape(-1,int(npoint_nav/2)).T[:,:(nb_cycles*nb_gating_spokes)]),cmap="gray")
    plt.title("{}".format(ch))

#TEST FABIAN CODE
# from pymia.filtering.registration import MultiModalRegistration,MultiModalRegistrationParams,RegistrationType
# import SimpleITK as sitk
#
#
#
# bottom = 50
# top = 150
# image=images_nav_mean
#
# nb_gating_spokes = image.shape[1]
# nb_slices = image.shape[0]
#
# npoint_image = image.shape[-1]
# ft = np.mean(image, axis=0)
# #ft=np.mean(image_nav_best_channel,axis=0)
#     # ft=image[0]
# image_nav_for_correl = image.reshape(-1, npoint_image)
#
# nb_images = image_nav_for_correl.shape[0]
# transf = []
# corrected_images=[]
# # adj=[]
# registration_nav = MultiModalRegistration(registration_type=RegistrationType.RIGID)
#
# for j in tqdm(range(nb_images)):
#     array_ref = np.tile(ft[j % nb_gating_spokes, :],(4,1))
#     array_to_align=np.tile(image_nav_for_correl[j, :].reshape(1,-1),(4,1))
#     fixed_image = sitk.GetImageFromArray(array_ref)
#     moving_image = sitk.GetImageFromArray(array_to_align)
#
#     # specify parameters to your needs
#     params = MultiModalRegistrationParams(fixed_image)
#     corrected_image=registration_nav.execute(moving_image, params)
#
#     transf.append(registration_nav.transform)
#     corrected_images.append(sitk.GetArrayFromImage(corrected_image))
#
# corrected_images_selected= np.array(corrected_images)[:,0,:]
# plt.figure();plt.imshow(corrected_images_selected.T)
# plt.figure();plt.imshow(image_nav_for_correl.T)

print("Estimating Movement...")
shifts = list(range(-30, 30))
bottom = -shifts[0]
top = nav_image_size[0]-shifts[-1]
displacements = calculate_displacement(image_nav_all_channels[4], bottom, top, shifts,lambda_tv=0.001)
plt.figure()
plt.plot(100+displacements[:nb_cycles*nb_gating_spokes])
plt.imshow(np.abs(image_nav_all_channels[4].reshape(-1,int(npoint_nav/2)).T[:,:(nb_cycles*nb_gating_spokes)]),cmap="gray")


displacement_for_binning = displacements
bin_width = 5
max_bin = np.max(displacement_for_binning)
min_bin = np.min(displacement_for_binning)

bins = np.arange(min_bin, max_bin + bin_width, bin_width)
#print(bins)
categories = np.digitize(displacement_for_binning, bins)
df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
df_groups = df_cat.groupby("cat").count()



#INVIVO BW 5
group_1 = (categories == 4)| (categories == 3)
group_2=(categories == 5)| (categories == 6)
group_3 = (categories == 7)

groups=[group_1,group_2,group_3]

#Building volume for most spokes group



if str.split(filename_volume_corrected_bestgroup,"/")[-1] not in os.listdir(folder):

    displacement_for_binning = displacements
    bin_width = 5
    max_bin = np.max(displacement_for_binning)
    min_bin = np.min(displacement_for_binning)

    maxi = 0
    for j in range(bin_width):
        min_bin = np.min(displacement_for_binning) + j
        bins = np.arange(min_bin, max_bin + bin_width, bin_width)
        # print(bins)
        categories = np.digitize(displacement_for_binning, bins)
        df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
        df_groups = df_cat.groupby("cat").count()
        curr_max = df_groups.displacement.max()
        if curr_max > maxi:
            maxi = curr_max
            idx_cat = df_groups.displacement.idxmax()
            retained_nav_spokes = (categories == idx_cat)

    retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
        axis=-1)

    spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
        nb_segments / nb_gating_spokes / 2) + 1:] - 1  # adjustment for change of partition
    spoke_groups = spoke_groups.flatten()

    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
    included_spokes[::int(nb_segments / nb_gating_spokes)] = False
    print("Filtering KData for movement...")
    kdata_retained_final_list = []
    for i in (range(nb_channels)):
        kdata_retained_final, traj_retained_final, retained_timesteps = correct_mvt_kdata(
            kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, ntimesteps,
            density_adj=True, log=False)
        kdata_retained_final_list.append(kdata_retained_final)

    #dico_kdata_retained_registration[j] = retained_timesteps



    radial_traj_3D_corrected = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor,
                                        npoint=npoint, nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    radial_traj_3D_corrected.traj_for_reconstruction = traj_retained_final

    volumes_corrected_best_group = simulate_radial_undersampled_images_multi(kdata_retained_final_list, radial_traj_3D_corrected,
                                                                  image_size, b1=b1_all_slices,
                                                                  ntimesteps=len(retained_timesteps), density_adj=False,
                                                                  useGPU=False, normalize_kdata=False, memmap_file=None,
                                                                  light_memory_usage=True, is_theta_z_adjusted=True,
                                                                  normalize_volumes=True)

    np.save(filename_volume_corrected_bestgroup,volumes_corrected_best_group)
else:
    volumes_corrected_best_group=np.load(filename_volume_corrected_bestgroup)



#INVIVO BW 5
# group_1 = (categories == 1) | (categories == 2)| (categories == 3)
# group_2 = (categories == 4)
# group_3 = (categories == 5)
# group_4 = (categories == 6) | (categories == 7)
# groups = [group_1, group_2, group_3,group_4]
# groups=[group_2,group_3]


#PHANTOM BW 8
# group_1 = (categories == 1) | (categories == 2)
# group_2 = (categories == 3)
# group_3 = (categories == 4) | (categories == 5)
#
# groups = [group_1, group_2, group_3]


# group_1=(categories==1)|(categories==2)|(categories==3)
# group_2=(categories==4)
# group_3=(categories==5)
# group_4=(categories==6)
# group_5=(categories==7)|(categories==8)|(categories==9)
#
# groups=[group_1,group_2,group_3,group_4,group_5]

# dico_kdata_retained_registration={}


dico_traj_retained = {}
for j, g in tqdm(enumerate(groups)):
    print("######################  BUILDING FULL VOLUME AND MASK FOR GROUP {} ##########################".format(j))
    retained_nav_spokes_index = np.argwhere(g).flatten()
    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
        axis=-1)

    spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
        nb_segments / nb_gating_spokes / 2) + 1:] - 1  # adjustment for change of partition
    spoke_groups = spoke_groups.flatten()

    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
    included_spokes[::int(nb_segments / nb_gating_spokes)] = False
    #print(np.sum(included_spokes))

    kdata_retained_final, traj_retained_final_volume, retained_timesteps = correct_mvt_kdata(
        kdata_all_channels_all_slices[0].reshape(nb_segments, -1), radial_traj, included_spokes, 1,
        density_adj=True, log=False)
    #print(traj_retained_final_volume.shape[1]/800)

    dico_traj_retained[j] = traj_retained_final_volume

nyquist_total_points = np.prod(image_size)*np.pi/2
for j,gr in enumerate(groups):
    print("Group {} : {} % of Nyquist criteria".format(j,np.round(dico_traj_retained[j].shape[1]/nyquist_total_points*100,2)))

for j,gr in enumerate(groups):
    print("Group {} : {} % of Kz Nyquist criteria".format(j,len(np.unique(dico_traj_retained[j][:,:,2]))/nb_slices*100,2))

all_kz = np.unique(radial_traj.get_traj()[:,:,-1])

df_sampling = pd.DataFrame(columns=range(len(groups)),index=all_kz,data=0)
df_sampling

kz=all_kz[10]
j=0

for j,g in tqdm(enumerate(groups)):
    traj_retained = dico_traj_retained[j]
    for kz in all_kz:
        curr_traj = traj_retained[traj_retained[:,:,2]==kz]
        df_sampling.loc[kz,j]=curr_traj.shape[0]/(npoint**2*np.pi/8)

df_sampling.loc["Total"]=[len(np.unique(dico_traj_retained[j][:,:,2]))/nb_slices for j,g in enumerate(groups)]

dico_volume = {}
dico_mask = {}

#j=2
#g=groups[j]


resol_factor=1.0
center_point=int(npoint/2)





for j, g in tqdm(enumerate(groups)):
    print("######################  BUILDING FULL VOLUME AND MASK FOR GROUP {} ##########################".format(j))
    retained_nav_spokes_index = np.argwhere(g).flatten()
    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
                             axis=-1)

    spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
        nb_segments / nb_gating_spokes / 2) + 1:] - 1  # adjustment for change of partition
    spoke_groups = spoke_groups.flatten()

    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
    included_spokes[::int(nb_segments / nb_gating_spokes)] = False
    print("Filtering KData for movement...")
    kdata_retained_final_list_volume = []
    for i in tqdm(range(nb_channels)):
        kdata_retained_final, traj_retained_final_volume, retained_timesteps = correct_mvt_kdata(
            kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, 1,
            density_adj=True, log=False)
        kdata_retained_final_list_volume.append(kdata_retained_final)

    radial_traj_3D_corrected_single_volume = Radial3D(total_nspokes=nb_allspokes,
                                                      undersampling_factor=undersampling_factor, npoint=npoint,
                                                      nb_slices=nb_slices, incoherent=incoherent, mode=mode)

    if resol_factor<1:
        traj_retained_final_volume=traj_retained_final_volume.reshape(1,-1,npoint,3)
        traj_retained_final_volume=traj_retained_final_volume[:,:,(center_point-int(resol_factor*npoint/2)):(center_point+int(resol_factor*npoint/2)),:]
        kdata_retained_final_list_volume=np.array(kdata_retained_final_list_volume)
        kdata_retained_final_list_volume=kdata_retained_final_list_volume.reshape(nb_channels,1,-1,npoint)
        kdata_retained_final_list_volume = kdata_retained_final_list_volume[:, :, :,(center_point - int(resol_factor * npoint / 2)):(
                    center_point + int(resol_factor * npoint / 2))]

    radial_traj_3D_corrected_single_volume.traj_for_reconstruction = traj_retained_final_volume

    volume_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list_volume,
                                                                 radial_traj_3D_corrected_single_volume, image_size,
                                                                 b1=b1_all_slices, density_adj=False, ntimesteps=1,
                                                                 useGPU=False, normalize_kdata=False, memmap_file=None,
                                                                 light_memory_usage=True, normalize_volumes=True,
                                                                 is_theta_z_adjusted=True)

    file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
                         "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[
                                  :-1])]) + "_fullvol_gr{}_resol_{}_b1.mha".format(j, "_".join(str.split(str(resol_factor), ".")))

    io.write(file_mha, np.abs(volume_corrected[0]), tags={"spacing": [dz, dx, dy]})

    dico_volume[j] = copy(volume_corrected[0])
    mask = build_mask_single_image_multichannel(kdata_retained_final_list_volume,
                                                radial_traj_3D_corrected_single_volume, image_size, b1=b1_all_slices,
                                                density_adj=False, threshold_factor=1 / 10, normalize_kdata=False,
                                                light_memory_usage=True, selected_spokes=None, is_theta_z_adjusted=True,
                                                normalize_volumes=True)
    dico_mask[j] = copy(mask)

volume_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,
                                                                 radial_traj, image_size,
                                                                 b1=None, density_adj=False, ntimesteps=1,
                                                                 useGPU=False, normalize_kdata=False, memmap_file=None,
                                                                 light_memory_usage=True, normalize_volumes=True,
                                                                 is_theta_z_adjusted=False)

file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
                         "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_fullvol_all_groups.mha"

io.write(file_mha,np.abs(volume_all[0]),tags={"spacing":[dz,dx,dy]})

animate_images(mask)

del volume_corrected
del kdata_retained_final_list_volume
del radial_traj_3D_corrected_single_volume
del mask

import cv2
import numpy as np




for j in dico_volume.keys():
    plt.figure(figsize=(10,10))
    #plt.imshow(denoise_tv_chambolle(np.abs((dico_mask[j]*dico_volume[j])[int(nb_slices/2),:,:]),weight=0.00001))
    plt.imshow(np.abs((dico_mask[j][int(nb_slices/2),:,:]*dico_volume[j][int(nb_slices/2),:,:])))
    #plt.imshow(np.abs((dico_mask[j][int(nb_slices / 2), :, :])))



#TEST FABIAN CODE
from pymia.filtering.registration import MultiModalRegistration,MultiModalRegistrationParams,RegistrationType
import SimpleITK as sitk



index_ref = 1
mask=dico_mask[index_ref]

if str.split(filename_mask_corrected_final, "/")[-1] not in os.listdir(folder):
    np.save(filename_mask_corrected_final,mask)

dico_homographies = {}
registration = MultiModalRegistration(registration_type=RegistrationType.RIGID)

for index_to_align in tqdm(dico_volume.keys()):
    dico_homographies[index_to_align] = {}
    for sl in range(nb_slices):
        if index_to_align==index_ref:
            dico_homographies[index_to_align][sl] = sitk.Transform(3, sitk.sitkIdentity)
        else:
            array_to_align = np.abs(dico_mask[index_to_align][sl] * dico_volume[index_to_align][sl])
            array_ref = np.abs(dico_mask[index_ref][sl] * dico_volume[index_ref][sl])

            fixed_image = sitk.GetImageFromArray(array_ref)
            moving_image = sitk.GetImageFromArray(array_to_align)

              # specify parameters to your needs
            params = MultiModalRegistrationParams(fixed_image)
            registration.execute(moving_image, params)

            dico_homographies[index_to_align][sl] = registration.transform



# sl = int(nb_slices / 2)
# test_index=2
#
# array_ref = np.abs(dico_mask[index_ref][sl] * dico_volume[index_ref][sl])
#
# array_to_align = np.abs(dico_mask[test_index][sl] * dico_volume[test_index][sl])
# moving_image = sitk.GetImageFromArray(array_to_align)
#
# registered_image = registration.execute(moving_image, dico_homographies[test_index][sl])
# registered_array = sitk.GetArrayFromImage(registered_image)
#
# animate_images([array_ref,registered_array])

#
from sklearn.metrics import mutual_info_score

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

test_index=0
for sl in range(nb_slices):
    print("############### SLICE {} ################".format(sl))
    array_ref = np.abs(dico_mask[index_ref][sl] * dico_volume[index_ref][sl])
    fixed_image = sitk.GetImageFromArray(array_ref)

    array_to_align = np.abs(dico_mask[test_index][sl] * dico_volume[test_index][sl])
    moving_image = sitk.GetImageFromArray(array_to_align)

    registered_image = sitk.Resample(moving_image, transform=dico_homographies[test_index][sl],
                                     interpolator=registration.resampling_interpolator,
                                     outputPixelType=moving_image.GetPixelIDValue())
    registered_array = sitk.GetArrayFromImage(registered_image)

    #animate_images([array_ref, registered_array])
    #animate_images([array_ref, array_to_align])

    print("MI score for unregistered image : {}".format(np.round(calc_MI(array_ref.flatten(),array_to_align.flatten(),bins=100),3)))
    print("MI score for registered image : {}".format(
        np.round(calc_MI(array_ref.flatten(),registered_array.flatten(),bins=100), 3)))



sl = int(nb_slices / 2)
sl=1
sl=31
#sl=30
test_index=0

array_ref = np.abs(dico_mask[index_ref][sl] * dico_volume[index_ref][sl])
fixed_image = sitk.GetImageFromArray(array_ref)

array_to_align = np.abs(dico_mask[test_index][sl] * dico_volume[test_index][sl])
moving_image = sitk.GetImageFromArray(array_to_align)

registered_image = sitk.Resample(moving_image,transform=dico_homographies[test_index][sl],interpolator=registration.resampling_interpolator,outputPixelType=moving_image.GetPixelIDValue())
registered_array = sitk.GetArrayFromImage(registered_image)

animate_images([array_ref,registered_array])
animate_images([array_ref,array_to_align])

animate_images([array_to_align,registered_array])

volumes_corrected_final=np.zeros((ntimesteps,nb_slices,int(npoint/2),int(npoint/2)),dtype="complex64")
ts_indices=np.zeros(len(groups)).astype(int)
count=np.zeros(ntimesteps).astype(int)
#total_weight=np.zeros(ntimesteps).astype(int)

del dico_mask
del dico_volume
#
# dico_kdata_retained_registration = {}
# dico_volumes_corrected = {}
#
# J=0
# g=groups[0]
for j, g in tqdm(enumerate(groups)):
    retained_nav_spokes_index = np.argwhere(g).flatten()
    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
                             axis=-1)

    spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
        nb_segments / nb_gating_spokes / 2) + 1:] - 1  # adjustment for change of partition
    spoke_groups = spoke_groups.flatten()

    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
    included_spokes[::int(nb_segments / nb_gating_spokes)] = False
    print("Filtering KData for movement...")
    kdata_retained_final_list = []
    for i in (range(nb_channels)):
        kdata_retained_final, traj_retained_final, retained_timesteps = correct_mvt_kdata(
            kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, ntimesteps,
            density_adj=True, log=False)
        kdata_retained_final_list.append(kdata_retained_final)

    #dico_kdata_retained_registration[j] = retained_timesteps



    radial_traj_3D_corrected = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor,
                                        npoint=npoint, nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    radial_traj_3D_corrected.traj_for_reconstruction = traj_retained_final

    volumes_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list, radial_traj_3D_corrected,
                                                                  image_size, b1=b1_all_slices,
                                                                  ntimesteps=len(retained_timesteps), density_adj=False,
                                                                  useGPU=False, normalize_kdata=False, memmap_file=None,
                                                                  light_memory_usage=True, is_theta_z_adjusted=True,
                                                                  normalize_volumes=True)

    print("Re-registering corrected volumes")
    for ts in tqdm(range(volumes_corrected.shape[0])):
        for sl in range(volumes_corrected.shape[1]):

            moving_image_real = sitk.GetImageFromArray(volumes_corrected[ts, sl, :, :].real)
            moving_image_imag = sitk.GetImageFromArray(volumes_corrected[ts, sl, :, :].imag)

            volumes_corrected[ts, sl, :, :] = sitk.GetArrayFromImage( sitk.Resample(moving_image_real, transform=dico_homographies[j][sl],
                                             interpolator=registration.resampling_interpolator,
                                             outputPixelType=moving_image_real.GetPixelIDValue())) + 1j * sitk.GetArrayFromImage( sitk.Resample(moving_image_imag, transform=dico_homographies[j][sl],
                                             interpolator=registration.resampling_interpolator,
                                             outputPixelType=moving_image_imag.GetPixelIDValue()))
            #registration.execute(sitk.GetImageFromArray(volumes_corrected[ts, sl, :, :].real),
            #                                                                 dico_homographies[j][sl]))+1j*sitk.GetArrayFromImage(registration.execute(sitk.GetImageFromArray(volumes_corrected[ts, sl, :, :].imag),
            #                                                                 dico_homographies[j][sl]))
    #dico_volumes_corrected[j] = copy(volumes_corrected)
    print("Forming final volumes with contribution from group {}".format(j))
    for ts in tqdm(range(ntimesteps)):
        if ts in retained_timesteps:
            volumes_corrected_final[ts]+=volumes_corrected[ts_indices[j]]*traj_retained_final[ts_indices[j]].shape[0]
            #count[ts]+=1
            count[ts] += traj_retained_final[ts_indices[j]].shape[0]
            ts_indices[j] += 1
            #total_weight[ts]+=traj_retained_final[ts_indices[j]].shape[0]


del volumes_corrected
del radial_traj_3D_corrected
del dico_homographies
del kdata_retained_final_list

count=np.expand_dims(count,axis=tuple(range(1,volumes_corrected_final.ndim)))
volumes_corrected_final/=count

np.save(filename_volume_corrected_final,volumes_corrected_final)
#
# animate_images(volumes_corrected_final[:,int(nb_slices/2),:,:])

######################################################################################""








#
# index_ref = 1
#
# dico_homographies = {}
#
# for index_to_align in dico_volume.keys():
#     dico_homographies[index_to_align] = {}
#     for sl in range(nb_slices):
#
#         # Open the image files.
#         array_to_align = np.abs(dico_mask[index_to_align][sl] * dico_volume[index_to_align][sl])
#         array_ref = np.abs(dico_mask[index_ref][sl] * dico_volume[index_ref][sl])
#
#         array_to_align = array_to_align / (array_to_align).max()
#         array_ref = array_ref / (array_ref).max()
#
#         uint_img_to_align = np.array(array_to_align * 255).astype('uint8')
#         uint_img_ref = np.array(array_ref * 255).astype('uint8')
#
#         # img1_color = np.abs() # Image to be aligned.
#         # img2_color = np.abs(dico_volume[1][int(nb_slices/2)])
#
#         # Convert to grayscale.
#         img1_color = cv2.cvtColor(uint_img_to_align, cv2.COLOR_GRAY2BGR)
#         img2_color = cv2.cvtColor(uint_img_ref, cv2.COLOR_GRAY2BGR)
#
#         img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
#         img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
#
#         # Find size of image1
#         sz = img1.shape
#
#         # Define the motion model
#         warp_mode = cv2.MOTION_TRANSLATION
#
#         # Define 2x3 or 3x3 matrices and initialize the matrix to identity
#         if warp_mode == cv2.MOTION_HOMOGRAPHY:
#             warp_matrix = np.eye(3, 3, dtype=np.float32)
#         else:
#             warp_matrix = np.eye(2, 3, dtype=np.float32)
#
#         # Specify the number of iterations.
#         number_of_iterations = 5000;
#
#         # Specify the threshold of the increment
#         # in the correlation coefficient between two iterations
#         termination_eps = 1e-10;
#
#         # Define termination criteria
#         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
#
#         # Run the ECC algorithm. The results are stored in warp_matrix.
#         (cc, warp_matrix) = cv2.findTransformECC(img2, img1, warp_matrix, warp_mode, criteria)
#
#         # if warp_mode == cv2.MOTION_HOMOGRAPHY :
#         # Use warpPerspective for Homography
#         #    transformed_img = cv2.warpPerspective (img1, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#         # else :
#         # Use warpAffine for Translation, Euclidean and Affine
#         #    transformed_img = cv2.warpAffine(img1, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
#
#         dico_homographies[index_to_align][sl] = warp_matrix
#
# sl = int(nb_slices / 2)
# test_index=0
#
# animate_multiple_images(np.abs(dico_volume[index_ref]),np.abs(dico_volume[test_index]))
#
# registered_volume=[]
# for sl in range(nb_slices):
#     registered_volume.append(scipy.ndimage.affine_transform(np.abs(dico_volume[test_index][sl].T), dico_homographies[test_index][sl]).T)
# registered_volume=np.array(registered_volume)
#
# animate_multiple_images(np.abs(dico_volume[index_ref]),registered_volume)
#
#
# animate_images([np.abs(dico_volume[test_index][sl]),
#                 np.abs(dico_volume[index_ref][sl])])
#
# animate_images([scipy.ndimage.affine_transform(np.abs(dico_volume[test_index][sl].T), dico_homographies[test_index][sl]).T,
#                 np.abs(dico_volume[index_ref][sl])])
#
# volumes_corrected_final=np.zeros((ntimesteps,nb_slices,int(npoint/2),int(npoint/2)),dtype="complex64")
# ts_indices=np.zeros(len(groups)).astype(int)
# count=np.zeros(ntimesteps).astype(int)
# #total_weight=np.zeros(ntimesteps).astype(int)
#
# del dico_mask
# del dico_volume
# #
# # dico_kdata_retained_registration = {}
# # dico_volumes_corrected = {}
# #
# # J=0
# # g=groups[0]
# for j, g in tqdm(enumerate(groups)):
#     retained_nav_spokes_index = np.argwhere(g).flatten()
#     spoke_groups = np.argmin(np.abs(
#         np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
#                                                                           nb_segments / nb_gating_spokes).reshape(1,
#                                                                                                                   -1)),
#                              axis=-1)
#     included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
#     included_spokes[::int(nb_segments / nb_gating_spokes)] = False
#     print("Filtering KData for movement...")
#     kdata_retained_final_list = []
#     for i in (range(nb_channels)):
#         kdata_retained_final, traj_retained_final, retained_timesteps = correct_mvt_kdata(
#             kdata_all_channels_all_slices[i].reshape(nb_segments, -1), radial_traj, included_spokes, ntimesteps,
#             density_adj=True, log=False)
#         kdata_retained_final_list.append(kdata_retained_final)
#
#     #dico_kdata_retained_registration[j] = retained_timesteps
#
#
#
#     radial_traj_3D_corrected = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor,
#                                         npoint=npoint, nb_slices=nb_slices, incoherent=incoherent, mode=mode)
#     radial_traj_3D_corrected.traj_for_reconstruction = traj_retained_final
#
#     volumes_corrected = simulate_radial_undersampled_images_multi(kdata_retained_final_list, radial_traj_3D_corrected,
#                                                                   image_size, b1=b1_all_slices,
#                                                                   ntimesteps=len(retained_timesteps), density_adj=False,
#                                                                   useGPU=False, normalize_kdata=False, memmap_file=None,
#                                                                   light_memory_usage=True, is_theta_z_adjusted=True,
#                                                                   normalize_volumes=True)
#
#     print("Re-registering corrected volumes")
#     for ts in tqdm(range(volumes_corrected.shape[0])):
#         for sl in range(volumes_corrected.shape[1]):
#             volumes_corrected[ts, sl, :, :] = scipy.ndimage.affine_transform(volumes_corrected[ts, sl, :, :].T,
#                                                                              dico_homographies[j][sl]).T
#     #dico_volumes_corrected[j] = copy(volumes_corrected)
#     print("Forming final volumes with contribution from group {}".format(j))
#     for ts in tqdm(range(ntimesteps)):
#         if ts in retained_timesteps:
#             volumes_corrected_final[ts]+=volumes_corrected[ts_indices[j]]*traj_retained_final[ts_indices[j]].shape[0]
#             #count[ts]+=1
#             count[ts] += traj_retained_final[ts_indices[j]].shape[0]
#             ts_indices[j] += 1
#             #total_weight[ts]+=traj_retained_final[ts_indices[j]].shape[0]
#

# del volumes_corrected
# del radial_traj_3D_corrected
# del dico_homographies
# del kdata_retained_final_list

# volumes_corrected_final=np.zeros((ntimesteps,nb_slices,int(npoint/2),int(npoint/2)),dtype=dico_volumes_corrected[0].dtype)
# ts_indices=np.zeros(len(dico_kdata_retained_registration.keys())).astype(int)
# count=np.zeros(ntimesteps).astype(int)


# for ts in tqdm(range(ntimesteps)):
#     count=0
#     for j in dico_kdata_retained_registration.keys():
#         if ts in dico_kdata_retained_registration[j]:
#             volumes_corrected_final[ts]+=dico_volumes_corrected[j][ts_indices[j]]
#             ts_indices[j]+=1
#             count+=1
#     volumes_corrected_final[ts]/=count

# for j in tqdm(dico_kdata_retained_registration.keys()):
#     for ts in tqdm(range(ntimesteps)):
#         if ts in dico_kdata_retained_registration[j]:
#             volumes_corrected_final[ts]+=dico_volumes_corrected[j][ts_indices[j]]
#             ts_indices[j]+=1
#             count[ts]+=1
#     del dico_kdata_retained_registration[j]
#     del dico_volumes_corrected[j]
#
# count=np.expand_dims(count,axis=tuple(range(1,volumes_corrected_final.ndim)))
# volumes_corrected_final/=count
#
# np.save(filename_volume_corrected_final,volumes_corrected_final)
#
# animate_images(volumes_corrected_final[:,int(nb_slices/2),:,:])

#volumes for slice taking into account coil sensi
print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=True,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True)
    np.save(filename_volume,volumes_all)
    # sl=20
    # ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
    del volumes_all

print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    selected_spokes = np.r_[10:400]
    selected_spokes=None
    mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/25, normalize_kdata=True,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask,mask)
    animate_images(mask)
    del mask

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


#dictfile = "mrf175_SimReco2_light.dict"
#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
dictfile = "mrf175_Dico2_Invivo.dict"
#dictfile = "mrf175_Dico2_Invivo_adjusted_TR7500.dict"
#filename_volume_corrected_final = str.split(filename,".dat") [0]+"_volumes_corrected_final_old{}.npy".format("")


mask = np.load(filename_mask)
volumes_all = np.load(filename_volume)
volumes_corrected_final=np.load(filename_volume_corrected_final)

filename_volume_corrected_disp5= "./data/InVivo/3D/20220113_CS/meas_MID00163_FID49558_raFin_3D_tra_1x1x5mm_FULL_50GS_read_volumes_corrected_disp5.npy"
volumes_corrected_disp5 = np.load(filename_volume_corrected_disp5)

center_z = int(nb_slices/2)
center_pixel=int(image_size[1]/2)

dplane = image_size[1]/8
dz = image_size[0]/8

x = int(np.random.normal(center_pixel,dplane))
y = int(np.random.normal(center_pixel,dplane))
z = int(np.random.normal(center_z,dz))



plt.close("all")
metric=np.real
plt.figure()
plt.plot(metric(volumes_all[:,z,x,y]),label="All spokes")
plt.plot(metric(volumes_corrected_final[:,z,x,y]),label="Corrected spokes")
plt.plot(metric(volumes_corrected_disp5[:,z,x,y]),label="Only one cycle")
plt.title("Volume series comparison {}".format((z,x,y)))
plt.legend()




#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])

sl = 10
suffix ="_bestgroup_testsl{}".format(sl)
suffix ="_bestgroup".format(sl)
mask_slice=np.zeros(mask.shape,dtype=mask.dtype)
mask_slice[sl,:,:]=1
mask*=mask_slice

ntimesteps=175
if not(load_map):
    niter = 0
    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=None,split=10,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps)
    all_maps=optimizer.search_patterns_test(dictfile,volumes_corrected_best_group,retained_timesteps=None)

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






























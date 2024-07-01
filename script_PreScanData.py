
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from trajectory import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np

from mutools import io


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
from scipy.spatial.transform import Rotation

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./data/InVivo/3D"

filename="./data/TestMarc/meas_MID00020_FID21818_JAMBE_raFin_CLI_EMPTY_ICE.dat"
#filename="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/phantom.018.v1/meas_MID00135_FID57305_raFin_3D_tra_1x1x5mm_FULL_new_test_prescan_orientation.dat"

#filename="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/patient.003.v13/meas_MID00021_FID42448_raFin_3D_tra_1x1x5mm_FULL_new.dat"
filename="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2020_Diaphragm_Imaging/2-Data/test_data_linux_vivo/meas_MID00156_FID00184_raMedic_3D.dat"
filename="./data/TestPreScan/meas_MID00156_FID00184_raMedic_3D.dat"

twix = twixtools.read_twix(filename,optional_additional_maps=["sWipMemBlock","sKSpace"],optional_additional_arrays=["SliceThickness"])

data_body,data_multicoil,data_noise=getPreScanImages(twix)

#noise correlation
corr_noise=np.real(np.corrcoef(data_noise))
sns.heatmap(corr_noise)

#Processing pre-scan data

data_body_processed=data_body[::2]
data_multicoil_processed=data_multicoil[::2]

data_body_processed=np.moveaxis(data_body_processed,[0,1,2,3],[2,3,0,1])
data_multicoil_processed=np.moveaxis(data_multicoil_processed,[0,1,2,3],[2,3,0,1])

data_body_processed=data_body_processed[::-1,::-1,::-1]
data_multicoil_processed=data_multicoil_processed[::-1,::-1,::-1]

image_body_processed=np.zeros_like(data_body_processed)
image_multicoil_processed=np.zeros_like(data_multicoil_processed)

for ch in range(image_body_processed.shape[-1]):
    image_body_processed[:,:,:,ch]=(sp.fft.fftshift(sp.fft.ifftn((sp.fft.ifftshift(data_body_processed[:,:,:,ch])))))
for ch in range(image_multicoil_processed.shape[-1]):
    image_multicoil_processed[:,:,:,ch]=(sp.fft.fftshift(sp.fft.ifftn((sp.fft.ifftshift(data_multicoil_processed[:,:,:,ch])))))

image_body_processed=np.sum(image_body_processed,axis=-1)


ch=0
for sl in range(image_multicoil_processed.shape[-3]):
    plt.figure()
    plt.imshow(np.abs(image_multicoil_processed[sl,:,:,ch]))











use_GPU = False
light_memory_usage=True
gen_mode="other"

filename_save=str.split(filename,".dat") [0]+".npy"
filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"
filename_save_calib=str.split(filename,".dat") [0]+"_calib.npy"
folder = "/".join(str.split(filename,"/")[:-1])


suffix = ""
filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"

filename_b1 = str.split(filename,".dat") [0]+"_b1.npy"
filename_volume = str.split(filename,".dat") [0]+"_volumes.npy"
filename_kdata = str.split(filename,".dat") [0]+"_kdata.npy"
filename_mask= str.split(filename,".dat") [0]+"_mask.npy"


#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"





use_GPU = False
light_memory_usage=True
#Parsed_File = rT.map_VBVD(filename)
#idx_ok = rT.detect_TwixImg(Parsed_File)
#RawData = Parsed_File[str(idx_ok)]["image"].readImage()

if str.split(filename_seqParams,"/")[-1] not in os.listdir(folder):
    if 'twix' not in locals():
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
    file.close()




meas_sampling_mode=dico_seqParams["alFree"][12]
nb_allspokes=dico_seqParams["alFree"][5]
undersampling_factor=dico_seqParams["US"]
nb_gating_spokes = dico_seqParams["alFree"][4]
nb_slices=dico_seqParams["nb_slices"]

x_FOV = dico_seqParams["x_FOV"]
y_FOV = dico_seqParams["y_FOV"]
z_FOV = dico_seqParams["z_FOV"]

#del dico_seqParams


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
        #k = 0
        for i, mdb in enumerate(mdb_list):
            if mdb.is_image_scan() :
                if not(mdb.mdh[14][9]):
                    mdb_data_shape = mdb.data.shape
                    mdb_dtype = mdb.data.dtype
                    nav_size_initialized = True
                    break

        for i, mdb in enumerate(mdb_list):
            if mdb.is_image_scan():
                if not (mdb.mdh[14][9]):
                    #print(mdb.mdh[14])
                    data.append(mdb)
                else:
                    data_for_nav.append(mdb)
                    #data.append(np.zeros(mdb_data_shape,dtype=mdb_dtype))

                #print("i : {} / k : {} / Line : {} / Part : {}".format(i, k, mdb.cLin, mdb.cPar))
                #k += 1
        data_for_nav = np.array([mdb.data for mdb in data_for_nav])
        data_for_nav = data_for_nav.reshape((int(nb_slices),int(nb_gating_spokes))+data_for_nav.shape[1:])

        if data_for_nav.ndim==3:
            data_for_nav=np.expand_dims(data_for_nav,axis=-2)

        data_for_nav = np.moveaxis(data_for_nav,-2,0)
        np.save(filename_nav_save, data_for_nav)

    data = np.array([mdb.data for mdb in data])
    data = data.reshape((-1, int(nb_allspokes)) + data.shape[1:])
    data = np.moveaxis(data, 2, 0)
    data = np.moveaxis(data, 2, 1)

    del mdb_list

    ##################################################
    try:
        del twix
    except:
        pass

    np.save(filename_save,data)
    #
    ##################################################
    #
    # Parsed_File = rT.map_VBVD(filename)
    # idx_ok = rT.detect_TwixImg(Parsed_File)
    # start_time = time.time()
    # data = Parsed_File[str(idx_ok)]["image"].readImage()
    # elapsed_time = time.time()
    # elapsed_time = elapsed_time - start_time
    #
    # progress_str = "Data read in %f s \n" % round(elapsed_time, 2)
    # print(progress_str)
    #
    # data = np.squeeze(data)
    #
    # if nb_gating_spokes>0:
    #     data = np.moveaxis(data, 0, -1)
    #     data = np.moveaxis(data, -2, 0)
    #
    # else:
    #     data = np.moveaxis(data, 0, -1)

    #np.save(filename_save, data)


else :
    data = np.load(filename_save)
    if nb_gating_spokes>0:
        data_for_nav=np.load(filename_nav_save)


try:
    del twix
except:
    pass

data_shape=data.shape
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
    del data

    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)


# Coil sensi estimation for all slices
print("Calculating Coil Sensitivity....")

if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices=calculate_sensitivity_map_3D(copy(kdata_all_channels_all_slices),radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage)
    np.save(filename_b1,b1_all_slices)
else:
    b1_all_slices=np.load(filename_b1)





sl=int(b1_all_slices.shape[1]/2)
list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
plot_image_grid(list_images,(5,6),title="Sensitivity map for slice {}".format(sl))


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






## Volume positions

# Slice position image
mdb_list = twix[1]['mdb']
for i, mdb in enumerate(mdb_list):
    if mdb.is_image_scan():
        param_rot_image=mdb.mdh[22]
        break

# Slice position prescan
mdb_list = twix[0]['mdb']
for i, mdb in enumerate(mdb_list):
    if mdb.is_image_scan():
        param_rot_prescan=mdb.mdh[22]
        break


r_prescan=Rotation.from_quat(param_rot_prescan[1])
r_image=Rotation.from_quat(param_rot_image[1])

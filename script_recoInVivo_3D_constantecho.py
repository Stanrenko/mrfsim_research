
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


localfile="/phantom.013.v3/meas_MID00236_FID27630_raFin_3D_tra_1x1x5mm_FULL_sel_2_2.dat"
localfile="/phantom.013.v3/meas_MID00237_FID27631_raFin_3D_tra_1x1x5mm_FULL_sel_2_4.dat"
localfile="/phantom.013.v3/meas_MID00238_FID27632_raFin_3D_tra_1x1x5mm_FULL_sel_2_6.dat"
localfile="/phantom.013.v3/meas_MID00239_FID27633_raFin_3D_tra_1x1x5mm_FULL_sel_2_8.dat"
#localfile="/phantom.013.v3/meas_MID00232_FID27626_raFin_3D_tra_1x1x5mm_FULL_sinc_2_2.dat"
#localfile="/phantom.013.v3/meas_MID00233_FID27627_raFin_3D_tra_1x1x5mm_FULL_sinc_2_4.dat"
#localfile="/phantom.013.v3/meas_MID00234_FID27628_raFin_3D_tra_1x1x5mm_FULL_sinc_2_6.dat"
#localfile="/phantom.013.v3/meas_MID00235_FID27629_raFin_3D_tra_1x1x5mm_FULL_sinc_2_8.dat"


filename = base_folder+localfile

filename_save=str.split(filename,".dat") [0]+".npy"
#filename_nav_save=str.split(base_folder+"/phantom.001.v1/phantom.001.v1.dat",".dat") [0]+"_nav.npy"
filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"

folder = "/".join(str.split(filename,"/")[:-1])

suffix="_allspokes8"

filename_b1 = str.split(filename,".dat") [0]+"_b1{}.npy".format("")
filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"

filename_volume = str.split(filename,".dat") [0]+"_volume{}.npy".format("")
filename_kdata = str.split(filename,".dat") [0]+"_kdata{}.npy".format("")
filename_mask= str.split(filename,".dat") [0]+"_mask{}.npy".format("")
#filename_mask='./data/InVivo/3D/patient.001.v1/meas_MID00215_FID60605_raFin_3D_tra_FULl_mask.npy'
#filename_mask='./data/InVivo/3D/patient.003.v3/meas_MID00021_FID13878_raFin_3D_tra_1x1x5mm_FULL_1400_old_full_mask.npy'
#filename_mask='./data/InVivo/3D/patient.003.v4/meas_MID00060_FID14882_raFin_3D_tra_1x1x5mm_FULL_1400_old_mask.npy'
#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"
#filename_mask='./data/InVivo/3D/patient.003.v7/meas_MID00021_FID18400_raFin_3D_tra_1x1x5mm_FULL_optim_reco_395_mask.npy'
#filename_mask='./data/InVivo/3D/patient.003.v7/meas_MID00025_FID18404_raFin_3D_tra_1x1x5mm_FULL_new_mask.npy'
#filename_mask='./data/InVivo/3D/patient.002.v4/meas_MID00165_FID18800_raFin_3D_tra_1x1x5mm_FULL_new_mask.npy'

window=8
density_adj_radial=True
use_GPU = True
light_memory_usage=True
#Parsed_File = rT.map_VBVD(filename)
#idx_ok = rT.detect_TwixImg(Parsed_File)
#RawData = Parsed_File[str(idx_ok)]["image"].readImage()

#filename_seqParams="./data/InVivo/3D/patient.001.v1/meas_MID00215_FID60605_raFin_3D_tra_FULl_seqParams.pkl"

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

try:
    use_navigator_dll=dico_seqParams["use_navigator_dll"]
except:
    use_navigator_dll=False

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
dummy_echos = dico_seqParams["alFree"][5]

ntimesteps=int(nb_segments/window)


x_FOV = dico_seqParams["x_FOV"]
y_FOV = dico_seqParams["y_FOV"]
z_FOV = dico_seqParams["z_FOV"]
#z_FOV=64
nb_part = dico_seqParams["nb_part"]
undersampling_factor = dico_seqParams["alFree"][9]
#undersampling_factor=1

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


#undersampling_factor=4


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
    data = data.reshape((-1,int(nb_segments)) + data.shape[1:])
    data=data[dummy_echos:]
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



nb_channels=data_shape[0]
nb_allspokes = data_shape[-3]
npoint = data_shape[-1]
nb_slices = data_shape[-2]*undersampling_factor
image_size = (nb_slices, int(npoint/2), int(npoint/2))


dx = x_FOV/(npoint/2)
dy = y_FOV/(npoint/2)
dz = z_FOV/nb_slices
#dz=4
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

# undersampling_factor=4
#
# nb_allspokes=kdata_all_channels_all_slices.shape[1]
# nb_channels=kdata_all_channels_all_slices.shape[0]
# npoint=kdata_all_channels_all_slices.shape[-1]
# nb_slices=kdata_all_channels_all_slices.shape[-2]*undersampling_factor
#
# image_size = (nb_slices, int(npoint/2), int(npoint/2))

print("Calculating Coil Sensitivity....")

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
#radial_traj.adjust_traj_for_window(window)

nb_segments=radial_traj.get_traj().shape[0]

if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage,hanning_filter=True)
    np.save(filename_b1,b1_all_slices)
    del kdata_all_channels_all_slices
    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    b1_all_slices=np.load(filename_b1)

sl=int(b1_all_slices.shape[1]/2)
list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))

if nb_channels==1:
    b1_all_slices=np.ones(b1_all_slices.shape)

if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    volume=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True)[0]
    np.save(filename_volume,volume)
else:
    volume=np.load(filename_volume)


animate_images(np.angle(volume[:, :, :]))

from mutools import io
file_mha = filename.split(".dat")[0] + "_volume_allspokes.mha"
io.write(file_mha,np.angle(volume),tags={"spacing":[dz,dx,dy]})

























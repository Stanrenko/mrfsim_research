
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


localfile="/patient.008.v3/meas_MID00437_FID27831_raFin_3D_tra_1x1x5mm_FULL_angio_sel.dat"
#localfile="/patient.008.v3/meas_MID00438_FID27832_raFin_3D_tra_1x1x5mm_FULL_angio_no_sel.dat"
#localfile="/patient.008.v3/meas_MID00439_FID27833_raFin_3D_tra_1x1x5mm_FULL_sinc_sel.dat"
#localfile="/patient.008.v3/meas_MID00440_FID27834_raFin_3D_tra_1x1x5mm_FULL_sinc_no_sel.dat"


dictfile="mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_2_25_reco3.95_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_2_25_reco3.95_w8_simmean.dict"
dictfile="mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_1_79_reco3.95_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_1_79_reco3.95_w8_simmean.dict"




localfile="/patient.003.v9/meas_MID00094_FID28108_raFin_3D_tra_1x1x5mm_FULL_sinc.dat"
localfile="/patient.003.v9/meas_MID00095_FID28109_raFin_3D_tra_1x1x5mm_FULL_sinc_non_sel.dat"
localfile="/patient.003.v9/meas_MID00092_FID28106_raFin_3D_tra_1x1x5mm_FULL_new.dat"
#localfile="/patient.003.v9/meas_MID00093_FID28107_raFin_3D_tra_1x1x5mm_FULL_new_nonsel.dat"

dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_78_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1_78_reco4_w8_simmean.dict"
dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2_23_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2_23_reco4_w8_simmean.dict"
#dictfile="mrf175_Dico2_Invivo_2_23.dict"
#dictfile_light="mrf175_Dico2_Invivo_2_23_light_for_matching.dict"


localfile="/patient.008.v5/meas_MID00117_FID32355_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile="/patient.008.v5/meas_MID00118_FID32356_raFin_3D_tra_1x1x5mm_FULL_bmy.dat"
#localfile="/patient.008.v5/meas_MID00119_FID32357_raFin_2D_tra_1x1x5mm_FULL_bmy.dat"


dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_78_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1_78_reco4_w8_simmean.dict"

dictfile="mrf_dictconf_Dico2_Invivo_new_fat_model_adjusted_1_78_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_new_fat_model_adjusted_1_78_reco4_w8_simmean.dict"

dictfile="mrf_dictconf_Dico2_Invivo_neg_fat_shift_adjusted_1_78_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_neg_fat_shift_adjusted_1_78_reco4_w8_simmean.dict"

#dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_68_reco4_w8_simmean.dict"
#dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1_68_reco4_w8_simmean.dict"

#localfile="/patient.008.v4/meas_MID00148_FID28313_raFin_3D_tra_1x1x5mm_FULL_new.dat"

# dictfile="mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_1_88_reco3.95_w8_simmean.dict"
# dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_1_88_reco3.95_w8_simmean.dict"
# dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_88_reco4_w8_simmean.dict"
# dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1_88_reco4_w8_simmean.dict"

localfile="/patient.003.v10/meas_MID00331_FID33652_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile="/patient.003.v10/meas_MID00332_FID33653_raFin_3D_tra_1x1x5mm_FULL_optim_v2.dat"

dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2_21_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2_21_reco4_w8_simmean.dict"


dictfile="mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v2_2_22_reco3.53_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v2_2_22_reco3.53_w8_simmean.dict"


localfile="/patient.009.v1/meas_MID00084_FID33958_raFin_3D_tra_1x1x5mm_FULL_new.dat"


dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2_21_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2_21_reco4_w8_simmean.dict"



#localfile="/patient.008.v6/meas_MID00021_FID34675_raFin_3D_tra_1x1x5mm_FULL_new_reco4.dat"
# localfile="/patient.008.v6/meas_MID00022_FID34676_raFin_3D_tra_1x1x5mm_FULL_760_reco4.dat"
# localfile="/patient.008.v6/meas_MID00023_FID34677_raFin_3D_tra_1x1x5mm_FULL_760_correl_reco375.dat"
localfile="/patient.008.v6/meas_MID00024_FID34678_raFin_3D_tra_1x1x5mm_FULL_760_v1_reco395.dat"
# localfile="/patient.008.v6/meas_MID00025_FID34679_raFin_3D_tra_1x1x5mm_FULL_760_v2_reco353.dat"

dictfile="mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_2_22_reco3.95_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_2_22_reco3.95_w8_simmean.dict"


# dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2_21_reco4_w8_simmean.dict"
# dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2_21_reco4_w8_simmean.dict"
#
#
#
# localfile="/phantom.014.v3/meas_MID00302_FID35244_raFin_3D_tra_1x1x5mm_FULL_new.dat"
# dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_88_reco4_w8_simmean.dict"
# dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1_88_reco4_w8_simmean.dict"
#
#
# localfile="/phantom.014.v4/meas_MID00024_FID35424_raFin_3D_tra_1x1x5mm_FULL_new.dat"
# dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_88_reco4_w8_simmean.dict"
# dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1_88_reco4_w8_simmean.dict"

#
# dictfile="mrf_dictconf_Dico2_Invivo_adjusted_760_2.22_reco4_w8_simmean.dict"
# dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_760_2.22_reco4_w8_simmean.dict"
# #
# dictfile="mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_correl_2.22_reco3.75_w8_simmean.dict"
# dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_correl_2.22_reco3.75_w8_simmean.dict"
# #
# dictfile="mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_2_22_reco3.95_w8_simmean.dict"
# dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1_2_22_reco3.95_w8_simmean.dict"
# #
# dictfile="mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v2_2_22_reco3.53_w8_simmean.dict"
# dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v2_2_22_reco3.53_w8_simmean.dict"




#localfile="/patient.001.v1/meas_MID00215_FID60605_raFin_3D_tra_FULl.dat"

#/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data Processed/patient.001.v1/meas_MID00215_FID60605_raFin_3D_tra_FULl_us4_kdata.npy

filename = base_folder+localfile

filename_save=str.split(filename,".dat") [0]+".npy"
#filename_nav_save=str.split(base_folder+"/phantom.001.v1/phantom.001.v1.dat",".dat") [0]+"_nav.npy"
filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"

folder = "/".join(str.split(filename,"/")[:-1])

suffix="_allspokes8"

filename_b1 = str.split(filename,".dat") [0]+"_b1{}.npy".format("")
filename_b1_2Dplus1 = str.split(filename,".dat") [0]+"_b12Dplus1{}.npy".format("")
filename_b1_bart = str.split(filename,".dat") [0]+"_b1_bart{}.npy".format("")

filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"

filename_volume = str.split(filename,".dat") [0]+"_volumes{}.npy".format("")
filename_volume_2Dplus1 = str.split(filename,".dat") [0]+"_volumes2Dplus1{}.npy".format("")

filename_kdata = str.split(filename,".dat") [0]+"_kdata{}.npy".format("")
filename_mask= str.split(filename,".dat") [0]+"_mask{}.npy".format("")
filename_mask='./data/InVivo/3D/patient.008.v6/meas_MID00021_FID34675_raFin_3D_tra_1x1x5mm_FULL_new_reco4_mask.npy'
#filename_mask='./data/InVivo/3D/patient.001.v1/meas_MID00215_FID60605_raFin_3D_tra_FULl_mask.npy'
#filename_mask='./data/InVivo/3D/patient.003.v3/meas_MID00021_FID13878_raFin_3D_tra_1x1x5mm_FULL_1400_old_full_mask.npy'
#filename_mask='./data/InVivo/3D/patient.003.v4/meas_MID00060_FID14882_raFin_3D_tra_1x1x5mm_FULL_1400_old_mask.npy'
#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"
#filename_mask='./data/InVivo/3D/patient.003.v7/meas_MID00021_FID18400_raFin_3D_tra_1x1x5mm_FULL_optim_reco_395_mask.npy'
#filename_mask='./data/InVivo/3D/patient.003.v7/meas_MID00025_FID18404_raFin_3D_tra_1x1x5mm_FULL_new_mask.npy'
#filename_mask='./data/InVivo/3D/patient.002.v4/meas_MID00165_FID18800_raFin_3D_tra_1x1x5mm_FULL_new_mask.npy'

return_cost=True

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
undersampling_factor=1

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
        data_for_nav = data_for_nav.reshape((int(nb_part+dummy_echos), int(nb_gating_spokes)) + data_for_nav.shape[1:])

        if data_for_nav.ndim == 3:
            data_for_nav = np.expand_dims(data_for_nav, axis=-2)
        data_for_nav=data_for_nav[dummy_echos:]
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
#image_size = (nb_slices, 550, 550)

#image_size = (nb_slices, int(npoint), int(npoint))

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

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
#radial_traj.adjust_traj_for_window(window)

nb_segments=radial_traj.get_traj().shape[0]

import torch
#help(torch.fft)
#torch.cuda.empty_cache()

radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)

#data_pytorch=torch.tensor(kdata_all_channels_all_slices,device="cpu")

#data_pytorch_zkxky=torch.fft.ifft(data_pytorch,dim=2)

#data_pytorch_zkxky=

data_numpy_zkxky=np.fft.ifft(kdata_all_channels_all_slices,axis=2)
data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
data_numpy_zkxky=np.moveaxis(data_numpy_zkxky,-2,1)

sl=int(nb_slices/2)
data_numpy_zkxky_slice=data_numpy_zkxky[:,sl]
data_numpy_zkxky_slice=data_numpy_zkxky_slice.reshape(nb_channels,ntimesteps,-1)

image_all_channels_sl=[]

traj=radial_traj_2D.get_traj_for_reconstruction(ntimesteps)

for ts in tqdm(range(ntimesteps)):
    t=traj[ts]
    image_all_channels_sl.append(finufft.nufft2d1(t[:, 0], t[:, 1], np.squeeze(data_numpy_zkxky_slice[:, ts]), image_size[1:]))

image_all_channels_sl=np.array(image_all_channels_sl)

ts=np.random.randint(ntimesteps)
plot_image_grid(np.abs(image_all_channels_sl[ts]),nb_row_col=(5,4))

ch=np.random.randint(nb_channels)
animate_images(image_all_channels_sl[:,ch])


data_numpy_zkxky=np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices,axes=2),axis=2),axes=2)

#data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
data_numpy_zkxky=np.moveaxis(data_numpy_zkxky,-2,0)

b1_all_slices_2Dplus1=calculate_sensitivity_map(data_numpy_zkxky,radial_traj_2D,image_size=image_size[1:],hanning_filter=True)
b1_all_slices_2Dplus1=np.moveaxis(b1_all_slices_2Dplus1,1,0)
sl=int(nb_slices/2)
plot_image_grid(np.abs(b1_all_slices_2Dplus1[:,sl]),nb_row_col=(5,4))

import scipy
n_comp=10
data_numpy_zkxky=np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices,axes=2),axis=2),axes=2).astype("complex64")
data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
data_numpy_zkxky=np.moveaxis(data_numpy_zkxky,-2,1)
data_numpy_zkxky_for_pca_all=np.zeros((n_comp,nb_slices,ntimesteps,8,npoint),dtype=data_numpy_zkxky.dtype)

for sl in tqdm(range(nb_slices)):
    data_numpy_zkxky_slice=data_numpy_zkxky[:,sl]
    data_numpy_zkxky_slice=data_numpy_zkxky_slice.reshape(nb_channels,ntimesteps,-1)

    data_numpy_zkxky_for_pca=data_numpy_zkxky_slice.reshape(nb_channels,-1)

    pca=PCAComplex(n_components_=n_comp)

    pca.fit(data_numpy_zkxky_for_pca.T)

    data_numpy_zkxky_for_pca_transformed=pca.transform(data_numpy_zkxky_for_pca.T)
    data_numpy_zkxky_for_pca_transformed=data_numpy_zkxky_for_pca_transformed.T

    data_numpy_zkxky_for_pca_all[:,sl,:,:,:]=data_numpy_zkxky_for_pca_transformed.reshape(10,ntimesteps,-1,npoint)


b1_all_slices_2Dplus1_pca=calculate_sensitivity_map(np.moveaxis(data_numpy_zkxky_for_pca_all,1,0).reshape(nb_slices,n_comp,nb_allspokes,-1),radial_traj_2D,image_size=image_size[1:],hanning_filter=True)
b1_all_slices_2Dplus1_pca=np.moveaxis(b1_all_slices_2Dplus1_pca,1,0)
sl=int(nb_slices/2)
plot_image_grid(np.abs(b1_all_slices_2Dplus1_pca[:,sl]),nb_row_col=(5,4))


maxtimesteps=95

data_numpy_zkxky_pca=data_numpy_zkxky_for_pca_all.reshape(n_comp*nb_slices,ntimesteps,-1)
traj=radial_traj_2D.get_traj_for_reconstruction(ntimesteps)
images_all_channels_all_slices_pca=np.zeros((maxtimesteps,)+image_size,dtype=data_numpy_zkxky_pca.dtype)

traj=traj.astype("float32")

for ts in tqdm(range(maxtimesteps)):
    t = traj[ts]
    fk=finufft.nufft2d1(t[:, 0], t[:, 1], data_numpy_zkxky_pca[:, ts], image_size[1:])
    fk=fk.reshape((n_comp,nb_slices,)+image_size[1:])
    images_all_channels_all_slices_pca[ts]=np.sum(b1_all_slices_2Dplus1_pca.conj() * fk, axis=0)

sl=int(nb_slices/2)
animate_images(images_all_channels_all_slices_pca[:,sl])



maxtimesteps=10

data_numpy_zkxky=np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices,axes=2),axis=2),axes=2).astype("complex64")
data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
data_numpy_zkxky=np.moveaxis(data_numpy_zkxky,-2,1)
data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels*nb_slices,ntimesteps,-1)
traj=radial_traj_2D.get_traj_for_reconstruction(ntimesteps)
images_all_channels_all_slices=np.zeros((maxtimesteps,)+image_size,dtype=data_numpy_zkxky.dtype)

traj=traj.astype("float32")

for ts in tqdm(range(maxtimesteps)):
    t = traj[ts]
    fk=finufft.nufft2d1(t[:, 0], t[:, 1], data_numpy_zkxky[:, ts], image_size[1:])
    fk=fk.reshape((nb_channels,nb_slices,)+image_size[1:])
    images_all_channels_all_slices[ts]=np.sum(b1_all_slices_2Dplus1.conj() * fk, axis=0)

sl=int(nb_slices/2)
animate_images(images_all_channels_all_slices[:,sl])

volumes_all = np.load(filename_volume)
mask = np.load(filename_mask)
signals = volumes_all[:maxtimesteps,mask>0]
signals_2Dplus1=images_all_channels_all_slices[:maxtimesteps,mask>0]
signals_2Dplus1_pca=images_all_channels_all_slices_pca[:maxtimesteps,mask>0]
nb_signals=signals.shape[1]

metric=np.real
num=np.random.randint(nb_signals)

plt.figure()
signal = signals[:,num]
signal_2Dplus1 = signals_2Dplus1[:,num]
signal_2Dplus1_pca = signals_2Dplus1_pca[:,num]

signal/=np.linalg.norm(signal)
signal_2Dplus1/=np.linalg.norm(signal_2Dplus1)
signal_2Dplus1_pca/=np.linalg.norm(signal_2Dplus1_pca)
plt.plot(metric(signal))
plt.plot(metric(signal_2Dplus1))
plt.plot(metric(signal_2Dplus1_pca))

#data_pytorch_zkxky=

# data_numpy_zkxky=np.fft.ifft(kdata_all_channels_all_slices,axis=2)
# data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
# data_numpy_zkxky=np.moveaxis(data_numpy_zkxky,-2,1)
# data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels,nb_slices,ntimesteps,-1)
# image_all_channels_all_slices=np.zeros((nb_channels,nb_slices,ntimesteps,)+image_size[1:],dtype=data_numpy_zkxky.dtype)
#
# for ts in tqdm(range(ntimesteps)):
#     t=traj[ts]
#     for ch in range(nb_channels):
#         for sl in range(nb_slices):
#             image_all_channels_all_slices[ch,sl,ts]=finufft.nufft2d1(t[:, 0], t[:, 1], data_numpy_zkxky[ch,sl, ts], image_size[1:])

#image_all_channels_all_slices=np.array(image_all_channels_all_slices)



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
plot_image_grid(list_images,(5,4),title="Sensitivity map for slice {}".format(sl))

#b1_all_slices=np.load(filename_b1_bart)

if nb_channels==1:
    b1_all_slices=np.ones(b1_all_slices.shape)



# import dask.array as da
# u_dico, s_dico, vh_dico = da.linalg.svd(da.from_array(b1_all_slices[:,int(nb_slices/2)].reshape(nb_channels,-1)))
# s_dico=np.array(s_dico)
# plt.figure();plt.plot(np.cumsum(s_dico)/np.sum(s_dico))
#
# vh_dico=np.array(vh_dico)

#volume_outofphase=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True)[0]
#animate_images(volume_outofphase)

# center_sl=int(nb_slices/2)
# center_point=int(npoint/2)
# res=16
# res_sl=2
# border=16
# border_sl=2
#
#
# mean_signal=np.mean(np.abs(kdata_all_channels_all_slices[0,:,(center_sl-res_sl):(center_sl+res_sl),(center_point-res):(center_point+res)]))
#
# std_signal=np.std(np.abs(kdata_all_channels_all_slices[0,:,:,np.r_[0:border,(npoint-border):npoint]][:,:,np.r_[0:border_sl,(nb_slices-border_sl):nb_slices]]))
#
# mean_signal/std_signal
#
# kdata_reshaped=kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
# mean_signal_by_ts=np.mean(np.abs(kdata_reshaped[0,:,:,(center_sl-res_sl):(center_sl+res_sl),(center_point-res):(center_point+res)]).reshape(ntimesteps,-1),axis=-1)
#
# std_signal_by_ts=np.std(np.abs(kdata_reshaped[0,:,:,:,np.r_[0:border,(npoint-border):npoint]][:,:,:,np.r_[0:border_sl,(nb_slices-border_sl):nb_slices]]),axis=(0,2,3))
#
#plt.figure();plt.plot(mean_signal_by_ts/std_signal_by_ts),plt.title("SNR by timestep")
# #
del kdata_all_channels_all_slices
kdata_all_channels_all_slices=np.load(filename_kdata)
# # #

radial_traj_anatomy=Radial3D(total_nspokes=400,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
radial_traj_anatomy.traj = radial_traj.get_traj()[800:1200]
volume_outofphase=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices[:,800:1200,:,:],radial_traj_anatomy,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True)[0]
# #
animate_images(volume_outofphase,cmap="gray")
#
from mutools import io
file_mha = filename.split(".dat")[0] + "_volume_oop_allspokes.mha"
io.write(file_mha,np.abs(volume_outofphase),tags={"spacing":[dz,dx,dy]})
# #


print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    kdata_all_channels_all_slices=np.load(filename_kdata)
    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True,normalize_iterative=True)
    np.save(filename_volume,volumes_all)
    ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
    #ani = animate_images(volumes_all[:, 5, :, :])
    del volumes_all

print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    #selected_spokes = np.r_[10:648,1099:1400]
    selected_spokes=None
    kdata_all_channels_all_slices = np.load(filename_kdata)
    mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/20, normalize_kdata=False,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask,mask)
    animate_images(mask)
    del mask

# with open("mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400.json","r") as file:
#     sequence_config=json.load(file)

# TE = np.array(sequence_config["TE"])
# unique_TE=np.unique(TE)
# np.argwhere(TE==unique_TE[0])[-1]
# np.argwhere(TE==unique_TE[-1])[0]


del kdata_all_channels_all_slices
#del b1_all_slices



########################## Dict mapping ########################################
seq = None

load_map=False
save_map=True

mask = np.load(filename_mask)
volumes_all = np.load(filename_volume_2Dplus1)



# sl=int(nb_slices/2)
# new_mask=np.zeros(mask.shape)
# new_mask[sl]=mask[sl]
# mask=new_mask

niter = 0

if niter>0:
    b1_all_slices=np.load(filename_b1)
else:
    b1_all_slices=None
return_cost=False
#animate_images(mask)
suffix="_2Dplus1_pca10"
if not(load_map):
    #niter = 0
    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=None,split=100,pca=True,threshold_pca=15,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps,b1=b1_all_slices,threshold_ff=0.9,dictfile_light=dictfile_light,mu=1,mu_TV=1,weights_TV=[1.,0.,0.],return_cost=return_cost)#,mu_TV=1,weights_TV=[1.,0.,0.])
    all_maps=optimizer.search_patterns_test_multi_2_steps_dico(dictfile,volumes_all,retained_timesteps=None)

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
    #file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
    file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format(suffix)
    file = open(file_map, "rb")
    all_maps = pickle.load(file)
    file.close()

# all_maps[0][-1]
# #plt.figure()
# animate_images(makevol(all_maps[0][-1],mask>0))
#
# signals_matched=all_maps[0][0][-1]
# signals_orig=volumes_all[:,mask>0]
#
# j=np.random.randint(signals_orig.shape[1])
# plt.close("all")
# plt.figure()
# metric=np.real
# plt.plot(metric(signals_orig[:,j]),label="orig")
# plt.plot(metric(signals_matched[:,j]),label="matched")
# plt.legend()
#


#file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format("")
curr_file=file_map
file = open(curr_file, "rb")
all_maps = pickle.load(file)
file.close()
for iter in list(all_maps.keys()):

    map_rebuilt=all_maps[iter][0]
    mask=all_maps[iter][1]

    map_rebuilt["wT1"][map_rebuilt["ff"] > 0.7] = 0.0

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

#
# plt.close("all")
#
# sl=55
# plt.figure()
# plt.title("Phase Map")
# plt.imshow(makevol(all_maps[iter][3],mask>0)[sl],cmap="jet",vmin=-1.5,vmax=1.5)
# plt.colorbar()
#
# plt.figure()
# plt.title("Correlation Map")
# plt.imshow(makevol(all_maps[iter][2],mask>0)[sl],cmap="jet")
# plt.colorbar()
#
# plt.figure()
# plt.title("Df Map")
# plt.imshow(map_for_sim["df"][sl],cmap="jet")
# plt.colorbar()














########################## Dict mapping ########################################


seq = None


load_map=False
save_map=True

ntimesteps=175


mask = np.load(filename_mask)
volumes_all = np.load(filename_volume)

#sl=int(nb_slices/2)
#new_mask=np.zeros(mask.shape)
#new_mask[sl]=mask[sl]
#mask=new_mask

dTEs=np.array([-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25])/1000
#dTEs=np.array([-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25])/1000


for dt in dTEs:
    if dt<0:
        dt_label="minus_{}".format(np.abs(dt*1000))
    else:
        dt_label=str(dt*1000)
    #animate_images(mask)
    suffix="_2StepsDico_{}".format(dt_label)

    dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2_23_dTE_{}_reco4_w8_simmean.dict".format(dt_label)
    dictfile_light = "mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2_23_dTE_{}_reco4_w8_simmean.dict".format(dt_label)



    if not(load_map):
        niter = 0
        optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=None,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps,b1=None,mu="Adaptative",threshold_ff=0.9,dictfile_light=dictfile_light,return_cost=return_cost)#,mu_TV=1,weights_TV=[1.,0.,0.])
        all_maps=optimizer.search_patterns_test_multi_2_steps_dico(dictfile,volumes_all,retained_timesteps=None)

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
        #file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
        file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format(suffix)
        file = open(file_map, "rb")
        all_maps = pickle.load(file)
        file.close()

    # all_maps[0][-1]
    # #plt.figure()
    # animate_images(makevol(all_maps[0][-1],mask>0))





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














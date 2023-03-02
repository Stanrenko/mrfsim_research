
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



#localfile="/phantom.009.v1/meas_MID00265_FID05670_raFin_3D_tra_1x1x4mm_FULL_new_fullReco.dat"
#localfile="/phantom.009.v1/meas_MID00266_FID05671_raFin_3D_tra_1x1x4mm_FULL_new_noReco.dat"
#localfile="/phantom.009.v1/meas_MID00267_FID05672_raFin_3D_tra_1x1x4mm_FULL_new_Optimized_noReco.dat"
#localfile="/phantom.009.v1/meas_MID00269_FID05674_raFin_3D_tra_1x1x4mm_FULL_new_noReco_dummy1.dat"
#localfile="/phantom.009.v1/meas_MID00268_FID05673_raFin_3D_tra_1x1x4mm_FULL_new_Optimized_noReco_dummy1.dat"
localfile="/phantom.009.v2/meas_MID00218_FID06324_raFin_3D_tra_1x1x5mm_FULL_fullReco.dat"
localfile="/phantom.009.v2/meas_MID00219_FID06325_raFin_3D_tra_1x1x5mm_FULL_noReco.dat"
#localfile="/phantom.009.v2/meas_MID00220_FID06326_raFin_3D_tra_1x1x5mm_FULL_optimized_correl.dat"
localfile="/phantom.009.v3/meas_MID00362_FID06677_raFin_3D_tra_1x1x5mm_FULL_optim.dat"
#localfile="/phantom.009.v4/meas_MID00148_FID07295_raFin_3D_tra_1x1x5mm_FULL_optim_correl_smooth_shorten.dat"
#localfile="/phantom.009.v4/meas_MID00147_FID07294_raFin_3D_tra_1x1x5mm_FULL_noReco.dat"
#localfile="/phantom.009.v4/meas_MID00146_FID07293_raFin_3D_tra_1x1x5mm_FULL_fullReco.dat"

#localfile="/phantom.009.v3bis/meas_MID00024_FID07402_raFin_3D_tra_1x1x5mm_FULL_new.dat"
#localfile="/phantom.009.v3bis/meas_MID00025_FID07403_raFin_3D_tra_1x1x5mm_FULL_new_2.dat"
#localfile="/phantom.009.v3bis/meas_MID00026_FID07404_raFin_3D_tra_1x1x5mm_FULL_noReco.dat"
#localfile="/phantom.009.v3bis/meas_MID00027_FID07405_raFin_3D_tra_1x1x5mm_FULL_optimCorrel.dat"
localfile="/phantom.009.v3bis/meas_MID00028_FID07406_raFin_3D_tra_1x1x5mm_FULL_optimCorrelShorten.dat"

dictfile = "mrf175_SimReco2_light_BW_540.dict"
dictfile = "mrf175_SimReco2_light_BW_540_noReco.dict"
dictfile="mrf175_SimReco2_light_BW_540_optimized_M0_local_optim_correl_smooth.dict"
#dictfile="mrf175_SimReco2_light_adjusted_M0_local_optim_correl_smooth_shorten.dict"
#dictfile="mrf175_Dico2_Invivo_optimized_M0_local_optim_correl_smooth_shorten.dict"


#dictfile = "mrf175_SimReco2_light_adjusted_noReco.dict"
#dictfile = "mrf175_SimReco2_light_adjusted_M0_local_optim.dict"
#dictfile = "mrf175_SimReco2_light_adjusted_M0_local_optim_correl_smooth_shorten.dict"
#dictfile = "mrf175_SimReco2_light_adjusted_M0_local_optim_correl_smooth.dict"

#dictfile = "mrf175_SimReco2_light_extended_T1_adjusted_noReco.dict"
#dictfile = "mrf175_SimReco2_light_extended_T1.dict"
#dictfile = "mrf175_SimReco2_light_adjusted_M0_T1_filter_DFFFTR_test.dict"
#dictfile = "mrf175_SimReco2_light_adjusted_M0_T1_filter_DFFFTR_test_delay1_6.dict"
#dictfile = "mrf175_SimReco2_light_extended_T1_adjusted_M0_T1_filter_DFFFTR_test_delay1_6.dict"
#dictfile = "mrf175_SimReco2_light_adjusted_M0_local_optim_correl.dict"


#localfile="/patient.003.v3/meas_MID00233_FID07380_raFin_3D_tra_1x1x5mm_FULL_new.dat"
#localfile="/patient.003.v3/meas_MID00234_FID07381_raFin_3D_tra_1x1x5mm_FULL_noReco.dat"
#localfile="/patient.003.v3/meas_MID00235_FID07382_raFin_3D_tra_1x1x5mm_FULL_correl_optim_smooth_540.dat"
#localfile="/patient.003.v3/meas_MID00236_FID07383_raFin_3D_tra_1x1x5mm_FULL_correl_optim_smooth_780.dat"
#localfile="/patient.003.v3/meas_MID00237_FID07384_raFin_3D_tra_1x1x5mm_FULL_correl_optim_smooth_shorten_780.dat"

#dictfile = "mrf175_Dico2_Invivo_BW_540.dict"
#dictfile = "mrf175_Dico2_Invivo_adjusted_Reco3130.dict"

#dictfile = "mrf175_Dico2_Invivo_BW_540_noReco.dict"
#dictfile = "mrf175_Dico2_Invivo_BW_540_optimized_M0_local_optim_correl_smooth.dict"
#dictfile = "mrf175_Dico2_Invivo_optimized_M0_local_optim_correl_smooth.dict"
#dictfile = "mrf175_Dico2_Invivo_optimized_M0_local_optim_correl_smooth_shorten.dict"


localfile="/phantom.009.v5/meas_MID00431_FID08945_raFin_3D_tra_1x1x5mm_FULL_new_optim_760_reco3.dat"
localfile="/phantom.009.v5/meas_MID00432_FID08946_raFin_3D_tra_1x1x5mm_FULL_new_760_reco4.dat"
localfile="/phantom.009.v5/meas_MID00433_FID08947_raFin_3D_tra_1x1x5mm_FULL_new_840_reco3.dat"
localfile="/phantom.009.v5/meas_MID00434_FID08948_raFin_3D_tra_1x1x5mm_FULL_new_1000_reco3.dat"
localfile="/phantom.009.v5/meas_MID00435_FID08949_raFin_3D_tra_1x1x5mm_FULL_new_1400_reco4.dat"
localfile="/phantom.009.v5/meas_MID00427_FID08941_raFin_3D_tra_1x1x5mm_FULL_orig.dat"

dictfile="mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_reco3.dict"
dictfile="mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_reco4.dict"
dictfile="mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp840_reco3.dict"
dictfile="mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1000_reco3.dict"
dictfile="mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_reco4.dict"
dictfile="mrf175_SimReco2_light_delay_1_94.dict"


localfile="/phantom.009.v6/meas_MID00356_FID09594_raFin_3D_tra_1x1x5mm_FULL_new.dat"
#localfile="/phantom.009.v6/meas_MID00357_FID09595_raFin_3D_tra_1x1x5mm_FULL_760_DE_FF_reco3.dat"
localfile="/phantom.009.v6/meas_MID00359_FID09596_raFin_3D_tra_1x1x5mm_FULL_760_DE_FF_reco3_8.dat"
#localfile="/phantom.009.v6/meas_MID00360_FID09597_raFin_3D_tra_1x1x5mm_FULL_1400_DE_FF_reco4.dat"

dictfile="mrf175_SimReco2_light_delay_1_94.dict"
#dictfile="mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_reco3.dict"
dictfile="mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_reco3.8.dict"
#dictfile="mrf_dictconf_SimReco2_light_adjusted_optimized_DE_Simu_FF_reco4.dict"


#localfile="/patient.003.v2/meas_MID00033_FID09694_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile="/patient.003.v2/meas_MID00034_FID09695_raFin_3D_tra_1x1x5mm_FULL_DE_reco3.dat"
#localfile="/patient.003.v2/meas_MID00035_FID09696_raFin_3D_tra_1x1x5mm_FULL_DE_reco4.dat"


#localfile="/patient.005.v1/meas_MID00189_FID11881_raFin_3D_tra_1x1x5mm_FULL_new.dat"
#localfile="/patient.005.v1/meas_MID00190_FID11882_raFin_3D_tra_1x1x5mm_FULL_DE_reco_3.dat"
#localfile="/patient.005.v1/meas_MID00191_FID11883_raFin_3D_tra_1x1x5mm_FULL_DE_reco_3_v2.dat"



localfile="/patient.003.v3/meas_MID00021_FID13878_raFin_3D_tra_1x1x5mm_FULL_1400_old_full.dat"
#localfile="/patient.003.v3/meas_MID00022_FID13879_raFin_3D_tra_1x1x5mm_US4_1400_old.dat"
#localfile="/patient.003.v3/meas_MID00023_FID13880_raFin_3D_tra_1x1x5mm_FULL_760_DE_reco3.dat"
#localfile="/patient.003.v3/meas_MID00024_FID13881_raFin_3D_tra_1x1x5mm_FULL_760_old_reco3.dat"


dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_87_reco4.dict"
dictfile="mrf_dictconf_Dico2_Invivo_lightDFB1_adjusted_1_87_reco4_w8_simmean.dict"
#dictfile="mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_1_87_reco3.dict"
#dictfile="mrf_dictconf_Dico2_Invivo_lightDFB1_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_1_87_reco3.dict"
#dictfile="mrf_dictconf_Dico2_Invivo_adjusted_760_reco3_w8_simmean.dict"
#dictfile="mrf_dictconf_Dico2_Invivo_lightDFB1_adjusted_760_reco3_w8_simmean.dict"


#mrf_dictconf_Dico2_Invivo_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v2_1_87_reco3
#mrf_dictconf_Dico2_Invivo_lightDFB1_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v2_1_87_reco3


localfile="/patient.003.v4/meas_MID00060_FID14882_raFin_3D_tra_1x1x5mm_FULL_1400_old.dat"
#localfile="/patient.003.v4/meas_MID00061_FID14883_raFin_3D_tra_1x1x5mm_FULL_760_DE_reco3.dat"
#localfile="/patient.003.v4/meas_MID00062_FID14884_raFin_3D_tra_1x1x5mm_FULL_760_random_v4_reco3_9.dat"
#localfile="/patient.003.v4/meas_MID00063_FID14885_raFin_3D_tra_1x1x5mm_FULL_760_random_v5_reco4.dat"

dictfile="mrf_dictconf_Dico2_Invivo_lightDFB1_adjusted_2_25_reco4_w8_simmean.dict"
#dictfile="mrf_dictconf_Dico2_Invivo_lightDFB1_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_2_25_reco3_w8_simmean.dict"
#dictfile="mrf_dictconf_Dico2_Invivo_lightDFB1_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v4_2_25_reco3.9_w8_simmean.dict"
#dictfile="mrf_dictconf_Dico2_Invivo_lightDFB1_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v5_2_25_reco4_w8_simmean.dict"


localfile="/patient.008.v5/meas_MID00117_FID32355_raFin_3D_tra_1x1x5mm_FULL_new.dat"
#localfile="/patient.008.v5/meas_MID00118_FID32356_raFin_3D_tra_1x1x5mm_FULL_bmy.dat"
#localfile="/patient.008.v5/meas_MID00119_FID32357_raFin_2D_tra_1x1x5mm_FULL_bmy.dat"


dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_78_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1_78_reco4_w8_simmean.dict"



undersampling_factor=2

filename = base_folder+localfile

filename_save=str.split(filename,".dat") [0]+".npy"
#filename_nav_save=str.split(base_folder+"/phantom.001.v1/phantom.001.v1.dat",".dat") [0]+"_nav.npy"
filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"

folder = "/".join(str.split(filename,"/")[:-1])

suffix="_allspokes8"

filename_b1 = str.split(filename,".dat") [0]+"_us{}_b1{}.npy".format(undersampling_factor,"")
filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"

filename_volume = str.split(filename,".dat") [0]+"_us{}_volumes{}.npy".format(undersampling_factor,"")
filename_kdata = str.split(filename,".dat") [0]+"_us{}_kdata{}.npy".format(undersampling_factor,"")
filename_mask= str.split(filename,".dat") [0]+"_mask{}.npy".format("")
#filename_mask='./data/InVivo/3D/patient.003.v4/meas_MID00060_FID14882_raFin_3D_tra_1x1x5mm_FULL_1400_old_mask.npy'

#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"


window=8
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
nb_part = dico_seqParams["nb_part"]
#undersampling_factor = dico_seqParams["alFree"][9]

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
nb_slices = data_shape[-2]
image_size = (nb_slices, int(npoint/2), int(npoint/2))


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
    #np.save(filename_kdata, data)
    #del data
    data = data.reshape(nb_channels, ntimesteps, -1, nb_slices,npoint)
    kdata_group = np.zeros((nb_channels, ntimesteps, int(nb_allspokes / ntimesteps), int(nb_slices / undersampling_factor), npoint),dtype=data.dtype)

    shift = 0

    for ts in range(ntimesteps):
        kdata_group[:, ts, :, :, :] = data[:, ts, :, shift::undersampling_factor, :]
        shift += 1
        shift = shift % (undersampling_factor)

    kdata_group = kdata_group.reshape(nb_channels, nb_allspokes, int(nb_slices / undersampling_factor), npoint)
    np.save(filename_kdata,kdata_group)
    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)



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
#
# del kdata_all_channels_all_slices
# kdata_all_channels_all_slices=np.load(filename_kdata)
# # #
# radial_traj_anatomy=Radial3D(total_nspokes=400,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
# radial_traj_anatomy.traj = radial_traj.get_traj()[800:1200]
# volume_outofphase=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True)[0]
# #
# animate_images(volume_outofphase)
#
# from mutools import io
# file_mha = filename.split(".dat")[0] + "_volume_oop_allspokes.mha"
# io.write(file_mha,np.abs(volume_outofphase),tags={"spacing":[dz,dx,dy]})



print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    kdata_all_channels_all_slices=np.load(filename_kdata)
    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=True,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True,normalize_iterative=True)
    np.save(filename_volume,volumes_all)
    # sl=20
    # ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
    del volumes_all

print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    #selected_spokes = np.r_[10:648,1099:1400]
    selected_spokes=None
    kdata_all_channels_all_slices = np.load(filename_kdata)
    mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/15, normalize_kdata=False,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask,mask)
    animate_images(mask)
    del mask

#with open("mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400.json","r") as file:
#    sequence_config=json.load(file)

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
volumes_all = np.load(filename_volume)

#sl=int(nb_slices/2)
#new_mask=np.zeros(mask.shape)
#new_mask[sl]=mask[sl]
#mask=new_mask

#animate_images(mask)
suffix="_2StepsDico_us{}".format(undersampling_factor)
if not(load_map):
    niter = 5
    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=100, pca=True,
                                 threshold_pca=20, log=False, useGPU_dictsearch=True, useGPU_simulation=False,
                                 gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                 b1=b1_all_slices, mu="Adaptative", weights_TV=[1, 0.2, 0.2], mu_TV=1,
                                 threshold_ff=0.9, dictfile_light=dictfile_light)
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
    file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
    file = open(file_map, "rb")
    all_maps = pickle.load(file)


return_cost=niter>0

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































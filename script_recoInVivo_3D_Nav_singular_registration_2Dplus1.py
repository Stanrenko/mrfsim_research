import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.u<se("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,BruteDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./data/InVivo/3D"


import twixtools

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

localfile="/phantom.003.v1/meas_MID00420_FID60810_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile="/phantom.003.v2/meas_MID00036_FID61109_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile="/phantom.003.v2/meas_MID00037_FID61110_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile="/phantom.003.v2/meas_MID00038_FID61111_raFin_3D_tra_1x1x5mm_FULL_new.dat"

#localfile = "/20220113_CS/meas_MID00164_FID49559_raFin_3D_tra_1x1x5mm_FULL_50GS_slice.dat"
#localfile = "/20220118_BM/meas_MID00151_FID49924_raFin_3D_tra_1x1x5mm_FULL_read_nav.dat"

#localfile="/phantom.001.v1/phantom.001.v1.dat"
#localfile="/phantom.001.v1/meas_MID00030_FID51057_raFin_3D_phantom_mvt_0"
localfile="/phantom.004.v1/meas_MID00172_FID62149_raFin_3D_tra_1x1x5mm_FULL_new.dat"#Box at the bottom border of bottle
localfile="/phantom.004.v1/meas_MID00173_FID62150_raFin_3D_tra_1x1x5mm_FULL_new.dat"#Box in the middle of bottle
localfile="/phantom.004.v1/meas_MID00174_FID62151_raFin_3D_tra_1x1x5mm_FULL_new.dat"#Box outside of bottle on top
localfile="/phantom.004.v1/meas_MID00177_FID62154_raFin_3D_tra_1x1x5mm_FULL_bottom.dat"#Box at the bottom border with more outside
localfile="/phantom.004.v1/meas_MID00178_FID62155_raFin_3D_tra_1x1x5mm_FULL_top.dat"#Box at the top border with more outside
localfile="/phantom.004.v2/meas_MID00244_FID62394_raFin_3D_tra_1x1x5mm_FULL_SliceSel500.dat"#Box at the top border with more outside
localfile="/phantom.004.v2/meas_MID00245_FID62395_raFin_3D_tra_1x1x5mm_FULL_SliceSel60.dat"#Box at the top border with more outside
localfile="/phantom.004.v2/meas_MID00247_FID62397_raFin_3D_tra_1x1x5mm_FULL_SliceSel180.dat"#Box at the top border with more outside
#localfile="/phantom.004.v2/meas_MID00248_FID62398_raFin_3D_tra_1x1x5mm_FULL_SliceSel90.dat"#Box at the top border with more outside
localfile="/phantom.004.v2/meas_MID00291_FID62683_raFin_3D_tra_1x1x5mm_FULL_Sl300RO100.dat"#Box at the top border with more outside
localfile="/phantom.004.v2/meas_MID00292_FID62684_raFin_3D_tra_1x1x5mm_FULL_Sl90RO100.dat"#Box at the top border with more outside
localfile="/phantom.004.v2/meas_MID00293_FID62685_raFin_3D_tra_1x1x5mm_FULL_Sl30RO100.dat"#Box at the top border with more outside
#localfile="/phantom.004.v2/meas_MID00294_FID62686_raFin_3D_tra_1x1x5mm_FULL_Sl30RO100_Out.dat"#Box at the top border with more outside

localfile="/phantom.005.v1/meas_MID00448_FID62840_raFin_3D_tra_1x1x5mm_FULL_P0_Sl400_RO50.dat"#Box at the top border with more outside
localfile="/phantom.005.v1/meas_MID00449_FID62841_raFin_3D_tra_1x1x5mm_FULL_P0_Sl30_RO50.dat"#Box at the top border with more outside
localfile="/phantom.005.v1/meas_MID00450_FID62842_raFin_3D_tra_1x1x5mm_FULL_P0_Sl30_RO50_Top.dat"#Box at the top border with more outside
localfile="/phantom.005.v1/meas_MID00451_FID62843_raFin_3D_tra_1x1x5mm_FULL_P0_Sl30_RO50_Bottom.dat"#Box at the top border with more outside
localfile="/phantom.005.v1/meas_MID00452_FID62844_raFin_3D_tra_1x1x5mm_FULL_P0_Sl30_RO50_Bottom_SlOut.dat"#Box at the top border with more outside
localfile="/phantom.005.v1/meas_MID00453_FID62845_raFin_3D_tra_1x1x5mm_FULL_P0_Sl400_RO50_Bottom_SlOut.dat"#Box at the top border with more outside
localfile="/phantom.005.v1/meas_MID00454_FID62846_raFin_3D_tra_1x1x5mm_FULL_P0_Sl30_RO50_BottomOut.dat"#Box at the top border with more outside
localfile="/phantom.005.v1/meas_MID00458_FID62850_raFin_3D_tra_1x1x5mm_FULL_P0_Sl200_RO50_FOV220.dat"#Box at the top border with more outside
localfile="/phantom.005.v1/meas_MID00459_FID62851_raFin_3D_tra_1x1x5mm_FULL_P0_Sl27_RO50_FOV220.dat"#Box at the top border with more outside

localfile="/phantom.001.v1/phantom.001.v1.dat"
#localfile="/phantom.001.v1/meas_MID00030_FID51057_raFin_3D_phantom_mvt_0"
#localfile="/phantom.006.v1/meas_MID00027_FID02798_raFin_3D_tra_1x1x5mm_FULL_FF.dat"#Box at the top border with more outside
#localfile="/phantom.006.v1/meas_MID00028_FID02799_raFin_3D_tra_1x1x5mm_FULL_new.dat"#Box at the top border with more outside
#localfile="/phantom.006.v1/meas_MID00029_FID02800_raFin_3D_tra_1x1x5mm_FULL_FF_TR4000.dat"#Box at the top border with more outside
#localfile="/phantom.006.v1/meas_MID00023_FID02830_raFin_3D_tra_1x1x5mm_FULL_FF_TR5000.dat"#Box at the top border with more outside
#localfile="/phantom.006.v2/"
localfile="/20210113/meas_MID00163_FID49558_raFin_3D_tra_1x1x5mm_FULL_50GS_read.dat"

localfile="/patient.002.v5/meas_MID00021_FID34064_raFin_3D_tra_1x1x5mm_FULL_new.dat"

localfile="/patient.003.v12/meas_MID00020_FID36427_raFin_3D_tra_1x1x5mm_FULL_new.dat"
dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2_28_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2_28_reco4_w8_simmean.dict"



localfile="/patient.008.v7/meas_MID00020_FID37032_raFin_3D_tra_1x1x5mm_FULL_new.dat"
dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2.26_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.26_reco4_w8_simmean.dict"


localfile="/patient.003.v13/meas_MID00021_FID42448_raFin_3D_tra_1x1x5mm_FULL_new.dat"
dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2.33_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.33_reco4_w8_simmean.dict"


filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

filename_save=str.split(filename,".dat") [0]+".npy"
#filename_nav_save=str.split(base_folder+"/phantom.001.v1/phantom.001.v1.dat",".dat") [0]+"_nav.npy"
filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"

folder = "/".join(str.split(filename,"/")[:-1])



suffix=""
low_freq_encode_corrected_perc=None
if low_freq_encode_corrected_perc is not None:
    suffix+="_{}".format("_".join(str.split(str(low_freq_encode_corrected_perc),".")))

filename_b1 = str.split(filename,".dat") [0]+"_b1{}.npy".format("")

filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"

filename_volume = str.split(filename,".dat") [0]+"_volumes{}.npy".format(suffix)
filename_volume_corrected = str.split(filename,".dat") [0]+"_volumes_corrected{}.npy".format(suffix)
filename_kdata = str.split(filename,".dat") [0]+"_kdata{}.npy".format("")
filename_mask= str.split(filename,".dat") [0]+"_mask{}.npy".format("")
filename_oop=str.split(filename,".dat") [0]+"_volumes_oop{}.npy".format(suffix)
filename_oop_corrected=str.split(filename,".dat") [0]+"_volumes_oop_corrected{}.npy".format(suffix)
filename_displacement_pt = str.split(filename,".dat") [0]+"_displacement_pt{}.npy".format("")
filename_displacement = str.split(filename,".dat") [0]+"_displacement{}.npy".format("")

filename_kdata_pt_corr = str.split(filename,".dat") [0]+"_kdata_no_dens_adj_pt_corr{}.npy".format("")


#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"


pt=True
density_adj_radial=True
use_GPU = False
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
                    data.append(mdb)
                else:
                    data_for_nav.append(mdb)
                    data.append(np.zeros(mdb_data_shape,dtype=mdb_dtype))

                #print("i : {} / k : {} / Line : {} / Part : {}".format(i, k, mdb.cLin, mdb.cPar))
                #k += 1
        data_for_nav = np.array([mdb.data for mdb in data_for_nav])
        data_for_nav = data_for_nav.reshape((int(nb_part),int(nb_gating_spokes))+data_for_nav.shape[1:])

        if data_for_nav.ndim==3:
            data_for_nav=np.expand_dims(data_for_nav,axis=-2)

        data_for_nav = np.moveaxis(data_for_nav,-2,0)
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
#
# if str.split(filename_save,"/")[-1] not in os.listdir(folder):
#     Parsed_File = rT.map_VBVD(filename)
#     idx_ok = rT.detect_TwixImg(Parsed_File)
#     start_time = time.time()
#     RawData = Parsed_File[str(idx_ok)]["image"].readImage()
#     #test=Parsed_File["0"]["noise"].readImage()
#     #test = np.squeeze(test)
#
#     elapsed_time = time.time()
#     elapsed_time = elapsed_time - start_time
#     progress_str = "Data read in %f s \n" % round(elapsed_time, 2)
#     print(progress_str)
#     ## Random map simulation
#
#     data = np.squeeze(RawData)
#     data = np.moveaxis(data, 0, -1)
#
#     np.save(filename_save,data)
#
# else :
#     data = np.load(filename_save)

#data = np.moveaxis(data, 0, -1)
# data=np.moveaxis(data,-2,-1)

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

#nb_channels=data_for_nav.shape[0]
#nb_allspokes = 1400
#npoint_nav = data_for_nav.shape[-1]
#nb_slices = data_for_nav.shape[1]


image_size = (nb_slices, int(npoint/2), int(npoint/2))
undersampling_factor=1

if nb_gating_spokes>0:
    npoint_nav=data_for_nav.shape[-1]



dx = x_FOV/(npoint/2)
dy = y_FOV/(npoint/2)
dz = z_FOV/nb_slices

#file_name_nav_mat=str.split(filename,".dat") [0]+"_nav.mat"
#savemat(file_name_nav_mat,{"Kdata":data_for_nav})

if str.split(filename_kdata,"/")[-1] in os.listdir(folder):
    del data



if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    # Density adjustment all slices

    if pt:
        data=np.load(filename_kdata_pt_corr)

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

kdata_shape=kdata_all_channels_all_slices.shape
#kdata_all_channels_all_slices=np.array(groupby(kdata_all_channels_all_slices,window,axis=1))
#ntimesteps=kdata_all_channels_all_slices.shape[0]
ntimesteps=int(nb_allspokes/window)
#kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(nb_channels,-1,nb_slices,npoint)

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


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
#radial_traj.adjust_traj_for_window(window)

nb_segments=radial_traj.get_traj().shape[0]


radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)

import scipy as sp

# data_numpy_zkxky=np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices,axes=2),axis=2,workers=24),axes=2).astype("complex64")
# data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
# data_numpy_zkxky=np.moveaxis(data_numpy_zkxky,-2,1)

#sl=int(nb_slices/2)
#data_numpy_zkxky_slice=data_numpy_zkxky[:,sl]
#data_numpy_zkxky_slice=data_numpy_zkxky_slice.reshape(nb_channels,ntimesteps,-1)

#=data_numpy_zkxky_slice.reshape(nb_channels,-1)

#pca=PCAComplex(n_components_=None)

#pca.fit(data_numpy_zkxky_for_pca.T)

# from copy import deepcopy
# n_comp=16
# data_numpy_zkxky_for_pca_all=np.zeros((n_comp,nb_slices,ntimesteps,8,npoint),dtype=data_numpy_zkxky.dtype)
# pca_dict={}
# for sl in tqdm(range(nb_slices)):
#     data_numpy_zkxky_slice=data_numpy_zkxky[:,sl]
#     data_numpy_zkxky_slice=data_numpy_zkxky_slice.reshape(nb_channels,ntimesteps,-1)
#
#     data_numpy_zkxky_for_pca=data_numpy_zkxky_slice.reshape(nb_channels,-1)
#
#     pca=PCAComplex(n_components_=n_comp)
#
#     pca.fit(data_numpy_zkxky_for_pca.T)
#
#     pca_dict[sl]=deepcopy(pca)
#
#     data_numpy_zkxky_for_pca_transformed=pca.transform(data_numpy_zkxky_for_pca.T)
#     data_numpy_zkxky_for_pca_transformed=data_numpy_zkxky_for_pca_transformed.T
#
#     data_numpy_zkxky_for_pca_all[:,sl,:,:,:]=data_numpy_zkxky_for_pca_transformed.reshape(n_comp,ntimesteps,-1,npoint)
#

# filename_kdata2Dplus1_pca  = str.split(filename,".dat") [0]+"_kdata2Dplus1_pca{}.npy".format(n_comp)
#
n_comp=16
filename_virtualcoils= str.split(filename,".dat") [0]+"_virtualcoils_{}.pkl".format(n_comp)
# import pickle
# with open(filename_virtualcoils,"wb") as file:
#     pickle.dump(pca_dict,file)
# np.save(filename_kdata2Dplus1_pca,data_numpy_zkxky_for_pca_all)
#
# filename_kdata2Dplus1= str.split(filename,".dat") [0]+"_kdata2Dplus1{}.npy".format("")
# np.save(filename_kdata2Dplus1,data_numpy_zkxky)
#
#
import pickle
with open(filename_virtualcoils,"rb") as file:
    pca_dict=pickle.load(file)

try:
    del kdata_all_channels_all_slices
except:
    pass
n_comp=16
filename_kdata2Dplus1_pca  = str.split(filename,".dat") [0]+"_kdata2Dplus1_pca{}.npy".format(n_comp)
data_numpy_zkxky_for_pca_all=np.load(filename_kdata2Dplus1_pca)


b1_all_slices_2Dplus1_pca=calculate_sensitivity_map(np.moveaxis(data_numpy_zkxky_for_pca_all,1,0).reshape(nb_slices,n_comp,nb_allspokes,-1),radial_traj_2D,image_size=image_size[1:],hanning_filter=True)
b1_all_slices_2Dplus1_pca=np.moveaxis(b1_all_slices_2Dplus1_pca,1,0)
sl=int(nb_slices/2)
plot_image_grid(np.abs(b1_all_slices_2Dplus1_pca[:,sl]),nb_row_col=(6,6))
#
#
# filename_b12Dplus1 = str.split(filename,".dat") [0]+"_b12Dplus1_{}.npy".format(n_comp)
# np.save(filename_b12Dplus1,b1_all_slices_2Dplus1_pca)
#
#
# filename_b12Dplus1 = str.split(filename,".dat") [0]+"_b12Dplus1_{}.npy".format(n_comp)
# b1_all_slices_2Dplus1_pca=np.load(filename_b12Dplus1)
#
# b1_all_slices=np.load(filename_b1)
#
# filename_kdata2Dplus1= str.split(filename,".dat") [0]+"_kdata2Dplus1{}.npy".format("")
# data_numpy_zkxky_for_pca_all=np.load(filename_kdata2Dplus1)
#
# maxtimesteps=ntimesteps
# data_numpy_zkxky_pca=data_numpy_zkxky_for_pca_all.reshape(n_comp*nb_slices,ntimesteps,-1)
# traj=radial_traj_2D.get_traj_for_reconstruction(ntimesteps)
# images_all_channels_all_slices_pca=np.zeros((maxtimesteps,)+image_size,dtype=data_numpy_zkxky_pca.dtype)
#
# traj=traj.astype("float32")
#
# for ts in tqdm(range(maxtimesteps)):
#     t = traj[ts]
#     fk=finufft.nufft2d1(t[:, 0], t[:, 1], data_numpy_zkxky_pca[:, ts], image_size[1:])
#     fk=fk.reshape((n_comp,nb_slices,)+image_size[1:])
#     images_all_channels_all_slices_pca[ts]=np.sum(b1_all_slices_2Dplus1_pca.conj() * fk, axis=0)
#
#
#
# #sl=int(nb_slices/2)
# #animate_images(images_all_channels_all_slices_pca[:,sl])
#
# filename_volume2Dplus1 = str.split(filename,".dat") [0]+"_volumes2Dplus1_{}.npy".format(n_comp)
# #filename_volume2Dplus1 = str.split(filename,".dat") [0]+"_volumes2Dplus1.npy"
# np.save(filename_volume2Dplus1,images_all_channels_all_slices_pca)
#
#
#
# n_comp=8
# filename_volume2Dplus1 = str.split(filename,".dat") [0]+"_volumes2Dplus1_{}.npy".format(n_comp)
# volumes_2Dplus1=images_all_channels_all_slices_pca
#
# volumes_all=np.load(filename_volume)
#
#
#
# ts=np.random.randint(ntimesteps)
# sl=np.random.randint(nb_slices)
#
#
# plt.close("all")
# fig,ax=plt.subplots(1,3)
# metric=np.real
# im0=ax[0].imshow(metric(volumes_all[ts,sl]/np.linalg.norm(volumes_all[ts,sl])))
# fig.colorbar(im0,ax=ax[0])
# im1=ax[1].imshow(metric(volumes_2Dplus1[ts,sl]/np.linalg.norm(volumes_2Dplus1[ts,sl])))
# fig.colorbar(im1,ax=ax[1])
# im2=ax[2].imshow(metric(volumes_2Dplus1[ts,sl]/np.linalg.norm(volumes_2Dplus1[ts,sl]))-metric(volumes_all[ts,sl]/np.linalg.norm(volumes_all[ts,sl])))
# fig.colorbar(im2,ax=ax[2])
#
#
#
#
# mask=np.load(filename_mask)
#
# signals=volumes_all[:,mask>0]
# signals_2Dplus1=volumes_2Dplus1[:,mask>0]
#
#
#
# num=np.random.randint(int(mask.sum()))
# signal=signals[:,num]
# signal_2Dplus1=signals_2Dplus1[:,num]
#
# signal/=np.linalg.norm(signal)
# signal_2Dplus1/=np.linalg.norm(signal_2Dplus1)
#
#
# plt.figure()
# plt.plot(signal,label="3D")
# plt.plot(signal_2Dplus1,label="2Dplus1")
# plt.legend()
#
#
#
# if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
#     kdata_all_channels_all_slices=np.load(filename_kdata)
#     res = 16
#     b1_all_slices=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage,hanning_filter=True)
#     np.save(filename_b1,b1_all_slices)
# else:
#     b1_all_slices=np.load(filename_b1)
#
#
# sl=int(b1_all_slices.shape[1]/2)+10
# list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
# plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))
#
# print("Building Volumes....")
# if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
#     kdata_all_channels_all_slices=np.load(filename_kdata)
#     volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_iterative=True)
#     np.save(filename_volume,volumes_all)
#     # sl=20
#     ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
#     del volumes_all
#
#
# if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
#      selected_spokes = np.r_[10:400]
#      kdata_all_channels_all_slices=np.load(filename_kdata)
#      selected_spokes=None
#      mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/25, normalize_kdata=False,light_memory_usage=True,selected_spokes=selected_spokes)
#      np.save(filename_mask,mask)
#      animate_images(mask)
#      del mask



#
# filename_kdata="test_constant_fullsquare_kdata.npy"
#
#
# if str.split(filename_kdata,"/")[-1] not in os.listdir(os.curdir):
#
#     with open("mrf_sequence.json") as f:
#         sequence_config = json.load(f)
#
#
#     seq = T1MRF(**sequence_config)
#
#     m = RandomMap3D("TestRandom3DMovement","",nb_slices=nb_slices,nb_empty_slices=0,undersampling_factor=1,repeat_slice=0,resting_time=4000,image_size=(int(npoint/2),int(npoint/2)),region_size=0,mask_reduction_factor=0,gen_mode="other")
#
#     m.build_timeline(seq)
#     base_images=np.zeros(image_size)
#     center_slice=int(nb_slices/2)
#     center_point=int(npoint/4)
#     dslice=4
#     dpoint=10
#     base_images[center_slice-dslice:center_slice+dslice,center_point-dpoint:center_point+dpoint,center_point-dpoint:center_point+dpoint]=1.0
#     base_images=np.ones(image_size)
#
#     base_images=np.expand_dims(base_images,axis=0)
#     m.images_series=np.vstack([base_images]*nb_segments)
#
#     #animate_images(m.images_series[:,8,:,:])
#
#     kdata_all_channels_all_slices=m.generate_kdata(radial_traj)
#     kdata_all_channels_all_slices=np.expand_dims(kdata_all_channels_all_slices,axis=0)
#
#     density = np.abs(np.linspace(-1, 1, npoint))
#     density = np.expand_dims(density,tuple(range(kdata_all_channels_all_slices.ndim-1)))
#     kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(1,nb_segments,-1,npoint)
#     kdata_all_channels_all_slices *= density
#
#
#
#     np.save(filename_kdata,kdata_all_channels_all_slices)
# else:
#     kdata_all_channels_all_slices=np.load(filename_kdata)

# density = np.abs(np.linspace(-1, 1, npoint))
# density = np.expand_dims(density,tuple(range(kdata_all_channels_all_slices.ndim-1)))
# kdata_all_channels_all_slices=np.ones((nb_channels,nb_segments,nb_slices,npoint),dtype="complex64")
# kdata_all_channels_all_slices *= density


if nb_gating_spokes>0:
    if str.split(filename_displacement, "/")[-1] not in os.listdir(folder):
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

        nav_image_size = (int(npoint_nav / 2),)



        print("Estimating Movement...")
        shifts = list(range(-15, 30))
        bottom = 15
        top = int(npoint_nav / 2) - 30
        displacements_all_channels = []

        #image_nav_all_channels = []
        for j in range(nb_channels):
            images_series_rebuilt_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[j], axis=0), nav_traj,
                                                                     nav_image_size, b1=None)
            image_nav_ch = np.abs(images_series_rebuilt_nav_ch)
            displacements = calculate_displacement(image_nav_ch, bottom, top, shifts, lambda_tv=0.001)
            displacements_all_channels.append(displacements)

        displacements_all_channels=np.array(displacements_all_channels)
            # plt.figure()
            # plt.imshow(image_nav_ch.reshape(-1, int(npoint / 2)).T, cmap="gray")
            # plt.title("Image channel {}".format(j))

        # plt.close("all")
        #image_nav_all_channels = np.array(image_nav_all_channels)

        displacements_all_channels=displacements_all_channels.reshape(nb_channels,nb_slices,-1)
        max_slices = nb_slices
        As_hat_normalized = np.zeros(displacements_all_channels.shape)
        As_hat_filtered = np.zeros(displacements_all_channels.shape)

        from scipy.signal import savgol_filter
        from statsmodels.nonparametric.smoothers_lowess import lowess

        for ch in tqdm(range(nb_channels)):
            for sl in range(max_slices):
                signal = displacements_all_channels[ch, sl, :]
                if np.max(signal) == np.min(signal):
                    signal = 0.5 * np.ones_like(signal)
                else:
                    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
                As_hat_normalized[ch, sl, :] = signal
                # mean=np.mean(signal)
                # std=np.std(signal)
                # ind=np.argwhere(signal<(mean-std)).flatten()
                # signal[ind]=signal[ind-1]
                signal_filtered = savgol_filter(signal, 3, 2)
                signal_filtered = lowess(signal_filtered, np.arange(len(signal_filtered)), frac=0.1)[:, 1]
                As_hat_filtered[ch, sl, :] = signal_filtered

        data_for_pca = As_hat_filtered.reshape(nb_channels, -1)
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1)
        pca.fit(data_for_pca.T)
        pcs = pca.components_ @ data_for_pca
        explained_variances_all_slices = pca.explained_variance_ratio_
        displacements = pcs[0]
        np.save(filename_displacement,displacements)
    else:
        displacements=np.load(filename_displacement)

    #displacements_pt=np.load(filename_displacement_pt)
    displacement_for_binning = displacements
    max_bin = np.max(displacement_for_binning)
    min_bin = np.min(displacement_for_binning)
    bin_width=(max_bin-min_bin)/5

    bins = np.arange(min_bin, max_bin + bin_width, bin_width)
    # print(bins)
    categories = np.digitize(displacement_for_binning, bins)
    df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
    df_groups = df_cat.groupby("cat").count()


# group_1=(categories==1)|(categories==2)
# group_2=(categories==3)
# group_3=(categories==4)
# group_4=(categories==5)
# group_5=(categories==6)
# group_6=(categories==7)
# group_7=(categories==8)
# group_8=(categories==9)
# group_9=(categories==10)|(categories==11)
#
#
# groups=[group_1,group_2,group_3,group_4,group_5,group_6,group_7,group_8,group_9]

group_1=(categories==1)
group_2=(categories==2)
group_3=(categories==3)
group_4=(categories==4)
group_5=(categories==5)


groups=[group_1,group_2,group_3,group_4,group_5]

nb_part=nb_slices
dico_traj_retained = {}
dico_retained_ts={}
for j, g in tqdm(enumerate(groups)):
    print("######################  BUILDING FULL VOLUME AND MASK FOR GROUP {} ##########################".format(j))
    retained_nav_spokes_index = np.argwhere(g).flatten()
    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
        axis=-1)

    spoke_groups=spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:]=spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:]-1 #adjustment for change of partition
    spoke_groups=spoke_groups.flatten()

    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
    #included_spokes[::int(nb_segments / nb_gating_spokes)] = False
    #print(np.sum(included_spokes))

    weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, 1)
    print(len(retained_timesteps))

    #print(traj_retained_final_volume.shape[1]/800)

    dico_traj_retained[j] = weights
    dico_retained_ts[j]=retained_timesteps

plt.close("all")

gr=0
weights_excl=dico_traj_retained[gr]
weights_excl=weights_excl.reshape(-1,nb_slices)
weights_excl[25]

gr=0
weights=dico_traj_retained[gr]
weights=weights.reshape(-1,nb_slices)
weights[25]

diff_weights=weights_excl-weights
vmax=np.max(diff_weights)
vmin=np.min(diff_weights)


plt.imshow(diff_weights.T,vmin=vmin,vmax=vmax,interpolation='nearest',cmap=plt.cm.binary,extent=[0, 1400, 0, 56], aspect=10)

all_weights=np.zeros((nb_allspokes,nb_slices))
fig,ax=plt.subplots(len(groups))
for gr in dico_traj_retained.keys():
    weights=dico_traj_retained[gr]
    retained_timesteps=dico_retained_ts[gr]
    selected=np.zeros(shape=weights.shape)

    selected[retained_timesteps]=(weights>0)*1
    selected=selected.reshape(-1,nb_slices)

    ax[gr].imshow(selected.T,vmin=0,vmax=1,interpolation='nearest',cmap=plt.cm.binary,extent=[0, 1400, 0, 56], aspect=10)
    ax[gr].set_title("gr {}".format(gr))
    all_weights+=selected


plt.figure()
plt.imshow(all_weights.T,vmin=0,vmax=1,interpolation='nearest',cmap=plt.cm.binary,extent=[0, 1400, 0, 56], aspect=10)


#
#
# n_comp=16
#
# filename_virtualcoils= str.split(filename,".dat") [0]+"_virtualcoils_{}.pkl".format(n_comp)
# import pickle
# with open(filename_virtualcoils,"rb") as file:
#     pca_dict=pickle.load(file)
#
# filename_b12Dplus1 = str.split(filename,".dat") [0]+"_b12Dplus1_{}.npy".format(n_comp)
# b1_all_slices_2Dplus1_pca=np.load(filename_b12Dplus1)
#
#
#
# traj_reco = radial_traj_2D.get_traj_for_reconstruction(ntimesteps).astype("float32")
# #traj_reco = traj_reco.reshape(-1, 2)
#
# mask_allgroups=np.zeros((len(groups),)+image_size,dtype="complex64")
#
#
# import cupy as cp
#
# list_comp=np.arange(n_comp)
# list_slices=np.arange(nb_slices)
#
# threshold_factor = 1 / 25
#
# kdata_all_channels_all_slices=np.load(filename_kdata)
#
# for gr in tqdm(dico_traj_retained.keys()):
#     current_volume = np.zeros(image_size, dtype="complex64")
#     mask=False
#
#     # data=copy(kdata_all_channels_all_slices)
#     # weights = np.expand_dims((dico_traj_retained[gr]>0)*1,axis=(0,-1))
#     weights = np.expand_dims(dico_traj_retained[gr], axis=(-1))
#     weights = 1 * (weights > 0)
#
#     # data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
#     # data*=weights
#     retained_timesteps=dico_retained_ts[gr]
#     data = np.zeros((nb_channels, len(retained_timesteps), 8, nb_slices, npoint), dtype=kdata_all_channels_all_slices.dtype)
#     traj_curr=traj_reco[retained_timesteps]
#     traj_curr = traj_curr.reshape(-1, 2)
#     for ch in tqdm(range(nb_channels)):
#         data[ch] = np.fft.fftshift(np.fft.ifft(
#             np.fft.ifftshift(kdata_all_channels_all_slices[ch].reshape(ntimesteps, 8, -1, npoint)[retained_timesteps] * weights, axes=2),
#             axis=2), axes=2)
#
#     data = np.moveaxis(data, -2, 1)
#
#     # data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
#     #images_series_rebuilt = np.zeros(image_size, dtype=np.complex64)
#
#     for sl in tqdm(list_slices):
#         data_curr = cp.asarray(data[:, sl])
#         data_curr = data_curr.reshape(nb_channels, -1)
#         pca=pca_dict[sl]
#
#         print("PCA")
#         data_curr_transformed = pca.transform(data_curr.T)
#         data_curr_transformed = data_curr_transformed.T
#
#         #data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
#         data_curr_transformed=data_curr_transformed.reshape(n_comp, len(retained_timesteps), -1)
#
#
#         for j in tqdm(list_comp):
#             kdata = data_curr_transformed[j].flatten()
#
#
#             fk_gpu = GPUArray(image_size[1:], dtype=np.complex64)
#             print("NUFFT")
#             plan = cufinufft(1, image_size[1:], 1, eps=1e-6, dtype=np.float32)
#             plan.set_pts(to_gpu(traj_curr[:, 0]), to_gpu(traj_curr[:, 1]))
#             plan.execute(to_gpu(kdata.get()), fk_gpu)
#
#             fk = np.squeeze(fk_gpu.get())
#             fk_gpu.gpudata.free()
#             plan.__del__()
#
#             current_volume[sl] += b1_all_slices_2Dplus1_pca[j,sl].conj() * fk
#
#
#
#         unique = np.histogram(np.abs(current_volume), 100)[1]
#         mask = mask | (np.abs(current_volume) > unique[int(len(unique) *threshold_factor)])
#         mask = ndimage.binary_closing(mask, iterations=3)
#
#         mask_allgroups[gr] = mask
#         #volumes_allspokes_allgroups[gr]=images_series_rebuilt
#
# np.save("mask_allgroups_reduced.npy",mask_allgroups)
#
# sl=int(nb_slices/2)
# animate_images(mask_allgroups[:,sl])
#
# mask=False
# for curr_mask in mask_allgroups:
#     mask=mask|curr_mask.astype(int)
#
# animate_images(mask)
#
# np.save("mask_aggregated.npy",mask)
#
#
# traj_reco = radial_traj_2D.get_traj_for_reconstruction(ntimesteps).astype("float32")
# #traj_reco = traj_reco.reshape(-1, 2)
#
# volumes_allspokes_allgroups=np.zeros((len(groups),)+image_size,dtype="complex64")
#
#
# import cupy as cp
#
# list_comp=np.arange(n_comp)
# list_slices=np.arange(nb_slices)
#
# for gr in tqdm(dico_traj_retained.keys()):
#
#     # data=copy(kdata_all_channels_all_slices)
#     # weights = np.expand_dims((dico_traj_retained[gr]>0)*1,axis=(0,-1))
#     weights = np.expand_dims(dico_traj_retained[gr], axis=(-1))
#     weights = 1 * (weights > 0)
#
#     # data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
#     # data*=weights
#     retained_timesteps=dico_retained_ts[gr]
#     data = np.zeros((nb_channels, len(retained_timesteps), 8, nb_slices, npoint), dtype=kdata_all_channels_all_slices.dtype)
#     traj_curr=traj_reco[retained_timesteps]
#     traj_curr = traj_curr.reshape(-1, 2)
#     for ch in tqdm(range(nb_channels)):
#         data[ch] = np.fft.fftshift(np.fft.ifft(
#             np.fft.ifftshift(kdata_all_channels_all_slices[ch].reshape(ntimesteps, 8, -1, npoint)[retained_timesteps] * weights, axes=2),
#             axis=2), axes=2)
#
#     data = np.moveaxis(data, -2, 1)
#
#     # data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
#     #images_series_rebuilt = np.zeros(image_size, dtype=np.complex64)
#
#     for sl in tqdm(list_slices):
#         data_curr = cp.asarray(data[:, sl])
#         data_curr = data_curr.reshape(nb_channels, -1)
#         pca=pca_dict[sl]
#
#         print("PCA")
#         data_curr_transformed = pca.transform(data_curr.T)
#         data_curr_transformed = data_curr_transformed.T
#
#         #data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
#         data_curr_transformed=data_curr_transformed.reshape(n_comp, len(retained_timesteps), -1)
#
#
#         for j in tqdm(list_comp):
#             kdata = data_curr_transformed[j].flatten()
#
#
#             fk_gpu = GPUArray(image_size[1:], dtype=np.complex64)
#             print("NUFFT")
#             plan = cufinufft(1, image_size[1:], 1, eps=1e-6, dtype=np.float32)
#             plan.set_pts(to_gpu(traj_curr[:, 0]), to_gpu(traj_curr[:, 1]))
#             plan.execute(to_gpu(kdata.get()), fk_gpu)
#
#             fk = np.squeeze(fk_gpu.get())
#             fk_gpu.gpudata.free()
#             plan.__del__()
#
#             volumes_allspokes_allgroups[gr,sl] += b1_all_slices_2Dplus1_pca[j,sl].conj() * fk
#
#         #volumes_allspokes_allgroups[gr]=images_series_rebuilt
#
# np.save("volumes_allspokes_allgroups_reduced.npy",volumes_allspokes_allgroups)
#
#
# sl=38
# animate_images(volumes_allspokes_allgroups[:,sl])
#
# plot_image_grid(np.abs(b1_all_slices_2Dplus1_pca[:,sl]),nb_row_col=(4,4))
#
#
# gr=0
#
# weights = np.expand_dims(dico_traj_retained[gr], axis=(-1))
# weights = 1 * (weights > 0)
# # data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
# # data*=weights
# retained_timesteps=dico_retained_ts[gr]
# data = np.zeros((nb_channels, len(retained_timesteps), 8, nb_slices, npoint), dtype=kdata_all_channels_all_slices.dtype)
# traj_curr=traj_reco[retained_timesteps]
# traj_curr = traj_curr.reshape(-1, 2)
# for ch in tqdm(range(nb_channels)):
#     data[ch] = np.fft.fftshift(np.fft.ifft(        np.fft.ifftshift(kdata_all_channels_all_slices[ch].reshape(ntimesteps, 8, -1, npoint)[retained_timesteps] * weights, axes=2),
#             axis=2), axes=2)
#
# data = np.moveaxis(data, -2, 1)
#
#     # data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
#     #images_series_rebuilt = np.zeros(image_size, dtype=np.complex64)
#
# j=9
# sl=38
# images=[]
#
# data_curr = cp.asarray(data[:, sl])
# data_curr = data_curr.reshape(nb_channels, -1)
# pca=pca_dict[sl]
# print("PCA")
# data_curr_transformed = pca.transform(data_curr.T)
# data_curr_transformed = data_curr_transformed.T
#
#         #data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
# data_curr_transformed=data_curr_transformed.reshape(n_comp, len(retained_timesteps), -1)
# for j in range(n_comp):
#     kdata = data_curr_transformed[j].flatten()
#     fk_gpu = GPUArray(image_size[1:], dtype=np.complex64)
#     print("NUFFT")
#     plan = cufinufft(1, image_size[1:], 1, eps=1e-6, dtype=np.float32)
#     plan.set_pts(to_gpu(traj_curr[:, 0]), to_gpu(traj_curr[:, 1]))
#     plan.execute(to_gpu(kdata.get()), fk_gpu)
#     fk = np.squeeze(fk_gpu.get())
#     images.append(fk)
#     fk_gpu.gpudata.free()
#     plan.__del__()
#
# images=np.array(images)
# plt.figure()
# plt.imshow(np.abs(fk))
#
# plot_image_grid(np.abs(images),nb_row_col=(4,4))


L0=10
filename_phi=str.split(dictfile,".dict") [0]+"_phi_L0_{}.npy".format(L0)

if filename_phi not in os.listdir():
    mrfdict = dictsearch.Dictionary()
    keys,values=read_mrf_dict(dictfile,np.arange(0.,1.01,0.1))

    import dask.array as da
    u,s,vh = da.linalg.svd(da.asarray(values))

    vh=np.array(vh)
    s=np.array(s)

    L0=10
    phi=vh[:L0]
    np.save(filename_phi,phi.astype("complex64"))
    del mrfdict
    del keys
    del values
    del u
    del s
    del vh
else:
    phi=np.load(filename_phi)

try:
    del kdata_all_channels_all_slices
except:
    pass

phi=phi.astype("complex64")
L0=10
all_volumes_singular=[]
output_shape = (L0,) + image_size

n_comp=16

filename_virtualcoils= str.split(filename,".dat") [0]+"_virtualcoils_{}.pkl".format(n_comp)
import pickle
with open(filename_virtualcoils,"rb") as file:
    pca_dict=pickle.load(file)

filename_b12Dplus1 = str.split(filename,".dat") [0]+"_b12Dplus1_{}.npy".format(n_comp)
b1_all_slices_2Dplus1_pca=np.load(filename_b12Dplus1)

traj_reco = radial_traj_2D.get_traj_for_reconstruction(1).astype("float32")
traj_reco = traj_reco.reshape(-1, 2)
#
#
# all_volumes_singular=[]
# kdata_all_channels_all_slices=np.load(filename_kdata)
#
# ###TO DO - need to do FFT + NUFFT in each loop
# for gr in tqdm(dico_traj_retained.keys()):
#     #data=copy(kdata_all_channels_all_slices)
#     #weights = np.expand_dims((dico_traj_retained[gr]>0)*1,axis=(0,-1))
#     weights = np.expand_dims(dico_traj_retained[gr], axis=(-1))
#
#     #data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
#     #data*=weights
#     data=np.zeros((nb_channels,ntimesteps,8,nb_slices,npoint),dtype=kdata_all_channels_all_slices.dtype)
#
#     for ch in tqdm(range(nb_channels)):
#         data[ch]=np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices[ch].reshape(ntimesteps,8,-1,npoint)*weights,axes=2),axis=2),axes=2)
#
#
#     data = np.moveaxis(data, -2, 1)
#
#     #data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
#     images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)
#     for sl in tqdm(range(nb_slices)):
#         data_curr = data[:, sl]
#         data_curr = data_curr.reshape(nb_channels, -1)
#         pca=pca_dict[sl]
#
#         data_curr_transformed = pca.transform(data_curr.T)
#         data_curr_transformed = data_curr_transformed.T
#
#         #data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
#         data_curr_transformed=data_curr_transformed.reshape(n_comp, ntimesteps, -1)
#
#
#         for j in tqdm(range(n_comp)):
#             kdata_singular = np.zeros((ntimesteps,npoint*window) + (L0,), dtype=data.dtype)
#             for ts in tqdm(range(ntimesteps)):
#                 kdata_singular[ ts, :, :] = data_curr_transformed[j, ts, :, None] @ (phi[:L0].conj().T[ts][None, :])
#             kdata_singular = np.moveaxis(kdata_singular, -1, 0)
#             kdata_singular = kdata_singular.reshape(L0, -1)
#
#
#             fk = finufft.nufft2d1(traj_reco[:, 0], traj_reco[:, 1], kdata_singular, image_size[1:])
#
#             images_series_rebuilt[:,sl] += np.expand_dims(b1_all_slices_2Dplus1_pca[j,sl].conj(), axis=0) * fk
#
#
#
#
#     all_volumes_singular.append(images_series_rebuilt)
#
#
#
# all_volumes_singular=np.array(all_volumes_singular)
#
# np.save("all_volumes_singular_test_weighted.npy",all_volumes_singular)
#
#
#
#
# np.save("all_volumes_singular_test.npy",all_volumes_singular)
#
# all_volumes_singular=np.load("all_volumes_singular_test_weighted.npy")
#
# sl=int(nb_slices/2)+10
#
# plot_image_grid(np.abs(all_volumes_singular[:,:,sl]).reshape((-1,)+image_size[1:]),nb_row_col=(len(groups),L0))





import cupy as cp

all_volumes_singular_gpu=[]
kdata_all_channels_all_slices=np.load(filename_kdata)
traj_reco = radial_traj_2D.get_traj_for_reconstruction(1).astype("float32")
traj_reco = traj_reco.reshape(-1, 2)

import scipy as sp

dtype=np.float32
###TO DO - need to do FFT + NUFFT in each loop
gr=1
# for gr in tqdm(dico_traj_retained.keys()):
#     #data=copy(kdata_all_channels_all_slices)
#     #weights = np.expand_dims((dico_traj_retained[gr]>0)*1,axis=(0,-1))
#     weights = np.expand_dims(dico_traj_retained[gr], axis=(-1))
#     weights=1*(weights>0)
#     weights=np.ones_like(weights)
#
#     #data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
#     #data*=weights
#     data=np.zeros((nb_channels,ntimesteps,8,nb_slices,npoint),dtype=kdata_all_channels_all_slices.dtype)
#
#     for ch in tqdm(range(nb_channels)):
#         data[ch]=np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices[ch].reshape(ntimesteps,8,-1,npoint)*weights,axes=2),axis=2,workers=24),axes=2)
#
#
#     data = np.moveaxis(data, -2, 1)
#
#     #data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
#     images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)
#     for sl in tqdm(range(nb_slices)):
#         data_curr = cp.asarray(data[:, sl])
#         data_curr = data_curr.reshape(nb_channels, -1)
#         pca=pca_dict[sl]
#
#         print("PCA")
#         data_curr_transformed = pca.transform(data_curr.T)
#         data_curr_transformed = data_curr_transformed.T
#
#         #data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
#         data_curr_transformed=data_curr_transformed.reshape(n_comp, ntimesteps, -1)
#
#
#         for j in tqdm(range(n_comp)):
#             kdata_singular = cp.zeros((ntimesteps,npoint*window) + (L0,), dtype=data.dtype)
#             for ts in tqdm(range(ntimesteps)):
#                 kdata_singular[ ts, :, :] = cp.matmul(data_curr_transformed[j, ts, :, None],(cp.asarray(phi[:L0]).conj().T[ts][None, :]))
#             kdata_singular = cp.moveaxis(kdata_singular, -1, 0)
#             kdata_singular = kdata_singular.reshape(L0, -1).get()
#             #
#             # fk_gpu = GPUArray((L0,)+image_size[1:], dtype=np.complex64)
#             # print("NUFFT")
#             # plan = cufinufft(1, image_size[1:], L0, eps=1e-6, dtype=dtype)
#             # plan.set_pts(to_gpu(traj_reco[:, 0].astype(dtype)), to_gpu(traj_reco[:, 1].astype(dtype)))
#             # plan.execute(to_gpu(kdata_singular.get()), fk_gpu)
#             #
#             # fk = np.squeeze(fk_gpu.get())
#             # fk_gpu.gpudata.free()
#             # plan.__del__()
#             #
#             fk=finufft.nufft2d1(traj_reco[:, 0], traj_reco[:, 1], kdata_singular, image_size[1:])
#             images_series_rebuilt[:,sl] += np.expand_dims(b1_all_slices_2Dplus1_pca[j,sl].conj(), axis=0) * fk
#
#
#     all_volumes_singular_gpu.append(images_series_rebuilt)
#
#
# for gr in tqdm(dico_traj_retained.keys()):
#     #data=copy(kdata_all_channels_all_slices)
#     #weights = np.expand_dims((dico_traj_retained[gr]>0)*1,axis=(0,-1))
#     weights = np.expand_dims(dico_traj_retained[gr], axis=(-1))
#     weights=1*(weights>0)
#     #weights=np.ones_like(weights)
#
#     #data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
#     #data*=weights
#     data=np.zeros((nb_channels,ntimesteps,8,nb_slices,npoint),dtype=kdata_all_channels_all_slices.dtype)
#
#     #for ch in tqdm(range(nb_channels)):
#     data=np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,8,-1,npoint)*weights[None],axes=3),axis=3,workers=24),axes=3)
#
#
#     data = np.moveaxis(data, -2, 1)
#
#     #data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
#     images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)
#
#     #data_curr_transformed_all=cp.zeros((nb_slices,n_comp,ntimesteps,window*npoint))
#
#     kdata_singular = cp.zeros((nb_slices, ntimesteps, npoint * window) + (L0,), dtype=data.dtype)
#     for sl in tqdm(range(nb_slices)):
#         data_curr = cp.asarray(data[:, sl])
#         data_curr = data_curr.reshape(nb_channels, -1)
#         pca=pca_dict[sl]
#
#         print("PCA")
#         data_curr_transformed = pca.transform(data_curr.T)
#         data_curr_transformed = data_curr_transformed.T
#
#         #data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
#         data_curr_transformed=data_curr_transformed.reshape(n_comp, ntimesteps, -1)
#
#         for j in tqdm(range(n_comp)):
#
#             for ts in tqdm(range(ntimesteps)):
#                 kdata_singular[ sl,ts, :, :] = cp.matmul(data_curr_transformed[j, ts, :, None],(cp.asarray(phi[:L0]).conj().T[ts][None, :]))
#             kdata_singular = cp.moveaxis(kdata_singular, -1, 0)
#             kdata_singular = kdata_singular.reshape(L0, -1).get()
#             # fk_gpu = GPUArray((L0,)+image_size[1:], dtype=np.complex64)
#             # print("NUFFT")
#             # plan = cufinufft(1, image_size[1:], L0, eps=1e-6, dtype=dtype)
#             # plan.set_pts(to_gpu(traj_reco[:, 0].astype(dtype)), to_gpu(traj_reco[:, 1].astype(dtype)))
#             # plan.execute(to_gpu(kdata_singular.get()), fk_gpu)
#             #
#             # fk = np.squeeze(fk_gpu.get())
#             # fk_gpu.gpudata.free()
#             # plan.__del__()
#             #
#             fk=finufft.nufft2d1(traj_reco[:, 0], traj_reco[:, 1], kdata_singular, image_size[1:])
#             images_series_rebuilt[:,sl] += np.expand_dims(b1_all_slices_2Dplus1_pca[j,sl].conj(), axis=0) * fk
#
#     all_volumes_singular_gpu.append(images_series_rebuilt)


import scipy as sp

for gr in tqdm(dico_traj_retained.keys()):
    # data=copy(kdata_all_channels_all_slices)
    # weights = np.expand_dims((dico_traj_retained[gr]>0)*1,axis=(0,-1))
    weights = np.expand_dims(dico_traj_retained[gr], axis=(-1))
    # weights=1*(weights>0)
    weights=np.ones_like(weights)

    # data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
    # data*=weights
    data = np.zeros((nb_channels, ntimesteps, 8, nb_slices, npoint), dtype=kdata_all_channels_all_slices.dtype)

    # for ch in tqdm(range(nb_channels)):
    data = np.fft.fftshift(
        sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices * weights, axes=2), axis=2, workers=24), axes=2)

    data = data.reshape((nb_channels, ntimesteps, 8, nb_slices, -1))
    data = np.moveaxis(data, -2, 1)

    # data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    # data_curr_transformed_all=cp.zeros((nb_slices,n_comp,ntimesteps,window*npoint))

    for sl in tqdm(range(nb_slices)):
        data_curr = cp.asarray(data[:, sl])
        data_curr = data_curr.reshape(nb_channels, -1)
        pca = pca_dict[sl]

        print("PCA")
        data_curr_transformed = pca.transform(data_curr.T)
        data_curr_transformed = data_curr_transformed.T

        # data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
        data_curr_transformed = data_curr_transformed.reshape(n_comp, ntimesteps, -1)

        for j in tqdm(range(n_comp)):
            kdata_singular = cp.zeros((ntimesteps, npoint * window) + (L0,), dtype="complex64")
            for ts in tqdm(range(ntimesteps)):
                kdata_singular[ts, :, :] = cp.matmul(data_curr_transformed[j, ts, :, None],
                                                     (cp.asarray(phi[:L0]).conj().T[ts][None, :]))
            kdata_singular = cp.moveaxis(kdata_singular, -1, 0)
            kdata_singular = kdata_singular.reshape(L0, -1).get()
            fk = finufft.nufft2d1(traj_reco[:, 0], traj_reco[:, 1], kdata_singular, image_size[1:])
            images_series_rebuilt[:, sl] += np.expand_dims(b1_all_slices_2Dplus1_pca[j, sl].conj(), axis=0) * fk


io.write("test_volumes_singular_all_spokes_pca16_l1.mha", np.abs(images_series_rebuilt[1]), tags={"spacing": [dz, dx, dy]})

#
# kdata_all_channels_all_slices=np.load(filename_kdata)
# traj=radial_traj.get_traj().reshape(ntimesteps,-1,3)
# kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,-1)
#
# kdata_singular=np.zeros((nb_channels,)+traj.shape[:-1]+(L0,),dtype=kdata_all_channels_all_slices.dtype)
# for ts in tqdm(range(ntimesteps)):
#     kdata_singular[:,ts,:,:]=kdata_all_channels_all_slices[:,ts,:,None]@(phi.conj().T[ts][None,:])
#
# kdata_singular=np.moveaxis(kdata_singular,-1,1)
#
# kdata_singular=kdata_singular.reshape(nb_channels,L0,-1)
# b1=np.load(filename_b1)
#
# del kdata_all_channels_all_slices
#
#
#
# volumes_singular=simulate_radial_undersampled_singular_images_multi(kdata_singular,radial_traj,image_size,density_adj=False,b1=b1,light_memory_usage=True)
#
# io.write("test_volumes_singular_all_spokes_l1.mha", np.abs(volumes_singular[1]), tags={"spacing": [dz, dx, dy]})


#np.save("test_volumes.npy",images_series_rebuilt)
#np.save("test_weights.npy",weights)

n_comp=16
volumes=np.load("test_volumes.npy")
weights=np.load("test_weights.npy")
weights=np.ones_like(weights)
filename_b12Dplus1 = str.split(filename,".dat") [0]+"_b12Dplus1_{}.npy".format(n_comp)
b1_all_slices=np.load(filename_b12Dplus1)

L0=volumes.shape[0]
size=volumes.shape[1:]
nb_channels=b1_all_slices.shape[0]

nb_slices=size[0]
trajectory=radial_traj_2D
#nb_allspokes = trajectory.paramDict["total_nspokes"]
traj = trajectory.get_traj_for_reconstruction(ntimesteps)

weights=weights.flatten()
traj = traj.reshape(-1, 2).astype("float32")
npoint = trajectory.paramDict["npoint"]

#num_k_samples = traj.shape[0]

output_shape = (L0,) + size
images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

k=0
import datetime
import scipy as sp
start=datetime.datetime.now()
curr_volumes = volumes * np.expand_dims(b1_all_slices[k], axis=0)
curr_kdata_slice=np.fft.fftshift(sp.fft.fft(np.fft.ifftshift(curr_volumes, axes=1),axis=1,workers=24), axes=1).astype("complex64")
end=datetime.datetime.now()
print((end-start)*nb_channels)

import datetime
start=datetime.datetime.now()
all_volumes=np.expand_dims(volumes,axis=0)* np.expand_dims(b1_all_slices, axis=1)
all_curr_kdata_slice=np.fft.fftshift(sp.fft.fft(np.fft.ifftshift(all_volumes, axes=2),axis=2,workers=24), axes=2).astype("complex64")
end=datetime.datetime.now()
print(end-start)

sl=0

l=0

fk_gpu = GPUArray((traj.shape[0]), dtype=np.complex64)
print("NUFFT")
plan = cufinufft(2, size[1:], 1, eps=1e-6, dtype=np.float32)
plan.set_pts(to_gpu(traj[:, 0]), to_gpu(traj[:, 1]))
plan.execute(to_gpu(curr_kdata_slice[l,sl]), fk_gpu)

fk = np.squeeze(fk_gpu.get())
fk_gpu.gpudata.free()
plan.__del__()

curr_kdata=np.zeros((L0,nb_slices,traj.shape[0]), dtype="complex64")
for sl in tqdm(range(nb_slices)):
    curr_kdata[:, sl] = finufft.nufft2d2(traj[:, 0],traj[:, 1],curr_kdata_slice[:, sl])

curr_kdata=np.zeros((L0,nb_slices,traj.shape[0]), dtype="complex64")

import datetime
start=datetime.datetime.now()
curr_kdata = finufft.nufft2d2(traj[:, 0],traj[:, 1],curr_kdata_slice.reshape((L0*nb_slices,)+size[1:])).reshape(L0,nb_slices,-1)
end=datetime.datetime.now()

curr_kdata=curr_kdata.reshape(L0,-1,npoint)
density = np.abs(np.linspace(-1, 1, npoint))
density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
curr_kdata*=density

curr_kdata = curr_kdata.reshape((L0,-1,npoint))
weights = np.expand_dims(weights, axis=(0,-1))
curr_kdata *= weights

curr_kdata = curr_kdata.reshape(L0, nb_slices,traj.shape[0])
curr_kdata = np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(curr_kdata,axes=1),axis=1,workers=24),axes=1)
curr_kdata=curr_kdata.astype(np.complex64)
for sl in tqdm(range(nb_slices)):
    ck_gpu = GPUArray((L0,) + size[1:], dtype=np.complex64)
    print("NUFFT")
    plan = cufinufft(1, size[1:], L0, eps=1e-6, dtype=np.float32)
    plan.set_pts(to_gpu(traj[:, 0]), to_gpu(traj[:, 1]))
    plan.execute(to_gpu(curr_kdata[:,sl]), ck_gpu)
    ck = np.squeeze(ck_gpu.get())
    ck_gpu.gpudata.free()
    plan.__del__()
    images_series_rebuilt[:,sl] += np.expand_dims(b1_all_slices[k,sl].conj(), axis=0) * ck


for sl in tqdm(range(nb_slices)):
    images_series_rebuilt[:,sl] = np.expand_dims(b1_all_slices[k,sl].conj(), axis=0) * finufft.nufft2d1(traj[:, 0],traj[:, 1],curr_kdata[:,sl],size[1:])

import datetime
start=datetime.datetime.now()
images_series_rebuilt=np.expand_dims(b1_all_slices[k,sl].conj(), axis=0)* (finufft.nufft2d1(traj[:, 0],traj[:, 1],curr_kdata.reshape(L0*nb_slices,-1),size[1:])).reshape((L0,)+size)
end=datetime.datetime.now()
animate_images(images_series_rebuilt[:,28])



def undersampling_operator_singular_new(volumes,trajectory,b1_all_slices,ntimesteps,density_adj=True,weights=None,retained_timesteps=None,light_memory_usage=False):
    """
    returns A.H @ W @ A @ volumes where A=F Fourier + sampling operator and W correspond to radial density adjustment
    """

    L0=volumes.shape[0]
    size=volumes.shape[1:]
    nb_channels=b1_all_slices.shape[0]

    nb_slices=size[0]

    #nb_allspokes = trajectory.paramDict["total_nspokes"]
    traj = trajectory.get_traj_for_reconstruction(ntimesteps)
    if retained_timesteps is not None:
        traj=traj[retained_timesteps]
    weights=weights.flatten()
    traj = traj.reshape(-1, 2).astype("float32")
    npoint = trajectory.paramDict["npoint"]

    #num_k_samples = traj.shape[0]

    output_shape = (L0,) + size
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    if weights is not None:
        weights = np.expand_dims(weights, axis=(0, -1))

    for k in tqdm(range(nb_channels)):

        curr_volumes = volumes * np.expand_dims(b1_all_slices[k], axis=0)
        #print(curr_volumes.shape)
        curr_kdata_slice=np.fft.fftshift(sp.fft.fft(
            np.fft.ifftshift(curr_volumes, axes=1),
            axis=1,workers=24), axes=1).astype("complex64")
        #curr_kdata=np.zeros((L0,nb_slices,traj.shape[0]), dtype="complex64")
        curr_kdata = finufft.nufft2d2(traj[:, 0],traj[:, 1],curr_kdata_slice.reshape((L0*nb_slices,)+size[1:])).reshape(L0,nb_slices,-1)

        if density_adj:
            curr_kdata=curr_kdata.reshape(L0,-1,npoint)
            density = np.abs(np.linspace(-1, 1, npoint))
            density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
            curr_kdata*=density

        #TO DO !!!!
        if weights is not None:
            curr_kdata = curr_kdata.reshape((L0,-1,npoint))
            curr_kdata *= weights

        curr_kdata = curr_kdata.reshape(L0, nb_slices,traj.shape[0])
        curr_kdata = np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(curr_kdata,axes=1),axis=1,workers=24),axes=1).astype(np.complex64)

        images_series_rebuilt+=np.expand_dims(b1_all_slices[k,sl].conj(), axis=0)* (finufft.nufft2d1(traj[:, 0],traj[:, 1],curr_kdata.reshape(L0*nb_slices,-1),size[1:])).reshape((L0,)+size)


    #images_series_rebuilt /= num_k_samples
    return images_series_rebuilt


#b1_all_slices.shape
#b1_all_slices=np.expand_dims(np.ones_like(volumes[0]),axis=0)
volumes_us=undersampling_operator_singular_new(volumes,radial_traj_2D,b1_all_slices,ntimesteps=ntimesteps,density_adj=True,weights=weights)

volumes_us/=traj.shape[0]
volumes_us*=traj.shape[0]

all_volumes_singular_gpu=np.array(all_volumes_singular_gpu)
np.save("all_volumes_singular_gpu_test_unweighted_pca{}.npy".format(n_comp),all_volumes_singular_gpu)

mask=np.load(filename_mask)

signals=volumes[:,mask>0]
signals_us=volumes_us[:,mask>0]

#signals/=np.linalg.norm(signals,axis=0,keepdims=True)
#signals_us/=np.linalg.norm(signals_us,axis=0,keepdims=True)

num=np.random.randint(signals.shape[1])

plt.figure()
plt.plot(signals[:,num])
plt.plot(signals_us[:,num])







sl=int(nb_slices/2)+10

plot_image_grid(np.abs(all_volumes_singular_gpu[:,:,sl]).reshape((-1,)+image_size[1:]),nb_row_col=(len(groups),L0))


animate_images(all_volumes_singular[:,0,sl])

comp=1
animate_images([all_volumes_singular[0,comp,sl],all_volumes_singular[1,comp,sl]],cmap="gray")

comp=1
animate_images([all_volumes_singular[1,comp,sl],all_volumes_singular[2,comp,sl]],cmap="gray")




phi=phi.astype("complex64")
L0=2
all_volumes_singular_orig=[]
output_shape = (L0,) + image_size

traj=radial_traj.get_traj().reshape(ntimesteps,-1,3)

traj_reco = radial_traj.get_traj_for_reconstruction(1).astype("float32")
traj_reco = traj_reco.reshape(-1, 3)

b1_all_slices=np.load(filename_b1)
kdata_all_channels_all_slices=np.load(filename_kdata)
for gr in tqdm(dico_traj_retained.keys()):
    #data=np.load(filename_kdata)
    #weights = np.expand_dims((dico_traj_retained[gr]>0)*1,axis=(0,-1))
    weights = np.expand_dims(dico_traj_retained[gr], axis=(-1))

    # data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
    # data*=weights
    data = np.zeros((nb_channels, ntimesteps, 8, nb_slices, npoint), dtype=kdata_all_channels_all_slices.dtype)

    for ch in tqdm(range(nb_channels)):
        data[ch] = kdata_all_channels_all_slices[ch].reshape(ntimesteps, 8, -1, npoint) * weights
    data = data.reshape(nb_channels, ntimesteps, -1)
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)
    for j in tqdm(range(nb_channels)):
        kdata_singular = np.zeros(traj.shape[:-1] + (L0,), dtype=data.dtype)
        for ts in tqdm(range(ntimesteps)):
            kdata_singular[ ts, :, :] = data[j, ts, :, None] @ (phi[:L0].conj().T[ts][None, :])
        kdata_singular = np.moveaxis(kdata_singular, -1, 0)
        kdata_singular = kdata_singular.reshape(L0, -1)

        fk = finufft.nufft3d1(traj_reco[:, 2], traj_reco[:, 0], traj_reco[:, 1], kdata_singular, image_size)

        images_series_rebuilt += np.expand_dims(b1_all_slices[j].conj(), axis=0) * fk




    all_volumes_singular_orig.append(images_series_rebuilt)



all_volumes_singular_orig=np.array(all_volumes_singular_orig)
np.save("all_volumes_singular_orig_test_weighted.npy",all_volumes_singular_orig)

sl=int(nb_slices/2)+10
L0=2
plot_image_grid(np.abs(all_volumes_singular_orig[:,:,sl]).reshape((-1,)+image_size[1:]),nb_row_col=(len(groups),L0))


import SimpleITK as sitk

def command_iteration(method):
    if method.GetOptimizerIteration() == 0:
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():7.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

comp=1
fixed = sitk.GetImageFromArray(np.abs(all_volumes_singular[1,comp,sl]))
#fixed.SetOrigin((0, 0, 0))
#fixed.SetSpacing(spacing)

moving = sitk.GetImageFromArray(np.abs(all_volumes_singular[2,comp,sl]))
#moving.SetOrigin((0, 0, 0))
#moving.SetSpacing(spacing)



R = sitk.ImageRegistrationMethod()

R.SetMetricAsCorrelation()

R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
R.SetOptimizerScalesFromIndexShift()



dimension = 2
offset = [2]*dimension # use a Python trick to create the offset list based on the dimension
transform = sitk.TranslationTransform(dimension, offset)

tx = transform

R.SetInitialTransform(tx)

R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

outTx = R.Execute(fixed, moving)


print("-------")
print(outTx)
print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
print(f" Iteration: {R.GetOptimizerIteration()}")
print(f" Metric value: {R.GetMetricValue()}")

#sitk.WriteTransform(outTx, args[3])

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)

resampler.SetTransform(outTx)

out = resampler.Execute(moving)

simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)


animate_images([sitk.GetArrayFromImage(fixed),sitk.GetArrayFromImage(moving)])
animate_images([sitk.GetArrayFromImage(moving),sitk.GetArrayFromImage(out)])

resampler_inverse = sitk.ResampleImageFilter()
resampler_inverse.SetReferenceImage(fixed)
resampler_inverse.SetInterpolator(sitk.sitkLinear)

inverseTx=outTx.GetInverse()
resampler_inverse.SetTransform(inverseTx)
animate_images([sitk.GetArrayFromImage(resampler.Execute(fixed)),sitk.GetArrayFromImage(moving)])


######Demons##################
def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    Args:
        image: The image we want to resample.
        shrink_factor: A number greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma and shrink factor.
    """
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]
    new_spacing = [((original_sz - 1) * original_spc) / (new_sz - 1)
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(),
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0,
                         image.GetPixelID())

import SimpleITK as sitk
def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform=None,
                      shrink_factors=None, smoothing_sigmas=None):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors: Shrink factors relative to the original image's size.
        smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform,
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(),
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # Run the registration.
    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1],
                                                                moving_images[-1],
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)


# Define a simple callback which allows us to monitor registration progress.
def iteration_callback(filter):
    print('\r{0}: {1:.2f}'.format(filter.GetElapsedIterations(), filter.GetMetric()), end='')


comp=0
fixed_3D = sitk.GetImageFromArray(np.abs(all_volumes_singular_gpu[0,comp,:]))
fixed_3D.SetSpacing([dx,dy,dz])
#fixed.SetOrigin((0, 0, 0))
#fixed.SetSpacing(spacing)

moving_3D = sitk.GetImageFromArray(np.abs(all_volumes_singular_gpu[3,comp,:]))
moving_3D.SetSpacing([dx,dy,dz])
#moving_3D.SetSpacing([dx,dy,dz])

sitk.WriteImage(fixed_3D,"fixed_3D.mha")
#moving_3D.SetSpacing([dx,dy,dz])
sitk.WriteImage(moving_3D,"moving_3D.mha")

# Select a Demons filter and configure it.
demons_filter = sitk.DemonsRegistrationFilter()

demons_filter.SetNumberOfIterations(200)
# Regularization (update field - viscous, total field - elastic).
demons_filter.SetSmoothDisplacementField(True)
demons_filter.SetStandardDeviations(0.01)

# Add our simple callback to the registration filter.
demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))

#initial_tfm = sitk.CenteredTransformInitializer(fixed_3D,
#                                                                    moving_3D,
#                                                                    sitk.Euler2DTransform(),
#                                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)

initial_tfm=None

# Run the registration.
tx = multiscale_demons(registration_algorithm=demons_filter,
                       fixed_image=fixed_3D,
                       moving_image=moving_3D,
                       initial_transform=initial_tfm,
                       shrink_factors=None,
                       smoothing_sigmas=10)



sitk.WriteImage(tx.GetDisplacementField(),"displacement_field_msdemons_test.nii")


import sys
import os


def command_iteration(filter):
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")


if len(sys.argv) < 4:
    print(
        f"Usage: {sys.argv[0]}"
        + " <fixedImageFilter> <movingImageFile> <outputTransformFile>"
    )
    sys.exit(1)

comp=0
fixed_3D = sitk.GetImageFromArray(np.abs(all_volumes_singular_gpu[0,comp,:]))
fixed_3D.SetSpacing([dx,dy,dz])
#fixed.SetOrigin((0, 0, 0))
#fixed.SetSpacing(spacing)

moving_3D = sitk.GetImageFromArray(np.abs(all_volumes_singular_gpu[2,comp,:]))
moving_3D.SetSpacing([dx,dy,dz])
#moving_3D.SetSpacing([dx,dy,dz])

sitk.WriteImage(fixed_3D,"fixed_3D.mha")
#moving_3D.SetSpacing([dx,dy,dz])
sitk.WriteImage(moving_3D,"moving_3D.mha")
matcher = sitk.HistogramMatchingImageFilter()
matcher.SetNumberOfHistogramLevels(1024)
matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
moving = matcher.Execute(moving_3D, fixed_3D)

# The basic Demons Registration Filter
# Note there is a whole family of Demons Registration algorithms included in
# SimpleITK
demons = sitk.DemonsRegistrationFilter()
demons.SetNumberOfIterations(50)
# Standard deviation for Gaussian smoothing of displacement field
demons.SetStandardDeviations(1.0)

demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

displacementField = demons.Execute(fixed_3D, moving_3D)

print("-------")
print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
print(f" RMS: {demons.GetRMSChange()}")

outTx = sitk.DisplacementFieldTransform(displacementField)

sitk.WriteTransform(outTx, sys.argv[3])


def demons_registration(fixed_image, moving_image, fixed_points=None, moving_points=None):
    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform()))

    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)

    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsDemons(10)  # intensities are equal if the difference is less than 10HU

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
        registration_method.AddCommand(sitk.sitkIterationEvent,
                                       lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points,
                                                                                   moving_points))

    return registration_method.Execute(fixed_image, moving_image)


tx = demons_registration(fixed_image = fixed_3D,
                         moving_image = moving_3D
                         )

outTx = sitk.DisplacementFieldTransform(tx)

sitk.WriteImage(outTx.GetDisplacementField(),"displacement_field_msdemons_test.nii")


data=np.load(filename_kdata)
gr=0
weights = np.expand_dims((dico_traj_retained[gr]>0)*1,axis=(0,-1))
data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
data*=weights

volumes_full= simulate_radial_undersampled_images_multi(data,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,normalize_iterative=True)[0]
animate_images(volumes_full)

del kdata_all_channels_all_slices
del b1_all_slices



########################## Dict mapping ########################################
seq = None

load_map=False
save_map=True

n_comp=16
mask = np.load(filename_mask)

x_min=np.min(np.argwhere(mask>0)[:,1])
y_min=np.min(np.argwhere(mask>0)[:,2])

x_max=np.max(np.argwhere(mask>0)[:,1])
y_max=np.max(np.argwhere(mask>0)[:,2])

mask[:,x_min:(x_max+1),y_min:(y_max+1)]=1

animate_images(mask)


volumes_allgroups_singular = np.load("all_volumes_singular_gpu_test_unweighted_pca{}.npy".format(n_comp))
#volumes_all = np.load(filename_volume)


mrfdict = dictsearch.Dictionary()
mrfdict.load(dictfile, force=True)
keys = mrfdict.keys
array_water = mrfdict.values[:, :, 0]
array_fat = mrfdict.values[:, :, 1]
array_water_projected=array_water@phi.T.conj()
array_fat_projected=array_fat@phi.T.conj()


mrfdict_light = dictsearch.Dictionary()
mrfdict_light.load(dictfile_light, force=True)
keys_light = mrfdict_light.keys
array_water = mrfdict_light.values[:, :, 0]
array_fat = mrfdict_light.values[:, :, 1]
array_water_light_projected=array_water@phi.T.conj()
array_fat_light_projected=array_fat@phi.T.conj()

# sl=int(nb_slices/2)
# new_mask=np.zeros(mask.shape)
# new_mask[sl]=mask[sl]
# mask=new_mask

niter = 0
ngroups=volumes_allgroups_singular.shape[0]


if niter>0:
    b1_all_slices=np.load(filename_b1)
else:
    b1_all_slices=None
return_cost=False
#animate_images(mask)

for gr in range(ngroups):
    volumes_all=volumes_allgroups_singular[gr]
    suffix="_2Dplus1_pca{}_gr{}".format(n_comp,gr)
    #suffix="_old"
    if not(load_map):
        #niter = 0
        optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=None,split=2000,pca=True,threshold_pca=10,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps,b1=b1_all_slices,threshold_ff=0.9,dictfile_light=(array_water_light_projected,array_fat_light_projected,keys_light),mu=1,mu_TV=1,weights_TV=[1.,0.,0.],return_cost=return_cost)#,mu_TV=1,weights_TV=[1.,0.,0.])
        all_maps=optimizer.search_patterns_test_multi_2_steps_dico((array_water_projected,array_fat_projected,keys),volumes_all,retained_timesteps=None)

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



gr=4

curr_file='./data/InVivo/3D/patient.003.v13/meas_MID00021_FID42448_raFin_3D_tra_1x1x5mm_FULL_new_2Dplus1_pca16_gr{}_MRF_map.pkl'.format(gr)

file = open(curr_file, "rb")
all_maps = pickle.load(file)
file.close()

k="wT1"
mask=all_maps[0][1]
map_curr=makevol(all_maps[0][0][k],mask>0)
map_curr=np.moveaxis(map_curr,0,-1)

target_file=str.split(str.split(curr_file,"/")[-1],".pkl")[0]+"{}.mat".format(k)
target_file="../PROST/DATA/"+target_file

savemat(target_file,{"img":map_curr})


all_volumes_singular=np.load("all_volumes_singular_test_weighted.npy")
gr=0

sl=28+10
image_size=(56,400,400)
L0=10
ngroups=5
plot_image_grid(np.abs(all_volumes_singular[:,:,sl]).reshape((-1,)+image_size[1:]),nb_row_col=(ngroups,L0))


l=0
dz=5
dx=1
dy=1
sl=28+10
for gr in list(range(4)):

    #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
    #map_Python.buildParamMap()
    file_mha ="volume_singular_pca{}_l{}_gr{}".format(n_comp,l,gr)
    io.write(file_mha,np.abs(all_volumes_singular_gpu[gr,l]),tags={"spacing":[dz,dx,dy]})





l=0
dz=5
dx=1
dy=1
sl=28+10
for gr in list(range(4)):

    #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
    #map_Python.buildParamMap()
    file_mha ="volume_singular_orig_l{}_gr{}".format(l,gr)
    io.write(file_mha,np.abs(all_volumes_singular_orig[gr,l]),tags={"spacing":[dz,dx,dy]})















#############Local low rank#############################
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.u<se("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,BruteDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat





n_comp=16

all_volumes_singular=np.load("all_volumes_singular_gpu_test_weighted_pca{}.npy".format(n_comp))

nb_gr,n_comp,nb_slices,npoint_im,npoint_im=all_volumes_singular.shape

gr=0
all_volumes_singular_currgr=np.abs(all_volumes_singular[gr])

sl=int(nb_slices/2)+10
plt.imshow(np.abs(all_volumes_singular_currgr[gt,sl]))

pixel=(sl,173,179)
window=(2,10,10)
values,patch=select_patch(pixel,all_volumes_singular_currgr,window=window)

mask_patch=np.zeros(all_volumes_singular_currgr[0].shape)
sls,rows,cols=zip(*[tuple(x) for x in patch.T])
mask_patch[sls,rows,cols]=1

plt.figure()
plt.imshow(mask_patch[sl]+1000*np.abs(all_volumes_singular_currgr[gr,sl]))

values=values.reshape(-1,400)

from sklearn.decomposition import PCA

pca=PCA(n_components=0.95)
pca.fit(values)
values_transf=pca.transform(values)


values_filtered=values_transf@pca.components_

values_filtered=values_filtered.reshape((n_comp,)+tuple(np.array(window)*2))
values=values.reshape((n_comp,)+tuple(np.array(window)*2))

comp_curr=0
animate_multiple_images(values[comp_curr],values_filtered[comp_curr])


sl=int(nb_slices/2)+10
window=(2,10,10)


base_folder = "./data/InVivo/3D"
localfile="/patient.003.v13/meas_MID00021_FID42448_raFin_3D_tra_1x1x5mm_FULL_new.dat"
filename = base_folder+localfile
filename_mask= str.split(filename,".dat") [0]+"_mask{}.npy".format("")
mask=np.load(filename_mask)

mask_currsl=mask[sl]

x_min=np.min(np.argwhere(mask_currsl>0)[:,0])
y_min=np.min(np.argwhere(mask_currsl>0)[:,1])
x_max=np.max(np.argwhere(mask_currsl>0)[:,0])
y_max=np.max(np.argwhere(mask_currsl>0)[:,1])

new_image=np.zeros(all_volumes_singular_currgr[:,sl].shape)

for x in tqdm(np.arange(x_min,x_max,window[1])):
    for y in np.arange(y_min, y_max, window[2]):
        values, patch = select_patch((sl,x,y), all_volumes_singular_currgr, window=window)

        mask_patch = np.zeros(all_volumes_singular_currgr[0].shape)
        rows, cols = zip(*[tuple(x[1:]) for x in patch.T])
        #mask_patch[sls, rows, cols] = 1

        patch_slice=patch.T[patch.T[:, 0] == patch.T[:, 0][0]][:,1:]
        rows, cols = zip(*[tuple(x) for x in patch_slice])

        #plt.figure()
        #plt.imshow(mask_patch[sl] + 1000 * np.abs(all_volumes_singular_currgr[gr, sl]))

        values = values.reshape(-1, 400)

        from sklearn.decomposition import PCA

        pca = PCA(n_components=0.9)
        pca.fit(values)
        values_transf = pca.transform(values)

        values_filtered = values_transf @ pca.components_

        #values_filtered = values_filtered.reshape((n_comp,) + tuple(np.array(window) * 2))
        values_filtered=values_filtered.reshape(n_comp,2*window[0],-1)
        values_filtered=values_filtered[:,3]
        new_image[:,rows,cols]=values_filtered



plt.figure()
plt.imshow(new_image[0])


plt.figure()
plt.imshow(all_volumes_singular_currgr[1,sl])

animate_images(new_image)


gr=3
all_volumes_singular_matlab=all_volumes_singular_gpu[gr]
all_volumes_singular_matlab=np.moveaxis(all_volumes_singular_matlab,1,-1)
all_volumes_singular_matlab=np.moveaxis(all_volumes_singular_matlab,0,-1)

savemat("../PROST/DATA/data_MRF_3D_CS_unweighted_gr{}.mat".format(gr),{"img":all_volumes_singular_matlab})



volumes_singular_denoised_all=[]
for gr in range(4):
    volumes_singular_denoised=loadmat("../PROST/data_MRF_3D_CS_unweighted_gr{}_denoised.mat".format(gr))["hd_prost_reco"]
    volumes_singular_denoised_all.append(volumes_singular_denoised)

volumes_singular_denoised_all=np.array(volumes_singular_denoised_all)
volumes_singular_denoised_all=np.moveaxis(volumes_singular_denoised_all,-1,1)
volumes_singular_denoised_all=np.moveaxis(volumes_singular_denoised_all,-1,2)


l=1
dz=5
dx=1
dy=1
for gr in list(range(4)):

    #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
    #map_Python.buildParamMap()
    file_mha ="volume_singular_denoised_l{}_gr{}".format(l,gr)
    io.write(file_mha,np.abs(volumes_singular_denoised_all[gr,l]),tags={"spacing":[dz,dx,dy]})

animate_images(volumes_singular_denoised_all[:,0,7])





volumes_singular_denoised_all=np.stack([volumes_singular_denoised,volumes_singular_denoised_1],axis=0)
volumes_singular_denoised_all=np.moveaxis(volumes_singular_denoised_all,-1,1)
volumes_singular_denoised_all=np.abs(volumes_singular_denoised_all)


sl=5
plot_image_grid(np.abs(volumes_singular_denoised_all[:,:,:,:,5].reshape(-1,400,201)),nb_row_col=(2,10))



threshold=1e-3

volumes_singular_denoised_all[volumes_singular_denoised_all>threshold]=1
volumes_singular_denoised_all[volumes_singular_denoised_all<=threshold]=volumes_singular_denoised_all[volumes_singular_denoised_all<=threshold]/threshold





volumes_singular_denoised_all=np.moveaxis(volumes_singular_denoised_all,-1,2)
import SimpleITK as sitk

comp=2
volume_fixed=np.abs(volumes_singular_denoised_all[0,comp,:])
volume_fixed/=np.max(volume_fixed)
fixed_3D = sitk.GetImageFromArray(volume_fixed)
fixed_3D.SetSpacing([dy,dx,dz])
#fixed.SetOrigin((0, 0, 0))
#fixed.SetSpacing(spacing)

volume_moving=np.abs(volumes_singular_denoised_all[3,comp,:])
volume_moving/=np.max(volume_moving)
moving_3D = sitk.GetImageFromArray(volume_moving)
moving_3D.SetSpacing([dy,dx,dz])
#moving_3D.SetSpacing([dx,dy,dz])

sitk.WriteImage(fixed_3D,"fixed_3D.mha")
#moving_3D.SetSpacing([dx,dy,dz])
sitk.WriteImage(moving_3D,"moving_3D.mha")

# Select a Demons filter and configure it.
demons_filter = sitk.DemonsRegistrationFilter()

demons_filter.SetNumberOfIterations(1000)
# Regularization (update field - viscous, total field - elastic).
demons_filter.SetSmoothDisplacementField(False)
demons_filter.SetStandardDeviations(2)

# Add our simple callback to the registration filter.
demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))

#initial_tfm = sitk.CenteredTransformInitializer(fixed_3D,
#                                                                    moving_3D,
#                                                                    sitk.Euler2DTransform(),
#                                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)

initial_tfm=None

# Run the registration.
tx = multiscale_demons(registration_algorithm=demons_filter,
                       fixed_image=fixed_3D,
                       moving_image=moving_3D,
                       initial_transform=initial_tfm,
                       shrink_factors=None,
                       smoothing_sigmas=None)



sitk.WriteImage(tx.GetDisplacementField(),"displacement_field_msdemons_test.nii")




















#############Local low rank#############################
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.u<se("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,BruteDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat


base_folder = "./data/InVivo/3D"

gr=0


return_cost=True
return_matched_signals=True
#keys=list(all_maps.keys())
keys=[0,3]
l=1

dz=5
dx=1
dy=1


for gr in tqdm([2,3]):
    file_map=base_folder + "/patient.003.v13/test_volume_comp_v2_gr{}_CF_iterative_2Dplus1_MRF_map.pkl".format(gr)

    #file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format("")
    curr_file=file_map
    file = open(curr_file, "rb")
    all_maps = pickle.load(file)
    file.close()


    for iter in keys:

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

        if return_matched_signals:
            matched_volumes=makevol(all_maps[iter][-1][l],mask>0)
            file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                                 "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "_it{}_l{}_{}.mha".format(
                iter,l, "matchedvolumes")
            io.write(file_mha, np.abs(matched_volumes), tags={"spacing": [dz, dx, dy]})

all_matched_volumes=[]

for gr in np.arange(4):
    file_mha= base_folder + "/patient.003.v13/test_volume_comp_v2_gr{}_CF_iterative_2Dplus1_MRF_map_it3_l1_matchedvolumes.mha".format(gr)
    matched_volume=io.read(file_mha)
    matched_volume=np.array(matched_volume)
    all_matched_volumes.append(matched_volume)

all_matched_volumes=np.array(all_matched_volumes)


sl=50
moving_image=np.concatenate([all_matched_volumes[:,sl],all_matched_volumes[1:-1,sl][::-1]],axis=0)

animate_images(moving_image,interval=10)

from PIL import Image
gif=[]
volume_for_gif = np.abs(moving_image)
for i in range(volume_for_gif.shape[0]):
    img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    img=img.convert("P")
    gif.append(img)

filename_gif = str.split(filename_volume,".npy") [0]+"_moving_singular_l1_sl{}.gif".format(sl)
gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)




import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.u<se("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,BruteDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat

file_map="test_volume_comp_v2_allgroups_CF_iterative_2Dplus1_MRF_map.pkl"
#file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format("")
curr_file=file_map
file = open(curr_file, "rb")
all_maps = pickle.load(file)
file.close()


iter=3

matched_signals=all_maps[iter][-1]
nb_singular_images=matched_signals.shape[0]
mask=all_maps[iter][1]
nb_signals=mask.sum()

matched_signals=matched_signals.reshape(nb_singular_images,-1,nb_signals)
nb_gr=matched_signals.shape[1]


l=1

dx=1
dy=1
dz=5
for gr in range(nb_gr):
    matched_volumes=makevol(matched_signals[l][gr],mask>0)
    file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                                 "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "_gr{}_it{}_l{}_{}.mha".format(gr,
                iter,l, "matchedvolumes")
    if file_mha.startswith("/"):
        file_mha=file_mha[1:]
    io.write(file_mha, np.abs(matched_volumes), tags={"spacing": [dz, dx, dy]})




all_matched_volumes=[]

for gr in np.arange(nb_gr):
    file_mha= 'test_volume_comp_v2_allgroups_CF_iterative_2Dplus1_MRF_map_gr{}_it3_l1_matchedvolumes.mha'.format(gr)
    matched_volume=io.read(file_mha)
    matched_volume=np.array(matched_volume)
    all_matched_volumes.append(matched_volume)

all_matched_volumes=np.array(all_matched_volumes)


sl=45
moving_image=np.concatenate([all_matched_volumes[:,sl],all_matched_volumes[1:-1,sl][::-1]],axis=0)

animate_images(moving_image,interval=10)

from PIL import Image
gif=[]
volume_for_gif = np.abs(moving_image)
for i in range(volume_for_gif.shape[0]):
    img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    img=img.convert("P")
    gif.append(img)

filename_gif = str.split(file_map,".npy") [0]+"_moving_singular_l1_sl{}.gif".format(sl)
gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)
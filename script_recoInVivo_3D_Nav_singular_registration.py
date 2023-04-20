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


#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"

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

print("Calculating Coil Sensitivity....")

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
#radial_traj.adjust_traj_for_window(window)

nb_segments=radial_traj.get_traj().shape[0]

if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage,hanning_filter=True)
    np.save(filename_b1,b1_all_slices)
else:
    b1_all_slices=np.load(filename_b1)


sl=int(b1_all_slices.shape[1]/2)
list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))

print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    kdata_all_channels_all_slices=np.load(filename_kdata)
    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_iterative=True)
    np.save(filename_volume,volumes_all)
    # sl=20
    ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
    del volumes_all


if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
     selected_spokes = np.r_[10:400]
     kdata_all_channels_all_slices=np.load(filename_kdata)
     selected_spokes=None
     mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/10, normalize_kdata=False,light_memory_usage=True,selected_spokes=selected_spokes)
     np.save(filename_mask,mask)
     animate_images(mask)
     del mask



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
            displacements = calculate_displacement(image_nav_ch, bottom, top, shifts, lambda_tv=0.01)
            displacements_all_channels.append(displacements)

        displacements_all_channels=np.array(displacements_all_channels)
            # plt.figure()
            # plt.imshow(image_nav_ch.reshape(-1, int(npoint_nav / 2)).T, cmap="gray")
            # plt.title("Image channel {}".format(j))
        np.save("image_nav_ch_test.npy",image_nav_ch)

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


group_1=(categories==1)
group_2=(categories==2)
group_3=(categories==3)
group_4=(categories==4)
group_5=(categories==5)


groups=[group_1,group_2,group_3,group_4,group_5]

nb_part=nb_slices
dico_traj_retained = {}
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

    weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, ntimesteps)
    print(len(retained_timesteps))

    #print(traj_retained_final_volume.shape[1]/800)

    dico_traj_retained[j] = weights


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
L0=2
all_volumes_singular=[]
output_shape = (L0,) + image_size

traj=radial_traj.get_traj().reshape(ntimesteps,-1,3)

traj_reco = radial_traj.get_traj_for_reconstruction(1).astype("float32")
traj_reco = traj_reco.reshape(-1, 3)

for gr in tqdm(dico_traj_retained.keys()):
    data=np.load(filename_kdata)
    #weights = np.expand_dims((dico_traj_retained[gr]>0)*1,axis=(0,-1))
    weights = np.expand_dims(dico_traj_retained[gr], axis=(0, -1))

    data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
    data*=weights
    data=data.reshape(nb_channels,ntimesteps,-1)

    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)
    for j in tqdm(range(nb_channels)):
        kdata_singular = np.zeros(traj.shape[:-1] + (L0,), dtype=data.dtype)
        for ts in tqdm(range(ntimesteps)):
            kdata_singular[ ts, :, :] = data[j, ts, :, None] @ (phi[:L0].conj().T[ts][None, :])
        kdata_singular = np.moveaxis(kdata_singular, -1, 0)
        kdata_singular = kdata_singular.reshape(L0, -1)

        fk = finufft.nufft3d1(traj_reco[:, 2], traj_reco[:, 0], traj_reco[:, 1], kdata_singular, image_size)

        images_series_rebuilt += np.expand_dims(b1_all_slices[j].conj(), axis=0) * fk




    all_volumes_singular.append(images_series_rebuilt)



all_volumes_singular=np.array(all_volumes_singular)

np.save("all_volumes_singular_test_weighted.npy",all_volumes_singular)


np.save("all_volumes_singular_test.npy",all_volumes_singular)

all_volumes_singular=np.load("all_volumes_singular_test_weighted.npy")

sl=int(nb_slices/2)+10

plot_image_grid(np.abs(all_volumes_singular[:,:,sl]).reshape((-1,)+image_size[1:]),nb_row_col=(len(groups),L0))

comp=1
animate_images([all_volumes_singular[0,comp,sl],all_volumes_singular[1,comp,sl]],cmap="gray")

comp=1
animate_images([all_volumes_singular[1,comp,sl],all_volumes_singular[2,comp,sl]],cmap="gray")


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



fixed_3D = sitk.GetImageFromArray(np.abs(all_volumes_singular[1,comp,:]))
fixed_3D.SetSpacing([dx,dy,dz])
#fixed.SetOrigin((0, 0, 0))
#fixed.SetSpacing(spacing)

moving_3D = sitk.GetImageFromArray(np.abs(all_volumes_singular[2,comp,:]))
moving_3D.SetSpacing([dx,dy,dz])
#moving_3D.SetSpacing([dx,dy,dz])

sitk.WriteImage(fixed_3D,"fixed_3D.mha")
#moving_3D.SetSpacing([dx,dy,dz])
sitk.WriteImage(moving_3D,"moving_3D.mha")

# Select a Demons filter and configure it.
demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
demons_filter.SetNumberOfIterations(20)
# Regularization (update field - viscous, total field - elastic).
demons_filter.SetSmoothDisplacementField(True)
demons_filter.SetStandardDeviations(2)

# Add our simple callback to the registration filter.
demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))

initial_tfm = sitk.CenteredTransformInitializer(fixed,
                                                                    moving,
                                                                    sitk.Euler2DTransform(),
                                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)

initial_tfm=None

# Run the registration.
tx = multiscale_demons(registration_algorithm=demons_filter,
                       fixed_image=fixed_3D,
                       moving_image=moving_3D,
                       initial_transform=initial_tfm,
                       shrink_factors=None,
                       smoothing_sigmas=None)



sitk.WriteImage(tx.GetDisplacementField(),"displacement_field_msdemons_test.nii")





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

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = None

load_map=False
save_map=True

dictfile = "mrf175_SimReco2_light.dict"
dictfile="mrf144w8_SeqFF_PWCR_SimRecoFFDf_light.dict"
dictfile="mrf144w8_SeqFF_PWCR_SimRecoFFDf_adjusted_light.dict"
dictfile="mrf144w8_SeqFF__SimRecoFFDf_light.dict"

#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
#dictfile = "mrf175_Dico2_Invivo.dict"

mask = np.load(filename_mask)
volumes_all = np.load(filename_volume)
#volumes_corrected=np.load(filename_volume_corrected)


#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_corrected[:,4,:,:])
#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])

retained_timesteps=None
ntimesteps=None
if not(load_map):
    niter = 0
    #optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=None,split=100,pca=True,threshold_pca=20,log=False,useGPU_dictsearch=False,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps)
    optimizer = BruteDictSearch(FF_list=np.arange(0, 1.01, 0.01), mask=mask, split=100, pca=True, threshold_pca=20,log=False, useGPU_dictsearch=False, ntimesteps=ntimesteps, log_phase=True)

    all_maps=optimizer.search_patterns(dictfile,volumes_all,retained_timesteps=retained_timesteps)

    if(save_map):
        import pickle

        #file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format(suffix)
        #file_map = filename.split(".dat")[0] + "_corrected_dens_adj{}_MRF_map.pkl".format(suffix)
        file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format(suffix)
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

from mutools import io
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


localfile="/patient.008.v7/meas_MID00020_FID37032_raFin_3D_tra_1x1x5mm_FULL_new.dat"
dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2.26_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.26_reco4_w8_simmean.dict"


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


#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"

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


else :
    data = np.load(filename_save)
    if nb_gating_spokes>0:
        data_for_nav=np.load(filename_nav_save)
#

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


print("Calculating Coil Sensitivity....")

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
#radial_traj.adjust_traj_for_window(window)

nb_segments=radial_traj.get_traj().shape[0]
























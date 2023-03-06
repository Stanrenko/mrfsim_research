
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

localfile="/phantom.001.v1/phantom.001.v1.dat"
localfile="/patient.008.v4/meas_MID00148_FID28313_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile="/patient.002.v5/meas_MID00021_FID34064_raFin_3D_tra_1x1x5mm_FULL_new.dat"
#localfile="/phantom.001.v1/meas_MID00030_FID51057_raFin_3D_phantom_mvt_0"
#localfile="/phantom.006.v1/meas_MID00027_FID02798_raFin_3D_tra_1x1x5mm_FULL_FF.dat"#Box at the top border with more outside
#localfile="/phantom.006.v1/meas_MID00028_FID02799_raFin_3D_tra_1x1x5mm_FULL_new.dat"#Box at the top border with more outside
#localfile="/phantom.006.v1/meas_MID00029_FID02800_raFin_3D_tra_1x1x5mm_FULL_FF_TR4000.dat"#Box at the top border with more outside
#localfile="/phantom.006.v1/meas_MID00023_FID02830_raFin_3D_tra_1x1x5mm_FULL_FF_TR5000.dat"#Box at the top border with more outside
#localfile="/phantom.006.v2/"


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

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
#radial_traj.adjust_traj_for_window(window)

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

ntimesteps=175
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
     mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,threshold_factor=1/30, normalize_kdata=False,light_memory_usage=True,selected_spokes=selected_spokes)
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

ntimesteps=175
if nb_gating_spokes>0:
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


    ch=9
    image_nav_ch =simulate_nav_images_multi(np.expand_dims(data_for_nav[ch],axis=0),nav_traj, nav_image_size)
    #plt.imshow(np.abs(b1_nav[ch].reshape(-1, int(npoint/2))))

    plt.figure()
    plt.imshow(np.abs(image_nav_ch.reshape(-1, int(npoint/2))), cmap="gray")
    plt.figure()
    plt.plot(np.abs(image_nav_ch.reshape(-1, int(npoint/2)))[10])

    print("Rebuilding Nav Images...")
    images_nav_mean = np.abs(simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, b1_nav_mean))
    plt.figure()
    plt.imshow(np.abs(images_nav_mean.reshape(-1, int(npoint/2))),cmap="gray")

    plt.figure()
    plt.plot(np.abs(images_nav_mean.reshape(-1, int(npoint/2)))[10,:])


    print("Estimating Movement...")
    shifts = list(range(-15, 30))
    bottom = 15
    top = int(npoint/2)-30
    displacements = calculate_displacement(image_nav_ch, bottom, top, shifts)

    plt.figure()
    plt.plot(displacements)

    displacement_for_binning = displacements
    bin_width = 5
    max_bin = np.max(displacement_for_binning)
    min_bin = np.min(displacement_for_binning)

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
            idx_cat = df_groups.displacement.idxmax()
            retained_nav_spokes = (categories == idx_cat)

    retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
    spoke_groups = np.argmin(np.abs(np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,nb_segments / nb_gating_spokes).reshape(1,-1)),axis=-1)
    if not (nb_segments == nb_gating_spokes):
        spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
        spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
            nb_segments / nb_gating_spokes / 2) + 1:] - 1
        spoke_groups = spoke_groups.flatten()

    included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])

    if low_freq_encode_corrected_perc is not None :
        included_spokes = included_spokes.reshape(nb_slices, nb_segments)
        width = int(nb_slices/2*low_freq_encode_corrected_perc)
        included_spokes[:int(nb_slices/2)-width,:]=True
        included_spokes[int(nb_slices/2)+width:,:]=True
        included_spokes=included_spokes.flatten()

    included_spokes[::int(nb_segments/nb_gating_spokes)]=False
    #included_spokes[:]=True

    # perc_retained=0.4
    # import random
    # indices_included_random=random.sample(range(spoke_groups.shape[0]),int(perc_retained*spoke_groups.shape[0]))
    # included_spokes=np.zeros(spoke_groups.shape[0])
    # included_spokes[indices_included_random]=1.0
    # included_spokes=included_spokes.astype(bool)

    #traj = radial_traj.get_traj()

    print("Calculating weights for movement...")

    weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, ntimesteps)




print("Rebuilding Images With Corrected volumes...")

b1_full = np.ones(image_size)
b1_full=np.expand_dims(b1_full,axis=0)

b1_full=b1_all_slices

if str.split(filename_volume_corrected,"/")[-1] not in os.listdir(folder):
    kdata_all_channels_all_slices=np.load(filename_kdata)
    volumes_corrected = simulate_radial_undersampled_images_multi_new(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_full,ntimesteps=ntimesteps,density_adj=False,useGPU=False,light_memory_usage=True,retained_timesteps=retained_timesteps,weights=weights)
    animate_images(volumes_corrected[:,int(nb_slices/2),:,:])
    np.save(filename_volume_corrected,volumes_corrected)
else:
    volumes_corrected=np.load(filename_volume_corrected)

kdata_all_channels_all_slices=np.load(filename_kdata)
volumes_full_corrected = simulate_radial_undersampled_images_multi_new(kdata_all_channels_all_slices, radial_traj,
                                                                  image_size, b1=b1_full, ntimesteps=ntimesteps,
                                                                  density_adj=False, useGPU=False,
                                                                  light_memory_usage=True,
                                                                  retained_timesteps=retained_timesteps,
                                                                  weights=weights,ntimesteps_final=1)[0]
# animate_images(volumes_full_corrected)
#
kdata_all_channels_all_slices=np.load(filename_kdata)
volumes_full= simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True,normalize_iterative=True)[0]
animate_images(volumes_full)
animate_multiple_images(volumes_full,volumes_full_corrected)

# kdata_all_channels_all_slices=np.load(filename_kdata)
# weights_one=np.ones(weights.shape)
# volumes_full_not_corrected = simulate_radial_undersampled_images_multi_new(kdata_all_channels_all_slices, radial_traj,
#                                                                   image_size, b1=b1_full, ntimesteps=ntimesteps,
#                                                                   density_adj=False, useGPU=False,
#                                                                   light_memory_usage=True,
#                                                                   retained_timesteps=retained_timesteps,
#                                                                   weights=weights_one,ntimesteps_final=1)[0]
#
# animate_multiple_images(volumes_full_not_corrected,volumes_full_corrected)



del kdata_all_channels_all_slices
del b1_all_slices
del volumes_corrected

########################## Dict mapping ########################################


seq = None

load_map=False
save_map=True

dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2_21_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2_21_reco4_w8_simmean.dict"

#dictfile="mrf144w8_SeqFF_PWCR_SimRecoFFDf_light.dict"
#dictfile="mrf144w8_SeqFF_PWCR_SimRecoFFDf_adjusted_light.dict"
#dictfile="mrf144w8_SeqFF__SimRecoFFDf_light.dict"

#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
#dictfile = "mrf175_Dico2_Invivo.dict"

mask = np.load(filename_mask)
volumes_all = np.load(filename_volume)
#volumes_all = np.load(filename_volume_corrected)
#volumes_corrected=np.load(filename_volume_corrected)


#ani = animate_images(volumes_all[:,4,:,:])
#ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
animate_images(mask)


#
# plt.figure()
# plt.plot(volumes_all[:,sl,200,200])

suffix=""
if not(load_map):
    niter = 0
    if niter>0:
        b1_all_slices=np.load(filename_b1)
    else:
        b1_all_slices=None


    #optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=100, pca=True,
    #                             threshold_pca=20, log=False, useGPU_dictsearch=True, useGPU_simulation=False,
    #                             gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
    #                             b1=b1_all_slices, mu=1,weights_TV=[1,0.2,0.2],mu_TV=1,weights=weights,threshold_ff=0.9,dictfile_light=dictfile_light)

    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=100, pca=True,
                                 threshold_pca=20, log=False, useGPU_dictsearch=True, useGPU_simulation=False,
                                 gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                 b1=b1_all_slices, mu=1, weights_TV=[1, 0.2, 0.2], mu_TV=1,
                                 threshold_ff=0.9, dictfile_light=dictfile_light)

    all_maps = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all, retained_timesteps=None)

if(save_map):
        import pickle

        #file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format(suffix)
        #file_map = filename.split(".dat")[0] + "_corrected_dens_adj{}_MRF_map.pkl".format(suffix)
        file_map = filename.split(".dat")[0] + "{}_new_iter_MRF_map.pkl".format(suffix)
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
        io.write(file_mha,map_for_sim[key],tags={"spacing":[dz,dx,dy]})


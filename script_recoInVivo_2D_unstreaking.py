
#import matplotlib
#matplotlib.use("TkAgg")
import numpy as np
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
import glob


save_kdata=True
save_maps = True

use_GPU = True
light_memory_usage=False

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/zz_Data_old/3_Comp_Matlab/InVivo"
#base_folder = "./data/InVivo/3D"
base_folder = "./data/InVivo"


localfile ="/20211105_TestCS_MRF/meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"
localfile ="/20211209_AL_Tongue/meas_MID00260_FID45164_JAMBES_raFin_CLI.dat"


#localfile = "/20211122_EV_MRF/meas_MID00146_FID42269_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"
#localfile = "/20211122_EV_MRF/meas_MID00147_FID42270_raFin_3D_tra_1x1x5mm_FULL_incoherent.dat"
#localfile = "/20211122_EV_MRF/meas_MID00148_FID42271_raFin_3D_tra_1x1x5mm_FULL_high_res.dat"
#localfile = "/20211122_EV_MRF/meas_MID00149_FID42272_raFin_3D_tra_1x1x5mm_USx2.dat"

# localfile = "/20211123_Phantom_MRF/meas_MID00317_FID42440_raFin_3D_tra_1x1x5mm_FULL_optimRG_vitro.dat"
# localfile = "/20211123_Phantom_MRF/meas_MID00318_FID42441_raFin_3D_tra_1x1x5mm_FULL_standardRG_vitro.dat"
# localfile = "/20211123_Phantom_MRF/meas_MID00319_FID42442_raFin_3D_tra_1x1x5mm_FULL_optimRNoG_vitro.dat"
# localfile = "/20211123_Phantom_MRF/meas_MID00320_FID42443_raFin_3D_tra_1x1x5mm_FULL_optimG_vitro.dat"
# localfile = "/20211123_Phantom_MRF/meas_MID00321_FID42444_raFin_3D_tra_1x1x5mm_FULL_standardRNoG_vitro.dat"

# localfile = "/20211129_BM/meas_MID00085_FID43316_raFin_3D_FULL_highRES_incoh.dat"
# localfile = "/20211129_BM/meas_MID00086_FID43317_raFin_3D_FULL_new_highRES_inco.dat"
# localfile = "/20211129_BM/meas_MID00087_FID43318_raFin_3D_FULL_new_highRES_stack.dat"
localfile = "/20210924_Exam/meas_MID00132_FID34531_CUISSES_raFin_CLI.dat"
localfile ="/Phantom20220310/meas_MID00198_FID57299_JAMBES_raFin_CLI.dat"
localfile ="/Phantom20220310/meas_MID00203_FID57304_JAMBES_raFin_CLI.dat"
localfile ="/Phantom20220310/meas_MID00241_FID57342_JAMBES_raFin_CLI.dat"
localfile ="/Phantom20220310/meas_MID00254_FID57355_JAMBES_raFin_CLI.dat"
localfile="/KB/meas_MID02754_FID765278_JAMBES_raFin_CLI_BILAT.dat"
localfile="/KB/MRF#SSkyra145100#F737243#M2960#D170322#T195218#JAMBES_raFin_CLIBILAT.dat"

filename = base_folder+localfile

#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00044_FID42066_raFin_3D_tra_1x1x5mm_us4_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"


filename_save=str.split(filename,".dat") [0]+".npy"
folder = "/".join(str.split(filename,"/")[:-1])


filename_b1 = str.split(filename,".dat") [0]+"_b1.npy"
filename_volume = str.split(filename,".dat") [0]+"_volumes.npy"
filename_kdata = str.split(filename,".dat") [0]+"_kdata.npy"


#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"

if str.split(filename_save,"/")[-1] not in os.listdir(folder):
    Parsed_File = rT.map_VBVD(filename)
    idx_ok = rT.detect_TwixImg(Parsed_File)
    start_time = time.time()
    RawData = Parsed_File[str(idx_ok)]["image"].readImage()
    #test=Parsed_File["0"]["noise"].readImage()
    #test = np.squeeze(test)



    elapsed_time = time.time()
    elapsed_time = elapsed_time - start_time
    progress_str = "Data read in %f s \n" % round(elapsed_time, 2)
    print(progress_str)
    ## Random map simulation

    data = np.squeeze(RawData)
    data = np.moveaxis(data, -1, 0)
    data = np.moveaxis(data, 1, -1)

    np.save(filename_save,data)

else :
    data = np.load(filename_save)

#data = np.moveaxis(data, 0, -1)
# data=np.moveaxis(data,-2,-1)

data_shape = data.shape

nb_channels = data.shape[1]

ntimesteps=175
nb_allspokes = data_shape[-2]
nspoke=int(nb_allspokes/ntimesteps)
npoint = data_shape[-1]
image_size = (int(npoint/2),int(npoint/2))
nb_slices = data_shape[0]


if str.split(filename_kdata,"/")[-1] in os.listdir(folder):
    del data

radial_traj=Radial(total_nspokes=nb_allspokes,npoint=npoint)

if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    # Density adjustment all slices
    print("Performing Density Adjustment....")
    density = np.abs(np.linspace(-1, 1, npoint))
    kdata_all_channels_all_slices = data.reshape(-1, npoint)
    del data
    kdata_all_channels_all_slices = (kdata_all_channels_all_slices*density).reshape(data_shape)
    #kdata_all_channels_all_slices = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in data]
    #kdata_all_channels_all_slices=np.array(kdata_all_channels_all_slices).reshape(data_shape)

    if save_kdata:
        np.save(filename_kdata, kdata_all_channels_all_slices)
        del kdata_all_channels_all_slices
        #kdata_all_channels_all_slices=open_memmap(filename_kdata)
        kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)





# Coil sensi estimation for all slices
print("Calculating Coil Sensitivity....")

if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
    res = 16
    b1_all_slices=calculate_sensitivity_map(kdata_all_channels_all_slices,radial_traj,res,image_size)
    np.save(filename_b1,b1_all_slices)
else:
    b1_all_slices=np.load(filename_b1)
sl=2
list_images = list(np.abs(b1_all_slices[sl]))
plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))

dico=build_dico_seqParams(filename, folder)

sl=2


kdata_all_channels=kdata_all_channels_all_slices[sl]
b1=b1_all_slices[sl]

mask = build_mask_single_image_multichannel(kdata_all_channels, radial_traj, image_size, b1=b1, density_adj=False,
                                                threshold_factor=1/20)

volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels, radial_traj, image_size, b1=b1,normalize_kdata=False,
                                                                density_adj=False)


radial_traj_anatomy=Radial(total_nspokes=300,npoint=npoint)
radial_traj_anatomy.traj = radial_traj.get_traj()[100:400]
#volume_outofphase=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices[:,800:1200,:,:],radial_traj_anatomy,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True)[0]

res=150
volumes_oop=[]
volumes_oop_filtered=[]
center_res = int(npoint / 2 - 1)

for ch in tqdm(range(nb_channels)):
    current_kdata = np.expand_dims(kdata_all_channels[ch, 100:400, :], axis=0)
    volume_outofphase = \
    simulate_radial_undersampled_images_multi(current_kdata, radial_traj_anatomy,
                                              image_size, b1=None, density_adj=False, ntimesteps=1,
                                              useGPU=False, normalize_kdata=False, memmap_file=None,
                                              light_memory_usage=True)[0]

    current_kdata_filtered=np.zeros(current_kdata.shape,dtype=current_kdata.dtype)
    current_kdata_filtered[:, :, (center_res - int(res / 2)):(center_res + int(res / 2))] = current_kdata[:, :,
                                                                                     (center_res - int(res / 2)):(
                                                                                             center_res + int(
                                                                                         res / 2))] * np.expand_dims(
        np.hanning(2 * int(res / 2)), axis=(0, 1))

    current_kdata_filtered[:, :, (npoint-50):npoint] = current_kdata[:, :,
                                                                                            (npoint-50):npoint] * np.expand_dims(
        np.hanning(100)[:50], axis=(0, 1))

    current_kdata_filtered[:, :, 0:50] = current_kdata[:, :,
                                                         0:50] * np.expand_dims(
        np.hanning(100)[50:], axis=(0, 1))

    volume_outofphase_filtered = \
        simulate_radial_undersampled_images_multi(current_kdata_filtered, radial_traj_anatomy,
                                                  image_size, b1=None, density_adj=False, ntimesteps=1,
                                                  useGPU=False, normalize_kdata=False, memmap_file=None,
                                                  light_memory_usage=True)[0]

    volumes_oop.append(np.abs(volume_outofphase))
    volumes_oop_filtered.append(np.abs(volume_outofphase_filtered))

plot_image_grid(volumes_oop,(6,6))
plot_image_grid(volumes_oop_filtered,(6,6))

streak_ratio=np.mean(np.abs(np.array(volumes_oop)-np.array(volumes_oop_filtered)),axis=(1,2))/np.mean(np.array(volumes_oop_filtered),axis=(1,2))

plt.figure()
plt.plot(streak_ratio)

plt.figure()
plt.plot(np.sort(streak_ratio))

ind_streak=np.argsort(streak_ratio)
plot_image_grid(list(np.array(volumes_oop)[ind_streak]),(6,6))


radial_traj_anatomy_us=Radial(total_nspokes=70,npoint=npoint,nb_slices=nb_slices)
radial_traj_anatomy_us.traj = radial_traj.get_traj()[100:800:10]
#volume_outofphase=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices[:,800:1200,:,:],radial_traj_anatomy,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True)[0]

volumes_oop_us=[]
for ch in tqdm(range(nb_channels)):
    current_kdata=np.expand_dims(kdata_all_channels[ch, 100:800:10, :],axis=0)
    volume_outofphase = \
    simulate_radial_undersampled_images_multi(current_kdata, radial_traj_anatomy_us,
                                              image_size, b1=None, density_adj=False, ntimesteps=1,
                                              useGPU=False, normalize_kdata=False, memmap_file=None,
                                              light_memory_usage=True)[0]



    volumes_oop_us.append(np.abs(volume_outofphase))




diff=np.array(volumes_oop_us)*np.linalg.norm(np.array(volumes_oop)[:,mask>0])/np.linalg.norm(np.array(volumes_oop_us)[:,mask>0])-np.array(volumes_oop)
diff[diff<0]=0
diff=list(diff)
streak_ratio=np.linalg.norm(np.array(diff)[:,mask>0],axis=(1))/np.linalg.norm(np.array(volumes_oop)[:,mask>0],axis=(1))

a=1

kdata_all_channels_unstreaked=kdata_all_channels/np.expand_dims(streak_ratio,axis=(1,2))**a
volumes_all_unstreaked = simulate_radial_undersampled_images_multi(kdata_all_channels_unstreaked, radial_traj, image_size, b1=b1,normalize_kdata=False,
                                                                density_adj=False)

animate_multiple_images(volumes_all,volumes_all_unstreaked)

dictfile = "mrf175_Dico2_Invivo.dict"

seq = None


optimizer = SimpleDictSearch(mask=mask, niter=0, seq=seq, trajectory=radial_traj, split=10, pca=True,
                                 threshold_pca=15, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                 gen_mode="other")
all_maps = optimizer.search_patterns_test(dictfile, volumes_all)

from mutools import io
for k in ["wT1","ff"]:
    iter = 0
    map_rebuilt = all_maps[iter][0]
    mask = all_maps[iter][1]

    values_simu = makevol(map_rebuilt[k], mask > 0)

    file_mha = filename.split(".dat")[0] + "_MRF_map_{}.mha".format(k)
    io.write(file_mha, np.expand_dims(values_simu,axis=0), tags={"spacing": [5, 1, 1]})

all_maps_unstreaked = optimizer.search_patterns_test(dictfile, volumes_all_unstreaked)

for k in ["wT1","ff"]:
    iter = 0
    map_rebuilt = all_maps_unstreaked[iter][0]
    mask = all_maps_unstreaked[iter][1]

    values_simu = makevol(map_rebuilt[k], mask > 0)

    file_mha = filename.split(".dat")[0] + "_MRF_unstreaked_map_{}.mha".format(k)
    io.write(file_mha, np.expand_dims(values_simu,axis=0), tags={"spacing": [5, 1, 1]})


# #
# # plt.figure()
# # plt.hist(np.array(diff)[8])
# #
# # plt.figure()
# # plt.hist(np.array(diff)[10])
#
# plot_image_grid(list(np.array(volumes_oop_us)[np.argsort(streak_ratio)]),(6,6),title="Volume OOP US map for slice {}".format(sl))
#
radial_traj_anatomy=Radial(total_nspokes=700,npoint=npoint)
radial_traj_anatomy.traj = radial_traj.get_traj()[100:800]
test_volume_oop_us=calculate_sensitivity_map(kdata_all_channels[:, 100:800, :],radial_traj_anatomy,300,image_size,hanning_filter=True)
test_volume_oop_us[:,mask>0]=0
plot_image_grid(list(np.abs(test_volume_oop_us)),(6,6),title="Volume OOP fully samples map for slice {}".format(sl))

def any_neighbor_zero(img, i, j):
    for k in range(-1,2):
      for l in range(-1,2):
         if img[i+k, j+k] == 0:
            return True
    return False

def zero_crossing(img):
    img[img > 0] = 1
    img[img < 0] = 0
    out_img = np.zeros(img.shape)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            if img[i,j] > 0 and any_neighbor_zero(img, i, j):
                out_img[i,j] = 255
    return out_img
phases_for_streak_identification=[]

for i in range(len(test_volume_oop_us)):
    curr_edges=ndimage.gaussian_laplace(np.abs(test_volume_oop_us[i]),sigma=3)
    curr_edges=zero_crossing(curr_edges)
    curr_edges[mask>0]=0
    phases_for_streak_identification.append(curr_edges)

plot_image_grid(phases_for_streak_identification,(6,6),title="Volume OOP fully samples map for slice {}".format(sl))

#
#
# from scipy import ndimage
# sigma=4
# volumes_oop_us=[]
#
# from scipy.ndimage.filters import median_filter
#
# radial_traj_anatomy_us=Radial(total_nspokes=10,npoint=npoint,nb_slices=nb_slices)
# radial_traj_anatomy_us.traj = radial_traj.get_traj()[110:120]
# #volume_outofphase=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices[:,800:1200,:,:],radial_traj_anatomy,image_size,b1=b1_all_slices,density_adj=False,ntimesteps=1,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=True)[0]
#
#
# for ch in tqdm(range(nb_channels)):
#     current_kdata=np.expand_dims(kdata_all_channels[ch, 110:120, :],axis=0)
#     volume_outofphase = \
#     simulate_radial_undersampled_images_multi(current_kdata, radial_traj_anatomy_us,
#                                               image_size, b1=None, density_adj=False, ntimesteps=1,
#                                               useGPU=False, normalize_kdata=False, memmap_file=None,
#                                               light_memory_usage=True)[0]
#
#     #curr_edges = ndimage.gaussian_laplace(np.angle(volume_outofphase), sigma=sigma)
#     #curr_edges = zero_crossing(curr_edges)
#     #curr_edges[mask > 0] = 0
#     float_img=np.real(volume_outofphase)/np.max(np.real(volume_outofphase))
#
#     im = np.array(float_img * 255, dtype=np.uint8)
#     # Blur the image for better edge detection
#     img_blur = cv2.GaussianBlur(im, (5, 5), 0)
#     threshold_lower = 50
#     threshold_upper = 250
#     edged = cv2.Canny(img_blur, threshold_lower, threshold_upper)
#     edged[mask > 0] = 0
#
#     volumes_oop_us.append(edged)
#
# plot_image_grid(volumes_oop_us,(6,6),title="Volume OOP fully samples map for slice {}".format(sl))
#
# # Convert to graycsale
# import cv2
# float_img=volumes_oop_us[8]/np.max(volumes_oop_us[8])
#
# im = np.array(float_img * 255, dtype = np.uint8)
# # Blur the image for better edge detection
# img_blur = cv2.GaussianBlur(im, (3,3), 0)
# plt.figure()
# plt.imshow(img_blur)
# threshold_lower = 100
# threshold_upper = 200
# edged = cv2.Canny(img_blur, threshold_lower, threshold_upper)
# edged[mask>0]=0
# plt.figure()
# plt.imshow(edged)
#
# plt.close("all")
# import cv2
#
# img = cv2.imread('image.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
# kern_size = 11
# gray_blurred = cv2.medianBlur(gray, kern_size)
# threshold_lower = 100
# threshold_upper = 220
# edged = cv2.Canny(gray_blurred, threshold_lower, threshold_upper)
# cv2.imshow('edged',edged)
#
# test=median_filter(volumes_oop_us[0], size=20)
# plt.plot()
#
# result = gaussian_laplace(img, sigma=sigma)
#
# diff=np.array(volumes_oop_us)*500000-np.array(volumes_oop)
# diff=list(diff)
#
# plot_image_grid(diff,(6,6),title="Volume OOP fully samples map for slice {}".format(sl))
#
# np.linalg.norm(diff,axis=(1,2))/np.linalg.norm(volumes_oop)






for sl in tqdm(range(nb_slices)):
    print("Processing slice {} out of {}".format(sl, nb_slices))
    kdata_all_channels = kdata_all_channels_all_slices[sl, :, :, :]
    b1 = b1_all_slices[sl]

    filename_mask = str.split(filename, ".dat")[0] + "_mask_{}.npy".format(sl)
    filename_volume = str.split(filename, ".dat")[0] + "_volumes_{}.npy".format(sl)

    print("Building Volumes....")
    if str.split(filename_volume, "/")[-1] not in os.listdir(folder):
        volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels, radial_traj, image_size, b1=b1,normalize_kdata=False,
                                                                density_adj=False)
        np.save(filename_volume, volumes_all)
    else:
        volumes_all = np.load(filename_volume)

    print("Building Mask....")

    if str.split(filename_mask, "/")[-1] not in os.listdir(folder):
        mask = build_mask_single_image_multichannel(kdata_all_channels, radial_traj, image_size, b1=b1, density_adj=False,
                                                threshold_factor=None)
        np.save(filename_mask, mask)
    else:
        mask = np.load(filename_mask)

    #animate_images(volumes_all)
    #volume_rebuilt = build_single_image_multichannel(kdata_all_channels, radial_traj,
   #                                                  image_size, b1=b1, density_adj=False)

    #plt.figure()
    #plt.imshow(np.abs(volume_rebuilt))

    ## Dict mapping

    #dictfile = "mrf175_SimReco2_.dict"
    dictfile = "mrf175_Dico2_Invivo.dict"

    with open("mrf_sequence.json") as f:
        sequence_config = json.load(f)

    seq = T1MRF(**sequence_config)
    file_map = filename.split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl)

    if str.split(file_map,"/")[-1] not in os.listdir(folder):
        niter = 0
        start_time = time.time()
        optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=10, pca=True,
                                     threshold_pca=15, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                     gen_mode="other")
        all_maps = optimizer.search_patterns_test(dictfile, volumes_all)
        end_time = time.time()
        print("Time taken for slice {} : {}".format(sl, end_time - start_time))
        if (save_maps):
            import pickle


            file = open(file_map, "wb")
            # dump information to that file
            pickle.dump(all_maps, file)
            # close the file
            file.close()

    else:
        import pickle

        file = open(file_map, "rb")
        all_maps = pickle.load(file)
        file.close()


path = r"/home/cslioussarenko/PythonRepositories"
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")

from mutools import io

keys = ["ff","wT1","attB1","df"]


dx = 1
dy = 1
dz = 8

for k in tqdm(keys) :
    map_all_slices = np.zeros((nb_slices,)+image_size)

    for sl in range(nb_slices):
        file_map = filename.split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl)
        file = open(file_map, "rb")
        all_maps = pickle.load(file)
        file.close()

        iter = 0
        map_rebuilt = all_maps[iter][0]
        mask = all_maps[iter][1]

        values_simu = makevol(map_rebuilt[k], mask > 0)
        map_all_slices[sl]=values_simu

    file_mha = filename.split(".dat")[0] + "_MRF_map_{}.mha".format(k)
    io.write(file_mha, map_all_slices, tags={"spacing": [dz, dx, dy]})









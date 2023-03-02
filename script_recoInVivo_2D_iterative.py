
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

localfile ="/3D/patient.008.v5/meas_MID00119_FID32357_raFin_2D_tra_1x1x5mm_FULL_bmy.dat"


filename = base_folder+localfile

#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00044_FID42066_raFin_3D_tra_1x1x5mm_us4_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"


filename_save=str.split(filename,".dat") [0]+".npy"
folder = "/".join(str.split(filename,"/")[:-1])


filename_b1 = str.split(filename,".dat") [0]+"_b1.npy"
filename_volume = str.split(filename,".dat") [0]+"_volumes.npy"
filename_kdata = str.split(filename,".dat") [0]+"_kdata.npy"

return_cost=True

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
    b1_all_slices=calculate_sensitivity_map(kdata_all_channels_all_slices,radial_traj,res,image_size,hanning_filter=True)
    np.save(filename_b1,b1_all_slices)
else:
    b1_all_slices=np.load(filename_b1)

sl=int(nb_slices/2)
list_images = list(np.abs(b1_all_slices[sl]))
plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))

#dico=build_dico_seqParams(filename, folder)



sl=int(nb_slices/2)

print("Processing slice {} out of {}".format(sl, nb_slices))
kdata_all_channels = kdata_all_channels_all_slices[sl, :, :, :]
b1 = b1_all_slices[sl]

filename_mask = str.split(filename, ".dat")[0] + "_mask_{}.npy".format(sl)
filename_volume = str.split(filename, ".dat")[0] + "_volumes_{}.npy".format(sl)

print("Building Volumes....")
if str.split(filename_volume, "/")[-1] not in os.listdir(folder):
    volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels, radial_traj, image_size, b1=b1,normalize_kdata=False,
                                                            density_adj=False,normalize_iterative=True)
    np.save(filename_volume, volumes_all)
else:
    volumes_all = np.load(filename_volume)

print("Building Mask....")

if str.split(filename_mask, "/")[-1] not in os.listdir(folder):
    mask = build_mask_single_image_multichannel(kdata_all_channels, radial_traj, image_size, b1=b1, density_adj=False,
                                                threshold_factor=1/25)
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
dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_68_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1_68_reco4_w8_simmean.dict"


file_map = filename.split(".dat")[0] + "_MRF_map_iterative_sl_{}.pkl".format(sl)

if str.split(file_map,"/")[-1] not in os.listdir(folder):
    niter = 10
    start_time = time.time()
    optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=radial_traj, split=100, pca=True,
                                 threshold_pca=20, log=False, useGPU_dictsearch=True, useGPU_simulation=False,
                                 gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                 b1=b1, threshold_ff=0.9, dictfile_light=dictfile_light, mu="Adaptative",mu_TV=0.1,
                                 weights_TV=[1., 1.],
                                 return_cost=return_cost)  # ,mu_TV=1,weights_TV=[1.,0.,0.])
    all_maps = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all, retained_timesteps=None)

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


mask=all_maps[0][1]

plt.close("all")
k="wT1"
plt.figure()
for iter in range(niter+1):
    plt.figure()
    plt.title("Iter {}".format(iter))
    plt.imshow(makevol(all_maps[iter][0][k],mask>0),cmap="inferno")
    plt.colorbar()



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

for iter in range(niter+1):
    for k in tqdm(keys) :
        map_all_slices = np.zeros((nb_slices,)+image_size)

        for sl in range(nb_slices):
            file_map = filename.split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl)
            file = open(file_map, "rb")
            all_maps = pickle.load(file)
            file.close()

            map_rebuilt = all_maps[iter][0]
            mask = all_maps[iter][1]

            values_simu = makevol(map_rebuilt[k], mask > 0)
            map_all_slices[sl]=values_simu

        file_mha = filename.split(".dat")[0] + "_MRF_map_{}.mha".format(k)
        io.write(file_mha, map_all_slices, tags={"spacing": [dz, dx, dy]})

    if return_cost:

        map_all_slices = np.zeros((nb_slices,) + image_size)

        for sl in range(nb_slices):
            file_map = filename.split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl)
            file = open(file_map, "rb")
            all_maps = pickle.load(file)
            file.close()

            map_rebuilt_correlation = all_maps[iter][2]
            mask = all_maps[iter][1]

            values_simu = makevol(map_rebuilt_correlation, mask > 0)
            map_all_slices[sl] = values_simu

        file_mha = filename.split(".dat")[0] + "_MRF_map_{}_correlation.mha".format(k)
        io.write(file_mha, map_all_slices, tags={"spacing": [dz, dx, dy]})

        map_all_slices = np.zeros((nb_slices,) + image_size)

        for sl in range(nb_slices):
            file_map = filename.split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl)
            file = open(file_map, "rb")
            all_maps = pickle.load(file)
            file.close()

            map_rebuilt_phase = all_maps[iter][3]
            mask = all_maps[iter][1]

            values_simu = makevol(map_rebuilt_phase, mask > 0)
            map_all_slices[sl] = values_simu

        file_mha = filename.split(".dat")[0] + "_MRF_map_phase.mha"
        io.write(file_mha, map_all_slices, tags={"spacing": [dz, dx, dy]})








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
                                                threshold_factor=1/25)
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
    dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_68_reco4_w8_simmean.dict"
    dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1_68_reco4_w8_simmean.dict"


    file_map = filename.split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl)

    if str.split(file_map,"/")[-1] not in os.listdir(folder):
        niter = 0
        start_time = time.time()
        optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=radial_traj, split=100, pca=True,
                                     threshold_pca=20, log=False, useGPU_dictsearch=True, useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                     b1=b1_all_slices, threshold_ff=0.9, dictfile_light=dictfile_light, mu=1, mu_TV=1,
                                     weights_TV=[1., 0., 0.],
                                     return_cost=return_cost)  # ,mu_TV=1,weights_TV=[1.,0.,0.])
        all_maps = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all, retained_timesteps=None)

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

if return_cost:

    map_all_slices = np.zeros((nb_slices,) + image_size)

    for sl in range(nb_slices):
        file_map = filename.split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl)
        file = open(file_map, "rb")
        all_maps = pickle.load(file)
        file.close()

        iter = 0
        map_rebuilt_correlation = all_maps[iter][2]
        mask = all_maps[iter][1]

        values_simu = makevol(map_rebuilt_correlation, mask > 0)
        map_all_slices[sl] = values_simu

    file_mha = filename.split(".dat")[0] + "_MRF_map_{}_correlation.mha".format(k)
    io.write(file_mha, map_all_slices, tags={"spacing": [dz, dx, dy]})

    map_all_slices = np.zeros((nb_slices,) + image_size)

    for sl in range(nb_slices):
        file_map = filename.split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl)
        file = open(file_map, "rb")
        all_maps = pickle.load(file)
        file.close()

        iter = 0
        map_rebuilt_phase = all_maps[iter][3]
        mask = all_maps[iter][1]

        values_simu = makevol(map_rebuilt_phase, mask > 0)
        map_all_slices[sl] = values_simu

    file_mha = filename.split(".dat")[0] + "_MRF_map_phase.mha"
    io.write(file_mha, map_all_slices, tags={"spacing": [dz, dx, dy]})










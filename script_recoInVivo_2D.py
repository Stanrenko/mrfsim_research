
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

localfile ="/3D/phantom.015.v2/meas_MID00371_FID50760_raFin_customIR_Reco.dat"

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
data=np.moveaxis(data,0,-1)
data=np.expand_dims(data,axis=0)

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



sl=0

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


ntimesteps=175
use_GPU = True
light_memory_usage=False

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/zz_Data_old/3_Comp_Matlab/InVivo"
#base_folder = "./data/InVivo/3D"
base_folder = "./data"


filename_mask=base_folder+"/TestMarc/meas_MID00020_FID21818_JAMBE_raFin_CLI_EMPTY_ICE_mask.npy"
filename_volume=base_folder+"/TestMarc/meas_MID00020_FID21818_JAMBE_raFin_CLI_EMPTY_ICE_volumes.npy"

mask=np.load(filename_mask)
volumes_all=np.load(filename_volume)


dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2.26_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.26_reco4_w8_simmean.dict"

dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1.14_reco5_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1.14_reco5_w8_simmean.dict"

optimizer = SimpleDictSearch(mask=mask,niter=0,seq=None,trajectory=None,split=2000,pca=True,threshold_pca=15,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps,b1=None,threshold_ff=0.9,dictfile_light=dictfile_light,mu=1,mu_TV=1,weights_TV=[1.,0.,0.],return_cost=False,clustering=True)#,mu_TV=1,weights_TV=[1.,0.,0.])
all_maps=optimizer.search_patterns_test_multi_2_steps_dico(dictfile,volumes_all,retained_timesteps=None)

plt.close("all")

k="ff"

image_map=makevol(all_maps[0][0][k],mask>0)
image_map[mask_ice==0]=0
image_map[image_map>0.35]=0.35
plt.figure()
plt.imshow(image_map.T)
plt.colorbar()


k="attB1"

image_map=makevol(all_maps[0][0][k],mask>0).T
image_map[mask_ice==0]=0
plt.figure()
plt.imshow(image_map)
plt.colorbar()




file="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/zz_Data_old/mask.csv"

import pandas as pd
mask_ice=pd.read_csv(file,header=None)
mask_ice=np.array(mask_ice)

plt.figure()
plt.imshow(mask_ice)


file="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/zz_Data_old/maps_final.csv"

#plt.close("all")

import pandas as pd
data=pd.read_csv(file,header=None)

curr_map=makevol(data.iloc[:,-1],mask_ice>0)
#curr_map[mask==0]=0
fig=plt.figure()
plt.imshow(curr_map)
plt.colorbar()





file="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/zz_Data_old/volumes_water.csv"

import pandas as pd
data=pd.read_csv(file,header=None)
data_real=data.iloc[:,::2]
data_real=data_real.applymap(lambda x:float(x[1:]))

plt.figure()
plt.imshow(np.array(data_real.iloc[:,0]).reshape(180,180))

images=np.array(data_real).T
images=[im.reshape(180,180) for im in images]
images=np.array(images)

animate_images(images)


file="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/zz_Data_old/volumes_real.csv"

import pandas as pd
data_real=pd.read_csv(file,header=None)

images=np.array(data_real).T
images=[im.reshape(180,180).T for im in images]
images=np.array(images)


file="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/zz_Data_old/volumes_imag.csv"

import pandas as pd
data_imag=pd.read_csv(file,header=None)

images_imag=np.array(data_imag).T
images_imag=[im.reshape(180,180).T for im in images_imag]
images_imag=np.array(images_imag)

volumes_all_ice=images + 1j*images_imag




animate_images(images)




animate_images(volumes_all)

nb_signals=mask.sum()
all_signals_python=np.real(volumes_all[:,mask>0])
all_signals_ice = images[:,mask>0]




k="wT1"
map_python=all_maps[0][0][k]
curr_map=np.array(data.iloc[:,0]).reshape(180,180).T
curr_map[mask==0]=0
map_ice=curr_map[mask>0]

error_map=map_ice-map_python

error_volume=makevol(error_map,mask>0)
plt.figure()
plt.imshow(error_volume)
plt.colorbar()

ind_max=np.argsort(error_map)



num_sig=ind_max[-2500]


ind=(135,128)
mask_test=np.zeros_like(mask)
mask_test[ind[0],ind[1]]=1
mask_test=mask_test[mask>0]
num_sig=np.argwhere(mask_test==1)[0][0]

#num_sig=np.random.randint(nb_signals)


signal_python=all_signals_python[:,num_sig]
signal_ice=all_signals_ice[:,num_sig]

signal_python/=np.linalg.norm(signal_python)
signal_ice/=np.linalg.norm(signal_ice)


plt.close("all")
plt.figure()
plt.title("Signal num {}".format(num_sig))
plt.plot(signal_python,label="Python")
plt.plot(signal_ice,label="ICE")
plt.legend()


mask_for_plot=np.zeros(nb_signals)
mask_for_plot[num_sig]=1
mask_for_plot=makevol(mask_for_plot,mask>0)
ind=np.argwhere(mask_for_plot==1)[0]


plt.figure()

plt.imshow(curr_map)
plt.scatter([ind[1]],[ind[0]],s=50,c="red")
#plt.colorbar()

curr_map[ind[0],ind[1]]
map_python[num_sig]


file="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/zz_Data_old/mask.csv"

import pandas as pd
mask_ice=pd.read_csv(file,header=None)
mask_ice=np.array(mask_ice)

plt.figure()
plt.imshow(mask_ice)

file="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/zz_Data_old/image_FullRes.csv"

import pandas as pd
image_full=pd.read_csv(file,header=None)
image_full=np.array(image_full)

plt.figure()
plt.imshow(image_full)

def build_mask(volumes):
    mask = False
    unique = np.histogram(np.abs(volumes), 100)[1]
    mask = mask | (np.mean(np.abs(volumes), axis=0) > unique[len(unique) // 10])
    mask = ndimage.binary_closing(mask, iterations=3)
    return mask*1

mask_from_volume=build_mask(image_full)
plt.figure()
plt.imshow(mask_from_volume)
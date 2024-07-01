
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,BruteDictSearch
from utils_mrf import *
from utils_reco import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
import twixtools
from PIL import Image
from mutools import io
import pandas as pd
import glob
from scipy.io import savemat


try :
    import cupy as cp
except:
    pass
# machines
path = r"/home/cslioussarenko/PythonRepositories"
#path = r"/Users/constantinslioussarenko/PythonGitRepositories/MyoMap"

import sys
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")

from machines import machine, Toolbox, Config, set_parameter, set_output, printer, file_handler, Parameter, RejectException, get_context

@machine
@set_parameter("folder", str, default=None, description="Folder with MRF raw K-space data")
@set_parameter("dictfile",str,default=None,description="Dictfile")
@set_parameter("dictfile_light",str,default=None,description="Dictfile light for clustering")
@set_parameter("target_folder",str,default=None,description="Results folder path from folder")
@set_parameter("return_mat",bool,default=False,description="return .mat")
@set_parameter("return_cost",bool,default=False,description="return cost")
@set_parameter("useGPU",bool,default=True,description="use GPU for dico matching")
@set_parameter("ignored_str",str,default="ignored",description="string filter for ignoring files")
def process_folder_mrf(folder,dictfile,dictfile_light,target_folder,return_mat,useGPU,return_cost,ignored_str):
    
    print("Processing folder {}".format(folder))
    filenames = glob.glob(folder+"/**/*.dat",recursive=True)
    print(filenames)

    for filename in filenames:
        if ignored_str in filename:
            continue
        folder="/".join(str.split(filename,"/")[:-1])
        if target_folder is None:
            results_folder=folder
        else:
            results_folder=folder+"/"+target_folder
        print("Results Folder {}".format(results_folder))
        
        print(os.listdir(folder))
        if not target_folder in os.listdir(folder):
            print("Making results folder")
            os.mkdir(results_folder)
        
        print(os.listdir(folder))
        
        filename_data = str.split(filename, ".dat")[0] + ".npy"
        filename_b1 = str.split(filename, ".dat")[0] + "_b1.npy"
        b1_image_file=str.split(filename_b1, ".npy")[0] + ".jpg"



        file_mha_wT1 = filename.split(".dat")[0] + "_MRF_map_wT1.mha"
        if str.split(file_mha_wT1,"/")[-1] in os.listdir(results_folder):
            print("Found T1 map for {}".format(filename))
            continue

        filename_data=str.replace(filename_data,folder,results_folder)
        filename_b1=str.replace(filename_b1,folder,results_folder)
        b1_image_file=str.replace(b1_image_file,folder,results_folder)

        if str.split(filename_data,"/")[-1] not in os.listdir(results_folder):
            print("Building .npy")
            try:
                Parsed_File = rT.map_VBVD(filename)
                idx_ok = rT.detect_TwixImg(Parsed_File)
                RawData = Parsed_File[str(idx_ok)]["image"].readImage()

            except :
                
                continue

            data = np.squeeze(RawData)
            if data.ndim==3:
                data=np.expand_dims(data,axis=-1)
            data=np.moveaxis(data,-1,0)
            data=np.moveaxis(data,1,-1)
            print(data.shape)
            np.save(filename_data,data)
        else:
            try:
                data=np.load(filename_data)
            except:
                print("Could not load kdata for {}".format(filename))
                continue

        
        
        ntimesteps=175
        nb_allspokes = data.shape[-2]
        npoint = data.shape[-1]
        image_size=(int(npoint/2),int(npoint/2))

        print(data.shape)
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(data.ndim - 1)))
        kdata_all_channels_all_slices = data*density
        del data
        kdata_all_channels_all_slices=kdata_all_channels_all_slices.astype("complex64")
        radial_traj=Radial(total_nspokes=nb_allspokes,npoint=npoint)

        try:
            print("Calculating Coil Sensitivities {}".format(filename_b1))
            if str.split(filename_b1,"/")[-1] not in os.listdir(results_folder):
                res=16
                b1_all_slices=calculate_sensitivity_map(kdata_all_channels_all_slices,radial_traj,res,image_size,hanning_filter=True)
                np.save(filename_b1,b1_all_slices)
            else:
                b1_all_slices=np.load(filename_b1)
        except:
            print("Failed calculating sensi {}".format(filename_b1))
            continue
        #Coil sensi estimation for all slices
        sl = int(b1_all_slices.shape[0]/2)

        list_images=list(np.abs(b1_all_slices[sl,:,:]))
        plot_image_grid(list_images,(6,6),title="Sensivity map for slice {}".format(sl),save_file=b1_image_file)
        
        
        for sl in range(0,kdata_all_channels_all_slices.shape[0]):

            filename_volume = str.split(filename, ".dat")[0] + "_volumes_sl{}.npy".format(sl)
            filename_mask=str.split(filename, ".dat")[0] + "_mask_sl{}.npy".format(sl)
            filename_gif = str.split(filename_volume,".npy") [0]+".gif"
            filename_mask_image = str.split(filename_mask,".npy") [0]+".jpg"
            file_map=str.split(filename, ".dat")[0] + "_MRF_map_sl{}.pkl".format(sl)

            filename_volume=str.replace(filename_volume,folder,results_folder)
            filename_mask=str.replace(filename_mask,folder,results_folder)
            filename_gif=str.replace(filename_gif,folder,results_folder)
            filename_mask_image=str.replace(filename_mask_image,folder,results_folder)
            file_map=str.replace(file_map,folder,results_folder)

            print("Processing slice {} out of {}".format(sl+1,kdata_all_channels_all_slices.shape[0]))
            kdata_all_channels=kdata_all_channels_all_slices[sl,:,:,:]
            b1=b1_all_slices[sl]

            print("Building volume {}".format(filename_volume))
            if not(str.split(filename_volume,"/")[-1] in os.listdir(results_folder)):
                volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False)
                np.save(filename_volume,volumes_all)
            else:
                volumes_all = np.load(filename_volume)

            gif=[]
            volume_for_gif = np.abs(volumes_all)
            for i in range(volume_for_gif.shape[0]):
                img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
                img=img.convert("P")
                gif.append(img)

            
            gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)


            try:
                print("Building mask {}".format(filename_mask))
                if str.split(filename_mask,"/")[-1] not in os.listdir(results_folder):
                    mask=build_mask_single_image_multichannel(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False,threshold_factor=1/25)
                    np.save(filename_mask,mask)
                else:
                    mask=np.load(filename_mask)
            except:
                
                sl=sl-1
                continue
                

            
            plt.figure()
            plt.imshow(mask)
            plt.savefig(filename_mask_image)

            
            niter = 0

            optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=None, split=100, pca=True,
                                             threshold_pca=15, log=False, useGPU_dictsearch=useGPU, useGPU_simulation=False,
                                             gen_mode="other",dictfile_light=dictfile_light,threshold_ff=0.9,ntimesteps=ntimesteps,return_cost=return_cost)
            
            print("Building map {}".format(file_map))
            if not(str.split(file_map,"/")[-1] in os.listdir(results_folder)):
                all_maps=optimizer.search_patterns_test_multi_2_steps_dico(dictfile,volumes_all)


                with open(file_map,"wb") as file:
                    pickle.dump(all_maps, file)

            else:
                with open(file_map, "rb") as file:
                    all_maps = pickle.load(file)
        dx = 1
        dy = 1
        dz = 8

        nb_slices=sl+1

        keys = ["ff","wT1","attB1","df"]
        iter=0
        dict_mat={}
        maps_all_slices={}
        
        for k in tqdm(keys) :
            map_all_slices = np.zeros((nb_slices,)+image_size)
            mask_all_slices = np.zeros((nb_slices,)+image_size)
            phase_all_slices = np.zeros((nb_slices,)+image_size)
            norm_all_slices = np.zeros((nb_slices,)+image_size)
            J_all_slices = np.zeros((nb_slices,)+image_size)
            
            for sl in range(nb_slices):
                file_map = filename.split(".dat")[0] + "_MRF_map_sl{}.pkl".format(sl)
                file_map = str.replace(file_map,folder,results_folder)
                file = open(file_map, "rb")
                all_maps = pickle.load(file)
                file.close()

                map_rebuilt = all_maps[iter][0]
                mask = all_maps[iter][1]
                J_optim=all_maps[iter][2]
                norm=all_maps[iter][4]
                phase=all_maps[iter][3]

                values_simu = makevol(map_rebuilt[k], mask > 0)
                map_all_slices[sl]=values_simu
                mask_all_slices[sl]=mask
                norm_all_slices[sl]=makevol(norm, mask > 0)
                phase_all_slices[sl]=makevol(phase, mask > 0)
                J_all_slices[sl]=makevol(J_optim, mask > 0)
                
            if return_mat:
                dict_mat[k]=map_all_slices
            
            maps_all_slices[k]=map_all_slices[mask_all_slices>0]

            file_mha = filename.split(".dat")[0] + "_MRF_map_{}.mha".format(k)
            file_mha = str.replace(file_mha,folder,results_folder)
            io.write(file_mha, map_all_slices, tags={"spacing": [dz, dx, dy]})

        maps_all_slices_to_save=[(maps_all_slices,mask_all_slices,J_all_slices[mask_all_slices>0],phase_all_slices[mask_all_slices>0],norm_all_slices[mask_all_slices>0])]
        file_map_all_slices = filename.split(".dat")[0] + "_MRF_map.pkl"
        file_map_all_slices = str.replace(file_map_all_slices,folder,results_folder)
        with open(file_map_all_slices,"wb") as file:
            pickle.dump(maps_all_slices_to_save, file)


        if return_mat:
            file_mat = filename.split(".dat")[0] + "_MRF_map.mat"
            file_mat = str.replace(file_mat,folder,results_folder)
            savemat(file_mat,dict_mat)
        
    return

toolbox.add_program("process_folder_mrf", process_folder_mrf)


if __name__ == "__main__":
    toolbox.cli()





# python script_recoInVivo_3D_machines.py build_maps --filename-volume data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_volumes_singular_allbins_registered.npy --filename-mask data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_mask.npy --filename-b1 data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_b12Dplus1_16.npy --filename-weights data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_weights.npy --dictfile mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean.dict --dictfile-light mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.22_reco4_w8_simmean.dict --optimizer-config opt_config_iterative_singular.json --file-deformation data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_deformation_map.npy --filename data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF.dat
# python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4.npy --file-model data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_vxm_model_weights.h5
# python script_recoInVivo_3D_machines.py build_volumes_singular_allbins_registered --filename-kdata data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_kdata.npy --filename-b1 data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_b12Dplus1_16.npy --filename-pca data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_virtualcoils_16.pkl --filename-weights data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_weights.npy --file-deformation-map data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_deformation_map.npy --filename-phi mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean_phi_L0_10.npy --L0 10
# python script_recoInVivo_3D_machines.py build_volumes_singular_allbins --filename-kdata data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_kdata.npy --filename-phi mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean_phi_L0_10.npy --L0 10 --n-comp 16

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
from utils_simu import *

from copy import deepcopy

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

BUMP_WINDOWS={
    "wT1":[0,200],
    "ff":[0,0.5]
}

BUMP_STD={
    "wT1":10,
    "ff":0.01
}
DEFAULT_CONFIG="config_simu.json"
DEFAULT_CONFIG_3D="config_simu_3D.json"

@machine
@set_parameter("config", type=Config, default=DEFAULT_CONFIG, description="Config File")
@set_parameter("base_folder",str,default="./3D",description="Base folder for saving results")
@set_parameter("useGPU",bool,default=True,description="use GPU for dico matching")
@set_parameter("sim_mode",["bloch","epg"],default="bloch",description="Simulation mode for ground truth images")
@set_parameter("save_volume",bool,default=False,description="save time serie of images")
def simulate(config,base_folder,useGPU,sim_mode,save_volume):

    print("###############################Phantoms used#############################################")
    print(config["file_phantom"])
    print("##############################################################################################")
    dictfiles=config["dictfiles"]
    seqfiles=config["seqfiles"]

    compression_factor=config["compression"]
    nb_simu=config["nb_simu"]

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")


    #seq=T1MRF(**sequence_config)
    
    name="Simu_{}".format(date_time)

    for num in range(nb_simu):
        print("######################################################Phantom {} out of {} ############################################################".format(num+1,nb_simu))

        map_index=np.random.randint(len(config["file_phantom"]))
        file = config["file_phantom"][map_index]
        print(file)

        m = MapFromFile(name+"_{}".format(num), file=file, rounding=True,gen_mode="loop")

        
        print("Randomizing parameter values inside each numerical phantom")
        maskROI=np.load(config["maskROI"][map_index])
            #print(maskROI.shape)
        ROI_bumped_count=config["ROI_count"]
        num_ROIs=len(np.unique(maskROI))
        bumped_ROIs=np.random.choice(num_ROIs,ROI_bumped_count)
        bumped_ROIs=np.unique(bumped_ROIs)
            #print(bumped_ROIs)        
        dico_bumps={}
        for k in BUMP_WINDOWS.keys():
            print(k)
            bump_values=np.zeros(shape=maskROI.shape)
            for roi in bumped_ROIs:
                if not (roi == 0):
                    shape=maskROI[maskROI==roi].shape
                    bump_values[maskROI==roi]=np.random.uniform(low=BUMP_WINDOWS[k][0],high=BUMP_WINDOWS[k][1])+np.random.normal(0,BUMP_STD[k],size=shape)
                    #print(bump_values[maskROI==roi])
            dico_bumps[k]=bump_values

        m.buildParamMap(dico_bumps=dico_bumps)
        if compression_factor>1:
            m.change_resolution(compression_factor)

        maskROI_for_plot=maskROI[::int(compression_factor),::int(compression_factor)]
        print(maskROI_for_plot.shape)
        maskROI_for_plot=maskROI_for_plot[m.mask>0]
        print(maskROI_for_plot.shape)
        npoint_image=m.image_size[-1]

        npoint = npoint_image*2
        image_size = (npoint_image, npoint_image)


        localfile="/"+m.name
        filename = base_folder+localfile

        filename_paramMap=filename+"_paramMap.pkl"
        filename_paramMask=filename+"_paramMask.npy"
        filename_maskROI=filename+"_ROI.npy"


        
        with open(filename_paramMap, "wb" ) as file:
            pickle.dump(m.paramMap, file)

        map_rebuilt = deepcopy(m.paramMap)
        mask = m.mask
        map_rebuilt["wT1"][map_rebuilt["ff"]>0.7]=0
        keys_simu = list(map_rebuilt.keys())
        values_simu = [makevol(map_rebuilt[k], mask > 0)[None,...] for k in keys_simu]
        map_for_sim = dict(zip(keys_simu, values_simu))

        np.save(filename_paramMask,mask)
        np.save(filename_maskROI,maskROI_for_plot)

        for key in ["ff", "wT1", "df", "attB1"]:
            file_mha = "/".join(["/".join(str.split(filename_paramMap, "/")[:-1]),
                                    "_".join(str.split(str.split(filename_paramMap, "/")[-1], ".")[:-1])]) + "_{}.mha".format(
                                                                                                                        key)
            io.write(file_mha, map_for_sim[key], tags={"spacing": [5, 1, 1]})


        for i,dictfile in enumerate(dictfiles):

            print("Processing phantom {} with dictionary {}".format(num+1,dictfile))
            dictfile_light=dictfile.replace("Invivo","Invivo_light_for_matching")
            split_for_reco=str.split(dictfile,"_reco")
            reco=float(str.split(split_for_reco[-1],"_")[0])
            print(reco)
            suffix=str.split(split_for_reco[0],"mrf_dictconf_Dico2_Invivo_overshoot_")[-1]
            print(suffix)
            file_map = filename + "_{}_MRF_map.pkl".format(suffix)
            file_volume = filename + "_{}_volumes.npy".format(suffix)

            group_size=int(str.split(str.split(split_for_reco[-1],"w")[-1],"_")[0])
            print(group_size)
            
            with open(seqfiles[i]) as f:
                sequence_config = json.load(f)

            

            if sim_mode=="epg":
                
                nrep=3
                rep=nrep-1
                sequence_config["T_recovery"]=reco*1000
                sequence_config["nrep"]=nrep
                sequence_config["rep"]=rep

                seq=T1MRFSS(**sequence_config)
                m.build_ref_images(seq)

            elif sim_mode=="bloch":
                min_TR_delay=(sequence_config["TR"][0]-sequence_config["TE"][0])/1000
                TR,FA,TE=load_sequence_file(seqfiles[i],reco,min_TR_delay)
                m.build_ref_images_bloch(TR,FA,TE)

            else:
                raise ValueError("Unknow simulation mode : {}".format(sim_mode))
            
            nb_allspokes = len(sequence_config["TE"])
            ntimesteps = int(nb_allspokes / group_size)

            radial_traj = Radial(total_nspokes=nb_allspokes, npoint=npoint)

            data = m.generate_kdata(radial_traj, useGPU=False,nthreads=1,fftw=0)
            data = np.array(data)
            data = data.reshape(nb_allspokes, -1, npoint)
            volumes_all = simulate_radial_undersampled_images(data, radial_traj, image_size, density_adj=True, useGPU=False,ntimesteps=ntimesteps,nthreads=1,fftw=0)
            
            if save_volume:
                np.save(file_volume,volumes_all)

            niter = 0

            optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=radial_traj, split=10, pca=True,
                                        threshold_pca=10, log=False, useGPU_dictsearch=useGPU, useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                        b1=None, mu="Adaptative", dens_adj=None, dictfile_light=dictfile_light,
                                        threshold_ff=0.9)  # ,kdata_init=kdata_all_channels_all_slices)#,mu_TV=0.5)#,kdata_init=data_no_noise)
            all_maps = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all, retained_timesteps=None)


            with open(file_map,"wb") as file:
                # dump information to that file
                pickle.dump(all_maps, file)
            
            
            regression_paramMaps_ROI(m.paramMap,all_maps[0][0],m.mask>0,all_maps[0][1]>0,maskROI_for_plot,adj_wT1=True,title="_".join(str.split("regROI_"+str.split(str.split(file_map,"/")[-1],".pkl")[0],".")),save=True)

            for iter in list(range(np.minimum(len(all_maps.keys()),2))):

                map_rebuilt=all_maps[iter][0]
                mask=all_maps[iter][1]

                keys_simu = list(map_rebuilt.keys())

                map_rebuilt["wT1"][map_rebuilt["ff"]>0.7]=0.0
                values_simu = [makevol(map_rebuilt[k], mask > 0)[None,...] for k in keys_simu]

                map_for_sim = dict(zip(keys_simu, values_simu))

                for key in ["ff","wT1","df","attB1"]:
                    file_mha = "/".join(["/".join(str.split(file_map,"/")[:-1]),"_".join(str.split(str.split(file_map,"/")[-1],".")[:-1])]) + "_it{}_{}.mha".format(iter,key)
                    io.write(file_mha,map_for_sim[key],tags={"spacing":[5,1,1]})


        
    
    return


@machine
@set_parameter("config", type=Config, default=DEFAULT_CONFIG_3D, description="Config File")
@set_parameter("base_folder",str,default="./3D",description="Base folder for saving results")
@set_parameter("useGPU",bool,default=True,description="use GPU for dico matching")
@set_parameter("sim_mode",["bloch","epg"],default="bloch",description="Simulation mode for ground truth images")
@set_parameter("save_volume",bool,default=False,description="save time serie of images")
@set_parameter("empty_slices",int,default=2,description="Empty slice on each side of the 3D volume")
def simulate_3D(config,base_folder,useGPU,sim_mode,save_volume,empty_slices):

    print("###############################Phantoms used#############################################")
    print(config["file_phantom"])
    print("##############################################################################################")
    dictfiles=config["dictfiles"]
    seqfiles=config["seqfiles"]

    compression_factor=config["compression"]
    nb_simu=config["nb_simu"]

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")


    name="Simu3D_{}".format(date_time)


    for num in range(nb_simu):
        print("######################################################Phantom {} out of {} ############################################################".format(num+1,nb_simu))


        file = config["file_phantom"]
        print(file)
        m = MapFromFile3D(name+"_{}".format(num), file=file, rounding=True,gen_mode="loop",nb_empty_slices=empty_slices)

        print("Randomizing parameter values inside each numerical phantom")
        maskROI=[]
        for f in config["maskROI"]:
            current_maskROI=np.load(f)
            maskROI.append(current_maskROI)
            #print(maskROI.shape)
        
        maskROI=np.array(maskROI)
        shape_maskROI_sl=maskROI.shape[1:]
        maskROI=np.vstack([np.zeros((empty_slices,)+shape_maskROI_sl),maskROI])
        maskROI=np.vstack([maskROI,np.zeros((empty_slices,)+shape_maskROI_sl)])

        print(maskROI.shape)
        
        ROI_bumped_count=config["ROI_count"]
        num_ROIs=len(np.unique(maskROI))
        bumped_ROIs=np.random.choice(num_ROIs,ROI_bumped_count)
        bumped_ROIs=np.unique(bumped_ROIs)
            #print(bumped_ROIs)        
        dico_bumps={}
        for k in BUMP_WINDOWS.keys():
            print(k)
            bump_values=np.zeros(shape=maskROI.shape)
            for roi in bumped_ROIs:
                if not (roi == 0):
                    shape=maskROI[maskROI==roi].shape
                    bump_values[maskROI==roi]=np.random.uniform(low=BUMP_WINDOWS[k][0],high=BUMP_WINDOWS[k][1])+np.random.normal(0,BUMP_STD[k],size=shape)
                    #print(bump_values[maskROI==roi])
            dico_bumps[k]=bump_values

        m.buildParamMap(dico_bumps=dico_bumps)
        if compression_factor>1:
            m.change_resolution(compression_factor)

        maskROI_for_plot=maskROI[:,::int(compression_factor),::int(compression_factor)]
        print(maskROI_for_plot.shape)
        maskROI_for_plot=maskROI_for_plot[m.mask>0]
        print(maskROI_for_plot.shape)
        npoint_image=m.image_size[-1]

        npoint = npoint_image*2
        image_size = m.image_size


        localfile="/"+m.name
        filename = base_folder+localfile

        filename_paramMap=filename+"_paramMap.pkl"
        filename_paramMask=filename+"_paramMask.npy"
        filename_maskROI=filename+"_ROI.npy"


        
        with open(filename_paramMap, "wb" ) as file:
            pickle.dump(m.paramMap, file)

        map_rebuilt = deepcopy(m.paramMap)
        mask = m.mask
        map_rebuilt["wT1"][map_rebuilt["ff"]>0.7]=0
        keys_simu = list(map_rebuilt.keys())
        values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
        map_for_sim = dict(zip(keys_simu, values_simu))

        np.save(filename_paramMask,mask)
        np.save(filename_maskROI,maskROI_for_plot)

        print(map_for_sim["wT1"].shape)

        for key in ["ff", "wT1", "df", "attB1"]:
            file_mha = "/".join(["/".join(str.split(filename_paramMap, "/")[:-1]),
                                    "_".join(str.split(str.split(filename_paramMap, "/")[-1], ".")[:-1])]) + "_{}.mha".format(
                                                                                                                        key)
            io.write(file_mha, map_for_sim[key], tags={"spacing": [5, 1, 1]})


        for i,dictfile in enumerate(dictfiles):

            print("Processing phantom {} with dictionary {}".format(num+1,dictfile))
            dictfile_light=dictfile.replace("Invivo","Invivo_light_for_matching")
            split_for_reco=str.split(dictfile,"_reco")
            reco=float(str.split(split_for_reco[-1],"_")[0])
            print(reco)
            suffix=str.split(split_for_reco[0],"mrf_dictconf_Dico2_Invivo_overshoot_")[-1]
            print(suffix)
            file_map = filename + "_{}_MRF_map.pkl".format(suffix)
            file_volume = filename + "_{}_volumes.npy".format(suffix)

            group_size=int(str.split(str.split(split_for_reco[-1],"w")[-1],"_")[0])
            print(group_size)
            
            with open(seqfiles[i]) as f:
                sequence_config = json.load(f)

            

            if sim_mode=="epg":
                
                nrep=3
                rep=nrep-1
                sequence_config["T_recovery"]=reco*1000
                sequence_config["nrep"]=nrep
                sequence_config["rep"]=rep

                seq=T1MRFSS(**sequence_config)
                m.build_ref_images(seq)

            elif sim_mode=="bloch":
                min_TR_delay=(sequence_config["TR"][0]-sequence_config["TE"][0])/1000
                TR,FA,TE=load_sequence_file(seqfiles[i],reco,min_TR_delay)
                m.build_ref_images_bloch(TR,FA,TE)

            else:
                raise ValueError("Unknow simulation mode : {}".format(sim_mode))
            
            nb_allspokes = len(sequence_config["TE"])
            ntimesteps = int(nb_allspokes / group_size)

            radial_traj = Radial3D(total_nspokes=nb_allspokes, npoint=npoint,nb_slices=m.mask.shape[0])

            data = m.generate_kdata(radial_traj, useGPU=False,nthreads=1,fftw=0)
            data = np.array(data)
            data = data.reshape(nb_allspokes, -1, npoint)
            volumes_all = simulate_radial_undersampled_images(data, radial_traj, image_size, density_adj=True, useGPU=False,ntimesteps=ntimesteps,nthreads=1,fftw=0)
            
            if save_volume:
                np.save(file_volume,volumes_all)

            niter = 0

            optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=radial_traj, split=10, pca=True,
                                        threshold_pca=10, log=False, useGPU_dictsearch=useGPU, useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                        b1=None, mu="Adaptative", dens_adj=None, dictfile_light=dictfile_light,
                                        threshold_ff=0.9)  # ,kdata_init=kdata_all_channels_all_slices)#,mu_TV=0.5)#,kdata_init=data_no_noise)
            all_maps = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all, retained_timesteps=None)


            with open(file_map,"wb") as file:
                # dump information to that file
                pickle.dump(all_maps, file)
            
            
            regression_paramMaps_ROI(m.paramMap,all_maps[0][0],m.mask>0,all_maps[0][1]>0,maskROI_for_plot,adj_wT1=True,title="_".join(str.split("regROI_"+str.split(str.split(file_map,"/")[-1],".pkl")[0],".")),save=True)

            for iter in list(range(np.minimum(len(all_maps.keys()),2))):

                map_rebuilt=all_maps[iter][0]
                mask=all_maps[iter][1]

                keys_simu = list(map_rebuilt.keys())

                map_rebuilt["wT1"][map_rebuilt["ff"]>0.7]=0.0
                values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]

                map_for_sim = dict(zip(keys_simu, values_simu))

                for key in ["ff","wT1","df","attB1"]:
                    file_mha = "/".join(["/".join(str.split(file_map,"/")[:-1]),"_".join(str.split(str.split(file_map,"/")[-1],".")[:-1])]) + "_it{}_{}.mha".format(iter,key)
                    io.write(file_mha,map_for_sim[key],tags={"spacing":[5,1,1]})


        
    
    return

toolbox.add_program("simulate", simulate)
toolbox.add_program("simulate_3D", simulate_3D)

if __name__ == "__main__":
    toolbox.cli()





# python script_recoInVivo_3D_machines.py build_maps --filename-volume data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_volumes_singular_allbins_registered.npy --filename-mask data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_mask.npy --filename-b1 data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_b12Dplus1_16.npy --filename-weights data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_weights.npy --dictfile mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean.dict --dictfile-light mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.22_reco4_w8_simmean.dict --optimizer-config opt_config_iterative_singular.json --file-deformation data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_deformation_map.npy --filename data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF.dat
# python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4.npy --file-model data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_vxm_model_weights.h5
# python script_recoInVivo_3D_machines.py build_volumes_singular_allbins_registered --filename-kdata data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_kdata.npy --filename-b1 data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_b12Dplus1_16.npy --filename-pca data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_virtualcoils_16.pkl --filename-weights data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_weights.npy --file-deformation-map data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_deformation_map.npy --filename-phi mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean_phi_L0_10.npy --L0 10
# python script_recoInVivo_3D_machines.py build_volumes_singular_allbins --filename-kdata data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_kdata.npy --filename-phi mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean_phi_L0_10.npy --L0 10 --n-comp 16
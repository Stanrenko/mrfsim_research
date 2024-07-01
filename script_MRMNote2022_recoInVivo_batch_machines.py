
from utils_simu import *
from utils_reco import *
from dictoptimizers import SimpleDictSearch
import json
from scipy.optimize import differential_evolution
import pickle
from datetime import datetime

path = r"/home/cslioussarenko/PythonRepositories"
#path = r"/Users/constantinslioussarenko/PythonGitRepositories/MyoMap"

import sys
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")

#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,BruteDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import glob
from mutools import io

from machines import machine, Toolbox, Config, set_parameter, set_output, printer, file_handler, Parameter, RejectException, get_context

DEFAULT_RUN_CONFIG=r"MRMNote2022_config.json"

#["CL1KB","CL1KF","CL1KN","MNAV","MNAW","MNAX","MNAZ","MNBC","MNBE","MNBF","MNBG","MNBH","MNBI"],
@machine
@set_parameter("run_config",type=Config,default=DEFAULT_RUN_CONFIG,description="Optimizer parameters")
def run_batch(run_config):

    folder_ROI = run_config["folder_ROI"]

    folder_results_matlab =run_config["folder_raw_data"]
    folder_results_matlab_bis=run_config["folder_raw_data_bis"]

    target_folder=run_config["target_folder"]

    ROI_folder_names = glob.glob(folder_ROI+"/MT*")
    #ROI_folder_names = glob.glob(folder_ROI+"/*")
    if "patient_names" in run_config:
        patient_names=run_config["patient_names"]
    else:
        patient_names=np.unique([str.split(str.split(p,"/")[-1],".")[0] for p in ROI_folder_names])
    #patient_names=['CL1KB','CL1KF','CL1KN','MNAV','MNAW','MNAX','MNAZ','MNBC','MNBE','MNBF','MNBG','MNBH','MNBI','MNBK','MNBL','MNBM','MNBN','MNBO','MNBP','MNBQ','MNBS','MNBU','MNBV']


    exam_types=["legs","thighs"]

    exam_code=run_config["exam_code"]

    all_results_cf={}
    all_results_matrix={}

    useGPU=True

    try:
        times_brute=list(np.load( target_folder + "/" + "times_brute.npy"))
        times_cf=list(np.load( target_folder + "/" + "times_cf.npy"))
        times_matrix=list(np.load( target_folder + "/" + "times_cf_clustering.npy"))
    except:
        times_brute = []
        times_cf = []
        times_matrix = []


    for p in patient_names:
        for exam_type in exam_types:

            print("##########################################################################################")
            print("PROCESSING {} FOR {}".format(p,exam_type))

            if exam_type=="thighs":
                image_type="CUISSES"
            elif exam_type=="legs":
                image_type = "JAMBES"
            else:
                raise ValueError("Unknown exam type")

            file_roi = folder_ROI+"/{}.{}/roi_T1/roi.mhd".format(p,exam_type)

            try:
                filename = sorted(glob.glob(folder_results_matlab+"/{}*/*_{}_*.dat".format(p,image_type)))[0]
                #folder_results = sorted(glob.glob(folder_results_matlab+"/{}*/*_{}_*[!.dat]".format(p,image_type)))[0]

                #folder_reco = sorted(glob.glob(folder_results+"/Reco[!_]*"))[-1]
            except:
                try:
                    filename = sorted(glob.glob(folder_results_matlab_bis + "/{}*/*_{}_*.dat".format(p, image_type)))[0]
                except:
                    try:
                        filename = sorted(glob.glob(folder_results_matlab_bis + "/{}*/bak/*_{}_*.dat".format(p, image_type)))[0]
                    except:
                        print("Could not load kdata or reco folder {} {}".format(p,exam_type))
                        continue


            filename_base = target_folder+"/"+"_".join(str.split(filename,"/")[-2:])


            try:
                maskROI = io.read(file_roi)
            except:
                print("Could not load ROI file for {} {}".format(p,exam_type))
                continue

            maskROI = np.moveaxis(maskROI,-1,0)
            maskROI = np.moveaxis(maskROI,-1,1)
            maskROI=np.array(maskROI)

            for j in range(maskROI.shape[0]):
                #maskROI[j]=np.flip((maskROI[j]),axis=1)
                maskROI[j]=np.rot90(maskROI[j], axes=(1, 0))

            filename_data = str.split(filename_base, ".dat")[0] + "_data.npy"

            if str.split(filename_data,"/")[-1] not in os.listdir(target_folder):
                try:
                    Parsed_File = rT.map_VBVD(filename)
                    idx_ok = rT.detect_TwixImg(Parsed_File)
                    RawData = Parsed_File[str(idx_ok)]["image"].readImage()

                except :
                    print("Could not load kdata for {} {}".format(p, exam_type))
                    continue

                data = np.squeeze(RawData)
                data=np.moveaxis(data,-1,0)
                data=np.moveaxis(data,1,-1)
                np.save(filename_data,data)
            else:
                try:
                    data=np.load(filename_data)
                except:
                    print("Could not load kdata for {} {}".format(p, exam_type))
                    continue


            ntimesteps=175
            nb_allspokes = data.shape[-2]
            npoint = data.shape[-1]
            image_size = maskROI.shape[1:]

            # Density adjustment all slices
            density = np.abs(np.linspace(-1, 1, npoint))
            kdata_all_channels_all_slices = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in data]
            kdata_all_channels_all_slices=np.array(kdata_all_channels_all_slices).reshape(data.shape)
            del data

            radial_traj=Radial(total_nspokes=nb_allspokes,npoint=npoint)

            #Coil sensi estimation for all slices
            filename_b1 = str.split(filename_base, ".dat")[0] + "_b1.npy"

            if str.split(filename_b1,"/")[-1] not in os.listdir(target_folder):
                res=16
                b1_all_slices=calculate_sensitivity_map(kdata_all_channels_all_slices,radial_traj,res,image_size)
                np.save(filename_b1,b1_all_slices)
            else:
                b1_all_slices=np.load(filename_b1)

            for sl in range(0,maskROI.shape[0]):

                filename_volume = str.split(filename_base, ".dat")[0] + "_volumes_sl{}.npy".format(sl)
                filename_mask=str.split(filename_base, ".dat")[0] + "_mask_sl{}.npy".format(sl)

                print("Processing slice {} out of {}".format(sl,maskROI.shape[0]-1))
                kdata_all_channels=kdata_all_channels_all_slices[sl,:,:,:]
                b1=b1_all_slices[sl]

                if not(str.split(filename_volume,"/")[-1] in os.listdir(target_folder)):
                    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False)
                    np.save(filename_volume,volumes_all)
                else:
                    volumes_all = np.load(filename_volume)

                if str.split(filename_mask,"/")[-1] not in os.listdir(target_folder):
                    mask=build_mask_single_image_multichannel(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False,threshold_factor=1/15)
                    np.save(filename_mask,mask)
                else:
                    mask=np.load(filename_mask)

                dictfile = "mrf175_Dico2_Invivo.dict"
                dictfile_light="mrf175_Dico2_Invivo_light_for_matching.dict"


                file_map_cf=str.split(filename_base, ".dat")[0] + "_MRF_map_cf_sl{}.pkl".format(sl)
                file_map_matrix = str.split(filename_base, ".dat")[0] + "_MRF_map_cf_clustering_sl{}.pkl".format(sl)
                file_map_brute = str.split(filename_base, ".dat")[0] + "_MRF_map_brute_sl{}.pkl".format(sl)
                file_map_brute_raw = str.split(filename_base, ".dat")[0] + "_MRF_map_brute_raw_sl{}.pkl".format(sl)


                niter = 0



                optimizer_clustering = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=radial_traj, split=2000, pca=True,
                                             threshold_pca=15, log=False, useGPU_dictsearch=useGPU, useGPU_simulation=False,
                                             gen_mode="other",dictfile_light=dictfile_light,threshold_ff=0.9,ntimesteps=ntimesteps)

                optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=radial_traj, split=100
                                             ,
                                                    pca=True,
                                                    threshold_pca=15, log=False, useGPU_dictsearch=useGPU,
                                                    useGPU_simulation=False,
                                                    gen_mode="other",ntimesteps=ntimesteps)

                optimizer_brute = BruteDictSearch(FF_list=np.arange(0,1.05,0.05),mask=mask,split=1,pca=True,threshold_pca=30,log=False,useGPU_dictsearch=useGPU,n_clusters_dico=100,pruning=0.05,ntimesteps=ntimesteps,dictfile_light=dictfile_light)
                #optimizer_brute = BruteDictSearch(FF_list=np.arange(0,1.05,0.05),mask=mask,split=1,pca=True,threshold_pca=15,log=False,useGPU_dictsearch=useGPU,n_clusters_dico=1000,pruning=0.05,ntimesteps=ntimesteps)
                #optimizer_brute_raw = BruteDictSearch(FF_list=np.arange(0,1.05,0.05),mask=mask,split=10,pca=True,threshold_pca=30,log=False,useGPU_dictsearch=True,n_clusters_dico=None,ntimesteps=ntimesteps)

                #all_maps = optimizer.search_patterns(dictfile, volumes_all)
                import pickle
                if not(str.split(file_map_cf,"/")[-1] in os.listdir(target_folder)):

                    start_time=time.time()
                    all_maps_cf=optimizer.search_patterns_test_multi(dictfile,volumes_all)
                    end_time=time.time()
                    dtime = (end_time - start_time)/mask.sum()*1000
                    print("CF Time taken per pixel for slice {} : {} ms".format(sl,dtime))

                    times_cf.append(dtime)

                    with open(file_map_cf,"wb") as file:
                        pickle.dump(all_maps_cf, file)

                else:
                    with open(file_map_cf, "rb") as file:
                        all_maps_cf = pickle.load(file)

                if not (str.split(file_map_matrix,"/")[-1] in os.listdir(target_folder)):

                    start_time = time.time()
                    all_maps_matrix = optimizer_clustering.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all)
                    end_time = time.time()
                    dtime = (end_time - start_time) / mask.sum()*1000
                    print("Matrix Time taken per pixel for slice {} : {} ms".format(sl, dtime))

                    times_matrix.append(dtime)

                    with open(file_map_matrix, "wb") as file:
                        pickle.dump(all_maps_matrix, file)

                else:
                    with open(file_map_matrix, "rb") as file:
                        all_maps_matrix = pickle.load(file)

                if not (str.split(file_map_brute,"/")[-1] in os.listdir(target_folder)):

                    start_time = time.time()
                    all_maps_brute = optimizer_brute.search_patterns_new_clustering(dictfile, volumes_all)
                    end_time = time.time()
                    dtime = (end_time - start_time) / mask.sum()*1000
                    print("Brute Time taken per pixel for slice {} : {} ms".format(sl, dtime))
                    times_brute.append(dtime)

                    with open(file_map_brute, "wb") as file:
                        pickle.dump(all_maps_brute, file)

                else:
                    with open(file_map_brute, "rb") as file:
                        all_maps_brute = pickle.load(file)

                #if not (str.split(file_map_brute_raw,"/")[-1] in os.listdir(target_folder)):
                #    print("Raw brute matching")

                #    start_time = time.time()
                #    all_maps_brute = optimizer_brute_raw.search_patterns(dictfile, volumes_all)
                #    end_time = time.time()
                #    dtime = (end_time - start_time) / mask.sum()*1000
                #    print("Brute Time raw taken per pixel for slice {} : {} ms".format(sl, dtime))
                #    #times_brute.append(dtime)



                np.save( target_folder + "/" + "times_cf.npy",np.array(times_cf))
                np.save( target_folder + "/" + "times_cf_clustering.npy",np.array(times_matrix))
                np.save( target_folder + "/" + "times_brute.npy",np.array(times_brute))

                proj_mask = mask

                maskROI_current = maskROI[sl,:,:][mask>0]



                compare_paramMaps(all_maps_brute[0][0],all_maps_cf[0][0],all_maps_brute[0][1]>0,all_maps_cf[0][1]>0,title1="{} {} Brute {}".format(p,exam_type,sl),title2="{} {} Closed-formula {}".format(p,exam_type,sl),proj_on_mask1=proj_mask>0,adj_wT1=True,save=True,fontsize=15,figsize=(40,10))
                compare_paramMaps(all_maps_brute[0][0], all_maps_matrix[0][0], all_maps_brute[0][1] > 0, all_maps_matrix[0][1] > 0,
                                  title1="{} {} Brute {}".format(p, exam_type, sl),
                                  title2="{} {} CF clustering {}".format(p, exam_type, sl), proj_on_mask1=proj_mask > 0,
                                  adj_wT1=True, save=True, fontsize=15, figsize=(40, 10))

                plt.close("all")

            # plt.figure();plt.imshow(makevol(maskROI_current,all_maps_python_current_slice[1]>0))
                # plt.figure();
                # plt.imshow(makevol(maskROI_current==20, all_maps_python_current_slice[1] > 0))

                try:
                    #df_python = metrics_paramMaps_ROI(all_maps_matlab_current_slice[0],all_maps_python_current_slice[0],all_maps_matlab_current_slice[1]>0,all_maps_python_current_slice[1]>0,maskROI=maskROI_current,adj_wT1=True,proj_on_mask1=proj_mask>0)
                    #df_python.to_csv("{} {} : Results_Comparison_Invivo slice {}".format(p,exam_type,sl))
                    regression_paramMaps_ROI(all_maps_brute[0][0],all_maps_cf[0][0],all_maps_brute[0][1]>0,all_maps_cf[0][1]>0,maskROI=maskROI_current,save=True,title="{} {}: Brute vs CF Invivo slice {}".format(p,exam_type,sl),kept_keys=["attB1","df","wT1","ff"],adj_wT1=True,fat_threshold=0.7,proj_on_mask1=proj_mask>0)
                    regression_paramMaps_ROI(all_maps_brute[0][0], all_maps_matrix[0][0], all_maps_brute[0][1] > 0, all_maps_matrix[0][1] > 0, maskROI=maskROI_current, save=True,
                                             title="{} {}: Brute vs CF Clustering Invivo slice {}".format(p, exam_type, sl),
                                             kept_keys=["attB1", "df", "wT1", "ff"], adj_wT1=True, fat_threshold=0.7,
                                             proj_on_mask1=proj_mask > 0)

                    plt.close("all")

                    results = get_ROI_values(all_maps_brute[0][0],all_maps_cf[0][0],all_maps_brute[0][1]>0,all_maps_cf[0][1]>0,maskROI=maskROI_current,kept_keys=["attB1","df","wT1","ff"],adj_wT1=True,fat_threshold=0.7,proj_on_mask1=proj_mask>0)
                    if all_results_cf=={}:
                        all_results_cf=results
                    else:
                        for k in all_results_cf.keys():
                            all_results_cf[k]=np.concatenate([results[k],all_results_cf[k]],axis=0)

                    results = get_ROI_values(all_maps_brute[0][0], all_maps_matrix[0][0], all_maps_brute[0][1] > 0, all_maps_matrix[0][1] > 0,
                                             maskROI=maskROI_current, kept_keys=["attB1", "df", "wT1", "ff"], adj_wT1=True,
                                             fat_threshold=0.7, proj_on_mask1=proj_mask > 0)
                    if all_results_matrix == {}:
                        all_results_matrix = results
                    else:
                        for k in all_results_matrix.keys():
                            all_results_matrix[k] = np.concatenate([results[k], all_results_matrix[k]], axis=0)

                    import pickle


                    file_all_results = "{}_ROI_All_results_CF.pkl".format(exam_code)
                    #
                    file = open(target_folder + "/" + file_all_results, "wb")
                    # dump information to that file
                    pickle.dump(all_results_cf, file)
                    # close the file
                    file.close()

                    file_all_results = "{}_ROI_All_results_CF_clustering.pkl".format(exam_code)
                    file = open(target_folder + "/" + file_all_results, "wb")
                    # dump information to that file
                    pickle.dump(all_results_matrix, file)
                    # close the file
                    file.close()


                except:
                    print("Could not perform the regression on ROI for {} {} {}".format(p,exam_type,sl))
                    continue

    np.save( target_folder + "/" + "times_cf.npy",np.array(times_cf))
    np.save( target_folder + "/" + "times_cf_clustering.npy",np.array(times_matrix))
    np.save( target_folder + "/" + "times_brute.npy",np.array(times_brute))


toolbox = Toolbox("script_MRMNote2022_recoInVivo_batch", description="run in vivo results for pattern matching algo comparison - MRM Note")
toolbox.add_program("run_batch", run_batch)


if __name__ == "__main__":
    toolbox.cli()
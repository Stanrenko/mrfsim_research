
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import glob
from mutools import io
from skimage.morphology import area_opening

plt.ioff()
plt.ion()



UNITS
folder_ROI = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/5_Results/MT"
#folder_ROI = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/5_Results/CLI"

#thighs / legs
folder_results_matlab =r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/3_Data/Patients"

save_volumes=True
save_maps=True

ROI_folder_names = glob.glob(folder_ROI+"/MT*")
#ROI_folder_names = glob.glob(folder_ROI+"/*")

patient_names=np.unique([str.split(str.split(p,"/")[-1],".")[0] for p in ROI_folder_names])
#patient_names=np.concatenate([["MTTX"],patient_names[4:]])

exam_types=["legs","thighs"]

# patient_names=["MTUH"]
# exam_types=["thighs"]

exam_type="legs"
p = patient_names[0]

patient_names=["MTTS"]
exam_types=["thighs"]
all_results={}
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
            folder_results = sorted(glob.glob(folder_results_matlab+"/{}*/*_{}_*[!.dat]".format(p,image_type)))[0]

            folder_reco = sorted(glob.glob(folder_results+"/Reco[!_]*"))[-1]
        except:
            print("Could not load kdata or reco folder {} {}".format(p,exam_type))
            continue

        try:
            matlab_map = folder_reco + "/MRFmaps0.mat"
            map_Matlab=MapFromFile("MapRebuiltMatlab",image_size=None,file=matlab_map,rounding=False,file_type="Result")
            map_Matlab.buildParamMap()
        except:
            print("Could not load map for {} {}".format(p,exam_type))
            continue

        try:
            maskROI = io.read(file_roi)
        except:
            print("Could not load ROI file for {} {}".format(p,exam_type))
            continue

        maskROI = np.moveaxis(maskROI,-1,0)
        maskROI = np.moveaxis(maskROI,-1,1)
        maskROI=np.array(maskROI)

        for j in range(maskROI.shape[0]):
            maskROI[j]=np.flip((maskROI[j]),axis=1)



        try:
            Parsed_File = rT.map_VBVD(filename)

            idx_ok = rT.detect_TwixImg(Parsed_File)
            #start_time = time.time()
            RawData = Parsed_File[str(idx_ok)]["image"].readImage()
            #test=Parsed_File["0"]["noise"].readImage()
            #test = np.squeeze(test)

            #elapsed_time = time.time()
            #elapsed_time = elapsed_time - start_time
            #progress_str = "Data read in %f s \n" % round(elapsed_time, 2)
            #print(progress_str)
            ## Random map simulation
        except :
            print("Could not load kdata for {} {}".format(p, exam_type))
            continue

        data = np.squeeze(RawData)
        data=np.moveaxis(data,-1,0)
        data=np.moveaxis(data,1,-1)

        nb_channels = data.shape[1]

        ntimesteps=175
        nb_allspokes = data.shape[-2]
        nspoke=int(nb_allspokes/ntimesteps)
        npoint = data.shape[-1]
        image_size = maskROI.shape[1:]

        # Density adjustment all slices
        density = np.abs(np.linspace(-1, 1, npoint))
        kdata_all_channels_all_slices = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in data]
        kdata_all_channels_all_slices=np.array(kdata_all_channels_all_slices).reshape(data.shape)

        radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint)

        #Coil sensi estimation for all slices
        res=16
        b1_all_slices=calculate_sensitivity_map(kdata_all_channels_all_slices,radial_traj,res,image_size)
        # sl=2
        # list_images = list(np.abs(b1_all_slices[sl]))
        # plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))

        # Selecting one slice

        for sl in range(0,maskROI.shape[0]):
        #for sl in [0]:
            #sl=4


            print("Processing slice {} out of {}".format(sl,maskROI.shape[0]-1))
            kdata_all_channels=kdata_all_channels_all_slices[sl,:,:,:]
            b1=b1_all_slices[sl]

            if not(str.split(filename,"/")[-1].split(".dat")[0] + "_volumes_sl_{}.npy".format(sl) in glob.glob("*")):
                volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False)
                if (save_volumes):
                    np.save(str.split(filename,"/")[-1].split(".dat")[0] + "_volumes_sl_{}.npy".format(sl),volumes_all)
            else:
                volumes_all = np.load(str.split(filename,"/")[-1].split(".dat")[0] + "_volumes_sl_{}.npy".format(sl))

            mask=build_mask_single_image_multichannel(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False,threshold_factor=1/15)

            volume_rebuilt = build_single_image_multichannel(kdata_all_channels, radial_traj,
                                                         image_size, b1=b1, density_adj=False)
            volume_rebuilt = np.flip(np.rot90(volume_rebuilt), axis=1)
            plt.figure();
            plt.imshow(np.abs(volume_rebuilt));
            plt.imshow(maskROI[sl], alpha=0.5)
            plt.title("Check ROI")

            # sl_test = 4
            # volume_rebuilt=build_single_image_multichannel(kdata_all_channels_all_slices[sl_test,:,:,:], radial_traj, image_size, b1=b1_all_slices[sl_test], density_adj=False)
            # volume_rebuilt=np.flip(np.rot90(volume_rebuilt), axis=1)
            #
            # plt.figure();
            # plt.imshow(np.abs(volume_rebuilt));
            # plt.imshow(maskROI[sl_test], alpha=0.2)
            # plt.title("Original")
            #
            # plt.figure();
            # plt.imshow(np.abs(np.abs(np.flip(volume_rebuilt,axis=1))));
            # plt.imshow(maskROI[sl_test], alpha=0.2)
            # plt.title("Flipped")

            # masks=[]
            # for j in range(maskROI.shape[0]):
            #     kdata_all_channels = kdata_all_channels_all_slices[j, :, :, :]
            #     b1 = b1_all_slices[j]
            #     masks.append(np.flip(np.rot90(build_mask_single_image_multichannel(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False,threshold_factor=1/15)),axis=1))
            #
            # animate_multiple_images(masks,map_Matlab.mask)
            # plt.figure()
            # plt.imshow(masks[0])
            # plt.figure()
            # plt.imshow(map_Matlab.mask[0])

            # plt.figure()
            # plt.imshow(mask)
            #
            # plt.figure()
            # plt.imshow(np.flip(np.rot90(map_Matlab.mask[1]),axis=1))
            #
            # error_mask = np.array(masks) - np.array(map_Matlab.mask)
            # plt.figure()
            # plt.imshow(error_mask[1])
            #
            # animate_images(error_mask)
            #
            # error_mask = np.array(masks[1:])-np.array(map_Matlab.mask[:-1])
            # plt.figure()
            # plt.imshow(error_mask[0])
            #
            # animate_images(error_mask)
            #
            # error_mask = np.array(masks[:-1]) - np.array(map_Matlab.mask[1:])
            # plt.figure()
            # plt.imshow(error_mask[0])
            #
            # error_mask = np.array(masks[2:]) - np.array(map_Matlab.mask[:-2])
            # plt.figure()
            # plt.imshow(error_mask[0])
            #
            #
            # animate_images(error_mask)

            ## Dict mapping

            #dictfile = "mrf175_SimReco2.dict"
            dictfile = "mrf175_Dico2_Invivo.dict"


            with open("mrf_sequence.json") as f:
                sequence_config = json.load(f)


            seq = T1MRF(**sequence_config)

            niter = 0
            start_time = time.time()
            optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=seq, trajectory=radial_traj, split=50, pca=True,
                                         threshold_pca=15, log=False, useGPU_dictsearch=True, useGPU_simulation=False,
                                         gen_mode="other")
            all_maps = optimizer.search_patterns(dictfile, volumes_all)
            end_time = time.time()
            print("Time taken for slice {} : {}".format(sl, end_time - start_time))


            if not(str.split(filename,"/")[-1].split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl) in glob.glob("*")):
                niter = 0
                start_time=time.time()
                optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=radial_traj,split=100,pca=True,threshold_pca=15,log=False,useGPU_dictsearch=False,useGPU_simulation=False,gen_mode="other")
                all_maps=optimizer.search_patterns(dictfile,volumes_all)
                end_time=time.time()
                print("Time taken for slice {} : {}".format(sl,end_time-start_time))
                if(save_maps):
                    import pickle

                    file_map = str.split(filename,"/")[-1].split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl)
                    file = open(file_map, "wb")
                    # dump information to that file
                    pickle.dump(all_maps, file)
                    # close the file
                    file.close()

            else:
                import pickle
                file_map = str.split(filename,"/")[-1].split(".dat")[0] + "_MRF_map_sl_{}.pkl".format(sl)
                file = open(file_map, "rb")
                all_maps = pickle.load(file)
                file.close()



            #matobj = loadmat(map_Matlab.paramDict["file"])["MRFmaps"]
            #map_wT1 = matobj["T1water_map"][0][0]

            #map_Matlab.plotParamMap("ff",sl=slice)

            all_maps_matlab_current_slice={}
            all_maps_matlab_current_slice[0]={}
            all_maps_matlab_current_slice[1]=map_Matlab.mask[sl,:,:]

            for k in map_Matlab.paramMap.keys():
                current_volume = makevol(map_Matlab.paramMap[k],map_Matlab.mask>0)[sl,:,:]
                all_maps_matlab_current_slice[0][k]=current_volume[all_maps_matlab_current_slice[1]>0]

            maps_python_current_slice=all_maps[0][0]
            mask_python_current_slice=all_maps[0][1]
            mask_python_current_slice=np.flip(np.rot90(mask_python_current_slice),axis=1)

            for k in maps_python_current_slice.keys():
                current_volume = makevol(maps_python_current_slice[k],all_maps[0][1]>0)
                current_volume = np.flip(np.rot90(current_volume),axis=1)
                maps_python_current_slice[k]=current_volume[mask_python_current_slice>0]

            all_maps_python_current_slice=(maps_python_current_slice,mask_python_current_slice)

            mask_ROI_matlab_slice = all_maps_matlab_current_slice[1]
            #Removing ROIs with non-dense dataset (aka due to improper mask reconstruction due to artefacts)
            mask_ROI_matlab_slice=area_opening(mask_ROI_matlab_slice,area_threshold=10)


            mask_ROI_python_slice = all_maps_python_current_slice[1]
            #Removing ROIs with non-dense dataset (aka due to improper mask reconstruction due to artefacts)
            mask_ROI_python_slice=area_opening(mask_ROI_python_slice,area_threshold=10)

            proj_mask = mask_ROI_matlab_slice*mask_ROI_python_slice

            maskROI_current = maskROI[sl,:,:][all_maps_matlab_current_slice[1]>0]

            plt.close("all")

            compare_paramMaps(all_maps_matlab_current_slice[0],all_maps_python_current_slice[0],all_maps_matlab_current_slice[1]>0,all_maps_python_current_slice[1]>0,title1="{} {} Matlab {}".format(p,exam_type,sl),title2="{} {} Python {}".format(p,exam_type,sl),proj_on_mask1=proj_mask>0,adj_wT1=True,save=True,fontsize=15,figsize=(40,10))

            # plt.figure();plt.imshow(makevol(maskROI_current,all_maps_python_current_slice[1]>0))
            # plt.figure();
            # plt.imshow(makevol(maskROI_current==20, all_maps_python_current_slice[1] > 0))

            try:
                #df_python = metrics_paramMaps_ROI(all_maps_matlab_current_slice[0],all_maps_python_current_slice[0],all_maps_matlab_current_slice[1]>0,all_maps_python_current_slice[1]>0,maskROI=maskROI_current,adj_wT1=True,proj_on_mask1=proj_mask>0)
                #df_python.to_csv("{} {} : Results_Comparison_Invivo slice {}".format(p,exam_type,sl))
                #regression_paramMaps_ROI(all_maps_matlab_current_slice[0],all_maps_python_current_slice[0],all_maps_matlab_current_slice[1]>0,all_maps_python_current_slice[1]>0,maskROI=maskROI_current,save=True,title="{} {}: Python vs Matlab Invivo slice {}".format(p,exam_type,sl),kept_keys=["attB1","df","wT1","ff"],adj_wT1=True,fat_threshold=0.7,proj_on_mask1=proj_mask>0)
                results = get_ROI_values(all_maps_matlab_current_slice[0],all_maps_python_current_slice[0],all_maps_matlab_current_slice[1]>0,all_maps_python_current_slice[1]>0,maskROI=maskROI_current,kept_keys=["attB1","df","wT1","ff"],adj_wT1=False,fat_threshold=0.7,proj_on_mask1=proj_mask>0)
                if all_results=={}:
                    all_results=results
                else:
                    for k in all_results.keys():
                        all_results[k]=np.concatenate([results[k],all_results[k]],axis=0)


            except:
                print("Could not perform the regression on ROI for {} {} {}".format(p,exam_type,sl))
                continue

import pickle
#
file_all_results = "CL_ROI_All_results_no_wT1adj.pkl"
#file_all_results = "MT_ROI_All_results_no_wT1adj.pkl"
#
file = open(file_all_results, "wb")
# dump information to that file
pickle.dump(all_results, file)
# close the file
file.close()

import pickle
import pandas as pd

file_all_results_CL = "CL_ROI_All_results.pkl"
file_all_results_MT = "MT_ROI_All_results.pkl"
file_all_results_CL = "CL_ROI_All_results_no_wT1adj.pkl"
file_all_results_MT = "MT_ROI_All_results_no_wT1adj.pkl"

file=open(file_all_results_CL,"rb")
all_results_CL = pickle.load(file)
file.close()


file=open(file_all_results_MT,"rb")
all_results_MT = pickle.load(file)
file.close()

df_combined = pd.DataFrame(columns=["ff","wT1","group"])

df_MT =  pd.DataFrame(columns=["ff","wT1","group"])
df_MT["ff"]=all_results_MT["ff"][all_results_MT["ff"][:,0]<0.7,0]
df_MT["wT1"]=all_results_MT["wT1"][all_results_MT["ff"][:,0]<0.7,0]
df_MT["group"]="Control"

df_CL =  pd.DataFrame(columns=["ff","wT1","group"])
df_CL["ff"]=all_results_CL["ff"][all_results_CL["ff"][:,0]<0.7,0]
df_CL["wT1"]=all_results_CL["wT1"][all_results_CL["ff"][:,0]<0.7,0]
df_CL["group"]="NMD"

df_combined = df_combined.append(df_CL)
df_combined = df_combined.append(df_MT)


import seaborn as sns
g=sns.pairplot(df_combined,diag_kind="kde",hue="group",plot_kws={'alpha':0.01})
g.fig.suptitle("Ref Method")


df_combined = pd.DataFrame(columns=["ff","wT1","group"])

df_MT =  pd.DataFrame(columns=["ff","wT1","group"])
df_MT["ff"]=all_results_MT["ff"][all_results_MT["ff"][:,1]<0.7,1]
df_MT["wT1"]=all_results_MT["wT1"][all_results_MT["ff"][:,1]<0.7,1]
df_MT["group"]="Control"

df_CL =  pd.DataFrame(columns=["ff","wT1","group"])
df_CL["ff"]=all_results_CL["ff"][all_results_CL["ff"][:,1]<0.7,1]
df_CL["wT1"]=all_results_CL["wT1"][all_results_CL["ff"][:,1]<0.7,1]
df_CL["group"]="NMD"

df_combined = df_combined.append(df_CL)
df_combined = df_combined.append(df_MT)


import seaborn as sns
g=sns.pairplot(df_combined,diag_kind="kde",hue="group",plot_kws={'alpha':0.01})
g.fig.suptitle("New Method")


import seaborn as sns
sns.pairplot(pd.DataFrame(all_results["ff"],columns=["ff ref","ff new"]),diag_kind="kde")

sns.histplot(pd.DataFrame(all_results["ff"],columns=["ff ref method","ff new method"])).set_title("Histogram of fat fraction over all NMD patients")

process_ROI_values(all_results,title="Comparison new vs ref all ROIs on Control patients",save=True,units=UNITS)
df_metrics_all=metrics_ROI_values(all_results,units=UNITS,name="All Control patients")
df_metrics_all.to_csv("MT_ROI_All_results.csv")

import glob
import pandas as pd
import numpy as np
list_exams=glob.glob("./figures/[MT]*Python vs Matlab Invivo slice*.png")
count_exams = len(list_exams)
df_exams=pd.DataFrame(columns=["Patient","Exam Type","Slice"])
for e in list_exams:
    ex = e.split("/")[-1]
    ex = ex.split(" ")
    patient = ex[0]
    exam_type = ex[1][:-1]
    sl = ex[-1][0]
    df_exams=df_exams.append(pd.DataFrame(np.array([patient,exam_type,sl]).reshape(1,-1),columns=["Patient","Exam Type","Slice"]))


df_exams.groupby("Exam Type").count()
df_exams.groupby("Patient").count()

print("Number of Patients :{}".format(df_exams.Patient.unique().shape[0]))

df_all = pd.DataFrame()
r=df_exams.iloc[0]
p=r["Patient"]
exam_type=r["Exam Type"]
sl = r["Slice"]
df=pd.read_csv("{} {} : Results_Comparison_Invivo slice {}".format(p,exam_type,sl),index_col=0)

for i in range(df_exams.shape[0]):
    r=df_exams.iloc[i]
    p=r["Patient"]
    exam_type=r["Exam Type"]
    sl = r["Slice"]
    df=pd.read_csv("{} {} : Results_Comparison_Invivo slice {}".format(p,exam_type,sl),index_col=0)
    if df_all.empty:
        df_all = df.copy()
    else:
        df_all=pd.merge(df,df_all,how="inner",left_index=True,right_index=True)

import matplotlib.pyplot as plt

print(df_all.index)

metric = "mean RMSE"
#metric = "R2"
param = "attB1"
plt.figure()
plt.hist(df_all.loc[metric+" "+param,:])
plt.title("Histogram of {} new vs ref".format(metric+" "+param))


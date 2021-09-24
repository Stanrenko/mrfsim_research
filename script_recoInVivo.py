
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time

filename="./data/InVivo/meas_MID00060_FID24042_JAMBES_raFin_CLI.dat"
filename="./data/InVivo/meas_MID00094_FID24076_JAMBES_raFin_CLI.dat"
#CS
filename="./data/InVivo/meas_MID00315_FID33126_JAMBES_raFin_CLI.dat"
filename="./data/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI.dat"

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
data=np.moveaxis(data,-1,0)
data=np.moveaxis(data,1,-1)

nb_channels = data.shape[1]

ntimesteps=175
nb_allspokes = data.shape[-2]
nspoke=int(nb_allspokes/ntimesteps)
npoint = data.shape[-1]
image_size = (256,256)

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
slice=2

kdata_all_channels=kdata_all_channels_all_slices[slice,:,:,:]
b1=b1_all_slices[slice]

##volumes for slice taking into account coil sensi

volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False)

##MASK

mask=build_mask_single_image_multichannel(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False,threshold_factor=1/15)

## Dict mapping

dictfile = "mrf175_SimReco2.dict"
dictfile = "mrf175_Dico2_Invivo.dict"


with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

niter = 0

optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=radial_traj,split=500,pca=True,threshold_pca=15,log=False,useGPU_dictsearch=False,useGPU_simulation=False,gen_mode="other")
all_maps=optimizer.search_patterns(dictfile,volumes_all)

iter=0
map_rebuilt=all_maps[iter][0]
mask=all_maps[iter][1]

keys_simu = list(map_rebuilt.keys())
values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
map_for_sim = dict(zip(keys_simu, values_simu))

map_Python = MapFromDict("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
map_Python.buildParamMap()

map_Python.plotParamMap()
map_Python.plotParamMap("ff")
map_Python.plotParamMap("wT1")

import pickle

file_map = filename.split(".dat")[0]+"_MRF_map.pkl"
file = open( file_map, "wb" )
    # dump information to that file
pickle.dump(all_maps, file)
    # close the file
file.close()

#Matlab
file_matlab = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/3_Comp_Matlab/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI/Reco8/MRFmaps0.mat"
map_Matlab=MapFromFile("MapRebuiltMatlab",image_size=(5,256,256),file=file_matlab,rounding=False,file_type="Result")
map_Matlab.buildParamMap()

map_Matlab.plotParamMap("ff",sl=slice)

all_maps_matlab_current_slice={}
all_maps_matlab_current_slice[0]={}
all_maps_matlab_current_slice[1]=map_Matlab.mask[slice,:,:]

for k in map_Matlab.paramMap.keys():
    current_volume = makevol(map_Matlab.paramMap[k],map_Matlab.mask>0)[slice,:,:]
    all_maps_matlab_current_slice[0][k]=current_volume[all_maps_matlab_current_slice[1]>0]

maps_python_current_slice=all_maps[0][0]
mask_python_current_slice=all_maps[0][1]
mask_python_current_slice=np.flip(np.rot90(mask_python_current_slice),axis=1)

for k in maps_python_current_slice.keys():
    current_volume = makevol(maps_python_current_slice[k],all_maps[0][1]>0)
    plt.figure()
    plt.imshow(current_volume)
    current_volume = np.flip(np.rot90(current_volume),axis=1)
    plt.figure()
    plt.imshow(current_volume)

    plt.figure()

    maps_python_current_slice[k]=current_volume[mask_python_current_slice>0]



all_maps_python_current_slice=(maps_python_current_slice,mask_python_current_slice)


compare_paramMaps(all_maps_matlab_current_slice[0],all_maps_python_current_slice[0],all_maps_matlab_current_slice[1]>0,all_maps_python_current_slice[1]>0,title1="Matlab",title2="Python",proj_on_mask1=True,adj_wT1=True,save=True)

maskROI=buildROImask(all_maps_python_current_slice[0],max_clusters=10)

df = metrics_paramMaps_ROI(all_maps_python_current_slice[0],all_maps_matlab_current_slice[0],all_maps_python_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,maskROI=maskROI,adj_wT1=True,proj_on_mask1=True)

regression_paramMaps_ROI(all_maps_python_current_slice[0],all_maps_matlab_current_slice[0],all_maps_python_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,maskROI=maskROI)

#regression_paramMaps(all_maps_python_current_slice[0],all_maps_matlab_current_slice[0],all_maps_python_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,mode="Boxplot")

# Check Dict
dict_conf = "mrf_dictconf_SimReco2.json"
file=loadmat(r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/2_Codes_info/Matlab/MRF_reco_linux/Dictionaries/Config_Dico_2D.mat")

import h5py
f = h5py.File(r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/2_Codes_info/Matlab/MRF_reco_linux/Dictionaries/Config_Dico_2D.mat","r")
paramsH5py_Matlab=f.get("paramDico")

paramMatlab = {}
for k in paramDico_Matlab.keys():
    paramMatlab[k]=np.array(paramsH5py_Matlab.get(k))

paramDico_Matlab ={}

paramDico_Matlab["water_T1"] =list(np.array(f[paramMatlab["T1"][0,0]]).flatten())
paramDico_Matlab["water_T2"] =list(np.array(f[paramMatlab["T2"][0,0]]).flatten())
paramDico_Matlab["fat_T1"] =list(np.array(f[paramMatlab["T1"][1,0]]).flatten())
paramDico_Matlab["fat_T2"] =list(np.array(f[paramMatlab["T2"][1,0]]).flatten())
paramDico_Matlab["ff"] = list(paramMatlab["FF"].flatten())
paramDico_Matlab["B1_att"] = list(paramMatlab["FA"].flatten())
paramDico_Matlab["delta_freqs"] = list(paramMatlab["Df"].flatten())
paramDico_Matlab["fat_amp"] = list(paramMatlab["FatAmp"].flatten())
paramDico_Matlab["fat_cshift"] = list(paramMatlab["FatShift"].flatten())

with open(dict_conf) as file:
    paramDico_Python = json.load(file)

k = "water_T1"

print(np.max(np.abs(np.array(paramDico_Matlab[k])-np.array(paramDico_Python[k]))))

print(paramDico_Matlab[k])
print(paramDico_Python[k])


print(len(paramDico_Matlab[k]))
print(len(paramDico_Python[k]))

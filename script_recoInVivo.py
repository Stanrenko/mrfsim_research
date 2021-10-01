
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
if (save_volume):
    np.save(filename.split(".dat")[0] + "_volumes.npy",volumes_all)

##MASK

mask=build_mask_single_image_multichannel(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False,threshold_factor=1/15)

## Dict mapping

dictfile = "mrf175_SimReco2.dict"
dictfile = "mrf175_Dico2_Invivo.dict"


with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)


load_map=True
save_map=False

if not(load_map):
    niter = 0

    optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=seq,trajectory=radial_traj,split=250,pca=True,threshold_pca=15,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other")
    all_maps=optimizer.search_patterns(dictfile,volumes_all)

    if(save_map):
        import pickle

        file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
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

iter=0
map_rebuilt=all_maps[iter][0]
mask=all_maps[iter][1]

keys_simu = list(map_rebuilt.keys())
values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
map_for_sim = dict(zip(keys_simu, values_simu))

map_Python = MapFromDict("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
map_Python.buildParamMap()

#map_Python.plotParamMap()
#map_Python.plotParamMap("ff")
map_Python.plotParamMap("df")
map_Python.build_ref_images(seq=seq)
rebuilt_image_series = map_Python.images_series
rebuilt_image_series= [np.mean(gp, axis=0) for gp in groupby(rebuilt_image_series, 8)]
rebuilt_image_series=np.array(rebuilt_image_series)


#ani=animate_images(rebuilt_image_series)
load_volume=True
if load_volume:
    volumes_all=np.load(filename.split(".dat")[0] + "_volumes.npy")

plt.figure()
pixel=(103,76)
metric=np.angle
signal_orig = metric(volumes_all[:,pixel[0],pixel[1]])
signal_orig=signal_orig/np.std(signal_orig)
signal_rebuilt = metric(rebuilt_image_series[:,pixel[0],pixel[1]])
signal_rebuilt=signal_rebuilt/np.std(signal_rebuilt)
plt.plot(signal_orig,label="Original")
plt.plot(signal_rebuilt,label="Python")
plt.legend()





######################################################################################################################################################
#Comp volume Matlab vs Python
volumes_python=np.load(filename.split(".dat")[0] + "_volumes.npy")

import h5py
folder_matlab="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/3_Comp_Matlab/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI/Reco8"

f = h5py.File(folder_matlab+"/ImgSeries.mat","r")
volumes_matlab=np.array(f.get("Img"))

slice=2
volumes_matlab_slice=volumes_matlab[slice,:,:,:].view("complex")
volumes_matlab_slice=np.rot90(np.flip(volumes_matlab_slice,axis=1),axes=(1,2))

plt.figure()
pixel=(103,76)
metric=np.real
signal_orig = metric(volumes_python[:,pixel[0],pixel[1]])
signal_orig=signal_orig/np.std(signal_orig)
signal_rebuilt = metric(volumes_matlab_slice[:,pixel[0],pixel[1]])
signal_rebuilt=signal_rebuilt/np.std(signal_rebuilt)
plt.plot(signal_orig,label="Python signal")
plt.plot(signal_rebuilt,label="Matlab signal")
plt.legend()


#######################################################################################################################################################
#Comp matlab vs Python
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import pickle

image_size=(256,256)

filename="./data/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI.dat"
file_map = filename.split(".dat")[0]+"_MRF_map.pkl"
file = open( file_map, "rb" )
all_maps=pickle.load(file)

slice=2


#Matlab
file_matlab = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/3_Comp_Matlab/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI/Reco8/MRFmaps0.mat"
map_Matlab=MapFromFile("MapRebuiltMatlab",image_size=(5,448,224),file=file_matlab,rounding=False,file_type="Result")
map_Matlab.buildParamMap()

matobj = loadmat(map_Matlab.paramDict["file"])["MRFmaps"]
map_wT1 = matobj["T1water_map"][0][0]

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
    current_volume = np.flip(np.rot90(current_volume),axis=1)
    maps_python_current_slice[k]=current_volume[mask_python_current_slice>0]

map_df_python =  makevol(maps_python_current_slice["df"],mask_python_current_slice>0)
map_df_matlab = makevol(all_maps_matlab_current_slice[0]["df"],all_maps_matlab_current_slice[1]>0)

map_df_python=np.rot90(np.flip(map_df_python,axis=0))
map_df_matlab=np.rot90(np.flip(map_df_matlab,axis=0))
error_df = np.abs(map_df_python-map_df_matlab)
max_diff_df = np.unravel_index(np.argmax(error_df),image_size)
plt.figure()
plt.imshow(error_df)
plt.colorbar()

all_maps_python_current_slice=(maps_python_current_slice,mask_python_current_slice)


compare_paramMaps(all_maps_matlab_current_slice[0],all_maps_python_current_slice[0],all_maps_matlab_current_slice[1]>0,all_maps_python_current_slice[1]>0,title1="Matlab",title2="Python",proj_on_mask1=True,adj_wT1=True,save=True)

maskROI=buildROImask(all_maps_python_current_slice[0],max_clusters=10)

from mutools import io
file_ROI = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/3_Comp_Matlab/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI/MRF_CS_Cuisses_ROI.mha"
maskROI = io.read(file_ROI)
maskROI = np.moveaxis(maskROI,-1,0)
maskROI = np.moveaxis(maskROI,-1,1)
maskROI=np.array(maskROI)
for j in range(maskROI.shape[0]):
    maskROI[j]=np.flip((maskROI[j]),axis=1)

maskROI = maskROI[slice,:,:][all_maps_python_current_slice[1]>0]


df = metrics_paramMaps_ROI(all_maps_python_current_slice[0],all_maps_matlab_current_slice[0],all_maps_python_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,maskROI=maskROI,adj_wT1=True,proj_on_mask1=True)
df.to_csv("Results_Comparison_Invivo")
regression_paramMaps_ROI(all_maps_python_current_slice[0],all_maps_matlab_current_slice[0],all_maps_python_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,maskROI=maskROI,save=True,title="Python vs Matlab Invivo slice {}".format(slice),kept_keys=["attB1","df","wT1","ff"],adj_wT1=True,fat_threshold=0.7)

from copy import deepcopy

map_python_rounded_ff=deepcopy(all_maps_python_current_slice[0])
map_matlab_rounded_ff=deepcopy(all_maps_matlab_current_slice[0])
map_python_rounded_ff["ff"]=np.round(map_python_rounded_ff["ff"],2)
map_matlab_rounded_ff["ff"]=np.round(map_matlab_rounded_ff["ff"],2)


regression_paramMaps(map_python_rounded_ff,map_matlab_rounded_ff,all_maps_python_current_slice[1]>0,all_maps_matlab_current_slice[1]>0,save=True,mode="Boxplot",fontsize=5)


#Check Dixon

dixon_folder = r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/3_Comp_Matlab/InVivo/meas_MID00333_FID33144_CUISSES_raFin_CLI/Dixon"
dixonmap_file = "b0map.mha"
dixonmask_file = "mask.mha"

from mutools import io
dixonmap = io.read(dixon_folder+"//"+dixonmap_file)
dixonmask = io.read(dixon_folder+"//"+dixonmask_file)

plt.close("all")




sl_dix=32
image_dixon = np.flip(np.rot90(np.array(dixonmap)[:,:,sl_dix]))
image_python =  makevol(maps_python_current_slice["df"],mask_python_current_slice>0)
image_matlab = makevol(all_maps_matlab_current_slice[0]["df"],all_maps_matlab_current_slice[1]>0)

center_image_y = int(image_dixon.shape[1]/2)
resol_y = int(3/4*image_python.shape[1])
image_dixon = image_dixon[:,center_image_y-resol_y:center_image_y+resol_y]


center_image_x =  int(image_python.shape[0]/2)
resol_x = int(image_dixon.shape[0]/2)
image_python = image_python[center_image_x-resol_x:center_image_x+resol_x,:]
image_matlab = image_matlab[center_image_x-resol_x:center_image_x+resol_x,:]

fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(10,30))
ax1.imshow(image_python)
ax1.set_title("Python df")
ax2.imshow(image_dixon)
ax2.set_title("Dixon df")
ax3.imshow(image_matlab)
ax3.set_title("Matlab df")

plt.figure()
plt.imshow()
plt.colorbar()

# Check Dict
dict_conf = "mrf_dictconf_Dico2_Invivo.json"
import h5py
f = h5py.File(r"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/2_Codes_info/Matlab/MRF_reco_linux/Dictionaries/Config_Dico_2D.mat","r")
paramsH5py_Matlab=f.get("paramDico")

paramMatlab = {}
for k in paramsH5py_Matlab.keys():
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

k = "fat_cshift"

print(np.max(np.abs(np.array(paramDico_Matlab[k])-np.array(paramDico_Python[k]))))

print(paramDico_Matlab[k])
print(paramDico_Python[k])


print(len(paramDico_Matlab[k]))
print(len(paramDico_Python[k]))

dict_conf = "mrf_dictconf_SimReco2.json"
dict_conf = "mrf_dictconf_Dico2_Invivo.json"
with open(dict_conf,"rb") as file:
    dico_conf=json.load(file)

dico_count={}
for k in dico_conf.keys():
    try:
        dico_count[k]=(len(dico_conf[k]),np.min(dico_conf[k]),np.max(dico_conf[k]))
    except:
        dico_count[k] = (1,dico_conf[k],dico_conf[k])

with open("mrf_sequence.json") as file:
    seq_conf=json.load(file)

seq_count={}
for k in seq_conf.keys():
    try:
        seq_count[k]=(len(seq_conf[k]),np.min(seq_conf[k]),np.max(seq_conf[k]))
    except:
        seq_count[k] = (1,seq_conf[k],seq_conf[k])
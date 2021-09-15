
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

mask=build_mask_single_image_multichannel(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False)

## Dict mapping

dictfile = "mrf175_SimReco2.dict"


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

m = MapFromDict("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
m.buildParamMap()

m.plotParamMap()
m.plotParamMap("ff")
m.plotParamMap("wT1")


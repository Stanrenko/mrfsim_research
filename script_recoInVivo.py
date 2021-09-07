
#import matplotlib
#matplotlib.use("TkAgg")
import numpy as np

from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
from utils_mrf import radial_golden_angle_traj,animate_images,animate_multiple_images,compare_patterns,translation_breathing,find_klargest_freq,SearchMrf,basicDictSearch,compare_paramMaps,regression_paramMaps,voronoi_volumes,transform_py_map
import json
from finufft import nufft1d1,nufft1d2,nufft2d2,nufft2d1
import imp
import re
import readTwix as rT
import time
import TwixObject

filename="./data/InVivo/meas_MID00060_FID24042_JAMBES_raFin_CLI.dat"
filename="./data/InVivo/meas_MID00094_FID24076_JAMBES_raFin_CLI.dat"


Parsed_File = rT.map_VBVD(filename)

idx_ok = rT.detect_TwixImg(Parsed_File)
start_time = time.time()
RawData = Parsed_File[str(idx_ok)]["image"].readImage()
test=Parsed_File["0"]["noise"].readImage()
test = np.squeeze(test)

elapsed_time = time.time()
elapsed_time = elapsed_time - start_time
progress_str = "Data read in %f s \n" % round(elapsed_time, 2)
print(progress_str)
## Random map simulation

data = np.squeeze(RawData)
data=np.moveaxis(data,-1,0)
data=np.moveaxis(data,1,-1)

slice=2
kdata_all_channels=data[slice,:,:,:]

nb_channels = kdata_all_channels.shape[0]

ch=3
plt.plot(np.abs(kdata_all_channels)[ch,:,:].T)

index=np.argwhere(np.abs(kdata_all_channels)[ch,:,:].flatten()>1e-3)
len(np.unique(np.squeeze(np.unravel_index(index,kdata_all_channels.shape[1:])[1])))

(np.abs(kdata_all_channels)[ch,:,:].flatten()>1e-3).sum()

list_channels=[]
for c in range(nb_channels):
    index = np.argwhere(np.abs(kdata_all_channels)[c, :, :].flatten() > 1e-3)
    if np.max(np.abs(np.abs(kdata_all_channels)[c,:,:].flatten()))<100:
        list_channels.append(c)

#### Rebuilding the map from undersampled images
ntimesteps=175
nspoke=int(kdata.shape[0]/ntimesteps)
npoint = kdata.shape[1]
image_size = (256,256)


radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint)

volumes_all_channels = np.zeros((nb_channels,ntimesteps)+image_size)


traj_all=radial_traj.get_traj().reshape(-1,2)
kdata_all = np.array(kdata_all_channels[ch].flatten(),dtype=np.complex128)
volume_rebuilt = finufft.nufft2d1(traj_all[:,0], traj_all[:,1], kdata_all, image_size)
plt.imshow(np.abs(volume_rebuilt))

plt.imshow(np.abs(kdata_all))


for i in tqdm(range(nb_channels)):
    volumes_all_channels[i]=simulate_radial_undersampled_images(kdata_all_channels[i],radial_traj,image_size,density_adj=True,useGPU=True)



volumes = np.sqrt(np.sum(np.abs(volumes_all_channels)**2,axis=0))

t=10
plt.imshow(np.abs(volumes_all_channels[4,t,:,:]))
plt.imshow(np.abs(volumes[t]))

## Dict mapping

dictfile = "mrf175_SimReco2.dict"


with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

niter = 0

mask = build_mask_single_image(kdata,radial_traj,image_size,useGPU=True)

optimizer = SimpleDictSearch(mask=mask,niter=0,seq=seq,trajectory=radial_traj,split=500,pca=True,threshold_pca=15,log=False,useAdjPred=False,useGPU_dictsearch=False,useGPU_simulation=True)
all_maps=optimizer.search_patterns(dictfile,volumes)

## Parallel reconstruction
with open("./"+str.split(filename,".")[1]+".npy","wb") as file:
    np.save(file,data)

slice=3
kdata_all_channels=data[slice,:,:,:]

nb_channels=kdata_all_channels.shape[0]

ch = 4

plt.plot(np.abs(kdata_all_channels[ch,0,:].T))

#### Rebuilding the map from undersampled images
ntimesteps=175
nspoke=int(kdata_all_channels.shape[1]/ntimesteps)
npoint = kdata_all_channels.shape[-1]
image_size=(256,256)

radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint)
test_volumes = simulate_radial_undersampled_images(kdata_all_channels[ch],radial_traj,image_size,density_adj=True,useGPU=False)
plt.imshow(np.abs(test_volumes[10]))


density_adj=True
if density_adj:
    density = np.abs(np.linspace(-1, 1, npoint))
    kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata_all_channels[ch]]

kdata=np.array(kdata).flatten()
traj = radial_traj.get_traj().reshape(-1,2)
test_volumes_all =  finufft.nufft2d1(traj[:,0], traj[:,1], kdata, image_size)
plt.imshow(np.abs(test_volumes_all))


volumes_all_channels=np.zeros((nb_channels,ntimesteps,)+image_size)

density = np.abs(np.linspace(-1, 1, npoint))
density=np.expand_dims(density,axis=(0,1))
kdata_all_channels_normalized=kdata_all_channels*density


traj=radial_traj.get_traj_for_reconstruction()
kdata_all_channels_normalized=kdata_all_channels_normalized.reshape((nb_channels,traj.shape[0],-1))
kdata_all_channels_normalized=np.moveaxis(kdata_all_channels_normalized,1,0)


images_series_rebuilt = [
                finufft.nufft2d1(t[:, 0], t[:, 1], s, image_size)
                for t, s in zip(traj, kdata_all_channels_normalized)
            ]

images_series_rebuilt=np.moveaxis(np.array(images_series_rebuilt),0,1)

volumes=np.sqrt(np.sum(np.abs(images_series_rebuilt)**2,axis=0))

ani=animate_images(volumes)
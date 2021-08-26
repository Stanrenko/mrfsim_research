
#import matplotlib
#matplotlib.use("TkAgg")

from mrfsim import T1MRF
from image_series import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
from movements import *
#from dictoptimizers import *
import glob
from tqdm import tqdm
import pickle
from scipy.io import savemat
from Transformers import PCAComplex


## Random map simulation

dictfile = "./mrf175_SimReco2_mid_point.dict"
dictfile = "./mrf175_SimReco2_window_1.dict"
dictfile = "./mrf175_SimReco2.dict"
#dictfile = "mrf175_CS.dict"

with open("./mrf_sequence.json") as f:
    sequence_config = json.load(f)

seq = T1MRF(**sequence_config)


size=(256,256)
useGPU_simulation=False
useGPU_dictsearch=False

load_maps=False
save_maps = False

load=True

type="KneePhantom"

ph_num=1

print("##################### {} : PHANTOM {} #########################".format(type,ph_num))
file_matlab_paramMap = "./data/{}/Phantom{}/paramMap.mat".format(type,ph_num)

###### Building Map
m = MapFromFile("{}{}".format(type,ph_num), image_size=size, file=file_matlab_paramMap, rounding=True,gen_mode="other")
m.buildParamMap()

##### Simulating Ref Images
m.build_ref_images(seq)

#### Rebuilding the map from undersampled images
ntimesteps=175
nspoke=8
npoint = 2*m.images_series.shape[1]

radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint)

if not(load):
    kdata = m.generate_kdata(radial_traj,useGPU=useGPU_simulation)
    with open("kdata_forLowRank_{}.pkl".format(m.name), "wb" ) as file:
        pickle.dump(kdata, file)

else:
    kdata = pickle.load( open( "kdata_no_mvt_sl{}us{}_{}.pkl".format(nb_total_slices,undersampling_factor,m.name), "rb" ) )



FF_list = list(np.arange(0.,1.05,0.05))

keys,values=read_mrf_dict(dictfile ,FF_list ,aggregate_components=True)

threshold_pca=15
pca_signal = PCAComplex(n_components_=threshold_pca)
pca_signal.fit(values)

V = pca_signal.components_




trajectory=radial_traj
traj=trajectory.get_traj_for_reconstruction()

# npoint = trajectory.paramDict["npoint"]
# nspoke = trajectory.paramDict["nspoke"]
# dtheta = np.pi / nspoke

if not(len(kdata)==len(traj)):
    kdata=np.array(kdata).reshape(len(traj),-1)

F = np.array(kdata).T
T = np.array(traj).T
m.image_size
x=np.arange(-int(m.image_size[0]/2),int(m.image_size[0]/2),1.0)#+0.5
y=np.arange(-int(m.image_size[1]/2),int(m.image_size[1]/2),1.0)#+0.5
X,Y = np.meshgrid(x,y)
X = X.reshape(1,-1)
Y = Y.reshape(1,-1)


#def J(U):



volumes = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=useGPU_simulation)

mask = build_mask_single_image(kdata,radial_traj,m.image_size)

# savemat("kdata_python.mat",{"KData":np.array(kdata)})
# savemat("images_ideal_python.mat", {"ImgIdeal": np.array(m.images_series)})
# savemat("images_rebuilt.mat", {"Img": np.array(volumes)})


# ani=animate_images([np.mean(gp, axis=0) for gp in groupby(m.images_series, nspoke)],cmap="gray")
# ani = animate_images(volumes, cmap="gray")

ani1,ani2 =animate_multiple_images([np.mean(gp, axis=0) for gp in groupby(m.images_series, nspoke)],volumes,cmap="gray")


# kdata_noGPU = m.generate_kdata(radial_traj, useGPU=False)
# volumes_noGPU = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=False)

optimizer = SimpleDictSearch(mask=mask,niter=0,seq=seq,trajectory=radial_traj,split=500,pca=True,threshold_pca=15,log=False,useAdjPred=False,useGPU_dictsearch=False,useGPU_simulation=False)
all_maps_adj=optimizer.search_patterns(dictfile,volumes)

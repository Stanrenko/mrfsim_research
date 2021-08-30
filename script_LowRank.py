
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
from dictoptimizers import SimpleDictSearch
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
    kdata = pickle.load( open( "kdata_forLowRank_{}.pkl".format(m.name), "rb" ) )



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

# F = np.array(kdata).T
# T = np.array(traj).T
# m.image_size
# x=np.arange(-int(m.image_size[0]/2),int(m.image_size[0]/2),1.0)#+0.5
# y=np.arange(-int(m.image_size[1]/2),int(m.image_size[1]/2),1.0)#+0.5
# X,Y = np.meshgrid(x,y)
# X = X.reshape(1,-1)
# Y = Y.reshape(1,-1)



#def J(U):



volumes = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=useGPU_simulation)

mask = build_mask_single_image(kdata,radial_traj,m.image_size)

# savemat("kdata_python.mat",{"KData":np.array(kdata)})
# savemat("images_ideal_python.mat", {"ImgIdeal": np.array(m.images_series)})
# savemat("images_rebuilt.mat", {"Img": np.array(volumes)})


# ani=animate_images([np.mean(gp, axis=0) for gp in groupby(m.images_series, nspoke)],cmap="gray")
# ani = animate_images(volumes, cmap="gray")

# ani1,ani2 =animate_multiple_images([np.mean(gp, axis=0) for gp in groupby(m.images_series, nspoke)],volumes,cmap="gray")
#

# kdata_noGPU = m.generate_kdata(radial_traj, useGPU=False)
# volumes_noGPU = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=False)

#

optimizer = SimpleDictSearch(mask=mask,niter=0,seq=seq,trajectory=radial_traj,split=500,pca=True,threshold_pca=15,log=False,useAdjPred=False,useGPU_dictsearch=False,useGPU_simulation=False)
all_maps_adj=optimizer.search_patterns(dictfile,volumes)

map_rebuilt=all_maps_adj[0][0]

keys_simu = list(map_rebuilt.keys())
values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
map_for_sim = dict(zip(keys_simu, values_simu))

# predict spokes
images_pred = MapFromDict("RebuiltMapFromParams", paramMap=map_for_sim, rounding=True,gen_mode="other")
images_pred.buildParamMap()

del map_for_sim
del keys_simu
del values_simu

images_pred.build_ref_images(seq)
pred_volumesi = images_pred.images_series

volumes_for_correction = np.array([np.mean(gp, axis=0) for gp in groupby(pred_volumesi, nspoke)])
volumes_for_correction=volumes_for_correction.reshape((volumes_for_correction.shape[0],-1))

trajectory=radial_traj
traj=trajectory.get_traj_for_reconstruction()

#
# ani1,ani2 =animate_multiple_images(pred_volumesi,m.images_series,cmap="gray")

from numpy.linalg import svd
from tqdm import tqdm
import pandas as pd

kx=np.arange(-np.pi,np.pi,2*np.pi/npoint)+np.pi/npoint
ky=np.arange(-np.pi,np.pi,2*np.pi/npoint)+np.pi/npoint
kx=np.concatenate([np.array([-np.pi]),kx,np.array([np.pi])])
ky=np.concatenate([np.array([-np.pi]),ky,np.array([np.pi])])

kdata_completed=list(kdata)
traj_completed=list(traj)

for i in tqdm(range(1,len(kx))):

    kx_ = kx[i-1]
    kx_next = kx[i]

    for j in range(1,len(ky)):

        ky_ = ky[j-1]
        ky_next=ky[j]

        indic_kx = (((traj[:,:,0]-kx_)*(traj[:,:,0]-kx_next))<=0).astype(int)
        indic_ky = (((traj[:,:,1]-ky_)*(traj[:,:,1]-ky_next))<=0).astype(int)

        indic = indic_kx * indic_ky
        indices_in_box=np.argwhere(indic==1)
        print("Number of kdata in box : {}".format(len(indices_in_box)))
        timesteps_for_k = np.unique(indices_in_box[:,0])

        all_timesteps = list(range(volumes_for_correction.shape[0]))
        missing_timesteps = list(set(all_timesteps) - set(timesteps_for_k))

        if len(missing_timesteps)==0:
            continue

        X = volumes_for_correction[timesteps_for_k,:]
        u, s, vh = np.linalg.svd(X, full_matrices=False)

        if s.size==0:
            continue

        index_retained = (s>0.01*s[0]).sum()
        u_red = u[:,:index_retained]
        vh_red = vh[:index_retained,:]
        s_red=s[:index_retained]

        df=pd.DataFrame(indices_in_box,columns=["Timesteps","Index"])
        df=df.drop_duplicates(subset="Timesteps")
        kdata_retained = kdata[df.Timesteps,df.Index]
        traj_retained=traj[df.Timesteps,df.Index,:]

        W = np.matmul(u_red.conj().T,kdata_retained)
        U = np.matmul(volumes_for_correction,np.matmul(vh_red.conj().T,np.diag(1/s_red)))

        kdata_interp = np.matmul(U,W)


        print(missing_timesteps)
        for t in missing_timesteps:
            traj_to_add = [(kx_+kx_next)/2,(ky_+ky_next)/2]
            traj_completed[t]=np.concatenate([traj_completed[t],[traj_to_add]])
            kdata_completed[t]=np.concatenate([kdata_completed[t],[kdata_interp[t]]])


kdata[df.iloc[0].Timesteps,df.iloc[0].Index]
kdata_interp[0]

np.unravel_index(np.argmax((((traj[:,:,0]-kx_)*(traj[:,:,0]-kx_next))<=0).astype(int)),traj.shape[:-1])


(((traj[:,:,0]-ky_)*(traj[:,:,0]-ky_next))<=0).astype(int)



for i in range(len(KX)):


    kx_ = KX[i]
    ky_ = KY[i]


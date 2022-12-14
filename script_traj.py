
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from trajectory import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np


from mutools import io

filename="./data/InVivo/3D/patient.002.v2.bis/meas_MID00155_FID17671_raFin_3D_tra_1x1x5mm_FULL_randomv1_reco395_SS_MRF_map_it0_wT1.mha"
t1_param_map_new=io.read(filename)
filename="./data/InVivo/3D/patient.002.v2.bis/meas_MID00155_FID17671_raFin_3D_tra_1x1x5mm_FULL_randomv1_reco395_SS_MRF_map_it0_ff.mha"
ff_param_map_new=io.read(filename)
filename="./data/InVivo/3D/patient.002.v2.bis/meas_MID00159_FID17675_raFin_3D_tra_1x1x5mm_FULL_1400_reco4_SS_MRF_map_it0_wT1.mha"
t1_param_map_old=io.read(filename)
filename="./data/InVivo/3D/patient.002.v2.bis/meas_MID00159_FID17675_raFin_3D_tra_1x1x5mm_FULL_1400_reco4_SS_MRF_map_it0_ff.mha"
ff_param_map_old=io.read(filename)

nb_slices=int(t1_param_map_new.shape[0])

sl=int(nb_slices/2)

curr_t1_param_map_new=np.array(t1_param_map_new[sl])
curr_t1_param_map_old=np.array(t1_param_map_old[sl])
curr_ff_param_map_new=np.array(ff_param_map_new[sl])
curr_ff_param_map_old=np.array(ff_param_map_old[sl])

curr_t1_param_map_new[curr_ff_param_map_new>0.7]=0
curr_t1_param_map_old[curr_ff_param_map_old>0.7]=0

plt.close("all")
plt.figure()
plt.imshow(np.flip(curr_t1_param_map_new.T),cmap="plasma",vmin=0,vmax=2000)
plt.colorbar()

plt.figure()
plt.imshow(np.flip(curr_ff_param_map_new.T),cmap="jet",vmin=0,vmax=1)
plt.colorbar()

plt.figure()
plt.imshow(np.flip(curr_t1_param_map_old.T),cmap="plasma",vmin=0,vmax=2000)
plt.colorbar()

plt.figure()
plt.imshow(np.flip(curr_ff_param_map_old.T),cmap="jet",vmin=0,vmax=1)
plt.colorbar()







import numpy as np

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss

nlist = 100
m = 8
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                  # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], 1) # sanity check
print(I)
print(D)
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
print(I[-5:])




#import matplotlib
#matplotlib.u<se("TkAgg")
import numpy as np
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,GaussianWeighting,BruteDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat
from mutools import io
from sklearn import linear_model
from scipy.optimize import minimize
from movements import TranslationBreathing

npoint=512
nb_allspokes=1400
window=8
ntimesteps=int(nb_allspokes/window)

radial_traj=Radial(total_nspokes=nb_allspokes,npoint=npoint)
image_size=(int(npoint/2),int(npoint/2))
volume=np.zeros(image_size,dtype="complex64")
volume[int(image_size[0]/2),int(image_size[1]/2)]=1
volumes=np.tile(volume,reps=(ntimesteps,1,1))
b1_all_slices=np.ones((1,)+image_size)

psf=undersampling_operator(volumes,radial_traj,b1_all_slices)

animate_images(psf)
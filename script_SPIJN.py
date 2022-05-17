
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
from dictoptimizers import *
from toy_model_keras import *
import tensorflow as tf
## Random map simulation

useGPU=False

dictfile = "mrf175.dict"
#dictfile = "mrf175_CS.dict"
dictfile = "mrf175_SimReco2.dict"
dictfile = "mrf175_SimReco2_light.dict"


with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

window = 8 #corresponds to nspoke by image
size=(256,256)

file_matlab_paramMap = "./data/Phantom1/paramMap.mat"

###### Building Map
#m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)

with open("mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

dict_config["ff"]=np.arange(0.,1.05,0.05)

window = 8 #corresponds to nspoke by image
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
size=(256,256)

predictions=[]

file_matlab_paramMap = "./data/paramMap.mat"

###### Building Map
m = RandomMap("TestRandom",dict_config,image_size=size,region_size=region_size,mask_reduction_factor=1/4)

m.buildParamMap()

#m.plotParamMap(save=True)

##### Simulating Ref Images
m.build_ref_images(seq)

ntimesteps=175
nspoke=8
npoint = 2*m.images_series.shape[1]

radial_traj=Radial(total_nspokes=ntimesteps*nspoke,npoint=npoint)
kdata = m.generate_kdata(radial_traj,useGPU=useGPU)

volumes = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=useGPU)
# mask = build_mask_single_image(kdata,radial_traj,m.image_size,useGPU=useGPU)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream
# plt.imshow(mask)
mask=m.mask

num_comp = 2
max_iter = 2
pca = True
threshold_pca = 15

keys, values = read_mrf_dict(dictfile, FF_list=None,aggregate_components=False)

values=np.moveaxis(values,-1,0)
values=values.reshape(-1,values.shape[-1])

all_signals = volumes[:, mask > 0]

if pca:
    pca_values = PCAComplex(n_components_=threshold_pca)

    pca_values.fit(values)

    print(
        "Components Retained {} out of {} timesteps".format(pca_values.n_components_, values.shape[1]))

    transformed_values = pca_values.transform(values)

    transformed_all_signals = np.transpose(
        pca_values.transform(np.transpose(all_signals)))


else:
    pca_values = None
    transformed_values = values
    transformed_all_signals = all_signals


transformed_values = transformed_values.T

transformed_values=np.concatenate([transformed_values.real,transformed_values.imag],axis=0)
transformed_all_signals=np.concatenate([transformed_all_signals.real,transformed_all_signals.imag],axis=0)


from SPIJN import *

Cr,Sfull,rel_err,C1,C=SPIJN(transformed_all_signals,transformed_values,num_comp=num_comp,max_iter=10)

fractions = Cr/np.expand_dims(Cr.sum(axis=1),axis=-1)

volume_fractions = makevol(fractions[:,0],mask>0)

plt.figure()
plt.imshow(volume_fractions)
plt.colorbar()

plt.figure()
plt.imshow(makevol(m.paramMap["ff"],mask>0))
plt.colorbar()



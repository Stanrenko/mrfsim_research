
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from utils_mrf import radial_golden_angle_traj,animate_images,animate_multiple_images,compare_patterns,translation_breathing,find_klargest_freq,SearchMrf,basicDictSearch,compare_paramMaps,regression_paramMaps,dictSearchMemoryOptim,voronoi_volumes,transform_py_map
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np


## Random map simulation

dictfile = "mrf175.dict"
size=(256,256)
nspoke=8

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)

seq = T1MRF(**sequence_config)

file_matlab_paramMap = "./data/paramMap.mat"
file_matlab_kSpace = "./data/KSpaceData.mat"

m= MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)
m.buildParamMap()


niter=0

#CS
map_rebuilt_CS,mask = dictSearchMemoryOptimIterativeExternalFile(dictfile,file_matlab_kSpace,seq,size,nspoke,niter=niter,pca=True,threshold_pca=15,split=2000,log=False,useAdjPred=False)

#PY
kdata, traj = load_data(file_matlab_kSpace)
npoint=traj.shape[1]
traj=np.reshape(groupby(traj, nspoke), (-1, npoint * nspoke))
kdata=np.reshape(groupby(kdata, nspoke), (-1, npoint * nspoke))

result=SearchMrf(kdata,traj,dictfile,niter,"brute","ls",size,nspoke,True)
map_rebuilt_PY=transform_py_map(result,mask)



regression_paramMaps(m.paramMap,map_rebuilt_CS[0],m.mask>0,mask>0,title="Orig vs CS")
regression_paramMaps(m.paramMap,map_rebuilt_PY,m.mask>0,mask>0,title="Orig vs PY")


compare_paramMaps(m.paramMap,map_rebuilt_CS[0],m.mask>0,mask>0,title1="Orig",title2="CS")
compare_paramMaps(m.paramMap,map_rebuilt_PY,m.mask>0,mask>0,title1="Orig",title2="PY")
compare_paramMaps(map_rebuilt_CS[0],map_rebuilt_PY,mask>0,title1="CS",title2="PY")




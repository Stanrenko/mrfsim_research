
#import matplotlib
#matplotlib.use("TkAgg")



from mrfsim import T1MRF
from image_series import *
from utils_mrf import radial_golden_angle_traj,animate_images,animate_multiple_images
import json

#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'


## Random map simulation

with open("/home/cslioussarenko/PythonRepositories/mrf-sim/mrf_sequence.json") as f:
    sequence_config = json.load(f)

seq = T1MRF(**sequence_config)

with open("/home/cslioussarenko/PythonRepositories/mrf-sim/mrf_dictconf.json") as f:
    dict_config = json.load(f)

dict_config["ff"]=np.arange(0.,1.05,0.05)

window = 8 #corresponds to nspoke by image
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
size=(256,256)

file_matlab_paramMap = "/home/cslioussarenko/PythonRepositories/mrf-sim/data/paramMap.mat"

###### Building Map
#m = RandomMap("TestRandom",dict_config,image_size=size,region_size=region_size,mask_reduction_factor=1/4)
m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap)
m.buildParamMap()
m.plotParamMap(figsize=(5,5))

#rand_map.plotParamMap("wT1",figsize=(5,5))

##### Simulating Ref Images
m.build_ref_images(seq,window)
#ani=animate_images(rand_map.images_series)

###### Undersampling k space

npoint = size[0]
total_nspoke=1400
nspoke=window

all_spokes=radial_golden_angle_traj(total_nspoke,npoint)
traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))

image_series_rebuilt = m.simulate_undersampled_images(traj)


#ani=animate_images(images_series)
#ani_r=animate_images(images_series_rebuilt)

ani1,ani2= animate_multiple_images(m.images_series,image_series_rebuilt)
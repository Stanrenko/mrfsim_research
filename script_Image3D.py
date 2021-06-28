
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
#dictfile = "mrf175_CS.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

window = 8 #corresponds to nspoke by image
size=(256,256)

file_matlab_paramMap = "./data/paramMap.mat"

###### Building Map
m = MapFromFile3D("TestPhantomV1",nb_slices=40,nb_empty_slices=12,file=file_matlab_paramMap,rounding=True)
m.buildParamMap()

#m.plotParamMap("wT1")

##### Simulating Ref Images
m.build_ref_images(seq,window)


undersampling_factor=4
npoint=512
nspoke=8
total_nspoke=nspoke*175
nb_slices=m.paramDict["nb_slices"]+2*m.paramDict["nb_empty_slices"]
size=m.image_size

nb_rep = int(nb_slices/undersampling_factor)

all_spokes=radial_golden_angle_traj(total_nspoke,npoint)
traj = np.reshape(all_spokes,(-1,nspoke*npoint))
#all_spokes_one_rep=np.reshape(all_spokes,(175,undersampling_factor,-1))
#traj=np.repeat(all_spokes_one_rep,nb_rep,axis=1)


k_z=np.zeros((175, nb_rep))
all_slices=np.linspace(-np.pi,np.pi,nb_slices)
k_z[0,:]=all_slices[::undersampling_factor]

for j in range(1,k_z.shape[0]):
    k_z[j,:]=np.sort(np.roll(all_slices,-j)[::undersampling_factor])

k_z=np.expand_dims(k_z,axis=-2)
traj = np.expand_dims(traj,axis=-1)
k_z,traj = np.broadcast_arrays(k_z,traj)
k_z = np.reshape(k_z,(175,-1))
traj= np.reshape(traj,(175,-1))


kdata = [
            finufft.nufft3d2(t.real, t.imag,z, p)
            for t, z,p in zip(traj,k_z, images_series)
        ]

density = np.abs(np.linspace(-1, 1, npoint))
kdata_test=[(np.reshape(k, (-1, npoint,nb_rep)) * np.resize(density,(npoint,nb_rep))).flatten() for k in kdata]


# kdata = (normalize_image_series(np.array(kdata)))

images_series_rebuilt = [
    finufft.nufft3d1(t.real, t.imag,z, s, size)
    for t,z, s in zip(traj,k_z, kdata_test)
]

plt.imshow(np.abs(images_series_rebuilt[0][35,:,:]))


size = m.image_size
images_series=m.images_series











npoint = 2*m.images_series.shape[1]
total_nspoke=8*175
nspoke=8

all_spokes=radial_golden_angle_traj(total_nspoke,npoint)
traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))

all_maps_adj=m.dictSearchMemoryOptimIterative(dictfile,seq,traj,npoint,niter=1,split=500,threshold_pca=15,log=False,useAdjPred=True,true_mask=False)
all_maps=m.dictSearchMemoryOptimIterative(dictfile,seq,traj,npoint,niter=1,split=500,threshold_pca=15,log=False,useAdjPred=False,true_mask=False)



regression_paramMaps(m.paramMap,all_maps_adj[1][0],m.mask>0,all_maps_adj[1][1]>0,title="Orig vs Adjusted Iterative")
regression_paramMaps(m.paramMap,all_maps[1][0],m.mask>0,all_maps[1][1]>0,title="Orig vs Iterative")

compare_paramMaps(m.paramMap,all_maps_adj[1][0],m.mask>0,all_maps_adj[1][1]>0,title1="Orig",title2="Adjusted Iterative")
compare_paramMaps(m.paramMap,all_maps[1][0],m.mask>0,all_maps[1][1]>0,title1="Orig",title2="Iterative")


from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

fig = plt.figure()
ax = p3.Axes3D(fig)

def gen(n):
    phi = 0
    while phi < 2*np.pi:
        yield np.array([np.cos(phi), np.sin(phi), phi])
        phi += 2*np.pi/n

def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])

N = 100
data = np.array(list(gen(N))).T
line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

# Setting the axes properties
ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 10.0])
ax.set_zlabel('Z')

ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=10000/N, blit=False)
#ani.save('matplot003.gif', writer='imagemagick')
plt.show()

import finufft
from cufinufft import cufinufft

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



## Random map simulation

dictfile = "mrf175.dict"
#dictfile = "mrf175_CS.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

window = 8 #corresponds to nspoke by image
size=(256,256)

file_matlab_paramMap = "./data/Phantom1/paramMap.mat"

###### Building Map
m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)

m.buildParamMap()

##### Simulating Ref Images
m.build_ref_images(seq,window)

ntimesteps=175
nspoke=8
npoint = 2*m.images_series.shape[1]

radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint)

traj = radial_traj.get_traj()
"""
Demonstrate the type 2 NUFFT using cuFINUFFT
"""



# Set up parameters for problem.
N1, N2 = m.images_series.shape[1], m.images_series.shape[2]                 # Size of uniform grid
M = traj.shape[1]                          # Number of nonuniform points
n_transf = 1                  # Number of input arrays
eps = 1e-6                      # Requested tolerance
dtype = np.float64              # Datatype (real)
complex_dtype = np.complex128    # Datatype (complex)


# Allocate memory for the nonuniform coefficients on the GPU.
c_gpu = GPUArray((n_transf, M), dtype=complex_dtype)
#c_finufft=np.zeros((n_transf,M)).astype(complex_dtype)

# Initialize the plan and set the points.
kdata_GPU=[]
start=datetime.now()
for i in list(range(m.images_series.shape[0])):
    fk = m.images_series[i,:,:]
    kx = traj[i, :, 0]
    ky = traj[i, :, 1]

    kx = kx.astype(dtype)
    ky = ky.astype(dtype)
    fk = fk.astype(complex_dtype)

    plan = cufinufft(2, (N1, N2), 1, eps=eps, dtype=dtype)
    plan.set_pts(to_gpu(kx), to_gpu(ky))
    plan.execute(c_gpu, to_gpu(fk))
    c = np.squeeze(c_gpu.get())
    kdata_GPU.append(c)
end=datetime.now()
print(end-start)

start=datetime.now()
kdata = [
                    finufft.nufft2d2(t[:, 0], t[:, 1], p)
                    for t, p in zip(traj, m.images_series)
                ]
end=datetime.now()
print(end-start)


start=datetime.now()
images = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size)
end=datetime.now()
print(end-start)

#images_GPU = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size)

start=datetime.now()
images_GPU = []

density = np.abs(np.linspace(-1, 1, npoint))
kdata_GPU = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata_GPU]

fk_gpu = GPUArray((n_transf, N1, N2), dtype=complex_dtype)
for i in list(range(m.images_series.shape[0])):
    c_retrieved = kdata_GPU[i]
    kx = traj[i, :, 0]
    ky = traj[i, :, 1]

    # Cast to desired datatype.
    kx = kx.astype(dtype)
    ky = ky.astype(dtype)
    c_retrieved = c_retrieved.astype(complex_dtype)

    # Allocate memory for the uniform grid on the GPU.


    # Initialize the plan and set the points.
    plan = cufinufft(1, (N1, N2), n_transf, eps=eps, dtype=dtype)
    plan.set_pts(to_gpu(kx), to_gpu(ky))

    # Execute the plan, reading from the strengths array c and storing the
    # result in fk_gpu.
    plan.execute(to_gpu(c_retrieved), fk_gpu)

    fk = np.squeeze(fk_gpu.get())
    images_GPU.append(fk)

images_GPU = np.array(images_GPU)
end=datetime.now()
print(end-start)





print("Max Diff on rebuilt images {}".format(np.max(np.abs(images_GPU-images))))
plt.plot(np.sort(np.abs(images_GPU-images).flatten())[::-1])



ani,aniGPU = animate_multiple_images(images,images_GPU)

ani_diff = animate_images(images-images_GPU)

#Rebuilt images







#
# plan_finufft=finufft.Plan(2,(N1, N2), n_transf, eps=eps, dtype=dtype)
# plan_finufft.setpts(kx, ky)
# plan_finufft.execute(c_finufft, fk)

# Execute the plan, reading from the uniform grid fk c and storing the result
# in c_gpu.

# Retreive the result from the GPU.

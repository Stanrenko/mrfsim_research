
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
import cupy as cp

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
npoint = 512

radial_traj=Radial(ntimesteps=ntimesteps,nspoke=nspoke,npoint=npoint)

if not(load):
    kdata = m.generate_kdata(radial_traj,useGPU=useGPU_simulation)
    with open("kdata_forLowRank_{}.pkl".format(m.name), "wb" ) as file:
        pickle.dump(kdata, file)

else:
    kdata = pickle.load( open( "kdata_forLowRank_{}.pkl".format(m.name), "rb" ) )

## Compressed sensing test

volumes = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=useGPU_simulation)

traj=radial_traj.get_traj_for_reconstruction()

if not(len(kdata)==len(traj)):
    kdata=np.array(kdata).reshape(len(traj),-1)

lambd=100

volumes_new = np.zeros((ntimesteps,)+size,dtype=np.complex128)
maxiter=100

for j,current_kdata in tqdm(enumerate(kdata)):

    # if j==0:
    #     m0 = np.zeros(size, dtype=np.complex128)
    #     maxiter=100
    # else:
    #     maxiter=10

    m0 = np.zeros(size, dtype=np.complex128)


    current_traj=traj[j]
    def J(m):
        global current_traj
        global current_kdata
        global lambd
        return J_fourier(m,current_traj,current_kdata) + lambd*J_sparse(m)

    def grad_J(m):
        global current_traj
        global current_kdata
        global lambd
        return grad_J_fourier(m,current_traj,current_kdata) + lambd*grad_J_sparse(m)

    m_opt = conjgrad(J, grad_J, m0, maxiter=maxiter)
    volumes_new[j]=m_opt
    m0 = m_opt

ani=animate_multiple_images(volumes,volumes_new)


mask = build_mask_single_image(kdata,radial_traj,m.image_size)

optimizer = SimpleDictSearch(mask=mask,niter=0,seq=seq,trajectory=radial_traj,split=500,pca=True,threshold_pca=15,log=False,useAdjPred=False,useGPU_dictsearch=False,useGPU_simulation=False)
all_maps_volumes=optimizer.search_patterns(dictfile,volumes)
all_maps_volumes_new=optimizer.search_patterns(dictfile,volumes_new)

maskROI= buildROImask_unique(m.paramMap)
    #maskROI=buildROImask(m.paramMap)
    # plt.close("all")
for i,mp in enumerate([all_maps_volumes,all_maps_volumes_new]):#all_maps_adj.keys():
    regression_paramMaps_ROI(m.paramMap, mp[0][0], m.mask > 0, mp[0][1] > 0,maskROI=maskROI,
                                 title="CompSens Test : ROI Orig vs Python {}".format(i), proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7,figsize=(30,15),fontsize=8,save=False)



compare_paramMaps(m.paramMap,all_maps_volumes_new[0][0],m.mask>0,all_maps_volumes_new[0][1]>0,adj_wT1=True,fat_threshold=0.7,title1="Orig",title2="Python Rebuilt CS",figsize=(30,10),fontsize=15,save=False,proj_on_mask1=True)
compare_paramMaps(m.paramMap,all_maps_volumes[0][0],m.mask>0,all_maps_volumes[0][1]>0,adj_wT1=True,fat_threshold=0.7,title1="Orig",title2="Python Rebuilt ",figsize=(30,10),fontsize=15,save=False,proj_on_mask1=True)


x=np.arange(-int(m.image_size[0]/2),int(m.image_size[0]/2),1.0)#+0.5
y=np.arange(-int(m.image_size[1]/2),int(m.image_size[1]/2),1.0)#+0.5
x=np.array(x,dtype=np.float16)
y=np.array(y,dtype=np.float16)
X,Y = np.meshgrid(x,y)
X = np.squeeze(X.reshape(1,-1))
Y = np.squeeze(Y.reshape(1,-1))
r = np.vstack((X,Y))

traj=radial_traj.get_traj_for_reconstruction()

def J(volumes,kdata,traj,r,split=10,lamb=0):
    nts=traj.shape[0]
    vol = volumes.reshape((volumes.shape[0],-1))
    if not (len(kdata) == len(traj)):
        kdata = np.array(kdata).reshape(len(traj), -1)
    nsamples=traj.shape[1]
    ngroups=int(nts/split)
    res=np.zeros((nts,nsamples),dtype=np.complex)
    grad=np.zeros((nts,vol.shape[-1]),dtype=np.complex)
    for t in tqdm(range(ngroups)):
        t_cur=t*split
        t_next=np.minimum((t+1)*split,nts)
        traj_cur=traj[t_cur:t_next]

        F_cur = np.matmul(traj_cur, r)
        F_cur_adj = F_cur.T.conj()
        F_cur_adj=np.moveaxis(F_cur_adj,-1,0)

        M_cur=vol[t_cur:t_next]
        M_cur=np.expand_dims(M_cur,axis=-1)

        kdata_cur=kdata[t_cur:t_next]
        print(kdata_cur.shape)
        print(F_cur.shape)
        print(M_cur.shape)
        current_res = (kdata_cur-np.squeeze(np.matmul(F_cur,M_cur)))
        res[t_cur:t_next]=current_res
        print(current_res.shape)
        print(F_cur_adj.shape)
        grad[t_cur:t_next]=np.squeeze(np.matmul(F_cur_adj,np.expand_dims(current_res,axis=-1)))
    return res,grad

def N(x,mu=1e-6):
    return np.sum(np.sqrt(np.abs(x)+mu))


def psi(m,type='db2',level=3,image_size=(256,256)):
    init_shape=m.shape
    m=m.reshape(image_size)
    c = pywt.wavedec2(m, type, level=level, mode="periodization")
    arr, slices = pywt.coeffs_to_array(c)
    arr = arr.reshape(init_shape)
    return arr,slices

def inv_psi(arr,slices,type="db2",image_size=(256,256)):
    init_shape = arr.shape
    arr=arr.reshape(image_size)
    coef = pywt.array_to_coeffs(arr, slices, output_format='wavedec2')
    volumes_rebuilt = pywt.waverec2(coef, type)
    volumes_rebuilt = volumes_rebuilt.reshape(init_shape)
    return volumes_rebuilt


def W(x,mu=1e-6):
    return np.diag(1/np.sqrt(np.abs(x)+mu))


def J_gpu(volumes,kdata,traj,r,split=10):
    kdata = cp.asarray(kdata)
    vol = cp.asarray(volumes.reshape((volumes.shape[0],-1)))
    traj=cp.asarray(traj)
    vol = cp.asarray(volumes.reshape((volumes.shape[0], -1)))
    r = cp.asarray(r)

    nts=traj.shape[0]

    if not (len(kdata) == len(traj)):
        kdata = kdata.reshape(len(traj), -1)
    nsamples=traj.shape[1]
    ngroups=int(nts/split)
    res=cp.zeros((nts,nsamples),dtype=cp.complex64)
    grad=cp.zeros((nts,vol.shape[-1]),dtype=cp.complex64)
    for t in tqdm(range(ngroups)):
        t_cur=t*split
        t_next=cp.minimum((t+1)*split,nts)
        traj_cur=traj[t_cur:t_next]

        F_cur = cp.matmul(traj_cur, r)
        F_cur_adj = F_cur.T.conj()
        F_cur_adj=cp.moveaxis(F_cur_adj,-1,0)

        M_cur=vol[t_cur:t_next]
        M_cur=np.expand_dims(M_cur,axis=-1)

        kdata_cur=kdata[t_cur:t_next]

        current_res = (kdata_cur-cp.squeeze(cp.matmul(F_cur,M_cur)))
        res[t_cur:t_next]=current_res

        grad[t_cur:t_next]=cp.squeeze(cp.matmul(F_cur_adj,cp.expand_dims(current_res,axis=-1)))
    return res.get(),grad.get()

volumes0 = np.zeros(volumes.shape)

res,grad=J(volumes0,np.array(kdata),traj,r,split=1)
grad_image=grad.reshape((grad.shape[0],)+m.image_size)
ts = 10
plt.imshow(np.abs(grad_image[ts]))
plt.colorbar()


from utils_mrf import animate_images
animate_images(grad_image)


F = np.matmul(traj,r)

####Wavelet filtering
import pywt

volumes = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=useGPU_simulation)

level=3
volumes_test=volumes[0,:,:]

c = pywt.wavedec2(volumes_test, 'db2', level=level)

#c[0] /= np.abs(c[0]).max()
#for detail_level in range(level):
#    c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
#    # show the normalized coefficients
arr, slices = pywt.coeffs_to_array(c)

#plt.imshow(np.abs(arr))

sorted_coef = np.sort(np.abs(arr.flatten()))[::-1]

cum_sum=np.cumsum(sorted_coef)
cum_sum=cum_sum/cum_sum[-1]
index_cut=(cum_sum>0.99).sum()
value = sorted_coef[index_cut]

#plt.plot(sorted_coef)

perc=90

perc_value=np.percentile(np.abs(arr),perc)
max_value=np.max(np.abs(arr))
arr_cut =arr.copy()
arr_cut[np.abs(arr_cut)<value]=0


#sorted_coef = np.sort(np.abs(arr_cut.flatten()))
#plt.plot(sorted_coef)

coef_cut = pywt.array_to_coeffs(arr_cut,slices,output_format='wavedec2')
volumes_cut = pywt.waverec2(coef_cut, 'db2')

plt.close("all")

plt.figure()
plt.imshow(np.abs(volumes_cut))
plt.title("Rebuilt After Filtering")

plt.figure()
plt.imshow(np.abs(volumes_test))
plt.title("Original")

plt.figure()
plt.imshow(np.abs(volumes_cut)-np.abs(volumes_test))
plt.title("Diff")

coef=pywt.array_to_coeffs(arr,slices,output_format='wavedec2')
volumes_rebuilt = pywt.waverec2(c, 'db2')

c_test=pywt.wavedec2(volumes[0,:,:], 'db2', level=level)
volumes_rebuilt = pywt.waverec2(c_test, 'db2')





import pywt
image = np.eye(256,256)

c = pywt.wavedec2(image, 'db2', level=3,mode="periodization")
arr, slices = pywt.coeffs_to_array(c)








FF_list = list(np.arange(0.,1.05,0.05))

keys,values=read_mrf_dict(dictfile ,FF_list ,aggregate_components=True)

threshold_pca=15
pca_signal = PCAComplex(n_components_=threshold_pca)
pca_signal.fit(values)

V = pca_signal.components_
v_hat = V.T.conj()

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
# x=np.array(x,dtype=np.float16)
# y=np.array(y,dtype=np.float16)
# X,Y = np.meshgrid(x,y)
# X = np.squeeze(X.reshape(1,-1))
# Y = np.squeeze(Y.reshape(1,-1))
# r = np.vstack((X,Y))


Np=kdata.size
d=kdata.flatten()

u_0 = np.zeros((m.images_series[0].size,threshold_pca))
eps=1
nit=3


def Fourier_Image(traj,I,image_size,useGPU=False,eps=1e-4):
    images_series=I.T.reshape((-1,)+image_size)
    if not (useGPU):
        kdata = [
            finufft.nufft2d2(t[:, 0], t[:, 1], p)
            for t, p in zip(traj, images_series)
        ]

    else:
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64
        N1, N2 = size[0], size[1]
        M = traj.shape[1]
        # Initialize the plan and set the points.
        kdata = []

        for i in tqdm(list(range(images_series.shape[0]))):
            # print("Allocating input")
            # start = datetime.now()

            fk = images_series[i, :, :]
            kx = traj[i, :, 0]
            ky = traj[i, :, 1]

            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            fk = fk.astype(complex_dtype)

            # end=datetime.now()
            # print(end-start)
            # print("Allocating Output")
            # start=datetime.now()

            fk_gpu = to_gpu(fk)
            c_gpu = GPUArray((M), dtype=complex_dtype)

            # end=datetime.now()
            # print(end-start)
            #
            # print("Executing FFT")
            # start=datetime.now()

            plan = cufinufft(2, (N1, N2), 1, eps=eps, dtype=dtype)
            plan.set_pts(to_gpu(kx), to_gpu(ky))
            plan.execute(c_gpu, fk_gpu)

            c = np.squeeze(c_gpu.get())

            fk_gpu.gpudata.free()
            c_gpu.gpudata.free()

            kdata.append(c)

            del c
            del kx
            del ky
            del fk
            del fk_gpu
            del c_gpu
            plan.__del__()

            # gc.collect()

        gc.collect()

    return np.array(kdata).flatten()

def Fourier_LowRank(traj,u,v_hat,image_size,useGPU=False):
    I = np.matmul(u,v_hat)
    return Fourier_Image(traj,I,image_size,useGPU)

def dG_LowRank(traj,v_hat,image_size,useGPU=False):
    L = v_hat.shape[0]
    N = image_size[0]*image_size[1]
    N_sample = traj.shape[0]*traj.shape[1]
    M = traj.shape[0]
    curr_I = np.zeros((N,M))
    dg_store = np.zeros((N, L, N_sample))
    for i in tqdm(range(N)):
        for j in range(L):
            curr_I[i,:]=v_hat[j,:]
            dg=Fourier_Image(traj,curr_I,image_size,useGPU)
            dg_store[i,j,:]=dg
    return dg_store

#dg_store = dG_LowRank(traj,v_hat,m.image_size,useGPU=True)

def  Grad_LowRank(traj,u,v_hat,d,image_size,useGPU=False,dg_store=None):
    N,L=u.shape
    N_sample = traj.shape[1]
    M=traj.shape[0]

    grad = np.zeros((N,L))
    #dg_store =  np.zeros((N,L,N_sample))
    curr_I = np.zeros((N,M))
    g_u = Fourier_LowRank(traj,u,v_hat,image_size,useGPU)
    for i in tqdm(range(N)):
        for j in range(L):
            curr_I[i,:]=v_hat[j,:]
            dg=Fourier_Image(traj,curr_I,image_size,useGPU)
            #dg_store[i,j,:]=dg
            grad[i,j]=np.vdot(dg,g_u-d)
    return grad

grad=Grad_LowRank(traj,u_0,v_hat,d,m.image_size,useGPU=True)






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
#
# from numpy.linalg import svd
# from tqdm import tqdm
# import pandas as pd
#
# kx=np.arange(-np.pi,np.pi,2*np.pi/npoint)+np.pi/npoint
# ky=np.arange(-np.pi,np.pi,2*np.pi/npoint)+np.pi/npoint
# kx=np.concatenate([np.array([-np.pi]),kx,np.array([np.pi])])
# ky=np.concatenate([np.array([-np.pi]),ky,np.array([np.pi])])
#
# kdata_completed=list(kdata)
# traj_completed=list(traj)
#
# for i in tqdm(range(1,len(kx))):
#
#     kx_ = kx[i-1]
#     kx_next = kx[i]
#
#     for j in range(1,len(ky)):
#
#         ky_ = ky[j-1]
#         ky_next=ky[j]
#
#         indic_kx = (((traj[:,:,0]-kx_)*(traj[:,:,0]-kx_next))<=0).astype(int)
#         indic_ky = (((traj[:,:,1]-ky_)*(traj[:,:,1]-ky_next))<=0).astype(int)
#
#         indic = indic_kx * indic_ky
#         indices_in_box=np.argwhere(indic==1)
#         print("Number of kdata in box : {}".format(len(indices_in_box)))
#         timesteps_for_k = np.unique(indices_in_box[:,0])
#
#         all_timesteps = list(range(volumes_for_correction.shape[0]))
#         missing_timesteps = list(set(all_timesteps) - set(timesteps_for_k))
#
#         if len(missing_timesteps)==0:
#             continue
#
#         X = volumes_for_correction[timesteps_for_k,:]
#         u, s, vh = np.linalg.svd(X, full_matrices=False)
#
#         if s.size==0:
#             continue
#
#         index_retained = (s>0.01*s[0]).sum()
#         u_red = u[:,:index_retained]
#         vh_red = vh[:index_retained,:]
#         s_red=s[:index_retained]
#
#         df=pd.DataFrame(indices_in_box,columns=["Timesteps","Index"])
#         df=df.drop_duplicates(subset="Timesteps")
#         kdata_retained = kdata[df.Timesteps,df.Index]
#         traj_retained=traj[df.Timesteps,df.Index,:]
#
#         W = np.matmul(u_red.conj().T,kdata_retained)
#         U = np.matmul(volumes_for_correction,np.matmul(vh_red.conj().T,np.diag(1/s_red)))
#
#         kdata_interp = np.matmul(U,W)
#
#
#         print(missing_timesteps)
#         for t in missing_timesteps:
#             traj_to_add = [(kx_+kx_next)/2,(ky_+ky_next)/2]
#             traj_completed[t]=np.concatenate([traj_completed[t],[traj_to_add]])
#             kdata_completed[t]=np.concatenate([kdata_completed[t],[kdata_interp[t]]])
#


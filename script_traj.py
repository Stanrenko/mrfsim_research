#
# #import matplotlib
# #matplotlib.use("TkAgg")
# from mrfsim import T1MRF
# from image_series import *
# from trajectory import *
# from utils_mrf import *
# import json
# from finufft import nufft1d1,nufft1d2
# from scipy import signal,interpolate
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import matplotlib.pyplot as plt
# import numpy as np
#
# from mutools import io
#
#
# #import matplotlib
# #matplotlib.u<se("TkAgg")
# from mrfsim import T1MRF
# from image_series import *
# from dictoptimizers import SimpleDictSearch
# from utils_mrf import *
# import json
# import readTwix as rT
# import time
# import os
# from numpy.lib.format import open_memmap
# from numpy import memmap
# import pickle
# from scipy.io import loadmat,savemat
# import twixtools
# from mutools import io
# import cv2
# import scipy
# from utils_reco import calculate_sensitivity_map
#
#
# import dask.array as da
#
# kdata_all_channels_all_slices=np.load("data/InVivo/3D/patient.010.v5/meas_MID00050_FID72061_raFin_3D_tra_0_8x0_8x3mm_FULL_new_mrf_us2_kdata.npy")
#
#
#
#
#
# data_shape=kdata_all_channels_all_slices.shape
# nb_channels=data_shape[0]
# nb_allspokes = data_shape[-3]
# npoint = data_shape[-1]
# nb_slices = data_shape[-2]
# image_size=(nb_slices,int(npoint/2),int(npoint/2))
#
# density = np.abs(np.linspace(-1, 1, npoint))
# # density=np.expand_dims(axis=0)
# density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))
# kdata_all_channels_all_slices/=density
#
# radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)
#
#
# data_numpy_zkxky=np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices,axes=2),axis=2,workers=24),axes=2).astype("complex64")
# #data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
# data_numpy_zkxky = data_numpy_zkxky.reshape(nb_channels, -1, nb_slices, npoint)
# data_numpy_zkxky=np.moveaxis(data_numpy_zkxky,-2,1)
#
# #window=data_numpy_zkxky.shape[3]
#
#
# n_comp=30
#
# #data_numpy_zkxky_for_pca_all=np.zeros((n_comp,nb_slices,ntimesteps,window,npoint),dtype=data_numpy_zkxky.dtype)
# data_numpy_zkxky_for_pca_all = np.zeros((n_comp, nb_slices, nb_allspokes, npoint),
#                                         dtype=data_numpy_zkxky.dtype)
#
#
#
#
#
# sl=0
#
# #PCA
# data_numpy_zkxky_slice=data_numpy_zkxky[:,sl]
# #data_numpy_zkxky_slice=data_numpy_zkxky_slice.reshape(nb_channels,ntimesteps,-1)
# data_numpy_zkxky_for_pca=data_numpy_zkxky_slice.reshape(nb_channels,-1)
# pca=PCAComplex(n_components_=n_comp)
# pca.fit(data_numpy_zkxky_for_pca.T)
#
# plt.close("all")
# plt.plot(pca.explained_variance_ratio_)
#
#
# #SVD
# data_numpy_zkxky_slice=data_numpy_zkxky[:,sl]
# #data_numpy_zkxky_slice=data_numpy_zkxky_slice.reshape(nb_channels,ntimesteps,-1)
# data_numpy_zkxky_for_pca=data_numpy_zkxky_slice.reshape(nb_channels,-1)
#
# u,s,vh = da.linalg.svd(da.asarray(data_numpy_zkxky_for_pca))
# vh=np.array(vh)
# plt.figure()
# plt.plot(np.cumsum(s**2)/np.sum(s**2))
# plt.plot(pca.explained_variance_ratio_)
# s=np.array(s)
#
# L0=30
# retrieved_data=u[:,:L0]@np.diag(s[:L0])@vh[:L0]
#
# ch=np.random.choice(nb_channels)
# ts=np.random.choice(nb_allspokes)
#
# plt.figure()
# plt.title("Sl {} ch {} ts {}".format(sl,ch,ts))
# plt.plot(retrieved_data.reshape(nb_channels,nb_allspokes,-1)[ch,ts],label="Retrieved")
# plt.plot(data_numpy_zkxky_slice.reshape(nb_channels,nb_allspokes,-1)[ch,ts],label="Original")
# plt.legend()
#
#
#
#
# pca_dict={}
#
# for sl in tqdm(range(nb_slices)):
#     data_numpy_zkxky_slice=data_numpy_zkxky[:,sl]
#     #data_numpy_zkxky_slice=data_numpy_zkxky_slice.reshape(nb_channels,ntimesteps,-1)
#
#     data_numpy_zkxky_for_pca=data_numpy_zkxky_slice.reshape(nb_channels,-1)
#
#     pca=PCAComplex(n_components_=n_comp)
#
#     pca.fit(data_numpy_zkxky_for_pca.T)
#
#     pca_dict[sl]=deepcopy(pca)
#
#     data_numpy_zkxky_for_pca_transformed=pca.transform(data_numpy_zkxky_for_pca.T)
#     data_numpy_zkxky_for_pca_transformed=data_numpy_zkxky_for_pca_transformed.T
#
#     data_numpy_zkxky_for_pca_all[:,sl,:,:]=data_numpy_zkxky_for_pca_transformed.reshape(n_comp,-1,npoint)
#
#
# density = np.abs(np.linspace(-1, 1, npoint))
# # density=np.expand_dims(axis=0)
# density = np.expand_dims(density, tuple(range(data_numpy_zkxky_for_pca_all.ndim - 1)))
# data_numpy_zkxky_for_pca_all*=density
#
# data_numpy_zkxky_for_pca_all
#
# plt.figure()
# for sl in np.arange(0,nb_slices)[::8]:
#     plt.plot(pca_dict[sl].explained_variance_ratio_,label=sl)
# plt.legend()
#
#
# b1_all_slices_2Dplus1_pca=calculate_sensitivity_map(np.moveaxis(data_numpy_zkxky_for_pca_all,1,0).reshape(nb_slices,n_comp,nb_allspokes,-1),radial_traj_2D,image_size=image_size[1:],hanning_filter=True,res=8)
# b1_all_slices_2Dplus1_pca=np.moveaxis(b1_all_slices_2Dplus1_pca,1,0)
#
#
# plt.close("all")
# for sl in range(nb_slices)[::10]:
#     plot_image_grid(np.abs(b1_all_slices_2Dplus1_pca[:,sl]),nb_row_col=(6,6))
#
#
#
#
#
#
# kdata_all_channels_all_slices=np.load("data/InVivo/3D/patient.010.v5/meas_MID00048_FID72059_raFin_3D_tra_0_8x0_8x3mm_FULL_new_respi_kdata.npy")
#
#
#
#
#
# data_shape=kdata_all_channels_all_slices.shape
# nb_channels=data_shape[0]
# nb_allspokes = data_shape[-3]
# npoint = data_shape[-1]
# nb_slices = data_shape[-2]
# image_size=(nb_slices,int(npoint/2),int(npoint/2))
#
# density = np.abs(np.linspace(-1, 1, npoint))
# # density=np.expand_dims(axis=0)
# density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))
# kdata_all_channels_all_slices/=density
#
# radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)
#
#
# data_numpy_zkxky=np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices,axes=2),axis=2,workers=24),axes=2).astype("complex64")
# #data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
# data_numpy_zkxky = data_numpy_zkxky.reshape(nb_channels, -1, nb_slices, npoint)
# data_numpy_zkxky=np.moveaxis(data_numpy_zkxky,-2,1)
#
# #window=data_numpy_zkxky.shape[3]
#
#
# n_comp=30
#
# #data_numpy_zkxky_for_pca_all=np.zeros((n_comp,nb_slices,ntimesteps,window,npoint),dtype=data_numpy_zkxky.dtype)
# data_numpy_zkxky_for_pca_all_respi = np.zeros((n_comp, nb_slices, nb_allspokes, npoint),
#                                         dtype=data_numpy_zkxky.dtype)
#
#
#
#
#
#
# pca_dict_respi={}
#
# for sl in tqdm(range(nb_slices)):
#     data_numpy_zkxky_slice=data_numpy_zkxky[:,sl]
#     #data_numpy_zkxky_slice=data_numpy_zkxky_slice.reshape(nb_channels,ntimesteps,-1)
#
#     data_numpy_zkxky_for_pca=data_numpy_zkxky_slice.reshape(nb_channels,-1)
#
#     pca=PCAComplex(n_components_=n_comp)
#
#     pca.fit(data_numpy_zkxky_for_pca.T)
#
#     pca_dict_respi[sl]=deepcopy(pca)
#
#     data_numpy_zkxky_for_pca_transformed=pca.transform(data_numpy_zkxky_for_pca.T)
#     data_numpy_zkxky_for_pca_transformed=data_numpy_zkxky_for_pca_transformed.T
#
#     data_numpy_zkxky_for_pca_all_respi[:,sl,:,:]=data_numpy_zkxky_for_pca_transformed.reshape(n_comp,-1,npoint)
#
#
# density = np.abs(np.linspace(-1, 1, npoint))
# # density=np.expand_dims(axis=0)
# density = np.expand_dims(density, tuple(range(data_numpy_zkxky_for_pca_all_respi.ndim - 1)))
# data_numpy_zkxky_for_pca_all_respi*=density
#
# plt.figure()
# for sl in np.arange(0,nb_slices)[::8]:
#     plt.plot(pca_dict_respi[sl].explained_variance_ratio_,label=sl)
# plt.legend()
#
#
#
# b1_all_slices_2Dplus1_pca_respi=calculate_sensitivity_map(np.moveaxis(data_numpy_zkxky_for_pca_all_respi,1,0).reshape(nb_slices,n_comp,nb_allspokes,-1),radial_traj_2D,image_size=image_size[1:],hanning_filter=True,res=8)
# b1_all_slices_2Dplus1_pca_respi=np.moveaxis(b1_all_slices_2Dplus1_pca_respi,1,0)
#
#
# for sl in range(nb_slices)[::10]:
#     plot_image_grid(np.abs(b1_all_slices_2Dplus1_pca_respi[:,sl]),nb_row_col=(6,6))
#
#
#
#
#
#
#
#
# #import matplotlib
# #matplotlib.use("TkAgg")
# from mrfsim import T1MRF
# from image_series import *
# from trajectory import *
# from utils_mrf import *
# import json
# from finufft import nufft1d1,nufft1d2
# from scipy import signal,interpolate
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import matplotlib.pyplot as plt
# import numpy as np
#
# from mutools import io
#
#
# #import matplotlib
# #matplotlib.u<se("TkAgg")
# from mrfsim import T1MRF
# from image_series import *
# from dictoptimizers import SimpleDictSearch
# from utils_mrf import *
# import json
# import readTwix as rT
# import time
# import os
# from numpy.lib.format import open_memmap
# from numpy import memmap
# import pickle
# from scipy.io import loadmat,savemat
# import twixtools
# from mutools import io
# import cv2
# import scipy
# from utils_reco import calculate_sensitivity_map
#
#
# import dask.array as da
#
# kdata_all_channels_all_slices=np.load("data/InVivo/3D/patient.010.v5/meas_MID00048_FID72059_raFin_3D_tra_0_8x0_8x3mm_FULL_new_respi_kdata.npy")
# kdata_all_channels_all_slices.shape
#
#
#
# data_shape=kdata_all_channels_all_slices.shape
# nb_channels=data_shape[0]
# nb_allspokes = data_shape[-3]
# npoint = data_shape[-1]
# nb_slices = data_shape[-2]
# image_size=(nb_slices,int(npoint/2),int(npoint/2))
#
# radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)
#
# ch=0
# kdata=np.expand_dims(kdata_all_channels_all_slices[ch],axis=0)
#
# data = np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(kdata, axes=2), axis=2, workers=24), axes=2)
#
# data = np.moveaxis(data, -2, 1)
#
# # data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
# images_series_rebuilt = np.zeros(image_size, dtype=np.complex64)
#
# # data_curr_transformed_all=cp.zeros((nb_slices,n_comp,ntimesteps,window*npoint))
#
# traj_reco = radial_traj_2D.get_traj_for_reconstruction(1).astype("float32")
# traj_reco = traj_reco.reshape(-1, 2)
#
# for sl in tqdm(range(nb_slices)):
#     data_curr = data[0, sl]
#     # data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
#     data_curr = data_curr.flatten().astype('complex64')
#
#     fk = finufft.nufft2d1(traj_reco[:, 0], traj_reco[:, 1], data_curr, image_size[1:])
#     images_series_rebuilt[sl] = fk
#
# animate_images(np.abs(images_series_rebuilt))
#
#
# def J_curr(x,u,lambd,is_weighted=False,mask=None,shift=0):
#     return lambd*(J_TV(x,axis=0,is_weighted=is_weighted,mask=mask,shift=shift)+0.2*J_TV(x,axis=1,is_weighted=is_weighted,mask=mask,shift=shift)+0.2*J_TV(x,axis=2,is_weighted=is_weighted,mask=mask,shift=shift))+0.5*np.linalg.norm(x-u)**2
#
# def grad_J_curr(x,u,lambd,is_weighted=False,mask=None,shift=0):
#     return lambd*(grad_J_TV(x,axis=0,is_weighted=is_weighted,mask=mask,shift=shift)+0.2*grad_J_TV(x,axis=1,is_weighted=is_weighted,mask=mask,shift=shift)+0.2*grad_J_TV(x,axis=2,is_weighted=is_weighted,mask=mask,shift=shift))+(x-u)
#
#
#
# mu=0.1
#
# u=2*mu*images_series_rebuilt
# u0=images_series_rebuilt
#
# lambd=0.005
# mask=None
# dens_adj=True
# niter=1
# iter_cg=5
# t0=1e-2
# radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)
#
# u_list_tv_0001_fista=[u]
# y=u
# t=1
# u = conjgrad(lambda x: J_curr(x,y, lambd*mu,mask=mask), lambda x: grad_J_curr(x,y,lambd*mu, mask=mask), y, tolgrad=1e-4, maxiter=iter_cg,
#                  alpha=0.05, beta=0.6, t0=t0, log=True, plot=True, filename_save=None, folder_logs="./logs",
#                  folder_figures="./figures")
#
# animate_multiple_images(np.abs(u0),np.abs(u),cmap="gray")
#
# t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
# y = u
# t = t_next
# u_list_tv_0001_fista.append(u)
#
# b1_all_slices_2Dplus1_pca=np.ones((1,)+image_size)
#
# for i in range(niter):
#     print("################ OUTER LOOP Iter {} ##############################".format(i))
#     u_prev = u
#     volumesi = undersampling_operator_singular_new(np.expand_dims(y, axis=0), radial_traj_2D, b1_all_slices_2Dplus1_pca,
#                                                    weights=None, density_adj=dens_adj)
#     volumesi = volumesi.squeeze()
#     grad = volumesi - u0
#     y = y - 2*mu * grad
#
#     u = conjgrad(lambda x: J_curr(x,y, lambd*mu,mask=mask), lambda x: grad_J_curr(x,y,lambd*mu, mask=mask), y, tolgrad=1e-4, maxiter=iter_cg,
#                  alpha=0.05, beta=0.6, t0=t0, log=True, plot=True, filename_save=None, folder_logs="./logs",
#                  folder_figures="./figures")
#
#     t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
#     y = u + (t - 1) / t_next * (u - u_prev)
#     t = t_next
#
#     u_list_tv_0001_fista.append(u)
#
#
# u_list_tv_0001_fista=np.array(u_list_tv_0001_fista)
#
# sl=np.random.randint(nb_slices-1)
# plot_image_grid(np.abs(u_list_tv_0001_fista[:,sl]),nb_row_col=(3,3),cmap="jet")
#
#
#
#
# mu=0.1
#
#
#
# lambd=0.005
# mask=None
# dens_adj=True
# niter=0
# iter_cg=5
# t0=1e-2
# radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)
# b1_all_slices_2Dplus1_pca=np.ones((1,)+image_size)
#
# image_coils_init=[]
# image_coils=[]
#
# for ch in tqdm(range(nb_channels)):
#     print("############# PROCESSING CHANNEL {} ##########################".format(ch))
#     kdata = np.expand_dims(kdata_all_channels_all_slices[ch], axis=0)
#
#     data = np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(kdata, axes=2), axis=2, workers=24), axes=2)
#
#     data = np.moveaxis(data, -2, 1)
#
#     # data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
#     images_series_rebuilt = np.zeros(image_size, dtype=np.complex64)
#
#     # data_curr_transformed_all=cp.zeros((nb_slices,n_comp,ntimesteps,window*npoint))
#
#     traj_reco = radial_traj_2D.get_traj_for_reconstruction(1).astype("float32")
#     traj_reco = traj_reco.reshape(-1, 2)
#
#     for sl in tqdm(range(nb_slices)):
#         data_curr = data[0, sl]
#         # data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
#         data_curr = data_curr.flatten().astype('complex64')
#
#         fk = finufft.nufft2d1(traj_reco[:, 0], traj_reco[:, 1], data_curr, image_size[1:])
#         images_series_rebuilt[sl] = fk
#
#     u = 2 * mu * images_series_rebuilt
#     u0 = images_series_rebuilt
#     image_coils_init.append(u0)
#
#     y = u
#     t = 1
#     u = conjgrad(lambda x: J_curr(x, y, lambd * mu, mask=mask), lambda x: grad_J_curr(x, y, lambd * mu, mask=mask), y,
#                  tolgrad=1e-4, maxiter=iter_cg,
#                  alpha=0.05, beta=0.6, t0=t0, log=True, plot=True, filename_save=None, folder_logs="./logs",
#                  folder_figures="./figures")
#
#     animate_multiple_images(np.abs(u0), np.abs(u), cmap="gray")
#
#     t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
#     y = u
#     t = t_next
#
#     for i in range(niter):
#         print("################ OUTER LOOP Iter {} ##############################".format(i))
#         u_prev = u
#         volumesi = undersampling_operator_singular_new(np.expand_dims(y, axis=0), radial_traj_2D,
#                                                        b1_all_slices_2Dplus1_pca,
#                                                        weights=None, density_adj=dens_adj)
#         volumesi = volumesi.squeeze()
#         grad = volumesi - u0
#         y = y - 2 * mu * grad
#
#         u = conjgrad(lambda x: J_curr(x, y, lambd * mu, mask=mask), lambda x: grad_J_curr(x, y, lambd * mu, mask=mask),
#                      y, tolgrad=1e-4, maxiter=iter_cg,
#                      alpha=0.05, beta=0.6, t0=t0, log=True, plot=True, filename_save=None, folder_logs="./logs",
#                      folder_figures="./figures")
#
#         t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
#         y = u + (t - 1) / t_next * (u - u_prev)
#         t = t_next
#
#     image_coils.append(u)
#
# plt.close("all")
#
# image_coils_init=np.array(image_coils_init)
# image_coils=np.array(image_coils)
#
# plt.close("all")
# sl=np.random.randint(nb_slices-1)
# sl=80
# plot_image_grid(np.abs(image_coils_init[:,sl]),nb_row_col=(6,6),same_range=True)
# plot_image_grid(np.abs(image_coils[:,sl]),nb_row_col=(6,6),same_range=True)
#
# plt.figure()
# plt.imshow(np.abs(image_coils_init[0,:,:,256]),aspect="auto")
# np.save("test_image_coils_denoised.npy",image_coils)
#
#
# import numpy as np
# import scipy as sp
# from tqdm import tqdm
#
# image_coils=np.load("test_image_coils_denoised.npy")
#
#
# fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax)
# ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax)
#
#
# sl=80
# kdata_cartesian=fft(image_coils[:,sl],(1,2))
#
#
# plt.imshow(np.abs(kdata_cartesian[0,0]))
# image_cartesian=ifft(kdata_cartesian,(1,2,3))
#
#
# kdata_cartesian=np.expand_dims(np.moveaxis(kdata_cartesian,0,-1),axis=0)
# kernelSize = 8
# CalibSize = 32
# t = 0.02
# c = 0.95
#
#
# maps=espirit(kdata_cartesian,kernelSize,CalibSize,t,c)
#
#
#
# def espirit(X, k, r, t, c):
#     """
#     Derives the ESPIRiT operator.
#
#     Arguments:
#       X: Multi channel k-space data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric
#          dimensions and (nc) is the channel dimension.
#       k: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel
#          will have dimensions (1, k, k, 8)
#       r: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the
#          calibration region will have dimensions (1, r, r, 8)
#       t: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
#          largest singular value are set to zero.
#       c: Crop threshold that determines eigenvalues "=1".
#     Returns:
#       maps: This is the ESPIRiT operator. It will have dimensions (sx, sy, sz, nc, nc) with (sx, sy, sz, :, idx)
#             being the idx'th set of ESPIRiT maps.
#     """
#
#     sx = np.shape(X)[0]
#     sy = np.shape(X)[1]
#     sz = np.shape(X)[2]
#     nc = np.shape(X)[3]
#
#     sxt = (sx//2-r//2, sx//2+r//2) if (sx > 1) else (0, 1)
#     syt = (sy//2-r//2, sy//2+r//2) if (sy > 1) else (0, 1)
#     szt = (sz//2-r//2, sz//2+r//2) if (sz > 1) else (0, 1)
#
#     # Extract calibration region.
#     C = X[sxt[0]:sxt[1], syt[0]:syt[1], szt[0]:szt[1], :].astype(np.complex64)
#
#     # Construct Hankel matrix.
#     p = (sx > 1) + (sy > 1) + (sz > 1)
#     A = np.zeros([(r-k+1)**p, k**p * nc]).astype(np.complex64)
#     print(A.shape)
#     idx = 0
#     print("Building the Autocalibration matrix of A")
#     for xdx in tqdm(range(max(1, C.shape[0] - k + 1))):
#       for ydx in range(max(1, C.shape[1] - k + 1)):
#         for zdx in range(max(1, C.shape[2] - k + 1)):
#           # numpy handles when the indices are too big
#           block = C[xdx:xdx+k, ydx:ydx+k, zdx:zdx+k, :].astype(np.complex64)
#           A[idx, :] = block.flatten()
#           idx = idx + 1
#
#     # Take the Singular Value Decomposition.
#     print("Performing SVD of A")
#     U, S, VH = np.linalg.svd(A, full_matrices=True)
#     V = VH.conj().T
#
#     # Select kernels.
#     n = np.sum(S >= t * S[0])
#     V = V[:, 0:n]
#
#     kxt = (sx//2-k//2, sx//2+k//2) if (sx > 1) else (0, 1)
#     kyt = (sy//2-k//2, sy//2+k//2) if (sy > 1) else (0, 1)
#     kzt = (sz//2-k//2, sz//2+k//2) if (sz > 1) else (0, 1)
#
#     # Reshape into k-space kernel, flips it and takes the conjugate
#     print("Building kernels")
#     kernels = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
#     kerdims = [(sx > 1) * k + (sx == 1) * 1, (sy > 1) * k + (sy == 1) * 1, (sz > 1) * k + (sz == 1) * 1, nc]
#     for idx in tqdm(range(n)):
#         kernels[kxt[0]:kxt[1],kyt[0]:kyt[1],kzt[0]:kzt[1], :, idx] = np.reshape(V[:, idx], kerdims)
#
#     # Take the iucfft
#     print("Building kernel images")
#     axes = (0, 1, 2)
#     kerimgs = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
#     for idx in tqdm(range(n)):
#         for jdx in range(nc):
#             ker = kernels[::-1, ::-1, ::-1, jdx, idx].conj()
#             kerimgs[:,:,:,jdx,idx] = fft(ker, axes) * np.sqrt(sx * sy * sz)/np.sqrt(k**p)
#
#     # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
#     print("building maps")
#     maps = np.zeros(np.append(np.shape(X), nc)).astype(np.complex64)
#     for idx in tqdm(range(0, sx)):
#         for jdx in range(0, sy):
#             for kdx in range(0, sz):
#
#                 Gq = kerimgs[idx,jdx,kdx,:,:]
#
#                 u, s, vh = np.linalg.svd(Gq, full_matrices=True)
#                 for ldx in range(0, nc):
#                     if (s[ldx]**2 > c):
#                         maps[idx, jdx, kdx, :, ldx] = u[:, ldx]
#
#     return maps
#
# def espirit_proj(x, esp):
#     """
#     Construct the projection of multi-channel image x onto the range of the ESPIRiT operator. Returns the inner
#     product, complete projection and the null projection.
#
#     Arguments:
#       x: Multi channel image data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric
#          dimensions and (nc) is the channel dimension.
#       esp: ESPIRiT operator as returned by function: espirit
#
#     Returns:
#       ip: This is the inner product result, or the image information in the ESPIRiT subspace.
#       proj: This is the resulting projection. If the ESPIRiT operator is E, then proj = E E^H x, where H is
#             the hermitian.
#       null: This is the null projection, which is equal to x - proj.
#     """
#     ip = np.zeros(x.shape).astype(np.complex64)
#     proj = np.zeros(x.shape).astype(np.complex64)
#     for qdx in range(0, esp.shape[4]):
#         for pdx in range(0, esp.shape[3]):
#             ip[:, :, :, qdx] = ip[:, :, :, qdx] + x[:, :, :, pdx] * esp[:, :, :, pdx, qdx].conj()
#
#     for qdx in range(0, esp.shape[4]):
#         for pdx in range(0, esp.shape[3]):
#           proj[:, :, :, pdx] = proj[:, :, :, pdx] + ip[:, :, :, qdx] * esp[:, :, :, pdx, qdx]
#
#     return (ip, proj, x - proj)
#
# #import matplotlib
# #matplotlib.u<se("TkAgg")
# from mrfsim import T1MRF
# from image_series import *
# from dictoptimizers import SimpleDictSearch
# from utils_mrf import *
# import json
# import readTwix as rT
# import time
# import os
# from numpy.lib.format import open_memmap
# from numpy import memmap
# import pickle
# from scipy.io import loadmat,savemat
# import twixtools
# from mutools import io
# import cv2
# import scipy
# from utils_reco import calculate_sensitivity_map
#
# os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
# sys.path.append(os.environ['TOOLBOX_PATH'] + "/python/")
# import cfl
# from bart import bart
#
#
# kdata_all_channels_all_slices=np.load("data/InVivo/3D/patient.010.v5/meas_MID00048_FID72059_raFin_3D_tra_0_8x0_8x3mm_FULL_new_respi_kdata.npy")
# kdata_all_channels_all_slices.shape
#
#
#
# data_shape=kdata_all_channels_all_slices.shape
# nb_channels=data_shape[0]
# nb_allspokes = data_shape[-3]
# npoint = data_shape[-1]
# nb_slices = data_shape[-2]
# image_size=(nb_slices,int(npoint/2),int(npoint/2))
#
# incoherent=False
# radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
#                        nb_slices=nb_slices, incoherent=incoherent)
#
# traj_python = radial_traj.get_traj()
# traj_python = np.transpose(traj_python)
#
#
#
#
#
# traj_python_for_bart = traj_python.astype("complex64")
# # traj_python_for_bart[:2,:,:]=traj_python
#
# traj_python_for_bart = traj_python_for_bart.reshape(3, npoint, -1)
# traj_python_for_bart[0] = traj_python_for_bart[0] / np.max(np.abs(traj_python_for_bart[0])) * int(
#     npoint / 4)
# traj_python_for_bart[1] = traj_python_for_bart[1] / np.max(np.abs(traj_python_for_bart[1])) * int(
#     npoint / 4)
#
# traj_python_for_bart[2] = traj_python_for_bart[2] / np.max(np.abs(traj_python_for_bart[2])) * int(
#     nb_slices / 2)
#
# cfl.writecfl("traj",traj_python_for_bart)
#
#
# ch=0
# data=np.expand_dims(kdata_all_channels_all_slices[ch],axis=0)
#
# kdata_multi_for_bart_full=data.reshape(1,-1,npoint).T
# #kdata_multi_for_bart_full=kdata_multi_for_bart_full.reshape(npoint,-1,nb_channels)
# kdata_multi_for_bart_full=np.expand_dims(kdata_multi_for_bart_full,axis=0)[:,:,:]
# #kdata_multi_for_bart_full=kdata_multi_for_bart_full[:,(center_res-res):(center_res+res)]
# cfl.writecfl("kdata_multi_full",kdata_multi_for_bart_full)
#
# coil_img=bart(1,"nufft -i -t traj kdata_multi_full".format(int(npoint/2),int(npoint/2),nb_slices))
# #coil_img=bart(1,"nufft -i -t -m 1 traj kdata_multi_full")
# cfl.writecfl("coil_img",coil_img)
# coil_img=cfl.readcfl("coil_img")
#
# plt.figure()
# plt.imshow(np.abs(coil_img[:,:,int(nb_slices/2)]))
#
# animate_images(np.abs(coil_img).T)
#
# plt.figure()
# plot_image_grid(np.moveaxis(np.abs(coil_img[:,:,int(nb_slices/2)]).squeeze(),-1,0),nb_row_col=(6,6))
#
#
#
#
#
#
# #import matplotlib
# #matplotlib.u<se("TkAgg")
# from mrfsim import T1MRF
# from image_series import *
# from dictoptimizers import SimpleDictSearch
# from utils_mrf import *
# import json
# import readTwix as rT
# import time
# import os
# from numpy.lib.format import open_memmap
# from numpy import memmap
# import pickle
# from scipy.io import loadmat,savemat
# import twixtools
# from mutools import io
# import cv2
# import scipy
# from utils_reco import calculate_sensitivity_map
#
#
# sl=0
# file_cli="./data/InVivo/3D/patient.003.v23/processed/meas_MID00258_FID82147_CUISSES_raFin_CLI_MRF_map_sl{}.pkl".format(sl)
# file_customir="./data/InVivo/3D/patient.003.v23/processed/meas_MID00259_FID82148_raFin_customIR_Reco_MRF_map_sl{}.pkl".format(sl)
#
# with open(file_cli,"rb") as file:
#     all_maps_cli=pickle.load(file)
#
#
# with open(file_customir,"rb") as file:
#     all_maps_customir=pickle.load(file)
#
#
# k="ff"
#
# import seaborn as sns
# plt.close("all")
# plt.figure()
# sns.kdeplot(data=all_maps_cli[0][0][k],bw_method=0.05)
# sns.kdeplot(data=all_maps_customir[0][0][k],bw_method=0.05)
#
#
# k="wT1"
#
#
# import seaborn as sns
# #plt.close("all")
# plt.figure()
# sns.kdeplot(data=
# all_maps_cli[0][0][k][all_maps_cli[0][0]["ff"]<0.7],bw_method=0.05)
# sns.kdeplot(data=
# all_maps_customir[0][0][k][all_maps_customir[0][0]["ff"]<0.7],bw_method=0.05)
#
#
# k="attB1"
#
# import seaborn as sns
# plt.close("all")
# plt.figure()
# sns.kdeplot(data=all_maps_cli[0][0][k],bw_method=0.05)
# sns.kdeplot(data=all_maps_customir[0][0][k],bw_method=0.05)
#
#
# k="df"
#
# import seaborn as sns
# plt.close("all")
# plt.figure()
# sns.kdeplot(data=all_maps_cli[0][0][k],bw_method=0.05)
# sns.kdeplot(data=all_maps_customir[0][0][k],bw_method=0.05)
#
#
#
#
#
#
#
# #import matplotlib
# #matplotlib.u<se("TkAgg")
# from mrfsim import T1MRF
# from image_series import *
# from dictoptimizers import SimpleDictSearch,BruteDictSearch
# from utils_mrf import *
# import json
# import readTwix as rT
# import time
# import os
# from numpy.lib.format import open_memmap
# from numpy import memmap
# import pickle
# from scipy.io import loadmat,savemat
# import twixtools
# from mutools import io
# import cv2
# import scipy
# from utils_reco import calculate_sensitivity_map
#
# #
# # volumes=h5py.File("/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data_Raw/patient.003.v23/meas_MID00258_FID82147_CUISSES_raFin_CLI/Reco1/ImgSeries.mat","r").get("Img")
# # volumes=np.array(volumes)
# # volumes_shape=volumes.shape
# # volumes=volumes.flatten()
# # volumes=np.array([complex(*x) for x in volumes])
# # volumes=volumes.reshape(volumes_shape)
# #
# # animate_images(np.abs(volumes[2]))
# #
# # np.save("./data/InVivo/3D/patient.003.v23/volumes_matlab_CLI.npy",volumes)
#
#
# volumes=np.load("./data/InVivo/3D/patient.003.v23/volumes_matlab_CLI.npy")
# mask=np.array(io.read("/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data_Raw/patient.003.v23/meas_MID00258_FID82147_CUISSES_raFin_CLI/Mask.mhd"))
#
#
# sl=-1
#
# plt.figure()
# plt.imshow(mask[:,:,sl])
#
# animate_images(np.moveaxis(volumes[sl],-1,-2))
#
#
# volumes_python=np.load("./data/InVivo/3D/patient.003.v23/processed/meas_MID00258_FID82147_CUISSES_raFin_CLI_volumes_sl{}.npy".format(sl))
#
#
#
# animate_multiple_images(np.flip(np.moveaxis(volumes[sl],-1,-2)),volumes_python,cmap="gray")
#
#
#
#
#
#
# dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1.14_reco5.0_w8_simmean.dict"
# dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1.14_reco5.0_w8_simmean.dict"
#
# dictfile="mrf_dictconf_SimReco2_adjusted_1.14_TI8_32_reco5.0_w8_simmean.dict"
# dictfile_light="mrf_dictconf_SimReco2_light_adjusted_1.14_TI8_32_reco5.0_w8_simmean.dict"
#
# filename_mask="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data_Raw/patient.003.v23/meas_MID00258_FID82147_CUISSES_raFin_CLI/Mask.mhd"
# filename_volume="./data/InVivo/3D/patient.003.v23/volumes_matlab_CLI.npy"
#
# sl=4
#
# mask_matlab=np.moveaxis(np.array(io.read(filename_mask)),-1,0)[sl]
# volumes_matlab=np.moveaxis(np.load(filename_volume),-1,-2)[sl]
#
# optimizer = SimpleDictSearch(mask=mask_matlab,niter=0,seq=None,trajectory=None,split=2000,pca=True,threshold_pca=15,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=175,b1=None,threshold_ff=0.9,dictfile_light=dictfile_light,mu=1,mu_TV=1,weights_TV=[1.,0.,0.],return_cost=False,clustering=True)#,mu_TV=1,weights_TV=[1.,0.,0.])
# all_maps_matlab=optimizer.search_patterns_test_multi_2_steps_dico(dictfile,volumes_matlab,retained_timesteps=None)
#
#
# optimizer_brute=BruteDictSearch(FF_list=np.arange(0,1.05,0.05),mask=mask_matlab,split=1,pca=True,threshold_pca=30,log=False,useGPU_dictsearch=True,n_clusters_dico=100,pruning=0.05,ntimesteps=175,dictfile_light=dictfile)
# all_maps_matlab=optimizer.search_patterns(dictfile,volumes_matlab)
#
#
# plt.close("all")
#
# k="wT1"
#
# image_map=makevol(all_maps_matlab[0][0][k],mask_matlab>0)
# #image_map[mask_ice.T==0]=0
# #image_map[mask_ice==0]=0
# #image_map[image_map>0.35]=0.35
# plt.figure()
# plt.imshow(image_map.T)
# plt.colorbar()
# plt.title("{} Map Matlab Volumes Python Matching Light".format(k))
#
#
#
#
#
# filename_mask_python="./data/InVivo/3D/patient.003.v23/processed/meas_MID00258_FID82147_CUISSES_raFin_CLI_mask_sl{}.npy".format(sl)
# filename_volume_python="./data/InVivo/3D/patient.003.v23/processed/meas_MID00258_FID82147_CUISSES_raFin_CLI_volumes_sl{}.npy".format(sl)
#
# if (sl==-1):
#     sl=4
#
# mask_python=np.load(filename_mask_python)
# volumes_python=np.load(filename_volume_python)
#
#
#
#
# optimizer = SimpleDictSearch(mask=mask_python,niter=0,seq=None,trajectory=None,split=2000,pca=True,threshold_pca=15,log=False,useGPU_dictsearch=True,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=175,b1=None,threshold_ff=0.9,dictfile_light=dictfile_light,mu=1,mu_TV=1,weights_TV=[1.,0.,0.],return_cost=False,clustering=True)#,mu_TV=1,weights_TV=[1.,0.,0.])
#
#
# all_maps_python=optimizer.search_patterns_test_multi_2_steps_dico(dictfile,volumes_python,retained_timesteps=None)
#
# k="attB1"
#
# image_map=np.flip(makevol(all_maps_python[0][0][k],mask_python>0),axis=1)
# #image_map[mask_ice.T==0]=0
# #image_map[mask_ice==0]=0
# #image_map[image_map>0.35]=0.35
# plt.figure()
# plt.imshow(image_map.T)
# plt.colorbar()
# plt.title("{} Map Python Volumes Python Matching".format(k))
#
#
# filename_map="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data_Raw/patient.003.v23/meas_MID00258_FID82147_CUISSES_raFin_CLI/Reco1/t1water_map.mhd"
# map_bmy=np.array(io.read(filename_map))[:,:,sl]
#
#
# plt.figure()
# plt.imshow(map_bmy.T)
# plt.colorbar()
# plt.title("{} Map Matlab Volumes Matlab Matching".format(k))
#





#import matplotlib
#matplotlib.u<se("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat
import twixtools
from mutools import io
import cv2
import scipy
from utils_reco import calculate_sensitivity_map
from utils_simu import *

sequence_file="./mrf_sequence_adjusted.json"
reco=4
min_TR_delay=2.22/1000

TR_,FA_,TE_=load_sequence_file(sequence_file,reco,min_TR_delay)

group_size=8

with open("./random_FA_US_varsp_seqOptim_config.json","rb") as file:
    config=json.load(file)

DFs = config["DFs"]
FFs = None
B1s = config["B1s"]
T1s = config["T1s"]
DFs_light = config["DFs_light"]
B1s_light = config["B1s_light"]
T1s_light = config["T1s_light"]

with open("./mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

fat_amp = np.array(dict_config["fat_amp"])
fat_shift = -np.array(dict_config["fat_cshift"])





s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
                                           amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
                                           return_fat_water=True,return_combined_signal=False)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

s_w = s_w.reshape(s_w.shape[0], -1).T
s_f = s_f.reshape(s_f.shape[0], -1).T

s_light, s_w_light, s_f_light, keys_light = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs_light, T1s_light, 300 / 1000, B1s_light, T_2w=40 / 1000,
                                                                   T_2f=80 / 1000,
                                                                   amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
                                                                   return_fat_water=True,
                                                                   return_combined_signal=False)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

s_w_light = s_w_light.reshape(s_w_light.shape[0], -1).T
s_f_light = s_f_light.reshape(s_f_light.shape[0], -1).T

#keys=np.array(keys)
#keys_light=np.array(keys_light)


keys=[(t[0]*1000,t[1]*1000,t[2],t[3]/1000) for t in keys]
keys_light=[(t[0]*1000,t[1]*1000,t[2],t[3]/1000) for t in keys_light]

# keys[:,0]=keys[:,0]*1000
# keys[:,1]=keys[:,1]*1000
# keys[:,-1]==keys[:,-1]/1000
# keys_light[:, 0] = keys_light[:, 0] * 1000
# keys_light[:, 1] = keys_light[:, 1] * 1000
# keys_light[:, -1] == keys_light[:, -1] / 1000

signals=s_w[:10].T

pca=True
threshold_pca_bc=10
split=100

dict_optim_bc_cf = SimpleDictSearch(mask=None, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True,dictfile_light=(s_w_light, s_f_light, keys_light),threshold_ff=0.9)

all_maps = dict_optim_bc_cf.search_patterns_test_multi_2_steps_dico((s_w, s_f, keys), signals)
matched_signals=all_maps[0][-1]

i=5
plt.figure()
plt.plot(signals[:,i])
plt.plot(matched_signals[:,i])

for k in all_maps[0][0].keys():
    print("{} : {}".format(k,all_maps[0][0][k][i]))

print(keys[i])

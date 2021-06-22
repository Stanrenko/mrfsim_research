
#import matplotlib
#matplotlib.use("TkAgg")
import numpy as np
from mrfsim import T1MRF
from image_series import *
from utils_mrf import radial_golden_angle_traj,animate_images,animate_multiple_images,compare_patterns,translation_breathing,find_klargest_freq,SearchMrf,basicDictSearch,compare_paramMaps,regression_paramMaps,dictSearchMemoryOptim,voronoi_volumes
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


## Random map simulation

dictfile = "mrf175.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)

seq = T1MRF(**sequence_config)

with open("mrf_dictconf.json") as f:
    dict_config = json.load(f)

dict_config["ff"]=np.arange(0.,1.05,0.05)

window = 8 #corresponds to nspoke by image
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
size=(256,256)

file_matlab_paramMap = "./data/paramMap.mat"

###### Building Map
#m = RandomMap("TestRandom",dict_config,image_size=size,region_size=region_size,mask_reduction_factor=1/4)
m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)
m.buildParamMap()
#m.plotParamMap(figsize=(5,5))
#m.plotParamMap("df",figsize=(5,5))

##### Simulating Ref Images
m.build_ref_images(seq,window)
m.change_resolution(compression_factor=3)
ani=animate_images(m.images_series)

#print(m.t[0])
#plt.imshow(np.angle(m.fat_series[0]))
#plt.colorbar()

image_series = m.images_series

##### Movement

direction = np.array([4,0])
shifts_t = lambda t:translation_breathing(t,direction)
m.translate_images(shifts_t,round=True)
#ani_shift=animate_images(m.images_series)
images_series_with_movement = m.images_series

#m.reset_image_series()

#pixel=(125,125)
#m.compare_patterns(pixel)


###### Undersampling k space

npoint = 2*image_series.shape[0]
total_nspoke=8*175
nspoke=8

all_spokes=radial_golden_angle_traj(total_nspoke,npoint)
traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))

image_series_rebuilt_with_movement = m.simulate_radial_undersampled_images(traj,density_adj=True,npoint=npoint)

# vol,vor =voronoi_volumes(np.transpose(np.array([traj[0].real,traj[0].imag])))
#
# from scipy.spatial import voronoi_plot_2d
#
# import matplotlib.pyplot as plt
# fig = voronoi_plot_2d(vor)
# plt.show()



#ani=animate_images(images_series)
#ani_r=animate_images(images_series_rebuilt)

#ani1,ani2= animate_multiple_images(image_series,image_series_rebuilt)

m.reset_image_series()
image_series_rebuilt = m.simulate_radial_undersampled_images(traj,density_adj=True,npoint=npoint)
print("Simulation US images")
image_series_rebuilt_2 = m.simulate_undersampled_images(traj,density_adj=True)

ani,ani1=animate_multiple_images(image_series_rebuilt,image_series_rebuilt_2)

#masked_images_rebuilt_mvt = np.transpose(np.array(image_series_rebuilt_with_movement)[:,m.mask>0])
#masked_images_rebuilt = np.transpose(np.array(image_series_rebuilt)[:,m.mask>0])
#masked_images = np.transpose(np.array(image_series)[:,m.mask>0])
#masked_images_mvt = np.transpose(np.array(images_series_with_movement)[:,m.mask>0])
#correl_window=10

# for j in range(1,image_series.shape[0]-correl_window):
#     current_correl=np.corrcoef(masked_images_rebuilt[:,(j-1):(j-1+correl_window)],masked_images_rebuilt[:,j:(j+correl_window)])
#     correlations.append(current_correl)


#fft_orig=np.fft.fft(masked_images,axis=1)
#fft_mvt=np.fft.fft(masked_images_mvt,axis=1)
#fft_rebuilt=np.fft.fft(masked_images_rebuilt,axis=1)
#fft_rebuilt_mvt=np.fft.fft(masked_images_rebuilt_mvt,axis=1)

########## Dictionary matching ##########################
# Original data
# kdata_allspokes = m.generate_kdata(traj)
# res_matching=SearchMrf(kdata_allspokes,traj, dictfile, 1, "brute", "ls", m.image_size,8,density_adj=True, setup_opts={}, search_opts= {})
# plt.imshow(np.abs(res_matching["b1map"]))

#
#
# m.translate_images(shifts_t,round=True)
# kdata_allspokes_mvt = m.generate_kdata(traj)
# res_matching_movement=SearchMrf(kdata_allspokes_mvt,all_spokes, dictfile, 1, "brute", "ls", m.image_size,8,density_adj=True, setup_opts={}, search_opts= {})
# plt.imshow(np.abs(res_matching_movement["b1map"]))





#
# var_w_pixel = var_w[idx_max]
# var_f_pixel = var_f[idx_max]
# sig_wf_pixel = sig_wf[idx_max]
#
# sig_ws_pixel = sig_ws[idx_max]
# sig_fs_pixel = sig_fs[idx_max]
#
# def J_(alpha):
#     return ((1-alpha)*sig_ws_pixel + alpha * sig_fs_pixel)/np.sqrt((1-alpha)**2*var_w_pixel+alpha**2*var_f_pixel+2*alpha*(1-alpha)*sig_wf_pixel)
#
# alpha = np.arange(0.001,1.0,0.01)
# plt.figure()
# plt.plot(alpha,J_(alpha))
# np.argmax(J_(alpha))

all_signals = m.images_series[:,m.mask>0]

#map_rebuilt = basicDictSearch(all_signals,dictfile)
#compare_paramMaps(m.paramMap,map_rebuilt,m.mask>0,adj_wT1=True)
#regression_paramMaps(m.paramMap,map_rebuilt,adj_wT1=True,fat_threshold=0.8)




#map_rebuilt=dictSearchMemoryOptim(all_signals,dictfile,pca=True,threshold_pca=0.999999,split=2000)
#map_rebuilt=dictSearchMemoryOptimIterative(image_series,dictfile,seq,traj,npoint,niter=1)
map_rebuilt=m.dictSearchMemoryOptimIterative(dictfile,seq,traj,npoint,niter=1,split=1000)
compare_paramMaps(m.paramMap,map_rebuilt,m.mask>0,adj_wT1=True)
regression_paramMaps(m.paramMap,map_rebuilt,adj_wT1=True,fat_threshold=0.8)

map_rebuilt_it1=m.dictSearchMemoryOptimIterative(dictfile,seq,traj,npoint,niter=0,split=1000)
compare_paramMaps(m.paramMap,map_rebuilt_it1,m.mask>0,adj_wT1=True)
regression_paramMaps(m.paramMap,map_rebuilt_it1,adj_wT1=True,fat_threshold=0.8)

map_rebuilt_us=dictSearchMemoryOptim(np.array(image_series_rebuilt)[:,m.mask>0],dictfile,pca=True,threshold_pca=0.999999,split=2000)
compare_paramMaps(m.paramMap,map_rebuilt_us,m.mask>0,adj_wT1=True)
regression_paramMaps(m.paramMap,map_rebuilt_us,adj_wT1=True,fat_threshold=0.8)



##############################################""
from sklearn.decomposition import PCA,TruncatedSVD
from scipy import linalg



mrfdict = dictsearch.Dictionary()
mrfdict.load(dictfile, force=True)

keys=mrfdict.keys
array_water = mrfdict.values[:, :, 0]
array_fat = mrfdict.values[:, :, 1]

del mrfdict

array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

nb_patterns=array_water.shape[0]
nb_signals= all_signals.shape[1]

del array_water
del array_fat


threshold_pca=0.999999




cov_fat = np.matmul(np.transpose(array_fat_unique.conj()),array_fat_unique)
cov_water = np.matmul(np.transpose(array_water_unique.conj()),array_water_unique)

fat_val,fat_vect  = np.linalg.eigh(cov_fat)
water_val,water_vect  = np.linalg.eigh(cov_water)

sorted_index_fat=np.argsort(fat_val)[::-1]
fat_val=fat_val[sorted_index_fat]
fat_vect=fat_vect[:,sorted_index_fat]

sorted_index_water=np.argsort(water_val)[::-1]
water_val=water_val[sorted_index_water]
water_vect=water_vect[:,sorted_index_water]

explained_variance_ratio_fat = np.cumsum(fat_val**2)/np.sum(fat_val**2)
n_components_fat = np.sum(explained_variance_ratio_fat<threshold_pca)+1


explained_variance_ratio_water = np.cumsum(water_val**2)/np.sum(water_val**2)
n_components_water = np.sum(explained_variance_ratio_water<threshold_pca)+1



fat_vect=fat_vect[:,:n_components_fat]
water_vect=water_vect[:,:n_components_water]

transformed_array_water_unique = np.matmul(array_water_unique,water_vect.conj())
transformed_array_fat_unique = np.matmul(array_fat_unique,fat_vect.conj())

#retrieved_array_water_unique = np.matmul(transformed_array_water_unique,np.transpose(water_vect))
#retrieved_array_fat_unique = np.matmul(transformed_array_fat_unique,np.transpose(fat_vect))

#plt.figure()
#plt.plot(retrieved_array_fat_unique[0,:].real)
#plt.plot(array_fat_unique[0,:].real)

transformed_all_signals_water = np.transpose(np.matmul(np.transpose(all_signals),water_vect.conj()))
transformed_all_signals_fat = np.transpose(np.matmul(np.transpose(all_signals),fat_vect.conj()))

var_w = np.sum(array_water_unique* array_water_unique.conj(), axis=1).real
var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

var_w = var_w[index_water_unique]
var_f = var_f[index_fat_unique]



print("Calculating correlations")
sig_ws_all_unique = np.matmul(transformed_array_water_unique, transformed_all_signals_water[:, :].conj()).real
sig_fs_all_unique = np.matmul(transformed_array_fat_unique, transformed_all_signals_fat[:, :].conj()).real




print("Calculating optimal fat fraction and best pattern per signal")
var_w = np.reshape(var_w, (-1, 1))
var_f = np.reshape(var_f, (-1, 1))
sig_wf = np.reshape(sig_wf, (-1, 1))

alpha_all_unique=np.zeros((nb_patterns,nb_signals))
J_all=np.zeros(alpha_all_unique.shape)

num_signal = alpha_all_unique.shape[1]
split=2000
num_group = int(num_signal/split )+1

for j in (range(num_group)):
    print(j/num_group)
    j_signal = j*split
    j_signal_next = np.minimum((j+1)*split,num_signal)
    current_sig_ws = sig_ws_all_unique[index_water_unique,j_signal:j_signal_next]
    current_sig_fs = sig_fs_all_unique[index_fat_unique,j_signal:j_signal_next]
    current_alpha_all_unique =  (sig_wf * current_sig_ws-var_w*current_sig_fs) / ((current_sig_ws + current_sig_fs) * sig_wf-var_w*current_sig_fs-var_f*current_sig_ws)
    alpha_all_unique[:,j_signal:j_signal_next] = current_alpha_all_unique
    J_all[:,j_signal:j_signal_next] = ((1-current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
    (1-current_alpha_all_unique)  ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (1-current_alpha_all_unique)  * sig_wf)

idx_max_all_unique = np.argmax(J_all, axis=0)
#idx_max_all = idx_max_all_unique[index_signals_unique]
#idx_max_all_unique = np.argmax(J_all, axis=0)
    #idx_max_all = idx_max_all_unique[index_signals_unique]
del J_all

print("Building the maps")
#alpha_all = alpha_all_unique[:, index_signals_unique]

del sig_ws_all_unique
del sig_fs_all_unique

params_all_unique = np.array([keys[idx] + (alpha_all_unique[idx, i],) for i, idx in enumerate(idx_max_all_unique)])
params_all = params_all_unique

map_rebuilt= {
            "wT1": params_all[:, 0],
            "fT1": params_all[:, 1],
            "attB1": params_all[:, 2],
            "df": params_all[:, 3],
            "ff": params_all[:, 4]

            }

compare_paramMaps(m.paramMap,map_rebuilt,m.mask>0,adj_wT1=True)
regression_paramMaps(m.paramMap,map_rebuilt,adj_wT1=True,fat_threshold=0.8)




# mapff_rebuilt = makevol(map_rebuilt["ff"],m.mask>0)
# mapwT1_rebuilt = makevol(map_rebuilt["wT1"],m.mask>0)
# mapwT1 = makevol(m.paramMap["wT1"],m.mask>0)
# mapwT1_rebuilt[mapff_rebuilt>0.8]=mapwT1[mapff_rebuilt>0.8]
# error = np.abs(mapwT1_rebuilt-mapwT1)
# i_max,j_max = np.unravel_index(np.argmax(error),error.shape)
# print(error[i_max,j_max])
#
# ###### SIGNAL MATCHING SINGLE PIXEL #######################
#
# pixel = (i_max,j_max)
#
# signal = m.images_series[:,pixel[0],pixel[1]]
# #signal_rebuilt =np.array(image_series_rebuilt)[:,pixel[0],pixel[1]]
#
# ground_truth={}
# for key in m.paramMap.keys():
#     volume = makevol(m.paramMap[key],m.mask>0)
#     param = volume[pixel[0],pixel[1]]
#     ground_truth[key]=param
#
# mrfdict = dictsearch.Dictionary()
# mrfdict.load(dictfile, force=True)
#
# array_water =  mrfdict.values[:,:,0]
# array_fat = mrfdict.values[:,:,1]
#
# var_w= np.sum(array_water*array_water.conj(),axis=1).real
# var_f= np.sum(array_fat*array_fat.conj(),axis=1).real
# sig_wf = np.sum(array_water*array_fat.conj(),axis=1).real
#
# array_water_unique,index_water_unique = np.unique(array_water,axis=0,return_inverse=True)
# array_fat_unique,index_fat_unique = np.unique(array_fat,axis=0,return_inverse=True)
#
# sig_ws_unique = np.matmul(array_water_unique,signal.conj()).real
# sig_fs_unique = np.matmul(array_fat_unique,signal.conj()).real
#
# sig_ws = sig_ws_unique[index_water_unique]
# sig_fs = sig_fs_unique[index_fat_unique]
#
# alpha0 = -(var_w*sig_fs-sig_wf*sig_ws)/((sig_ws+sig_fs)*(sig_wf-var_f))
# J = ((1-alpha0)*sig_ws + alpha0 * sig_fs)/np.sqrt((1-alpha0)**2*var_w+alpha0**2*var_f+2*alpha0*(1-alpha0)*sig_wf)
# idx_max = np.argmax(J)
#
# var_s =np.sum(signal*signal.conj()).real
# final_correl = J[idx_max]/np.sqrt(var_s)
# params = mrfdict.keys[idx_max]
# ff = alpha0[idx_max]
# print("Correl : {}".format(np.round(final_correl,4)))
# print("Params vs Ground Truth")
# print("wT1: {} vs {}".format(params[0],ground_truth["wT1"]))
# print("fT1: {} vs {}".format(params[1],ground_truth["fT1"]))
# print("attB1: {} vs {}".format(params[2],ground_truth["attB1"]))
# print("df: {} vs {}".format(params[3],ground_truth["df"]))
# print("ff: {} vs {}".format(np.round(ff,2),ground_truth["ff"]))
#
# plt.figure()
# plt.plot(signal.real,label="Signal")
# plt.plot(array_water[idx_max].real*(1-ff)+ff*array_fat[idx_max].real,label="Matched Pattern")
# plt.legend()
#
# keys=np.array(mrfdict.keys)
# dic_keys = {}
#
# dic_keys["wT1"]=np.unique(keys[:,0])
# dic_keys["fT1"]=np.unique(keys[:,1])
# dic_keys["attB1"]=np.unique(keys[:,2])
# dic_keys["df"]=np.unique(keys[:,3])
#
#
# dic_nearest={}
# for i,param in enumerate(dic_keys.keys()):
#     idx_nearest=np.argmin(np.abs(dic_keys[param]-ground_truth[param]))
#     print(idx_nearest)
#     dic_nearest[param]=dic_keys[param][idx_nearest]
#
# params_closest = tuple(dic_nearest.values())
# idx_params_closest = mrfdict.keys.index(tuple(dic_nearest.values()))
# signal_closest_params = mrfdict[params_closest][:,0]*(1-ground_truth["ff"])+mrfdict[params_closest][:,1]*(ground_truth["ff"])
# plt.figure()
# plt.plot(signal.real,label="Signal")
# plt.plot(array_water[idx_max].real*(1-ff)+ff*array_fat[idx_max].real,label="Matched Pattern")
# plt.plot(signal_closest_params.real,label="Closest Params Pattern")
# plt.legend()
#
# ff_params_closest = alpha0[idx_params_closest]
#
# var_w_pixel = var_w[idx_params_closest]
# var_f_pixel = var_f[idx_params_closest]
# sig_wf_pixel = sig_wf[idx_params_closest]
#
# sig_ws_pixel = sig_ws[idx_params_closest]
# sig_fs_pixel = sig_fs[idx_params_closest]
#
# def J_(alpha):
#     return ((1-alpha)*sig_ws_pixel + alpha * sig_fs_pixel)/np.sqrt((1-alpha)**2*var_w_pixel+alpha**2*var_f_pixel+2*alpha*(1-alpha)*sig_wf_pixel)
#
# def J_prime(alpha,eps=0.00001):
#     return (J_(alpha+eps)-J_(alpha-eps))/(2*eps)
#
# alpha = np.arange(0.001,1.0,0.01)
# plt.figure()
# plt.plot(alpha,J_(alpha))
# plt.figure()
# plt.plot(alpha,J_prime(alpha))
# alpha[np.argmax(J_(alpha))]






#all_signals_rebuilt=np.array(image_series_rebuilt)[:,m.mask>0]
#map_rebuilt_undersampling = basicDictSearch(all_signals_rebuilt,dictfile)


#
# red_factor = 1/3
# mask = np.zeros(size)
# mask[int(size[0] *red_factor):int(size[0]*(1-red_factor)), int(size[1] *red_factor):int(size[1]*(1-red_factor))] = 1.0
#
# test = mask
#
# density = np.abs(np.linspace(-1, 1, npoint))
# f_samples = traj[0]
#
# ft_test = finufft.nufft2d2(f_samples.real, f_samples.imag, test)
# ft_test /= len(ft_test)
#
# ft_test_density_adj=(np.reshape(ft_test,(-1,npoint))*density).flatten()
#
# retrieved_test = finufft.nufft2d1(f_samples.real, f_samples.imag, ft_test_density_adj, size)
# plt.imshow(np.abs(retrieved_test))
# plt.colorbar()
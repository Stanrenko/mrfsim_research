
#import matplotlib
#matplotlib.use("TkAgg")
import json
import os

from image_series import *
from mrfsim import T1MRF
from utils_mrf import radial_golden_angle_traj, translation_breathing, SearchMrf,animate_multiple_images,animate_images,compare_patterns

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

file_matlab_paramMap = "/home/cslioussarenko/PythonRepositories/mrf-sim/data/paramMap.mat"

###### Building Map
m = RandomMap("TestRandom",dict_config,image_size=size,region_size=region_size,mask_reduction_factor=1/4)
#m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)
m.buildParamMap()
#m.plotParamMap(figsize=(5,5))
#m.plotParamMap("df",figsize=(5,5))

##### Simulating Ref Images
m.build_ref_images(seq,window)
#ani=animate_images(m.images_series,metric=np.angle)

#print(m.t[0])
#plt.imshow(np.angle(m.fat_series[0]))
#plt.colorbar()

image_series = m.images_series

##### Movement

direction = np.array([15,0])
shifts_t = lambda t:translation_breathing(t,direction)
m.translate_images(shifts_t,round=True)
#ani_shift=animate_images(m.images_series)
images_series_with_movement = m.images_series

#m.reset_image_series()

#pixel=(125,125)
#m.compare_patterns(pixel)


###### Undersampling k space

npoint = 512
total_nspoke=8*175
nspoke=8

all_spokes=radial_golden_angle_traj(total_nspoke,npoint)
traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))

image_series_rebuilt_with_movement = m.simulate_radial_undersampled_images(traj,density_adj=True,npoint=npoint)

#ani=animate_images(images_series)
#ani_r=animate_images(images_series_rebuilt)



m.reset_image_series()
image_series_rebuilt = m.simulate_radial_undersampled_images(traj,density_adj=True,npoint=npoint)


#ani1,ani2= animate_multiple_images(image_series_rebuilt,image_series_rebuilt_with_movement)

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
kdata_allspokes = np.array(m.generate_kdata(traj))
res_matching=SearchMrf(kdata_allspokes,traj, dictfile, 1, "brute", "ls", m.image_size,8,density_adj=True, setup_opts={}, search_opts= {})
plt.imshow(np.abs(res_matching["b1map"]))


m.translate_images(shifts_t,round=True)
kdata_allspokes_mvt = m.generate_kdata(traj)
res_matching_movement=SearchMrf(kdata_allspokes_mvt,all_spokes, dictfile, 1, "brute", "ls", m.image_size,8,density_adj=True, setup_opts={}, search_opts= {})
plt.imshow(np.abs(res_matching_movement["ffmap"]))

pixel =(125,125)
signal = m.images_series[:,pixel[0],pixel[1]]
signal_rebuilt = np.array(image_series_rebuilt)[:,pixel[0],pixel[1]]
mrfdict = dictsearch.Dictionary();mrfdict.load(dictfile, force=True)

all_signals=m.images_series[:,m.mask>0]

array_water = mrfdict.values[:,:,0]
array_fat = mrfdict.values[:,:,1]

sig_w = np.std(array_water,axis=1)
sig_f = np.std(array_fat,axis=1)
m_w = np.mean(array_water,axis=1)
m_f =np.mean(array_fat,axis=1)

sig_w_f = np.sum((array_water*array_fat.conj()),axis=1)

sig_w_s = np.matmul(array_water,signal.conj())
sig_w_f = np.matmul(array_fat,signal.conj())



red_factor = 1/3
mask = np.zeros(size)
mask[int(size[0] *red_factor):int(size[0]*(1-red_factor)), int(size[1] *red_factor):int(size[1]*(1-red_factor))] = 1.0

test = mask

density = np.abs(np.linspace(-1, 1, npoint))
f_samples = traj[0]

ft_test = finufft.nufft2d2(f_samples.real, f_samples.imag, test)
ft_test /= len(ft_test)

ft_test_density_adj=(np.reshape(ft_test,(-1,npoint))*density).flatten()

retrieved_test = finufft.nufft2d1(f_samples.real, f_samples.imag, ft_test_density_adj, size)
plt.imshow(np.abs(retrieved_test))
plt.colorbar()
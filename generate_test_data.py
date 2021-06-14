
#import matplotlib
#matplotlib.use("TkAgg")
import numpy as np
from mrfsim import T1MRF
from image_series import *
from utils_mrf import radial_golden_angle_traj,animate_images,animate_multiple_images,compare_patterns,translation_breathing,find_klargest_freq
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal
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
m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)
m.buildParamMap()
#m.plotParamMap(figsize=(5,5))
#rand_map.plotParamMap("wT1",figsize=(5,5))

##### Simulating Ref Images
m.build_ref_images(seq,window)
#ani=animate_images(m.images_series)

image_series = m.images_series

#####



direction=np.flip(np.array([4,0]))
direction = np.array([4,0])

shifts_t = lambda t:translation_breathing(t,direction)
m.translate_images(shifts_t,round=True)
#ani_shift=animate_images(m.images_series)
images_series_with_movement = m.images_series

#m.reset_image_series()

pixel=(125,125)
m.compare_patterns(pixel)




###### Undersampling k space

npoint = size[0]
total_nspoke=1400
nspoke=window

all_spokes=radial_golden_angle_traj(total_nspoke,npoint)
traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))

image_series_rebuilt_with_movement = m.simulate_undersampled_images(traj)


#ani=animate_images(images_series)
#ani_r=animate_images(images_series_rebuilt)

#ani1,ani2= animate_multiple_images(m.images_series,image_series_rebuilt)

m.reset_image_series()
image_series_rebuilt = m.simulate_undersampled_images(traj)

compare_patterns(pixel,np.array(image_series_rebuilt),np.array(image_series_rebuilt_with_movement),title_1="Images rebuilt {} spokes".format(nspoke),title_2="Images with movement rebuilt {} spokes".format(nspoke))

correlations=[]

masked_images_rebuilt_mvt = np.transpose(np.array(image_series_rebuilt_with_movement)[:,m.mask>0])
masked_images_rebuilt = np.transpose(np.array(image_series_rebuilt)[:,m.mask>0])
masked_images = np.transpose(np.array(image_series)[:,m.mask>0])
masked_images_mvt = np.transpose(np.array(images_series_with_movement)[:,m.mask>0])
#correl_window=10

# for j in range(1,image_series.shape[0]-correl_window):
#     current_correl=np.corrcoef(masked_images_rebuilt[:,(j-1):(j-1+correl_window)],masked_images_rebuilt[:,j:(j+correl_window)])
#     correlations.append(current_correl)


fft_orig=np.fft.fft(masked_images,axis=1)
fft_mvt=np.fft.fft(masked_images_mvt,axis=1)
fft_rebuilt=np.fft.fft(masked_images_rebuilt,axis=1)
fft_rebuilt_mvt=np.fft.fft(masked_images_rebuilt_mvt,axis=1)
min_t = np.min(m.t)
max_t = np.max(m.t)
a = 2*np.pi/(max_t-min_t)
b = np.pi*(min_t+max_t)/(min_t-max_t)
t_for_nufft = a*m.t+b

fft_orig=np.array([nufft1d1(t_for_nufft,masked_images[j,:],masked_images.shape[1]) for j in range(masked_images.shape[0])])
fft_mvt=np.array([nufft1d1(t_for_nufft,masked_images_mvt[j,:],masked_images_mvt.shape[1]) for j in range(masked_images_mvt.shape[0])])
fft_rebuilt=np.array([nufft1d1(t_for_nufft,masked_images_rebuilt[j,:],masked_images_rebuilt.shape[1]) for j in range(masked_images_rebuilt.shape[0])])
fft_rebuilt_mvt=np.array([nufft1d1(t_for_nufft,masked_images_rebuilt_mvt[j,:],masked_images_rebuilt_mvt.shape[1]) for j in range(masked_images_rebuilt_mvt.shape[0])])

plt.figure()
plt.plot(np.abs(fft_orig[int(len(fft_orig)/2),:]),label="No movement")
plt.plot(np.abs(fft_mvt[int(len(fft_orig)/2),:]),label="With movement")
plt.title("Full Sampling")
plt.legend()

plt.figure()
plt.plot(np.mean(np.abs(fft_orig),axis=0),label="No movement")
plt.plot(np.mean(np.abs(fft_mvt),axis=0),label="With movement")
plt.title("Mean of all pixels - Full Sampling")
plt.legend()

plt.figure()
plt.plot(np.abs(fft_rebuilt[int(len(fft_orig)/2),:]),label="No movement")
plt.plot(np.abs(fft_rebuilt_mvt[int(len(fft_orig)/2),:]),label="With Movement")
plt.title("Radial Sampling {} Spokes".format(nspoke))
plt.legend()

plt.figure()
plt.plot(np.mean(np.abs(fft_rebuilt),axis=0),label="No movement")
plt.plot(np.mean(np.abs(fft_rebuilt_mvt),axis=0),label="With movement")
plt.title("Mean of all pixels - Radial Sampling {} Spokes".format(nspoke))
plt.legend()


# Fourier Transform of the breathing function
t_list=np.arange(0.,max_t,max_t/175)
shifts_t_regular_spacing = np.array([shifts_t(t_) for t_ in t_list])
#plt.plot(t_list,shifts_t_regular_spacing[:,0])
ft_movement=np.fft.fft(shifts_t_regular_spacing[:,0])
max_freq=np.argmax((np.abs(ft_movement))[1:])
second_max_freq = np.argsort((np.abs(ft_movement))[1:int(len(ft_movement)/2)])[-2]
period = max_t/175*175/max_freq

# Fourier Transform on all images with radial sampling
#ft_movement_in_image=np.fft.fftshift(np.abs(np.mean(fft_rebuilt_mvt,axis=0)))
ft_movement_in_image=np.fft.fftshift(np.mean(np.abs(fft_rebuilt_mvt),axis=0))
max_freq_image=find_klargest_freq(ft_movement_in_image,k=1)
#first_10_comp=list(map(lambda k_:find_klargest_freq(ft_movement_in_image,k_),list(range(1,10))))
period_image = max_t/175*175/max_freq_image
freq_image=1/period_image

plt.plot(ft_movement_in_image)

# Quick test - Fourier Transform on all images with full sampling
ft_movement_in_image=np.fft.fftshift(np.mean(np.abs(fft_mvt),axis=0))
max_freq_image=find_klargest_freq(ft_movement_in_image,k=1)
#first_10_comp=list(map(lambda k_:find_klargest_freq(ft_movement_in_image,k_),list(range(1,10))))
period_image = max_t/175*175/max_freq_image
freq_image=1/period_image
freq_image_theo=1/300


fs =  (max_t/175)**(-1) # Sample frequency (Hz)
sos = signal.butter(10, [freq_image*(1-0.1),freq_image*(1+0.1),], 'bs', fs=fs, output='sos')
filtered = signal.sosfilt(sos, shifts_t_regular_spacing[:,0])


notch_freq = freq_image_theo  # Frequency to be removed from signal (Hz)
quality_factor = 30.0  # Quality factor
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
filtered_1=signal.filtfilt(b_notch, a_notch, shifts_t_regular_spacing[:,0])
b_notch, a_notch = signal.iirnotch(second_max_freq/max_t, quality_factor, fs)
filtered=signal.filtfilt(b_notch, a_notch, filtered_1)

plt.plot(t_list, filtered)
plt.plot(t_list, shifts_t_regular_spacing[:,0])



x = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
y = np.exp(-x/3.0)
flinear = interpolate.interp1d(x, y)
fcubic = interpolate.interp1d(x, y, kind='cubic')

xnew = np.arange(0.001, 20, 1)
ylinear = flinear(xnew)
ycubic = fcubic(xnew)

fig,(ax1,ax2) = plt.subplots(1,2)

filtered_image_orig=signal.sosfilt(sos,np.array(image_series_rebuilt)[:,pixel[0],pixel[1]])
filtered_image_move=signal.sosfilt(sos,np.array(image_series_rebuilt_with_movement)[:,pixel[0],pixel[1]])

title_1 = "Orig rebuilt"
title_2 = "Movement rebuilt "

ax1.plot(np.real(filtered_image_orig),label=title_1+" - real part")
ax1.plot(np.real(filtered_image_move), label=title_2+" - real part")
ax1.legend()
ax2.plot(np.imag(filtered_image_orig),
         label=title_1+" - imaginary part")
ax2.plot(np.imag(filtered_image_move),
         label=title_2+" - imaginary part")
ax2.legend()

plt.show()


fig,(ax1,ax2) = plt.subplots(1,2)

filtered_image_orig=signal.sosfilt(sos,image_series[:,pixel[0],pixel[1]])
filtered_image_move=signal.sosfilt(sos,np.array(images_series_with_movement)[:,pixel[0],pixel[1]])

title_1 = "Orig"
title_2 = "Movement "

ax1.plot(np.real(filtered_image_orig),label=title_1+" - real part")
ax1.plot(np.real(filtered_image_move), label=title_2+" - real part")
ax1.legend()
ax2.plot(np.imag(filtered_image_orig),
         label=title_1+" - imaginary part")
ax2.plot(np.imag(filtered_image_move),
         label=title_2+" - imaginary part")
ax2.legend()

plt.show()
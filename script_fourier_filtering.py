
#import matplotlib
#matplotlib.use("TkAgg")
import numpy as np
from mrfsim import T1MRF
from image_series import *
from utils_mrf import radial_golden_angle_traj,animate_images,animate_multiple_images,compare_patterns,translation_breathing,find_klargest_freq
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
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


#fft_orig=np.fft.fft(masked_images,axis=1)
#fft_mvt=np.fft.fft(masked_images_mvt,axis=1)
#fft_rebuilt=np.fft.fft(masked_images_rebuilt,axis=1)
#fft_rebuilt_mvt=np.fft.fft(masked_images_rebuilt_mvt,axis=1)

##### TEMPORAL FOURIER ANALYSIS  ####################"
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

######### FINDING MAIN SECONDARY FREQ (BREATHING FREQ) ###################
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

#######FILTERING AT BREATHING FREQ ###############"
fs =  (max_t/175)**(-1) # Sample frequency (Hz)
sos = signal.butter(10, [freq_image_theo*(1-0.1),freq_image_theo*(1+0.1),], 'bs', fs=fs, output='sos')
filtered = signal.sosfilt(sos, shifts_t_regular_spacing[:,0])


# notch_freq = freq_image_theo  # Frequency to be removed from signal (Hz)
# quality_factor = 30.0  # Quality factor
# b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
# filtered_1=signal.filtfilt(b_notch, a_notch, shifts_t_regular_spacing[:,0])
# b_notch, a_notch = signal.iirnotch(second_max_freq/max_t, quality_factor, fs)
# filtered=signal.filtfilt(b_notch, a_notch, filtered_1)

plt.figure()
plt.plot(t_list, filtered)
plt.plot(t_list, shifts_t_regular_spacing[:,0])


################# FILTERING NON UNIFORMLY SAMPLED DATA #############################
resampled_images_rebuilt_pixel = (interpolate.interp1d(m.t,np.real(np.array(image_series_rebuilt)[:,pixel[0],pixel[1]]),fill_value="extrapolate"))(t_list)+1j*(interpolate.interp1d(m.t,np.imag(np.array(image_series_rebuilt)[:,pixel[0],pixel[1]]),fill_value="extrapolate"))(t_list)
resampled_images_rebuilt_movement_pixel = (interpolate.interp1d(m.t,np.real(np.array(image_series_rebuilt_with_movement)[:,pixel[0],pixel[1]]),fill_value="extrapolate"))(t_list)+1j*(interpolate.interp1d(m.t,np.imag(np.array(image_series_rebuilt_with_movement)[:,pixel[0],pixel[1]]),fill_value="extrapolate"))(t_list)

filtered_image_orig=signal.sosfilt(sos,resampled_images_rebuilt_pixel)
filtered_image_move=signal.sosfilt(sos,resampled_images_rebuilt_movement_pixel)

resampled_filtered_images_rebuilt_movement_pixel_grid_orig = (interpolate.interp1d(t_list,np.real(filtered_image_move),fill_value="extrapolate"))(m.t)+1j*(interpolate.interp1d(t_list,np.imag(filtered_image_move),fill_value="extrapolate"))(m.t)

image_orig = np.array(image_series)[:,pixel[0],pixel[1]]
image_orig_rebuilt = np.array(image_series_rebuilt)[:,pixel[0],pixel[1]]

image_orig_rebuilt_movement =np.array(image_series_rebuilt_with_movement)[:,pixel[0],pixel[1]]
image_orig_rebuilt_movement_filtered = resampled_filtered_images_rebuilt_movement_pixel_grid_orig

fig,(ax1,ax2) = plt.subplots(1,2)
title_1 = "Orig"
title_2 = "Movement rebuilt "
title_3 = "Movement rebuilt post filter"

ax1.plot(np.real(image_orig),label=title_1+" - real part")
ax1_1=ax1.twinx()
ax1_1.plot(np.real(image_orig_rebuilt_movement), label=title_2+" - real part")
ax1_1.plot(np.real(image_orig_rebuilt_movement_filtered), label=title_3+" - real part")
ax1.legend()
ax1_1.legend()
ax2.plot(np.imag(image_orig),
         label=title_1+" - imaginary part")
ax2_1=ax2.twinx()
ax2_1.plot(np.imag(image_orig_rebuilt_movement),
         label=title_2+" - imaginary part")
ax2_1.plot(np.imag(image_orig_rebuilt_movement_filtered), label=title_3+" - imag part")
ax2.legend()
ax2_1.legend()

plt.show()


corr_orig = np.abs(np.corrcoef(image_orig,image_orig_rebuilt))[0,1]
corr_due_to_movement = np.abs(np.corrcoef(image_orig,image_orig_rebuilt_movement))[0,1]
corr_post_filtering = np.abs(np.corrcoef(image_orig,image_orig_rebuilt_movement_filtered))[0,1]

print("The correlation with dictionary signal without movement is : {}".format(corr_orig))
print("With movement, the correlation goes down to {}".format(corr_due_to_movement))
print("When filtering for breathing frequency, correlation goes to {}".format(corr_post_filtering))



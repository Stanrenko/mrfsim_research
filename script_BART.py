
import numpy as np
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,GaussianWeighting,BruteDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat
from mutools import io
from sklearn import linear_model
from scipy.optimize import minimize
from movements import TranslationBreathing
from bart import bart
import matplotlib.pyplot as plt

import cfl


base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./2D"

dictjson="mrf_dictconf_SimReco2_light.json"
#dictfile = "mrf175_SimReco2.dict"
#dictfile = "mrf175_SimReco2_window_1.dict"
#dictfile = "mrf175_SimReco2_window_21.dict"
#dictfile = "mrf175_SimReco2_window_55.dict"
#dictfile = "mrf175_Dico2_Invivo.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)

seq = T1MRF(**sequence_config)


nb_phantom = 5

ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 512

image_size = (int(npoint/2), int(npoint/2))
nspoke=int(nb_allspokes/ntimesteps)

with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.01,0.01)

region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
mask_reduction_factor=1/4


name = "SquareSimu2D_bart"
name="Knee2D_bart"

use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

folder = "/".join(str.split(filename,"/")[:-1])

all_results_cf={}

num=0

print("################## PROCESSING Phantom {} ##################".format(num))
filename_paramMap=filename+"_paramMap_{}.pkl".format(num)
filename_paramMask=filename+"_paramMask_{}.npy".format(num)
filename_groundtruth = filename+"_groundtruth_{}.npy".format(num)

filename_kdata = filename+"_kdata_{}.npy".format(num)

filename_volume = filename+"_volumes_{}.npy".format(num)
filename_mask= filename+"_mask_{}.npy".format(num)
file_map_cf = filename + "_{}_cf_MRF_map.pkl".format(num)



#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



if "Square" in name:
    m_ = RandomMap(name,dict_config,resting_time=4000,image_size=image_size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)

else:
    m_=MapFromFile(name,file = "./data/KneePhantom/Phantom1/paramMap_Control.mat",rounding=True)


if str.split(filename_paramMap,"/")[-1] not in os.listdir(folder):
    m_.buildParamMap()
    with open(filename_paramMap, "wb" ) as file:
        pickle.dump(m_.paramMap, file)

    map_rebuilt = m_.paramMap
    mask = m_.mask

    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    np.save(filename_paramMask,mask)

    for key in ["ff", "wT1", "df", "attB1"]:
        file_mha = "/".join(["/".join(str.split(filename_paramMap, "/")[:-1]),
                             "_".join(str.split(str.split(filename_paramMap, "/")[-1], ".")[:-1])]) + "_{}.mha".format(
            key)
        io.write(file_mha, map_for_sim[key], tags={"spacing": [5, 1, 1]})

else:
    with open(filename_paramMap, "rb") as file:
        m_.paramMap=pickle.load(file)
    m_.mask=np.load(filename_paramMask)



m_.build_ref_images(seq)

if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
    np.save(filename_groundtruth,m_.images_series[::nspoke])

tmp_traj = bart(1, "traj -x {0} -y {1} -G -r".format(npoint,nb_allspokes))
cfl.writecfl("tmp_traj",tmp_traj)
traj = bart(1, "scale 0.5 tmp_traj")
cfl.writecfl("traj",traj)


radial_traj=Radial(nb_allspokes=nb_allspokes,npoint=npoint)
traj_for_python=traj.astype(float).T
traj_for_python*=np.pi/np.max(traj_for_python)
radial_traj.traj=traj_for_python[:,:,:2]





nb_channels=4
nb_means=int(nb_channels**(1/2))


means_x=np.arange(1,nb_means+1)*(1/(nb_means+1))*image_size[0]
means_y=np.arange(1,nb_means+1)*(1/(nb_means+1))*image_size[1]

sig_x=(image_size[0]/(2*(nb_means+1)))**2
sig_y=(image_size[1]/(2*(nb_means+1)))**2

x = np.arange(image_size[0])
y = np.arange(image_size[1])


X,Y = np.meshgrid(x,y)
pixels=np.stack([X,Y], axis=-1)

pixels=pixels.reshape(-1,2)

from scipy.stats import multivariate_normal
b1_maps=[]
for mu_x in means_x:
    for mu_y in means_y:
        b1_maps.append(multivariate_normal.pdf(pixels, mean=[mu_x,mu_y], cov=np.diag([sig_x,sig_y])))

b1_maps = np.array(b1_maps)
b1_maps=b1_maps/np.expand_dims(np.max(b1_maps,axis=-1),axis=-1)
b1_maps=b1_maps.reshape((nb_channels,)+image_size)
plt.figure()
plt.imshow(b1_maps.reshape(-1,256))

kdata_multi=m_.generate_kdata_multi(radial_traj,b1_maps)
m_.build_ref_images(seq)
#animate_images(m_.images_series)
kdata_multi_for_bart_full=kdata_multi.reshape(nb_channels,nb_allspokes,npoint).T
kdata_multi_for_bart_full=np.expand_dims(kdata_multi_for_bart_full,axis=0)
cfl.writecfl("kdata_multi_full",kdata_multi_for_bart_full)


kdata_multi=kdata_multi.reshape(nb_channels,175,8,npoint)
density = np.abs(np.linspace(-1, 1, npoint))
kdata_multi=kdata_multi*np.expand_dims(density,axis=(0,1))
res=16
b1_all_slices=calculate_sensitivity_map(kdata_multi.reshape(nb_channels,nb_allspokes,npoint),radial_traj,res,image_size,hanning_filter=True)
plot_image_grid(np.abs(b1_all_slices),nb_row_col=(2,2))


coil_img=bart(1,"nufft -i -t traj kdata_multi_full")
cfl.writecfl("coil_img",coil_img)
plot_image_grid(np.moveaxis(np.abs(coil_img).squeeze(),-1,0),nb_row_col=(2,2))

#sens done in Bart espirit
#bart fft -u $(bart bitmask 0 1) coil_img ksp
#bart ecalib -m1 ksp sens
#sens=cfl.readcfl("sens")
import os
os.system("bart fft -u $(bart bitmask 0 1) coil_img ksp_test")
sens=bart(1,"ecalib -m1 ksp")
cfl.writecfl("sens",sens)

plot_image_grid(np.moveaxis(np.abs(sens).squeeze(),-1,0)*np.expand_dims(mask,axis=0),nb_row_col=(2,2))
plot_image_grid(np.abs(b1_all_slices)*np.expand_dims(mask,axis=0),nb_row_col=(2,2))
plot_image_grid(b1_maps*np.expand_dims(mask,axis=0),nb_row_col=(2,2))

plot_image_grid(np.moveaxis(np.abs(sens).squeeze(),-1,0)*np.expand_dims(mask,axis=0)-b1_maps*np.expand_dims(mask,axis=0),nb_row_col=(2,2))
plot_image_grid(np.abs(b1_all_slices)*np.expand_dims(mask,axis=0)-b1_maps*np.expand_dims(mask,axis=0),nb_row_col=(2,2))

#Dynamic reconstruction
traj_reshaped = traj.reshape(3,npoint,175,8)
traj_reshaped=np.moveaxis(traj_reshaped,-1,-2)
traj_reshaped=np.expand_dims(traj_reshaped,axis=(3,4,5,6,7,8,9))
cfl.writecfl("traj_reshaped",traj_reshaped)


kdata_multi_for_bart_reshaped=kdata_multi_for_bart_full.reshape(1,npoint,175,8,4)
kdata_multi_for_bart_reshaped=np.moveaxis(kdata_multi_for_bart_reshaped,2,-1)
kdata_multi_for_bart_reshaped=np.expand_dims(kdata_multi_for_bart_reshaped,axis=(4,5,6,7,8,9))
cfl.writecfl("kdata_multi_for_bart_reshaped",kdata_multi_for_bart_reshaped)

density = np.abs(np.linspace(-1, 1, npoint))
kdata_multi_for_bart_reshaped_dens_adj=kdata_multi_for_bart_reshaped*np.expand_dims(density,axis=(0,2,3,4,5,6,7,8,9,10))
cfl.writecfl("kdata_multi_for_bart_reshaped_dens_adj",kdata_multi_for_bart_reshaped_dens_adj)

density = np.abs(np.linspace(-1, 1, npoint))
#density_adj_bart=np.zeros(kdata_multi_for_bart_reshaped.shape)
tile_shape=list(kdata_multi_for_bart_reshaped.shape)
tile_shape[3]=1
tile_shape[1]=1
tile_shape=tuple(tile_shape)
density_adj_bart=np.sqrt(np.tile(np.expand_dims(density,axis=(0,2,3,4,5,6,7,8,9,10)),tile_shape))
cfl.writecfl("dens_adj_bart",density_adj_bart)


kdata_multi_for_bart_reshaped=kdata_multi_for_bart_full.reshape(1,npoint,175,8,4)
kdata_multi_for_bart_reshaped=np.moveaxis(kdata_multi_for_bart_reshaped,2,-1)
kdata_multi_for_bart_reshaped=np.expand_dims(kdata_multi_for_bart_reshaped,axis=(4,5,6,7,8,9))
cfl.writecfl("kdata_multi_for_bart_reshaped",kdata_multi_for_bart_reshaped)

kdata_multi_for_bart_reshaped_sbreco=np.moveaxis(kdata_multi_for_bart_reshaped,-1,-2)
traj_reshaped_sbreco=np.moveaxis(traj_reshaped,-1,-2)
cfl.writecfl("kdata_multi_for_bart_reshaped_sbreco",kdata_multi_for_bart_reshaped_sbreco)
cfl.writecfl("traj_reshaped_sbreco",traj_reshaped_sbreco)


#bart pics -RT:$(bart bitmask 10):0:0.01 -t traj_reshaped kdata_multi_for_bart_reshaped sens out
import os
os.system("bart pics -RT:$(bart bitmask 10):0:0.01 -t traj_reshaped kdata_multi_for_bart_reshaped sens out")
os.system("bart pics -RT:$(bart bitmask 0 1 2):0:0.0001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out_tv_spatial")
#os.system("bart pics -p dens_adj_bart -RL:$(bart bitmask 0 1 2):$(bart bitmask 0 1 2):0.00001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out_llr")
# TV temp > TV spatial > LLR with dens adj (LLR without dens adj does not work at all)

out=cfl.readcfl("out")
animate_images(np.moveaxis(out.squeeze(),-1,0))

plt.close("all")

volumes_all = simulate_radial_undersampled_images_multi(kdata_multi.reshape(nb_channels,175,-1), radial_traj, image_size, b1=b1_all_slices,normalize_kdata=False,
                                                                density_adj=False)
animate_images(volumes_all)



dictfile = "mrf175_SimReco2_light.dict"
dictfile = "mrf175_SimReco2.dict"


mask=m_.mask
optimizer = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=10, pca=True,
                                     threshold_pca=15, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                     gen_mode="other",ntimesteps=175)
all_maps_python = optimizer.search_patterns_test_multi(dictfile, volumes_all)

all_maps_bart = optimizer.search_patterns_test_multi(dictfile, np.moveaxis(out.squeeze(),-1,0))







k="attB1"
plt.figure()
fig,ax=plt.subplots(1,3)
vmin=np.min(all_maps_python[0][0][k])
vmax=np.max(all_maps_python[0][0][k])

ax[0].imshow(makevol(all_maps_python[0][0][k],mask>0),cmap="inferno",vmin=vmin, vmax=vmax)
ax[1].imshow(makevol(m_.paramMap[k],mask>0),cmap="inferno",vmin=vmin, vmax=vmax)
im=ax[2].imshow(makevol(all_maps_bart[0][0][k],mask>0),cmap="inferno",vmin=vmin, vmax=vmax)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)


maskROI = buildROImask_unique(m_.paramMap)
it=0

regression_paramMaps_ROI(m_.paramMap, all_maps_python[it][0], mask > 0, all_maps_python[it][1] > 0, maskROI, adj_wT1=True,
                              save=False,fontsize_axis=10,marker_size=2)


regression_paramMaps_ROI(m_.paramMap, all_maps_bart[it][0], mask > 0, all_maps_bart[it][1] > 0, maskROI, adj_wT1=True,
                              save=False,fontsize_axis=10,marker_size=2)


maskROI = buildROImask_unique(m_.paramMap)
dic_maps={}
dic_maps["python"]=all_maps_python
dic_maps["bart"]=all_maps_bart
it=0

df_result=pd.DataFrame()
k="wT1"
for key in dic_maps.keys():
    roi_values=get_ROI_values(m_.paramMap,dic_maps[key][it][0],m_.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI,fat_threshold=0.7)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
            #roi_values.sort_values(by=["Obs Mean"],inplace=True)
    error=list((roi_values["Pred Mean"]-roi_values["Obs Mean"]))
    if df_result.empty:
        df_result=pd.DataFrame(data=error,columns=[key + " Iteration {}".format(it)])
    else:
        df_result[key + " Iteration {}".format(it)]=error

columns=[df_result.columns[-1]]+list(df_result.columns[:-1])
df_result=df_result[columns]

plt.figure()
df_result.boxplot(grid=False, rot=45, fontsize=10,showfliers=False)
plt.axhline(y=0,linestyle="dashed",color="k",linewidth=0.5)
plt.title("Errors vs ground truth")



df_result=pd.DataFrame()
for key in dic_maps.keys():
    roi_values=get_ROI_values(m_.paramMap,dic_maps[key][it][0],m_.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI,fat_threshold=0.7)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
            #roi_values.sort_values(by=["Obs Mean"],inplace=True)
    std=list(roi_values["Pred Std"])
    if df_result.empty:
        df_result=pd.DataFrame(data=std,columns=[key + " Iteration {}".format(it)])
    else:
        df_result[key + " Iteration {}".format(it)]=std

columns=[df_result.columns[-1]]+list(df_result.columns[:-1])
df_result=df_result[columns]

plt.figure()
df_result.boxplot(grid=False, rot=45, fontsize=10,showfliers=False)
plt.axhline(y=0,linestyle="dashed",color="k",linewidth=0.5)
plt.title("Std per ROI distribution")


#Multiple Iter

os.system("bart pics -p dens_adj_bart -RL:$(bart bitmask 0 1 2):$(bart bitmask 0 1 2):0.00001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out_llr")


import os
dictfile = "mrf175_SimReco2.dict"

mrfdict = dictsearch.Dictionary()
keys,values=read_mrf_dict(dictfile,np.arange(0.,1.01,0.05))
values=values.T
cfl.writecfl("dico_bart",values)

os.system("bart svd -e dico_bart U S V")

S=cfl.readcfl("S")
U=cfl.readcfl("U")

# create the temporal basis
nCoe=5 # use 4 coefficients
os.system("bart extract 1 0 {} U basis".format(nCoe))
#basis=cfl.readcfl("basis")
#basis.shape
os.system("bart transpose 1 6 basis basis")
os.system("bart transpose 0 5 basis basis_{}".format(nCoe))
basis=cfl.readcfl("basis_{}".format(nCoe))


os.system("bart transpose 10 5 traj_reshaped traj_reshaped_sbreco")
os.system("bart transpose 10 5 kdata_multi_for_bart_reshaped kdata_multi_for_bart_reshaped_sbreco")



import os

dictfile = "mrf175_SimReco2.dict"
iter=0
nCoe=10
basis=cfl.readcfl("basis_{}".format(nCoe))
cfl.writecfl("basis_used",basis)
bart_command="bart pics {} -i1 -RT:$(bart bitmask 10):0:0.01 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#looks like the best for now
#bart_command="bart pics {} -i1 -RT:$(bart bitmask 0 1 2):0:0.0001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"
#bart_command="bart pics {} -i1 -s 0.01 -p dens_adj_bart -RL:$(bart bitmask 0 1 2):$(bart bitmask 0 1 2):0.00001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"# does not work well
#bart_command="bart pics {} -i1 -s 0.01 -p dens_adj_bart -RL:$(bart bitmask 0 1 2):0:0.0000001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#does not work well
#bart_command="bart pics {} -m -p dens_adj_bart -i1 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#second best
#bart_command="bart pics {} -m -i1 -RW:$(bart bitmask 0 1 2):0:0.001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"
bart_command="bart pics -B basis_used {} -i1 -t traj_reshaped_sbreco kdata_multi_for_bart_reshaped_sbreco sens out{}"#looks like the best for now
bart_command="bart pics  -m -B basis_used {} -i1 -t traj_reshaped_sbreco kdata_multi_for_bart_reshaped_sbreco sens out{}"#looks like the best for now


#traj_reshaped_sbreco=cfl.readcfl("traj_reshaped_sbreco")
#kdata_multi_for_bart_reshaped_sbreco=cfl.readcfl("kdata_multi_for_bart_reshaped_sbreco")

os.system(bart_command.format("",iter))
out=cfl.readcfl("out{}".format(iter))


# plt.figure()
# animate_images(np.moveaxis(np.matmul(out.squeeze(),basis.squeeze().T),-1,0))

mask=m_.mask
optimizer = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=10, pca=True,
                                     threshold_pca=15, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                     gen_mode="other",ntimesteps=175,return_matched_signals=True)
#np.matmul(out.squeeze(),basis.squeeze().T.conj()).shape

niter=3
all_maps_bart_all_iter={}

for iter in tqdm(range(1,niter+1)):
    if "basis_used" in bart_command:
        out=np.matmul(out.squeeze(),basis.squeeze().T)
    all_maps_bart,matched_signals = optimizer.search_patterns_test_multi(dictfile, np.moveaxis(out.squeeze(),-1,0))

    all_maps_bart_all_iter[iter-1]=all_maps_bart[0]

    if iter==niter:
        break

    if "basis_used" in bart_command:
        matched_signals = np.matmul(basis.squeeze().T.conj(), matched_signals)
    matched_signals=[makevol(s,mask>0) for s in matched_signals]

    matched_signals=np.array(matched_signals)
    matched_signals_bart=np.moveaxis(matched_signals,0,-1)
    if "basis_used" in bart_command:
        matched_signals_bart = np.expand_dims(matched_signals_bart, axis=(2, 3, 4,5))
    else:
        matched_signals_bart=np.expand_dims(matched_signals_bart,axis=(2,3,4,5,6,7,8,9))
    cfl.writecfl("image_start",matched_signals_bart)

    os.system(bart_command.format("-W image_start", iter))
    out=cfl.readcfl("out{}".format(iter))

k="wT1"
plt.figure()
fig,ax=plt.subplots(1,4)
vmin=np.min(all_maps_python[0][0][k])
vmax=np.max(all_maps_python[0][0][k])
#vmin=550
#vmax=2000

ax[1].imshow(makevol(all_maps_python[0][0][k],mask>0),cmap="inferno",vmin=vmin, vmax=vmax)
ax[1].set_title("Python")
ax[0].imshow(makevol(m_.paramMap[k],mask>0),cmap="inferno",vmin=vmin, vmax=vmax)
ax[0].set_title("Ground Truth")
im=ax[2].imshow(makevol(all_maps_bart_all_iter[0][0][k],mask>0),cmap="inferno",vmin=vmin, vmax=vmax)
ax[2].set_title("Bart Iter 1")
im=ax[3].imshow(makevol(all_maps_bart_all_iter[niter-1][0][k],mask>0),cmap="inferno",vmin=vmin, vmax=vmax)
ax[3].set_title("Bart Iter {}".format(niter-1))

fig.suptitle("Subspace reco", fontsize=14)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)


maskROI = buildROImask_unique(m_.paramMap)
it=0

regression_paramMaps_ROI(m_.paramMap, all_maps_python[it][0], mask > 0, all_maps_python[it][1] > 0, maskROI, adj_wT1=True,
                              save=False,fontsize_axis=10,marker_size=2)

it=1
regression_paramMaps_ROI(m_.paramMap, all_maps_bart_all_iter[it][0], mask > 0, all_maps_bart_all_iter[it][1] > 0, maskROI, adj_wT1=True,
                              save=False,fontsize_axis=10,marker_size=2)


maskROI = buildROImask_unique(m_.paramMap)
dic_maps={}
dic_maps["python"]=all_maps_python
for it in all_maps_bart_all_iter.keys():
    dic_maps["bart_it{}".format(it)]=[all_maps_bart_all_iter[it]]

it=0
df_result=pd.DataFrame()
k="ff"
for key in sorted(list(dic_maps.keys())):
    roi_values=get_ROI_values(m_.paramMap,dic_maps[key][it][0],m_.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI,fat_threshold=0.7)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
            #roi_values.sort_values(by=["Obs Mean"],inplace=True)
    error=list((roi_values["Pred Mean"]-roi_values["Obs Mean"]))
    if df_result.empty:
        df_result=pd.DataFrame(data=error,columns=[key + " Iteration {}".format(it)])
    else:
        df_result[key + " Iteration {}".format(it)]=error

columns=[df_result.columns[-1]]+list(df_result.columns[:-1])
df_result=df_result[columns]

plt.figure()
df_result.boxplot(grid=False, rot=45, fontsize=10,showfliers=False)
plt.axhline(y=0,linestyle="dashed",color="k",linewidth=0.5)
plt.title("Errors vs ground truth")



df_result=pd.DataFrame()
for key in sorted(list(dic_maps.keys())):
    roi_values=get_ROI_values(m_.paramMap,dic_maps[key][it][0],m_.mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI,fat_threshold=0.7)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
            #roi_values.sort_values(by=["Obs Mean"],inplace=True)
    std=list(roi_values["Pred Std"])
    if df_result.empty:
        df_result=pd.DataFrame(data=std,columns=[key + " Iteration {}".format(it)])
    else:
        df_result[key + " Iteration {}".format(it)]=std

columns=[df_result.columns[-1]]+list(df_result.columns[:-1])
df_result=df_result[columns]

plt.figure()
df_result.boxplot(grid=False, rot=45, fontsize=10,showfliers=False)
plt.axhline(y=0,linestyle="dashed",color="k",linewidth=0.5)
plt.title("Std per ROI distribution")
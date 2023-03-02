
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from trajectory import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np

from mutools import io

folder="./data/KneePhantom/Phantom1/"
file_mask="maskFull_Control_multislice_31.npy"
file_dico="dicoMasks_Control_multislice_retained_31.pkl"

mask_full=np.load(folder+file_mask)

with open(folder+file_dico,"rb") as file:
    dico_retained=pickle.load(file)


#animate_images(mask_full)

keys=list(dico_retained.keys())

volumes_all=np.zeros((len(keys),95,int((mask_full>0).sum())),dtype="complex64")

for num in tqdm(range(len(keys))):
    mask=dico_retained[keys[num]]
    #animate_images(mask)

    volumes=np.tile(mask,(95,1,1,1))
    volumes=volumes.astype("complex64")
    b1_all_slices=np.ones((1,)+mask.shape)

    radial_traj=Radial3D(total_nspokes=760,undersampling_factor=2,npoint=512,nb_slices=24,incoherent=False,mode="old")


    volumes_us=undersampling_operator(volumes,radial_traj,b1_all_slices,density_adj=True,light_memory_usage=True)
    volumes_all[num]=volumes_us[:,mask_full>0]


np.save("volumes_Control_multislice_retained_31.npy",volumes_all,allow_pickle=True)




from utils_simu import *

with open("./mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

fat_amp = np.array(dict_config["fat_amp"])
fat_shift = -np.array(dict_config["fat_cshift"])
group_size=8

TR_,FA_,TE_=load_sequence_file("mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1.json",3.95,1.87/1000)

params=np.array(keys)

volumes_simu=np.zeros(volumes_all.shape[1:],dtype="complex64")

for j in tqdm(range(params.shape[0])):
    param=params[j]
    ff=param[1]
    df=param[-1]
    b1=param[-2]
    wT1=param[0]/1000

    s, s_w, s_f, keys_signals = simulate_gen_eq_signal(TR_, FA_, TE_, ff, df, wT1, 300 / 1000,b1, T_2w=40 / 1000, T_2f=80 / 1000,
                                                   amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
                                                   return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s=np.expand_dims(s.squeeze(),axis=1)

    volumes_simu += s*volumes_all[j]

volumes_simu_image=[makevol(v,mask_full>0) for v in volumes_simu]
volumes_simu_image=np.array(volumes_simu_image)



animate_images(volumes_simu_image[:,12])



output_file=folder+"paramMap_Control_multislice.mat"
matMap_deformed=loadmat(output_file)
map_wT1=matMap_deformed["paramMap"][0][0][0]
map_b1=matMap_deformed["paramMap"][0][0][2]
map_df=matMap_deformed["paramMap"][0][0][3]
map_ff=matMap_deformed["paramMap"][0][0][1]

dico_gt={}
dico_gt["wT1"]=map_wT1[mask_full>0]
dico_gt["df"]=map_df[mask_full>0]
dico_gt["attB1"]=map_b1[mask_full>0]
dico_gt["ff"]=map_ff[mask_full>0]

output_file=folder+"paramMap_Control_multislice_masked_31.pkl"
with open(output_file,"wb") as file:
    pickle.dump(dico_gt,file)


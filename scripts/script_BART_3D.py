
import sys
os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
sys.path.append(os.environ['TOOLBOX_PATH'] + "/python/")
import numpy as np
from image_series import *
from mrfsim.dictoptimizers import SimpleDictSearch,GaussianWeighting,BruteDictSearch
from mrfsim.utils_mrf import *
import json
import os
import pickle
from mutools import io
from mrfsim.movements import TranslationBreathing
from bart import bart
import matplotlib.pyplot as plt

import cfl
import os



use_GPU = True
light_memory_usage=False
window=8

base_folder = "./data/InVivo"

localfile ="/3D/patient.009.v1/meas_MID00084_FID33958_raFin_3D_tra_1x1x5mm_FULL_new.dat"
localfile="/3D/patient.002.v5/meas_MID00021_FID34064_raFin_3D_tra_1x1x5mm_FULL_new.dat"

filename = base_folder+localfile

filename="/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/2_Data Raw/phantom.009.v2/meas_MID00218_FID06324_raFin_3D_tra_1x1x5mm_FULL_fullReco.dat"

#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00044_FID42066_raFin_3D_tra_1x1x5mm_us4_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"


filename_save=str.split(filename,".dat") [0]+".npy"
folder = "/".join(str.split(filename,"/")[:-1])


filename_b1 = str.split(filename,".dat") [0]+"_b1.npy"
filename_b1_bart = str.split(filename,".dat") [0]+"_b1_bart.npy"

filename_volume = str.split(filename,".dat") [0]+"_volumes.npy"
filename_kdata = str.split(filename,".dat") [0]+"_no_dens_adj_kdata.npy"
filename_nav_save=str.split(filename,".dat") [0]+"_nav.npy"
filename_seqParams = str.split(filename,".dat") [0]+"_seqParams.pkl"
filename_mask= str.split(filename,".dat") [0]+"_mask{}.npy".format("")


localfile="/patient.008.v7/meas_MID00020_FID37032_raFin_3D_tra_1x1x5mm_FULL_new.dat"
dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2.26_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.26_reco4_w8_simmean.dict"



return_cost=True

if str.split(filename_seqParams,"/")[-1] not in os.listdir(folder):

    twix = twixtools.read_twix(filename,optional_additional_maps=["sWipMemBlock","sKSpace"],optional_additional_arrays=["SliceThickness"])

    if np.max(np.argwhere(np.array(twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"])>0))>=16:
        use_navigator_dll = True
    else:
        use_navigator_dll = False



    alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
    x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
    y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
    z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]

    nb_part = twix[-1]["hdr"]["Meas"]["Partitions"]

    dico_seqParams = {"alFree":alFree,"x_FOV":x_FOV,"y_FOV":y_FOV,"z_FOV":z_FOV,"use_navigator_dll":use_navigator_dll,"nb_part":nb_part}

    del alFree

    file = open(filename_seqParams, "wb")
    pickle.dump(dico_seqParams, file)
    file.close()

else:
    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()



try:
    del twix
except:
    pass

try:
    use_navigator_dll=dico_seqParams["use_navigator_dll"]
except:
    use_navigator_dll=False

if use_navigator_dll:
    meas_sampling_mode=dico_seqParams["alFree"][14]
    nb_gating_spokes = dico_seqParams["alFree"][6]
else:
    meas_sampling_mode = dico_seqParams["alFree"][12]
    nb_gating_spokes = 0

if nb_gating_spokes>0:
    meas_orientation =  dico_seqParams["alFree"][11]
    if meas_orientation==1:
        nav_direction = "READ"
    elif meas_orientation==2:
        nav_direction = "PHASE"
    elif meas_orientation==3:
        nav_direction = "SLICE"

nb_segments = dico_seqParams["alFree"][4]
dummy_echos = dico_seqParams["alFree"][5]

ntimesteps=int(nb_segments/window)


x_FOV = dico_seqParams["x_FOV"]
y_FOV = dico_seqParams["y_FOV"]
z_FOV = dico_seqParams["z_FOV"]
#z_FOV=64
nb_part = dico_seqParams["nb_part"]
undersampling_factor = dico_seqParams["alFree"][9]
#undersampling_factor=1

del dico_seqParams

if meas_sampling_mode==1:
    incoherent=False
    mode = None
elif meas_sampling_mode==2:
    incoherent = True
    mode = "old"
elif meas_sampling_mode==3:
    incoherent = True
    mode = "new"


#undersampling_factor=4


if str.split(filename_save,"/")[-1] not in os.listdir(folder):
    if 'twix' not in locals():
        print("Re-loading raw data")
        twix = twixtools.read_twix(filename)

    mdb_list = twix[-1]['mdb']
    if nb_gating_spokes == 0:
        data = []

        for i, mdb in enumerate(mdb_list):
            if mdb.is_image_scan():
                data.append(mdb)



    else:
        print("Reading Navigator Data....")
        data_for_nav = []
        data = []
        nav_size_initialized = False
        # k = 0
        for i, mdb in enumerate(mdb_list):
            if mdb.is_image_scan():
                if not (mdb.mdh[14][9]):
                    mdb_data_shape = mdb.data.shape
                    mdb_dtype = mdb.data.dtype
                    nav_size_initialized = True
                    break

        for i, mdb in enumerate(mdb_list):
            if mdb.is_image_scan():
                if not (mdb.mdh[14][9]):
                    data.append(mdb)
                else:
                    data_for_nav.append(mdb)
                    data.append(np.zeros(mdb_data_shape, dtype=mdb_dtype))

                # print("i : {} / k : {} / Line : {} / Part : {}".format(i, k, mdb.cLin, mdb.cPar))
                # k += 1
        data_for_nav = np.array([mdb.data for mdb in data_for_nav])
        data_for_nav = data_for_nav.reshape((int(nb_part+dummy_echos), int(nb_gating_spokes)) + data_for_nav.shape[1:])

        if data_for_nav.ndim == 3:
            data_for_nav = np.expand_dims(data_for_nav, axis=-2)
        data_for_nav=data_for_nav[dummy_echos:]
        data_for_nav = np.moveaxis(data_for_nav, -2, 0)
        np.save(filename_nav_save, data_for_nav)

    data = np.array([mdb.data for mdb in data])
    data = data.reshape((-1,int(nb_segments)) + data.shape[1:])
    data=data[dummy_echos:]
    data = np.moveaxis(data, 2, 0)
    data = np.moveaxis(data, 2, 1)

    del mdb_list

    ##################################################
    try:
        del twix
    except:
        pass

    np.save(filename_save,data)



else :
    data = np.load(filename_save)
    if nb_gating_spokes>0:
        data_for_nav=np.load(filename_nav_save)

try:
    del twix
except:
    pass

data_shape = data.shape

#data_for_nav=data_for_nav[:,:nb_gating_spokes,:,:]
#data_for_nav = np.moveaxis(data_for_nav,-2,1)



nb_channels=data_shape[0]
nb_allspokes = data_shape[-3]
npoint = data_shape[-1]
nb_slices = data_shape[-2]*undersampling_factor
image_size = (nb_slices, int(npoint/2), int(npoint/2))


dx = x_FOV/(npoint/2)
dy = y_FOV/(npoint/2)
dz = z_FOV/nb_slices
#dz=4
#file_name_nav_mat=str.split(filename,".dat") [0]+"_nav.mat"
#savemat(file_name_nav_mat,{"Kdata":data_for_nav})

if str.split(filename_kdata,"/")[-1] in os.listdir(folder):
    del data


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)
#radial_traj.adjust_traj_for_window(window)

nb_segments=radial_traj.get_traj().shape[0]

if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):
    # Density adjustment all slices
    #print("Performing Density Adjustment....")
    #density = np.abs(np.linspace(-1, 1, npoint))
    #kdata_all_channels_all_slices = data.reshape(-1, npoint)
    #del data
    #kdata_all_channels_all_slices = (kdata_all_channels_all_slices*density).reshape(data_shape)
    #kdata_all_channels_all_slices = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in data]
    #kdata_all_channels_all_slices=np.array(kdata_all_channels_all_slices).reshape(data_shape)
    np.save(filename_kdata, data)
    del data
    #kdata_all_channels_all_slices=open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)





# Coil sensi estimation for all slices
print("Calculating Coil Sensitivity....")

if str.split(filename_b1,"/")[-1] not in os.listdir(folder):
    res = 8
    b1_all_slices=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=light_memory_usage,hanning_filter=True,density_adj=True)
    np.save(filename_b1,b1_all_slices)
    del kdata_all_channels_all_slices
    kdata_all_channels_all_slices = np.load(filename_kdata)

else:
    b1_all_slices=np.load(filename_b1)

sl=int(b1_all_slices.shape[1]/2)
list_images = list(np.abs(b1_all_slices[:,sl,:,:]))
plot_image_grid(list_images,(6,6),title="Sensitivity map for slice {}".format(sl))


kdata_all_channels_all_slices = np.load(filename_kdata)

print("Building Volumes....")
if str.split(filename_volume,"/")[-1] not in os.listdir(folder):
    kdata_all_channels_all_slices=np.load(filename_kdata)
    volumes_all=simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=True,ntimesteps=ntimesteps,useGPU=False,normalize_kdata=False,memmap_file=None,light_memory_usage=light_memory_usage,normalize_volumes=True,normalize_iterative=True)
    np.save(filename_volume,volumes_all)
    # ani = animate_images(volumes_all[:,int(nb_slices/2),:,:])
    #ani = animate_images(volumes_all[:, 5, :, :])
    #del volumes_all
else:
    volumes_all=np.load(filename_volume)

print("Building Mask....")
if str.split(filename_mask,"/")[-1] not in os.listdir(folder):
    #selected_spokes = np.r_[10:648,1099:1400]
    selected_spokes=None
    kdata_all_channels_all_slices = np.load(filename_kdata)
    mask=build_mask_single_image_multichannel(kdata_all_channels_all_slices,radial_traj,image_size,b1=b1_all_slices,density_adj=True,threshold_factor=1/20, normalize_kdata=False,light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask,mask)
    animate_images(mask)
else:
    mask = np.load(filename_mask)



kdata_all_channels=np.load(filename_kdata)

traj_python=radial_traj.get_traj()
traj_python=np.transpose(traj_python)


traj_python_for_bart=traj_python.astype("complex64")[:,:,-200:]
#traj_python_for_bart[:2,:,:]=traj_python
traj_python_for_bart[0,:,:]=traj_python_for_bart[0,:,:]/np.max(np.abs(traj_python_for_bart[0,:,:]))*int(npoint/4)
traj_python_for_bart[1,:,:]=traj_python_for_bart[1,:,:]/np.max(np.abs(traj_python_for_bart[1,:,:]))*int(npoint/4)
traj_python_for_bart[2,:,:]=traj_python_for_bart[2,:,:]/np.max(np.abs(traj_python_for_bart[2,:,:]))*int(nb_slices/2)

cfl.writecfl("traj",traj_python_for_bart)

kdata_multi_for_bart_full=kdata_all_channels.reshape(nb_channels,nb_allspokes,-1).T
kdata_multi_for_bart_full=np.expand_dims(kdata_multi_for_bart_full,axis=0)[:,:,-200:]
cfl.writecfl("kdata_multi_full",kdata_multi_for_bart_full)

coil_img=bart(1,"nufft -i -t traj kdata_multi_full")
cfl.writecfl("coil_img",coil_img)

coil_img=cfl.readcfl("coil_img")

plot_image_grid(np.moveaxis(np.abs(coil_img[:,:,int(nb_slices/2)]).squeeze(),-1,0),nb_row_col=(4,4))

animate_images(coil_img[:,:,:,0].T)

#sens done in Bart espirit
#bart fft -u $(bart bitmask 0 1) coil_img ksp
#bart ecalib -m1 ksp sens
#sens=cfl.readcfl("sens")
import os
os.system("bart fft -u $(bart bitmask 0 1 2) coil_img ksp")
sens=bart(1,"ecalib -m1 ksp")
cfl.writecfl("sens",sens)

sens=cfl.readcfl("sens")

sl=int(nb_slices/2)
plot_image_grid(np.moveaxis(np.abs(sens[:,:,sl]).squeeze(),-1,0)*np.expand_dims(mask[sl],axis=0),nb_row_col=(4,4))
plot_image_grid(np.abs(b1_all_slices[sl])*np.expand_dims(mask[sl],axis=0),nb_row_col=(4,4))

# b1_all_slices_bart = sens
# b1_all_slices_bart=np.moveaxis(b1_all_slices_bart,-2,0)
# b1_all_slices_bart=np.moveaxis(b1_all_slices_bart,-1,0)
# np.save(filename_b1_bart,b1_all_slices_bart)

cc_test=bart(1,"cc -M ksp")

ksp_ref=cfl.readcfl("ksp")

num_vcoils = 8
ksp_ref_cc = bart(1, 'ccapply -p {}'.format(num_vcoils), ksp_ref, cc_test)
kdata_multi_for_bart_full_cc = bart(1, 'ccapply -p {}'.format(num_vcoils), kdata_multi_for_bart_full, cc_test)
cfl.writecfl("kdata_multi_for_bart_full_cc",kdata_multi_for_bart_full_cc)


sens_cc=bart(1,"ecalib -m1",ksp_ref_cc)

sl=int(nb_slices/2)
plot_image_grid(np.moveaxis(np.abs(sens_cc[:,:,sl]).squeeze(),-1,0)*np.expand_dims(mask[sl],axis=0),nb_row_col=(4,4))

plt.figure()
plt.plot(np.abs(sens[135,135,:,:]))

cfl.writecfl("sens_cc",sens_cc)


#Dynamic reconstruction
traj=cfl.readcfl("traj")
traj_reshaped = traj.reshape(3,-1,175,8)
traj_reshaped=np.moveaxis(traj_reshaped,-1,-2)
traj_reshaped=np.expand_dims(traj_reshaped,axis=(3,4,5,6,7,8,9))
cfl.writecfl("traj_reshaped",traj_reshaped)


kdata_multi_for_bart_reshaped=kdata_multi_for_bart_full.reshape(1,-1,175,8,nb_channels)
kdata_multi_for_bart_reshaped=np.moveaxis(kdata_multi_for_bart_reshaped,2,-1)
kdata_multi_for_bart_reshaped=np.expand_dims(kdata_multi_for_bart_reshaped,axis=(4,5,6,7,8,9))
cfl.writecfl("kdata_multi_for_bart_reshaped",kdata_multi_for_bart_reshaped)

kdata_multi_for_bart_reshaped_cc = bart(1, 'ccapply -p {}'.format(num_vcoils), kdata_multi_for_bart_reshaped, cc_test)
cfl.writecfl("kdata_multi_for_bart_reshaped_cc",kdata_multi_for_bart_reshaped_cc)


density = np.abs(np.linspace(-1, 1, npoint))
tile_shape=list(kdata_multi_for_bart_reshaped.shape)
tile_shape[3]=1
tile_shape[1]=1
tile_shape=tuple(tile_shape)
density_adj_bart=np.sqrt(np.tile(np.expand_dims(density,axis=(0,2,3,4,5,6,7,8,9,10)),tile_shape))
cfl.writecfl("dens_adj_bart",density_adj_bart)
#
#
#
# import os
# #dictfile="mrf_dictconf_Dico2_Invivo_adjusted_1_68_reco4_w8_simmean.dict"
# #dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_1_68_reco4_w8_simmean.dict"
#
# dictfile="mrf_dictconf_SimReco2_adjusted_1_68_reco4_w8_simmean.dict"
# mrfdict = dictsearch.Dictionary()
# keys,values=read_mrf_dict(dictfile,np.arange(0.,1.01,0.05))
# values=values.T
# cfl.writecfl("dico_bart",values)
#
# os.system("bart svd -e dico_bart U S V")
#
# S=cfl.readcfl("S")
# U=cfl.readcfl("U")
#
# # create the temporal basis
# nCoe=10 # use 4 coefficients
# os.system("bart extract 1 0 {} U basis".format(nCoe))
# #basis=cfl.readcfl("basis")
# #basis.shape
# os.system("bart transpose 1 6 basis basis")
# os.system("bart transpose 0 5 basis basis_{}".format(nCoe))
# basis=cfl.readcfl("basis_{}".format(nCoe))
#
#
# os.system("bart transpose 10 5 traj_reshaped traj_reshaped_sbreco")
# os.system("bart transpose 10 5 kdata_multi_for_bart_reshaped kdata_multi_for_bart_reshaped_sbreco")
#


import os

#dictfile = "mrf175_SimReco2.dict"
dictfile="mrf_dictconf_Dico2_Invivo_adjusted_2_21_reco4_w8_simmean.dict"
dictfile_light="mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2_21_reco4_w8_simmean.dict"

iter=0
nCoe=10
basis=cfl.readcfl("basis_{}".format(nCoe))
cfl.writecfl("basis_used",basis)

bart_command="bart pics {} -i1 -RT:$(bart bitmask 10):0:0.01 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#looks like the best for now
#bart_command="bart pics {} -i1 -RT:$(bart bitmask 0 1 2):0:0.0001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"
#bart_command="bart pics {} -i1 -s 0.01 -p dens_adj_bart -RL:$(bart bitmask 0 1 2):$(bart bitmask 0 1 2):0.00001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"# does not work well
#bart_command="bart pics {} -i1 -s 0.01 -p dens_adj_bart -RL:$(bart bitmask 0 1 2):0:0.0000001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#does not work well
#bart_command="bart pics {} -m -p dens_adj_bart -i1 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#second best
bart_command="bart pics {} -i1 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#second best

#bart_command="bart pics {} -m -p dens_adj_bart -i1 -t traj_reshaped kdata_multi_for_bart_reshaped_cc sens_cc out{}"#second best
bart_command="bart pics -e --no-toeplitz --lowmem-stack=8 {} --fista -i1 -RT:$(bart bitmask 2):0:0.0001 -t traj_reshaped kdata_multi_for_bart_reshaped_cc sens_cc out{}"#second best

#bart_command="bart pics {} -m -i1 -RW:$(bart bitmask 0 1 2):0:0.001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"
#bart_command="bart pics -B basis_used {} -i1 -t traj_reshaped_sbreco kdata_multi_for_bart_reshaped_sbreco sens out{}"#looks like the best for now
#bart_command="bart pics -m -C1 -B basis_used {} -i1 -t traj_reshaped_sbreco kdata_multi_for_bart_reshaped_sbreco sens out{}"#looks like the best for now


#traj_reshaped_sbreco=cfl.readcfl("traj_reshaped_sbreco")
#kdata_multi_for_bart_reshaped_sbreco=cfl.readcfl("kdata_multi_for_bart_reshaped_sbreco")

os.system(bart_command.format("",iter))
out=cfl.readcfl("out{}".format(iter))

#sl=int(nb_slices/2)
#animate_images(np.moveaxis(out[:,:,sl].squeeze(),-1,0))
#animate_images(volumes_all)

# plt.figure()
# animate_images(np.moveaxis(np.matmul(out.squeeze(),basis.squeeze().T),-1,0))

#mask=m_.mask
optimizer = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=10, pca=True,
                                     threshold_pca=15, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                     gen_mode="other",ntimesteps=175,return_matched_signals=True)

optimizer = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=radial_traj, split=100, pca=True,
                                     threshold_pca=15, log=False, useGPU_dictsearch=True, useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                     b1=b1_all_slices, threshold_ff=0.9, dictfile_light=dictfile_light,
                                     return_matched_signals=True)  # ,mu_TV=1,weights_TV=[1.,0.,0.])
#np.matmul(out.squeeze(),basis.squeeze().T.conj()).shape


niter=5
all_maps_bart_all_iter={}

for iter in tqdm(range(1,niter+1)):
    if "basis_used" in bart_command:
        out=np.matmul(out.squeeze(),basis.squeeze().T)
    all_maps_bart,matched_signals = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, np.moveaxis(np.moveaxis(out.squeeze(),-1,0),-1,1),retained_timesteps=None)

    all_maps_bart_all_iter[iter-1]=all_maps_bart[0]

    # all_maps_bart, matched_signals = optimizer.search_patterns_test_multi(dictfile, np.moveaxis(out.squeeze(), -1, 0))
    # all_maps_bart_all_iter[iter - 1] = all_maps_bart[0]


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

curr_file = filename.split(".dat")[0] + "{}_MRF_map.pkl".format("_bart_fista_reg_z")

import pickle
with open(curr_file,"wb") as file:
    pickle.dump(all_maps_bart_all_iter,file)

all_maps = all_maps_bart_all_iter

for iter in list(all_maps.keys()):

    map_rebuilt=all_maps[iter][0]
    mask=all_maps[iter][1]

    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
    #map_Python.buildParamMap()


    for key in ["ff","wT1","df","attB1"]:
        file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_it{}_{}.mha".format(iter,key)
        io.write(file_mha,map_for_sim[key],tags={"spacing":[dz,dx,dy]})

    if return_cost:
        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "_it{}_{}.mha".format(
            iter, "correlation")
        io.write(file_mha, makevol(all_maps[iter][2],mask>0), tags={"spacing": [dz, dx, dy]})

        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "_it{}_{}.mha".format(
            iter, "phase")
        io.write(file_mha, makevol(all_maps[iter][3],mask>0), tags={"spacing": [dz, dx, dy]})



all_maps_python,matched_signals=optimizer.search_patterns_test_multi_2_steps_dico(dictfile,volumes_all)

plt.close("all")

sl=int(nb_slices/2)
k="wT1"
plt.figure()
fig,ax=plt.subplots(1,niter+1)
vmin=np.min(all_maps_python[0][0][k])
vmax=np.max(all_maps_python[0][0][k])
#vmin=550
#vmax=2000
ax[0].imshow(makevol(all_maps_python[0][0][k][sl],mask>0)[sl],cmap="inferno",vmin=vmin, vmax=vmax)
ax[0].set_title("Python",fontsize=8)
ax[0].set_axis_off()

for i in range(niter):
    im=ax[i+1].imshow(makevol(all_maps_bart_all_iter[i][0][k],mask>0)[sl],cmap="inferno",vmin=vmin, vmax=vmax)
    ax[i+1].set_title("Bart Iter {}".format(i),fontsize=8)
    ax[i+1].set_axis_off()

title="ADMM {} No dens adj CC".format(k)
fig.suptitle(title, fontsize=14)


fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(im, cax=cbar_ax)
plt.savefig("BART {}".format(title),dpi=600)


maskROI = buildROImask_unique(m_.paramMap)
it=0

regression_paramMaps_ROI(all_maps_python[it][0], all_maps_python[it][0], mask > 0, all_maps_python[it][1] > 0, maskROI, adj_wT1=True,
                              save=False,fontsize_axis=10,marker_size=2)

it=1
regression_paramMaps_ROI(all_maps_python[it][0], all_maps_bart_all_iter[it][0], mask > 0, all_maps_bart_all_iter[it][1] > 0, maskROI, adj_wT1=True,
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
    roi_values=get_ROI_values(m_.paramMap,dic_maps[key][it][0],mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI,fat_threshold=0.7)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
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
    roi_values=get_ROI_values(dic_maps["python"][it][0],dic_maps[key][it][0],mask>0,dic_maps[key][it][1]>0,return_std=True,adj_wT1=True,maskROI=maskROI,fat_threshold=0.7)[k].loc[:,["Obs Mean","Pred Mean","Pred Std"]]
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












######SIMU########
### Movements simulation

#import matplotlib
#matplotlib.u<se("TkAgg")
import numpy as np
from image_series import *
from mrfsim.dictoptimizers import SimpleDictSearch,GaussianWeighting
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
from utils_simu import *

base_folder = "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&0_2021_MR_MyoMaps/3_Data/4_3D/Invivo"
base_folder = "./3D"

#dictfile = "mrf175_SimReco2_light.dict"
dictjson="mrf_dictconf_SimReco2_light.json"
dictfile="mrf_dictconf_SimReco2_adjusted_1_87_reco4_w8_simmean.dict"
dictfile_light='./mrf_dictconf_SimReco2_light_matching_adjusted_1_87_reco4_w8_simmean.dict'
suffix="_fullReco"
with open("./mrf_sequence_adjusted_1_87.json") as f:
   sequence_config = json.load(f)
Treco=4000

nrep=2
rep=nrep-1
TR_total = np.sum(sequence_config["TR"])

#Treco = TR_total-np.sum(sequence_config["TR"])

##other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=nrep
sequence_config["rep"]=rep

seq=T1MRFSS(**sequence_config)

nb_filled_slices = 16
nb_empty_slices=2
repeat_slice=16
nb_slices = nb_filled_slices+2*nb_empty_slices

undersampling_factor=1

name = "SquareSimu3DMT"


use_GPU = False
light_memory_usage=True
gen_mode="other"


localfile="/"+name
filename = base_folder+localfile

#filename="./data/InVivo/3D/20211221_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
#filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"

folder = "/".join(str.split(filename,"/")[:-1])

suffix=""

filename_paramMap=filename+"_paramMap_sl{}_rp{}.pkl".format(nb_slices,repeat_slice)
filename_paramMask=filename+"_paramMask_sl{}_rp{}.npy".format(nb_slices,repeat_slice)
filename_groundtruth = filename+"_groundtruth_volumes_sl{}_rp{}{}.npy".format(nb_slices,repeat_slice,"")

filename_kdata = filename+"_kdata_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_kdata_gt = filename+"_kdata_mvt_gt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_nav = filename+"_nav_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_volume = filename+"_volumes_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_volume_corrected = filename+"_volumes_corrected_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

filename_mask= filename+"_mask_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")
filename_mask_corrected= filename+"_mask_corrected_mvt_sl{}_rp{}_us{}{}.npy".format(nb_slices,repeat_slice,undersampling_factor,"")

file_map = filename + "_mvt_sl{}_rp{}_us{}{}_MRF_map.pkl".format(nb_slices,repeat_slice,undersampling_factor,suffix)



#filename="./data/InVivo/Phantom20211028/meas_MID00028_FID39712_JAMBES_raFin_CLI.dat"



ntimesteps=175
nb_channels=1
nb_allspokes = 1400
npoint = 128



incoherent=True
mode="old"

image_size = (nb_slices, int(npoint/2), int(npoint/2))
nb_segments=nb_allspokes
nspoke=int(nb_segments/ntimesteps)
size = image_size[1:]

with open(dictjson) as f:
    dict_config = json.load(f)
dict_config["ff"]=np.arange(0.,1.05,0.05)

if "Square" in name:
    region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
    mask_reduction_factor=1/4


    m_ = RandomMap3D(name,dict_config,nb_slices=nb_filled_slices,nb_empty_slices=nb_empty_slices,undersampling_factor=undersampling_factor,repeat_slice=repeat_slice,resting_time=4000,image_size=size,region_size=region_size,mask_reduction_factor=mask_reduction_factor,gen_mode=gen_mode)

# elif name=="KneePhantom":
#     num =1
#     file_matlab_paramMap = "./data/{}/Phantom{}/paramMap.mat".format(name,num)
#
#     m = MapFromFile(name,image_size=image_size,file=file_matlab_paramMap,rounding=True,gen_mode="other")

else:
    raise ValueError("Unknown Name")



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

#if str.split(filename_groundtruth,"/")[-1] not in os.listdir(folder):
#    np.save(filename_groundtruth,m_.images_series[::nspoke])

# i=0
# image=m.images_series[i]


# file_mha = "/".join(["/".join(str.split(filename_volume, "/")[:-1]),
#                              "_".join(str.split(str.split(filename_volume, "/")[-1], ".")[:-1])]) + "_ideal_ts{}.mha".format(
#                                                                                                                    i)
# io.write(file_mha, np.abs(image), tags={"spacing": [5, 1, 1]})


radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode,is_random=False)

nb_channels=1


direction=np.array([0.0,8.0,0.0])
move = TranslationBreathing(direction,T=4000,frac_exp=0.7)

m_.add_movements([move])


if str.split(filename_kdata,"/")[-1] not in os.listdir(folder):

    #images = copy(m.images_series)
    data=m_.generate_kdata(radial_traj,useGPU=use_GPU)
    data=np.array(data)
    data=np.expand_dims(data,axis=0)
    np.save(filename_kdata, data)

else:
    #kdata_all_channels_all_slices = open_memmap(filename_kdata)
    data = np.load(filename_kdata)

nb_segments=radial_traj.get_traj().shape[0]

b1_full = np.ones(image_size)
b1_all_slices=np.expand_dims(b1_full,axis=0)



print("Processing Nav Data...")

nb_gating_spokes=50
ts_between_spokes=int(nb_allspokes/50)
timesteps = list(np.arange(1400)[::ts_between_spokes])

if str.split(filename_nav,"/")[-1] not in os.listdir(folder):

    nav_z=Navigator3D(direction=[1,0,0.0],applied_timesteps=timesteps,npoint=npoint)
    kdata_nav = m_.generate_kdata(nav_z,useGPU=use_GPU)

    kdata_nav=np.array(kdata_nav)

    data_for_nav = np.expand_dims(kdata_nav,axis=0)
    data_for_nav =  np.moveaxis(data_for_nav,1,2)
    data_for_nav = data_for_nav.astype("complex64")
    np.save(filename_nav,data_for_nav)

else:
    data_for_nav=np.load(filename_nav)

npoint_nav=data_for_nav.shape[-1]
nb_slices_nav=data_for_nav.shape[1]
nb_gating_spokes=data_for_nav.shape[-2]

all_timesteps = np.arange(nb_allspokes)
nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices_nav,
                       applied_timesteps=list(nav_timesteps))

nav_image_size = (int(npoint_nav / 2),)

print("Calculating Sensitivity Maps for Nav Images...")
b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
b1_nav_mean = np.mean(b1_nav, axis=(1, 2))

print("Rebuilding Nav Images...")
images_nav_mean = np.abs(simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, b1_nav_mean))


print("Estimating Movement...")
shifts = list(range(-10, 10))
bottom = 10
top = 54
displacements = calculate_displacement(images_nav_mean, bottom, top, shifts)

displacement_for_binning = displacements
bin_width = 2
max_bin = np.max(displacement_for_binning)
min_bin = np.min(displacement_for_binning)
plt.figure();plt.plot(displacements)

maxi = 0
for j in range(bin_width):
    min_bin = np.min(displacement_for_binning) + j
    bins = np.arange(min_bin, max_bin + bin_width, bin_width)
    #print(bins)
    categories = np.digitize(displacement_for_binning, bins)
    df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
    df_groups = df_cat.groupby("cat").count()
    curr_max = df_groups.displacement.max()
    if curr_max > maxi:
        maxi = curr_max
        df_groups_max=copy(df_groups)
        min_bin_max=min_bin
        max_bin_max=max_bin
        idx_cat = df_groups.displacement.idxmax()
        retained_nav_spokes = (categories == idx_cat)

retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
spoke_groups = np.argmin(np.abs(np.arange(0, nb_segments * nb_slices, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_slices,nb_segments / nb_gating_spokes).reshape(1,-1)),axis=-1)

if not (nb_segments == nb_gating_spokes):
    spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
        nb_segments / nb_gating_spokes / 2) + 1:] - 1
    spoke_groups = spoke_groups.flatten()

included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
displacements_real=m_.list_movements[0].paramDict["transformation"](m_.t.reshape(-1,1))[:,1]
#plt.figure();plt.plot(displacements_real[::(int(nb_allspokes/nb_gating_spokes))]);plt.plot(displacements+5,marker='x')
#plt.figure();plt.plot(displacements_real);plt.plot(included_spokes)
included_spokes=displacements_real>6

included_spokes[::int(nb_segments/nb_gating_spokes)]=False

weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, ntimesteps)


traj_python=radial_traj.get_traj()
traj_python=np.transpose(traj_python)


import cfl

traj_python_for_bart=traj_python.astype("complex64")
#traj_python_for_bart[:2,:,:]=traj_python
traj_python_for_bart[:2,:,:]=traj_python_for_bart[:2,:,:]/np.max(traj_python_for_bart[:2,:,:])*int(npoint/4)
traj_python_for_bart[2,:,:]=traj_python_for_bart[2,:,:]/np.max(traj_python_for_bart[2,:,:])*int(nb_slices/2)

cfl.writecfl("traj_simu",traj_python_for_bart)



kdata_multi_for_bart_full=data.reshape(nb_channels,nb_allspokes,-1).T
kdata_multi_for_bart_full=np.expand_dims(kdata_multi_for_bart_full,axis=0)
cfl.writecfl("kdata_multi_full_simu",kdata_multi_for_bart_full)

#sens done in Bart espirit
#bart fft -u $(bart bitmask 0 1) coil_img ksp
#bart ecalib -m1 ksp sens
#sens=cfl.readcfl("sens")
import os
sens=cfl.readcfl("sens")
b1_all_slices_for_bart=np.moveaxis(b1_all_slices,1,-1)
b1_all_slices_for_bart=np.moveaxis(b1_all_slices_for_bart,0,-1)

cfl.writecfl("sens_simu",b1_all_slices_for_bart)


#Dynamic reconstruction
traj=cfl.readcfl("traj_simu")
traj_reshaped = traj.reshape(3,-1,175,8)
traj_reshaped=np.moveaxis(traj_reshaped,-1,-2)
traj_reshaped=np.expand_dims(traj_reshaped,axis=(3,4,5,6,7,8,9))
cfl.writecfl("traj_simu_reshaped",traj_reshaped)


kdata_multi_for_bart_reshaped=kdata_multi_for_bart_full.reshape(1,-1,175,8,nb_channels)
kdata_multi_for_bart_reshaped=np.moveaxis(kdata_multi_for_bart_reshaped,2,-1)
kdata_multi_for_bart_reshaped=np.expand_dims(kdata_multi_for_bart_reshaped,axis=(4,5,6,7,8,9))
cfl.writecfl("kdata_multi_for_bart_simu_reshaped",kdata_multi_for_bart_reshaped)


weights_for_bart=np.tile(np.expand_dims(weights,axis=-1),tuple(np.ones(weights_for_bart.ndim).astype(int))+(npoint,))
weights_for_bart=weights_for_bart.reshape(nb_channels,175,8,-1)
weights_for_bart=np.moveaxis(weights_for_bart,0,-1)
weights_for_bart=np.moveaxis(weights_for_bart,0,-1)
weights_for_bart=np.moveaxis(weights_for_bart,0,1)

weights_for_bart=np.expand_dims(weights_for_bart,axis=(0,4,5,6,7,8,9))
cfl.writecfl("weights_simu",weights_for_bart)



import cfl
iter=0

bart_command="bart pics {} -i1 -RT:$(bart bitmask 10):0:0.01 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#looks like the best for now
#bart_command="bart pics {} -i1 -RT:$(bart bitmask 0 1 2):0:0.0001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"
#bart_command="bart pics {} -i1 -s 0.01 -p dens_adj_bart -RL:$(bart bitmask 0 1 2):$(bart bitmask 0 1 2):0.00001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"# does not work well
#bart_command="bart pics {} -i1 -s 0.01 -p dens_adj_bart -RL:$(bart bitmask 0 1 2):0:0.0000001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#does not work well
#bart_command="bart pics {} -m -p dens_adj_bart -i1 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#second best
bart_command="bart pics {} -i1 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"#second best

#bart_command="bart pics {} -m -p dens_adj_bart -i1 -t traj_reshaped kdata_multi_for_bart_reshaped_cc sens_cc out{}"#second best
bart_command="bart pics --no-toeplitz --lowmem-stack=8 {} -m -i1 -RT:$(bart bitmask 2):0:0.0001 -t traj_simu_reshaped kdata_multi_for_bart_simu_reshaped sens_simu out{}"#second best
bart_command="bart pics {} -p weights_simu -m -i1 -RT:$(bart bitmask 2):0:0.0001 -t traj_simu_reshaped kdata_multi_for_bart_simu_reshaped sens_simu out{}"#second best
bart_command="bart pics {} -p weights_simu -i1 -t traj_simu_reshaped kdata_multi_for_bart_simu_reshaped sens_simu out{}"#second best


#bart_command="bart pics {} -m -i1 -RW:$(bart bitmask 0 1 2):0:0.001 -t traj_reshaped kdata_multi_for_bart_reshaped sens out{}"
#bart_command="bart pics -B basis_used {} -i1 -t traj_reshaped_sbreco kdata_multi_for_bart_reshaped_sbreco sens out{}"#looks like the best for now
#bart_command="bart pics -m -C1 -B basis_used {} -i1 -t traj_reshaped_sbreco kdata_multi_for_bart_reshaped_sbreco sens out{}"#looks like the best for now


#traj_reshaped_sbreco=cfl.readcfl("traj_reshaped_sbreco")
#kdata_multi_for_bart_reshaped_sbreco=cfl.readcfl("kdata_multi_for_bart_reshaped_sbreco")

os.system(bart_command.format("",iter))
out=cfl.readcfl("out{}".format(iter))

sl=int(nb_slices/2)
animate_images(np.moveaxis(out[:,:,sl].squeeze(),-1,0))

mask = np.load(filename_mask_corrected)

optimizer = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=radial_traj, split=100, pca=True,
                                     threshold_pca=15, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=ntimesteps,
                                     b1=b1_all_slices, threshold_ff=0.9, dictfile_light=dictfile_light,
                                     return_matched_signals=True)  # ,mu_TV=1,weights_TV=[1.,0.,0.])
#np.matmul(out.squeeze(),basis.squeeze().T.conj()).shape

return_cost=True

niter=50
all_maps_bart_all_iter={}

for iter in tqdm(range(1,niter+1)):
    if "basis_used" in bart_command:
        out=np.matmul(out.squeeze(),basis.squeeze().T)
    all_maps_bart,matched_signals = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, np.moveaxis(np.moveaxis(out.squeeze(),-1,0),-1,1),retained_timesteps=None)

    all_maps_bart_all_iter[iter-1]=all_maps_bart[0]

    # all_maps_bart, matched_signals = optimizer.search_patterns_test_multi(dictfile, np.moveaxis(out.squeeze(), -1, 0))
    # all_maps_bart_all_iter[iter - 1] = all_maps_bart[0]


    if iter==niter:
        break

    if "basis_used" in bart_command:
        matched_signals = np.matmul(basis.squeeze().T.conj(), matched_signals)
    matched_signals=[makevol(s,mask>0) for s in matched_signals]

    matched_signals=np.array(matched_signals)
    matched_signals_bart=np.moveaxis(matched_signals,0,-1)
    matched_signals_bart = np.moveaxis(matched_signals_bart, 0, -2)
    if "basis_used" in bart_command:
        matched_signals_bart = np.expand_dims(matched_signals_bart, axis=(3, 4,5))
    else:
        matched_signals_bart=np.expand_dims(matched_signals_bart,axis=(3,4,5,6,7,8,9))
    cfl.writecfl("image_start",matched_signals_bart)

    os.system(bart_command.format("-W image_start", iter))
    out=cfl.readcfl("out{}".format(iter))

curr_file = filename.split(".dat")[0] + "{}_MRF_map.pkl".format("_bart_admm_reg_z")

import pickle
with open(curr_file,"wb") as file:
    pickle.dump(all_maps_bart_all_iter,file)

all_maps = all_maps_bart_all_iter

dx=1
dy=1
dz=5
for iter in list(all_maps.keys()):

    map_rebuilt=all_maps[iter][0]
    mask=all_maps[iter][1]

    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
    #map_Python.buildParamMap()


    for key in ["ff","wT1","df","attB1"]:
        file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_it{}_{}.mha".format(iter,key)
        io.write(file_mha,map_for_sim[key],tags={"spacing":[dz,dx,dy]})

    if return_cost:
        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "_it{}_{}.mha".format(
            iter, "correlation")
        io.write(file_mha, makevol(all_maps[iter][2],mask>0), tags={"spacing": [dz, dx, dy]})

        file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                             "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "_it{}_{}.mha".format(
            iter, "phase")
        io.write(file_mha, makevol(all_maps[iter][3],mask>0), tags={"spacing": [dz, dx, dy]})

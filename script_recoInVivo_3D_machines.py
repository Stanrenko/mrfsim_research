
#import matplotlib
#matplotlib.use("TkAgg")
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
import twixtools

# machines
path = r"/home/cslioussarenko/PythonRepositories"
#path = r"/Users/constantinslioussarenko/PythonGitRepositories/MyoMap"

import sys
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")

from machines import machine, Toolbox, Config, set_parameter, set_output, printer, file_handler, Parameter, RejectException, get_context


@machine
@set_parameter("filename", str, default=None, description="Siemens K-space data .dat file")
@set_parameter("suffix",str,default="")
def build_kdata(filename,suffix):

    filename_kdata = str.split(filename, ".dat")[0] + suffix + "_kdata.npy"
    filename_save = str.split(filename, ".dat")[0] + ".npy"
    filename_seqParams = str.split(filename, ".dat")[0] + "_seqParams.pkl"

    folder = "/".join(str.split(filename, "/")[:-1])

    if str.split(filename_seqParams, "/")[-1] not in os.listdir(folder):

        twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock", "sKSpace"],
                                   optional_additional_arrays=["SliceThickness"])

        alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
        x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
        y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
        z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]

        dico_seqParams = {"alFree": alFree, "x_FOV": x_FOV, "y_FOV": y_FOV, "z_FOV": z_FOV}

        del alFree

        file = open(filename_seqParams, "wb")
        pickle.dump(dico_seqParams, file)
        file.close()

    else:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)

    if str.split(filename_save, "/")[-1] not in os.listdir(folder):
        if 'twix' not in locals():
            print("Re-loading raw data")
            twix = twixtools.read_twix(filename)

        mapped = twixtools.map_twix(twix)

        try:
            del twix
        except:
            pass

        data = mapped[-1]['image']
        del mapped

        data = data[:].squeeze()
        data = np.moveaxis(data, 0, -2)
        data = np.moveaxis(data, 1, 0)

        np.save(filename_save, data)

    else:
        data = np.load(filename_save)

    try:
        del twix
    except:
        pass

    npoint = data.shape[-1]
    #image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    print("Performing Density Adjustment....")
    density = np.abs(np.linspace(-1, 1, npoint))
    density = np.expand_dims(density, tuple(range(data.ndim - 1)))
    data *= density
    np.save(filename_kdata, data)

    return

@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("suffix",str,default="")
def build_coil_sensi(filename_kdata,sampling_mode,undersampling_factor,suffix):

    kdata_all_channels_all_slices = np.load(filename_kdata)
    filename_b1 = str.split(filename_kdata, "_kdata.npy")[0] + "_b1" + suffix +".npy"

    sampling_mode_list = str.split(sampling_mode,"_")

    if sampling_mode_list[0]=="stack":
        incoherent=False
    else:
        incoherent=True

    if len(sampling_mode_list)>1:
        mode=sampling_mode_list[1]
    else:
        mode="old"

    data_shape = kdata_all_channels_all_slices.shape
    nb_allspokes = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2]
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                           nb_slices=nb_slices, incoherent=incoherent, mode=mode)

    res = 16
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))
    b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                 useGPU=False, light_memory_usage=True)
    np.save(filename_b1, b1_all_slices)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("ntimesteps", int, default=175, description="Number of timesteps for the image serie")
@set_parameter("use_GPU", bool, default=True, description="Use GPU")
@set_parameter("light_mem", bool, default=True, description="Memory usage")
@set_parameter("suffix",str,default="")
def build_volumes(filename_kdata,sampling_mode,undersampling_factor,ntimesteps,use_GPU,light_mem,suffix):

    print("I was here") 
    kdata_all_channels_all_slices = np.load(filename_kdata)
    print("I was here 2") 
    filename_b1 = str.split(filename_kdata, "_kdata.npy")[0] + "_b1.npy"
    print(filename_b1) 
    b1_all_slices=np.load(filename_b1)

    filename_volume = str.split(filename_kdata, "_kdata.npy")[0] + "_volumes"+suffix+".npy"
    print(filename_volume)
    sampling_mode_list = str.split(sampling_mode, "_")
    print(sampling_mode_list)
    if sampling_mode_list[0] == "stack":
        incoherent = False
    else:
        incoherent = True

    if len(sampling_mode_list) > 1:
        mode = sampling_mode_list[1]
    else:
        mode = "old"

    data_shape = kdata_all_channels_all_slices.shape
    nb_allspokes = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2]
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                           nb_slices=nb_slices, incoherent=incoherent, mode=mode)

    volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices, radial_traj, image_size,
                                                            b1=b1_all_slices, density_adj=False, ntimesteps=ntimesteps,
                                                            useGPU=use_GPU, normalize_kdata=True, memmap_file=True,
                                                            light_memory_usage=light_mem,normalize_volumes=True)
    np.save(filename_volume, volumes_all)
    return

@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("suffix",str,default="")
def build_mask(filename_kdata,sampling_mode,undersampling_factor,suffix):
    kdata_all_channels_all_slices = np.load(filename_kdata)

    print(filename_kdata) 
    filename_b1 = str.split(filename_kdata, "_kdata.npy")[0] + "_b1.npy"
    print(filename_b1) 
    b1_all_slices=np.load(filename_b1)

    filename_mask = str.split(filename_kdata, "_kdata.npy")[0] + "_mask"+suffix+".npy"

    sampling_mode_list = str.split(sampling_mode, "_")

    if sampling_mode_list[0] == "stack":
        incoherent = False
    else:
        incoherent = True

    if len(sampling_mode_list) > 1:
        mode = sampling_mode_list[1]
    else:
        mode = "old"

    data_shape = kdata_all_channels_all_slices.shape
    nb_allspokes = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2]
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                           nb_slices=nb_slices, incoherent=incoherent, mode=mode)

    selected_spokes=np.r_[10:400] 
    selected_spokes=None 
    mask = build_mask_single_image_multichannel(kdata_all_channels_all_slices, radial_traj, image_size,
                                                b1=b1_all_slices, density_adj=False, threshold_factor=None,
                                                normalize_kdata=True, light_memory_usage=True,selected_spokes=selected_spokes)
    np.save(filename_mask, mask)
    return


toolbox = Toolbox("script_recoInVivo_3D_machines", description="Reading Siemens 3D MRF data and performing image series reconstruction")
toolbox.add_program("build_kdata", build_kdata)
toolbox.add_program("build_coil_sensi", build_coil_sensi)
toolbox.add_program("build_volumes", build_volumes)
toolbox.add_program("build_mask", build_mask)

if __name__ == "__main__":
    toolbox.cli()

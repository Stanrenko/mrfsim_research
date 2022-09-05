
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,BruteDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
import twixtools
from PIL import Image
from mutools import io
import pandas as pd


try :
    import cupy as cp
except:
    pass
# machines
path = r"/home/cslioussarenko/PythonRepositories"
#path = r"/Users/constantinslioussarenko/PythonGitRepositories/MyoMap"

import sys
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")

from machines import machine, Toolbox, Config, set_parameter, set_output, printer, file_handler, Parameter, RejectException, get_context

DEFAULT_OPT_CONFIG="opt_config.json"

@machine
@set_parameter("filename", str, default=None, description="Siemens K-space data .dat file")
@set_parameter("suffix",str,default="")
def build_kdata(filename,suffix):

    filename_kdata = str.split(filename, ".dat")[0] + suffix + "_kdata.npy"
    filename_save = str.split(filename, ".dat")[0] + ".npy"
    filename_nav_save = str.split(filename, ".dat")[0] + "_nav.npy"
    filename_seqParams = str.split(filename, ".dat")[0] + "_seqParams.pkl"

    folder = "/".join(str.split(filename, "/")[:-1])


    if str.split(filename_seqParams, "/")[-1] not in os.listdir(folder):

        twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock", "sKSpace"],
                                   optional_additional_arrays=["SliceThickness"])

        if np.max(np.argwhere(np.array(twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]) > 0)) >= 16:
            use_navigator_dll = True
        else:
            use_navigator_dll = False

        alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
        x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
        y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
        z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]

        nb_part = twix[-1]["hdr"]["Meas"]["Partitions"]

        dico_seqParams = {"alFree": alFree, "x_FOV": x_FOV, "y_FOV": y_FOV, "z_FOV": z_FOV,
                          "use_navigator_dll": use_navigator_dll, "nb_part": nb_part}

        del alFree

        file = open(filename_seqParams, "wb")
        pickle.dump(dico_seqParams, file)
        file.close()

    else:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

    use_navigator_dll = dico_seqParams["use_navigator_dll"]

    if use_navigator_dll:
        meas_sampling_mode = dico_seqParams["alFree"][14]
        nb_gating_spokes = dico_seqParams["alFree"][6]
    else:
        meas_sampling_mode = dico_seqParams["alFree"][12]
        nb_gating_spokes = 0

    nb_segments = dico_seqParams["alFree"][4]
    nb_part = dico_seqParams["nb_part"]

    del dico_seqParams

    if str.split(filename_save, "/")[-1] not in os.listdir(folder):
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
            data_for_nav = data_for_nav.reshape((int(nb_part), int(nb_gating_spokes)) + data_for_nav.shape[1:])

            if data_for_nav.ndim == 3:
                data_for_nav = np.expand_dims(data_for_nav, axis=-2)

            data_for_nav = np.moveaxis(data_for_nav, -2, 0)
            np.save(filename_nav_save, data_for_nav)

        data = np.array([mdb.data for mdb in data])
        data = data.reshape((-1, int(nb_segments)) + data.shape[1:])
        data = np.moveaxis(data, 2, 0)
        data = np.moveaxis(data, 2, 1)

        del mdb_list

        ##################################################
        try:
            del twix
        except:
            pass

        np.save(filename_save, data)

    else:
        data = np.load(filename_save)
        if nb_gating_spokes > 0:
            data_for_nav = np.load(filename_nav_save)

    npoint = data.shape[-1]
    #image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    print("Performing Density Adjustment....")
    density = np.abs(np.linspace(-1, 1, npoint))
    density = np.expand_dims(density, tuple(range(data.ndim - 1)))
    data *= density
    np.save(filename_kdata, data)

    return



@machine
@set_parameter("filename", str, default=None, description="Siemens K-space data .dat file")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("nspokes_per_z_encoding", int, default=8, description="Number of spokes per z encoding")
@set_parameter("nb_ref_lines", int, default=24, description="Number of reference lines for Grappa calibration")
@set_parameter("suffix",str,default="")
def build_data_for_grappa_simulation(filename,undersampling_factor,nspokes_per_z_encoding,nb_ref_lines,suffix):

    filename_kdata = str.split(filename, ".dat")[0] + suffix + "_kdata.npy"
    filename_save = str.split(filename, ".dat")[0] + ".npy"
    filename_seqParams = str.split(filename, ".dat")[0] + "_seqParams.pkl"
    filename_save_us = str.split(filename, ".dat")[0] + "_us{}ref{}_us.npy".format(undersampling_factor, nb_ref_lines)
    filename_save_calib = str.split(filename, ".dat")[0] + "_us{}ref{}_calib.npy".format(undersampling_factor,
                                                                                         nb_ref_lines)


    folder = "/".join(str.split(filename, "/")[:-1])

    if str.split(filename_seqParams, "/")[-1] not in os.listdir(folder):

        twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock", "sKSpace"],
                                   optional_additional_arrays=["SliceThickness"])

        alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
        x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
        y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
        z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]

        nb_slices = twix[-1]["hdr"]["Meas"]["lPartitions"]

        dico_seqParams = {"alFree": alFree, "x_FOV": x_FOV, "y_FOV": y_FOV, "z_FOV": z_FOV, "US": undersampling_factor,
                          "nb_slices": nb_slices, "nb_ref_lines": nb_ref_lines}

        del alFree

        file = open(filename_seqParams, "wb")
        pickle.dump(dico_seqParams, file)
        file.close()

    else:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)

    nb_segments = dico_seqParams["alFree"][4]
    nb_slices = dico_seqParams["nb_slices"]
    del dico_seqParams

    if str.split(filename_save_us, "/")[-1] not in os.listdir(folder):

        if str.split(filename_save, "/")[-1] not in os.listdir(folder):
            if 'twix' not in locals():
                print("Re-loading raw data")
                twix = twixtools.read_twix(filename)
            mdb_list = twix[-1]['mdb']
            data = []

            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    data.append(mdb)

            data = np.array([mdb.data for mdb in data])
            data = data.reshape((-1, int(nb_segments)) + data.shape[1:])
            data = np.moveaxis(data, 2, 0)
            data = np.moveaxis(data, 2, 1)

            try:
                del twix
            except:
                pass

            np.save(filename_save, data)

        else:
            data = np.load(filename_save)

        data = groupby(data, nspokes_per_z_encoding, axis=1)
        # data=np.array(data)

        all_lines = np.arange(nb_slices).astype(int)

        shift = 0
        data_calib = []

        for ts in tqdm(range(len(data))):
            lines_measured = all_lines[shift::undersampling_factor]
            lines_to_estimate = list(set(all_lines) - set(lines_measured))
            center_line = int(nb_slices / 2)
            lines_ref = all_lines[(center_line - int(nb_ref_lines / 2)):(center_line + int(nb_ref_lines / 2))]

            data_calib.append(data[ts][:, :, lines_ref, :])
            data[ts] = data[ts][:, :, lines_measured, :]
            shift += 1
            shift = shift % (undersampling_factor)

        data_calib = np.array(data_calib)
        data = np.array(data)

        data = np.moveaxis(data, 0, 1)
        data_calib = np.moveaxis(data_calib, 0, 1)

        data = data.reshape(data.shape[0], -1, data.shape[-2], data.shape[-1])
        data_calib = data_calib.reshape(data_calib.shape[0], -1, data_calib.shape[-2], data_calib.shape[-1])

        np.save(filename_save_us, data)
        np.save(filename_save_calib, data_calib)


    try:
        del twix
    except:
        pass


    return


@machine
@set_parameter("filename_save_calib", str, default=None, description="Calib data .npy file")
@set_parameter("nspokes_per_z_encoding", int, default=8, description="Number of spokes per z encoding")
@set_parameter("len_kery", int, default=2, description="Number of lines in each kernel")
@set_parameter("len_kerx", int, default=7, description="Number of readout points in each line of the kernel")
@set_parameter("calibration_mode", ["Standard","Tikhonov"], default="Standard", description="Calibration methodology for Grappa")
@set_parameter("lambd", float, default=None, description="Regularization parameter")
@set_parameter("suffix", str, default="")
def calib_and_estimate_kdata_grappa(filename_save_calib, nspokes_per_z_encoding,len_kery,len_kerx,calibration_mode,lambd,suffix):
    data_calib = np.load(filename_save_calib)
    nb_channels=data_calib.shape[0]

    filename_split = str.split(filename_save_calib,"_us")
    filename = filename_split[0]+".dat"
    filename_seqParams = str.split(filename, ".dat")[0] + "_seqParams.pkl"

    filename_split_params = filename_split[1].split("ref")
    undersampling_factor = int(filename_split_params[0])
    nb_ref_lines=int(filename_split_params[1].split("_")[0])

    filename_save_us = str.split(filename, ".dat")[0] + "_us{}ref{}_us.npy".format(undersampling_factor, nb_ref_lines)

    if calibration_mode == "Tikhonov":
        if lambd is None:
            lambd = 0.01
            print("Warning : lambd was not set for Tikhonov calibration. Using default value of 0.01")

        str_lambda=str(lambd)
        str_lambda=str.split(str_lambda,".")
        str_lambda="_".join(str_lambda)
        filename_kdata_grappa = str.split(filename, ".dat")[0] + "_{}_{}_us{}ref{}_kdata_grappa.npy".format(calibration_mode,str_lambda,undersampling_factor,
                                                                                                  nb_ref_lines)

        filename_currtraj_grappa = str.split(filename, ".dat")[0] + "_{}_{}_us{}ref{}_currtraj_grappa.npy".format(calibration_mode,str_lambda,
            undersampling_factor, nb_ref_lines)

    else:
        filename_kdata_grappa = str.split(filename, ".dat")[0] + "_{}__us{}ref{}_kdata_grappa.npy".format(
            undersampling_factor,
            nb_ref_lines)

        filename_currtraj_grappa = str.split(filename, ".dat")[0] + "_us{}ref{}_currtraj_grappa.npy".format(
            undersampling_factor, nb_ref_lines)

    print(filename_kdata_grappa)

    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()

    data = np.load(filename_save_us)

    indices_kz = list(range(1, (undersampling_factor)))

    kernel_y = np.arange(len_kery)
    kernel_x = np.arange(-(int(len_kerx / 2) - 1), int(len_kerx / 2))

    pad_y = ((np.array(kernel_y) < 0).sum(), (np.array(kernel_y) > 0).sum())
    pad_x = ((np.array(kernel_x) < 0).sum(), (np.array(kernel_x) > 0).sum())

    data_calib_padded = np.pad(data_calib, ((0, 0), (0, 0), pad_y, pad_x), mode="constant")
    data_padded = np.pad(data, ((0, 0), (0, 0), pad_y, pad_x), mode="constant")

    kdata_all_channels_completed_all_ts = []
    curr_traj_completed_all_ts = []

    replace_calib_lines = True
    useGPU = True

    shift=0

    nb_segments = dico_seqParams["alFree"][4]
    nb_allspokes= nb_segments
    npoint=data.shape[-1]
    nb_slices = dico_seqParams["nb_slices"]

    meas_sampling_mode = dico_seqParams["alFree"][12]
    if meas_sampling_mode == 1:
        incoherent = False
        mode = None
    elif meas_sampling_mode == 2:
        incoherent = True
        mode = "old"
    elif meas_sampling_mode == 3:
        incoherent = True
        mode = "new"

    radial_traj_all = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint, nb_slices=nb_slices,
                               incoherent=incoherent, mode=mode, nspoke_per_z_encoding=nspokes_per_z_encoding)
    traj_all = radial_traj_all.get_traj().reshape(nb_allspokes, nb_slices, npoint, 3)

    all_lines = np.arange(nb_slices).astype(int)


    for ts in tqdm(range(nb_allspokes)):

        if ((ts) % nspokes_per_z_encoding == 0) and (not (ts == 0)):
            shift += 1
            shift = shift % (undersampling_factor)

        lines_measured = all_lines[shift::undersampling_factor]
        lines_to_estimate = list(set(all_lines) - set(lines_measured))
        center_line = int(nb_slices / 2)
        lines_ref = all_lines[(center_line - int(nb_ref_lines / 2)):(center_line + int(nb_ref_lines / 2))]

        calib_lines = []
        calib_lines_ref_indices = []
        for i in indices_kz:
            current_calib_lines = []
            current_calib_lines_ref_indices = []

            for j in range(undersampling_factor * (kernel_y[0]) + i,
                           nb_ref_lines - (kernel_y[-1] * undersampling_factor - i)):
                current_calib_lines.append(lines_ref[j])
                current_calib_lines_ref_indices.append(j)
            calib_lines.append(current_calib_lines)
            calib_lines_ref_indices.append(current_calib_lines_ref_indices)

        calib_lines = np.array(calib_lines).astype(int)
        calib_lines_ref_indices = np.array(calib_lines_ref_indices).astype(int)

        weights = np.zeros((len(calib_lines), nb_channels, nb_channels * len(kernel_x) * len(kernel_y)),
                           dtype=data_calib.dtype)

        for index_ky_target in range(len(calib_lines_ref_indices)):
            calib_lines_ref_index = calib_lines_ref_indices[index_ky_target]
            F_target_calib = data_calib[:, ts, calib_lines_ref_indices[index_ky_target], :]  # .reshape(nb_channels,-1)
            F_source_calib = np.zeros((nb_channels * len(kernel_x) * len(kernel_y),) + F_target_calib.shape[1:],
                                      dtype=F_target_calib.dtype)

            for i, l in enumerate(calib_lines_ref_indices[index_ky_target]):
                l0 = l - (index_ky_target + 1)
                for j in range(npoint):
                    local_indices = np.stack(np.meshgrid(pad_y[0] + l0 + undersampling_factor * np.array(kernel_y),
                                                         pad_x[0] + j + np.array(kernel_x)), axis=0).T.reshape(-1, 2)
                    F_source_calib[:, i, j] = data_calib_padded[:, ts, local_indices[:, 0],
                                              local_indices[:, 1]].flatten()

            F_source_calib = F_source_calib.reshape(nb_channels * len(kernel_x) * len(kernel_y), -1)
            F_target_calib = F_target_calib.reshape(nb_channels, -1)

            if calibration_mode == "Tikhonov":
                F_source_calib_cupy = cp.asarray(F_source_calib)
                F_target_calib_cupy = cp.asarray(F_target_calib)
                weights[index_ky_target] = (F_target_calib_cupy @ F_source_calib_cupy.conj().T @ cp.linalg.inv(
                    F_source_calib_cupy @ F_source_calib_cupy.conj().T + lambd * cp.eye(
                        F_source_calib_cupy.shape[0]))).get()

            else:
                weights[index_ky_target] = (
                            cp.asarray(F_target_calib) @ cp.linalg.pinv(cp.asarray(F_source_calib))).get()


        F_target_estimate = np.zeros((nb_channels, len(lines_to_estimate), npoint), dtype=data.dtype)

        for i, l in (enumerate(lines_to_estimate)):
            F_source_estimate = np.zeros((nb_channels * len(kernel_x) * len(kernel_y), npoint),
                                         dtype=data_calib.dtype)
            index_ky_target = (l) % (undersampling_factor) - 1
            try:
                l0 = np.argwhere(np.array(lines_measured - shift) == l - (index_ky_target + 1))[0][0]
            except:
                print("Line {} for Ts {} not estimated".format(l, ts))
                continue
            for j in range(npoint):
                local_indices = np.stack(
                    np.meshgrid(pad_y[0] + l0 + np.array(kernel_y), pad_x[0] + j + np.array(kernel_x)),
                    axis=0).T.reshape(-1, 2)
                F_source_estimate[:, j] = data_padded[:, ts, local_indices[:, 0], local_indices[:, 1]].flatten()
            F_target_estimate[:, i, :] = weights[index_ky_target] @ F_source_estimate

        F_target_estimate[:,:,:pad_x[0]]=0
        F_target_estimate[:, :, -pad_x[1]:] = 0

        max_measured_line=np.max(lines_measured)
        min_measured_line=np.min(lines_measured)

        for i, l in (enumerate(lines_to_estimate)):
            if (l<min_measured_line)or(l>max_measured_line):
                F_target_estimate[:, i, :]=0

        for i, l in (enumerate(lines_to_estimate)):
            if l in lines_ref.flatten():
                ind_line = np.argwhere(lines_ref.flatten() == l).flatten()[0]
                F_target_estimate[:, i, :] = data_calib[:, ts, ind_line, :]

        kdata_all_channels_completed = np.concatenate([data[:, ts], F_target_estimate], axis=1)
        curr_traj_estimate = traj_all[ts, lines_to_estimate, :, :]
        curr_traj_completed = np.concatenate([traj_all[ts,lines_measured,:,:], curr_traj_estimate],
                                             axis=0)

        kdata_all_channels_completed = kdata_all_channels_completed.reshape((nb_channels, -1))
        curr_traj_completed = curr_traj_completed.reshape((-1, 3))

        kdata_all_channels_completed_all_ts.append(kdata_all_channels_completed)
        curr_traj_completed_all_ts.append(curr_traj_completed)

    kdata_all_channels_completed_all_ts = np.array(kdata_all_channels_completed_all_ts)
    kdata_all_channels_completed_all_ts = np.moveaxis(kdata_all_channels_completed_all_ts, 0, 1)
    curr_traj_completed_all_ts = np.array(curr_traj_completed_all_ts)

    for ts in tqdm(range(curr_traj_completed_all_ts.shape[0])):
        ind = np.lexsort((curr_traj_completed_all_ts[ts][:, 0], curr_traj_completed_all_ts[ts][:, 2]))
        curr_traj_completed_all_ts[ts] = curr_traj_completed_all_ts[ts, ind, :]
        kdata_all_channels_completed_all_ts[:, ts] = kdata_all_channels_completed_all_ts[:, ts, ind]

    kdata_all_channels_completed_all_ts = kdata_all_channels_completed_all_ts.reshape(nb_channels, nb_allspokes,
                                                                                      nb_slices, npoint)
    #kdata_all_channels_completed_all_ts[:, :, :, :pad_x[0]] = 0
    #kdata_all_channels_completed_all_ts[:, :, :, -pad_x[1]:] = 0
    #kdata_all_channels_completed_all_ts[:, :, :((undersampling_factor - 1) * pad_y[0]), :] = 0
    #kdata_all_channels_completed_all_ts[:, :, (-(undersampling_factor - 1) * pad_y[1]):, :] = 0

    np.save(filename_kdata_grappa, kdata_all_channels_completed_all_ts)
    np.save(filename_currtraj_grappa, curr_traj_completed_all_ts)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("dens_adj", bool, default=False, description="Memory usage")
@set_parameter("suffix",str,default="")
def build_coil_sensi(filename_kdata,filename_traj,sampling_mode,undersampling_factor,dens_adj,suffix):

    kdata_all_channels_all_slices = np.load(filename_kdata)
    filename_b1 = ("_b1"+suffix).join(str.split(filename_kdata, "_kdata"))

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
    nb_slices = data_shape[2]*undersampling_factor
    #image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts = np.load(filename_traj)
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    res = 16
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))
    b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                 useGPU=False, light_memory_usage=True,density_adj=dens_adj)

    image_file=str.split(filename_b1, ".npy")[0] + suffix + ".jpg"

    sl = int(b1_all_slices.shape[1]/2)

    list_images=list(np.abs(b1_all_slices[:,sl,:,:]))
    plot_image_grid(list_images,(6,6),title="Sensivitiy map for slice".format(sl),save_file=image_file)

    np.save(filename_b1, b1_all_slices)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("ntimesteps", int, default=175, description="Number of timesteps for the image serie")
@set_parameter("use_GPU", bool, default=True, description="Use GPU")
@set_parameter("light_mem", bool, default=True, description="Memory usage")
@set_parameter("dens_adj", bool, default=False, description="Density adjustment for radial data")
@set_parameter("suffix",str,default="")
def build_volumes(filename_kdata,filename_traj,sampling_mode,undersampling_factor,ntimesteps,use_GPU,light_mem,dens_adj,suffix):

    kdata_all_channels_all_slices = np.load(filename_kdata)
    filename_b1 = ("_b1" + suffix).join(str.split(filename_kdata, "_kdata"))

    b1_all_slices=np.load(filename_b1)
    filename_volume = ("_volumes"+suffix).join(str.split(filename_kdata, "_kdata"))
    print(filename_volume)
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
    nb_slices = data_shape[2]*undersampling_factor
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts=np.load(filename_traj)
        radial_traj=Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                                  nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices, radial_traj, image_size,
                                                            b1=b1_all_slices, density_adj=dens_adj, ntimesteps=ntimesteps,
                                                            useGPU=use_GPU, normalize_kdata=False, memmap_file=None,
                                                            light_memory_usage=light_mem,normalize_volumes=True)
    np.save(filename_volume, volumes_all)


    gif=[]
    sl=int(volumes_all.shape[1]/2)
    volume_for_gif = np.abs(volumes_all[:,sl,:,:])
    for i in range(volume_for_gif.shape[0]):
        img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
        img=img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_volume,".npy") [0]+".gif"
    gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("filename_nav_save", str, default=None, description="Saved K-space navigator data .npy file")
@set_parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("light_mem", bool, default=True, description="Memory usage")
@set_parameter("dens_adj", bool, default=False, description="Density adjustment for radial data")
@set_parameter("suffix",str,default="")
def build_data_autofocus(filename_kdata,filename_traj,filename_nav_save,sampling_mode,undersampling_factor,light_mem,dens_adj,suffix):
    # b1_all_slices=b1_full

    print("Loading Files...")
    kdata_all_channels_all_slices = np.load(filename_kdata)

    print("Loading coil sensi")
    filename_b1 = ("_b1" + suffix).join(str.split(filename_kdata, "_kdata"))
    b1_all_slices = np.load(filename_b1)

    print("Parsing parameters")
    filename_kdata_modif = ("_kdata_modif" + suffix).join(str.split(filename_kdata, "_kdata"))
    #print(filename_volume)
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
    nb_segments = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2] * undersampling_factor
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    print(data_shape)

    print("Processing Nav Data...")
    data_for_nav = np.load(filename_nav_save)

    print(data_for_nav.shape)

    nb_allspokes = nb_segments
    nb_slices = data_for_nav.shape[1]
    nb_gating_spokes=data_for_nav.shape[2]
    nb_channels = data_for_nav.shape[0]
    npoint_for_nav = data_for_nav.shape[-1]


    all_timesteps = np.arange(nb_allspokes)
    nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

    nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_for_nav, nb_slices=nb_slices,
                           applied_timesteps=list(nav_timesteps))

    nav_image_size = (int(npoint_for_nav / 2),)

    print("Rebuilding Nav Images...")
    #print("Calculating Sensitivity Maps for Nav Images...")
    #b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
    #b1_nav_mean = np.mean(b1_nav, axis=(1, 2))

    #images_nav = np.abs(
    #    simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, b1=b1_nav_mean))

    best_ch=9
    images_nav = np.abs(simulate_nav_images_multi(np.expand_dims(data_for_nav[best_ch],axis=0), nav_traj, nav_image_size, b1=None))




    #print("Estimating Movement...")
    shifts = list(range(-30, 30))
    #bottom = -shifts[0]
    #top = nav_image_size[0]-shifts[-1]

    bottom = 40
    top=120
    displacements = calculate_displacement(images_nav, bottom, top, shifts, lambda_tv=0.00)

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts=np.load(filename_traj)
        radial_traj=Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                                  nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_slices, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_slices,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
        axis=-1)

    if not (nb_segments == nb_gating_spokes):
        spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
        spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
            nb_segments / nb_gating_spokes / 2) + 1:] - 1
        spoke_groups = spoke_groups.flatten()

    displacements_extrapolated = np.array([displacements[j] for j in spoke_groups])

    #x = np.arange(-1.0, 1.01, 0.1)
    #y = np.array([0])
    #z = np.arange(-0.5, 0.51, 0.1)

    #print("Calculating Entropy for all movement ranges...")
    #entropy_all = []
    #ent_min = np.inf
    #for dx in tqdm(x):
    #    entropy_xzy = []
    #    for dz in z:
    #        entropy_zy=[]
    #        for dy in y:
    #            alpha = np.array([dx, dy, dz])
    #            dr = np.expand_dims(alpha, axis=(0, 1)) * np.expand_dims(
    #                displacements_extrapolated.reshape(nb_slices, nb_segments).T, axis=(2))
    #            modif = np.exp(
    #                1j * np.sum((radial_traj.get_traj().reshape(nb_segments, -1, npoint, 3) * np.expand_dims(dr, axis=2)),
    #                            axis=-1))
    #            data_modif = kdata_all_channels_all_slices * modif
    #            volume_full_modif = \
    #                simulate_radial_undersampled_images_multi(data_modif, radial_traj, image_size, b1=b1_all_slices,
    #                                                          density_adj=dens_adj, ntimesteps=1, useGPU=False,
    #                                                          normalize_kdata=False, memmap_file=None,
    #                                                          light_memory_usage=light_mem,
    #                                                          normalize_volumes=True)[0]
    #            ent = calc_grad_entropy(volume_full_modif)
    #            entropy_zy.append(ent)
    #            if ent < ent_min:
    #                modif_final = modif
    #                alpha_min = alpha
    #                ent_min = ent

    #        entropy_xzy.append(entropy_zy)
    #    entropy_all.append(entropy_xzy)

    #print("Alpha min: {}".format(alpha_min))

    #filename_entropy = str.split(filename_kdata, ".npy")[0] + "_entropy.npy"
    #np.save(filename_entropy,np.array(entropy_all))

    ###########################
    #X, Y = np.meshgrid(x, y)


    #fig = plt.figure(figsize=(15, 15))
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X, Y, np.array(entropy_all), rstride=1, cstride=1, cmap='hot', linewidth=0,
    #                       antialiased=False)

    #filename_entropy_image = str.split(filename_kdata, ".npy")[0] + "_entropy.jpg"
    #plt.savefig(filename_entropy_image)
    ##################################""

    alpha = np.array([1, 0, 0])
    dr = np.expand_dims(alpha, axis=(0, 1)) * np.expand_dims(
                    displacements_extrapolated.reshape(nb_slices, nb_segments).T, axis=(2))
    modif_final = np.exp(
                    1j * np.sum((radial_traj.get_traj().reshape(nb_segments, -1, npoint, 3) * np.expand_dims(dr, axis=2)),
                                axis=-1))

    np.save(filename_kdata_modif, kdata_all_channels_all_slices*modif_final)


    # gif=[]
    # sl=int(volumes_all.shape[1]/2)
    # volume_for_gif = np.abs(volumes_all[:,sl,:,:])
    # for i in range(volume_for_gif.shape[0]):
    #     img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    #     img=img.convert("P")
    #     gif.append(img)
    #
    # filename_gif = str.split(filename_volume,".npy") [0]+".gif"
    # gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return



@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("filename_df_groups_global", str, default=None, description="Number of spokes per cycle per acquisition")
@set_parameter("filename_categories_global", str, default=None, description="Cycle for each spoke for each acquisition")
@set_parameter("nb_gating_spokes", int, default=50, description="Number of gating spokes per repetition")
@set_parameter("files_config", type=Config, default=None, description="Kdata filenames to use for aggregation")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")


def aggregate_kdata(filename_kdata,filename_df_groups_global,filename_categories_global,files_config,nb_gating_spokes,undersampling_factor):
    # b1_all_slices=b1_full

    print("Loading Files...")
    kdata_all_channels_all_slices = np.load(filename_kdata)

    folder = "/".join(str.split(filename_kdata, "/")[:-1])
    base_folder="/".join(str.split(filename_kdata, "/")[:-2])

    bin_width=int(str.split(filename_categories_global, "bw")[-1][:-4])



    data_shape = kdata_all_channels_all_slices.shape
    nb_channels=data_shape[0]
    nb_segments = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2] * undersampling_factor
    nb_part = nb_slices

    print(data_shape)
    print(folder)

    files = files_config["files"]
    print(files)

    filename_kdata_final = base_folder + str.split(files[0], "_1.dat")[0] + "_bw{}_aggregated_kdata.npy".format(
        bin_width)

    categories_global = np.load(filename_categories_global)
    df_groups_global = pd.read_pickle(filename_df_groups_global)

    idx_cat = df_groups_global.displacement.idxmax()

    kdata_final = np.zeros(data_shape, dtype=kdata_all_channels_all_slices.dtype)
    del kdata_all_channels_all_slices
    count = np.zeros((nb_segments, nb_slices))

    for i, localfile in tqdm(enumerate(files)):

        filename = base_folder + localfile

        retained_nav_spokes = (categories_global[i] == idx_cat)

        retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
        spoke_groups = np.argmin(np.abs(
            np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                              nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                      -1)),
                                 axis=-1)

        if not (nb_segments == nb_gating_spokes):
            spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
            spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
                nb_segments / nb_gating_spokes / 2) + 1:] - 1
            spoke_groups = spoke_groups.flatten()

        included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])

        included_spokes[::int(nb_segments / nb_gating_spokes)] = False
        included_spokes = included_spokes.reshape(nb_slices, nb_segments)
        included_spokes = included_spokes.T

        filename_kdata = str.split(filename, ".dat")[0] + "_kdata{}.npy".format("")

        print("Loading kdata for file {}".format(localfile))
        kdata_all_channels_all_slices = np.load(filename_kdata)
        #print(included_spokes.shape)
        kdata_all_channels_all_slices = kdata_all_channels_all_slices.reshape(nb_channels, nb_segments, nb_slices,
                                                                              npoint)
        print("Aggregating kdata for file {}".format(localfile))
        for ch in tqdm(range(nb_channels)):
            kdata_final[ch] += (np.expand_dims(1 * included_spokes, axis=-1)) * kdata_all_channels_all_slices[ch]
        count += (1 * included_spokes)

    count[count == 0] = 1

    print("Normalizing Final Kdata")
    for ch in tqdm(range(nb_channels)):
        kdata_final[ch] /= np.expand_dims(count, axis=(-1))

    np.save(filename_kdata_final, kdata_final)


    # gif=[]
    # sl=int(volumes_all.shape[1]/2)
    # volume_for_gif = np.abs(volumes_all[:,sl,:,:])
    # for i in range(volume_for_gif.shape[0]):
    #     img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    #     img=img.convert("P")
    #     gif.append(img)
    #
    # filename_gif = str.split(filename_volume,".npy") [0]+".gif"
    # gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("dens_adj", bool, default=False, description="Memory usage")
@set_parameter("threshold", float, default=None, description="Threshold for mask")
@set_parameter("suffix",str,default="")
def build_mask(filename_kdata,filename_traj,sampling_mode,undersampling_factor,dens_adj,threshold,suffix):
    kdata_all_channels_all_slices = np.load(filename_kdata)

    filename_b1 = ("_b1" + suffix).join(str.split(filename_kdata, "_kdata"))

    b1_all_slices=np.load(filename_b1)

    filename_mask =("_mask"+suffix).join(str.split(filename_kdata, "_kdata"))
    print(filename_mask)

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
    nb_slices = data_shape[2]*undersampling_factor
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts = np.load(filename_traj)
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    selected_spokes=np.r_[10:400] 
    selected_spokes=None 
    mask = build_mask_single_image_multichannel(kdata_all_channels_all_slices, radial_traj, image_size,
                                                b1=b1_all_slices, density_adj=dens_adj, threshold_factor=threshold,
                                                normalize_kdata=False, light_memory_usage=True,selected_spokes=selected_spokes,normalize_volumes=True)


    np.save(filename_mask, mask)

    gif = []

    for i in range(mask.shape[0]):
        img = Image.fromarray(np.uint8(mask[i] / np.max(mask[i]) * 255), 'L')
        img = img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_mask, ".npy")[0] + ".gif"
    gif[0].save(filename_gif, save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return


@machine
@set_parameter("filename_volume", str, default=None, description="MRF time series")
@set_parameter("filename_mask", str, default=None, description="Mask")
@set_parameter("dictfile", str, default=None, description="Dictionary file")
@set_parameter("optimizer_config",type=Config,default=DEFAULT_OPT_CONFIG,description="Optimizer parameters")
@set_parameter("slices",type=Config,default=None,description="Slices to consider for pattern matching")
def build_maps(filename_volume,filename_mask,dictfile,optimizer_config,slices):
    file_map = filename_volume.split(".npy")[0] + "_MRF_map.pkl"
    volumes_all = np.load(filename_volume)
    mask=np.load(filename_mask)

    opt_type = optimizer_config["type"]

    if slices is not None:
        sl = slices["slices"]
        if not(len(sl)==0):
            mask_slice = np.zeros(mask.shape, dtype=mask.dtype)
            mask_slice[sl] = 1
            mask *= mask_slice
            sl=[str(s) for s in sl]
            file_map = filename_volume.split(".npy")[0] + "_sl{}_{}_MRF_map.pkl".format("_".join(sl),opt_type)


    if (slices is not None) and ("spacing" in slices) and not(len(slices["spacing"])==0):
        spacing=slices["spacing"]
    else:
        spacing = [5, 1, 1]

    dz = spacing[0]
    dx = spacing[1]
    dy = spacing[2]

    threshold_pca=optimizer_config["pca"]
    split=optimizer_config["split"]
    useGPU = optimizer_config["useGPU"]

    if opt_type=="CF":
        optimizer = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=True,
                                     threshold_pca=threshold_pca, log=False, useGPU_dictsearch=useGPU,
                                     useGPU_simulation=False, gen_mode="other")
        all_maps = optimizer.search_patterns_test(dictfile, volumes_all)

    elif opt_type=="Matrix":
        optimizer = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=True,
                                 threshold_pca=threshold_pca, log=False, useGPU_dictsearch=useGPU,
                                 useGPU_simulation=False, gen_mode="other")
        all_maps = optimizer.search_patterns_matrix(dictfile, volumes_all)

    elif opt_type=="Brute":
        optimizer=BruteDictSearch(FF_list=np.arange(0,1.01,0.05),mask=mask,split=split, pca=True,
                                 threshold_pca=threshold_pca, log=False, useGPU_dictsearch=useGPU
                                 )
        all_maps = optimizer.search_patterns(dictfile, volumes_all)

    file = open(file_map, "wb")
    pickle.dump(all_maps, file)
    file.close()

    for iter in list(all_maps.keys()):

        map_rebuilt = all_maps[iter][0]
        mask = all_maps[iter][1]

        keys_simu = list(map_rebuilt.keys())
        values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
        map_for_sim = dict(zip(keys_simu, values_simu))

        for key in ["ff", "wT1", "df", "attB1"]:
            file_mha = "/".join(["/".join(str.split(file_map, "/")[:-1]),
                                 "_".join(str.split(str.split(file_map, "/")[-1], ".")[:-1])]) + "_it{}_{}.mha".format(
                iter, key)
            io.write(file_mha, map_for_sim[key], tags={"spacing": [dz, dx, dy]})

    return



@machine

@set_parameter("files_config",type=Config,default=None,description="Files to consider for aggregate kdata reconstruction")
@set_parameter("disp_config",type=Config,default=None,description="Parameters for movement identification")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")

def build_data_nacq(files_config,disp_config,undersampling_factor):

    base_folder = "./data/InVivo/3D"
    files=files_config["files"]
    shifts=list(range(disp_config["shifts"][0],disp_config["shifts"][1]))
    ch=disp_config["channel"]
    bin_width=disp_config["bin_width"]

    folder = base_folder + "/".join(str.split(files[0], "/")[:-1])


    filename_categories_global = folder + "/categories_global_bw{}.npy".format(bin_width)
    filename_df_groups_global = folder + "/df_groups_global_bw{}.pkl".format(bin_width)

    categories_global = []
    df_groups_global = pd.DataFrame()

    for localfile in files:

        filename = base_folder + localfile
        filename_nav_save = str.split(filename, ".dat")[0] + "_nav.npy"
        folder = "/".join(str.split(filename, "/")[:-1])
        filename_kdata = str.split(filename, ".dat")[0] + "_kdata{}.npy".format("")
        filename_disp_image=str.split(filename, ".dat")[0] + "_nav_image.jpg".format("")


        dico_seqParams = build_dico_seqParams(filename, folder)

        use_navigator_dll = dico_seqParams["use_navigator_dll"]

        if use_navigator_dll:
            meas_sampling_mode = dico_seqParams["alFree"][14]
            nb_gating_spokes = dico_seqParams["alFree"][6]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]
            nb_gating_spokes = 0


        nb_segments = dico_seqParams["alFree"][4]

        del dico_seqParams

        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

        data, data_for_nav = build_data(filename, folder, nb_segments, nb_gating_spokes)

        data_shape = data.shape

        nb_allspokes = data_shape[-3]
        npoint = data_shape[-1]
        nb_slices = data_shape[-2]

        if str.split(filename_kdata, "/")[-1] in os.listdir(folder):
            del data

        if str.split(filename_kdata, "/")[-1] not in os.listdir(folder):
            # Density adjustment all slices

            density = np.abs(np.linspace(-1, 1, npoint))
            density = np.expand_dims(density, tuple(range(data.ndim - 1)))

            print("Performing Density Adjustment....")
            data *= density
            np.save(filename_kdata, data)
            del data



        print("Calculating Coil Sensitivity....")

        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        nb_segments = radial_traj.get_traj().shape[0]

        if nb_gating_spokes > 0:
            print("Processing Nav Data...")
            data_for_nav = np.load(filename_nav_save)

            nb_allspokes = nb_segments
            nb_slices = data_for_nav.shape[1]
            nb_channels = data_for_nav.shape[0]
            npoint_nav = data_for_nav.shape[-1]

            all_timesteps = np.arange(nb_allspokes)
            nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

            nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                                   applied_timesteps=list(nav_timesteps))

            nav_image_size = (int(npoint_nav / 2),)

            print("Building nav image for channel {}...".format(ch))
            # b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
            # b1_nav_mean = np.mean(b1_nav, axis=(1, 2))

            image_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[ch], axis=0), nav_traj, nav_image_size)

            plt.figure()
            plt.plot(np.abs(image_nav_ch.reshape(-1, int(npoint / 2)))[5*nb_gating_spokes, :])
            plt.savefig(filename_disp_image)

            print("Estimating Movement...")

            bottom = -shifts[0]
            top = nav_image_size[0]-shifts[-1]


            displacements = calculate_displacement(image_nav_ch, bottom, top, shifts, 0.001)

            displacement_for_binning = displacements

            max_bin = np.max(displacement_for_binning)
            min_bin = shifts[0]

            bins = np.arange(min_bin, max_bin + bin_width, bin_width)
            # print(bins)
            categories = np.digitize(displacement_for_binning, bins)
            df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T,
                                  columns=["displacement", "cat"])
            df_groups = df_cat.groupby("cat").count()
            curr_max = df_groups.displacement.max()

            if df_groups_global.empty:
                df_groups_global = df_groups
            else:
                df_groups_global += df_groups

            categories_global.append(categories)

    #################################################################################################################################"

    categories_global = np.array(categories_global)

    np.save(filename_categories_global, categories_global)
    df_groups_global.to_pickle(filename_df_groups_global)



    return


toolbox = Toolbox("script_recoInVivo_3D_machines", description="Reading Siemens 3D MRF data and performing image series reconstruction")
toolbox.add_program("build_kdata", build_kdata)
toolbox.add_program("build_coil_sensi", build_coil_sensi)
toolbox.add_program("build_volumes", build_volumes)
toolbox.add_program("build_mask", build_mask)
toolbox.add_program("build_data_for_grappa_simulation", build_data_for_grappa_simulation)
toolbox.add_program("calib_and_estimate_kdata_grappa", calib_and_estimate_kdata_grappa)
toolbox.add_program("build_data_autofocus", build_data_autofocus)
toolbox.add_program("aggregate_kdata", aggregate_kdata)
toolbox.add_program("build_maps", build_maps)
toolbox.add_program("build_data_nacq", build_data_nacq)

if __name__ == "__main__":
    toolbox.cli()


#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch,BruteDictSearch
from utils_mrf import weights_aggregate_center_part
from utils_reco import *
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
from utils_simu import *


try :
    import cupy as cp
except:
    pass


# third party imports
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
from copy import copy
from utils_reco import unpad,format_input_voxelmorph,format_input_voxelmorph_3D
from skimage.transform import resize
from tqdm import tqdm

#print(tf.config.experimental.list_physical_devices("GPU"))

# local imports
import voxelmorph as vxm
import neurite as ne
from mutools import io
import matplotlib.pyplot as plt

import SimpleITK as sitk
import wandb
from wandb.keras import WandbMetricsLogger,WandbModelCheckpoint
import torchio as tio
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from keras import backend
from utils_reco import apply_deformation_to_complex_volume

from skimage.transform import resize


DEFAULT_TRAIN_CONFIG="config_train_voxelmorph.json"
DEFAULT_TRAIN_CONFIG_3D="config_train_voxelmorph_3D.json"

def build_volumes_iterative_allbins_registered_allindex(volumes, b1_all_slices_2Dplus1_pca, all_weights, mu, mu_TV,mu_bins,weights_TV,
                                                        niter, gamma, dens_adj, nb_rep_center_part,
                                                        deformation_map, beta,resolution):

    if not(beta==0)and(deformation_map is None):
        raise ValueError("Deformation map should be supplied when beta is not zero")


    weights_TV /= np.sum(weights_TV)
    print("Loading Volumes")
    # To fit the input expected by the undersampling function with L0=1 in our case

    nbins,nb_slices,n,n=volumes.shape
    nb_allspokes = all_weights.shape[2]

    volumes = volumes.astype("complex64")
    nb_channels=b1_all_slices_2Dplus1_pca.shape[0]

    if resolution is not None:
        print("Resampling volume at resolution {}".format(resolution))
        volumes_reshaped = np.zeros((nbins, nb_slices, resolution, resolution), dtype=volumes.dtype)
        b1_all_slices_2Dplus1_pca_reshaped=np.zeros((nb_channels, nb_slices, resolution, resolution), dtype=b1_all_slices_2Dplus1_pca.dtype)

        for gr in range(nbins):
            for sl in range(nb_slices):
                volumes_reshaped[gr,sl]=resize(volumes[gr,sl].real,(resolution,resolution))+1j*resize(volumes[gr,sl].imag,(resolution,resolution))
        volumes=volumes_reshaped

        for ch in range(nbins):
            for sl in range(nb_slices):
                b1_all_slices_2Dplus1_pca_reshaped[ch,sl]=resize(b1_all_slices_2Dplus1_pca[ch,sl].real,(resolution,resolution))+1j*resize(b1_all_slices_2Dplus1_pca[ch,sl].imag,(resolution,resolution))

        b1_all_slices_2Dplus1_pca=b1_all_slices_2Dplus1_pca_reshaped

        n=resolution

        if deformation_map is not None:
            deformation_map=resample_deformation(deformation_map,resolution)

    npoint = 2 * n

    radial_traj_2D = Radial(total_nspokes=nb_allspokes, npoint=npoint)
    volumes = np.expand_dims(volumes, axis=1)


    if nb_rep_center_part > 1:
        all_weights = weights_aggregate_center_part(all_weights, nb_rep_center_part)

    print("Volumes shape {}".format(volumes.shape))
    print("Weights shape {}".format(all_weights.shape))



    if beta==0:
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        deformation_map = np.stack([X, Y], axis=0)
        deformation_map = np.tile(deformation_map,reps=(2,nbins,nb_slices,1,1))

    deformation_map_allindex = []
    inv_deformation_map_allindex = []
    volumes_registered_allindex = []

    print("Extraction deformation map for all bin reference")
    for index_ref in range(nbins):
        deformation_map_allindex.append(change_deformation_map_ref(deformation_map, index_ref))

    print("Building inverse deformation map for all bin reference")
    for index_ref in range(nbins):
        deformation_map = deformation_map_allindex[index_ref]
        print("Calculating inverse deformation map")
        inv_deformation_map = np.zeros_like(deformation_map)
        for gr in tqdm(range(nbins)):
            if beta==0:
                inv_deformation_map[:, gr] = deformation_map[:, gr]
            else:
                inv_deformation_map[:, gr] = calculate_inverse_deformation_map(deformation_map[:, gr])
        inv_deformation_map_allindex.append(inv_deformation_map)

        volumes_registered = np.zeros(volumes.shape[2:], dtype=volumes.dtype)

        print("Registering initial volumes")
        for gr in range(nbins):
            if (beta == 0) and not (gr == index_ref):
                continue

            if beta is None:
                volumes_registered += apply_deformation_to_complex_volume(volumes[gr].squeeze(), deformation_map[:, gr])

            else:
                if gr == index_ref:
                    volumes_registered += (1-beta+beta/nbins) * apply_deformation_to_complex_volume(volumes[gr].squeeze(),
                                                                                     deformation_map[:, gr])
                else:
                    volumes_registered += beta/nbins * apply_deformation_to_complex_volume(volumes[gr].squeeze(),
                                                                                           deformation_map[:, gr])
        volumes_registered_allindex.append(volumes_registered)

    deformation_map_allindex = np.array(deformation_map_allindex)
    inv_deformation_map_allindex = np.array(inv_deformation_map_allindex)

    volumes_registered_allindex = np.array(volumes_registered_allindex)
    volumes0_allindex = copy(volumes_registered_allindex)
    volumes_registered_allindex = mu * volumes0_allindex

    for i in tqdm(range(niter)):
        print("#############################  Correcting volumes for iteration {} #################################".format(i))
        all_grad_norm = 0
        for index_ref in range(nbins):

            for gr in tqdm(range(nbins)):

                if (beta==0)and not(gr==index_ref):
                    continue

                volumesi = apply_deformation_to_complex_volume(volumes_registered_allindex[index_ref],
                                                               inv_deformation_map_allindex[index_ref, :, gr])
                volumesi = undersampling_operator_singular_new(np.expand_dims(volumesi, axis=0), radial_traj_2D,
                                                               b1_all_slices_2Dplus1_pca, weights=all_weights[gr],
                                                               density_adj=dens_adj,
                                                               nb_rep_center_part=nb_rep_center_part)
                volumesi = apply_deformation_to_complex_volume(volumesi.squeeze(),
                                                               deformation_map_allindex[index_ref, :, gr])

                if beta is not None:
                    if gr == index_ref:
                        volumesi *= (1-beta+beta/nbins)
                    else:
                        volumesi *= beta/nbins

                if gr == 0:
                    final_volumesi = volumesi
                else:
                    final_volumesi += volumesi

            grad = final_volumesi - volumes0_allindex[index_ref]
            volumes_registered_allindex[index_ref] = volumes_registered_allindex[index_ref] - mu * grad

            if (mu_TV is not None) and (not (mu_TV == 0)):
                print("Applying TV regularization")

                grad_norm = np.linalg.norm(grad)
                all_grad_norm += grad_norm ** 2
                print("grad norm {}".format(grad_norm))
                del grad
                grad_TV = np.zeros_like(volumes_registered_allindex[index_ref])

                for ind_w, w in (enumerate(weights_TV)):
                    if w > 0:
                        grad_TV += (w * grad_J_TV(volumes_registered_allindex[index_ref], ind_w, is_weighted=False,
                                                  shift=0))

                        # grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                grad_TV_norm = np.linalg.norm(grad_TV)
                # signals = matched_signals + mu * grad

                print("grad_TV_norm {}".format(grad_TV_norm))

                volumes_registered_allindex[index_ref] -= mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                del grad_TV
                del grad_TV_norm

        if (mu_bins is not None) and (not (mu_bins == 0)):
            grad_TV_bins = grad_J_TV(volumes_registered_allindex, 0, is_weighted=False, shift=0)
            grad_TV_bins_norm = np.linalg.norm(grad_TV_bins)

            volumes_registered_allindex -= mu * mu_bins * grad_TV_bins / grad_TV_bins_norm * all_grad_norm
            print("grad_TV_bins norm {}".format(grad_TV_bins_norm))
            del grad_TV_bins


    volumes_registered_allindex = np.squeeze(volumes_registered_allindex)
    if gamma is not None:
        volumes_registered_allindex = gamma_transform(volumes_registered_allindex)

    # volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights)
    # np.save(filename_volumes,volumes_allbins)

    return volumes_registered_allindex


def train_voxelmorph(filename_volumes, config_train, suffix, init_weights, resolution):
    all_volumes = np.abs(np.load(filename_volumes))
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes = all_volumes.astype("float32")

    nb_gr, nb_slices, npoint, npoint = all_volumes.shape

    if resolution is not None:
        all_volumes = resize(all_volumes, (nb_gr, nb_slices, resolution, resolution))

    if resolution is None:
        file_model = filename_volumes.split(".npy")[0] + "_vxm_model_weights{}.h5".format(suffix)
        file_checkpoint = "/".join(filename_volumes.split("/")[:-1]) + "/model_checkpoint.h5"
    else:
        file_model = filename_volumes.split(".npy")[0] + "_vxm_model_weights_res{}{}.h5".format(resolution, suffix)
        file_checkpoint = "/".join(filename_volumes.split("/")[:-1]) + "/model_checkpoint_res{}.h5".format(resolution)
    print(file_checkpoint)

    config_train["file_volume"] = filename_volumes.split("/")[-1]
    config_train["resolution"] = resolution
    file_project = file_model.split("/")[-1]
    file_project = file_project[:22]
    run = wandb.init(
        project=file_project,
        config=config_train
    )

    # pad_amount=config_train["padding"]
    loss = config_train["loss"]
    decay = config_train["lr_decay"]
    # pad_amount=tuple(tuple(l) for l in pad_amount)

    # Finding the power of 2 "closest" and longer than  x dimension
    n = all_volumes.shape[-1]
    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad = int(pad_1 / 2)
    else:
        pad = int(pad_2 / 2)

    if n % 2 == 0:
        pad = 0

    pad_amount = ((0, 0), (pad, pad), (pad, pad))
    print(pad_amount)
    nb_features = config_train["nb_features"]
    # configure unet features
    # nb_features = [
    #    [256, 32, 32, 32],         # encoder features
    #    [32, 32, 32, 32, 32, 16]  # decoder features
    # ]

    # nb_features = [
    #    [32, 64, 128, 128],         # encoder features
    #    [128, 128, 64, 64, 32, 16]  # decoder features
    # ]

    optimizer = config_train["optimizer"]  # "Adam"
    lambda_param = config_train["lambda"]  # 0.05
    nb_epochs = config_train["nb_epochs"]  # 200
    batch_size = config_train["batch_size"]  # 16
    lr = config_train["lr"]

    x_train_fixed, x_train_moving = format_input_voxelmorph(all_volumes, pad_amount)

    # configure unet input shape (concatenation of moving and fixed images)
    inshape = x_train_fixed.shape[1:]

    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # voxelmorph has a variety of custom loss classes

    if loss == "MSE":
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss == "NCC":
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
    elif loss == "MI":
        losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    else:
        raise ValueError("Loss should be either MSE or Mutual Information (MI)")

    loss_weights = [1, lambda_param]

    print("Compiling Model")
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    if lr is not None:
        backend.set_value(vxm_model.optimizer.learning_rate, lr)

    train_generator = vxm_data_generator(x_train_fixed, x_train_moving, batch_size=batch_size)

    nb_examples = x_train_fixed.shape[0]

    steps_per_epoch = int(nb_examples / batch_size) + 1

    curr_scheduler = lambda epoch, lr: scheduler(epoch, lr, decay)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)

    callback_checkpoint = WandbModelCheckpoint(filepath=file_checkpoint, save_best_only=True, save_weights_only=True,
                                               monitor="vxm_dense_transformer_loss")

    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,
                                   callbacks=[Schedulecallback, WandbMetricsLogger(), callback_checkpoint])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0] + "_loss.jpg")


filename_volumes_allbins="data/InVivo/3D/patient.010.v3/meas_MID00120_FID65014_raFin_3D_tra_1x1x5mm_FULL_new_mrf_volume_singular_l1_allbins.npy"
filename_weights="data/InVivo/3D/patient.010.v3/meas_MID00120_FID65014_raFin_3D_tra_1x1x5mm_FULL_new_mrf_weights.npy"
filename_b1="data/InVivo/3D/patient.010.v3/meas_MID00120_FID65014_raFin_3D_tra_1x1x5mm_FULL_new_mrf_b12Dplus1_16.npy"



volumes_allbins=np.load(filename_volumes_allbins)
all_weights=np.load(filename_weights)
b1=np.load(filename_b1)

resolution=64
mu=1
mu_TV=1
gamma=None
deformation_map=None
beta=0
niter=2
dens_adj=True
nb_rep_center_part=10
mu_bins=2
weights_TV = np.array([1.0, 0.5,0.5])

volumes_allbins_registered=build_volumes_iterative_allbins_registered_allindex(volumes_allbins,b1,all_weights,mu,mu_TV,mu_bins,weights_TV,niter,gamma,dens_adj,nb_rep_center_part,deformation_map,beta,resolution)

sl=46
animate_images(volumes_allbins_registered[:,sl,:,:])

animate_images(gamma_transform(np.abs(volumes_allbins_registered[:,sl,:,:]),gamma=0.5))
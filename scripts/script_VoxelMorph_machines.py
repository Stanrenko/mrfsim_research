# imports
# third party imports
import numpy as np
import tensorflow as tf
import json
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
from copy import copy
from mrfsim.utils_reco import unpad,format_input_voxelmorph,format_input_voxelmorph_3D,plot_deformation_map,apply_deformation_to_complex_volume,pad_to_multiple
from skimage.transform import resize

print(tf.config.experimental.list_physical_devices("GPU"))

# local imports
import voxelmorph as vxm
import neurite as ne
from mrfsim import io

import matplotlib.pyplot as plt
try:
    import SimpleITK as sitk
except:
    pass
import wandb
from wandb.integration.keras import WandbMetricsLogger,WandbModelCheckpoint
try:
    import torchio as tio
    import torch
    import torchvision.transforms as T
except:
    pass

from sklearn.model_selection import train_test_split
from keras import backend

import machines as ma
from machines import Toolbox
import subprocess

from scipy.ndimage import zoom,map_coordinates

def get_total_memory_mb():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"]
    )
    return int(result.decode("utf-8").strip().split("\n")[0])

total_mem = get_total_memory_mb()
fraction = 0.8

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    try:
        mem_limit = int(total_mem * fraction)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)]
        )
        print(f"GPU memory limited to {mem_limit} MB")
    except RuntimeError as e:
        print("Error:", e)


DEFAULT_TRAIN_CONFIG="../config/config_train_voxelmorph.json"
DEFAULT_TRAIN_CONFIG_3D="../config/config_train_voxelmorph_3D.json"
# You should most often have this import together with all other imports at the top,
# but we include here here explicitly to show where data comes from



def upsample_flow_3D(flow, target_shape,order=1):
    assert flow.ndim == 4 and flow.shape[-1] == 3

    if flow.shape[:3] == target_shape:
        return flow

    scale = [t/s for t, s in zip(target_shape, flow.shape[:3])]

    flow_up = np.stack([
        zoom(flow[..., i], scale, order=order) for i in range(3)
    ], axis=-1)

    # scale displacement
    for i in range(3):
        flow_up[..., i] *= scale[i]

    return flow_up


def upsample_volume(vol, target_shape,order=1):
    factors = [t/s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=order)

def downsample_3D(vol, factor):
    if factor is None or factor == 1:
        return vol
    return zoom(vol, (1/factor, 1/factor, 1/factor), order=1)


def warp_flow_3D(flow, mapz, mapx, mapy):
    """
    Warp flow using current deformation map φ
    """
    coords = [mapz, mapx, mapy]

    warped = np.zeros_like(flow)
    for i in range(3):
        warped[..., i] = map_coordinates(
            flow[..., i],
            coords,
            order=1,
            mode='nearest'
        )
    return warped

def vxm_data_generator(x_data_fixed, x_data_moving, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data_fixed.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data_moving.shape[0], size=batch_size)
        moving_images = x_data_moving[idx1, ..., np.newaxis]
        # idx2 = np.random.randint(0, x_data_fixed.shape[0], size=batch_size)
        fixed_images = x_data_fixed[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

def vxm_data_generator_3D(x_data_fixed, x_data_moving, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data_fixed.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data_moving.shape[0], size=batch_size)
        moving_images = x_data_moving[idx1, ...,np.newaxis]
        # idx2 = np.random.randint(0, x_data_fixed.shape[0], size=batch_size)
        fixed_images = x_data_fixed[idx1, ...,np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

def vxm_data_generator_torchio(x_data_fixed, x_data_moving, transform,batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data_fixed.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data_moving.shape[0], size=batch_size)
        moving_images=[]
        fixed_images=[]
        for j in idx1:
            #print(j)
            moving_image = x_data_moving[np.newaxis,j, ..., np.newaxis]
            # idx2 = np.random.randint(0, x_data_fixed.shape[0], size=batch_size)
            fixed_image = x_data_fixed[np.newaxis,j, ..., np.newaxis]
            subject=tio.Subject(moving=tio.ScalarImage(tensor=moving_image),fixed=tio.ScalarImage(tensor=fixed_image))
            subject_transf=transform(subject)
            moving_images.append(subject_transf.moving.data.squeeze()[...,np.newaxis].numpy())
            fixed_images.append(subject_transf.fixed.data.squeeze()[..., np.newaxis].numpy())

        inputs = [np.array(moving_images), np.array(fixed_images)]


        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [np.array(fixed_images), zero_phi]

        yield (inputs, outputs)


def vxm_data_generator_torchvision(x_data_fixed, x_data_moving, transform_list,batch_size=32,probabilities=None):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data_fixed.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    transform_minmax = T.Lambda(lambda x: x - x.min() / (x.max() - x.min()))
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data_moving.shape[0], size=batch_size)
        moving_images=[]
        fixed_images=[]



        for j in idx1:
            #print(j)
            moving_image = torch.from_numpy(x_data_moving[np.newaxis,j])
            # idx2 = np.random.randint(0, x_data_fixed.shape[0], size=batch_size)
            fixed_image = torch.from_numpy(x_data_fixed[np.newaxis,j])
            t=np.random.choice(transform_list,p=probabilities)
            #print("Transform for j {}: {}".format(j,t))

            transf= T.Compose([t, T.Lambda(lambda x : (x-x.min())/(x.max()-x.min()))])
            moving_images.append(transf(moving_image).numpy().squeeze()[...,np.newaxis])
            fixed_images.append(transf(fixed_image).numpy().squeeze()[..., np.newaxis])

        inputs = [np.array(moving_images), np.array(fixed_images)]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [np.array(fixed_images), zero_phi]

        yield (inputs, outputs)

@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG,description="Config training")
@ma.parameter("kept_bins",str,default=None,description="Bins to keep for training")
@ma.parameter("suffix",str,default="",description="suffix")
@ma.parameter("resolution",int,default=None,description="image resolution")
@ma.parameter("nepochs",int,default=None,description="Number of epochs (overwrites config)")
@ma.parameter("lr",float,default=None,description="Learning rate (overwrites config)")
@ma.parameter("init_weights",str,default=None,description="Weights initialization from .h5 file")
@ma.parameter("us",int,default=None,description="Select one every us slice")
@ma.parameter("excluded",int,default=5,description="Excluded slices on both extremities")
@ma.parameter("axis",int,default=None,description="Change registration axis")
def train_voxelmorph(filename_volumes,file_config_train,suffix,init_weights,resolution,nepochs,lr,kept_bins,axis,us,excluded):

    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes = np.abs(np.load(filename_volumes))
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")


    if kept_bins is not None:
        kept_bins_list=np.array(str.split(kept_bins,",")).astype(int)
        print(kept_bins_list)
        all_volumes=all_volumes[kept_bins_list]


    nb_gr,nb_slices,npoint,npoint=all_volumes.shape


    if axis is not None:
        all_volumes=np.moveaxis(all_volumes,axis+1,1)
        # all_volumes=resize(all_volumes,(nb_gr,npoint,npoint,npoint))
        all_volumes=all_volumes[:,::int(npoint/nb_slices)]


    if us is not None:
        # all_volumes=resize(all_volumes,(nb_gr,npoint,npoint,npoint))
        all_volumes=all_volumes[:,::us]

    print(all_volumes.shape)

    if nepochs is not None:
        config_train["nb_epochs"]=nepochs

    if lr is not None:
        config_train["lr"]=lr

    if resolution is not None:
        all_volumes=resize(all_volumes,(nb_gr,nb_slices,resolution,resolution))

    if resolution is None:
        file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights{}.h5".format(suffix)
        file_checkpoint="/".join(filename_volumes.split("/")[:-1])+"/model_checkpoint.h5"
    else:
        file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights_res{}{}.h5".format(resolution,suffix)
        file_checkpoint="/".join(filename_volumes.split("/")[:-1])+"/model_checkpoint_res{}.h5".format(resolution)
    print(file_checkpoint)
    # run=wandb.init(
    #     project=str.replace(file_model.split("/")[-1].split("_FULL")[0],"raFin_3D_","")+file_model.split("/")[-1].split("_FULL")[1],
    #     config=config_train
    # )

    run=wandb.init(
        project="project_test",
        config=config_train
    )

    #pad_amount=config_train["padding"]
    loss = config_train["loss"]
    decay=config_train["lr_decay"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)

    #Finding the power of 2 "closest" and longer than  x dimension
    n = np.maximum(all_volumes.shape[-1],all_volumes.shape[-2])
    pad_1=2**(int(np.log2(n))+1)-n
    pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2<0:
        pad=int(pad_1/2)
    else:
        pad = int(pad_2 / 2)

    # if n%2==0:
    #     pad=0
    pad_x=int((2*pad+n-all_volumes.shape[-2])/2)
    pad_y=int((2*pad+n-all_volumes.shape[-1])/2)

    pad_amount = ((0,0),(pad_x,pad_x), (pad_y,pad_y))
    print(pad_amount)
    nb_features=config_train["nb_features"]
    # configure unet features
    #nb_features = [
    #    [256, 32, 32, 32],         # encoder features
    #    [32, 32, 32, 32, 32, 16]  # decoder features
    #]

    #nb_features = [
    #    [32, 64, 128, 128],         # encoder features
    #    [128, 128, 64, 64, 32, 16]  # decoder features
    #]

    optimizer=config_train["optimizer"] #"Adam"
    lambda_param = config_train["lambda"] # 0.05
    nb_epochs = config_train["nb_epochs"] # 200
    batch_size=config_train["batch_size"] # 16
    lr=config_train["lr"]

    x_train_fixed,x_train_moving=format_input_voxelmorph(all_volumes,pad_amount,sl_down=excluded,sl_top=-excluded)
    

    # configure unet input shape (concatenation of moving and fixed images)
    inshape = x_train_fixed.shape[1:]

    print(inshape)

    print(nb_features)

    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    print("Model defined")
    # voxelmorph has a variety of custom loss classes

    if loss=="MSE":
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss == "NCC":
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
    elif loss =="MI":
        losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    else:
        raise ValueError("Loss should be either MSE or Mutual Information (MI)")

    loss_weights = [1, lambda_param]

    print("Compiling Model")
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    if lr is not None:
        backend.set_value(vxm_model.optimizer.learning_rate,lr)

    train_generator = vxm_data_generator(x_train_fixed,x_train_moving,batch_size=batch_size)


    nb_examples=x_train_fixed.shape[0]

    
    steps_per_epoch = int(nb_examples/batch_size)+1
    
    if "min_lr" in config_train:
        min_lr=config_train["min_lr"]
    else:
        min_lr=0.0002

    if "decay_start" in config_train:
        decay_start=config_train["decay_start"]
    else:
        decay_start=20

    curr_scheduler=lambda epoch,lr: scheduler(epoch,lr,decay,min_lr,decay_start)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)

    callback_checkpoint=WandbModelCheckpoint(filepath=file_checkpoint,save_best_only=True,save_weights_only=True,monitor="vxm_dense_transformer_loss")


    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,callbacks=[Schedulecallback,WandbMetricsLogger(),callback_checkpoint])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0]+"_loss.jpg")

    return


# @ma.machine()
# @ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
# @ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG_3D,description="Config training")
# @ma.parameter("suffix",str,default="",description="suffix")
# @ma.parameter("init_weights",str,default=None,description="Weights initialization from .h5 file")
# @ma.parameter("kept_bins",str,default=None,description="Bins to keep for training")
# @ma.parameter("nepochs",int,default=None,description="Number of epochs (overwrites config)")
# @ma.parameter("excluded",int,default=0,description="Excluded slices on both extremities")
# @ma.parameter("downsample",int,default=1,description="Downsampling")
# def train_voxelmorph_3D(filename_volumes,file_config_train,suffix,init_weights,kept_bins,nepochs,excluded,downsample):

#     with open(file_config_train,"r") as f:
#         config_train=json.load(f)

#     all_volumes = np.abs(np.load(filename_volumes))
#     print("Volumes shape {}".format(all_volumes.shape))
#     all_volumes=all_volumes.astype("float32")

#     if kept_bins is not None:
#         kept_bins_list=np.array(str.split(kept_bins,",")).astype(int)
#         print(kept_bins_list)
#         all_volumes=all_volumes[kept_bins_list]
    
#     nb_gr,nb_slices,npoint,npoint=all_volumes.shape

#     if nepochs is not None:
#         config_train["nb_epochs"]=nepochs

#     file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights_3D{}.h5".format(suffix)
#     file_checkpoint="/".join(filename_volumes.split("/")[:-1])+"/model_checkpoint_3D.h5"
#     print(file_checkpoint)
#     run=wandb.init(
#         project="project_test_3D",
#         config=config_train
#     )

#     #pad_amount=config_train["padding"]
#     loss = config_train["loss"]
#     decay=config_train["lr_decay"]
#     #pad_amount=tuple(tuple(l) for l in pad_amount)

#     # #Finding the power of 2 "closest" and longer than  x dimension
#     # n=all_volumes.shape[-1]
#     # pad_1=2**(int(np.log2(n))+1)-n
#     # pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

#     # if pad_2<0:
#     #     pad=int(pad_1/2)
#     # else:
#     #     pad = int(pad_2 / 2)

#     # n=all_volumes.shape[1]
#     # pad_1=2**(int(np.log2(n))+1)-n
#     # pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

#     # if pad_2<0:
#     #     pad_z=int(pad_1/2)
#     # else:
#     #     pad_z = int(pad_2 / 2)

#     # pad_amount = ((0,0),(pad_z,pad_z),(pad,pad), (pad,pad))
#     # print(pad_amount)

#     nb_features=config_train["nb_features"]
#     # configure unet features
#     #nb_features = [
#     #    [256, 32, 32, 32],         # encoder features
#     #    [32, 32, 32, 32, 32, 16]  # decoder features
#     #]

#     #nb_features = [
#     #    [32, 64, 128, 128],         # encoder features
#     #    [128, 128, 64, 64, 32, 16]  # decoder features
#     #]

#     optimizer=config_train["optimizer"] #"Adam"
#     lambda_param = config_train["lambda"] # 0.05
#     nb_epochs = config_train["nb_epochs"] # 200
#     batch_size=config_train["batch_size"] # 16
    
#     lr=config_train["lr"]
    

#     if downsample != 1:
#         all_volumes = np.array([downsample_3D(v, downsample) for v in all_volumes])

#     x_train_fixed,x_train_moving=format_input_voxelmorph_3D(all_volumes,sl_down=excluded,sl_top=-excluded)
#     # add channel
    
    



#     # configure unet input shape (concatenation of moving and fixed images)
#     inshape = x_train_fixed.shape[1:]
#     print("Input shape {}".format(inshape))
#     vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=7,int_downsize=1)

#     # voxelmorph has a variety of custom loss classes

#     if loss=="MSE":
#         losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
#     elif loss == "NCC":
#         losses = [vxm.losses.NCC(win=9).loss, vxm.losses.Grad('l2').loss]
#         # losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
#     elif loss =="MI":
#         losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
#     else:
#         raise ValueError("Loss should be either MSE or Mutual Information (MI)")

#     loss_weights = [1, lambda_param]

#     print("Compiling Model")
#     vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

#     if lr is not None:
#         backend.set_value(vxm_model.optimizer.learning_rate,lr)

#     train_generator = vxm_data_generator_3D(x_train_fixed,x_train_moving,batch_size=batch_size)


#     nb_examples=x_train_fixed.shape[0]

    
#     steps_per_epoch = int(nb_examples/batch_size)+1
    

#     if "min_lr" in config_train:
#         min_lr=config_train["min_lr"]
#     else:
#         min_lr=0.0002


#     curr_scheduler=lambda epoch,lr: scheduler(epoch,lr,decay,min_lr)
#     Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)
    
#     callback_checkpoint=WandbModelCheckpoint(filepath=file_checkpoint,save_best_only=True,save_weights_only=True,monitor="vxm_dense_transformer_loss")


#     if init_weights is not None:
#         vxm_model.load_weights(init_weights)

#     hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,callbacks=[Schedulecallback,WandbMetricsLogger(),callback_checkpoint])

#     vxm_model.save_weights(file_model)

#     run.finish()

#     plt.figure()
#     plt.plot(hist.epoch, hist.history["loss"], '.-')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.savefig(file_model.split(".h5")[0]+"_loss.jpg")


# ── new loss class — place above the training function ───────────────────────
class ChannelMagnitudeLoss:
    """
    Penalize magnitude of specific velocity channels.
    Designed to match the vxm .loss callable interface.

    suppressed_channels : list of channel indices to penalize
    weights             : per-channel penalty weight (same length as suppressed_channels)
    """
    def __init__(self, suppressed_channels=(1, 2), weights=(10.0, 10.0)):
        self.suppressed_channels = suppressed_channels
        self.weights             = weights

    def loss(self, _, y_pred):
        total = tf.cast(0.0, tf.float32)
        for ch, w in zip(self.suppressed_channels, self.weights):
            total += w * tf.reduce_mean(tf.square(y_pred[..., ch]))
        return total


# @ma.machine()
# @ma.parameter("filename_volumes", str,  default=None,                  description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
# @ma.parameter("file_config_train", str, default=DEFAULT_TRAIN_CONFIG_3D, description="Config training")
# @ma.parameter("suffix",            str,  default="",                    description="suffix")
# @ma.parameter("init_weights",      str,  default=None,                  description="Weights initialization from .h5 file")
# @ma.parameter("kept_bins",         str,  default=None,                  description="Bins to keep for training")
# @ma.parameter("nepochs",           int,  default=None,                  description="Number of epochs (overwrites config)")
# @ma.parameter("excluded",          int,  default=0,                     description="Excluded slices on both extremities")
# @ma.parameter("downsample",        int,  default=1,                     description="Downsampling")
# def train_voxelmorph_3D(filename_volumes, file_config_train, suffix, init_weights,
#                         kept_bins, nepochs, excluded, downsample):

#     with open(file_config_train, "r") as f:
#         config_train = json.load(f)

#     all_volumes = np.abs(np.load(filename_volumes)).astype("float32")
#     print("Volumes shape {}".format(all_volumes.shape))

#     if kept_bins is not None:
#         kept_bins_list = np.array(str.split(kept_bins, ",")).astype(int)
#         print(kept_bins_list)
#         all_volumes = all_volumes[kept_bins_list]

#     nb_gr, nb_slices, npoint, npoint = all_volumes.shape

#     if nepochs is not None:
#         config_train["nb_epochs"] = nepochs

#     file_model      = filename_volumes.split(".npy")[0] + "_vxm_model_weights_3D{}.h5".format(suffix)
#     file_checkpoint = "/".join(filename_volumes.split("/")[:-1]) + "/model_checkpoint_3D.h5"
#     print(file_checkpoint)

#     run = wandb.init(project="project_test_3D", config=config_train)

#     # ── read config ───────────────────────────────────────────────────────────
#     loss          = config_train["loss"]
#     decay         = config_train["lr_decay"]
#     nb_features   = config_train["nb_features"]
#     optimizer     = config_train["optimizer"]
#     lambda_param  = config_train["lambda"]
#     nb_epochs     = config_train["nb_epochs"]
#     batch_size    = config_train["batch_size"]
#     lr            = config_train["lr"]
#     min_lr        = config_train.get("min_lr", 0.0002)

#     # ── channel magnitude loss config (optional keys in config_train) ─────────
#     # Example entries in your JSON config:
#     #   "use_channel_magnitude_loss": true,
#     #   "channel_magnitude_suppressed": [1, 2],   <- channels to penalize
#     #   "channel_magnitude_weights":    [10, 10], <- one weight per channel
#     #   "lambda_channel":               1.0        <- global scale for this loss
#     use_channel_loss         = config_train.get("use_channel_magnitude_loss", False)
#     suppressed_channels      = config_train.get("channel_magnitude_suppressed", [1, 2])
#     channel_magnitude_weights = config_train.get("channel_magnitude_weights",
#                                                   [10.0] * len(suppressed_channels))
#     lambda_channel           = config_train.get("lambda_channel", 1.0)
#     ncc_win                  = config_train.get("ncc_win", 9)
#     int_steps                = config_train.get("int_steps", 7)
#     int_downsize                = config_train.get("int_downsize", 1)
    

#     # ── optional downsampling ─────────────────────────────────────────────────
#     if downsample != 1:
#         all_volumes = np.array([downsample_3D(v, downsample) for v in all_volumes])

#     x_train_fixed, x_train_moving = format_input_voxelmorph_3D(
#         all_volumes, sl_down=excluded, sl_top=-excluded
#     )

#     inshape = x_train_fixed.shape[1:]
#     print("Input shape {}".format(inshape))

#     vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=int_steps, int_downsize=int_downsize)

#     # ── build velocity extractor so we can pass velocity to channel loss ───────
#     # VoxelMorph model outputs: [moved_image, flow]
#     # We need the velocity (pre-integration) for the channel magnitude loss.
#     # Find VecInt layer and expose its input (= raw velocity).
#     vecint_layer = next(
#         (l for l in vxm_model.layers if isinstance(l, vxm.layers.VecInt)), None
#     )
#     if vecint_layer is None:
#         raise ValueError("No VecInt layer found — check int_steps > 0")

#     # extended model: outputs [moved, flow, velocity]
#     # used only during compile/fit so the loss can receive the velocity tensor
#     if use_channel_loss:
#         extended_model = tf.keras.Model(
#             inputs=vxm_model.inputs,
#             outputs=[
#                 vxm_model.outputs[0],   # moved image   → similarity loss
#                 vxm_model.outputs[1],   # integrated flow → smoothness loss
#                 vecint_layer.input,     # raw velocity   → channel magnitude loss
#             ]
#         )
#     else:
#         extended_model = vxm_model

#     # ── losses ────────────────────────────────────────────────────────────────
#     if loss == "MSE":
#         sim_loss = vxm.losses.MSE().loss
#     elif loss == "NCC":
#         sim_loss = vxm.losses.NCC(win=ncc_win).loss
#     elif loss == "MI":
#         sim_loss = vxm.losses.MutualInformation().loss
#     else:
#         raise ValueError("Loss should be MSE, NCC or MI")

#     smooth_loss = vxm.losses.Grad('l2').loss

#     if use_channel_loss:
#         losses = [
#             sim_loss,                  # on moved image
#             smooth_loss,               # on integrated flow
#             ChannelMagnitudeLoss(      # on raw velocity
#                 suppressed_channels=suppressed_channels,
#                 weights=channel_magnitude_weights,
#             ).loss,
#         ]
#         loss_weights = [1.0, lambda_param, lambda_channel]
#     else:
#         losses       = [sim_loss, smooth_loss]
#         loss_weights = [1.0, lambda_param]

#     print("Compiling model — losses: {}, weights: {}".format(
#         [l.__self__.__class__.__name__ if hasattr(l,'__self__') else l.__qualname__
#          for l in losses],
#         loss_weights
#     ))

#     extended_model.compile(
#         optimizer=optimizer,
#         loss=losses,
#         loss_weights=loss_weights
#     )

#     if lr is not None:
#         backend.set_value(extended_model.optimizer.learning_rate, lr)

#     # ── data generator ────────────────────────────────────────────────────────
#     # When use_channel_loss=True the model has 3 outputs, so the generator
#     # must yield 3 zero targets instead of 2.
#     def vxm_data_generator_3D_extended(x_fixed, x_moving, batch_size=8,
#                                         n_outputs=2):
#         nb   = x_fixed.shape[0]
#         zeros_shape_flow = x_fixed.shape[1:-1] + (3,)   # (Dz,Dx,Dy,3)
#         while True:
#             idx = np.random.randint(0, nb, size=batch_size)
#             inputs  = [x_moving[idx], x_fixed[idx]]
#             # target for similarity loss = fixed image
#             # targets for flow / velocity losses = zeros (regularization)
#             targets = [x_fixed[idx],
#                        np.zeros((batch_size, *zeros_shape_flow), dtype=np.float32)]
#             if n_outputs == 3:
#                 targets.append(
#                     np.zeros((batch_size, *zeros_shape_flow), dtype=np.float32)
#                 )
#             yield inputs, targets

#     n_outputs     = 3 if use_channel_loss else 2
#     nb_examples   = x_train_fixed.shape[0]
#     steps_per_epoch = int(nb_examples / batch_size) + 1

#     train_generator = vxm_data_generator_3D_extended(
#         x_train_fixed, x_train_moving,
#         batch_size=batch_size,
#         n_outputs=n_outputs
#     )

#     # ── callbacks ─────────────────────────────────────────────────────────────
#     curr_scheduler   = lambda epoch, lr: scheduler(epoch, lr, decay, min_lr)
#     schedule_cb      = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)

#     # monitor the transformer (similarity) loss for checkpointing
#     monitor_loss = "vxm_dense_transformer_loss"
#     checkpoint_cb = WandbModelCheckpoint(
#         filepath=file_checkpoint,
#         save_best_only=True,
#         save_weights_only=True,
#         monitor=monitor_loss
#     )

#     # ── custom W&B callback to log individual loss components ─────────────────
#     class LogLossComponents(tf.keras.callbacks.Callback):
#         """Log each loss component to W&B under a readable name."""
#         def __init__(self, use_channel_loss):
#             super().__init__()
#             self.use_channel_loss = use_channel_loss

#         def on_epoch_end(self, epoch, logs=None):
#             logs = logs or {}
#             log_dict = {"epoch": epoch}

#             # Keras names outputs as 'loss', then per-output losses
#             # Keys depend on output layer names — print logs.keys() once to confirm
#             for k, v in logs.items():
#                 log_dict[k] = v
            
#             print(logs.keys())

#             # add explicit readable keys if present
#             key_map = {
#                 "vxm_dense_transformer_loss":  "loss/similarity",
#                 "vxm_dense_flow_loss":         "loss/smoothness",
#             }
#             if self.use_channel_loss:
#                 # third output is velocity — key depends on VecInt layer name
#                 key_map["vxm_dense_flow_1_loss"] = "loss/channel_magnitude"

#             for keras_key, readable_key in key_map.items():
#                 if keras_key in logs:
#                     log_dict[readable_key] = logs[keras_key]

#             wandb.log(log_dict)

#     log_cb = LogLossComponents(use_channel_loss=use_channel_loss)

#     if init_weights is not None:
#         extended_model.load_weights(init_weights)

#     # ── train ─────────────────────────────────────────────────────────────────
#     hist = extended_model.fit_generator(
#         train_generator,
#         epochs=nb_epochs,
#         steps_per_epoch=steps_per_epoch,
#         verbose=2,
#         callbacks=[schedule_cb, WandbMetricsLogger(), checkpoint_cb, log_cb]
#     )

#     # save weights — compatible with original vxm_model (shared weights)
#     vxm_model.save_weights(file_model)

#     run.finish()

#     # ── loss plot ─────────────────────────────────────────────────────────────
#     plt.figure()
#     plt.plot(hist.epoch, hist.history["loss"], '.-', label="total")
#     if use_channel_loss and "vec_int_loss" in hist.history:
#         plt.plot(hist.epoch, hist.history["vec_int_loss"], '.-',
#                  label="channel magnitude")
#     plt.plot(hist.epoch,
#              hist.history.get("vxm_dense_transformer_loss", []),
#              '.-', label="similarity")
#     plt.plot(hist.epoch,
#              hist.history.get("vxm_dense_flow_loss", []),
#              '.-', label="smoothness")
#     plt.legend()
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.savefig(file_model.split(".h5")[0] + "_loss.jpg")

@ma.machine()
@ma.parameter("filename_volumes", str,  default=None,                  description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_config_train", str, default=DEFAULT_TRAIN_CONFIG_3D, description="Config training")
@ma.parameter("suffix",            str,  default="",                    description="suffix")
@ma.parameter("init_weights",      str,  default=None,                  description="Weights initialization from .h5 file")
@ma.parameter("kept_bins",         str,  default=None,                  description="Bins to keep for training")
@ma.parameter("nepochs",           int,  default=None,                  description="Number of epochs (overwrites config)")
@ma.parameter("excluded",          int,  default=0,                     description="Excluded slices on both extremities")
@ma.parameter("downsample",        int,  default=1,                     description="Downsampling")
def train_voxelmorph_3D(filename_volumes, file_config_train, suffix, init_weights,
                        kept_bins, nepochs, excluded, downsample):

    with open(file_config_train, "r") as f:
        config_train = json.load(f)

    all_volumes = np.abs(np.load(filename_volumes)).astype("float32")
    print("Volumes shape {}".format(all_volumes.shape))

    if kept_bins is not None:
        kept_bins_list = np.array(str.split(kept_bins, ",")).astype(int)
        print(kept_bins_list)
        all_volumes = all_volumes[kept_bins_list]

    nb_gr, nb_slices, npoint, npoint = all_volumes.shape

    if nepochs is not None:
        config_train["nb_epochs"] = nepochs

    file_model      = filename_volumes.split(".npy")[0] + "_vxm_model_weights_3D{}.h5".format(suffix)
    file_checkpoint = "/".join(filename_volumes.split("/")[:-1]) + "/model_checkpoint_3D.h5"
    print(file_checkpoint)

    run = wandb.init(project="project_test_3D", config=config_train)

    # ── read config ───────────────────────────────────────────────────────────
    loss             = config_train["loss"]
    decay            = config_train["lr_decay"]
    nb_features      = config_train["nb_features"]
    optimizer        = config_train["optimizer"]
    lambda_param     = config_train["lambda"]
    nb_epochs        = config_train["nb_epochs"]
    batch_size       = config_train["batch_size"]
    lr               = config_train["lr"]
    min_lr           = config_train.get("min_lr", 0.0002)
    use_channel_loss          = config_train.get("use_channel_magnitude_loss", False)
    suppressed_channels       = config_train.get("channel_magnitude_suppressed", [1, 2])
    channel_magnitude_weights = config_train.get("channel_magnitude_weights",
                                                  [10.0] * len(suppressed_channels))
    lambda_channel   = config_train.get("lambda_channel", 1.0)
    ncc_win          = config_train.get("ncc_win", 9)
    int_steps        = config_train.get("int_steps", 7)
    int_downsize     = config_train.get("int_downsize", 1)
    log_images_every = config_train.get("log_images_every", 100)

    # ── optional downsampling ─────────────────────────────────────────────────
    if downsample != 1:
        all_volumes = np.array([downsample_3D(v, downsample) for v in all_volumes])

    x_train_fixed, x_train_moving = format_input_voxelmorph_3D(
        all_volumes, sl_down=excluded, sl_top=-excluded
    )

    inshape = x_train_fixed.shape[1:]
    print("Input shape {}".format(inshape))

    # ── pick one fixed/moving pair for visual logging ─────────────────────────
    # use the last training pair (most motion, most informative)
    vis_fixed  = x_train_fixed[-1:]    # (1, Dz, Dx, Dy, 1)
    vis_moving = x_train_moving[-1:]   # (1, Dz, Dx, Dy, 1)

    # unpack spatial dims for slice indices
    Dz, Dx, Dy = inshape[:3]
    sl_z = Dz // 2
    sl_x = Dx // 2
    sl_y = Dy // 2

    def _make_residual_figure(fixed_vol, moving_vol, registered_vol,
                           sl_z, sl_x, sl_y):
        """
        fixed_vol, moving_vol, registered_vol: (Dz, Dx, Dy) numpy arrays
        sl_z, sl_x, sl_y: slice indices for the three orientations
        """
        slices_fixed      = [fixed_vol[sl_z, :, :],
                            fixed_vol[:, sl_x, :],
                            fixed_vol[:, :, sl_y]]
        slices_moving     = [moving_vol[sl_z, :, :],
                            moving_vol[:, sl_x, :],
                            moving_vol[:, :, sl_y]]
        slices_registered = [registered_vol[sl_z, :, :],
                            registered_vol[:, sl_x, :],
                            registered_vol[:, :, sl_y]]

        orient_labels = [f"Axial (z={sl_z})",
                        f"Coronal (x={sl_x})",
                        f"Sagittal (y={sl_y})"]

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        fig.suptitle("Registration quality", fontsize=12, fontweight="bold")

        for row, (s_fix, s_mov, s_reg, orient) in enumerate(
            zip(slices_fixed, slices_moving, slices_registered, orient_labels)
        ):
            res_before = s_mov - s_fix
            res_after  = s_reg - s_fix
            vmax = max(np.abs(res_before).max(), np.abs(res_after).max()) + 1e-9

            # col 0: moving − fixed
            ax = axes[row, 0]
            im = ax.imshow(res_before.T, cmap="coolwarm",
                        vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Moving − Fixed\n{orient}", fontsize=8)
            ax.axis("off")

            # col 1: registered − fixed
            ax = axes[row, 1]
            im = ax.imshow(res_after.T, cmap="coolwarm",
                        vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Registered − Fixed\n{orient}", fontsize=8)
            ax.axis("off")

            # col 2: registered image
            ax = axes[row, 2]
            im = ax.imshow(s_reg.T, cmap="gray",
                        vmin=0, vmax=1, origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Registered\n{orient}", fontsize=8)
            ax.axis("off")

        fig.tight_layout()
        return fig

    # ── model ─────────────────────────────────────────────────────────────────
    vxm_model = vxm.networks.VxmDense(
        inshape, nb_features, int_steps=int_steps, int_downsize=int_downsize
    )

    vecint_layer = next(
        (l for l in vxm_model.layers if isinstance(l, vxm.layers.VecInt)), None
    )
    if vecint_layer is None:
        raise ValueError("No VecInt layer found — check int_steps > 0")

    if use_channel_loss:
        extended_model = tf.keras.Model(
            inputs=vxm_model.inputs,
            outputs=[
                vxm_model.outputs[0],
                vxm_model.outputs[1],
                vecint_layer.input,
            ]
        )
    else:
        extended_model = vxm_model

    # ── losses ────────────────────────────────────────────────────────────────
    if loss == "MSE":
        sim_loss = vxm.losses.MSE().loss
    elif loss == "NCC":
        sim_loss = vxm.losses.NCC(win=ncc_win).loss
    elif loss == "MI":
        sim_loss = vxm.losses.MutualInformation().loss
    else:
        raise ValueError("Loss should be MSE, NCC or MI")

    smooth_loss = vxm.losses.Grad('l2').loss

    if use_channel_loss:
        losses = [
            sim_loss,
            smooth_loss,
            ChannelMagnitudeLoss(
                suppressed_channels=suppressed_channels,
                weights=channel_magnitude_weights,
            ).loss,
        ]
        loss_weights = [1.0, lambda_param, lambda_channel]
    else:
        losses       = [sim_loss, smooth_loss]
        loss_weights = [1.0, lambda_param]

    print("Compiling model — losses: {}, weights: {}".format(
        [l.__self__.__class__.__name__ if hasattr(l, '__self__') else l.__qualname__
         for l in losses],
        loss_weights
    ))

    extended_model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights
    )

    if lr is not None:
        backend.set_value(extended_model.optimizer.learning_rate, lr)

    # ── data generator ────────────────────────────────────────────────────────
    def vxm_data_generator_3D_extended(x_fixed, x_moving, batch_size=8,
                                        n_outputs=2):
        nb = x_fixed.shape[0]
        zeros_shape_flow = x_fixed.shape[1:-1] + (3,)
        while True:
            idx     = np.random.randint(0, nb, size=batch_size)
            inputs  = [x_moving[idx], x_fixed[idx]]
            targets = [x_fixed[idx],
                       np.zeros((batch_size, *zeros_shape_flow), dtype=np.float32)]
            if n_outputs == 3:
                targets.append(
                    np.zeros((batch_size, *zeros_shape_flow), dtype=np.float32)
                )
            yield inputs, targets

    n_outputs       = 3 if use_channel_loss else 2
    nb_examples     = x_train_fixed.shape[0]
    steps_per_epoch = int(nb_examples / batch_size) + 1

    train_generator = vxm_data_generator_3D_extended(
        x_train_fixed, x_train_moving,
        batch_size=batch_size,
        n_outputs=n_outputs
    )

    # ── callbacks ─────────────────────────────────────────────────────────────
    curr_scheduler = lambda epoch, lr: scheduler(epoch, lr, decay, min_lr)
    schedule_cb    = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)

    monitor_loss  = "vxm_dense_transformer_loss"
    checkpoint_cb = WandbModelCheckpoint(
        filepath=file_checkpoint,
        save_best_only=True,
        save_weights_only=True,
        monitor=monitor_loss
    )

    class LogLossComponents(tf.keras.callbacks.Callback):
        """Log loss components and periodic registration quality images to W&B."""

        def __init__(self, use_channel_loss, vis_fixed, vis_moving,
                    log_images_every=100):
            super().__init__()
            self.use_channel_loss = use_channel_loss
            self.vis_fixed        = vis_fixed
            self.vis_moving       = vis_moving
            self.log_images_every = log_images_every

        def on_epoch_end(self, epoch, logs=None):
            logs     = logs or {}
            log_dict = {"epoch": epoch}

            print(logs.keys())
            for k, v in logs.items():
                log_dict[k] = v

            key_map = {
                "vxm_dense_transformer_loss": "loss/similarity",
                "vxm_dense_flow_loss":        "loss/smoothness",
            }
            if self.use_channel_loss:
                key_map["vxm_dense_flow_1_loss"] = "loss/channel_magnitude"

            for keras_key, readable_key in key_map.items():
                if keras_key in logs:
                    log_dict[readable_key] = logs[keras_key]

            # ── image logging ─────────────────────────────────────────────────────
            if (epoch + 1) % self.log_images_every == 0 or epoch == 0:
                try:
                    # run inference
                    pred = vxm_model.predict(
                        [self.vis_moving, self.vis_fixed], verbose=0
                    )

                    # pred[0] shape: (1, Dz, Dx, Dy, 1) for 3D
                    #                (1, Dz, Dx, 1)      for 2D — shouldn't happen here
                    print(f"[img log] pred[0].shape = {pred[0].shape}")
                    print(f"[img log] vis_fixed.shape = {self.vis_fixed.shape}")
                    print(f"[img log] vis_moving.shape = {self.vis_moving.shape}")

                    # safely extract spatial volume — remove batch and channel dims
                    # works for both (1,Dz,Dx,Dy,1) and unexpected shapes
                    registered = np.squeeze(pred[0])           # (Dz, Dx, Dy)
                    fixed_vol  = np.squeeze(self.vis_fixed)    # (Dz, Dx, Dy)
                    moving_vol = np.squeeze(self.vis_moving)   # (Dz, Dx, Dy)

                    print(f"[img log] squeezed shapes: "
                        f"fixed={fixed_vol.shape} "
                        f"moving={moving_vol.shape} "
                        f"registered={registered.shape}")

                    # guard: only log if we got 3D volumes
                    if fixed_vol.ndim != 3:
                        print(f"[img log] WARNING: expected 3D volume after squeeze, "
                            f"got {fixed_vol.ndim}D — skipping image log")
                        wandb.log(log_dict)
                        return

                    Dz, Dx, Dy = fixed_vol.shape
                    sl_z = Dz // 2
                    sl_x = Dx // 2
                    sl_y = Dy // 2

                    fig = _make_residual_figure(
                        fixed_vol, moving_vol, registered,
                        sl_z, sl_x, sl_y
                    )

                    log_dict["registration/residuals"] = wandb.Image(
                        fig,
                        caption=f"Epoch {epoch + 1} — residuals & registered slices"
                    )
                    plt.close(fig)

                    mae_before = float(np.abs(moving_vol - fixed_vol).mean())
                    mae_after  = float(np.abs(registered  - fixed_vol).mean())
                    log_dict["registration/mae_before"] = mae_before
                    log_dict["registration/mae_after"]  = mae_after
                    log_dict["registration/mae_ratio"]  = mae_after / (mae_before + 1e-9)

                except Exception as e:
                    # never crash training because of logging
                    print(f"[img log] ERROR at epoch {epoch + 1}: {e}")
                    import traceback
                    traceback.print_exc()

            wandb.log(log_dict)

    log_cb = LogLossComponents(
        use_channel_loss  = use_channel_loss,
        vis_fixed         = vis_fixed,
        vis_moving        = vis_moving,
        log_images_every  = log_images_every,
    )

    if init_weights is not None:
        extended_model.load_weights(init_weights)

    # ── train ─────────────────────────────────────────────────────────────────
    hist = extended_model.fit_generator(
        train_generator,
        epochs          = nb_epochs,
        steps_per_epoch = steps_per_epoch,
        verbose         = 2,
        callbacks       = [schedule_cb, WandbMetricsLogger(), checkpoint_cb, log_cb]
    )

    vxm_model.save_weights(file_model)
    run.finish()

    # ── loss plot ─────────────────────────────────────────────────────────────
    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-', label="total")
    if use_channel_loss and "vxm_dense_flow_1_loss" in hist.history:
        plt.plot(hist.epoch, hist.history["vxm_dense_flow_1_loss"], '.-',
                 label="channel magnitude")
    plt.plot(hist.epoch,
             hist.history.get("vxm_dense_transformer_loss", []),
             '.-', label="similarity")
    plt.plot(hist.epoch,
             hist.history.get("vxm_dense_flow_loss", []),
             '.-', label="smoothness")
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(file_model.split(".h5")[0] + "_loss.jpg")



@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG,description="Config training")
@ma.parameter("suffix",str,default="",description="suffix")
@ma.parameter("init_weights",str,default=None,description="Weights initialization from .h5 file")
def train_voxelmorph_torchio(filename_volumes,file_config_train,suffix,init_weights):
    
    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes = np.load(filename_volumes)
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")

    file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights{}.h5".format(suffix)
    run=wandb.init(
        project=file_model.split("/")[-1],
        config=config_train
    )

    #pad_amount=config_train["padding"]
    loss = config_train["loss"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)
    decay=config_train["lr_decay"]

    #Finding the power of 2 "closest" and longer than  x dimension
    n=all_volumes.shape[-1]
    pad_1=2**(int(np.log2(n))+1)-n
    pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2<0:
        pad=int(pad_1/2)
    else:
        pad = int(pad_2 / 2)

    pad_amount = ((0,0),(pad,pad), (pad,pad))
    print(pad_amount)
    nb_features=config_train["nb_features"]
    # configure unet features
    #nb_features = [
    #    [256, 32, 32, 32],         # encoder features
    #    [32, 32, 32, 32, 32, 16]  # decoder features
    #]

    #nb_features = [
    #    [32, 64, 128, 128],         # encoder features
    #    [128, 128, 64, 64, 32, 16]  # decoder features
    #]

    
    #  "nb_features": [
    #      [64, 128, 256, 256, 256],
    #      [256, 256, 256, 128, 128, 64, 32, 16]
    #        ],


    optimizer=config_train["optimizer"] #"Adam"
    lambda_param = config_train["lambda"] # 0.05
    nb_epochs = config_train["nb_epochs"] # 200
    batch_size=config_train["batch_size"] # 16
    

    x_train_fixed,x_train_moving=format_input_voxelmorph(all_volumes,pad_amount,normalize=False)



    # configure unet input shape (concatenation of moving and fixed images)
    inshape = x_train_fixed.shape[1:]

    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # voxelmorph has a variety of custom loss classes

    if loss=="MSE":
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss =="MI":
        losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    else:
        raise ValueError("Loss should be either MSE or Mutual Information (MI)")

    loss_weights = [1, lambda_param]

    print("Compiling Model")
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    
    noise=tio.RandomNoise(std=(0,0.05))
    blur=tio.RandomBlur(std=(0,2))
    gamma=tio.RandomGamma()
    noise_transform={
        blur:0.5,
        noise:0.5,
    }
    intensity_transform={
        gamma:1.0,
    }

    spatial_transforms = {
        #tio.RandomElasticDeformation(max_displacement=1): 0.2,
        tio.RandomAffine(): 1.0,
    }

    transf=tio.Compose([tio.OneOf(noise_transform,p=0.3),tio.OneOf(intensity_transform,p=0.3),tio.OneOf(spatial_transforms,p=0.3),tio.RescaleIntensity()])

    train_generator = vxm_data_generator_torchio(x_train_fixed,x_train_moving,batch_size=batch_size,transform=transf)

    nb_examples=x_train_fixed.shape[0]

    steps_per_epoch = int(nb_examples/batch_size)+1
    
    curr_scheduler=lambda epoch,lr: scheduler(epoch,lr,decay)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)


    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,callbacks=[Schedulecallback,WandbMetricsLogger()])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0]+"_loss.jpg")

@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG,description="Config training")
@ma.parameter("suffix",str,default="",description="suffix")
@ma.parameter("init_weights",str,default=None,description="Weights initialization from .h5 file")
def train_voxelmorph_torchvision(filename_volumes,file_config_train,suffix,init_weights):
    
    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes = np.load(filename_volumes)
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=np.abs(all_volumes).astype("float32")

    file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights_torchaugment{}.h5".format(suffix)
    file_checkpoint="/".join(filename_volumes.split("/")[:-1])+"/model_checkpoint.h5"
    print(file_checkpoint)
    run=wandb.init(
        project=file_model.split("/")[-1],
        config=config_train
    )

    #pad_amount=config_train["padding"]
    loss = config_train["loss"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)
    decay=config_train["lr_decay"]

    #Finding the power of 2 "closest" and longer than  x dimension
    n=all_volumes.shape[-1]
    pad_1=2**(int(np.log2(n))+1)-n
    pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2<0:
        pad=int(pad_1/2)
    else:
        pad = int(pad_2 / 2)

    pad_amount = ((0,0),(pad,pad), (pad,pad))
    print(pad_amount)
    nb_features=config_train["nb_features"]
    # configure unet features
    #nb_features = [
    #    [256, 32, 32, 32],         # encoder features
    #    [32, 32, 32, 32, 32, 16]  # decoder features
    #]

    #nb_features = [
    #    [32, 64, 128, 128],         # encoder features
    #    [128, 128, 64, 64, 32, 16]  # decoder features
    #]

    optimizer=config_train["optimizer"] #"Adam"
    lambda_param = config_train["lambda"] # 0.05
    nb_epochs = config_train["nb_epochs"] # 200
    batch_size=config_train["batch_size"] # 16

    all_groups_combination=config_train["all_groups_combination"]
    perc_augmentation = config_train["perc_augmentation"]
    test_size = config_train["test_size"]

    x_fixed,x_moving=format_input_voxelmorph(all_volumes,pad_amount,normalize=True,all_groups_combination=all_groups_combination)
    x_train_fixed, x_test_fixed, x_train_moving, x_test_moving = train_test_split(x_fixed, x_moving, test_size=test_size, random_state=42)

    print("Train size : {}".format(x_train_fixed.shape))
    print("Test size : {}".format(x_test_fixed.shape))

    # configure unet input shape (concatenation of moving and fixed images)
    inshape = x_train_fixed.shape[1:]

    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # voxelmorph has a variety of custom loss classes

    if loss=="MSE":
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss =="MI":
        losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    else:
        raise ValueError("Loss should be either MSE or Mutual Information (MI)")

    loss_weights = [1, lambda_param]

    print("Compiling Model")
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    
    transform_color=T.Lambda(lambda x : T.functional.adjust_contrast(T.functional.adjust_brightness(x,1.5),1.5))
    identity=T.Lambda(lambda x : x)
    transform_list=[T.GaussianBlur(9,2),T.GaussianBlur(5,2),transform_color,T.Lambda(lambda x : x + 0.05*torch.randn_like(x)),T.Lambda(lambda x : x + 0.1*torch.randn_like(x)),identity]
    probabilities= [perc_augmentation/(len(transform_list)-1)]*(len(transform_list)-1)+[1-perc_augmentation]
    train_generator = vxm_data_generator_torchvision(x_train_fixed,x_train_moving,batch_size=batch_size,transform_list=transform_list,probabilities=probabilities)

    validation_generator=vxm_data_generator_torchvision(x_test_fixed,x_test_moving,batch_size=batch_size,transform_list=transform_list,probabilities=[0]*(len(transform_list)-1)+[1.0])

    nb_examples_training=x_train_fixed.shape[0]
    nb_examples_test = x_test_fixed.shape[0]
    steps_per_epoch_training = int(nb_examples_training/batch_size)+1
    steps_per_epoch_test = int(nb_examples_test / batch_size) + 1
    
    curr_scheduler=lambda epoch,lr: scheduler(epoch,lr,decay)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)


    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    callback_checkpoint=WandbModelCheckpoint(filepath=file_checkpoint,save_best_only=True,save_weights_only=True)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, validation_data=validation_generator,steps_per_epoch=steps_per_epoch_training,validation_steps=steps_per_epoch_test, verbose=2,callbacks=[Schedulecallback,WandbMetricsLogger(log_freq=8),callback_checkpoint])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0]+"_loss.jpg")


@ma.machine()
@ma.parameter("file_config_train", str, default=DEFAULT_TRAIN_CONFIG, description="Config training")
@ma.parameter("kept_bins", str, default=None, description="Bins to keep for training")
@ma.parameter("suffix", str, default="", description="suffix")
@ma.parameter("init_weights", str, default=None, description="Weights initialization from .h5 file")
def train_voxelmorph_torchvision_multiple_patients(file_config_train, suffix, init_weights, kept_bins):
    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    filenames = config_train["filenames"]
    folder = config_train["folder"]

    file_model = folder + "/multiple_patients_vxm_model_weights_torchaugment{}.h5".format(suffix)
    file_checkpoint = folder + "/multiple_patients_model_checkpoint.h5"
    print(file_checkpoint)
    run = wandb.init(
        project=file_model.split("/")[-1],
        config=config_train
    )

    # pad_amount=config_train["padding"]
    loss = config_train["loss"]
    # pad_amount=tuple(tuple(l) for l in pad_amount)
    decay = config_train["lr_decay"]

    n = 0
    volumes_all_patients = []
    n_all_patients = []

    for file_volume in filenames:
        all_volumes = np.load(folder + file_volume)
        print("Volumes shape {}".format(all_volumes.shape))
        all_volumes = np.abs(all_volumes).astype("float32")
        if kept_bins is not None:
            kept_bins_list = np.array(str.split(kept_bins, ",")).astype(int)
            print(kept_bins_list)
            all_volumes = all_volumes[kept_bins_list]
        n_curr = all_volumes.shape[-1]
        n_all_patients.append(n_curr)
        if n_curr > n:
            n = n_curr
        volumes_all_patients.append(all_volumes)
    pads = ((n - np.array(n_all_patients)) / 2).astype(int)
    volumes_all_patients = [np.pad(v, ((0, 0), (0, 0), (pads[i], pads[i]), (pads[i], pads[i])), "constant") for i, v in
                            enumerate(volumes_all_patients)]

    for v in volumes_all_patients:
        print(v.shape)

    # Finding the power of 2 "closest" and longer than  x dimension

    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad = int(pad_1 / 2)
    else:
        pad = int(pad_2 / 2)

    if n%2==0:
        pad=0

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

    all_groups_combination = config_train["all_groups_combination"]
    perc_augmentation = config_train["perc_augmentation"]
    test_size = config_train["test_size"]

    lr = config_train["lr"]

    x_fixed_all = []
    x_moving_all = []

    for v in volumes_all_patients:
        x_fixed, x_moving = format_input_voxelmorph(v, pad_amount, normalize=True,
                                                    all_groups_combination=all_groups_combination)
        x_fixed_all.append(x_fixed)
        x_moving_all.append(x_moving)
        print(x_fixed.shape)

    x_fixed_all = np.concatenate(x_fixed_all, axis=0)
    x_moving_all = np.concatenate(x_moving_all, axis=0)
    x_train_fixed, x_test_fixed, x_train_moving, x_test_moving = train_test_split(x_fixed_all, x_moving_all,
                                                                                  test_size=test_size, random_state=42)

    print("Train size : {}".format(x_train_fixed.shape))
    print("Test size : {}".format(x_test_fixed.shape))

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

    transform_color = T.Lambda(lambda x: T.functional.adjust_contrast(T.functional.adjust_brightness(x, 1.5), 1.5))
    identity = T.Lambda(lambda x: x)
    transform_list = [T.GaussianBlur(9, 2), T.GaussianBlur(5, 2), transform_color,
                      T.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
                      T.Lambda(lambda x: x + 0.1 * torch.randn_like(x)), identity]
    probabilities = [perc_augmentation / (len(transform_list) - 1)] * (len(transform_list) - 1) + [
        1 - perc_augmentation]
    train_generator = vxm_data_generator_torchvision(x_train_fixed, x_train_moving, batch_size=batch_size,
                                                     transform_list=transform_list, probabilities=probabilities)

    validation_generator = vxm_data_generator_torchvision(x_test_fixed, x_test_moving, batch_size=batch_size,
                                                          transform_list=transform_list,
                                                          probabilities=[0] * (len(transform_list) - 1) + [1.0])

    nb_examples_training = x_train_fixed.shape[0]
    nb_examples_test = x_test_fixed.shape[0]
    steps_per_epoch_training = int(nb_examples_training / batch_size) + 1
    steps_per_epoch_test = int(nb_examples_test / batch_size) + 1

    if "min_lr" in config_train:
        min_lr = config_train["min_lr"]

    curr_scheduler = lambda epoch, lr: scheduler(epoch, lr, decay, min_lr)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)

    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    callback_checkpoint = WandbModelCheckpoint(filepath=file_checkpoint, save_best_only=True, save_weights_only=True)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, validation_data=validation_generator,
                                   steps_per_epoch=steps_per_epoch_training, validation_steps=steps_per_epoch_test,
                                   verbose=2,
                                   callbacks=[Schedulecallback, WandbMetricsLogger()])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0] + "_loss.jpg")

@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_model",str,default=None,description="Trained Model weights")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG,description="Config training")

def evaluate_model(filename_volumes,file_model,file_config_train):
    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    dx=1# Might want to change that (but not that important for now given that the registration is 2D)
    dy=1
    dz=5

    all_volumes = np.load(filename_volumes)
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")
    nb_gr=all_volumes.shape[0]
    
    
    
    #pad_amount=config_train["padding"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)

    n = all_volumes.shape[-1]
    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad = int(pad_1 / 2)
    else:
        pad = int(pad_2 / 2)

    pad_amount = ((0, 0), (pad, pad), (pad, pad))


    nb_features=config_train["nb_features"]

    for gr in range(nb_gr-1):
        x_val_fixed,x_val_moving=format_input_voxelmorph(all_volumes[[gr,gr+1]],pad_amount)
        inshape=x_val_fixed.shape[1:]

        vxm_model=vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
        vxm_model.load_weights(file_model)

        
        val_input=[x_val_moving[...,None],x_val_fixed[...,None]]
        val_pred=vxm_model.predict(val_input)

        field_array=np.zeros(shape=val_pred[1].shape[:-1]+(3,),dtype=val_pred[1].dtype)
        field_array[:,:,:,:2]=val_pred[1]
        field=sitk.GetImageFromArray(field_array,isVector=True)
        field.SetSpacing([dx,dy,dz])
        moving_3D=sitk.GetImageFromArray(val_input[0][:,:,:,0])
        moving_3D.SetSpacing([dx,dy,dz])
        fixed_3D=sitk.GetImageFromArray(val_input[1][:,:,:,0])
        fixed_3D.SetSpacing([dx,dy,dz])
        moved_3D=sitk.GetImageFromArray(val_pred[0][:,:,:,0])
        moved_3D.SetSpacing([dx,dy,dz])

        sitk.WriteImage(field,file_model.split(".h5")[0]+"_displacement_field_vm_gr{}.nii".format(gr))
        sitk.WriteImage(moving_3D,file_model.split(".h5")[0]+"_moving_vm_gr{}.mha".format(gr))
        sitk.WriteImage(fixed_3D,file_model.split(".h5")[0]+"_fixed_vm_gr{}.mha".format(gr))
        sitk.WriteImage(moved_3D,file_model.split(".h5")[0]+"_moved_vm_gr{}.mha".format(gr))


@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_model",str,default=None,description="Trained Model weights")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG,description="Config training")
@ma.parameter("file_deformation",str,default=None,description="Initial deformation if doing multiple pass of registration algo")
@ma.parameter("niter",int,default=1,description="Number of iterations for registration")
@ma.parameter("resolution",int,default=None,description="Image resolution")
@ma.parameter("metric",["abs","phase","real","imag"],default="abs",description="Metric to register")
@ma.parameter("axis",int,default=None,description="Change registration axis")
def register_allbins_to_baseline(filename_volumes,file_model,file_config_train,niter,file_deformation,resolution,metric,axis):

    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes = np.load(filename_volumes)


    if resolution is None:
        filename_registered_volumes=filename_volumes.split(".npy")[0]+"_registered.npy"
        filename_deformation = filename_volumes.split(".npy")[0] + "_deformation_map.npy"
    else:
        filename_registered_volumes=filename_volumes.split(".npy")[0]+"_registered_res{}.npy".format(resolution)
        filename_deformation = filename_volumes.split(".npy")[0] + "_deformation_map_res{}.npy".format(resolution)



    if metric=="abs":
        all_volumes = np.abs(all_volumes)
    elif metric=="phase":
        all_volumes = np.angle(all_volumes)
        filename_registered_volumes = str.replace(filename_registered_volumes,"registered","registered_phase")
        filename_deformation = str.replace(filename_deformation, "deformation_map", "deformation_map_phase")

    elif metric=="real":
        all_volumes = np.real(all_volumes)
        filename_deformation = str.replace(filename_deformation, "deformation_map", "deformation_map_real")

    elif metric=="imag":
        all_volumes = np.imag(all_volumes)
        filename_deformation = str.replace(filename_deformation, "deformation_map", "deformation_map_imag")

    else:
        raise ValueError("metric unknown - choose from abs/phase/real/imag")

    if file_deformation is not None:
        deformation_map=np.load(file_deformation)
    else:
        deformation_map=None

    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")
    nb_gr,nb_slices,npoint,npoint=all_volumes.shape

    if resolution is not None:
        all_volumes=resize(all_volumes,(nb_gr,nb_slices,resolution,resolution))
    
    if axis is not None:
        all_volumes=np.moveaxis(all_volumes,axis+1,1)
        # all_volumes=resize(all_volumes,(nb_gr,npoint,npoint,npoint))
    
    #pad_amount=config_train["padding"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)
    n = np.maximum(all_volumes.shape[-1],all_volumes.shape[-2])
    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad = int(pad_1 / 2)
    else:
        pad = int(pad_2 / 2)

    # if n%2==0:
    #     pad=0
    pad_x=int((2*pad+n-all_volumes.shape[-2])/2)
    pad_y=int((2*pad+n-all_volumes.shape[-1])/2)

    pad_amount = ((0, 0), (pad_x, pad_x), (pad_y, pad_y))
    print(pad_amount)

    nb_features=config_train["nb_features"]
    inshape=np.pad(all_volumes[0],pad_amount,mode="constant").shape[1:]
    print(inshape)
    vxm_model=vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    vxm_model.load_weights(file_model)

    # Filtering out slices with only 0 as it seems to be buggy
    # sl_down_non_zeros = 0
    # while not (np.any(all_volumes[:, sl_down_non_zeros])):
    #     sl_down_non_zeros += 1
    #
    # sl_top_non_zeros = nb_slices
    # while not (np.any(all_volumes[:, sl_top_non_zeros-1])):
    #     sl_top_non_zeros -= 1
    #
    # print(sl_top_non_zeros)
    # print(sl_down_non_zeros)
    # all_volumes=all_volumes[:,sl_down_non_zeros:sl_top_non_zeros]
    registered_volumes=copy(all_volumes)
    mapxbase_all=np.zeros_like(all_volumes)
    mapybase_all = np.zeros_like(all_volumes)
    print(registered_volumes.shape)


    i=0
    while i<niter:
        print("Registration for iter {}".format(i+1))
        for gr in range(nb_gr):
            registered_volumes[gr],mapxbase_all[gr],mapybase_all[gr]=register_motionbin(vxm_model,all_volumes,gr,pad_amount,deformation_map)

        all_volumes=copy(registered_volumes)
        deformation_map=np.stack([mapxbase_all,mapybase_all],axis=0)
        print(deformation_map.shape)
        i+=1

    if axis is not None:
        registered_volumes=np.moveaxis(registered_volumes,1,axis+1)
        # registered_volumes=resize(registered_volumes,(nb_gr,nb_slices,npoint,npoint))
        # registered_volumes=registered_volumes[:,::int(npoint/nb_slices)]
        deformation_map=np.moveaxis(deformation_map,2,axis+2)
        # deformation_map=resize(deformation_map,(2,nb_gr,nb_slices,npoint,npoint))


    if resolution is not None:
        deformation_map=resize(deformation_map,(2,nb_gr,nb_slices,npoint,npoint),order=3)
    np.save(filename_registered_volumes,registered_volumes)
    np.save(filename_deformation, deformation_map)
    
@ma.machine()
@ma.parameter("file_deformation",str,default=None,description="deformation")
@ma.parameter("gr",int,default=None,description="Respiratory bin")
@ma.parameter("sl",int,default=None,description="Slice")
@ma.parameter("axis",int,default=None,description="Change registration axis")
def plot_deformation_flow(file_deformation,gr,sl,axis):
    deformation_map=np.load(file_deformation)
    if gr is None:
        gr=deformation_map.shape[1]-1
    if axis is not None:
        deformation_map=np.moveaxis(deformation_map,axis+2,2)
    file_plot=file_deformation.split(".npy")[0]+"_gr{}sl{}.jpg".format(gr,sl)
    print(file_plot)
    plot_deformation_map(deformation_map[:,gr,sl],us=4,save_file=file_plot)


@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_deformation",str,default=None,description="Initial deformation if doing multiple pass of registration algo")
@ma.parameter("axis",int,default=None,description="Registration axis")
def apply_deformation_map(filename_volumes,file_deformation,axis):
    all_volumes = np.load(filename_volumes)
    filename_registered_volumes = filename_volumes.split(".npy")[0] + "_registered_by_deformation.npy"

    # if metric=="abs":
    #     all_volumes = np.abs(all_volumes)
    # elif metric=="phase":
    #     all_volumes = np.angle(all_volumes)
    #     filename_registered_volumes = str.replace(filename_registered_volumes,"registered","registered_phase")
    #
    # elif metric=="real":
    #     all_volumes = np.real(all_volumes)
    #     filename_registered_volumes = str.replace(filename_registered_volumes,"registered","registered_real")
    #
    # elif metric=="imag":
    #     all_volumes = np.imag(all_volumes)
    #     filename_registered_volumes = str.replace(filename_registered_volumes,"registered","registered_imag")
    #
    # else:
    #     raise ValueError("metric unknown - choose from abs/phase/real/imag")

    deformation_map=np.load(file_deformation)

    # if axis is not None:
    #     all_volumes=np.moveaxis(all_volumes,axis+1,1)
    #     deformation_map=np.moveaxis(deformation_map,axis+2,2)

    deformed_volumes = np.zeros_like(all_volumes)
    nb_gr=all_volumes.shape[0]
    print(deformed_volumes.dtype)
    print(all_volumes.dtype)

    

    for gr in range(nb_gr):
        deformed_volumes[gr]= apply_deformation_to_complex_volume(all_volumes[gr], deformation_map[:,gr],axis=axis)

    # if axis is not None:
    #     deformed_volumes=np.moveaxis(deformed_volumes,1,axis+1)
        

    np.save(filename_registered_volumes,deformed_volumes)

    return

# @ma.machine()
# @ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
# @ma.parameter("file_model",str,default=None,description="Trained Model weights")
# @ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG_3D,description="Config training")
# @ma.parameter("file_deformation",str,default=None,description="Initial deformation if doing multiple pass of registration algo")
# @ma.parameter("niter",int,default=1,description="Number of iterations for registration")
# @ma.parameter("excluded",int,default=0,description="Excluded slices on both extremities")
# @ma.parameter("downsample",int,default=1,description="Downsampling input")
# def register_allbins_to_baseline_3D(filename_volumes,file_model,file_config_train,niter,file_deformation,excluded,downsample):

#     with open(file_config_train,"r") as f:
#         config_train=json.load(f)

#     all_volumes_full = np.abs(np.load(filename_volumes))
#     filename_registered_volumes=filename_volumes.split(".npy")[0]+"_registered_3D.npy"
#     filename_deformation = filename_volumes.split(".npy")[0] + "_deformation_map_3D.npy"

#     if file_deformation is not None:
#         deformation_map=np.load(file_deformation)
#     else:
#         deformation_map=None

#     print("Volumes shape {}".format(all_volumes_full.shape))
#     all_volumes_full=all_volumes_full.astype("float32")
#     nb_gr=all_volumes_full.shape[0]
    
    

#     if excluded >0:
#         all_volumes_full=all_volumes_full[:,excluded:-excluded]
    

#     if downsample != 1:
#         all_volumes = np.array([downsample_3D(v, downsample)
#             for v in all_volumes_full
#         ])
#     else:
#         all_volumes = copy(all_volumes_full)

#     print(all_volumes.shape)
#     pad_amount=pad_to_multiple(all_volumes[0],mult=16,offset=0)
#     print(pad_amount)

    

#     nb_features=config_train["nb_features"]
#     inshape=np.pad(all_volumes[0],pad_amount,mode="constant").shape
#     #print(inshape)
#     #print(inshape)
#     vxm_model=vxm.networks.VxmDense(inshape, nb_features, int_steps=7)
#     vxm_model.load_weights(file_model)
    
#     registered_volumes=copy(all_volumes)
#     mapxbase_all=np.zeros_like(all_volumes_full)
#     mapybase_all = np.zeros_like(all_volumes_full)
#     mapzbase_all = np.zeros_like(all_volumes_full)
#     i=0
#     while i<niter:
#         print("Registration for iter {}".format(i+1))
#         for gr in range(nb_gr):
#             registered_volumes[gr],mapzbase_all[gr],mapxbase_all[gr],mapybase_all[gr]=register_motionbin_3D(vxm_model,all_volumes,gr,pad_amount,deformation_map,full_shape=all_volumes_full.shape[1:])

#         all_volumes=copy(registered_volumes)
#         deformation_map=np.stack([mapzbase_all,mapxbase_all,mapybase_all],axis=0)
#         print(deformation_map.shape)
#         i+=1

#     if downsample != 1:
#         registered_volumes = np.array([
#             upsample_volume(v, all_volumes_full.shape[1:],order=2)
#             for v in registered_volumes
#         ])

#     print(registered_volumes.shape)
#     np.save(filename_registered_volumes,registered_volumes)
#     np.save(filename_deformation, deformation_map)
    


@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_model",str,default=None,description="Trained Model weights")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG_3D,description="Config training")
@ma.parameter("file_deformation",str,default=None,description="Initial deformation if doing multiple pass of registration algo")
@ma.parameter("niter",int,default=1,description="Number of iterations for registration")
@ma.parameter("excluded",int,default=0,description="Excluded slices on both extremities")
@ma.parameter("downsample",int,default=1,description="Downsampling input")
def register_allbins_to_baseline_3D(filename_volumes,file_model,file_config_train,niter,file_deformation,excluded,downsample):

    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes_full = np.abs(np.load(filename_volumes))
    filename_registered_volumes=filename_volumes.split(".npy")[0]+"_registered_3D.npy"
    filename_deformation = filename_volumes.split(".npy")[0] + "_deformation_map_3D.npy"

    if file_deformation is not None:
        deformation_map=np.load(file_deformation)
    else:
        deformation_map=None

    print("Volumes shape {}".format(all_volumes_full.shape))
    all_volumes_full=all_volumes_full.astype("float32")
    nb_gr=all_volumes_full.shape[0]
    
    

    if excluded >0:
        all_volumes_full=all_volumes_full[:,excluded:-excluded]
    

    if downsample != 1:
        all_volumes = np.array([downsample_3D(v, downsample)
            for v in all_volumes_full
        ])
    else:
        all_volumes = copy(all_volumes_full)

    print(all_volumes.shape)
    pad_amount=pad_to_multiple(all_volumes[0],mult=16,offset=0)
    print(pad_amount)

    

    nb_features=config_train["nb_features"]
    inshape=np.pad(all_volumes[0],pad_amount,mode="constant").shape
    #print(inshape)
    #print(inshape)
    vxm_model=vxm.networks.VxmDense(inshape, nb_features, int_steps=7)#,int_downsize=1)
    vxm_model.load_weights(file_model)
    
    registered_volumes=copy(all_volumes)
    flow_all = np.zeros((nb_gr, *all_volumes_full.shape[1:], 3), dtype=np.float32)
    i=0
    while i<niter:
        print("Registration for iter {}".format(i+1))
        for gr in range(nb_gr):
            registered_volumes[gr], flow_all[gr] = register_motionbin_3D(
                                                    vxm_model, all_volumes, gr, pad_amount,
                                                    deformation_map, full_shape=all_volumes_full.shape[1:]
                                                )

        all_volumes=copy(registered_volumes)
        deformation_map=flow_all
        print(deformation_map.shape)
        i+=1

    if downsample != 1:
        registered_volumes = np.array([
            upsample_volume(v, all_volumes_full.shape[1:],order=2)
            for v in registered_volumes
        ])

    print(registered_volumes.shape)
    np.save(filename_registered_volumes,registered_volumes)
    np.save(filename_deformation, flow_all)



def register_motionbin(vxm_model,all_volumes,gr,pad_amount,deformation_map=None):
    curr_gr=gr
    moving_volume=np.pad(all_volumes[curr_gr],pad_amount,mode="constant")
    nb_slices=all_volumes.shape[1]

    print(all_volumes.shape)
    

    if deformation_map is None:
        # print("Here")
        mapx_base, mapy_base = np.meshgrid(np.arange(all_volumes.shape[-1]), np.arange(all_volumes.shape[-2]))
        mapx_base=np.tile(mapx_base,reps=(nb_slices,1,1))
        mapy_base = np.tile(mapy_base, reps=(nb_slices, 1, 1))
        # print("Here 2")
    else:
        #print("Applying existing deformation map")
        mapx_base=deformation_map[0,gr]
        mapy_base=deformation_map[1,gr]

    #print("mapx_base shape: {}".format(mapx_base.shape))
    #print("mapy_base shape: {}".format(mapy_base.shape))
    #print(mapx_base.shape)
    while curr_gr>0:
        # print(all_volumes[curr_gr-1].shape)
        # print(moving_volume.shape)

        input=np.stack([np.pad(all_volumes[curr_gr-1],pad_amount,mode="constant"),moving_volume],axis=0)
        # print(input.shape)
        x_val_fixed,x_val_moving=format_input_voxelmorph(input,((0,0),(0,0),(0,0)),sl_down=0,sl_top=nb_slices,exclude_zero_slices=False)
        val_input=[x_val_moving[...,None],x_val_fixed[...,None]]
        #print(val_input.shape)


        val_pred=vxm_model.predict(val_input)
        moving_volume=val_pred[0][:,:,:,0]
        #print(val_pred[1][:,:,:].shape)
        
        mapx_base=mapx_base+unpad(val_pred[1][:,:,:,1],pad_amount)
        mapy_base=mapy_base+unpad(val_pred[1][:,:,:,0],pad_amount)

        curr_gr=curr_gr-1
    print("Moving volume shape : {}".format(moving_volume.shape))

    if gr==0:
        moving_volume/=np.max(moving_volume,axis=(1,2),keepdims=True)
    unpadded_moving_volume=unpad(moving_volume,pad_amount)
    # print("Unpadded Moving volume shape : {}".format(unpadded_moving_volume.shape))
    # print(mapx_base.shape)
    # print(mapy_base.shape)
    print("Norm unpadded_moving_volume : {}".format(np.linalg.norm(unpadded_moving_volume)))
    print("Max unpadded_moving_volume : {}".format(np.max(unpadded_moving_volume)))
    print("Min unpadded_moving_volume : {}".format(np.min(unpadded_moving_volume)))


    
          
    return unpadded_moving_volume,mapx_base,mapy_base


# def register_motionbin_3D(vxm_model,all_volumes,gr,pad_amount,deformation_map=None,full_shape=None):
#     curr_gr=gr
#     moving_volume=np.pad(all_volumes[curr_gr],pad_amount,mode="constant")
#     nb_slices=all_volumes.shape[1]

    


#     print(all_volumes.shape)
#     print(full_shape)
#     if deformation_map is None:
#         # mapz_base,mapx_base, mapy_base = np.meshgrid(np.arange(all_volumes.shape[1]),np.arange(all_volumes.shape[2]), np.arange(all_volumes.shape[3]),indexing="ij")
#         Dz, Dx, Dy = full_shape

#         mapz_base, mapx_base, mapy_base = np.meshgrid(
#             np.arange(Dz), np.arange(Dx), np.arange(Dy), indexing="ij"
#         )
            
#     else:
#         #print("Applying existing deformation map")
#         mapz_base=deformation_map[0,gr]
#         mapx_base=deformation_map[1,gr]
#         mapy_base=deformation_map[2,gr]

#     # print("mapx_base shape: {}".format(mapx_base.shape))
#     # print("mapy_base shape: {}".format(mapy_base.shape))
#     # print("mapz_base shape: {}".format(mapz_base.shape))
    
#     while curr_gr>0:
#         # print("curr_gr {}".format(curr_gr))
#         # print("all_volumes[curr_gr-1].shape {}".format(all_volumes[curr_gr-1].shape))
#         # print("moving_volume.shape {}".format(moving_volume.shape))
#         input=np.stack([np.pad(all_volumes[curr_gr-1],pad_amount,mode="constant"),moving_volume],axis=0)
#         #print(input.shape)
#         x_val_fixed,x_val_moving=format_input_voxelmorph_3D(input,pad=False,sl_down=0,sl_top=nb_slices)
#         #print(x_val_fixed.shape)
#         val_input=[x_val_moving,x_val_fixed]
#         #print(val_input.shape)

#         val_pred=vxm_model.predict(val_input)
#         moving_volume=val_pred[0][0,:,:,:,0]
#         flow=val_pred[1][0]
#         print(f"flow gr {gr} curr_gr {curr_gr} shape {flow.shape}")
#         np.save(f"flow_gr{gr}curr_gr{curr_gr}.npy",flow)
#         flow=upsample_flow_3D(flow,moving_volume.shape)
#         # print("Moving volume shape after registration {}".format(moving_volume.shape))
#         # print("flowshape {}".format(flow.shape))
#         flow[...,0]=unpad(flow[...,0],pad_amount)
#         flow[...,1]=unpad(flow[...,1],pad_amount)
#         flow[...,2]=unpad(flow[...,2],pad_amount)
#         flow=upsample_flow_3D(flow,full_shape)
#         # print("flowshape after upsample and unpad {}".format(flow.shape))
#         flow=warp_flow_3D(flow,mapz_base,mapx_base,mapy_base)

#         mapz_base=mapz_base+flow[...,0]
#         mapx_base=mapx_base+flow[...,1]
#         mapy_base=mapy_base+flow[...,2]
#         curr_gr=curr_gr-1
        
#     #print("Moving volume shape {}:".format(moving_volume.shape))
#     return unpad(moving_volume,pad_amount),mapz_base,mapx_base,mapy_base

def register_motionbin_3D(vxm_model, all_volumes, gr, pad_amount,
                         deformation_map=None, full_shape=None):

    curr_gr = gr
    moving_volume = np.pad(all_volumes[curr_gr], pad_amount, mode="constant")
    nb_slices = all_volumes.shape[1]

    Dz, Dx, Dy = full_shape

    # --- initialize total flow ---
    if deformation_map is None:
        total_flow = np.zeros((Dz, Dx, Dy, 3), dtype=np.float32)
    else:
        total_flow = deformation_map[gr].copy()  # now directly a flow

    while curr_gr > 0:

        input = np.stack([
            np.pad(all_volumes[curr_gr-1], pad_amount, mode="constant"),
            moving_volume
        ], axis=0)

        x_val_fixed, x_val_moving = format_input_voxelmorph_3D(
            input, pad=False, sl_down=0, sl_top=nb_slices
        )

        val_pred = vxm_model.predict([x_val_moving, x_val_fixed])

        moving_volume = val_pred[0][0, :, :, :, 0]
        flow = val_pred[1][0]
        print(f"Flow vxm output shape {flow.shape}")
        
        print(f"volume vxm output shape {moving_volume.shape}")
        # --- process flow ---
        flow = upsample_flow_3D(flow, moving_volume.shape)

        flow[..., 0] = unpad(flow[..., 0], pad_amount)
        flow[..., 1] = unpad(flow[..., 1], pad_amount)
        flow[..., 2] = unpad(flow[..., 2], pad_amount)

        print(f"Flow shape after unpadding {flow.shape}")

        flow = upsample_flow_3D(flow, full_shape)

        print(f"Flow shape after upsampling {flow.shape}")

        # --- IMPORTANT: compose flows ---
        # flow = warp_flow_3D(
        #     flow,
        #     total_flow[..., 0] + np.arange(Dz)[:, None, None],
        #     total_flow[..., 1] + np.arange(Dx)[None, :, None],
        #     total_flow[..., 2] + np.arange(Dy)[None, None, :]
        # )

        coords_z = np.arange(Dz)[:, None, None] + total_flow[..., 0]
        coords_x = np.arange(Dx)[None, :, None] + total_flow[..., 1]
        coords_y = np.arange(Dy)[None, None, :] + total_flow[..., 2]

        total_flow = warp_flow_3D(flow, coords_z, coords_x, coords_y) + total_flow


        # coords_z = np.arange(Dz)[:, None, None] + flow[..., 0]
        # coords_x = np.arange(Dx)[None, :, None] + flow[..., 1]
        # coords_y = np.arange(Dy)[None, None, :] + flow[..., 2]

        # warped_total = warp_flow_3D(total_flow, coords_z, coords_x, coords_y)
        # total_flow   = warped_total + flow

        # total_flow = total_flow + flow

        curr_gr -= 1

    return unpad(moving_volume, pad_amount), total_flow


def scheduler(epoch, lr,decay=0.005,min_lr=None,decay_start=20):
  if epoch < decay_start:
    return lr
  else:
    if min_lr is None:
        return lr * tf.math.exp(-decay)
    else:
        return np.maximum(lr * tf.math.exp(-decay),min_lr)



toolbox = Toolbox("script_VoxelMorph_machines.", description="Volume registration with Voxelmorph")
toolbox.add_program("train_voxelmorph", train_voxelmorph)
toolbox.add_program("train_voxelmorph_3D", train_voxelmorph_3D)
toolbox.add_program("train_voxelmorph_torchio", train_voxelmorph_torchio)
toolbox.add_program("train_voxelmorph_torchvision", train_voxelmorph_torchvision)
toolbox.add_program("train_voxelmorph_torchvision_multiple_patients", train_voxelmorph_torchvision_multiple_patients)
toolbox.add_program("evaluate_model", evaluate_model)
toolbox.add_program("register_allbins_to_baseline", register_allbins_to_baseline)
toolbox.add_program("register_allbins_to_baseline_3D", register_allbins_to_baseline_3D)
toolbox.add_program("apply_deformation_map", apply_deformation_map)
toolbox.add_program("plot_deformation_flow", plot_deformation_flow)

if __name__ == "__main__":
    toolbox.cli()


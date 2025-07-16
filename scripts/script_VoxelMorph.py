# imports
import os, sys

# third party imports
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
from copy import copy

print(tf.config.experimental.list_physical_devices("GPU"))

# local imports
import voxelmorph as vxm
import neurite as ne
from mutools import io
import matplotlib.pyplot as plt

# You should most often have this import together with all other imports at the top,
# but we include here here explicitly to show where data comes from

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




l=1
it=5

filename_template="./test_volume_comp_v2_allgroups_CF_iterative_2Dplus1_MRF_map_mubins1_muTV1_gr*_it{}_l{}_matchedvolumes.mha".format(it,l)
filename_template=str.replace(filename_template,"*","{}")

gr = 0
all_volumes = []
for gr in range(4):
    volume = io.read(filename_template.format(gr))
    all_volumes.append(volume)

all_volumes = np.array(all_volumes)


sl=45
gr=1
plt.imshow(all_volumes[gr,sl],cmap="gray")
plt.figure()
plt.imshow(all_volumes[gr+1,sl],cmap="gray")


#gr=1
fixed_volume=[]
moving_volume=[]

for gr in range(3):


    fixed_volume.append(all_volumes[gr,5:-5])
    moving_volume.append(all_volumes[gr+1,5:-5])

fixed_volume=np.array(fixed_volume)
moving_volume=np.array(moving_volume)

fixed_volume=fixed_volume.reshape(-1,fixed_volume.shape[-2],fixed_volume.shape[-1])
moving_volume=moving_volume.reshape(-1,moving_volume.shape[-2],moving_volume.shape[-1])

fixed_volume/=np.max(fixed_volume,axis=(1,2),keepdims=True)
moving_volume/=np.max(moving_volume,axis=(1,2),keepdims=True)

pad_amount = ((0,0),(56,56), (56,56))

# fix data
fixed_volume = np.pad(fixed_volume, pad_amount, 'constant')
moving_volume = np.pad(moving_volume, pad_amount, 'constant')

# configure unet input shape (concatenation of moving and fixed images)
ndim = 2
unet_input_features = 2
inshape = fixed_volume.shape[1:]

# configure unet features
nb_features = [
    [256, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

nb_features = [
    [32, 64, 128, 128],         # encoder features
    [128, 128, 64, 64, 32, 16]  # decoder features
]

vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

sl=40
#x_train_fixed=np.expand_dims(fixed_volume[sl],axis=0)
#x_train_moving=np.expand_dims(moving_volume[sl],axis=0)

# x_train_fixed=fixed_volume[35:45]
# x_train_moving=moving_volume[35:45]

x_train_fixed=copy(fixed_volume)
x_train_moving=copy(moving_volume)


x_train_fixed[x_train_fixed>0.2]=0.2
x_train_moving[x_train_moving>0.2]=0.2

x_train_fixed/=np.max(x_train_fixed,axis=(1,2),keepdims=True)
x_train_moving/=np.max(x_train_moving,axis=(1,2),keepdims=True)


batch_size=16
train_generator = vxm_data_generator(x_train_fixed,x_train_moving,batch_size=batch_size)



# # let's test it
# train_generator = vxm_data_generator(x_train_fixed,x_train_moving)
# in_sample, out_sample = next(train_generator)
#
# # visualize
# images = [img[0, :, :, 0] for img in in_sample + out_sample]
# titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
# ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);

nb_examples=x_train_fixed.shape[0]

nb_epochs = 200
steps_per_epoch = int(nb_examples/batch_size)+1
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)

vxm_model.save_weights("vxm_model_ss_large_whole_pop_weights.h5")

vxm_model_bis=vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
vxm_model_bis.load_weights("vxm_model_large_whole_pop_weights.h5")

import matplotlib.pyplot as plt

def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

plot_history(hist)

x_val_fixed=fixed_volume[45:50]
x_val_moving=moving_volume[45:50]

x_val_fixed[x_val_fixed>0.2]=0.2
x_val_moving[x_val_moving>0.2]=0.2

x_val_fixed/=np.max(x_val_fixed,axis=(1,2),keepdims=True)
x_val_moving/=np.max(x_val_moving,axis=(1,2),keepdims=True)

val_generator = vxm_data_generator(x_val_fixed,x_val_moving, batch_size = 1)
val_input, _ = next(val_generator)
val_input, _ = next(val_generator)
val_input, _ = next(val_generator)
val_input, _ = next(val_generator)
#val_pred = vxm_model.predict(val_input)
val_pred = vxm_model_bis.predict(val_input)


# visualize
images = [img[0, :, :, 0] for img in val_input + val_pred]
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

plt.close("all")

ne.plot.flow([val_pred[1].squeeze()], width=10)


images_array=np.array(images)

plt.close("all")
from utils_mrf import *
animate_images(images_array[[2,1]])


animate_images(images_array[[0,1]])

plt.figure()
plt.imshow(images_array[-1])


plt.figure()
plt.imshow(images_array[1])
plt.imshow(images_array[2])



test=images_array[2]
test[test>0.2]=0.2
plt.imshow(test)






# imports
import os, sys

# third party imports
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

# local imports
import voxelmorph as vxm
import neurite as ne
from mutools import io
import matplotlib.pyplot as plt

# You should most often have this import together with all other imports at the top,
# but we include here here explicitly to show where data comes from

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
        print(idx1)
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




l=1
it=5

filename_template="./test_volume_comp_v2_allgroups_CF_iterative_2Dplus1_MRF_map_mubins1_muTV1_gr*_it{}_l{}_matchedvolumes.mha".format(it,l)
filename_template=str.replace(filename_template,"*","{}")

gr = 0
all_volumes = []
for gr in range(4):
    volume = io.read(filename_template.format(gr))
    all_volumes.append(volume)

all_volumes = np.array(all_volumes)


sl=45
gr=1


gr=1
fixed_volume=all_volumes[gr,5:-5]
moving_volume=all_volumes[gr+1,5:-5]
fixed_volume/=np.max(fixed_volume,axis=(1,2),keepdims=True)
moving_volume/=np.max(moving_volume,axis=(1,2),keepdims=True)

pad_amount = ((0,0),(56,56), (56,56))

# fix data
fixed_volume = np.pad(fixed_volume, pad_amount, 'constant')
moving_volume = np.pad(moving_volume, pad_amount, 'constant')

# configure unet input shape (concatenation of moving and fixed images)
ndim = 2
unet_input_features = 2
inshape = fixed_volume.shape[1:]

# configure unet features
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

nb_features = [
    [32, 64, 128, 128],         # encoder features
    [128, 128, 64, 64, 32, 16]  # decoder features
]

vxm_model_bis=vxm.networks.VxmDense(inshape, nb_features, int_steps=0)





#vxm_model_bis.load_weights("vxm_model_large_whole_pop_weights.h5")



x_val_fixed=copy(fixed_volume)[40:45]
x_val_moving=copy(moving_volume)[40:45]


x_val_fixed[x_val_fixed>0.2]=0.2
x_val_moving[x_val_moving>0.2]=0.2

x_val_fixed/=np.max(x_val_fixed,axis=(1,2),keepdims=True)
x_val_moving/=np.max(x_val_moving,axis=(1,2),keepdims=True)

val_generator = vxm_data_generator(x_val_fixed,x_val_moving, batch_size = 1)
val_input, _ = next(val_generator)
val_input, _ = next(val_generator)
val_input, _ = next(val_generator)
val_input, _ = next(val_generator)
#val_pred = vxm_model.predict(val_input)
val_pred = vxm_model_bis.predict(val_input)

volume_all_phases=[val_input[1],val_input[0]]
volume_all_phases_mc=[val_input[1],val_pred[0]]

volume_all_phases=np.squeeze(np.array(volume_all_phases))
volume_all_phases_mc=np.squeeze(np.array(volume_all_phases_mc))


moving_image=np.concatenate([volume_all_phases,volume_all_phases[1:-1][::-1]],axis=0)
moving_image_mc=np.concatenate([volume_all_phases_mc,volume_all_phases_mc[1:-1][::-1]],axis=0)



from PIL import Image
gif=[]
volume_for_gif = np.abs(moving_image)
for i in range(volume_for_gif.shape[0]):
    img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    img=img.convert("P")
    gif.append(img)

filename_gif = "moving_l1_sl{}.gif".format(sl)
gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)



from PIL import Image
gif=[]
volume_for_gif = np.abs(moving_image_mc)
for i in range(volume_for_gif.shape[0]):
    img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    img=img.convert("P")
    gif.append(img)

filename_gif = "moving_corrected_l1_sl{}.gif".format(sl)
gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)




# visualize
images = [img[0, :, :, 0] for img in val_input + val_pred]
images = images + [val_pred[1][0,:,:,1]]
titles = ['moving', 'fixed', 'moved', 'flow x',"flow y"]
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)


images_array=np.array(images)
from utils_mrf import *
animate_images(images_array[[0,1]])
animate_images(images_array[[2,1]])

from mutools import io
io.savemat(val_pred[1])


import SimpleITK as sitk

dx=1
dy=1
dz=5


gr=1
fixed_volume=all_volumes[gr,5:-5]
moving_volume=all_volumes[gr+1,5:-5]
fixed_volume/=np.max(fixed_volume,axis=(1,2),keepdims=True)
moving_volume/=np.max(moving_volume,axis=(1,2),keepdims=True)

pad_amount = ((0,0),(56,56), (56,56))

# fix data
fixed_volume = np.pad(fixed_volume, pad_amount, 'constant')
moving_volume = np.pad(moving_volume, pad_amount, 'constant')


x_val_fixed=copy(fixed_volume)
x_val_moving=copy(moving_volume)


x_val_fixed[x_val_fixed>0.2]=0.2
x_val_moving[x_val_moving>0.2]=0.2

x_val_fixed/=np.max(x_val_fixed,axis=(1,2),keepdims=True)
x_val_moving/=np.max(x_val_moving,axis=(1,2),keepdims=True)


val_input=[x_val_moving[...,None],x_val_fixed[...,None]]
val_pred=vxm_model_bis.predict(val_input)

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

sitk.WriteImage(field,"displacement_field_voxelmorph_whole_basegr{}.nii".format(gr))
sitk.WriteImage(moving_3D,"moving_voxelmorph_whole_basegr{}.mha".format(gr))
sitk.WriteImage(fixed_3D,"fixed_voxelmorph_whole_basegr{}.mha".format(gr))
sitk.WriteImage(moved_3D,"moved_voxelmorph_whole_basegr{}.mha".format(gr))

##Whole volume

from mutools import io
sl=45
volume_all_phases=[]
volume_all_phases_mc=[]




moving_image=np.concatenate([volume_all_phases,volume_all_phases[1:-1][::-1]],axis=0)
moving_image_mc=np.concatenate([volume_all_phases_mc,volume_all_phases_mc[1:-1][::-1]],axis=0)




from PIL import Image
gif=[]
volume_for_gif = np.abs(moving_image)
for i in range(volume_for_gif.shape[0]):
    img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    img=img.convert("P")
    gif.append(img)

filename_gif = "moving_l1_sl{}.gif".format(sl)
gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)



from PIL import Image
gif=[]
volume_for_gif = np.abs(moving_image_mc)
for i in range(volume_for_gif.shape[0]):
    img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    img=img.convert("P")
    gif.append(img)

filename_gif = "moving_corrected_l1_sl{}.gif".format(sl)
gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)




sl=45
volume_all_phases=all_volumes[:,5:-5][:,sl:(sl+1)]
pad_amount_all_phases = ((0,0),(0,0),(56,56), (56,56))
volume_all_phases=np.pad(volume_all_phases, pad_amount_all_phases, 'constant')

volume_all_phases_mc=[volume_all_phases[1][...,None]]

for gr in range(1,3):
    curr_fixed_volume=volume_all_phases[1]
    curr_moving_volume=volume_all_phases[gr+1]
    x_val_fixed = copy(curr_fixed_volume)
    x_val_moving = copy(curr_moving_volume)

    #x_val_fixed[x_val_fixed > 0.2] = 0.2
    #x_val_moving[x_val_moving > 0.2] = 0.2

    x_val_fixed /= np.max(x_val_fixed, axis=(1, 2), keepdims=True)
    x_val_moving /= np.max(x_val_moving, axis=(1, 2), keepdims=True)
    val_pred = vxm_model_bis.predict([x_val_moving[...,None],x_val_fixed[...,None]])
    volume_all_phases_mc.append(val_pred[0])

volume_all_phases_mc=np.array(volume_all_phases_mc)

volume_all_phases=np.squeeze(volume_all_phases)[[1,2,3]]
volume_all_phases_mc=np.squeeze(volume_all_phases_mc)


moving_image=np.concatenate([volume_all_phases,volume_all_phases[1:-1][::-1]],axis=0)
moving_image_mc=np.concatenate([volume_all_phases_mc,volume_all_phases_mc[1:-1][::-1]],axis=0)

animate_images(volume_all_phases)

animate_images(volume_all_phases_mc)



from PIL import Image
gif=[]
volume_for_gif = np.abs(moving_image)
for i in range(volume_for_gif.shape[0]):
    img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    img=img.convert("P")
    gif.append(img)

filename_gif = "moving_l1_sl{}.gif".format(sl)
gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)



from PIL import Image
gif=[]
volume_for_gif = np.abs(moving_image_mc)
for i in range(volume_for_gif.shape[0]):
    img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    img=img.convert("P")
    gif.append(img)

filename_gif = "moving_corrected_l1_sl{}.gif".format(sl)
gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)





# imports
import os, sys

# third party imports
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
from copy import copy

print(tf.config.experimental.list_physical_devices("GPU"))

# local imports
import voxelmorph as vxm
import neurite as ne
from mutools import io
import matplotlib.pyplot as plt

# You should most often have this import together with all other imports at the top,
# but we include here here explicitly to show where data comes from

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




all_volumes = np.load("./data/InVivo/3D/patient.009.v2/meas_MID00148_FID57318_raFin_3D_tra_1x1x5mm_FULL_new_allvolumes_pca16.npy")


fixed_volume=[]
moving_volume=[]

for gr in range(4):


    fixed_volume.append(all_volumes[gr,5:-5])
    moving_volume.append(all_volumes[gr+1,5:-5])

fixed_volume=np.array(fixed_volume)
moving_volume=np.array(moving_volume)

fixed_volume=fixed_volume.reshape(-1,fixed_volume.shape[-2],fixed_volume.shape[-1])
moving_volume=moving_volume.reshape(-1,moving_volume.shape[-2],moving_volume.shape[-1])

fixed_volume/=np.max(fixed_volume,axis=(1,2),keepdims=True)
moving_volume/=np.max(moving_volume,axis=(1,2),keepdims=True)

pad_amount = ((0,0),(56,56), (56,56))
#pad_amount = ((0,0),(0,0), (0,0))


# fix data
fixed_volume = np.pad(fixed_volume, pad_amount, 'constant')
moving_volume = np.pad(moving_volume, pad_amount, 'constant')

# configure unet input shape (concatenation of moving and fixed images)
ndim = 2
unet_input_features = 2
inshape = fixed_volume.shape[1:]

# configure unet features
nb_features = [
    [256, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

nb_features = [
    [32, 64, 128, 128],         # encoder features
    [128, 128, 64, 64, 32, 16]  # decoder features
]

vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

#x_train_fixed=np.expand_dims(fixed_volume[sl],axis=0)
#x_train_moving=np.expand_dims(moving_volume[sl],axis=0)

# x_train_fixed=fixed_volume[35:45]
# x_train_moving=moving_volume[35:45]

x_train_fixed=copy(fixed_volume)
x_train_moving=copy(moving_volume)


#x_train_fixed[x_train_fixed>0.1]=0.1
#x_train_moving[x_train_moving>0.1]=0.1

x_train_fixed/=np.max(x_train_fixed,axis=(1,2),keepdims=True)
x_train_moving/=np.max(x_train_moving,axis=(1,2),keepdims=True)

plt.close("all")
num_samples=x_train_fixed.shape[0]
plt.figure()
samp=np.random.randint(num_samples)
samp=152
plt.imshow(x_train_fixed[samp])


batch_size=16
train_generator = vxm_data_generator(x_train_fixed,x_train_moving,batch_size=batch_size)



# # let's test it
# train_generator = vxm_data_generator(x_train_fixed,x_train_moving)
# in_sample, out_sample = next(train_generator)
#
# # visualize
# images = [img[0, :, :, 0] for img in in_sample + out_sample]
# titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
# ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);

nb_examples=x_train_fixed.shape[0]

nb_epochs = 200
steps_per_epoch = int(nb_examples/batch_size)+1
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)

vxm_model.save_weights("vxm_model_ss_large_whole_pop_weights.h5")




import SimpleITK as sitk

dx=1
dy=1
dz=5


gr=3
fixed_volume=all_volumes[gr,5:-5,:,:]
moving_volume=all_volumes[gr+1,5:-5,:,:]
fixed_volume/=np.max(fixed_volume,axis=(1,2),keepdims=True)
moving_volume/=np.max(moving_volume,axis=(1,2),keepdims=True)

pad_amount = ((0,0),(0,0), (0,0))
pad_amount = ((0,0),(56,56), (56,56))

# fix data
fixed_volume = np.pad(fixed_volume, pad_amount, 'constant')
moving_volume = np.pad(moving_volume, pad_amount, 'constant')


x_val_fixed=copy(fixed_volume)
x_val_moving=copy(moving_volume)


#x_val_fixed[x_val_fixed>0.2]=0.2
#x_val_moving[x_val_moving>0.2]=0.2

x_val_fixed/=np.max(x_val_fixed,axis=(1,2),keepdims=True)
x_val_moving/=np.max(x_val_moving,axis=(1,2),keepdims=True)


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

sitk.WriteImage(field,"displacement_field_voxelmorph_whole_basegr{}.nii".format(gr))
sitk.WriteImage(moving_3D,"moving_voxelmorph_whole_basegr{}.mha".format(gr))
sitk.WriteImage(fixed_3D,"fixed_voxelmorph_whole_basegr{}.mha".format(gr))
sitk.WriteImage(moved_3D,"moved_voxelmorph_whole_basegr{}.mha".format(gr))
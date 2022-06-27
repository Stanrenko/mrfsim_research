
try:
    import matplotlib.pyplot as plt
except:
    pass
import matplotlib.animation as animation
from mutools.optim.dictsearch import dictsearch,groupmatch
from functools import reduce
from mrfsim import makevol,parse_options,groupby,load_data
import numpy as np
import finufft
from scipy import ndimage
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.spatial import Voronoi,ConvexHull
from Transformers import PCAComplex
try:
    import freud
except:
    pass
try:

    import seaborn as sns
except:
    pass 
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import pandas as pd
import itertools
from sigpy.mri import spiral
import cv2
import pywt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.metrics import structural_similarity as ssim
import collections
import gc
try:
    import pycuda
    import pycuda.autoinit
    from pycuda.gpuarray import GPUArray, to_gpu
    from cufinufft import cufinufft

except:
    pass

from copy import copy
import psutil
from datetime import datetime

def read_mrf_dict(dict_file ,FF_list ,aggregate_components=True):

    mrfdict = dictsearch.Dictionary()
    mrfdict.load(dict_file, force=True)

    if aggregate_components :
        epg_water = mrfdict.values[: ,: ,0]
        epg_fat = mrfdict.values[: ,: ,1]
        ff = np.zeros(mrfdict.values.shape[:-1 ] +(len(FF_list),))
        ff_matrix =np.tile(np.array(FF_list) ,ff.shape[:-1 ] +(1,))

        water_signal =np.expand_dims(mrfdict.values[: ,: ,0] ,axis=-1 ) *(1-ff_matrix)
        fat_signal =np.expand_dims(mrfdict.values[: ,: ,1] ,axis=-1 ) *(ff_matrix)

        signal =water_signal +fat_signal

        signal_reshaped =np.moveaxis(signal ,-1 ,-2)
        signal_reshaped =signal_reshaped.reshape((-1 ,signal_reshaped.shape[-1]))

        keys_with_ff = list(itertools.product(mrfdict.keys, FF_list))
        keys_with_ff = [(*res, f) for res, f in keys_with_ff]

        return keys_with_ff,signal_reshaped

    else:
        return mrfdict.keys,mrfdict.values



def animate_images(images_series,interval=200,metric=np.abs,cmap=None):
    fig, ax = plt.subplots()
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i, image in enumerate(images_series):

        im = ax.imshow(metric(image), animated=True,cmap=cmap)
        if i == 0:
            ax.imshow(metric(image),cmap=cmap)  # show an initial one first
        ims.append([im])

    return animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=10 * interval)

def animate_multiple_images(images_series,images_series_rebuilt,interval=200,cmap=None):
    nb_frames=len(images_series)
    fig, ax = plt.subplots()
    fig_rebuilt, ax_rebuilt = plt.subplots()
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    ims_rebuilt = []
    for i in range(nb_frames):
        im = ax.imshow(np.abs(images_series[i]), animated=True,cmap=cmap)
        if i == 0:
            ax.imshow(np.abs(images_series[i]),cmap=cmap)  # show an initial one first
        ims.append([im])

        im_rebuilt = ax_rebuilt.imshow(np.abs(images_series_rebuilt[i]), animated=True,cmap=cmap)
        if i == 0:
            ax_rebuilt.imshow(np.abs(images_series_rebuilt[i]),cmap=cmap)  # show an initial one first
        ims_rebuilt.append([im_rebuilt])

    return animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=10 * interval),animation.ArtistAnimation(fig_rebuilt, ims_rebuilt, interval=interval, blit=True,
                                    repeat_delay=10 * interval),

def cartesian_traj_2D(npoint_x,npoint_y,k_max=np.pi):
    #kx = -k_max + np.arange(npoint_x) * 2 * k_max / (npoint_x - 1)
    #ky = -k_max + np.arange(npoint_y) * 2 * k_max / (npoint_y - 1)
    kx = np.arange(-k_max, k_max, 2 * k_max / npoint_x) + k_max / npoint_x
    ky = np.arange(-k_max, k_max, 2 * k_max / npoint_y) + k_max / npoint_y

    KX, KY = np.meshgrid(kx, ky)
    return np.stack([KX.flatten(), KY.flatten()], axis=-1)

def cartesian_traj_3D(total_nspoke, npoint_x, npoint_y, nb_slices, undersampling_factor=4):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    base_traj=cartesian_traj_2D(npoint_x,npoint_y)
    traj = np.tile(base_traj, (total_nspoke, 1, 1))

    k_z = np.zeros((timesteps, nb_rep))
    all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    k_z[0, :] = all_slices[::undersampling_factor]
    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])




def radial_golden_angle_traj(total_nspoke,npoint,k_max=np.pi):
    golden_angle=111.246*np.pi/180
    #base_spoke = np.arange(-k_max, k_max, 2 * k_max / npoint, dtype=np.complex_)
    base_spoke = -k_max+np.arange(npoint)*2*k_max/(npoint-1)
    all_rotations = np.exp(1j * np.arange(total_nspoke) * golden_angle)
    all_spokes = np.matmul(np.diag(all_rotations), np.repeat(base_spoke.reshape(1, -1), total_nspoke, axis=0))
    return all_spokes


def radial_golden_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
    #traj = np.reshape(all_spokes, (-1, nspoke * npoint))

    k_z = np.zeros((timesteps, nb_rep))
    all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    k_z[0, :] = all_slices[::undersampling_factor]
    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)
    traj = np.expand_dims(all_spokes, axis=-2)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    # k_z = np.reshape(k_z, (timesteps, -1))
    # traj = np.reshape(traj, (timesteps, -1))

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])

def radial_golden_angle_traj_3D_incoherent(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4,mode="old"):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
    golden_angle = 111.246 * np.pi / 180
    if mode=="old":
        all_rotations = np.exp(1j * np.arange(nb_rep) * total_nspoke * golden_angle)
    elif mode=="new":
        all_rotations = np.exp(1j * np.arange(nb_rep) * golden_angle)
    else:
        raise ValueError("Unknown value for mode")
    all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_rep, axis=1)
    traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes

    k_z = np.zeros((timesteps, nb_rep))
    all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    k_z[0, :] = all_slices[::undersampling_factor]
    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])


def radial_golden_angle_traj_random_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4,frac_center=0.25,mode="old",incoherent=True):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
        #traj = np.reshape(all_spokes, (-1, nspoke * npoint))

    if incoherent:
        golden_angle = 111.246 * np.pi / 180
        if mode=="old":
            all_rotations = np.exp(1j * np.arange(nb_rep) * total_nspoke * golden_angle)
        elif mode=="new":
            all_rotations = np.exp(1j * np.arange(nb_rep) * golden_angle)
        else:
            raise ValueError("Unknown value for mode")
        all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_rep, axis=1)
        traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes
    else:
        traj = np.expand_dims(all_spokes, axis=-2)

    k_z = np.zeros((timesteps, nb_rep))
    all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    kz_center=all_slices[(int(nb_slices/2)+np.array(range(int(-frac_center*nb_rep/2),int(frac_center*nb_rep/2),1)))]
    kz_border = [k for k in all_slices if k not in kz_center]
    nb_border=nb_rep-len(kz_center)
    for j in range(k_z.shape[0]):
        k_z[j, :] = np.sort(np.concatenate([np.random.choice(kz_border,size=int(nb_border),replace=False),kz_center]))

    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)

    k_z, traj = np.broadcast_arrays(k_z, traj)

    # k_z = np.reshape(k_z, (timesteps, -1))
    # traj = np.reshape(traj, (timesteps, -1))

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])

def spiral_golden_angle_traj(total_spiral,fov, N, f_sampling, R, ninterleaves, alpha, gm, sm):
    golden_angle = 111.246 * np.pi / 180
    base_spiral = spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
    base_spiral = base_spiral[:,0]+1j*base_spiral[:,1]
    all_rotations = np.exp(1j * np.arange(total_spiral) * golden_angle)
    all_spirals = np.matmul(np.diag(all_rotations), np.repeat(base_spiral.reshape(1, -1), total_spiral, axis=0))
    return all_spirals

def spiral_golden_angle_traj_v2(total_spiral,nspiral,fov, N, f_sampling, R, ninterleaves, alpha, gm, sm):
    golden_angle = 111.246 * np.pi / 180
    angle = 2*np.pi/nspiral
    base_spiral = spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
    base_spiral = base_spiral[:,0]+1j*base_spiral[:,1]
    all_rotations = np.exp(1j * np.arange(total_spiral) * angle)
    all_spirals = np.matmul(np.diag(all_rotations), np.repeat(base_spiral.reshape(1, -1), total_spiral, axis=0))

    disk_rotations = np.exp(1j * np.arange(int(total_spiral/nspiral)) * golden_angle)
    disk_rotations = np.repeat(disk_rotations,nspiral)
    all_spirals = np.matmul(np.diag(disk_rotations),all_spirals )

    return all_spirals

def create_random_map(list_params,region_size,size,mask):
    basis = np.random.choice(list_params,(int(size[0]/region_size),int(size[1]/region_size)))
    map = np.repeat(np.repeat(basis, region_size, axis=1), region_size, axis=0) * mask
    return map

def compare_patterns(pixel_number,images_1,images_2,title_1="image_1",title_2="image_2"):

    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.plot(np.real(images_1[:,pixel_number[0],pixel_number[1]]),label=title_1+" - real part")
    ax1.plot(np.real(images_2[:, pixel_number[0],pixel_number[1]]), label=title_2+" - real part")
    ax1.legend()
    ax2.plot(np.imag(images_1[:, pixel_number[0], pixel_number[1]]),
             label=title_1+" - imaginary part")
    ax2.plot(np.imag(images_2[:, pixel_number[0], pixel_number[1]]),
             label=title_2+" - imaginary part")
    ax2.legend()

    ax3.plot(np.abs(images_1[:, pixel_number[0], pixel_number[1]]),
             label=title_1+" - norm")
    ax3.plot(np.abs(images_2[:, pixel_number[0], pixel_number[1]]),
             label=title_2+" - norm")
    ax3.legend()

    plt.show()


def translation_breathing(t,direction,T=4000,frac_expiration=0.7):
    def base_pattern(t):
        lambda1=5/(frac_expiration*T)
        lambda2=20/((1-frac_expiration)*T)

        return ((1-np.exp(-lambda1*t))*((t<(frac_expiration*T))*1) + ((t>=(frac_expiration*T))*1)*((1-np.exp(-lambda1*frac_expiration*T))* np.exp(-lambda2*t)/np.exp(-lambda2*frac_expiration*T)))*direction

    return base_pattern(t-(t/T).astype(int)*T)

def find_klargest_freq(ft, k=1, remove_central_peak=True):
    n_max = len(ft) - 1
    n_min = 0
    if remove_central_peak:
        # Removing the central peak in fourier transform (corresponds roughly to PSF)
        while (ft[n_max - 1] <= ft[n_max]):
            n_max = n_max - 1

        while (ft[n_min + 1] <= ft[n_min]):
            n_min = n_min + 1

    freq_image = np.argsort((ft)[n_min:n_max])[-k]

    if freq_image + n_min < 175 / 2:
        freq_image = freq_image + n_min
    else:
        freq_image = len(ft) - 1 - (freq_image + n_min)

    return freq_image

def SearchMrf(kdata,trajectory, dictfile, niter, method, metric, shape,density_adj=False, setup_opts={}, search_opts= {}):
    """ Estimate parameters """
    # constants
    shape = tuple(shape)

    nspoke=trajectory.paramDict["nspoke"]
    npoint=trajectory.paramDict["npoint"]
    traj = trajectory.get_traj()

    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
    else:
        density=np.ones(npoint)

    # printer(f"Load dictionary: {dictfile}")
    mrfdict = dictsearch.Dictionary()
    mrfdict.load(dictfile, force=True)

    print(f"Init solver ({method})")
    if method == "brute":
        solver = dictsearch.DictSearch()
        setupopts = {"pca": True, **parse_options(setup_opts)}
        searchopts = {"metric": metric, "parallel": True, **parse_options(search_opts)}
    elif method == "group":
        solver = groupmatch.GroupMatch()
        setupopts = {"pca": True, "group_ratio": 0.05, **parse_options(setup_opts)}
        searchopts = {"metric": metric, "parallel": True, "group_threshold": 1e-1, **parse_options(search_opts)}
    solver.setup(mrfdict.keys, mrfdict.values, **setupopts)

    # group trajectories and kspace
    #traj = np.reshape(groupby(traj, nspoke), (-1, npoint * nspoke))
    kdata = np.array([(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata])

    #kdata = np.reshape(groupby(kdata * density, nspoke), (-1, npoint * nspoke))

    print(f"Build volumes ({nspoke} groups)")

    # NUFFT
    kdata /= np.sum(np.abs(kdata)**2)**0.5 / len(kdata)
    volumes = [
        finufft.nufft2d1(t.real, t.imag, s, shape)
        for t, s in zip(traj, kdata)
    ]

    # init mask
    mask = False
    volumes0 = volumes
    kdata0 = kdata
    info = {}
    for i in range(niter + 1):

        # auto mask
        unique = np.histogram(np.abs(volumes), 100)[1]
        mask = mask | (np.mean(np.abs(volumes), axis=0) > unique[len(unique) // 10])
        mask = ndimage.binary_closing(mask, iterations=3)

        print(f"Search data (iteration {i})")
        obs = np.transpose([vol[mask] for vol in volumes])
        res = solver.search(obs, **searchopts)

        info[f"iteration {i}"] = solver.info

        if i == niter:
            break

        # generate prediction volumes
        pred = np.asarray(solver.predict(res)).T

        # predict spokes
        kdata = [
            finufft.nufft2d2(t.real, t.imag, makevol(p, mask))
            for t, p in zip(traj, pred)
        ]
        kdatai = np.array([(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata])

        # NUFFT
        kdatai /= np.sum(np.abs(kdatai)**2)**0.5 / len(kdatai)
        volumesi = [
            finufft.nufft2d1(t.real, t.imag, s, shape)
            for t, s in zip(traj, kdatai)
        ]

        # correct volumes
        volumes = [2 * vol0 - voli for vol0, voli in zip(volumes0, volumesi)]


    # make maps
    wt1map = makevol([p[0] for p in res.parameters], mask)
    ft1map = makevol([p[1] for p in res.parameters], mask)
    b1map = makevol([p[2] for p in res.parameters], mask)
    dfmap = makevol([p[3] for p in res.parameters], mask)
    wmap =  makevol([s[0] for s in res.scales], mask)
    fmap =  makevol([s[1] for s in res.scales], mask)
    ffmap = makevol([s[1]/(s[0] + s[1]) for s in res.scales], mask)



    return {
        "mask": mask,
        "wt1map": wt1map,
        "ft1map": ft1map,
        "b1map": b1map,
        "dfmap": dfmap,
        "wmap": wmap,
        "fmap": fmap,
        "ffmap": ffmap,
        "info": {"search": info, "options": solver.options},
    }

def basicDictSearch(all_signals,dictfile):
    #Basic dic search with component separation
    #
    mrfdict = dictsearch.Dictionary()
    mrfdict.load(dictfile, force=True)

    array_water = mrfdict.values[:, :, 0]
    array_fat = mrfdict.values[:, :, 1]

    var_w = np.sum(array_water * array_water.conj(), axis=1).real
    var_f = np.sum(array_fat * array_fat.conj(), axis=1).real
    sig_wf = np.sum(array_water * array_fat.conj(), axis=1).real

    print("Removing duplicate dictionary entries and signals")
    array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
    array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)
    all_signals_unique, index_signals_unique = np.unique(all_signals, axis=1, return_inverse=True)

    print("Calculating correlations")
    sig_ws_all_unique = np.matmul(array_water_unique, all_signals_unique[:, :].conj()).real
    sig_fs_all_unique = np.matmul(array_fat_unique, all_signals_unique[:, :].conj()).real
    sig_ws_all = sig_ws_all_unique[index_water_unique, :]
    sig_fs_all = sig_fs_all_unique[index_fat_unique, :]

    print("Calculating optimal fat fraction and best pattern per signal")
    var_w = np.reshape(var_w, (-1, 1))
    var_f = np.reshape(var_f, (-1, 1))
    sig_wf = np.reshape(sig_wf, (-1, 1))

    alpha_all_unique = (sig_wf * sig_ws_all-var_w*sig_fs_all) / ((sig_ws_all + sig_fs_all) * sig_wf-var_w*sig_fs_all-var_f*sig_ws_all)
    one_minus_alpha_all = 1 - alpha_all_unique
    J_all = (one_minus_alpha_all * sig_ws_all + alpha_all_unique * sig_fs_all) / np.sqrt(
        one_minus_alpha_all ** 2 * var_w + alpha_all_unique ** 2 * var_f + 2 * alpha_all_unique * one_minus_alpha_all * sig_wf)
    idx_max_all_unique = np.argmax(J_all, axis=0)
    #idx_max_all = idx_max_all_unique[index_signals_unique]

    print("Building the maps")
    #alpha_all = alpha_all_unique[:, index_signals_unique]

    params_all_unique = np.array([mrfdict.keys[idx] + (alpha_all_unique[idx, i],) for i, idx in enumerate(idx_max_all_unique)])
    params_all = params_all_unique[index_signals_unique]



    return {
            "wT1": params_all[:, 0],
            "fT1": params_all[:, 1],
            "attB1": params_all[:, 2],
            "df": params_all[:, 3],
            "ff": params_all[:, 4]

            }







def compare_paramMaps(map1,map2,mask1,mask2=None,fontsize=5,title1="Orig Map",title2="Rebuilt Map",adj_wT1=False,fat_threshold=0.8,proj_on_mask1=False,save=False,figsize=(30,10),units=None,vmax_error=None,extent=None,kept_keys=None):
    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())

    if kept_keys is not None:
        keys_1 = keys_1 & set(kept_keys)
        keys_2 = keys_2 & set(kept_keys)
    if mask2 is None:
        mask2 = mask1
    for k in (keys_1 & keys_2):
        fig,axes=plt.subplots(1,3,figsize=figsize)
        vol1 = makevol(map1[k],mask1)
        vol2= makevol(map2[k],mask2)

        if proj_on_mask1 is not None:
            if type(proj_on_mask1) is bool:#Projection on mask 1
                vol2=vol2*(mask1*1)
            else:#projection on external mask
                vol1 = vol1*proj_on_mask1
                vol2 = vol2*proj_on_mask1

        if adj_wT1 and k=="wT1":
            ff = makevol(map2["ff"],mask2)
            vol2[ff>fat_threshold]=vol1[ff>fat_threshold]

        error=vol2-vol1

        if units is None:
            graph_title1 = title1+" {}".format(k)
            graph_title2 = title2 + " {}".format(k)
            error_title = "Error {}".format(k)
        else:
            graph_title1 = title1+" {} ({})".format(k,units[k])
            graph_title2 = title2 + " {} ({})".format(k, units[k])
            error_title = "Error {} ({})".format(k,units[k])


        if extent is not None:
            centrum_x=int(vol1.shape[0]/2)
            centrum_y = int(vol2.shape[1] / 2)
            vol1=vol1[centrum_x-extent:centrum_x+extent,centrum_y-extent:centrum_y+extent]
            vol2 = vol2[centrum_x - extent:centrum_x + extent, centrum_y - extent:centrum_y + extent]
            error = error[centrum_x - extent:centrum_x + extent, centrum_y - extent:centrum_y + extent]

        minmin=np.min([np.min(vol1),np.min(vol2)])
        maxmax=np.max([np.max(vol1),np.max(vol2)])

        im1=axes[0].imshow(vol1,vmin=minmin,vmax=maxmax,aspect="auto")
        axes[0].set_title(graph_title1,fontdict={"fontsize":fontsize})
        axes[0].xaxis.set_visible(False)
        axes[0].yaxis.set_visible(False)
        #cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        #cbar1.ax.tick_params(labelsize=fontsize)

        im2 = axes[1].imshow(vol2,vmin=minmin,vmax=maxmax,aspect="auto")
        axes[1].set_title(graph_title2,fontdict={"fontsize":fontsize})
        axes[1].xaxis.set_visible(False)
        axes[1].yaxis.set_visible(False)


        if (vmax_error is None):
            im3 = axes[2].imshow(error,aspect="auto")
        else:
            im3 = axes[2].imshow(error,vmin=-vmax_error[k],vmax=vmax_error[k],aspect="auto")
        axes[2].set_title(error_title,fontdict={"fontsize":fontsize})
        axes[2].xaxis.set_visible(False)
        axes[2].yaxis.set_visible(False)

        #fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
        #                    wspace=0.08, hspace=0.02)

        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=fontsize)

        cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=fontsize)
        if save:
            fig.savefig("./figures/{}_vs_{}_{}".format(title1,title2,k))

def compare_paramMaps_3D(map1,map2,mask1,mask2=None,slice=0,fontsize=5,title1="Orig Map",title2="Rebuilt Map",adj_wT1=False,fat_threshold=0.8,proj_on_mask1=False,save=False):
    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    if mask2 is None:
        mask2 = mask1
    for k in (keys_1 & keys_2):
        fig,axes=plt.subplots(1,3)
        vol1 = makevol(map1[k],mask1)[slice,:,:]
        vol2= makevol(map2[k],mask2)[slice,:,:]
        if proj_on_mask1:
            vol2=vol2*(mask1[slice,:,:]*1)
        if adj_wT1 and k=="wT1":
            ff = makevol(map2["ff"],mask2)[slice,:,:]
            vol2[ff>fat_threshold]=vol1[ff>fat_threshold]

        error=(vol1-vol2)

        im1=axes[0].imshow(vol1)
        axes[0].set_title(title1+" "+k)
        axes[0].tick_params(axis='x', labelsize=fontsize)
        axes[0].tick_params(axis='y', labelsize=fontsize)
        cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=fontsize)

        im2 = axes[1].imshow(vol2)
        axes[1].set_title(title2+" "+k)
        axes[1].tick_params(axis='x', labelsize=fontsize)
        axes[1].tick_params(axis='y', labelsize=fontsize)
        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=fontsize)

        im3 = axes[2].imshow(error)
        axes[2].set_title("Error {}".format(k))
        axes[2].tick_params(axis='x', labelsize=fontsize)
        axes[2].tick_params(axis='y', labelsize=fontsize)
        cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=fontsize)
        if save:
            plt.savefig("./figures/{}_vs_{}_Slice_{}_{}".format(title1,title2,slice,k))

def regression_paramMaps(map1,map2,mask1=None,mask2=None,title="Maps regression plots",fontsize=5,adj_wT1=False,fat_threshold=0.8,mode="Standard",proj_on_mask1=False,save=False):

    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    nb_keys=len(keys_1 & keys_2)
    fig,ax = plt.subplots(1,nb_keys)

    for i,k in enumerate(keys_1 & keys_2):
        obs = map1[k]
        pred = map2[k]

        if mask1 is not None:
            if mask2 is None:
                mask2 = mask1
            mask_union = mask1 | mask2
            mat_obs = makevol(map1[k],mask1)
            mat_pred = makevol(map2[k],mask2)
            if proj_on_mask1:
                mat_pred = mat_pred*(mask1*1)

            obs = mat_obs[mask_union]
            pred = mat_pred[mask_union]

        if adj_wT1 and k=="wT1":
            ff = map2["ff"]
            obs = obs[ff < fat_threshold]
            pred = pred[ff < fat_threshold]

        x_min = np.min(obs)
        x_max = np.max(obs)

        if x_min==x_max:
            fig.delaxes(ax[i])
            continue

        mean=np.mean(obs)
        ss_tot = np.sum((obs-mean)**2)
        ss_res = np.sum((obs-pred)**2)
        bias = np.mean((pred-obs))
        r_2 = 1-ss_res/ss_tot

        dx = (x_max - x_min) / 10
        x_ = np.arange(x_min, x_max+dx,dx )

        if mode=="Standard":
            ax[i].scatter(obs,pred,s=1)
            ax[i].plot(x_, x_, "r")
        elif mode=="Boxplot":
            unique_obs=np.unique(obs)
            sns.boxplot(ax=ax[i],x=obs,y=pred)
            locs=ax[i].get_xticks()
            print(locs)
            sns.lineplot(ax=ax[i],x=locs,y=unique_obs)
        else:
            raise ValueError("mode should be Standard/Boxplot")

        ax[i].set_title(k+" R2:{} Bias:{}".format(np.round(r_2,2),np.round(bias,2)),fontsize=2*fontsize)
        ax[i].tick_params(axis='x', labelsize=fontsize)
        ax[i].tick_params(axis='y', labelsize=fontsize)

    plt.suptitle(title)

    if save:
        plt.savefig("./figures/{}".format(title))


def regression_paramMaps_ROI(map1, map2, mask1=None, mask2=None, maskROI=None, title="Maps regression plots",
                             fontsize=5, adj_wT1=False, fat_threshold=0.8, mode="Standard", proj_on_mask1=True,plt_std=False,
                             figsize=(15, 10),save=False,kept_keys=None,min_ROI_count=15,units=None,fontsize_axis=None):

    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    if kept_keys is not None:
        keys_1 = keys_1 & set(kept_keys)
        keys_2 = keys_2 & set(kept_keys)

    nb_keys = len(keys_1 & keys_2)

    if maskROI is None:
        maskROI = buildROImask(map1)

    #Removing ROIs with less than min_ROI_count values
    freq = collections.Counter(maskROI)
    maskROI = np.array([ele if freq[ele] > min_ROI_count else 0 for ele in maskROI])

    fig, ax = plt.subplots(1, nb_keys, figsize=figsize)

    for i, k in enumerate(sorted(keys_1 & keys_2)):
        obs = map1[k]
        pred = map2[k]

        if mask1 is not None:
            if mask2 is None:
                mask2 = mask1
            mask_union = mask1 | mask2
            mat_obs = makevol(map1[k], mask1)
            mat_pred = makevol(map2[k], mask2)
            mat_ROI = makevol(maskROI, mask1)
            if proj_on_mask1 is not None:
                if type(proj_on_mask1) is bool:#Projection on mask1
                    mat_pred = mat_pred * (mask1 * 1)
                    mat_ROI = mat_ROI * (mask1 * 1)
                    mat_obs = mat_obs * (mask1 * 1)
                    mask_union = mask1

                else : #projection on externally provided mask
                    mat_pred = mat_pred * (proj_on_mask1 * 1)
                    mat_ROI = mat_ROI * (proj_on_mask1 * 1)
                    mat_obs = mat_obs * (proj_on_mask1 * 1)
                    mask_union = proj_on_mask1

            obs = mat_obs[mask_union]
            pred = mat_pred[mask_union]
            maskROI_current = mat_ROI[mask_union]

            # print(obs)

        if adj_wT1 and k == "wT1":
            ff = makevol(map1["ff"], mask1)
            ff = ff[mask_union]
            obs = obs[ff < fat_threshold]
            pred = pred[ff < fat_threshold]
            maskROI_current = maskROI_current[ff < fat_threshold]

        df_obs = pd.DataFrame(columns=["Data", "Groups"],
                              data=np.stack([obs.flatten(), maskROI_current.flatten()], axis=-1))
        df_pred = pd.DataFrame(columns=["Data", "Groups"],
                               data=np.stack([pred.flatten(), maskROI_current.flatten()], axis=-1))
        obs = np.array(df_obs.groupby("Groups").mean())[1:]
        pred = np.array(df_pred.groupby("Groups").mean())[1:]
        # obs_std = np.array(df_obs.groupby("Groups").std())[1:]

        # print(list(pred_std.reshape(1,-1)))

        x_min = np.min(obs)
        x_max = np.max(obs)

        if x_min == x_max:
            fig.delaxes(ax[i])
            continue

        mean = np.mean(obs)
        ss_tot = np.sum((obs - mean) ** 2)
        ss_res = np.sum((obs - pred) ** 2)
        bias = np.mean((pred - obs))
        r_2 = 1 - ss_res / ss_tot

        dx = (x_max - x_min) / 10
        x_ = np.arange(x_min, x_max + dx, dx)

        if mode == "Standard":
            if plt_std:
                pred_std = np.array(df_pred.groupby("Groups").std())[1:]
                ax[i].errorbar(obs, pred, list(pred_std.flatten()), linestyle='None', marker='^')
            else:
                ax[i].scatter(obs,pred,s=1)
            ax[i].plot(x_, x_, "r")
        elif mode == "Boxplot":
            unique_obs = np.unique(obs)
            sns.boxplot(ax=ax[i], x=obs, y=pred)
            locs = ax[i].get_xticks()
            sns.lineplot(ax=ax[i], x=locs, y=unique_obs)
        else:
            raise ValueError("mode should be Standard/Boxplot")
        if units is None:
            graph_title = k + " R2:{} Bias:{}".format(np.round(r_2, 4), np.round(bias, 3))
        else:
            graph_title = k + " R2:{} Bias:{} ({})".format(np.round(r_2, 4), np.round(bias, 3), units[k])

        ax[i].set_title(graph_title, fontsize=2 * fontsize)

        if fontsize_axis is None:
            ax[i].tick_params(axis='x', labelsize=fontsize)
            ax[i].tick_params(axis='y', labelsize=fontsize)

        else:
            ax[i].tick_params(axis='x', labelsize=fontsize_axis)
            ax[i].tick_params(axis='y', labelsize=fontsize_axis)


    plt.suptitle(title)
    if save:
        fig.savefig("./figures/{}".format(title))


def process_ROI_values(all_results, save=False,mode="Standard",title="Results comparison all ROIs",plt_std=False,figsize=(15, 10),fontsize=5,fontsize_axis=None,units=None):

    nb_keys=len(all_results.keys())
    fig, ax = plt.subplots(1, nb_keys, figsize=figsize)

    for i,k in enumerate(all_results.keys()):
        obs=all_results[k][:,0]
        pred = all_results[k][:, 1]

        x_min = np.min(obs)
        x_max = np.max(obs)

        if x_min == x_max:
            fig.delaxes(ax[i])
            continue

        mean = np.mean(obs)
        ss_tot = np.sum((obs - mean) ** 2)
        ss_res = np.sum((obs - pred) ** 2)
        bias = np.mean((pred - obs))
        r_2 = 1 - ss_res / ss_tot

        dx = (x_max - x_min) / 10
        x_ = np.arange(x_min, x_max + dx, dx)

        if mode == "Standard":
            if plt_std:
                pred_std = np.array(df_pred.groupby("Groups").std())[1:]
                ax[i].errorbar(obs, pred, list(pred_std.flatten()), linestyle='None', marker='^')
            else:
                ax[i].scatter(obs, pred, s=1)
            ax[i].plot(x_, x_, "r")
        elif mode == "Boxplot":
            unique_obs = np.unique(obs)
            sns.boxplot(ax=ax[i], x=obs, y=pred)
            locs = ax[i].get_xticks()
            sns.lineplot(ax=ax[i], x=locs, y=unique_obs)
        else:
            raise ValueError("mode should be Standard/Boxplot")

        if units is None:
            graph_title = k + " R2:{} Bias:{}".format(np.round(r_2, 4), np.round(bias, 3))
        else :
            graph_title = k + " R2:{} Bias:{} ({})".format(np.round(r_2, 4), np.round(bias, 3),units[k])
        ax[i].set_title(graph_title, fontsize=2 * fontsize)

        if fontsize_axis is None:
            ax[i].tick_params(axis='x', labelsize=fontsize)
            ax[i].tick_params(axis='y', labelsize=fontsize)
        else:
            ax[i].tick_params(axis='x', labelsize=fontsize_axis)
            ax[i].tick_params(axis='y', labelsize=fontsize_axis)

    plt.suptitle(title)
    if save:
        plt.savefig("./figures/{}".format(title))

def metrics_ROI_values(all_results,units=None,name="Results"):

    nb_keys=len(all_results.keys())
    df = pd.DataFrame(columns=[name])
    for i,k in enumerate(all_results.keys()):
        obs=all_results[k][:,0]
        pred = all_results[k][:, 1]


        mean = np.mean(obs)
        ss_tot = np.sum((obs - mean) ** 2)
        ss_res = np.sum((obs - pred) ** 2)
        bias = np.mean((pred - obs))
        r_2 = 1 - ss_res / ss_tot

        error = np.linalg.norm(obs-pred)/np.sqrt(len(obs))

        if units is None:
            r2_label = "R2 {}".format(k)
            bias_label = "Bias {}".format(k)
            rmse_label ="RMSE {}".format(k)

        else :
            r2_label = "R2 {} (a.u)".format(k)
            bias_label = "Bias {} ({})".format(k,units[k])
            rmse_label = "RMSE {} ({})".format(k,units[k])

        df=df.append(pd.DataFrame(columns=[name],index=[r2_label],data=r_2))
        df=df.append(pd.DataFrame(columns=[name], index=[bias_label], data=bias))
        df=df.append(pd.DataFrame(columns=[name], index=[rmse_label], data=error))

    return df

def get_ROI_values(map1, map2, mask1=None, mask2=None, maskROI=None, adj_wT1=False, fat_threshold=0.8,proj_on_mask1=True,plt_std=False,
                             kept_keys=None,min_ROI_count=15):

    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    if kept_keys is not None:
        keys_1 = keys_1 & set(kept_keys)
        keys_2 = keys_2 & set(kept_keys)

    nb_keys = len(keys_1 & keys_2)

    if maskROI is None:
        maskROI = buildROImask(map1)

    #Removing ROIs with less than min_ROI_count values
    freq = collections.Counter(maskROI)
    maskROI = np.array([ele if freq[ele] > min_ROI_count else 0 for ele in maskROI])


    results={}
    for i, k in enumerate(sorted(keys_1 & keys_2)):
        obs = map1[k]
        pred = map2[k]

        if mask1 is not None:
            if mask2 is None:
                mask2 = mask1
            mask_union = mask1 | mask2
            mat_obs = makevol(map1[k], mask1)
            mat_pred = makevol(map2[k], mask2)
            mat_ROI = makevol(maskROI, mask1)
            if proj_on_mask1 is not None:
                if type(proj_on_mask1) is bool:#Projection on mask1
                    mat_pred = mat_pred * (mask1 * 1)
                    mat_ROI = mat_ROI * (mask1 * 1)
                    mat_obs = mat_obs * (mask1 * 1)
                    mask_union = mask1

                else : #projection on externally provided mask
                    mat_pred = mat_pred * (proj_on_mask1 * 1)
                    mat_ROI = mat_ROI * (proj_on_mask1 * 1)
                    mat_obs = mat_obs * (proj_on_mask1 * 1)
                    mask_union = proj_on_mask1

            obs = mat_obs[mask_union]
            pred = mat_pred[mask_union]
            maskROI_current = mat_ROI[mask_union]

            # print(obs)

        if adj_wT1 and k == "wT1":
            ff = makevol(map1["ff"], mask1)
            ff = ff[mask_union]
            obs = obs[ff < fat_threshold]
            pred = pred[ff < fat_threshold]
            maskROI_current = maskROI_current[ff < fat_threshold]

        df_obs = pd.DataFrame(columns=["Data", "Groups"],
                              data=np.stack([obs.flatten(), maskROI_current.flatten()], axis=-1))
        df_pred = pd.DataFrame(columns=["Data", "Groups"],
                               data=np.stack([pred.flatten(), maskROI_current.flatten()], axis=-1))
        obs = np.array(df_obs.groupby("Groups").mean())[1:]
        pred = np.array(df_pred.groupby("Groups").mean())[1:]
        # obs_std = np.array(df_obs.groupby("Groups").std())[1:]

        # print(list(pred_std.reshape(1,-1)))
        results[k]=np.concatenate([obs,pred],axis=1)
    return results

def metrics_paramMaps_ROI(map_ref, map2, mask_ref=None, mask2=None, maskROI=None,
                              adj_wT1=False, fat_threshold=0.8, proj_on_mask1=True,name="Result",min_ROI_count=15,units=None
                             ):

    df = pd.DataFrame(columns=[name])

    keys_1 = set(map_ref.keys())
    keys_2 = set(map2.keys())
    nb_keys = len(keys_1 & keys_2)

    if maskROI is None:
        maskROI = buildROImask(map_ref)

    # Removing ROIs with less than min_ROI_count values
    freq = collections.Counter(maskROI)
    maskROI = np.array([ele if freq[ele] > min_ROI_count else 0 for ele in maskROI])

    for i, k in enumerate(keys_1 & keys_2):
        print(i)

        mask_union = mask_ref | mask2
        mat_obs = makevol(map_ref[k], mask_ref)
        mat_pred = makevol(map2[k], mask2)
        mat_ROI = makevol(maskROI, mask_ref)
        if proj_on_mask1 is not None:
            if type(proj_on_mask1) is bool:  # Projection on mask1
                mat_pred = mat_pred * (mask_ref * 1)
                mat_ROI = mat_ROI * (mask_ref * 1)
                mat_obs = mat_obs * (mask_ref * 1)
                mask_union = mask_ref

            else:  # projection on externally provided mask
                mat_pred = mat_pred * (proj_on_mask1 * 1)
                mat_ROI = mat_ROI * (proj_on_mask1 * 1)
                mat_obs = mat_obs * (proj_on_mask1 * 1)
                mask_union = proj_on_mask1

        obs = mat_obs[mask_union]
        pred = mat_pred[mask_union]
        maskROI_current = mat_ROI[mask_union]

        if adj_wT1 and k == "wT1":
            ff = makevol(map_ref["ff"], mask_ref)
            ff = ff[mask_union]
            obs = obs[ff < fat_threshold]
            pred = pred[ff < fat_threshold]
            maskROI_current = maskROI_current[ff < fat_threshold]

        df_all = pd.DataFrame(columns=["Data_Obs", "Data_Pred", "Groups"],
                              data=np.stack([obs.flatten(), pred.flatten(), maskROI_current.flatten()], axis=-1))

        ssim_values = df_all.groupby("Groups").apply(lambda x: ssim(x.Data_Obs, x.Data_Pred))
        mean_ssim = ssim_values.mean()
        std_ssim = ssim_values.std()

        mean_obs = np.array(df_all[["Data_Obs","Groups"]].groupby("Groups").mean())[1:]
        mean_pred = np.array(df_all[["Data_Pred", "Groups"]].groupby("Groups").mean())[1:]


        x_min = np.min(mean_obs)
        x_max = np.max(mean_obs)

        if x_min == x_max:
            continue

        mean = np.mean(mean_obs)
        ss_tot = np.sum((mean_obs - mean) ** 2)
        ss_res = np.sum((mean_obs - mean_pred) ** 2)
        bias = np.mean((mean_pred - mean_obs))
        r_2 = 1 - ss_res / ss_tot


        df_error = pd.DataFrame(columns=["Data", "Groups"],
                                data=np.stack(
                                    [(pred.flatten() - obs.flatten()) ** 2, maskROI_current.flatten()],
                                    axis=-1))
        errors = np.sqrt(np.array(df_error.groupby("Groups").mean())[1:])
        error = np.mean(errors)
        std_error = np.std(errors)

        if units is None:
            r2_label = "R2 {}".format(k)
            bias_label = "Bias {}".format(k)
            rmse_label ="mean RMSE {}".format(k)
            std_rmse_label = "std RMSE {}".format(k)
            ssim_label = "mean SSIM {}".format(k)
            std_ssim_label ="std SSIM {}".format(k)
        else :
            r2_label = "R2 {} (a.u)".format(k)
            bias_label = "Bias {} ({})".format(k,units[k])
            rmse_label = "mean RMSE {} ({})".format(k,units[k])
            std_rmse_label = "std RMSE {} ({})".format(k, units[k])
            ssim_label = "mean SSIM {} (a.u)".format(k)
            std_ssim_label = "std SSIM {} (a.u)".format(k)

        df=df.append(pd.DataFrame(columns=[name],index=[r2_label],data=r_2))
        df=df.append(pd.DataFrame(columns=[name], index=[bias_label], data=bias))
        df=df.append(pd.DataFrame(columns=[name], index=[rmse_label], data=error))
        df=df.append(pd.DataFrame(columns=[name], index=[std_rmse_label], data=std_error))
        df = df.append(pd.DataFrame(columns=[name], index=[ssim_label], data=mean_ssim))
        df = df.append(pd.DataFrame(columns=[name], index=[std_ssim_label], data=std_ssim))

    return df

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def voronoi_volumes_freud(points,box_size=2*np.pi):

    box = freud.box.Box.cube(box_size + 0.00001)

    voro = freud.locality.Voronoi()
    voro.compute(system=(box, points))

    volumes = np.array(voro.volumes)
    #volumes /= volumes.sum()

    return volumes

def voronoi_volumes(points,min_x=None,min_y=None,max_x=None,max_y=None,min_z=None,max_z=None,eps=0.1):
    dim =points.shape[1]

    vor = Voronoi(points)

    if dim==3:
        regions, vertices = voronoi_finite_polygons_2d(vor)

    else:
        regions, vertices = voronoi_finite_polygons_3d(vor)

    if min_x is None:
        min_x = vor.min_bound[0] - eps

    if max_x is None:
        max_x = vor.max_bound[0] + eps


    if min_y is None:
        min_y = vor.min_bound[1] - eps

    if max_y is None:
        max_y = vor.max_bound[1] + eps


    if dim==3:
        if min_z is None:
            min_z = vor.min_bound[2] - eps

        if max_z is None:
            max_z = vor.max_bound[2] + eps


        mins = np.tile((min_x, min_y,min_z), (vertices.shape[0], 1))
        maxs = np.tile((max_x, max_y,min_z), (vertices.shape[0], 1))

    else:
        mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
        maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))

    bounded_vertices = np.max((vertices, mins), axis=0)
    bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

    vol = np.zeros(len(regions))
    for i, indices in enumerate(regions):
        if -1 in indices:  # some regions can be opened - SHOULD NOT HAPPEN WITH NEW IMPLEMENTATION
            # indices.remove(-1)
            print("Warning : Had to set a voronoi volume to 0 for {}".format(np.round(v.points[i], 2)))
            vol[i] = 0
        else:
            vol[i] = ConvexHull(bounded_vertices[indices]).volume
        # except:
        #     print("Warning : Had to set a voronoi volume to 0 for {}".format(np.round(v.points[i],2)))
        #     vol[i]=0

    return vol, vor


def normalize_image_series(images_series):
    shapes=list(images_series.shape)
    last_dimensions_collapse =reduce(lambda x, y: x*y, shapes[1:])
    normalization = np.reshape(np.sum(np.abs(np.reshape(images_series,(-1,last_dimensions_collapse)) ** 2), axis=-1) ** 0.5, (-1,)+tuple(np.ones(len(shapes[1:])).astype(int)))
    images_series /= normalization
    return images_series


def transform_py_map(res,mask):
    map_py = res.copy()
    map_py.pop("info")
    map_py.pop("mask")
    keys = list(map_py.keys()).copy()

    for k in keys:
        map_py[str.split(k, "map")[0]] = map_py.pop(k)[mask > 0]
    map_py["attB1"] = map_py.pop("b1")
    map_py["fT1"] = map_py.pop("ft1")
    map_py["wT1"] = map_py.pop("wt1")
    return map_py

def build_mask(volumes):
    mask = False
    unique = np.histogram(np.abs(volumes), 100)[1]
    mask = mask | (np.mean(np.abs(volumes), axis=0) > unique[len(unique) // 10])
    mask = ndimage.binary_closing(mask, iterations=3)
    return mask*1


def build_mask_single_image(kdata,trajectory,size,useGPU=False,eps=1e-6,threshold_factor=1/7):
    mask = False

    npoint=trajectory.paramDict["npoint"]
    traj = trajectory.get_traj_for_reconstruction()

    # kdata /= np.sum(np.abs(kdata) ** 2) ** 0.5 / len(kdata)

    density = np.abs(np.linspace(-1, 1, npoint))
    kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata]
    # kdata = (normalize_image_series(np.array(kdata)))
    kdata_all = np.concatenate(kdata)
    traj_all = np.concatenate(list(traj))
    if traj_all.shape[-1]==2: # For slices

        if not(useGPU):
            volume_rebuilt = finufft.nufft2d1(traj_all[:,0], traj_all[:,1], kdata_all, size)
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2), dtype=complex_dtype)


            c_retrieved = kdata_all
            kx = traj_all[:, 0]
            ky = traj_all[:, 1]

            # Cast to desired datatype.
            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            c_retrieved = c_retrieved.astype(complex_dtype)

            kx_gpu = to_gpu(kx)
            ky_gpu = to_gpu(ky)
            c_retrieved_gpu = to_gpu(c_retrieved)

            # Allocate memory for the uniform grid on the GPU.

            # Initialize the plan and set the points.
            plan = cufinufft(1, (N1, N2), 1, eps=eps, dtype=dtype)
            plan.set_pts(kx_gpu, ky_gpu)

            # Execute the plan, reading from the strengths array c and storing the
            # result in fk_gpu.
            plan.execute(c_retrieved_gpu, fk_gpu)

            fk = np.squeeze(fk_gpu.get())
            volume_rebuilt = np.array(fk)
            kx_gpu.gpudata.free()
            ky_gpu.gpudata.free()
            fk_gpu.gpudata.free()
            c_retrieved_gpu.gpudata.free()

            plan.__del__()


        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[int(len(unique)*threshold_factor)])
        #mask = ndimage.binary_closing(mask, iterations=10)


    elif traj_all.shape[-1]==3: # For volumes
        if not(useGPU):
            volume_rebuilt = finufft.nufft3d1(traj_all[:, 2],traj_all[:, 0], traj_all[:, 1], kdata_all, size)
        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)

            c_retrieved = kdata_all
            kx = traj_all[:, 0]
            ky = traj_all[:, 1]
            kz = traj_all[:, 2]

            # Cast to desired datatype.
            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            c_retrieved = c_retrieved.astype(complex_dtype)

            # Allocate memory for the uniform grid on the GPU.

            # Initialize the plan and set the points.
            plan = cufinufft(1, (N1, N2, N3), 1, eps=eps, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

            # Execute the plan, reading from the strengths array c and storing the
            # result in fk_gpu.
            plan.execute(to_gpu(c_retrieved), fk_gpu)

            fk = np.squeeze(fk_gpu.get())
            volume_rebuilt = np.array(fk)
            plan.__del__()

        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[int(len(unique)*threshold_factor)])
        mask = ndimage.binary_closing(mask, iterations=3)

    return mask

def build_mask_single_image_multichannel(kdata,trajectory,size,density_adj=True,eps=1e-6,b1=None,threshold_factor=None,useGPU=False,normalize_kdata=True,light_memory_usage=False,is_theta_z_adjusted=False,selected_spokes=None,normalize_volumes=True):
    '''

    :param kdata: shape nchannels*ntimesteps*point_per_timestep
    :param trajectory: shape ntimesteps * point_per_timestep * ndim (2 or 3)
    :param size: image size
    :param density_adj:
    :param eps:
    :param b1: coil sensitivity map
    :return: mask of size size
    '''
    mask = False

    if (selected_spokes is not None):
        trajectory_for_mask = copy(trajectory)
        #selected_spokes = np.r_[20:800,1200:1400]
        trajectory_for_mask.traj = trajectory.get_traj()[selected_spokes]
    else:
        trajectory_for_mask = trajectory

    if (selected_spokes is not None):
        volume_rebuilt = build_single_image_multichannel(kdata[:,selected_spokes,:,:],trajectory_for_mask,size,density_adj,eps,b1,useGPU=useGPU,normalize_kdata=normalize_kdata,light_memory_usage=light_memory_usage,is_theta_z_adjusted=is_theta_z_adjusted,normalize_volumes=normalize_volumes)
    else:
        volume_rebuilt = build_single_image_multichannel(kdata, trajectory_for_mask, size,
                                                         density_adj, eps, b1, useGPU=useGPU,
                                                         normalize_kdata=normalize_kdata,
                                                         light_memory_usage=light_memory_usage,
                                                         is_theta_z_adjusted=is_theta_z_adjusted,normalize_volumes=normalize_volumes)

    traj = trajectory.get_traj_for_reconstruction()


    if traj.shape[-1]==2: # For slices

        if threshold_factor is None:
            threshold_factor = 1/7

        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[int(len(unique) *threshold_factor)])
        #mask = ndimage.binary_closing(mask, iterations=3)


    elif traj.shape[-1]==3: # For volumes

        if threshold_factor is None:
            threshold_factor = 1/20

        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[int(len(unique) *threshold_factor)])
        #mask = ndimage.binary_closing(mask, iterations=3)

    return mask


def build_single_image_multichannel(kdata,trajectory,size,density_adj=True,eps=1e-6,b1=None,useGPU=False,normalize_kdata=False,light_memory_usage=False,is_theta_z_adjusted=False,normalize_volumes=False):
    '''

    :param kdata: shape nchannels*ntimesteps*point_per_timestep
    :param trajectory: shape ntimesteps * point_per_timestep * ndim (2 or 3)
    :param size: image size
    :param density_adj:
    :param eps:
    :param b1: coil sensitivity map
    :return: mask of size size
    '''
    volume_rebuilt=simulate_radial_undersampled_images_multi(kdata,trajectory,size,density_adj,eps,is_theta_z_adjusted,b1,1,useGPU,None,normalize_kdata,light_memory_usage,True)[0]
    return volume_rebuilt

def generate_kdata(volumes,trajectory,useGPU=False,eps=1e-6):
    traj=trajectory.get_traj()

    if traj.shape[-1]==2:# For slices
        if not(useGPU):
            kdata = [
                    finufft.nufft2d2(t[:,0], t[:,1], p)
                    for t, p in zip(traj, volumes)
                ]
        else:
            # Allocate memory for the nonuniform coefficients on the GPU.
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            N1,N2 = volumes.shape[1],volumes.shape[2]
            M = traj.shape[1]
            c_gpu = GPUArray((1, M), dtype=complex_dtype)
            # Initialize the plan and set the points.
            kdata=[]
            for i in list(range(volumes.shape[0])):
                fk = volumes[i, :, :]
                kx = traj[i, :, 0]
                ky = traj[i, :, 1]

                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                fk = fk.astype(complex_dtype)

                plan = cufinufft(2, (N1, N2), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kx), to_gpu(ky))
                plan.execute(c_gpu, to_gpu(fk))
                c = np.squeeze(c_gpu.get())
                kdata.append(c)
                plan.__del__()

    elif traj.shape[-1]==3:# For volumes
        if not (useGPU):
            kdata = [
                finufft.nufft3d2(t[:, 2],t[:, 0], t[:, 1], p)
                for t, p in zip(traj, volumes)
            ]
        else:
            # Allocate memory for the nonuniform coefficients on the GPU.
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            N1, N2,N3 = volumes.shape[1], volumes.shape[2],volumes.shape[3]
            M = traj.shape[1]
            c_gpu = GPUArray((M), dtype=complex_dtype)
            # Initialize the plan and set the points.
            kdata = []
            for i in list(range(volumes.shape[0])):
                fk = volumes[i, :, :]
                kx = traj[i, :, 0]
                ky = traj[i, :, 1]
                kz = traj[i, :, 2]

                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                kz = kz.astype(dtype)
                fk = fk.astype(complex_dtype)

                plan = cufinufft(2, (N1, N2,N3), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))
                plan.execute(c_gpu, to_gpu(fk))
                c = np.squeeze(c_gpu.get())
                kdata.append(c)
                plan.__del__()
    return kdata

def buildROImask(map,max_clusters=40):
    # ROI using KNN clustering
    if "wT1" not in map:
        raise ValueError("wT1 should be in the param Map to build the ROI")

    #print(map["wT1"].shape)
    orig_data = map["wT1"].reshape(-1, 1)
    data = orig_data + np.random.normal(size=orig_data.shape,scale=0.1)
    model = KMeans(n_clusters=np.minimum(len(np.unique(orig_data)),max_clusters))
    model.fit(data)
    groups = model.labels_ + 1
    #print(groups.shape)
    return groups

def buildROImask_unique(map):
    # ROI using regions with same value of T1
    if "wT1" not in map:
        raise ValueError("wT1 should be in the param Map to build the ROI")

    unique_wT1 = np.unique(map["wT1"])
    maskROI = np.zeros(map["wT1"].shape)
    for i, value in enumerate(unique_wT1):
        maskROI[map["wT1"] == value] = i + 1

    return maskROI

def simulate_radial_undersampled_images(kdata,trajectory,size,density_adj=True,useGPU=False,eps=1e-6,is_theta_z_adjusted=False,ntimesteps=175):
#Deals with single channel data / howver kdata can be a list of arrays (meaning each timestep does not need to have the same number of spokes/partitions)
    traj=trajectory.get_traj_for_reconstruction(ntimesteps)
    npoint = trajectory.paramDict["npoint"]
    nb_allspokes = trajectory.paramDict["total_nspokes"]
    nspoke = int(nb_allspokes / ntimesteps)

    if not(is_theta_z_adjusted):
        dtheta = np.pi / nspoke
        dz = 1/trajectory.paramDict["nb_rep"]

    else:
        dtheta=1
        dz = 1/(2*np.pi)


    if not(len(kdata)==len(traj)):
        kdata=np.array(kdata).reshape(len(traj),-1)

    if not(kdata[0].shape[0]==traj[0].shape[0]):
        raise ValueError("Incompatible Kdata and Trajectory shapes")

    kdata = [k / (2*npoint)*dz * dtheta for k in kdata]



    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata]

    #kdata = (normalize_image_series(np.array(kdata)))

    if traj[0].shape[-1] == 2:  # 2D
        if not(useGPU):
            images_series_rebuilt = [
                finufft.nufft2d1(t[:,0], t[:,1], s, size)
                for t, s in zip(traj, kdata)
            ]
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2), dtype=complex_dtype)
            images_GPU = []
            for i in list(range(len(kdata))):

                c_retrieved = kdata[i]
                kx = traj[i][:, 0]
                ky = traj[i][:, 1]

                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.


                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(to_gpu(c_retrieved), fk_gpu)

                fk = np.squeeze(fk_gpu.get())
                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt=np.array(images_GPU)
    elif traj[0].shape[-1] == 3:  # 3D
        if not(useGPU):
            #images_series_rebuilt = [
            #    finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size)
            #    for t, s in zip(traj, kdata)
            #]

            images_series_rebuilt=[]
            for t,s in tqdm(zip(traj,kdata)):
                s=s.astype(np.complex64)
                t=t.astype(np.float32)
                images_series_rebuilt.append(finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size))

        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64

            images_GPU=[]
            for i in list(range(len(kdata))):
                fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)
                c_retrieved = kdata[i]
                kx = traj[i][ :, 0]
                ky = traj[i][ :, 1]
                kz = traj[i][ :, 2]


                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                kz = kz.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.
                c_retrieved_gpu = to_gpu(c_retrieved)

                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2,N3), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(c_retrieved_gpu, fk_gpu)

                fk = np.squeeze(fk_gpu.get())

                fk_gpu.gpudata.free()
                c_retrieved_gpu.gpudata.free()

                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt = np.array(images_GPU)

    #images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return np.array(images_series_rebuilt)

def simulate_radial_undersampled_images_density_optim(kdata,trajectory,size,density_adj=True,useGPU=False,eps=1e-6,is_theta_z_adjusted=False,ntimesteps=175):
#Deals with single channel data / howver kdata can be a list of arrays (meaning each timestep does not need to have the same number of spokes/partitions)
    traj=trajectory.get_traj_for_reconstruction()
    npoint = trajectory.paramDict["npoint"]
    nb_allspokes=trajectory.paramDict["total_nspokes"]
    nspoke=int(nb_allspokes/ntimesteps)

    if not(is_theta_z_adjusted):
        dtheta = np.pi / nspoke
        dz = 1/trajectory.paramDict["nb_rep"]

    else:
        dtheta=np.pi
        dz = 1


    if not(len(kdata)==len(traj)):
        kdata=np.array(kdata).reshape(len(traj),-1)

    if not(kdata[0].shape[0]==traj[0].shape[0]):
        raise ValueError("Incompatible Kdata and Trajectory shapes")

    kdata = [k / (npoint)*dz * dtheta for k in kdata]

    if density_adj:
        traj_left=np.moveaxis(traj,1,-1)
        traj_right=np.expand_dims(traj_left,axis=-1)
        traj_right=np.moveaxis(traj_right,2,-1)
        traj_left = np.expand_dims(traj_left,axis=-1)
        for ts in tqdm(range(traj.shape[0])):
            density=traj_left[ts]-traj_right[ts]
            density=np.sinc(density)**2
            density=np.prod(density,axis=0)
            density = 1/np.sum(density,axis=1)
            kdata[ts] = kdata[ts]*density

    #kdata = (normalize_image_series(np.array(kdata)))

    if traj[0].shape[-1] == 2:  # 2D
        if not(useGPU):
            images_series_rebuilt = [
                finufft.nufft2d1(t[:,0], t[:,1], s, size)
                for t, s in zip(traj, kdata)
            ]
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2), dtype=complex_dtype)
            images_GPU = []
            for i in list(range(len(kdata))):

                c_retrieved = kdata[i]
                kx = traj[i][:, 0]
                ky = traj[i][:, 1]

                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.


                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(to_gpu(c_retrieved), fk_gpu)

                fk = np.squeeze(fk_gpu.get())
                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt=np.array(images_GPU)
    elif traj[0].shape[-1] == 3:  # 3D
        if not(useGPU):
            #images_series_rebuilt = [
            #    finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size)
            #    for t, s in zip(traj, kdata)
            #]

            images_series_rebuilt=[]
            for t,s in tqdm(zip(traj,kdata)):
                images_series_rebuilt.append(finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size))

        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64

            images_GPU=[]
            for i in list(range(len(kdata))):
                fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)
                c_retrieved = kdata[i]
                kx = traj[i][ :, 0]
                ky = traj[i][ :, 1]
                kz = traj[i][ :, 2]


                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                kz = kz.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.
                c_retrieved_gpu = to_gpu(c_retrieved)

                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2,N3), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(c_retrieved_gpu, fk_gpu)

                fk = np.squeeze(fk_gpu.get())

                fk_gpu.gpudata.free()
                c_retrieved_gpu.gpudata.free()

                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt = np.array(images_GPU)

    #images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return np.array(images_series_rebuilt)

# def simulate_radial_undersampled_images_multi(kdata,trajectory,size,density_adj=True,eps=1e-6,is_theta_z_adjusted=False,b1=None,ntimesteps=175,useGPU=False,memmap_file=None,normalize_kdata=False,light_memory_usage=False,normalize_volumes=True):
# #Deals with single channel data / howver kdata can be a list of arrays (meaning each timestep does not need to have the same number of spokes/partitions)
#
#     #if light_memory_usage and not(useGPU):
#     #    print("Warning : light memory usage is not used without GPU")
#
#     traj=trajectory.get_traj_for_reconstruction(ntimesteps)
#     npoint = trajectory.paramDict["npoint"]
#     nb_allspokes=trajectory.paramDict["total_nspokes"]
#     nb_channels=kdata.shape[0]
#     nspoke=int(nb_allspokes/ntimesteps)
#
#
#     if kdata.dtype=="complex64":
#         traj=traj.astype(np.float32)
#
#
#     if not(is_theta_z_adjusted):
#         dtheta = np.pi / nspoke
#         dz = 1/trajectory.paramDict["nb_rep"]
#
#     else:
#         dtheta=np.pi
#         dz = 1
#
#
#     if not(kdata.shape[1]==len(traj)):
#         kdata=kdata.reshape(kdata.shape[0],len(traj),-1)
#
#
#
#
#     if density_adj:
#         density = np.abs(np.linspace(-1, 1, npoint))
#         kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata]
#
#
#     if normalize_kdata:
#         print("Normalizing Kdata")
#         if not(light_memory_usage):
#             C = np.mean(np.linalg.norm(kdata,axis=1),axis=-1)
#             kdata /= np.expand_dims(C,axis=(1,2))
#         else:
#             for i in tqdm(range(nb_channels)):
#                 kdata[i]/=np.mean(np.linalg.norm(kdata[i],axis=0))
#     else:
#         kdata *= dz * dtheta / npoint
#
#     #kdata = (normalize_image_series(np.array(kdata)))
#
#
#     output_shape = (ntimesteps,)+size
#
#     flushed=False
#
#     if memmap_file is not None :
#         from tempfile import mkdtemp
#         import os.path as path
#         file_memmap=path.join(mkdtemp(),"memmap_volumes.dat")
#         images_series_rebuilt = np.memmap(file_memmap,dtype="complex64",mode="w+",shape=output_shape)
#
#     else:
#         images_series_rebuilt = np.zeros(output_shape,dtype=np.complex64)
#
#     print("Performing NUFFT")
#     if traj[0].shape[-1] == 2:  # 2D
#
#         for i,t in tqdm(enumerate(traj)):
#             fk = finufft.nufft2d1(t[:,0], t[:,1], kdata[:,i,:], size)
#
#             #images_series_rebuilt = np.moveaxis(images_series_rebuilt, 0, 1)
#             if b1 is None:
#                 images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
#             else:
#                 images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)
#
#     elif traj[0].shape[-1] == 3:  # 3D
#         if not (useGPU):
#             for i, t in tqdm(enumerate(traj)):
#                 if not(light_memory_usage):
#                     fk = finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], kdata[:, i, :], size)
#                     if b1 is None:
#                         images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
#                     else:
#                         images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)
#
#                 else:
#                     flush_condition=(memmap_file is not None)and((psutil.virtual_memory().cached+psutil.virtual_memory().free)/1e9<2)and(not(flushed))
#                     if flush_condition:
#                         print("Flushed Memory")
#                         offset=i*images_series_rebuilt.itemsize*i*np.prod(images_series_rebuilt.shape[1:])
#                         i0 = i
#                         new_shape=(output_shape[0]-i,)+output_shape[1:]
#                         images_series_rebuilt.flush()
#                         del images_series_rebuilt
#                         flushed=True
#                         normalize_volumes = False
#                         images_series_rebuilt = np.memmap(memmap_file,dtype="complex64",mode="r+",shape=new_shape,offset=offset)
#
#                     if flushed :
#                         for j in tqdm(range(nb_channels)):
#                             fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kdata[j, i, :], size)
#                             if b1 is None:
#                                 images_series_rebuilt[i-i0] += np.abs(fk) ** 2
#                             else:
#                                 images_series_rebuilt[i-i0] += b1[j].conj() * fk
#
#                         if b1 is None:
#                             images_series_rebuilt[i] = np.sqrt(images_series_rebuilt[i])
#
#                     else:
#                         for j in tqdm(range(nb_channels)):
#                             fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kdata[j, i, :], size)
#                             if b1 is None:
#                                 images_series_rebuilt[i] += np.abs(fk) ** 2
#                             else:
#                                 images_series_rebuilt[i] += b1[j].conj() * fk
#
#                         if b1 is None:
#                             images_series_rebuilt[i] = np.sqrt(images_series_rebuilt[i])
#         else:
#             N1, N2, N3 = size[0], size[1], size[2]
#             dtype = np.float32  # Datatype (real)
#             complex_dtype = np.complex64
#
#             for i in tqdm(list(range(kdata.shape[1]))):
#                 if not(light_memory_usage):
#                     fk_gpu = GPUArray((nb_channels,N1, N2, N3), dtype=complex_dtype)
#                     c_retrieved = kdata[:,i,:]
#                     kx = traj[i][:, 0]
#                     ky = traj[i][:, 1]
#                     kz = traj[i][:, 2]
#
#                     # Cast to desired datatype.
#                     kx = kx.astype(dtype)
#                     ky = ky.astype(dtype)
#                     kz = kz.astype(dtype)
#                     c_retrieved = c_retrieved.astype(complex_dtype)
#
#                     # Allocate memory for the uniform grid on the GPU.
#                     c_retrieved_gpu = to_gpu(c_retrieved)
#
#                     # Initialize the plan and set the points.
#                     plan = cufinufft(1, (N1, N2, N3), nb_channels, eps=eps, dtype=dtype)
#                     plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))
#
#                     # Execute the plan, reading from the strengths array c and storing the
#                     # result in fk_gpu.
#                     plan.execute(c_retrieved_gpu, fk_gpu)
#
#                     fk = np.squeeze(fk_gpu.get())
#
#                     fk_gpu.gpudata.free()
#                     c_retrieved_gpu.gpudata.free()
#
#                     if b1 is None:
#                         images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
#                     else:
#                         images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)
#
#                     plan.__del__()
#                 else:
#                     #fk = np.zeros(output_shape,dtype=complex_dtype)
#                     for j in tqdm(range(nb_channels)):
#                         fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)
#                         c_retrieved = kdata[j, i, :]
#                         kx = traj[i][:, 0]
#                         ky = traj[i][:, 1]
#                         kz = traj[i][:, 2]
#
#                         # Cast to desired datatype.
#                         kx = kx.astype(dtype)
#                         ky = ky.astype(dtype)
#                         kz = kz.astype(dtype)
#                         c_retrieved = c_retrieved.astype(complex_dtype)
#
#                         # Allocate memory for the uniform grid on the GPU.
#                         c_retrieved_gpu = to_gpu(c_retrieved)
#
#                         # Initialize the plan and set the points.
#                         plan = cufinufft(1, (N1, N2, N3), 1, eps=eps, dtype=dtype)
#                         plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))
#
#                         # Execute the plan, reading from the strengths array c and storing the
#                         # result in fk_gpu.
#                         plan.execute(c_retrieved_gpu, fk_gpu)
#
#                         fk = np.squeeze(fk_gpu.get())
#
#                         fk_gpu.gpudata.free()
#                         c_retrieved_gpu.gpudata.free()
#
#                         if b1 is None:
#                             images_series_rebuilt[i] += np.abs(fk) ** 2
#                         else:
#                             images_series_rebuilt[i] += b1[j].conj() * fk
#                         plan.__del__()
#
#                     if b1 is None:
#                         images_series_rebuilt[i] = np.sqrt(images_series_rebuilt[i])
#
#
#
#
#         del kdata
#         gc.collect()
#
#         if flushed:
#             images_series_rebuilt.flush()
#             del images_series_rebuilt
#             images_series_rebuilt=np.memmap(memmap_file,dtype="complex64",mode="r",shape=output_shape)
#
#         if (normalize_volumes)and(b1 is not None):
#             print("Normalizing by Coil Sensi")
#             if light_memory_usage:
#                 b1_norm = np.sum(np.abs(b1) ** 2)
#                 for i in tqdm(range(images_series_rebuilt.shape[0])):
#                     images_series_rebuilt[i] = images_series_rebuilt[i] / b1_norm
#             else:
#                 images_series_rebuilt /= np.sum(np.abs(b1) ** 2)
#
#
#
#     #images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))
#
#
#
#     return images_series_rebuilt


def convolution_kernel_radial_single_channel(traj,dk,npoint,size,density_adj=False):
    dtheta = 1
    dz = 1 / (2 * np.pi)


    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        #density=np.expand_dims(axis=0)

        dk =(np.reshape(dk, (-1, npoint)) * density).flatten()


    dk *= dz * dtheta / (2*npoint)

    if dk.dtype == "complex64":
        traj=traj.astype("float32")
        print(traj.dtype)

    fk = finufft.nufft3d1(traj[:, 2], traj[:, 0], traj[:, 1], dk, size)
    return fk





def simulate_radial_undersampled_images_multi(kdata, trajectory, size, density_adj=True, eps=1e-6,
                                              is_theta_z_adjusted=False, b1=None, ntimesteps=175, useGPU=False,
                                              memmap_file=None, normalize_kdata=False, light_memory_usage=False,
                                              normalize_volumes=True):
    # Deals with single channel data / howver kdata can be a list of arrays (meaning each timestep does not need to have the same number of spokes/partitions)

    # if light_memory_usage and not(useGPU):
    #    print("Warning : light memory usage is not used without GPU")

    traj = trajectory.get_traj_for_reconstruction(ntimesteps)
    print(traj[0].shape)
    npoint = trajectory.paramDict["npoint"]
    nb_allspokes = trajectory.paramDict["total_nspokes"]
    nb_channels = len(kdata)
    nspoke = int(nb_allspokes / ntimesteps)



    if not (is_theta_z_adjusted):
        dtheta = np.pi / nspoke
        dz = 1 / trajectory.paramDict["nb_rep"]

    else:
        dtheta = 1
        dz = 1/(2*np.pi)

    if not (len(kdata[0]) == len(traj)):
        kdata = kdata.reshape(nb_channels, len(traj), -1)

    if type(density_adj) is bool:
        if density_adj:
            density_adj="Radial"

    if density_adj=="Radial":
        density = np.abs(np.linspace(-1, 1, npoint))
        #density=np.expand_dims(axis=0)
        for j in tqdm(range(nb_channels)):
            kdata[j] =[(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata[j]]

    elif density_adj=="Voronoi":
        print("Calculating Voronoi Density Adj")
        density=[]
        for i in tqdm(range(len(traj))):
            curr_dens=voronoi_volumes_freud(traj[i])
            curr_dens_shape=curr_dens.shape
            curr_dens=curr_dens.reshape(-1,npoint)
            curr_dens[:,0]=curr_dens[:,1]
            curr_dens[:, npoint-1] = curr_dens[:, npoint-2]
            curr_dens=curr_dens.reshape(curr_dens_shape)
            curr_dens /= curr_dens.sum()
            density.append(curr_dens)

        # density = [
        #     voronoi_volumes_freud(traj[i]) for i in
        #     tqdm(range(len(traj)))]
        for j in tqdm(range(nb_channels)):
            kdata[j] = [k * density[i] for i, k in enumerate(kdata[j])]

    if kdata[0][0].dtype == "complex64":
        try:
            traj=traj.astype("float32")
        except:
            for i in range(traj.shape[0]):
                traj[i] = traj[i].astype("float32")
        print(traj[0].dtype)

    if normalize_kdata:
        print("Normalizing Kdata")
        if not (light_memory_usage):
            C = np.mean(np.linalg.norm(kdata, axis=1), axis=-1)
            kdata /= np.expand_dims(C, axis=(1, 2))
        else:
            for i in tqdm(range(nb_channels)):
                #try:
                #    kdata[i]/=np.mean(np.linalg.norm(kdata[i],axis=0))
                #except:
                kdata[i] /= np.sum(np.abs(np.array(list((itertools.chain(*kdata[i]))))))
    else:
        for i in tqdm(range(nb_channels)):
            kdata[i] *= dz * dtheta / (2*npoint)

    # kdata = (normalize_image_series(np.array(kdata)))

    output_shape = (ntimesteps,) + size

    flushed = False

    if memmap_file is not None:
        from tempfile import mkdtemp
        import os.path as path
        file_memmap = path.join(mkdtemp(), "memmap_volumes.dat")
        images_series_rebuilt = np.memmap(file_memmap, dtype="complex64", mode="w+", shape=output_shape)

    else:
        images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    print("Performing NUFFT")
    if traj[0].shape[-1] == 2:  # 2D

        for i, t in tqdm(enumerate(traj)):
            fk = finufft.nufft2d1(t[:, 0], t[:, 1], kdata[:, i, :], size)

            # images_series_rebuilt = np.moveaxis(images_series_rebuilt, 0, 1)
            if b1 is None:
                images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
            else:
                images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)

    elif traj[0].shape[-1] == 3:  # 3D
        if not (useGPU):

            for i, t in tqdm(enumerate(traj)):
                if not (light_memory_usage):

                    fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kdata[:, i, :], size)
                    if b1 is None:
                        images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
                    else:
                        images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)

                else:
                    flush_condition = (memmap_file is not None) and (
                                (psutil.virtual_memory().cached + psutil.virtual_memory().free) / 1e9 < 2) and (
                                          not (flushed))
                    if flush_condition:
                        print("Flushed Memory")
                        offset = i * images_series_rebuilt.itemsize * i * np.prod(images_series_rebuilt.shape[1:])
                        i0 = i
                        new_shape = (output_shape[0] - i,) + output_shape[1:]
                        images_series_rebuilt.flush()
                        del images_series_rebuilt
                        flushed = True
                        normalize_volumes = False
                        images_series_rebuilt = np.memmap(memmap_file, dtype="complex64", mode="r+", shape=new_shape,
                                                          offset=offset)

                    if flushed:
                        for j in tqdm(range(nb_channels)):
                            print(t.shape)
                            print(kdata[j][i].shape)
                            fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kdata[j][i], size)
                            if b1 is None:
                                images_series_rebuilt[i - i0] += np.abs(fk) ** 2
                            else:
                                images_series_rebuilt[i - i0] += b1[j].conj() * fk

                        if b1 is None:
                            images_series_rebuilt[i] = np.sqrt(images_series_rebuilt[i])

                    else:
                        for j in tqdm(range(nb_channels)):

                            index_non_zero_kdata=np.nonzero(kdata[j][i])
                            kdata_current=kdata[j][i][index_non_zero_kdata]
                            t_current=t[index_non_zero_kdata]
                            print(t_current.shape)
                            print(kdata_current.shape)
                            fk = finufft.nufft3d1(t_current[:, 2], t_current[:, 0], t_current[:, 1], kdata_current, size)
                            if b1 is None:
                                images_series_rebuilt[i] += np.abs(fk) ** 2
                            else:
                                images_series_rebuilt[i] += b1[j].conj() * fk

                        if b1 is None:
                            images_series_rebuilt[i] = np.sqrt(images_series_rebuilt[i])
        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64

            for i in tqdm(list(range(kdata[0].shape[0]))):
                if not (light_memory_usage):
                    fk_gpu = GPUArray((nb_channels, N1, N2, N3), dtype=complex_dtype)
                    c_retrieved = kdata[:, i, :]
                    kx = traj[i][:, 0]
                    ky = traj[i][:, 1]
                    kz = traj[i][:, 2]

                    # Cast to desired datatype.
                    kx = kx.astype(dtype)
                    ky = ky.astype(dtype)
                    kz = kz.astype(dtype)
                    c_retrieved = c_retrieved.astype(complex_dtype)

                    # Allocate memory for the uniform grid on the GPU.
                    c_retrieved_gpu = to_gpu(c_retrieved)

                    # Initialize the plan and set the points.
                    plan = cufinufft(1, (N1, N2, N3), nb_channels, eps=eps, dtype=dtype)
                    plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

                    # Execute the plan, reading from the strengths array c and storing the
                    # result in fk_gpu.
                    plan.execute(c_retrieved_gpu, fk_gpu)

                    fk = np.squeeze(fk_gpu.get())

                    fk_gpu.gpudata.free()
                    c_retrieved_gpu.gpudata.free()

                    if b1 is None:
                        images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
                    else:
                        images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)

                    plan.__del__()
                else:
                    # fk = np.zeros(output_shape,dtype=complex_dtype)
                    for j in tqdm(range(nb_channels)):
                        fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)
                        index_non_zero_kdata=np.nonzero(kdata[j][i])
                        c_retrieved = kdata[j][i][index_non_zero_kdata]
                        kx = traj[i][index_non_zero_kdata][:, 0]
                        ky = traj[i][index_non_zero_kdata][:, 1]
                        kz = traj[i][index_non_zero_kdata][:, 2]

                        # Cast to desired datatype.
                        kx = kx.astype(dtype)
                        ky = ky.astype(dtype)
                        kz = kz.astype(dtype)
                        c_retrieved = c_retrieved.astype(complex_dtype)

                        # Allocate memory for the uniform grid on the GPU.
                        c_retrieved_gpu = to_gpu(c_retrieved)

                        # Initialize the plan and set the points.
                        plan = cufinufft(1, (N1, N2, N3), 1, eps=eps, dtype=dtype)
                        plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

                        # Execute the plan, reading from the strengths array c and storing the
                        # result in fk_gpu.
                        plan.execute(c_retrieved_gpu, fk_gpu)

                        fk = np.squeeze(fk_gpu.get())

                        fk_gpu.gpudata.free()
                        c_retrieved_gpu.gpudata.free()

                        if b1 is None:
                            images_series_rebuilt[i] += np.abs(fk) ** 2
                        else:
                            images_series_rebuilt[i] += b1[j].conj() * fk
                        plan.__del__()

                    if b1 is None:
                        images_series_rebuilt[i] = np.sqrt(images_series_rebuilt[i])

        del kdata
        gc.collect()

        if flushed:
            images_series_rebuilt.flush()
            del images_series_rebuilt
            images_series_rebuilt = np.memmap(memmap_file, dtype="complex64", mode="r", shape=output_shape)

        if (normalize_volumes) and (b1 is not None):
            print("Normalizing by Coil Sensi")
            if light_memory_usage:
                b1_norm = np.sum(np.abs(b1) ** 2,axis=0)
                for i in tqdm(range(images_series_rebuilt.shape[0])):
                    images_series_rebuilt[i] /= b1_norm
            else:
                images_series_rebuilt /= np.expand_dims(np.sum(np.abs(b1) ** 2,axis=0),axis=0)


    # images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return images_series_rebuilt

def simulate_undersampled_images(kdata,trajectory,size,density_adj=True,useGPU=False,eps=1e-6):
    # Strong Assumption : from one time step to the other, the sampling is just rotated, hence voronoi volumes can be calculated only once
    print("Simulating Undersampled Images")
    traj=trajectory.get_traj_for_reconstruction()

    if not(len(kdata)==len(traj)):
        kdata=np.array(kdata).reshape(len(traj),-1)

    kdata = np.array(kdata) / (2*np.pi) **2

    if density_adj:
        print("Performing density adjustment using Voronoi cells")
        #density = voronoi_volumes(np.transpose(np.array([traj[0,:, 0], traj[0,:, 1]])),min_x=-np.pi,min_y=-np.pi,max_x=np.pi,max_y=np.pi)[0]
        #if traj[0].shape[-1] == 2:#2D
        #    density = [voronoi_volumes(traj[i],min_x=-np.pi,min_y=-np.pi,max_x=np.pi,max_y=np.pi)[0] for i in tqdm(range(len(kdata)))]
        #else:#3D
        density = [
                voronoi_volumes(traj[i], min_x=-np.pi, min_y=-np.pi,
                                max_x=np.pi, max_y=np.pi, min_z=-np.pi, max_z=np.pi)[0] for i in
                tqdm(range(len(kdata)))]
        kdata = np.array([k * density[i] for i, k in enumerate(kdata)])

    #kdata = (normalize_image_series(np.array(kdata)))

    if traj[0].shape[-1] == 2:  # 2D
        if not (useGPU):
            images_series_rebuilt = [
                finufft.nufft2d1(t[:, 0], t[:, 1], s, size)
                for t, s in zip(traj, kdata)
            ]
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2), dtype=complex_dtype)
            images_GPU = []
            for i in list(range(len(kdata))):
                c_retrieved = kdata[i]
                kx = traj[i, :, 0]
                ky = traj[i, :, 1]

                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.

                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(to_gpu(c_retrieved), fk_gpu)

                fk = np.squeeze(fk_gpu.get())
                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt = np.array(images_GPU)
    elif traj[0].shape[-1] == 3:  # 3D
        if not (useGPU):
            # images_series_rebuilt = [
            #    finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size)
            #    for t, s in zip(traj, kdata)
            # ]

            images_series_rebuilt = []
            for t, s in tqdm(zip(traj, kdata)):
                images_series_rebuilt.append(finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], s, size))

        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64

            images_GPU = []
            for i in list(range(len(kdata))):
                fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)
                c_retrieved = kdata[i]
                kx = traj[i, :, 0]
                ky = traj[i, :, 1]
                kz = traj[i, :, 2]

                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                kz = kz.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.
                c_retrieved_gpu = to_gpu(c_retrieved)

                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2, N3), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(c_retrieved_gpu, fk_gpu)

                fk = np.squeeze(fk_gpu.get())

                fk_gpu.gpudata.free()
                c_retrieved_gpu.gpudata.free()

                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt = np.array(images_GPU)

    # images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return np.array(images_series_rebuilt)

def plot_evolution_params(map_ref, mask_ref, all_maps, maskROI=None, adj_wT1=True, title="Evolution",metric="R2", fat_threshold=0.7,
                          proj_on_mask1=True, fontsize=5, figsize=(15, 40),save=False):
    keys_1 = set(map_ref.keys())
    keys_2 = set(all_maps[0][0].keys())
    nb_keys = len(keys_1 & keys_2)
    fig, ax = plt.subplots(nb_keys, figsize=figsize)

    if maskROI is None:
        maskROI = buildROImask(map_ref)

    for i, k in enumerate(keys_1 & keys_2):
        print(i)
        result_list = []
        std_list = []
        it_list = []
        for it, value in all_maps.items():

            map2 = value[0]
            mask2 = value[1] > 0

            mask_union = mask_ref | mask2
            mat_obs = makevol(map_ref[k], mask_ref)
            mat_pred = makevol(map2[k], mask2)
            mat_ROI = makevol(maskROI, mask_ref)
            if proj_on_mask1:
                mat_pred = mat_pred * (mask_ref * 1)
                mat_obs = mat_obs * (mask_ref * 1)
                mat_ROI = mat_ROI * (mask_ref * 1)
                mask_union = mask_ref

            obs = mat_obs[mask_union]
            pred = mat_pred[mask_union]
            maskROI_current = mat_ROI[mask_union]

            if adj_wT1 and k == "wT1":
                ff = makevol(map_ref["ff"], mask_ref)
                ff = ff[mask_union]
                obs = obs[ff < fat_threshold]
                pred = pred[ff < fat_threshold]
                maskROI_current = maskROI_current[ff < fat_threshold]

            df_obs = pd.DataFrame(columns=["Data", "Groups"],
                                  data=np.stack([obs.flatten(), maskROI_current.flatten()], axis=-1))
            df_pred = pd.DataFrame(columns=["Data", "Groups"],
                                   data=np.stack([pred.flatten(), maskROI_current.flatten()], axis=-1))
            mean_obs = np.array(df_obs.groupby("Groups").mean())[1:]
            mean_pred = np.array(df_pred.groupby("Groups").mean())[1:]

            x_min = np.min(mean_obs)
            x_max = np.max(mean_pred)

            if x_min == x_max:
                fig.delaxes(ax[i])
                break

            if metric == "R2":
                mean = np.mean(mean_obs)
                ss_tot = np.sum((mean_obs - mean) ** 2)
                ss_res = np.sum((mean_obs - mean_pred) ** 2)
                bias = np.mean((mean_pred - mean_obs))
                r_2 = 1 - ss_res / ss_tot
                result_list.append(r_2)
                it_list.append(it)



            elif metric == "RMSE":
                print(obs.shape)
                print(pred.shape)
                df_error = pd.DataFrame(columns=["Data", "Groups"],
                                        data=np.stack(
                                            [(pred.flatten() - obs.flatten()) ** 2, maskROI_current.flatten()],
                                            axis=-1))
                errors = np.sqrt(np.array(df_error.groupby("Groups").mean())[1:])
                error = np.mean(errors)
                std_error = np.std(errors)
                result_list.append(error)
                std_list.append(std_error)
                it_list.append(it)


            else:
                raise ValueError("Metric should be RMSE or R2")

        if x_min == x_max:
            continue

        if metric == "R2":
            it_list, result_list = tuple(zip(*sorted(zip(it_list, result_list))))
            ax[i].plot(it_list, result_list, "r")
        elif metric == "RMSE":
            # ax[i].plot(range(n_it), result_list, "r")
            it_list, result_list,std_list=tuple(zip(*sorted(zip(it_list, result_list,std_list))))
            ax[i].errorbar(it_list, result_list, std_list)

        ax[i].set_title(k + " Evolution over Iteration", fontsize=2 * fontsize)
        ax[i].tick_params(axis='x', labelsize=fontsize)
        ax[i].tick_params(axis='y', labelsize=fontsize)

    plt.suptitle("{} : {}".format(title,metric))

    if save :
        plt.savefig("./figures/{} : {}".format(title,metric))


def create_cuda_context():
    pycuda.driver.init()
    dev=pycuda.driver.Device(0)
    context=dev.make_context()
    return context

def wavelet_denoising(image,retained_coef=0.99,level=3):
    c = pywt.wavedec2(image, 'db2', level=level)
    arr, slices = pywt.coeffs_to_array(c)

    #Selection of the appropriate cut off
    sorted_coef = np.sort(np.abs(arr.flatten()))[::-1]
    cum_sum = np.cumsum(sorted_coef)
    cum_sum = cum_sum / cum_sum[-1]
    index_cut = (cum_sum > retained_coef).sum()
    value = sorted_coef[index_cut]
    # Cut off
    arr_cut = arr.copy()
    arr_cut[np.abs(arr_cut) < value] = 0

    # sorted_coef = np.sort(np.abs(arr_cut.flatten()))
    # plt.plot(sorted_coef)

    #Image reconstruction
    coef_cut = pywt.array_to_coeffs(arr_cut, slices, output_format='wavedec2')
    image_cut = pywt.waverec2(coef_cut, 'db2')
    return image_cut

def calculate_condition_mvt_correction(t,transf,perc):


    shifts = transf(t.flatten().reshape(-1, 1))[:, 1]

    # traj_for_selection=traj_for_selection.reshape(t.shape+traj_for_selection.shape[-2:])
    # traj_for_selection=traj_for_selection.reshape((m.paramDict["nb_rep"],ntimesteps,-1)+traj_for_selection.shape[-2:])
    #
    # kdata_for_selection=kdata_for_selection.reshape(t.shape+kdata_for_selection.shape[-1:])
    # kdata_for_selection=kdata_for_selection.reshape((m.paramDict["nb_rep"],ntimesteps,-1)+kdata_for_selection.shape[-1:])

    threshold = np.percentile(shifts, perc)
    cond = (shifts > threshold)
    return cond


def correct_mvt_kdata(kdata,trajectory,cond,ntimesteps,density_adj=True,log=False):

    kdata=np.array(kdata)
    traj=trajectory.get_traj()

    mode=trajectory.paramDict["mode"]
    incoherent=trajectory.paramDict["incoherent"]


    nb_rep = int(cond.shape[0]/traj.shape[0])
    npoint = int(traj.shape[1] / nb_rep)
    nspoke = int(traj.shape[0] / ntimesteps)

    traj_for_selection = np.array(groupby(traj, npoint, axis=1))
    kdata_for_selection = np.array(groupby(kdata, npoint, axis=1))

    traj_for_selection = traj_for_selection.reshape(cond.shape[0], -1, 3)
    kdata_for_selection = kdata_for_selection.reshape(cond.shape[0], -1)

    indices = np.unravel_index(np.argwhere(cond).T,(nb_rep,ntimesteps,nspoke))
    retained_indices = np.squeeze(np.array(indices).T)

    traj_retained = traj_for_selection[cond, :, :]
    kdata_retained = kdata_for_selection[cond, :]

    ## DENSITY CORRECTION
    if density_adj:
        df = pd.DataFrame(columns=["rep", "ts", "spoke", "kz", "theta"], index=range(nb_rep * ntimesteps * nspoke))
        df["rep"] = np.repeat(list(range(nb_rep)), ntimesteps * nspoke)
        df["ts"] = list(np.repeat(list(range(ntimesteps)), (nspoke))) * nb_rep
        df["spoke"] = list(range(nspoke)) * nb_rep * ntimesteps

        df["kz"] = traj_for_selection[:, :, 2][:, 0]
        golden_angle = 111.246 * np.pi / 180

        if not(incoherent):
            df["theta"] = np.array(list(np.mod(np.arange(0, int(df.shape[0] / nb_rep)) * golden_angle, np.pi)) * nb_rep)
        elif incoherent:
            if mode=="old":
                df["theta"] = np.array(list(np.mod(np.arange(0, int(df.shape[0])) * golden_angle, np.pi)))
            elif mode=="new":
                df["theta"] = np.mod((np.array(list(np.arange(0, nb_rep) * golden_angle)).reshape(-1,1)+np.array(list(np.arange(0, int(df.shape[0] / nb_rep)) * golden_angle)).reshape(1,-1)).flatten(),np.pi)


        df_retained = df.iloc[np.nonzero(cond)]
        kz_by_timestep = df_retained.groupby("ts")["kz"].unique()
        theta_by_rep_timestep = df_retained.groupby(["ts", "rep"])["theta"].unique()

        df_retained = df_retained.join(kz_by_timestep, on="ts", rsuffix="_s")
        df_retained = df_retained.join(theta_by_rep_timestep, on=["ts", "rep"], rsuffix="_s")

        # Theta weighting
        df_retained["theta_s"] = df_retained["theta_s"].apply(lambda x: np.sort(x))
        #df_retained["theta_s"] = df_retained["theta_s"].apply(
        #    lambda x: np.concatenate([[x[-1] - np.pi], x, [x[0] + np.pi]]))
        df_retained["theta_s"] = df_retained["theta_s"].apply(
                lambda x: np.unique(np.concatenate([[0], x, [np.pi]])))
        diff_theta=(df_retained.theta - df_retained["theta_s"])
        theta_inside_boundary=(df_retained["theta"]!=0)*(df_retained["theta"]!=np.pi)
        df_retained["theta_inside_boundary"] = theta_inside_boundary

        min_theta = df_retained.groupby(["ts", "rep"])["theta"].min()
        max_theta = df_retained.groupby(["ts", "rep"])["theta"].max()
        df_retained = df_retained.join(min_theta, on=["ts", "rep"], rsuffix="_min")
        df_retained = df_retained.join(max_theta, on=["ts", "rep"], rsuffix="_max")
        is_min_theta = (df_retained["theta"]==df_retained["theta_min"])
        is_max_theta = (df_retained["theta"] == df_retained["theta_max"])
        df_retained["is_min_theta"] = is_min_theta
        df_retained["is_max_theta"] = is_max_theta

        df_retained["theta_weight"] = theta_inside_boundary*diff_theta.apply(lambda x: (np.sort(x[x>=0])[1]+np.sort(-x[x<=0])[1])/2 if ((x>=0).sum()>1) and ((x<=0).sum()>1) else 0)+\
                                      (1-theta_inside_boundary)*diff_theta.apply(lambda x: np.sort(np.abs(x))[1]/2)

        df_retained["theta_weight_before_correction"] = df_retained["theta_weight"]

        df_retained["theta_weight"] = df_retained["theta_weight"]+ (theta_inside_boundary)* ((is_min_theta)*df_retained["theta"]+(is_max_theta)*(np.pi-df_retained["theta"]))/2

        df_retained.loc[df_retained["theta_weight"].isna(), "theta_weight"] = 1.0
        sum_weights = df_retained.groupby(["ts", "rep"])["theta_weight"].sum()
        df_retained = df_retained.join(sum_weights, on=["ts", "rep"], rsuffix="_sum")
        #df_retained["theta_weight"] = df_retained["theta_weight"] / df_retained["theta_weight_sum"]

        # KZ weighting
        #df_retained.loc[df_retained.ts == 138].to_clipboard()
        df_retained["kz_s"] = df_retained["kz_s"].apply(lambda x: np.unique(np.concatenate([[-np.pi], x, [np.pi]])))
        diff_kz=(df_retained.kz - df_retained["kz_s"])
        kz_inside_boundary=(df_retained["kz"].abs()!=np.pi)
        df_retained["kz_inside_boundary"]=kz_inside_boundary

        min_kz = df_retained.groupby(["ts"])["kz"].min()
        max_kz = df_retained.groupby(["ts"])["kz"].max()
        df_retained = df_retained.join(min_kz, on=["ts"], rsuffix="_min")
        df_retained = df_retained.join(max_kz, on=["ts"], rsuffix="_max")

        is_min_kz = (df_retained["kz"] == df_retained["kz_min"])
        is_max_kz = (df_retained["kz"] == df_retained["kz_max"])

        df_retained["is_min_kz"] = is_min_kz
        df_retained["is_max_kz"] = is_max_kz

        df_retained["kz_weight"] = kz_inside_boundary*diff_kz.apply(lambda x: (np.sort(x[x >= 0])[1] + np.sort(-x[x <= 0])[1]) / 2 if ((x >= 0).sum() > 1) and (
            ((x <= 0).sum() > 1)) else 0)+(1-kz_inside_boundary)*diff_kz.apply(lambda x: np.sort(np.abs(x))[1]/2)



        df_retained["kz_weight"] = df_retained["kz_weight"] + (kz_inside_boundary) * (
                    (is_min_kz) * (df_retained["kz"]+np.pi) + (is_max_kz) * (np.pi - df_retained["kz"])) / 2

        df_retained.loc[df_retained["kz_weight"].isna(), "kz_weight"] = 1.0
        sum_weights = df_retained.drop_duplicates(subset=["kz","ts"])
        sum_weights = sum_weights.groupby(["ts"])["kz_weight"].apply(lambda x: x.sum())
        df_retained = df_retained.join(sum_weights, on=["ts"], rsuffix="_sum")
        #df_retained["kz_weight"] = df_retained["kz_weight"] / df_retained["kz_weight_sum"]

        if log:
            now = datetime.now()
            df_retained.to_csv("./log/df_density_correction_{}.csv".format(now))

    dico_traj = {}
    dico_kdata = {}
    for i, index in enumerate(retained_indices):
        curr_slice = index[0]
        ts = index[1]
        curr_spoke = index[2]
        if ts not in dico_traj:
            dico_traj[ts] = []
            dico_kdata[ts] = []

        # dico_traj[ts]=[*dico_traj[ts],*traj_retained[i]]
        # dico_kdata[ts]=[*dico_kdata[ts],*kdata_retained[i]]



        dico_traj[ts].append(traj_retained[i])
        if density_adj:
            theta_weight = df_retained.iloc[i]["theta_weight"]
            kz_weight = df_retained.iloc[i]["kz_weight"]
            dico_kdata[ts].append(kdata_retained[i]*theta_weight*kz_weight)
        else:
            dico_kdata[ts].append(kdata_retained[i])

    retained_timesteps = list(dico_traj.keys())
    retained_timesteps.sort()

    traj_retained_final = []
    kdata_retained_final = []

    # for ts in tqdm(range(len(retained_timesteps))):
    #     traj_retained_final.append(np.array(dico_traj[ts]))
    #     kdata_retained_final.append(np.array(dico_kdata[ts]))

    for ts in tqdm(retained_timesteps):
        traj_retained_final.append(np.array(dico_traj[ts]).flatten().reshape(-1, 3))
        kdata_retained_final.append(np.array(dico_kdata[ts]).flatten())

    traj_retained_final = np.array(traj_retained_final)
    kdata_retained_final = np.array(kdata_retained_final)

    return kdata_retained_final,traj_retained_final,retained_timesteps


def plot_image_grid(list_images,nb_row_col,figsize=(10,10),title="",cmap=None,save_file=None):
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=nb_row_col,  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, list_images):
        ax.imshow(im,cmap=cmap)

    plt.show()
    if save_file is not None:
        plt.savefig(save_file)

def calculate_sensitivity_map(kdata,trajectory,res=16,image_size=(256,256)):
    traj_all = trajectory.get_traj()
    traj_all=traj_all.reshape(-1,traj_all.shape[-1])
    npoint = kdata.shape[-1]
    center_res = int(npoint / 2 - 1)


    kdata_for_sensi = np.zeros(kdata.shape, dtype=np.complex128)

    if kdata.ndim==3:#Tried :: syntax but somehow it introduces errors in the allocation
        kdata_for_sensi[:,:, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:,:,
                                                                                   (center_res - int(res / 2)):(
                                                                                               center_res + int(
                                                                                           res / 2))]
    else:
        kdata_for_sensi[:, :,:, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:, :,:,
                                                                                         (center_res - int(res / 2)):(
                                                                                                 center_res + int(
                                                                                             res / 2))]

    kdata_all = kdata_for_sensi.reshape(np.prod(kdata.shape[:-2]), -1)
    print(traj_all.shape)
    print(kdata_all.shape)
    coil_sensitivity = finufft.nufft2d1(traj_all[:, 0], traj_all[:, 1], kdata_all, image_size)
    coil_sensitivity=coil_sensitivity.reshape(*kdata.shape[:-2],*image_size)
    print(coil_sensitivity.shape)

    if coil_sensitivity.ndim==3:
        print("Ndim 3)")
        b1 = coil_sensitivity / np.linalg.norm(coil_sensitivity, axis=0)
        b1 = b1 / np.max(np.abs(b1.flatten()))
    else:#first dimension contains slices
        print("Ndim > 3")
        b1=coil_sensitivity.copy()
        for i in range(coil_sensitivity.shape[0]):
            b1[i]=coil_sensitivity[i] / np.linalg.norm(coil_sensitivity[i], axis=0)
            b1[i]=b1[i] / np.max(np.abs(b1[i].flatten()))
    return b1

def calculate_sensitivity_map_3D(kdata,trajectory,res=16,image_size=(1,256,256),useGPU=False,eps=1e-6,light_memory_usage=False,density_adj=False):
    traj_all = trajectory.get_traj()
    traj_all = traj_all.reshape(-1, traj_all.shape[-1])
    npoint = kdata.shape[-1]
    nb_channels = kdata.shape[0]
    center_res = int(npoint / 2 - 1)

    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        # density=np.expand_dims(axis=0)
        for j in tqdm(range(nb_channels)):
            kdata[j] = np.array([(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata[j]]).reshape(kdata.shape[1:])

    if not(light_memory_usage):
        kdata_for_sensi = np.zeros(kdata.shape, dtype=np.complex128)
        kdata_for_sensi[:, :, :, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:,
                                                                                            :, :,
                                                                                            (center_res - int(res / 2)):(
                                                                                                    center_res + int(
                                                                                                res / 2))]

        del kdata
        kdata_for_sensi = kdata_for_sensi.reshape(nb_channels, -1)

    print("Performing NUFFT on filtered data for sensitivity calculation")
    if not(useGPU):
        if not(light_memory_usage):
            index_non_zero_kdata=np.nonzero(kdata_for_sensi[0])
            kdata_for_sensi=kdata_for_sensi[index_non_zero_kdata]
            traj_all=traj_all[index_non_zero_kdata]
            coil_sensitivity = finufft.nufft3d1(traj_all[:, 2], traj_all[:, 0], traj_all[:, 1], kdata_for_sensi, image_size)
        else:
            coil_sensitivity=np.zeros((nb_channels,)+image_size,dtype=np.complex128)
            kdata_for_sensi = np.zeros(kdata.shape[1:], dtype=np.complex128)
            for i in tqdm(range(nb_channels)):
                kdata_for_sensi[:, :, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[i,
                                                                                                    :, :,
                                                                                                    (center_res - int(
                                                                                                        res / 2)):(
                                                                                                            center_res + int(
                                                                                                        res / 2))]
                flattened_kdata_for_sensi=kdata_for_sensi.flatten()
                index_non_zero_kdata = np.nonzero(flattened_kdata_for_sensi)
                flattened_kdata_for_sensi=flattened_kdata_for_sensi[index_non_zero_kdata]
                traj_current = traj_all[index_non_zero_kdata]
                coil_sensitivity[i] = finufft.nufft3d1(traj_current[:, 2], traj_current[:, 0], traj_current[:, 1], flattened_kdata_for_sensi, image_size)


    else:
        N1, N2, N3 = image_size[0], image_size[1], image_size[2]
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64

        fk_gpu = GPUArray((nb_channels, N1, N2, N3), dtype=complex_dtype)

        kx = traj_all[:, 0]
        ky = traj_all[:, 1]
        kz = traj_all[:, 2]

        kx = kx.astype(dtype)
        ky = ky.astype(dtype)
        kz = kz.astype(dtype)

        c_retrieved_gpu = to_gpu(kdata_for_sensi.astype(complex_dtype))
        plan = cufinufft(1, (N1, N2, N3), nb_channels, eps=eps, dtype=dtype)
        plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

        plan.execute(c_retrieved_gpu, fk_gpu)

        coil_sensitivity = np.squeeze(fk_gpu.get())

        fk_gpu.gpudata.free()
        c_retrieved_gpu.gpudata.free()
        plan.__del__()

    del kdata_for_sensi
    print("Normalizing sensi")

    #b1 = coil_sensitivity / np.linalg.norm(coil_sensitivity, axis=0)
    #del coil_sensitivity
    #b1 = b1 / np.max(np.abs(b1.flatten()))
    coil_sensitivity /= np.linalg.norm(coil_sensitivity, axis=0)
    coil_sensitivity /= np.max(np.abs(coil_sensitivity.flatten()))


    return coil_sensitivity


def J_fourier(m, traj, kdata):

    Fu_m = finufft.nufft2d2(traj[:, 0], traj[:, 1], m)
    return np.linalg.norm(Fu_m - kdata) ** 2


def J_sparse(m, typ="db4", mode="periodization", mu=1e-6):
    return N(coef_to_array(psi(m, typ, mode=mode)), mu)


def grad_J_sparse(m, typ="db4", mode="periodization", mu=1e-6):
    psi_m = coef_to_array(psi(m, typ, mode=mode))
    return np.real(inv_psi(array_to_coef(W(psi_m, mu) * psi_m)))


def grad_J_fourier(m, traj, kdata, npoint=512, density_adj=True):
    image_size = m.shape
    Fu_m = finufft.nufft2d2(traj[:, 0], traj[:, 1], m)
    kdata_error = Fu_m - kdata
    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        kdata_error = (np.reshape(kdata_error, (-1, npoint)) * density).flatten()
    error_volume = finufft.nufft2d1(traj[:, 0], traj[:, 1], kdata_error, image_size)

    return 2 * np.real(error_volume)

def psi(m,typ='db4',mode="periodization"):
    coef = pywt.dwt2(m, typ,mode=mode)
    return coef

def inv_psi(c,typ='db4',mode="periodization"):
    m = pywt.idwt2(c, typ,mode=mode)
    return m

def N(x,mu=1e-6):
    return np.sum(np.sqrt(np.abs(x)**2+mu))

def W(x,mu=1e-6):
    return 1/np.sqrt(np.abs(x)**2+mu)

def coef_to_array(c):
    cA, (cH, cV, cD) = c
    image_1 = np.concatenate([cA,cH],axis=1)
    image_2 = np.concatenate([cV,cD],axis=1)
    psi_m = np.concatenate([image_1,image_2],axis=0)
    return psi_m

def array_to_coef(array):
    N = array.shape[0]
    mid = int(N/2)
    cA = array[:mid,:mid]
    cH = array[:mid,mid:N]
    cV = array[mid:N,:mid]
    cD = array[mid:N,mid:N]
    c=cA, (cH, cV, cD)
    return c


def conjgrad(J,grad_J,m0,tolgrad=1e-4,maxiter=100,alpha=0.05,beta=0.6,t0=1,log=False,plot=False,filename_save=None):
    '''
    J : function from W (domain of m) to R
    grad_J : function from W to W - gradient of J
    m0 : initial value of m
    '''
    k=0
    m=m0
    if log:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        norm_g_list=[]

    g=grad_J(m)
    d_m=-g
    #store = [m]

    if plot:
        plt.ion()
        fig, axs = plt.subplots(1, 2, figsize=(30, 10))
        axs[0].set_title("Evolution of cost function")
    while (np.linalg.norm(g)>tolgrad)and(k<maxiter):
        norm_g = np.linalg.norm(g)
        if log:
            print("################ Iter {} ##################".format(k))
            norm_g_list.append(norm_g)
        print("Grad norm for iter {}: {}".format(k,norm_g))
        if k%10==0:
            print(k)
            if filename_save is not None:
                np.save(filename_save,m)
        t = t0
        J_m = J(m)
        print("J for iter {}: {}".format(k,J_m))
        J_m_next = J(m+t*d_m)
        slope = np.real(np.dot(g.flatten(),d_m.flatten()))
        if plot:
            axs[0].scatter(k,J_m,c="r",marker="+")
            axs[1].cla()
            axs[1].set_title("Line search for iteration {}".format(k))
            t_array = np.arange(0.,t0,t0/100)
            axs[1].plot(t_array,J_m+t_array*slope)
            axs[1].scatter(0,J_m,c="b",marker="x")
            plt.draw()

        while(J_m_next>J_m+alpha*t*slope):
            print(t)
            t = beta*t
            if plot:
                axs[1].scatter(t,J_m_next,c="b",marker="x")
            J_m_next=J(m+t*d_m)




        m = m + t*d_m
        g_prev = g
        g = grad_J(m)
        gamma = np.linalg.norm(g)**2/np.linalg.norm(g_prev)**2
        d_m = -g + gamma*d_m
        k=k+1
        #store.append(m)

    if log:
        norm_g_list=np.array(norm_g_list)
        np.save('./logs/conjgrad_{}.npy'.format(date_time),norm_g_list)

    return m


def graddesc(J,grad_J,m0,tolgrad=1e-4,maxiter=100,alpha=0.05,beta=0.6,t0=1,log=False):
    '''
    J : function from W (domain of m) to R
    grad_J : function from W to W - gradient of J
    m0 : initial value of m
    '''
    k=0
    m=m0
    if log:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        norm_g_list=[]

    g=grad_J(m)
    d_m=-g
    #store = [m]

    while (np.linalg.norm(g)>tolgrad)and(k<maxiter):
        norm_g = np.linalg.norm(g)
        if log:
            print("################ Iter {} ##################".format(k))
            norm_g_list.append(norm_g)
        print("Grad norm for iter {}: {}".format(k,norm_g))
        if k%10==0:
            print(k)
        t = t0
        J_m = J(m)
        print("J for iter {}: {}".format(k,J_m))
        while(J(m+t*d_m)>J_m+alpha*t*np.real(np.dot(g.flatten(),d_m.flatten()))):
            print(t)
            t = beta*t

        m = m + t*d_m
        g = grad_J(m)
        d_m = -g
        k=k+1
        #store.append(m)

    if log:
        norm_g_list=np.array(norm_g_list)
        np.save('./logs/graddesc_{}.npy'.format(date_time),norm_g_list)

    return m

def graddesc_linsearch(J,grad_J,m0,tolgrad=1e-4,maxiter=100,alpha=0.05,beta=0.6,t0=1,log=False):
    '''
    J : function from W (domain of m) to R
    grad_J : function from W to W - gradient of J
    m0 : initial value of m
    '''
    k=0
    m=m0
    if log:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        norm_g_list=[]

    g=grad_J(m)
    d_m=-g
    #store = [m]

    while (np.linalg.norm(g)>tolgrad)and(k<maxiter):
        norm_g = np.linalg.norm(g)
        if log:
            print("################ Iter {} ##################".format(k))
            norm_g_list.append(norm_g)
        print("Grad norm for iter {}: {}".format(k,norm_g))
        if k%10==0:
            print(k)

        J_m = J(m)
        t = t0
        print("J for iter {}: {}".format(k,J_m))
        while(J(m+t*d_m)>J_m+alpha*t*np.real(np.dot(g.flatten(),d_m.flatten()))):
            print(t)
            t = beta*t

        m = m + t*d_m
        g = grad_J(m)
        d_m = -g
        k=k+1
        #store.append(m)

    if log:
        norm_g_list=np.array(norm_g_list)
        np.save('./logs/graddesc_linsearch_{}.npy'.format(date_time),norm_g_list)

    return m


def graddesc(J,grad_J,m0,tolgrad=1e-4,maxiter=100,alpha=0.1,log=False):
    '''
    J : function from W (domain of m) to R
    grad_J : function from W to W - gradient of J
    m0 : initial value of m
    '''
    k=0
    m=m0
    if log:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        norm_g_list=[]

    g=grad_J(m)
    d_m=-g
    #store = [m]

    while (np.linalg.norm(g)>tolgrad)and(k<maxiter):
        norm_g = np.linalg.norm(g)
        if log:
            print("################ Iter {} ##################".format(k))
            norm_g_list.append(norm_g)
        print("Grad norm for iter {}: {}".format(k,norm_g))
        if k%10==0:
            print(k)

        J_m = J(m)
        print("J for iter {}: {}".format(k,J_m))

        m = m + alpha*d_m/norm_g*np.linalg.norm(J_m)
        g = grad_J(m)
        d_m = -g
        k=k+1
        #store.append(m)

    if log:
        norm_g_list=np.array(norm_g_list)
        np.save('./logs/graddesc_{}.npy'.format(date_time),norm_g_list)

    return m

def simulate_image_series_from_maps(map_rebuilt,mask_rebuilt,window=8):
    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask_rebuilt > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    map_ = MapFromDict("RebuiltMapFromParam", paramMap=map_for_sim)
    map_.buildParamMap()

    map_.build_ref_images(seq=seq)
    rebuilt_image_series = map_.images_series
    rebuilt_image_series= [np.mean(gp, axis=0) for gp in groupby(rebuilt_image_series, window)]
    rebuilt_image_series=np.array(rebuilt_image_series)
    return rebuilt_image_series,map_for_sim


def calculate_sensitivity_map_3D_for_nav(kdata, trajectory, res=16, image_size=(400,)):
    traj = trajectory.get_traj()
    nb_channels = kdata.shape[0]
    npoint = kdata.shape[-1]
    nb_slices = kdata.shape[1]
    nb_gating_spokes = kdata.shape[2]
    center_res = int(npoint / 2 - 1)
    kdata = kdata.reshape((nb_channels, -1, npoint))
    b1_nav = np.zeros((nb_channels, kdata.shape[1],) + image_size, dtype=np.complex128)
    kdata_for_sensi = np.zeros(kdata.shape[1:], dtype=np.complex128)
    for i in tqdm(range(nb_channels)):
        kdata_for_sensi[:, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[i, :,
                                                                                      (center_res - int(res / 2)):(
                                                                                                  center_res + int(
                                                                                              res / 2))]
        b1_nav[i] = finufft.nufft1d1(traj[0, :, 2], kdata_for_sensi, image_size)

        print("Normalizing sensi")

        # b1 = coil_sensitivity / np.linalg.norm(coil_sensitivity, axis=0)
        # del coil_sensitivity
        # b1 = b1 / np.max(np.abs(b1.flatten()))
    b1_nav /= np.linalg.norm(b1_nav, axis=0)
    b1_nav /= np.max(np.abs(b1_nav.flatten()))

    b1_nav = b1_nav.reshape((nb_channels, nb_slices, nb_gating_spokes, int(npoint / 2)))
    return b1_nav


def simulate_nav_images_multi(kdata, trajectory, image_size=(400,), b1=None):
    traj = trajectory.get_traj()
    nb_channels = kdata.shape[0]
    npoint = kdata.shape[-1]
    nb_slices = kdata.shape[1]
    nb_gating_spokes = kdata.shape[2]

    print(kdata.dtype)
    print(traj.dtype)

    if kdata.dtype == "complex64":
        traj=traj.astype("float64")

    if kdata.dtype == "complex128":
        traj=traj.astype("float128")

    if b1 is not None:
        if b1.ndim == 2:
            b1 = np.expand_dims(b1, axis=(1, 2))
        elif b1.ndim == 3:
            b1 = np.expand_dims(b1, axis == (1))

    traj = traj.astype(np.float32)

    kdata = kdata.reshape((nb_channels, -1, npoint))
    images_series_rebuilt_nav = np.zeros((nb_slices, nb_gating_spokes, int(npoint / 2)), dtype=np.complex64)
    # all_channels_images_nav = np.zeros((nb_channels,nb_slices,nb_gating_spokes,int(npoint/2)),dtype=np.complex64)

    for i in tqdm(range(nb_channels)):
        fk = finufft.nufft1d1(traj[0, :, 2], kdata[i, :, :], image_size)
        fk = fk.reshape((nb_slices, nb_gating_spokes, int(npoint / 2)))

        # all_channels_images_nav[i]=fk

        if b1 is None:
            images_series_rebuilt_nav += np.abs(fk) ** 2
        else:
            images_series_rebuilt_nav += b1[i].conj() * fk

    if b1 is None:
        images_series_rebuilt_nav = np.sqrt(images_series_rebuilt_nav)

    return images_series_rebuilt_nav


def calculate_displacement(image, bottom, top, shifts,lambda_tv=0.001):
    nb_gating_spokes = image.shape[1]
    nb_slices = image.shape[0]
    npoint_image = image.shape[-1]
    ft = np.mean(image, axis=0)
    # ft=np.mean(image_nav_best_channel,axis=0)
    # ft=image[0]
    image_nav_for_correl = image.reshape(-1, npoint_image)
    nb_images = image_nav_for_correl.shape[0]
    correls = []
    mvt = []
    # adj=[]
    for j in tqdm(range(nb_images)):
        corrs = np.zeros(len(shifts))
        for i, shift in enumerate(shifts):
            corr = np.corrcoef(np.concatenate([ft[j % nb_gating_spokes, bottom:top].reshape(1, -1),
                                               image_nav_for_correl[j, bottom + shift:top + shift].reshape(1, -1)],
                                              axis=0))[0, 1]
            # corr = np.linalg.norm(image_nav_for_correl[0, bottom:top]-image_nav_for_correl[j + 1, (bottom + shift):(top + shift)])
            corrs[i] = corr
        # adjustment=np.sum(dft_x[j+1]*dft_t[j])/np.sum(dft_x[j+1]**2)
        # adj.append(adjustment)
        if (j % nb_gating_spokes == 0):
            J = corrs


        else:
            J = corrs - lambda_tv * (np.array(shifts) - mvt[j - 1]) ** 2  # penalty to not be too far from last disp

        current_mvt = shifts[np.argmax(J)]
        mvt.append(current_mvt)
        # correls.append(corrs)
    # correls_array = np.array(correls)
    # mvt = [shifts[i] for i in np.argmax(correls_array, axis=-1)]
    # mvt=[shifts[i] for i in np.argmin(correls_array,axis=-1)]
    # mvt=np.array(mvt)+np.array(adj)
    # mvt=np.concatenate([[0],mvt]).astype(int)
    # mvt=np.array(mvt).reshape(int(nb_slices),int(nb_gating_spokes))
    # displacement=-np.cumsum(mvt,axis=-1).flatten()
    displacement = np.array(mvt)
    return displacement, correls_array



def mrisensesim(size, ncoils=8, array_cent=None, coil_width=2, n_rings=None, phi=0):
    """Apply simulated sensitivity maps. Based on a script by Florian Knoll.
    Args:
        size (tuple): Size of the image array for the sensitivity coils.
        nc_range (int, default: 8): Number of coils to simulate.
        array_cent (tuple, default: 0): Location of the center of the coil
            array.
        coil_width (double, default: 2): Parameter governing the width of the
            coil, multiplied by actual image dimension.
        n_rings (int, default: ncoils // 4): Number of rings for a
            cylindrical hardware set-up.
        phi (double, default: 0): Parameter for rotating coil geometry.
    Returns:
        list: A list of dimensions (ncoils, (N)), specifying spatially-varying
            sensitivity maps for each coil.
    """
    if array_cent is None:
        c_shift = [0, 0, 0]
    elif len(array_cent) < 3:
        c_shift = array_cent + (0,)
    else:
        c_shift = array_cent

    c_width = coil_width * min(size)

    if len(size) > 2:
        if n_rings is None:
            n_rings = ncoils // 4

    c_rad = min(size[0:1]) / 2
    smap = []
    if len(size) > 2:
        zz, yy, xx = np.meshgrid(
            range(size[2]), range(size[1]), range(size[0]), indexing="ij"
        )
    else:
        yy, xx = np.meshgrid(range(size[1]), range(size[0]), indexing="ij")

    if ncoils > 1:
        x0 = np.zeros((ncoils,))
        y0 = np.zeros((ncoils,))
        z0 = np.zeros((ncoils,))

        for i in range(ncoils):
            if len(size) > 2:
                theta = np.radians((i - 1) * 360 / (ncoils + n_rings) + phi)
            else:
                theta = np.radians((i - 1) * 360 / ncoils + phi)
            x0[i] = c_rad * np.cos(theta) + size[0] / 2
            y0[i] = c_rad * np.sin(theta) + size[1] / 2
            if len(size) > 2:
                z0[i] = (size[2] / (n_rings + 1)) * (i // n_rings)
                smap.append(
                    np.exp(
                        -1
                        * ((xx - x0[i]) ** 2 + (yy - y0[i]) ** 2 + (zz - z0[i]) ** 2)
                        / (2 * c_width)
                    )
                )
            else:
                smap.append(
                    np.exp(-1 * ((xx - x0[i]) ** 2 + (yy - y0[i]) ** 2) / (2 * c_width))
                )
    else:
        x0 = c_shift[0]
        y0 = c_shift[1]
        z0 = c_shift[2]
        if len(size) > 2:
            smap = np.exp(
                -1 * ((xx - x0) ** 2 + (yy - y0) ** 2 + (zz - z0) ** 2) / (2 * c_width)
            )
        else:
            smap = np.exp(-1 * ((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * c_width))

    side_mat = np.arange(int(size[0] // 2) - 20, 1, -1)
    side_mat = np.reshape(side_mat, (1,) + side_mat.shape) * np.ones(shape=(size[1], 1))
    cent_zeros = np.zeros(shape=(size[1], size[0] - side_mat.shape[1] * 2))

    ph = np.concatenate((side_mat, cent_zeros, side_mat), axis=1) / 10
    if len(size) > 2:
        ph = np.reshape(ph, (1,) + ph.shape)

    for i, s in enumerate(smap):
        smap[i] = s * np.exp(i * 1j * ph * np.pi / 180)

    return smap
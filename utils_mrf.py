
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mutools.optim.dictsearch import dictsearch,groupmatch
from functools import reduce
from mrfsim import makevol,parse_options,groupby,load_data
import numpy as np
import finufft
from scipy import ndimage
from sklearn.decomposition import PCA
import tqdm
from scipy.spatial import Voronoi,ConvexHull
from Transformers import PCAComplex
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import pandas as pd
import itertools
from sigpy.mri import spiral

try:
    import pycuda.autoinit
    from pycuda.gpuarray import GPUArray, to_gpu
    from cufinufft import cufinufft
except:
    pass

def read_mrf_dict(dict_file ,FF_list ,aggregate_components=True):

    mrfdict = dictsearch.Dictionary()
    mrfdict.load(dict_file, force=True)

    if aggregate_components :
        epg_water = mrfdict.values[: ,: ,0]
        epg_fat = mrfdict.values[: ,: ,1]
        ff = np.zeros(mrfdict.values.shape[:-1 ] +(len(FF_list),))
        ff_matrix =np.tile(np.array(FF_list) ,ff.shape[:-1 ] +(1,))

        water_signal =np.expand_dims(mrfdict.values[: ,: ,0] ,axis=-1 ) *ff_matrix
        fat_signal =np.expand_dims(mrfdict.values[: ,: ,1] ,axis=-1 ) *( 1 -ff_matrix)

        signal =water_signal +fat_signal

        signal_reshaped =np.moveaxis(signal ,-1 ,-2)
        signal_reshaped =signal_reshaped.reshape((-1 ,175))

        keys_with_ff = list(itertools.product(mrfdict.keys, FF_list))
        keys_with_ff = [(*res, f) for res, f in keys_with_ff]

        return keys_with_ff,signal_reshaped

    else:
        return mrfdict.keys,mrfdict.values



def animate_images(images_series,interval=200,metric=np.abs):
    fig, ax = plt.subplots()
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i, image in enumerate(images_series):

        im = ax.imshow(metric(image), animated=True)
        if i == 0:
            ax.imshow(metric(image))  # show an initial one first
        ims.append([im])

    return animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=10 * interval)

def animate_multiple_images(images_series,images_series_rebuilt,interval=200):
    nb_frames=len(images_series)
    fig, ax = plt.subplots()
    fig_rebuilt, ax_rebuilt = plt.subplots()
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    ims_rebuilt = []
    for i in range(nb_frames):
        im = ax.imshow(np.abs(images_series[i]), animated=True)
        if i == 0:
            ax.imshow(np.abs(images_series[i]))  # show an initial one first
        ims.append([im])

        im_rebuilt = ax_rebuilt.imshow(np.abs(images_series_rebuilt[i]), animated=True)
        if i == 0:
            ax_rebuilt.imshow(np.abs(images_series_rebuilt[i]))  # show an initial one first
        ims_rebuilt.append([im_rebuilt])

    return animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=10 * interval),animation.ArtistAnimation(fig_rebuilt, ims_rebuilt, interval=interval, blit=True,
                                    repeat_delay=10 * interval),

def radial_golden_angle_traj(total_nspoke,npoint,k_max=np.pi):
    golden_angle=-111.25*np.pi/180
    base_spoke = np.arange(-k_max, k_max, 2 * k_max / npoint, dtype=np.complex_)
    all_rotations = np.exp(1j * np.arange(total_nspoke) * golden_angle)
    all_spokes = np.matmul(np.diag(all_rotations), np.repeat(base_spoke.reshape(1, -1), total_nspoke, axis=0))
    return all_spokes


def radial_golden_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
    traj = np.reshape(all_spokes, (-1, nspoke * npoint))

    k_z = np.zeros((timesteps, nb_rep))
    all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    k_z[0, :] = all_slices[::undersampling_factor]
    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    k_z = np.expand_dims(k_z, axis=-1)
    traj = np.expand_dims(traj, axis=-2)
    k_z, traj = np.broadcast_arrays(k_z, traj)
    k_z = np.reshape(k_z, (timesteps, -1))
    traj = np.reshape(traj, (timesteps, -1))

    return np.stack([traj.real,traj.imag, k_z], axis=-1)

def spiral_golden_angle_traj(total_spiral,fov, N, f_sampling, R, ninterleaves, alpha, gm, sm):
    golden_angle = -111.25 * np.pi / 180
    base_spiral = spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
    base_spiral = base_spiral[:,0]+1j*base_spiral[:,1]
    all_rotations = np.exp(1j * np.arange(total_spiral) * golden_angle)
    all_spirals = np.matmul(np.diag(all_rotations), np.repeat(base_spiral.reshape(1, -1), total_spiral, axis=0))
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







def compare_paramMaps(map1,map2,mask1,mask2=None,fontsize=5,title1="Orig Map",title2="Rebuilt Map",adj_wT1=False,fat_threshold=0.8,proj_on_mask1=False,save=False,figsize=(30,10)):
    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    if mask2 is None:
        mask2 = mask1
    for k in (keys_1 & keys_2):
        fig,axes=plt.subplots(1,3,figsize=figsize)
        vol1 = makevol(map1[k],mask1)
        vol2= makevol(map2[k],mask2)
        if proj_on_mask1:
            vol2=vol2*(mask1*1)
        if adj_wT1 and k=="wT1":
            ff = makevol(map2["ff"],mask2)
            vol2[ff>fat_threshold]=vol1[ff>fat_threshold]

        error=vol2-vol1

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
            plt.savefig("./figures/{}_vs_{}_{}".format(title1,title2,k))

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
                             figsize=(15, 10),save=False):
    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    nb_keys = len(keys_1 & keys_2)

    if maskROI is None:
        maskROI = buildROImask(map1)

    fig, ax = plt.subplots(1, nb_keys, figsize=figsize)

    for i, k in enumerate(keys_1 & keys_2):
        obs = map1[k]
        pred = map2[k]

        if mask1 is not None:
            if mask2 is None:
                mask2 = mask1
            mask_union = mask1 | mask2
            mat_obs = makevol(map1[k], mask1)
            mat_pred = makevol(map2[k], mask2)
            mat_ROI = makevol(maskROI, mask1)
            if proj_on_mask1:
                mat_pred = mat_pred * (mask1 * 1)
                mat_ROI = mat_ROI * (mask1 * 1)
                mat_obs = mat_obs * (mask1 * 1)
                mask_union = mask1

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

        if k == "ff":
            print(np.sort(obs.flatten()))
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

        ax[i].set_title(k + " R2:{} Bias:{}".format(np.round(r_2, 4), np.round(bias, 3)), fontsize=2 * fontsize)
        ax[i].tick_params(axis='x', labelsize=fontsize)
        ax[i].tick_params(axis='y', labelsize=fontsize)

    plt.suptitle(title)
    if save:
        plt.savefig("./figures/{}".format(title))

def metrics_paramMaps_ROI(map_ref, map2, mask_ref=None, mask2=None, maskROI=None,
                              adj_wT1=False, fat_threshold=0.8, proj_on_mask1=True,name="Result",
                             ):

    df = pd.DataFrame(columns=[name])

    keys_1 = set(map_ref.keys())
    keys_2 = set(map2.keys())
    nb_keys = len(keys_1 & keys_2)

    if maskROI is None:
        maskROI = buildROImask(map_ref)

    for i, k in enumerate(keys_1 & keys_2):
        print(i)


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

        df=df.append(pd.DataFrame(columns=[name],index=["R2 {}".format(k)],data=r_2))
        df=df.append(pd.DataFrame(columns=[name], index=["Bias {}".format(k)], data=bias))
        df=df.append(pd.DataFrame(columns=[name], index=["mean RMSE {}".format(k)], data=error))
        df=df.append(pd.DataFrame(columns=[name], index=["std RMSE {}".format(k)], data=std_error))

    return df


def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            indices.remove(-1)

        vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol,v

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


def build_mask_single_image(kdata,trajectory,size,useGPU=False,eps=1e-6):
    mask = False

    npoint=trajectory.paramDict["npoint"]
    traj = trajectory.get_traj()

    # kdata /= np.sum(np.abs(kdata) ** 2) ** 0.5 / len(kdata)

    density = np.abs(np.linspace(-1, 1, npoint))
    kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata]
    # kdata = (normalize_image_series(np.array(kdata)))
    kdata_all = np.reshape(kdata, (-1,))

    if traj.shape[-1]==2: # For slices
        traj_all = np.reshape(traj, (-1,2))
        if not(useGPU):
            volume_rebuilt = finufft.nufft2d1(traj_all[:,0], traj_all[:,1], kdata_all, size)
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((1, N1, N2), dtype=complex_dtype)


            c_retrieved = kdata_all
            kx = traj_all[:, 0]
            ky = traj_all[:, 1]

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
            volume_rebuilt = np.array(fk)
            plan.__del__()


        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[len(unique) // 7])
        #mask = ndimage.binary_closing(mask, iterations=10)


    elif traj.shape[-1]==3: # For volumes
        traj_all = np.reshape(traj, (-1, 3))
        if not(useGPU):
            volume_rebuilt = finufft.nufft3d1(traj_all[:, 2],traj_all[:, 0], traj_all[:, 1], kdata_all, size)
        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((1, N1, N2, N3), dtype=complex_dtype)

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
        mask = mask | (np.abs(volume_rebuilt) > unique[len(unique) // 7])
        mask = ndimage.binary_closing(mask, iterations=3)

    return mask


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
            c_gpu = GPUArray((1, M), dtype=complex_dtype)
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

def buildROImask(map):
    # ROI using KNN clustering
    if "wT1" not in map:
        raise ValueError("wT1 should be in the param Map to build the ROI")

    #print(map["wT1"].shape)
    orig_data = map["wT1"].reshape(-1, 1)
    data = orig_data + np.random.normal(size=orig_data.shape,scale=0.1)
    model = KMeans(n_clusters=np.minimum(len(np.unique(orig_data)),10))
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

def simulate_radial_undersampled_images(kdata,trajectory,size,density_adj=True,useGPU=False,eps=1e-6):

    traj=trajectory.get_traj()
    npoint = trajectory.paramDict["npoint"]
    nspoke = trajectory.paramDict["nspoke"]

    dtheta = np.pi / nspoke
    kdata = np.array(kdata) / (npoint * trajectory.paramDict["nb_rep"]) * dtheta

    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata]

    #kdata = (normalize_image_series(np.array(kdata)))

    if traj.shape[-1] == 2:  # 2D
        if not(useGPU):
            images_series_rebuilt = [
                finufft.nufft2d1(t[:,0], t[:,1], s, size)
                for t, s in zip(traj, kdata)
            ]
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((1, N1, N2), dtype=complex_dtype)
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
            images_series_rebuilt=np.array(images_GPU)
    elif traj.shape[-1] == 3:  # 3D
        if not(useGPU):
            images_series_rebuilt = [
                finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size)
                for t, s in zip(traj, kdata)
            ]

        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((1, N1, N2, N3), dtype=complex_dtype)
            images_GPU=[]
            for i in list(range(len(kdata))):

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


                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2,N3), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(to_gpu(c_retrieved), fk_gpu)

                fk = np.squeeze(fk_gpu.get())
                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt = np.array(images_GPU)

    #images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return np.array(images_series_rebuilt)

def simulate_undersampled_images(kdata,trajectory,size,density_adj=True,useGPU=False,eps=1e-6):
    # Strong Assumption : from one time step to the other, the sampling is just rotated, hence voronoi volumes can be calculated only once
    print("Simulating Undersampled Images")
    traj=trajectory.get_traj()

    kdata = np.array(kdata) / (2*np.pi) **2

    if density_adj:
        print("Performing density adjustment using Voronoi cells")
        density = voronoi_volumes(np.transpose(np.array([traj[0,:, 0], traj[0,:, 1]])))[0]
        kdata = np.array([k * density for i, k in enumerate(kdata)]) / (2 * np.pi) ** 2

    #kdata = (normalize_image_series(np.array(kdata)))

    if traj.shape[-1] == 2:  # 2D
        if not(useGPU):
            images_series_rebuilt = [
                finufft.nufft2d1(t[:,0], t[:,1], s, size)
                for t, s in zip(traj, kdata)
            ]
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((1, N1, N2), dtype=complex_dtype)
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
            images_series_rebuilt=np.array(images_GPU)
    elif traj.shape[-1] == 3:  # 3D

        raise ValueError("Generic undersampled trajectory simulation not covered yet for 3D")
        # if not(useGPU):
        #     images_series_rebuilt = [
        #         finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size)
        #         for t, s in zip(traj, kdata)
        #     ]
        #
        # else:
        #     N1, N2, N3 = size[0], size[1], size[2]
        #     dtype = np.float32  # Datatype (real)
        #     complex_dtype = np.complex64
        #     fk_gpu = GPUArray((1, N1, N2, N3), dtype=complex_dtype)
        #     images_GPU=[]
        #     for i in list(range(len(kdata))):
        #
        #         c_retrieved = kdata[i]
        #         kx = traj[i, :, 0]
        #         ky = traj[i, :, 1]
        #         kz = traj[i, :, 2]
        #
        #
        #         # Cast to desired datatype.
        #         kx = kx.astype(dtype)
        #         ky = ky.astype(dtype)
        #         kz = kz.astype(dtype)
        #         c_retrieved = c_retrieved.astype(complex_dtype)
        #
        #         # Allocate memory for the uniform grid on the GPU.
        #
        #
        #         # Initialize the plan and set the points.
        #         plan = cufinufft(1, (N1, N2,N3), 1, eps=eps, dtype=dtype)
        #         plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))
        #
        #         # Execute the plan, reading from the strengths array c and storing the
        #         # result in fk_gpu.
        #         plan.execute(to_gpu(c_retrieved), fk_gpu)
        #
        #         fk = np.squeeze(fk_gpu.get())
        #         images_GPU.append(fk)
        #         plan.__del__()
        #     images_series_rebuilt = np.array(images_GPU)

    #images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mutools.optim.dictsearch import dictsearch,groupmatch
from mrfsim import makevol,parse_options,groupby
import numpy as np
import finufft
from scipy import ndimage
from sklearn.decomposition import PCA
import tqdm

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


def translation_breathing(t,direction,T=300,frac_expiration=0.7):
    def base_pattern(t):
        lambda1=5/(frac_expiration*T)
        lambda2=20/((1-frac_expiration)*T)
        if t<(frac_expiration*T):
            return (1-np.exp(-lambda1*t))*direction
        else:
            return ((1-np.exp(-lambda1*frac_expiration*T))* np.exp(-lambda2*t)/np.exp(-lambda2*frac_expiration*T))*direction

    return base_pattern(t-int(t/T)*T).flatten()


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

def SearchMrf(kdata,traj, dictfile, niter, method, metric, shape,nspoke,density_adj=False, setup_opts={}, search_opts= {}):
    """ Estimate parameters """
    # constants
    shape = tuple(shape)


    # density compensation
    npoint = traj.shape[1]
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
        kdatai = [d * np.tile(density, nspoke) for d in kdata]

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

def dictSearchMemoryOptim(all_signals,dictfile,pca=True,threshold_pca = 0.999999,split=2000):
    mrfdict = dictsearch.Dictionary()
    mrfdict.load(dictfile, force=True)

    keys = mrfdict.keys
    array_water = mrfdict.values[:, :, 0]
    array_fat = mrfdict.values[:, :, 1]

    del mrfdict

    array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
    array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)
    all_signals_unique, index_signals_unique = np.unique(all_signals, axis=1, return_inverse=True)

    nb_water_timesteps = array_water_unique.shape[1]
    nb_fat_timesteps = array_fat_unique.shape[1]
    nb_patterns = array_water.shape[0]
    nb_signals_unique = all_signals_unique.shape[1]
    nb_signals = all_signals.shape[1]

    duplicate_signals=True
    if nb_signals_unique==nb_signals:
        print("No duplicate signals")
        duplicate_signals = False
        all_signals_unique=all_signals


    del array_water
    del array_fat

    if pca:
        print("Performing PCA")

        mean_fat = np.mean(array_fat_unique,axis=0)
        mean_water = np.mean(array_water_unique,axis=0)

        cov_fat = np.matmul(np.transpose((array_fat_unique-mean_fat).conj()), (array_fat_unique-mean_fat))
        cov_water = np.matmul(np.transpose((array_water_unique-mean_water).conj()), (array_water_unique-mean_water))

        fat_val, fat_vect = np.linalg.eigh(cov_fat)
        water_val, water_vect = np.linalg.eigh(cov_water)

        sorted_index_fat = np.argsort(fat_val)[::-1]
        fat_val = fat_val[sorted_index_fat]
        fat_vect = fat_vect[:, sorted_index_fat]

        sorted_index_water = np.argsort(water_val)[::-1]
        water_val = water_val[sorted_index_water]
        water_vect = water_vect[:, sorted_index_water]

        explained_variance_ratio_fat = np.cumsum(fat_val ** 2) / np.sum(fat_val ** 2)
        n_components_fat = np.sum(explained_variance_ratio_fat < threshold_pca) + 1

        explained_variance_ratio_water = np.cumsum(water_val ** 2) / np.sum(water_val ** 2)
        n_components_water = np.sum(explained_variance_ratio_water < threshold_pca) + 1

        print("Water Components Retained {} out of {} timesteps".format(n_components_water,nb_water_timesteps))
        print("Fat Components Retained {} out of {} timesteps".format(n_components_fat, nb_fat_timesteps))

        fat_vect = fat_vect[:, :n_components_fat]
        water_vect = water_vect[:, :n_components_water]

        transformed_array_water_unique = np.matmul(array_water_unique, water_vect.conj())
        transformed_array_fat_unique = np.matmul(array_fat_unique, fat_vect.conj())

        # retrieved_array_water_unique = np.matmul(transformed_array_water_unique,np.transpose(water_vect))
        # retrieved_array_fat_unique = np.matmul(transformed_array_fat_unique,np.transpose(fat_vect))

        # plt.figure()
        # plt.plot(retrieved_array_fat_unique[0,:].real)
        # plt.plot(array_fat_unique[0,:].real)

        transformed_all_signals_water = np.transpose(np.matmul(np.transpose(all_signals_unique), water_vect.conj()))
        transformed_all_signals_fat = np.transpose(np.matmul(np.transpose(all_signals_unique), fat_vect.conj()))

        sig_ws_all_unique = np.matmul(transformed_array_water_unique, transformed_all_signals_water[:, :].conj()).real
        sig_fs_all_unique = np.matmul(transformed_array_fat_unique, transformed_all_signals_fat[:, :].conj()).real

    else:
        sig_ws_all_unique = np.matmul(array_water_unique, all_signals_unique[:, :].conj()).real
        sig_fs_all_unique = np.matmul(array_fat_unique, all_signals_unique[:, :].conj()).real



    var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
    var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
    sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

    var_w = var_w[index_water_unique]
    var_f = var_f[index_fat_unique]

    print("Calculating optimal fat fraction and best pattern per signal")
    var_w = np.reshape(var_w, (-1, 1))
    var_f = np.reshape(var_f, (-1, 1))
    sig_wf = np.reshape(sig_wf, (-1, 1))

    alpha_all_unique = np.zeros((nb_patterns, nb_signals_unique))
    J_all = np.zeros(alpha_all_unique.shape)

    num_group = int(nb_signals_unique / split) + 1

    for j in tqdm.tqdm(range(num_group)):
        j_signal = j * split
        j_signal_next = np.minimum((j + 1) * split, nb_signals_unique)
        current_sig_ws = sig_ws_all_unique[index_water_unique, j_signal:j_signal_next]
        current_sig_fs = sig_fs_all_unique[index_fat_unique, j_signal:j_signal_next]
        current_alpha_all_unique = (sig_wf * current_sig_ws - var_w * current_sig_fs) / (
                    (current_sig_ws + current_sig_fs) * sig_wf - var_w * current_sig_fs - var_f * current_sig_ws)
        alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
        J_all[:, j_signal:j_signal_next] = ((
                                                        1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
            (1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                        1 - current_alpha_all_unique) * sig_wf)

    idx_max_all_unique = np.argmax(J_all, axis=0)
    del J_all

    print("Building the maps")

    del sig_ws_all_unique
    del sig_fs_all_unique

    params_all_unique = np.array([keys[idx] + (alpha_all_unique[idx, i],) for i, idx in enumerate(idx_max_all_unique)])



    if duplicate_signals:
        params_all = params_all_unique[index_signals_unique]
    else:
        params_all = params_all_unique


    map_rebuilt = {
        "wT1": params_all[:, 0],
        "fT1": params_all[:, 1],
        "attB1": params_all[:, 2],
        "df": params_all[:, 3],
        "ff": params_all[:, 4]

    }

    return map_rebuilt





# def dictSearchMemoryOptimPCAPatterns(all_signals,dictfile,pca=True,threshold_pca = 0.999999,split=2000):
#     mrfdict = dictsearch.Dictionary()
#     mrfdict.load(dictfile, force=True)
#
#     keys = mrfdict.keys
#     array_water = mrfdict.values[:, :, 0]
#     array_fat = mrfdict.values[:, :, 1]
#
#     del mrfdict
#
#     array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
#     array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)
#     all_signals_unique, index_signals_unique = np.unique(all_signals, axis=1, return_inverse=True)
#
#     nb_water = array_water_unique.shape[0]
#     nb_fat = array_fat_unique.shape[0]
#     nb_patterns = array_water.shape[0]
#     nb_signals_unique = all_signals_unique.shape[1]
#     nb_signals = all_signals.shape[1]
#
#     duplicate_signals=True
#     if nb_signals_unique==nb_signals:
#         print("No duplicate signals")
#         duplicate_signals = False
#         all_signals_unique=all_signals
#
#
#     del array_water
#     del array_fat
#
#     if pca:
#         print("Performing PCA")
#         mean_fat = np.mean(array_fat_unique,axis=1).reshape(-1,1)
#         mean_water = np.mean(array_water_unique,axis=1).reshape(-1,1)
#         cov_fat = np.matmul((array_fat_unique-mean_fat).conj(), np.transpose(array_fat_unique-mean_fat))
#         cov_water = np.matmul((array_water_unique-mean_water).conj(), np.transpose(array_water_unique-mean_water))
#
#         del mean_water
#         del mean_fat
#
#         fat_val, fat_vect = np.linalg.eigh(cov_fat)
#         water_val, water_vect = np.linalg.eigh(cov_water)
#
#         del cov_water
#         del cov_fat
#
#         sorted_index_fat = np.argsort(fat_val)[::-1]
#         fat_val = fat_val[sorted_index_fat]
#         fat_vect = fat_vect[:, sorted_index_fat]
#
#         sorted_index_water = np.argsort(water_val)[::-1]
#         water_val = water_val[sorted_index_water]
#         water_vect = water_vect[:, sorted_index_water]
#
#         explained_variance_ratio_fat = np.cumsum(fat_val>0 ** 2) / np.sum(fat_val>0 ** 2)
#         n_components_fat = np.sum(explained_variance_ratio_fat < threshold_pca) + 1
#
#         explained_variance_ratio_water = np.cumsum(water_val>0 ** 2) / np.sum(water_val>0 ** 2)
#         n_components_water = np.sum(explained_variance_ratio_water < threshold_pca) + 1
#
#         print("Water Components Retained {} out of {}".format(n_components_water,nb_water))
#         print("Fat Components Retained {} out of {}".format(n_components_fat, nb_fat))
#
#
#         fat_vect = fat_vect[:, :n_components_fat]
#         water_vect = water_vect[:, :n_components_water]
#
#         transformed_array_water_unique = (np.matmul( np.transpose(water_vect),array_water_unique))
#         transformed_array_fat_unique = (np.matmul( np.transpose(fat_vect),array_fat_unique))
#
#         #retrieved_array_water_unique = np.matmul((water_vect.conj()),transformed_array_water_unique)
#         #retrieved_array_fat_unique = np.matmul((fat_vect.conj()),transformed_array_fat_unique)
#
#         #plt.figure()
#         #plt.plot(retrieved_array_fat_unique[0,:].real)
#         #plt.plot(array_fat_unique[0,:].real)
#
#
#         sig_ws_all_unique = np.matmul(transformed_array_water_unique, all_signals_unique[:, :].conj()).real
#         sig_fs_all_unique = np.matmul(transformed_array_fat_unique, all_signals_unique[:, :].conj()).real
#
#     else:
#         sig_ws_all_unique = np.matmul(array_water_unique, all_signals_unique[:, :].conj()).real
#         sig_fs_all_unique = np.matmul(array_fat_unique, all_signals_unique[:, :].conj()).real
#
#
#
#     var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
#     var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
#     sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real
#
#     var_w = var_w[index_water_unique]
#     var_f = var_f[index_fat_unique]
#
#     print("Calculating optimal fat fraction and best pattern per signal")
#     var_w = np.reshape(var_w, (-1, 1))
#     var_f = np.reshape(var_f, (-1, 1))
#     sig_wf = np.reshape(sig_wf, (-1, 1))
#
#     alpha_all_unique = np.zeros((nb_patterns, nb_signals_unique))
#     J_all = np.zeros(alpha_all_unique.shape)
#
#     num_group = int(nb_signals_unique / split) + 1
#
#     for j in tqdm.tqdm(range(num_group)):
#         j_signal = j * split
#         j_signal_next = np.minimum((j + 1) * split, nb_signals_unique)
#         current_sig_ws = sig_ws_all_unique[index_water_unique, j_signal:j_signal_next]
#         current_sig_fs = sig_fs_all_unique[index_fat_unique, j_signal:j_signal_next]
#         current_alpha_all_unique = (sig_wf * current_sig_ws - var_w * current_sig_fs) / (
#                     (current_sig_ws + current_sig_fs) * sig_wf - var_w * current_sig_fs - var_f * current_sig_ws)
#         alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
#         J_all[:, j_signal:j_signal_next] = ((
#                                                         1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
#             (1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
#                         1 - current_alpha_all_unique) * sig_wf)
#
#     idx_max_all_unique = np.argmax(J_all, axis=0)
#     del J_all
#
#     print("Building the maps")
#
#     del sig_ws_all_unique
#     del sig_fs_all_unique
#
#     params_all_unique = np.array([keys[idx] + (alpha_all_unique[idx, i],) for i, idx in enumerate(idx_max_all_unique)])
#
#     if duplicate_signals:
#         params_all = params_all_unique[index_signals_unique]
#     else:
#         params_all = params_all_unique
#
#
#     map_rebuilt = {
#         "wT1": params_all[:, 0],
#         "fT1": params_all[:, 1],
#         "attB1": params_all[:, 2],
#         "df": params_all[:, 3],
#         "ff": params_all[:, 4]
#
#     }
#
#     return map_rebuilt

def compare_paramMaps(map1,map2,mask,fontsize=5,title1="Orig Map",title2="Rebuilt Map",adj_wT1=False,fat_threshold=0.8):
    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    for k in (keys_1 & keys_2):
        fig,axes=plt.subplots(1,3)
        vol1 = makevol(map1[k],mask)
        vol2= makevol(map2[k],mask)
        if adj_wT1 and k=="wT1":
            ff = makevol(map2["ff"],mask)
            vol2[ff>fat_threshold]=vol1[ff>fat_threshold]

        error=np.abs(vol1-vol2)

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

def regression_paramMaps(map1,map2,fontsize=5,adj_wT1=False,fat_threshold=0.8):

    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    nb_keys=len(keys_1 & keys_2)
    fig,ax = plt.subplots(1,nb_keys)
    for i,k in enumerate(keys_1 & keys_2):
        obs = map1[k]
        pred = map2[k]

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

        ax[i].scatter(obs,pred,s=1)
        ax[i].plot(x_, x_,"r")
        ax[i].set_title(k+" R2:{} Bias:{}".format(np.round(r_2,2),np.round(bias,2)),fontsize=2*fontsize)
        ax[i].tick_params(axis='x', labelsize=fontsize)
        ax[i].tick_params(axis='y', labelsize=fontsize)


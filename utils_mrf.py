
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mutools.optim.dictsearch import dictsearch,groupmatch
from mrfsim import makevol,parse_options,groupby
import numpy as np
import finufft
from scipy import ndimage

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
    traj = np.reshape(groupby(traj, nspoke), (-1, npoint * nspoke))
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
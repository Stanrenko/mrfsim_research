""" build MRF T1 dict and map T1 volumes """
import pathlib
import itertools
import numpy as np
from scipy.io import loadmat
from epgpy import epg
from machines import machine, Toolbox, Config, set_parameter, set_output, printer, file_handler
from mutools import io
from mutools.optim.dictsearch import dictsearch, groupmatch, utils

DEFAULT_SEQUENCE = "mrf_sequence.json"
DEFAULT_CONFIG = "mrf_dict_config.json"

# build dictionary
@machine()
@set_parameter("sequence_config", type=Config, default=DEFAULT_SEQUENCE)
@set_parameter("dict_config", type=Config, default=DEFAULT_CONFIG)
@set_parameter("dict_file", str)
def build_dict(sequence_config, dict_config, dict_file):
    """ Build MRF-T1map dictionary """
    # build sequence
    printer("Build MRF-T1map sequence.")
    seq = T1MRF(**sequence_config)
    lenseq = len(sequence_config["TR"])

    # generate signals
    # 'water_T1', 'water_T2', 'fat_T1', 'fat_T2', 'delta_freqs', 'B1_att', 'fat_amp', 'fat_cshifts
    wT1 = dict_config["water_T1"]
    fT1 = dict_config["fat_T1"]
    wT2 = dict_config["water_T2"]
    fT2 = dict_config["fat_T2"]
    df = dict_config["delta_freqs"]
    df = [- value/1000 for value in df] # temp
    att = dict_config["B1_att"]
    fat_amp = dict_config["fat_amp"]
    fat_cs = dict_config["fat_cshift"]
    fat_cs = [- value/1000 for value in fat_cs] # temp

    # merge spokes
    nwin = dict_config["window_length"]

    def slidewin(arr, n, axis=0):
        npad = -(-arr.shape[0] // n) * n - arr.shape[0]
        padding = [(0, 0)] * arr.ndim
        padding[axis] = (npad//2, -(-npad // 2))
        arr = np.pad(arr, padding, mode="edge")
        # convolve
        arr = np.moveaxis(arr, axis, -1)
        conv = np.mean(arr.reshape(arr.shape[:-1] + (-1, n)), axis=-1)
        return np.moveaxis(conv, -1, axis)

    # water
    printer("Generate water signals.")
    water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])
    water = slidewin(water, nwin, axis=0)

    # fat
    printer("Generate fat signals.")
    eval = "dot(signal, amps)"
    args = {"amps": fat_amp}
    # merge df and fat_cs df to dict
    fatdf = [[cs + f for cs in fat_cs] for f in df]
    fat = seq(T1=[fT1], T2=fT2, att=[[att]], g=[[[fatdf]]], eval=eval, args=args)
    fat = slidewin(fat, nwin, axis=0)

    # join water and fat
    printer("Build dictionary.")
    keys = list(itertools.product(wT1, fT1, att, df))
    values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
    values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

    printer("Save dictionary.")
    mrfdict = dictsearch.Dictionary(keys, values)
    mrfdict.save(dict_file)



@machine()
@set_output("t1map_mrf", type="t1map")
@set_parameter("path", str, description="Path of input data")
@set_parameter("dict_file", str, description="Path of dictionary")
@set_parameter("traj_file", str, description="Path of trajectory file", default=None)
def search(path, dict_file, traj_file):
    """ Estimate parameters """
    printer(f"Load input data")
    data = load_data(path)

    if traj_file:
        traj = load_kspace(traj_file)
    else:
        traj = None

    printer(f"Load dictionary: {dict_file}")
    mrfdict = dictsearch.Dictionary()
    mrfdict.load(dict_file)

    printer(f"Init solver")
    solver = groupmatch.GroupMatch()
    solver.setup(mrfdict.keys, mrfdict.values)

    # auto mask
    vols = np.asarray(data["volumes"])
    unique = np.histogram(np.abs(vols), 100)[1]
    mask = np.mean(np.abs(vols), axis=0) > unique[len(unique) // 10]

    printer(f"Search data (first pass)")
    obs = np.transpose([vol[mask] for vol in vols])
    res = solver.search(obs)

    # second pass
    if traj is not None:
        pred = solver.predict(res)
        breakpoint()
        # volumes = [makevol(values, mask) for values in pred]
        # group spokes into groups of size 8
        coords = [gp.ravel() for gp in groupby(traj, size=8, mode="constant")]
        assert len(traj) == len(pred)
        # compute NUFFT of solution
        spokes = [
            finufft.nufft2d2(coords.real, coords.imag, makevol(values, mask))
            for values, coord in zip(pred, coords)
        ]

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
        "info": {"search": solver.info, "options": solver.options},
    }


# MRF sequence
class T1MRF:
    def __init__(self, FA, TI, TE, TR, B1):
        """ build sequence """
        seqlen = len(TE)
        self.inversion = epg.T(180, 0) # perfect inversion
        seq = [epg.Offset(TI)]
        for i in range(seqlen):
            echo = [
                epg.T(FA * B1[i], 90),
                epg.Wait(TE[i]),
                epg.ADC,
                epg.Wait(TR[i] - TE[i]),
                epg.SPOILER,
            ]
            seq.extend(echo)
        self._seq = seq

    def __call__(self, T1, T2, g, att, **kwargs):
        """ simulate sequence """
        seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g)]
        return np.asarray(epg.simulate(seq, **kwargs))


# load input data
def load_data(filename):
    """ load volumes """
    filename = pathlib.Path(filename)
    if filename.suffix == ".mat":
        vols = np.moveaxis(loadmat(filename)["ImgSeries"], -1, 0)
    else:
        raise NotImplementedError(f"Unknown data type: {filename}")
    return {"volumes": vols}

# load kspace data
def load_kspace(filename):
    """ load k-space data """
    matobj = loadmat(filename)
    traj = matobj["KSpaceTraj"].T
    return traj

def groupby(arr, n, axis=0, mode="edge"):
    """ group array into groups of size 'n' """

    ngroup = -(-arr.shape[axis] // n)
    if arr.shape[axis] % n != 0:
        # pad array
        padding = [(0,0)] * arr.ndim
        nzero = n - np.mod(arr.shape[axis], n)
        padding[axis] = (nzero//2, -(-nzero//2))
        arr = np.pad(arr, padding, mode=mode)
    arr = np.moveaxis(arr, axis, 0)
    arr = arr.reshape((ngroup, -1) + arr.shape[1:])
    return list(np.moveaxis(arr, 1, axis + 1))


def makevol(values, mask):
    """ fill volume """
    values = np.asarray(values)
    new = np.zeros(mask.shape, dtype=values.dtypes)
    new[mask] = values
    return new

# t1map handler
def t1map_saver(dirname, data):
    dirname = pathlib.Path(dirname)
    io.write(dirname / "mask.mha", data["mask"].astype("uint8"))
    io.write(dirname / "wt1map.mha", data["wt1map"], nan_as=-1)
    io.write(dirname / "ft1map.mha", data["ft1map"], nan_as=-1)
    io.write(dirname / "b1map.mha", data["b1map"], nan_as=-1)
    io.write(dirname / "dfmap.mha", data["dfmap"], nan_as=-1)
    io.write(dirname / "wmap.mha", data["wmap"], nan_as=-1)
    io.write(dirname / "fmap.mha", data["fmap"], nan_as=-1)
    io.write(dirname / "ffmap.mha", data["ffmap"], nan_as=-1)
    io.config.write(dirname / "info.yml", data["info"])

t1map_handler = file_handler(save=t1map_saver)


toolbox = Toolbox("MRF-sim", description="build MRF T1 dict and map T1 volumes")
toolbox.add_program("build-dict", build_dict)
toolbox.add_program("search", search)
toolbox.add_handler("t1map", t1map_handler)


if __name__ == "__main__":
    toolbox.cli()

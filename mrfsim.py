""" build MRF T1 dict and map T1 volumes """
import pathlib
import itertools
import numpy as np
from scipy import ndimage
import sys
path = r"/home/cslioussarenko/PythonRepositories"
#path = r"/Users/constantinslioussarenko/PythonGitRepositories/MyoMap"
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")



# matplotl

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib import pyplot as plt
except:
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except:
        pass
# misc
import pandas as pd
from scipy.io import loadmat,savemat
from epgpy import epg
import finufft

# machines
from machines import machine, Toolbox, Config, set_parameter, set_output, printer, file_handler, Parameter, RejectException, get_context

# mutools
from mutools import io
from mutools.tables import aggregate
from mutools.optim.dictsearch import dictsearch, groupmatch, utils
#from mutools.toolbox.main import GetResults, handlers_common

DEFAULT_SEQUENCE = "mrf_sequence.json"
DEFAULT_CONFIG = "mrf_dictconf.json"

# build dictionary
@machine
@set_parameter("sequence_config", type=Config, default=DEFAULT_SEQUENCE)
@set_parameter("dict_config", type=Config, default=DEFAULT_CONFIG)
@set_parameter("window", int, default=None, description="Window size")
@set_parameter("dictfile", str, description="Dictionary file")
@set_parameter("overwrite", bool, default=False, description="Overwrite existing dictionary")
@set_parameter("sim_mode", str, default="mean", description="Aggregation mode for simulation")
def GenDict(sequence_config, dict_config, window, dictfile, overwrite,sim_mode):
    """ Generate MRF-T1map dictionary """

    dictfile = pathlib.Path(dictfile)
    if dictfile.is_file() and not overwrite:
        raise RejectException(f"File already exists: {dictfile}")

    # build sequence
    printer("Build MRF-T1map sequence.")
    seq = T1MRF(**sequence_config)
    lenseq = len(sequence_config["TR"])

    # generate signals
    wT1 = dict_config["water_T1"]
    fT1 = dict_config["fat_T1"]
    wT2 = dict_config["water_T2"]
    fT2 = dict_config["fat_T2"]
    att = dict_config["B1_att"]
    df = dict_config["delta_freqs"]
    df = [- value / 1000 for value in df] # temp
    # df = np.linspace(-0.1, 0.1, 101)

    fat_amp = dict_config["fat_amp"]
    fat_cs = dict_config["fat_cshift"]
    fat_cs = [- value / 1000 for value in fat_cs] # temp

    # other options
    if not window:
        window = dict_config["window_size"]

    # water
    printer("Generate water signals.")
    water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])
    if sim_mode == "mean":
        water = [np.mean(gp, axis=0) for gp in groupby(water, window)]
    elif sim_mode == "mid_point":
        water = water[(int(window / 2) - 1):-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    # fat
    printer("Generate fat signals.")
    eval = "dot(signal, amps)"
    args = {"amps": fat_amp}
    # merge df and fat_cs df to dict
    fatdf = [[cs + f for cs in fat_cs] for f in df]
    fat = seq(T1=[fT1], T2=fT2, att=[[att]], g=[[[fatdf]]], eval=eval, args=args)
    if sim_mode == "mean":
        fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
    elif sim_mode == "mid_point":
        fat = fat[(int(window / 2) - 1):-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    # join water and fat
    printer("Build dictionary.")
    keys = list(itertools.product(wT1, fT1, att, df))
    values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
    values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

    printer("Save dictionary.")
    mrfdict = dictsearch.Dictionary(keys, values)
    mrfdict.save(dictfile, overwrite=overwrite)


@machine
@set_output("t1map_mrf")
@set_parameter("path", str, description="Path of k-space data")
@set_parameter("dictfile", str, description="Path of dictionary")
@set_parameter("niter", int, description="Number of iteration", default=0)
@set_parameter("method", ["brute", "group"], description="Search method")
@set_parameter("metric", ["ls", "nnls"], description="Cost function")
@set_parameter("shape", [int, int], description="Image shape", default=[256, 256])
@set_parameter("setup_opts", [str], description="Setup options", default=[])
@set_parameter("search_opts", [str], description="Search options", default=[])
def SearchMrf(path, dictfile, niter, method, metric, shape, setup_opts, search_opts):
    """ Estimate parameters """
    # constants
    nspoke = 8 # spoke groups
    shape = tuple(shape)

    printer(f"Load input data")
    kdata, traj = load_data(path)

    # density compensation
    npoint = traj.shape[1]
    density = np.abs(np.linspace(-1, 1, npoint))

    # printer(f"Load dictionary: {dictfile}")
    mrfdict = dictsearch.Dictionary()
    mrfdict.load(dictfile, force=True)

    printer(f"Init solver ({method})")
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
    kdata = np.reshape(groupby(kdata * density**0.5, nspoke), (-1, npoint * nspoke))

    printer(f"Build volumes ({nspoke} groups)")

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

        printer(f"Search data (iteration {i})")
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


param_path = Parameter(str, description="Path of ParamMap.mat file")
@machine(output="refdata", path=param_path)
def GtMrf(path):
    """ extract ROI from ParamMap.mat"""
    data = load_parammap(path)
    return data

@machine(inputs="t1map::t1map_mrf & refdata", output="results")
def ResultsInRoi(t1map, refdata):
    """ Compare results and ground truth """
    roi = refdata["roi"]
    labels = refdata.get("labels", {})

    # dynamics
    T1_DYN = 2000
    FF_DYN = 1

    table = []
    labelset = np.unique(roi[roi > 0])
    for label in labelset:
        mask = (roi == label) * t1map["mask"]
        if mask.sum() == 0:
            continue

        labelname = labels.get(label, f"label {label}")
        row = {"label": labelname}

        ref_wt1 = refdata["wt1map"][mask]
        est_wt1 = t1map["wt1map"][mask]
        row["WT1_REF"] = np.mean(ref_wt1)
        row["WT1_EST"] = np.mean(est_wt1)
        row["WT1_ERR"] = row["WT1_EST"] - row["WT1_REF"]
        row["WT1_STD"] = np.std(est_wt1)
        row["WT1_RMSE"] = np.mean((ref_wt1 - est_wt1)**2)**0.5
        row["WT1_PSNR"] = 20 * np.log10(T1_DYN / row["WT1_RMSE"])

        ref_ff = refdata["ffmap"][mask]
        est_ff = t1map["ffmap"][mask]
        row["FF_REF"] = np.mean(ref_ff)
        row["FF_EST"] = np.mean(est_ff)
        row["FF_ERR"] = row["FF_EST"] - row["FF_REF"]
        row["FF_STD"] = np.std(est_ff)
        row["FF_RMSE"] = np.mean((ref_ff - est_ff)**2)**0.5
        row["FF_PSNR"] = 20 * np.log10(T1_DYN / row["FF_RMSE"])

        row["DF_REF"] = np.mean(refdata["dfmap"][mask])
        row["DF_EST"] = - 1000 * np.mean(t1map["dfmap"][mask])
        row["B1_REF"] = np.mean(refdata["b1map"][mask])
        row["B1_EST"] = np.mean(t1map["b1map"][mask])


        table.append(row)

    # delta T1 and FF ?
    mask = roi > 0
    delta_wT1 = (refdata["wt1map"] - t1map["wt1map"]) * roi
    delta_FF = (refdata["ffmap"] - t1map["ffmap"]) * roi

    table = pd.DataFrame(table)
    return {
        "tables": {"results": table},
        "volumes": {
            "wt1diff": delta_wT1,
            "ffdiff": delta_FF,
        }
    }

@machine(inputs="results", output="output::results", aggregate=True)
def AggregateResults(results):
    """ Merge result tables """
    context = get_context()
    targets = context.targets["results"]
    combine = []
    for target, res in zip(targets, results):
        table = res["tables"]["results"]
        table["index"] = str(target.index)
        table["branch"] = str(target.branch)
        combine.append(table)

    table = aggregate.combine(combine, on=["index", "branch", "label"])
    return {"tables": {"results": table}}



#
# MRF sequence

class T1MRF:
    def __init__(self, FA, TI, TE, TR, B1):
        """ build sequence """
        seqlen = len(TE)
        self.TR=TR
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

    def __call__(self, T1, T2, g, att, calc_deriv=False,**kwargs):
        """ simulate sequence """
        seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        if not(calc_deriv):
            return np.asarray(epg.simulate(seq, **kwargs))
        else:
            return epg.simulate(seq,calc_deriv=calc_deriv, **kwargs)

#
# utils

# load kspace data
def load_data(filename):
    """ load k-space data """
    filename = str(pathlib.Path(filename).expanduser().resolve())
    matobj = loadmat(filename)
    traj = 2 * np.pi * matobj["KSpaceTraj"].T
    data = matobj["KSpaceData"].T
    return data, traj

def load_parammap(filename):
    filename = str(pathlib.Path(filename).expanduser().resolve())
    matobj = loadmat(filename)["paramMap"]
    wt1map = matobj["T1"][0, 0]
    dfmap = matobj["Df"][0, 0]
    b1map = matobj["B1"][0, 0]
    ffmap = matobj["FF"][0, 0]
    mask = wt1map > 0
    labelset = np.unique(wt1map)
    roi = np.digitize(wt1map, labelset) - 1

    return {
        "roi": roi,
        "wt1map": wt1map,
        "dfmap": dfmap,
        "b1map": b1map,
        "ffmap": ffmap,
    }


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
    new = np.zeros(mask.shape, dtype=values.dtype)
    new[mask] = values
    return new

def trycast(value, types=[int, float], default=str):
    """ return cast value for multiple types """
    for type in types:
        try:
            return type(value)
        except ValueError:
            pass
    if default is not None:
        return default(value)
    return value


def parse_options(seq, sep="="):
    """ parse and cast sequence of 'key=value' string """
    return {
        item[0]: trycast(item[1], [int, float])
        for item in map(lambda x: x.split(sep), seq)
    }

#
# handlers

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

def t1map_loader(dirname):
    dirname = pathlib.Path(dirname)
    data = {}
    data["mask"] = io.read(dirname / "mask.mha", astype=bool)
    data["wt1map"] = io.read(dirname / "wt1map.mha", nan_as=-1)
    data["ft1map"] = io.read(dirname / "ft1map.mha", nan_as=-1)
    data["b1map"] = io.read(dirname / "b1map.mha", nan_as=-1)
    data["dfmap"] = io.read(dirname / "dfmap.mha", nan_as=-1)
    data["wmap"] = io.read(dirname / "wmap.mha", nan_as=-1)
    data["fmap"] = io.read(dirname / "fmap.mha", nan_as=-1)
    data["ffmap"] = io.read(dirname / "ffmap.mha", nan_as=-1)
    return data

t1map_handler = file_handler(save=t1map_saver, load=t1map_loader)


def gt_saver(dirname, data):
    dirname = pathlib.Path(dirname)
    io.write(dirname / "roi.mha", data["roi"].astype("uint8"))
    io.write(dirname / "wt1map.mha", data["wt1map"], nan_as=-1)
    io.write(dirname / "b1map.mha", data["b1map"], nan_as=-1)
    io.write(dirname / "dfmap.mha", data["dfmap"], nan_as=-1)
    io.write(dirname / "ffmap.mha", data["ffmap"], nan_as=-1)

def gt_loader(dirname):
    dirname = pathlib.Path(dirname)
    data = {}
    data["roi"] = io.read(dirname / "roi.mha", astype="uint8")
    data["wt1map"] = io.read(dirname / "wt1map.mha", nan_as=-1)
    data["b1map"] = io.read(dirname / "b1map.mha", nan_as=-1)
    data["dfmap"] = io.read(dirname / "dfmap.mha", nan_as=-1)
    data["ffmap"] = io.read(dirname / "ffmap.mha", nan_as=-1)
    if (dirname / "labels.txt").is_file():
        data["labels"] = io.read_labels(dirname/"labels.txt")
    return data

gt_handler = file_handler(save=gt_saver, load=gt_loader)


def results_saver(dirname, data):
    dirname = pathlib.Path(dirname)
    for name in data.get("tables", {}):
        filename = dirname / name
        table = data["tables"][name]
        table.to_excel(filename.with_suffix(".xlsx"), index=False)
        table.to_csv(filename.with_suffix(".csv"), index=False)
    for vol in data.get("volumes", {}):
        io.write(dirname / vol, data["volumes"][vol])

def results_loader(dirname):
    dirname = pathlib.Path(dirname)
    data = {"tables": {}}
    for filename in dirname.glob("*.csv"):
        table = pd.read_csv(filename)
        data["tables"][filename.stem] = table
    return data

results_handler = file_handler(save=results_saver, load=results_loader)

#
# build toolbox

toolbox = Toolbox("mrfsim", description="build MRF T1 dict and map T1 volumes")
toolbox.add_program("gendict", GenDict)
toolbox.add_program("search", SearchMrf)
toolbox.add_program("getref", GtMrf)
toolbox.add_program("getres", ResultsInRoi)
toolbox.add_program("aggres", AggregateResults)

toolbox.add_handler("t1map_mrf", t1map_handler)
toolbox.add_handler("refdata", gt_handler)
toolbox.add_handler("results", results_handler)



if __name__ == "__main__":
    toolbox.cli()

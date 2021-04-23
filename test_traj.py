""" test trajectory simulation with NUFFT """

import itertools
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import finufft

from epgpy import epg
from mutools import io


DATADIR = Path("data")


# image series
# imgs = loadmat(DATADIR / "ImgSeries0.mat")["ImgSeries"]
imgs8 = loadmat(DATADIR / "ImgSeries8Spokes_block.mat")["ImgSeries"]

# constants
wT2 = 40
fT2 = 80
fat_amp = [0.0586, 0.0109, 0.0618, 0.1412, 0.66, 0.0673]
fat_cshift = [-101.1, 208.3, 281.0, 305.7, 395.6, 446.2]

# phantom
gtparams = loadmat(DATADIR / "paramMap.mat")["paramMap"]
wt1map = gtparams["T1"][0, 0]
dfmap = gtparams["Df"][0, 0]
famap = gtparams["B1"][0, 0]
ffmap = gtparams["FF"][0, 0]
mask = wt1map > 0
ft1map = [350] * mask



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


# generate images
config = io.config.read("mrf_sequence.json")
mrfseq = T1MRF(**config)

# water signal
wsig = mrfseq(T1=wt1map[mask], T2=wT2, att=famap[mask], g=-dfmap[mask]/1000)

# fat signal
opts = {"eval": "dot(signal, wfat)", "args": {"wfat": fat_amp}}
fsig = mrfseq(T1=ft1map[mask], T2=fT2, att=famap[mask], g=-(np.c_[dfmap[mask]] + fat_cshift)/1000, **opts)

# merge signals
sig = wsig * (1 - ffmap[mask]) + fsig * ffmap[mask]

# make volumes
def makevol(values, mask):
    """ fill volume """
    values = np.asarray(values)
    new = np.zeros(mask.shape, dtype=values.dtype)
    new[mask] = values
    return new

OUTDIR = Path("outdir")
OUTDIR.mkdir(exist_ok=True)
# io.write(OUTDIR / 'sig0', makevol(np.abs(sig[0]), mask))

# load trajectory
kspace = loadmat(DATADIR / "KSpaceData.mat")
traj = kspace["KSpaceTraj"].T
kdata = kspace["KSpaceData"].T

npoint = traj.shape[1]
density = np.abs(np.linspace(-1, 1, npoint))

# simdata
kdata2 = np.array([
    finufft.nufft2d2(t.real, t.imag, makevol(s, mask))
    for t,s in zip(traj * np.pi * 2, sig)
])

# undersampled volume
uvol0 = finufft.nufft2d1(
    traj[0:8].real.ravel() * 2 * np.pi,
    traj[0:8].imag.ravel() * 2 * np.pi,
    (kdata[0:8] * density**0.5 * 255).ravel(),
    mask.shape,
)
io.write(OUTDIR / "uvol0", np.abs(uvol0))

uvol = finufft.nufft2d1(
    traj[0:8].real.ravel() * 2 * np.pi,
    traj[0:8].imag.ravel() * 2 * np.pi,
    (kdata2[0:8] * density / 255 * np.sqrt(mask.size)).ravel(),
    # kdata[0:8].ravel(),
    mask.shape,
)
io.write(OUTDIR / "uvol", np.abs(uvol))

""" Simulate the T1 map MRF sequence

"""
import itertools
from pathlib import Path
import h5py
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from epgpy import epg

# load sequence values
DIR_PARAMS = Path("mrf_params")
TE = loadmat(DIR_PARAMS / "TE_fingerprint.mat")["TE"].ravel()
TR = loadmat(DIR_PARAMS / "TR_fingerprint.mat")["TR"].ravel()
B1 = loadmat(DIR_PARAMS / "B1_fingerprint.mat")["B1"].ravel()
FA = 5 # degree
TI = 8.32 # ms
nspoke = 1400

class T1MRF:
    def __init__(self):
        # build sequence
        self.inversion = epg.T(180, 0)
        seq = [epg.Offset(TI)]
        for i in range(nspoke):
            dur1 = TE[i]
            dur2 = TR[i] - TE[i]
            echo = [
                epg.T(FA * B1[i], 90),
                epg.Wait(dur1),
                epg.ADC,
                epg.Wait(dur2),
                epg.SPOILER,
            ]
            seq.extend(echo)
        self._seq = seq

    def __call__(self, adc_time=False, **kwargs):
        """ modify sequence """
        seq = [self.inversion, epg.modify(self._seq, **kwargs)]
        sim = epg.simulate(seq, adc_time=adc_time)
        return tuple(np.array(arr) for arr in sim)


#
# load ref dict
paramfile = DIR_PARAMS / "paramDico_175_light.mat"
print(f"Loading parameters: {paramfile}")
h5 = h5py.File(paramfile, "r")
dict_Df = np.asarray(h5["paramDico/Df"]).ravel()
dict_FA = np.asarray(h5["paramDico/FA"]).ravel()
dict_T1_water = np.asarray(h5[h5["paramDico/T1"][0][0]]).ravel()
dict_T1_fat = np.asarray(h5[h5["paramDico/T1"][1][0]]).ravel()
# values
values_water = np.asarray(h5["paramDico/dico_water"]).T
values_water = values_water["real"] + 1j * values_water["imag"]
values_fat = np.asarray(h5["paramDico/dico_fat"]).T
values_fat = values_fat["real"] + 1j * values_fat["imag"]

# compare EPG sequence and dict
seq = T1MRF()

def sample_dict_water():
    ielement = np.random.randint(len(dict_T1_water) * len(dict_Df) * len(dict_FA))
    indices = np.unravel_index(ielement, values_water.shape[:-1])
    T1_ref = dict_T1_water[indices[0]]
    df_ref = dict_Df[indices[1]]
    att_ref = dict_FA[indices[2]]
    ref = values_water[indices]
    sim = seq(T1=T1_ref, T2=35, att=att_ref, g=-df_ref / 1000)
    sim = [np.mean(sim[i:i + 8]) for i in range(0, nspoke, 8)]
    return (T1_ref, att_ref, df_ref), ref, sim

def sample_dict_fat():
    ielement = np.random.randint(len(dict_T1_fat) * len(dict_Df) * len(dict_FA))
    indices = np.unravel_index(ielement, values_fat.shape[:-1])
    T1_ref = dict_T1_fat[indices[0]]
    df_ref = dict_Df[indices[1]]
    att_ref = dict_FA[indices[2]]
    ref = values_fat[indices]
    sim = seq(T1=T1_ref, T2=120, att=att_ref, g=(-df_ref / 1000 - 0.42))
    sim = [np.mean(sim[i:i + 8]) for i in range(0, nspoke, 8)]
    return (T1_ref, att_ref, df_ref), ref, sim


fig, axes = plt.subplots(nrows=2, ncols=2)

labels = []
for i in range(5):
    wparams, ref, sim = sample_dict_water()
    lines = axes[0, 0].plot(np.real(ref), alpha=0.5)
    color = lines[0].get_color()
    axes[0, 0].plot(np.real(sim), "x", alpha=0.5, color=color)

    axes[0, 1].plot(np.imag(ref), alpha=0.5, color=color)
    axes[0, 1].plot(np.imag(sim), "x", alpha=0.5, color=color)

    fparams, ref, sim = sample_dict_fat()
    axes[1, 0].plot(np.real(sim), alpha=0.5, color=color)
    axes[1, 0].plot(np.real(ref), "x", alpha=0.5, color=color)

    axes[1, 1].plot(np.imag(ref), alpha=0.5, color=color)
    axes[1, 1].plot(np.imag(sim), "x", alpha=0.5, color=color)

    labels.append([wparams, fparams, lines[0]])


fig.suptitle("Reference dict vs EPG simulation")
axes[0,0].set_title("water (real)")
axes[0,1].set_title("water (imag)")
axes[0,1].legend([l[2] for l in labels], [l[0] for l in labels])
# axes[1].grid("both")
axes[1, 0].set_title("fat (real)")
axes[1, 1].set_title("fat (imag)")
axes[1, 1].legend([l[2] for l in labels], [l[1] for l in labels])
axes[1, 0].set_xlabel("spoke index")
axes[1, 1].set_xlabel("spoke index")
plt.show()

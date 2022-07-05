
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import *
from image_series import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
from movements import *
from dictoptimizers import *
import glob
from tqdm import tqdm
import pickle
from scipy.io import savemat

## Random map simulation



class T1MRFSS:
    def __init__(self, FA, TI, TE, TR, B1,T_recovery,nrep):
        print(T_recovery)
        print(nrep)
        """ build sequence """
        seqlen = len(TE)
        self.TR=TR
        self.inversion = epg.T(180, 0) # perfect inversion
        self.T_recovery=T_recovery
        self.nrep=nrep
        seq=[]
        for r in range(nrep):
            curr_seq = [epg.Offset(TI)]
            for i in range(seqlen):
                echo = [
                    epg.T(FA * B1[i], 90),
                    epg.Wait(TE[i]),
                    epg.ADC,
                    epg.Wait(TR[i] - TE[i]),
                    epg.SPOILER,
                ]
                curr_seq.extend(echo)
            recovery=[epg.Wait(T_recovery)]
            curr_seq.extend(recovery)
            self.len_rep = len(curr_seq)
            seq.extend(curr_seq)
        self._seq = seq

    def __call__(self, T1, T2, g, att, calc_deriv=False,**kwargs):
        """ simulate sequence """
        seq=[]
        for r in range(self.nrep):
            curr_seq=self._seq[r*self.len_rep:(r+1)*(self.len_rep)]
            curr_seq=[self.inversion, epg.modify(curr_seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
            seq.extend(curr_seq)
        #seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        if not(calc_deriv):
            return np.asarray(epg.simulate(seq, **kwargs))
        else:
            return epg.simulate(seq,calc_deriv=calc_deriv, **kwargs)

#


dictfile = "mrf175_SimReco2_mid_point.dict"
dictfile = "mrf175_SimReco2_light.dict"
#dictfile = "mrf175_CS.dict"


with open("./mrf_sequence.json") as f:
    sequence_config = json.load(f)

with open("./mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)


# generate signals
wT1 = dict_config["water_T1"]
fT1 = dict_config["fat_T1"]
wT2 = dict_config["water_T2"]
fT2 = dict_config["fat_T2"]
att = dict_config["B1_att"]
df = dict_config["delta_freqs"]
df = [- value / 1000 for value in df] # temp
# df = np.linspace(-0.1, 0.1, 101)

rep=2
TR_total = 7000

Treco = TR_total-np.sum(sequence_config["TR"])
# other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=rep


seq=T1MRFSS(**sequence_config)

water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])




indices=[]
for i in range(1,water.ndim):
    indices.append(np.random.choice(water.shape[i]))

plt.figure()
plt.title("Starting point readout as a function of rep : T1 {} B1 {} Df {}".format(wT1[-1], att[indices[2]], df[indices[3]]))
plt.plot(np.real(water[::1400,-1,indices[1],indices[2],indices[3]]),label=k)

fig,ax=plt.subplots(2,1)
fig.suptitle("T1 {} B1 {} Df {}".format(wT1[-1],att[indices[2]],df[indices[3]]))
ax[0].plot(np.real(water[:,-1,indices[1],indices[2],indices[3]].reshape(rep,1400)).T)
ax[0].set_title("Real Part")
ax[1].plot(np.imag(water[:,-1,indices[1],indices[2],indices[3]].reshape(rep,1400)).T)
ax[1].set_title("Imag Part")

plt.close("all")
num_sig=5
for j in range(num_sig):
    indices=[]
    for i in range(1,water.ndim):
        indices.append(np.random.choice(water.shape[i]))

    plt.figure()
    plt.title("Starting point readout as a function of rep : T1 {} B1 {} Df {}".format(wT1[indices[0]], att[indices[2]], df[indices[3]]))
    plt.plot(np.real(water[::1400,indices[0],indices[1],indices[2],indices[3]]),label=k)

    fig,ax=plt.subplots(2,1)
    fig.suptitle("T1 {} B1 {} Df {}".format(wT1[indices[0]],att[indices[2]],df[indices[3]]))
    ax[0].plot(np.real(water[:,indices[0],indices[1],indices[2],indices[3]].reshape(rep,1400)).T)
    ax[0].set_title("Real Part")
    ax[1].plot(np.imag(water[:,indices[0],indices[1],indices[2],indices[3]].reshape(rep,1400)).T)
    ax[1].set_title("Imag Part")

# fat_amp = dict_config["fat_amp"]
# fat_cs = dict_config["fat_cshift"]
# fat_cs = [- value / 1000 for value in fat_cs] # temp


















#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import *
from image_series import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
from movements import *
from dictoptimizers import *
import glob
from tqdm import tqdm
import pickle
from scipy.io import savemat



with open("./mrf_sequence_adjusted.json") as f:
    sequence_config = json.load(f)

with open("./mrf_dictconf_Dico2_Invivo.json") as f:
    dict_config = json.load(f)


# generate signals
wT1 = dict_config["water_T1"]
fT1 = dict_config["fat_T1"]
wT2 = dict_config["water_T2"]
fT2 = dict_config["fat_T2"]
att = dict_config["B1_att"]
df = dict_config["delta_freqs"]
df = [- value / 1000 for value in df] # temp
# df = np.linspace(-0.1, 0.1, 101)

rep=2
TR_total = 7500

Treco = TR_total-np.sum(sequence_config["TR"])
# other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=rep


seq=T1MRFSS(**sequence_config)

water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])






dictfile = "mrf175_SimReco2_mid_point.dict"
dictfile = "mrf175_SimReco2_light.dict"
#dictfile = "mrf175_CS.dict"
dictfile = "mrf175_Dico2_Invivo_adjusted_TR7500.dict"



sim_mode="mean"
overwrite=False

fat_amp = dict_config["fat_amp"]
fat_cs = dict_config["fat_cshift"]
fat_cs = [- value / 1000 for value in fat_cs] # temp

# other options

window = dict_config["window_size"]

# water
printer("Generate water signals.")
water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])
water=water.reshape((rep,-1)+water.shape[1:])[-1]

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
fat=fat.reshape((rep,-1)+fat.shape[1:])[-1]

if sim_mode == "mean":
    fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
elif sim_mode == "mid_point":
    fat = fat[(int(window / 2) - 1):-1:window]
else:
    raise ValueError("Unknow sim_mode")

water=np.array(water)
fat=np.array(fat)
# join water and fat
printer("Build dictionary.")
keys = list(itertools.product(wT1, fT1, att, df))
values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

printer("Save dictionary.")
mrfdict = dictsearch.Dictionary(keys, values)
mrfdict.save(dictfile, overwrite=overwrite)







##############Sequence no inversion
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import *
from image_series import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
from movements import *
from dictoptimizers import *
import glob
from tqdm import tqdm
import pickle
from scipy.io import savemat




class T1MRFNoInv:
    def __init__(self, FA, TI, TE, TR, B1):
        """ build sequence """
        seqlen = len(TE)
        self.TR=TR
        #self.inversion = epg.T(180, 0) # perfect inversion
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
        seq = [epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        if not(calc_deriv):
            return np.asarray(epg.simulate(seq, **kwargs))
        else:
            return epg.simulate(seq,calc_deriv=calc_deriv, **kwargs)


with open("./mrf_sequence_adjusted.json") as f:
    sequence_config = json.load(f)

with open("./mrf_dictconf_SimReco2_light.json") as f:
    dict_config = json.load(f)


# generate signals
wT1 = dict_config["water_T1"]
fT1 = dict_config["fat_T1"]
wT2 = dict_config["water_T2"]
fT2 = dict_config["fat_T2"]
att = dict_config["B1_att"]
df = dict_config["delta_freqs"]
df = [- value / 1000 for value in df] # temp
# df = np.linspace(-0.1, 0.1, 101)

seq=T1MRFNoInv(**sequence_config)





dictfile = "mrf175_SimReco2_mid_point.dict"
dictfile = "mrf175_SimReco2_light.dict"
#dictfile = "mrf175_CS.dict"
dictfile = "mrf175_SimReco2_light_adjusted_NoInv.dict"



sim_mode="mean"
overwrite=False

fat_amp = dict_config["fat_amp"]
fat_cs = dict_config["fat_cshift"]
fat_cs = [- value / 1000 for value in fat_cs] # temp

# other options

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

water=np.array(water)
fat=np.array(fat)
# join water and fat
printer("Build dictionary.")
keys = list(itertools.product(wT1, fT1, att, df))
values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

printer("Save dictionary.")
mrfdict = dictsearch.Dictionary(keys, values)
mrfdict.save(dictfile, overwrite=overwrite)

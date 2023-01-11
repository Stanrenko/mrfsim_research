import pandas as pd
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import *
from image_series import *
from utils_mrf import *
from utils_simu import *

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





dictfile = "mrf175_SimReco2_mid_point.dict"
dictfile = "mrf175_SimReco2_light.dict"
#dictfile = "mrf175_CS.dict"


with open("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_optim_FF.json") as f:
    sequence_config = json.load(f)

#with open("./mrf_sequence_adjusted.json") as f:
#    sequence_config = json.load(f)

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

rep=4
TR_total = np.sum(sequence_config["TR"])

Treco = 3000
# other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=rep


nb_allspokes=len(sequence_config["TE"])

seq=T1MRFSS(**sequence_config)

water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])


indices=[]
for i in range(1,water.ndim):
    indices.append(np.random.choice(water.shape[i]))

fig,ax=plt.subplots()
ax.set_title("Worst T1 Starting point readout as a function of rep : T1 {} B1 {} Df {}".format(wT1[-1], att[0], df[4]))
ax.plot(range(rep),np.real(water[::nb_allspokes,-1,0,0,4]),color="grey")
ax.set_xticks(list(range(rep)))


fig,ax=plt.subplots(2,1)
fig.suptitle("T1 {} B1 {} Df {}".format(wT1[-1],att[0],df[4]))
ax[0].plot(np.real(water[:,-1,0,0,4]))
ax[0].set_title("Real Part")
ax[1].plot(np.imag(water[:,-1,0,0,4]))
ax[1].set_title("Imag Part")

plt.close("all")
num_sig=5
for j in range(num_sig):
    indices=[]
    for i in range(1,water.ndim):
        indices.append(np.random.choice(water.shape[i]))

    plt.figure()
    plt.title("Starting point readout as a function of rep : T1 {} B1 {} Df {}".format(wT1[indices[0]], att[indices[2]], df[indices[3]]))
    plt.plot(np.real(water[::nb_allspokes,indices[0],indices[1],indices[2],indices[3]]))

    fig,ax=plt.subplots(2,1)
    fig.suptitle("T1 {} B1 {} Df {}".format(wT1[indices[0]],att[indices[2]],df[indices[3]]))
    ax[0].plot(np.real(water[:,indices[0],indices[1],indices[2],indices[3]].reshape(rep,nb_allspokes)).T)
    ax[0].set_title("Real Part")
    ax[1].plot(np.imag(water[:,indices[0],indices[1],indices[2],indices[3]].reshape(rep,nb_allspokes)).T)
    ax[1].set_title("Imag Part")

plt.close("all")
num_sig=5
fixed_indices = []
for i in range(2, water.ndim):
    fixed_indices.append(np.random.choice(water.shape[i]))

plt.figure()
for j in range(num_sig):
    indices=[np.random.choice(water.shape[1])]
    indices=indices+fixed_indices

    plt.title("Starting point readout as a function of rep : T1 {} B1 {} Df {}".format(wT1[indices[0]], att[indices[2]], df[indices[3]]))
    plt.plot(np.real(water[::nb_segments,indices[0],indices[1],indices[2],indices[3]]),label=wT1[indices[0]])
plt.legend()

fig,ax=plt.subplots(2,1)
for j in range(num_sig):
    indices = [np.random.choice(water.shape[1])]
    indices = indices + fixed_indices
    fig.suptitle("T1 {} B1 {} Df {}".format(wT1[indices[0]],att[indices[2]],df[indices[3]]))
    ax[0].plot(np.real(water[:,indices[0],indices[1],indices[2],indices[3]].reshape(rep,1400)).T,label=wT1[indices[0]])
    ax[0].set_title("Real Part")
    ax[1].plot(np.imag(water[:,indices[0],indices[1],indices[2],indices[3]].reshape(rep,1400)).T,label=wT1[indices[0]])
    ax[1].set_title("Imag Part")
plt.legend()

# fat_amp = dict_config["fat_amp"]
# fat_cs = dict_config["fat_cshift"]
# fat_cs = [- value / 1000 for value in fat_cs] # temp

water_reshaped=water.reshape((rep,1400)+water.shape[1:])
diff_rep = np.linalg.norm(water_reshaped[-1]-water_reshaped[-2],axis=0)
diff_rep_rel=diff_rep/np.linalg.norm(water_reshaped[1],axis=0)


print(np.max(diff_rep))
print(np.unravel_index(np.argmax(diff_rep),diff_rep.shape))
plt.figure()
plt.hist(diff_rep.flatten())

print(np.max(diff_rep_rel))
print(np.unravel_index(np.argmax(diff_rep_rel),diff_rep_rel.shape))
plt.figure()
plt.hist(diff_rep_rel.flatten())

max_error_index=np.unravel_index(np.argmax(diff_rep),diff_rep.shape)
error=np.abs(water_reshaped[-1,:,max_error_index[0],max_error_index[1],max_error_index[2],max_error_index[3]]-water_reshaped[-2,:,max_error_index[0],max_error_index[1],max_error_index[2],max_error_index[3]])
plt.figure()
plt.title("Max error : T1 {} B1 {} Df {}".format(wT1[max_error_index[0]], att[max_error_index[2]], df[max_error_index[3]]))
plt.plot(error)


ind_error_on_pattern=np.argmax(error)

ind_start=0
num_ts=10

rep1=1
rep2=2

plt.figure()
plt.plot(np.arange(num_ts),np.real(water_reshaped[rep1,ind_start:ind_start+num_ts,max_error_index[0],max_error_index[1],max_error_index[2],max_error_index[3]]),label="Real rep {}".format(rep1))
plt.plot(np.arange(num_ts),np.real(water_reshaped[rep2,ind_start:ind_start+num_ts,max_error_index[0],max_error_index[1],max_error_index[2],max_error_index[3]]),"ro",label="Real rep {}".format(rep2))
plt.legend()

plt.figure()
plt.plot(np.arange(num_ts),np.imag(water_reshaped[rep1,ind_start:ind_start+num_ts,max_error_index[0],max_error_index[1],max_error_index[2],max_error_index[3]]),label="Imag rep {}".format(rep1))
plt.plot(np.arange(num_ts),np.imag(water_reshaped[rep2,ind_start:ind_start+num_ts,max_error_index[0],max_error_index[1],max_error_index[2],max_error_index[3]]),"ro",label="Imag rep {}".format(rep2))
plt.legend()

plt.figure()
plt.plot(np.arange(num_ts),np.abs(water_reshaped[rep1,ind_start:ind_start+num_ts,max_error_index[0],max_error_index[1],max_error_index[2],max_error_index[3]]),label="Mag rep {}".format(rep1))
plt.plot(np.arange(num_ts),np.abs(water_reshaped[rep2,ind_start:ind_start+num_ts,max_error_index[0],max_error_index[1],max_error_index[2],max_error_index[3]]),"ro",label="Mag rep {}".format(rep2))
plt.legend()





###########MRF Steady State Dico Gen##################

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



with open("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400.json") as f:
    sequence_config = json.load(f)

#with open("./mrf_sequence_adjusted_BW_540.json") as f:
#    sequence_config = json.load(f)


with open("./mrf_dictconf_Dico2_Invivo.json") as f:
    dict_config = json.load(f)

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

rep=2
TR_total = np.sum(sequence_config["TR"])

Treco = TR_total-np.sum(sequence_config["TR"])
Treco=4000
#Treco=3130
#Treco=4000

# other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=rep


seq=T1MRFSS(**sequence_config)






dictfile = "mrf175_SimReco2_mid_point.dict"
dictfile = "mrf175_SimReco2_light.dict"
#dictfile = "mrf175_CS.dict"
dictfile = "mrf175_Dico2_Invivo_adjusted_TR7000.dict"
dictfile = "mrf_dictconf_SimReco2_light_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_reco4.dict"
dictfile = "mrf175_SimReco2_light.dict"



sim_mode="mean"
overwrite=True

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





#### MRF Dico Gen##########

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

with open("./mrf_sequence.json") as f:
    sequence_config = json.load(f)


with open("./mrf_dictconf_Dico2_Invivo_light_for_matching.json") as f:
    dict_config = json.load(f)

#with open("./mrf_dictconf_SimReco2.json") as f:
#    dict_config = json.load(f)

# generate signals
wT1 = dict_config["water_T1"]
fT1 = dict_config["fat_T1"]
wT2 = dict_config["water_T2"]
fT2 = dict_config["fat_T2"]
att = dict_config["B1_att"]
df = dict_config["delta_freqs"]
df = [- value / 1000 for value in df] # temp
# df = np.linspace(-0.1, 0.1, 101)

#min_TR_delay=1.84
#TE_list=np.array(sequence_config["TE"])
#TR_list=list(TE_list+min_TR_delay)
#sequence_config["TR"]=TR_list

seq=T1MRF(**sequence_config)

dictfile = "mrf175_Dico2_Invivo_light_for_matching.dict"

sim_mode="mean"
overwrite=True

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








import numpy as np
import pandas as pd
import json
with open("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1.json") as f:
    sequence_config = json.load(f)

TE=sequence_config["TE"]
TE=np.array(TE)

TE=np.maximum(TE,2.2)
TE=np.round(TE,2)
#TE=np.maximum(np.round(TE,2),2.2)

TE_file="TE_temp.text"

with open(TE_file,"w") as file:
    for te in TE:
        file.write(str(te)+"\n")

#pd.DataFrame(np.maximum(np.round(TE,2),2.2)).to_clipboard()

FA=sequence_config["B1"]
FA=np.array(FA)
FA=np.round(FA,3)
#pd.DataFrame(np.round(FA,3)).to_clipboard(excel=False,index=False)

FA_file="FA_temp.text"

with open(FA_file,"w") as file:
    for fa in FA:
        file.write(str(fa)+"\n")
print(np.min(np.array(TE)))
print(np.max(np.array(TE)))
print(np.min(np.array(FA)))
print(np.max(np.array(FA)))

from utils_simu import *
generate_epg_dico_T1MRFSS_from_sequence_file("mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1.json","./mrf_dictconf_Dico2_Invivo.json",3.95)



generate_epg_dico_T1MRFSS_from_sequence_file("mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF.json","./mrf_dictconf_SimReco2.json",3)





#############PCA##############@

import pandas as pd
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
from Transformers import *

dictfile="mrf_dictconf_SimReco2_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_reco3.dict"

mrfdict = dictsearch.Dictionary()
mrfdict.load(dictfile, force=True)
keys = mrfdict.keys
array_water = mrfdict.values[:, :, 0]
array_fat = mrfdict.values[:, :, 1]

array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

del pca_water
pca_water = PCAComplex(n_components_=10)

pca_water.fit(array_water_unique)
#pca_fat.fit(array_fat_unique)


plt.figure()
plt.plot(pca_water.explained_variance_ratio_)

pca_water.plot_retrieved_signal(array_water_unique,i=np.random.choice(array_water_unique.shape[0]))

pca_fat = PCAComplex(n_components_=5)

pca_fat.fit(array_fat_unique)
#pca_fat.fit(array_fat_unique)


plt.figure()
plt.plot(pca_fat.explained_variance_ratio_)

pca_fat.plot_retrieved_signal(array_fat_unique,i=np.random.choice(array_fat_unique.shape[0]))







































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
    def __init__(self, FA, TE,TI, TR, B1):
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
#del sequence_config["TI"]
sequence_config["TI"]=0.01

seq=T1MRFNoInv(**sequence_config)





dictfile = "mrf175_SimReco2_mid_point.dict"
dictfile = "mrf175_SimReco2_light.dict"
#dictfile = "mrf175_CS.dict"
dictfile = "mrf175_SimReco2_light_adjusted_NoInv.dict"



sim_mode="mean"
overwrite=True

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






seq = T1MRF(**sequence_config)

wT1=1380
wT2=40
water = seq(T1=wT1, T2=wT2, att=[[1.0]], g=[[[0.0]]])

water.shape

plt.plot(np.squeeze(water))







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




suffix="_PWCR"
#suffix="_PWWeighted"
#suffix="_"

nb_allspokes = 144
nspoke=8
nb_segments=nb_allspokes
ntimesteps=int(nb_segments/nspoke)

#suffix="_plateau600"
#suffix="_constantTE_last"
#suffix=""

with open("mrf{}_SeqFF{}_config.json".format(nb_allspokes,suffix)) as f:
    sequence_config = json.load(f)


with open("mrf_dictconf_SimReco2_light.json") as f:
    dict_config = json.load(f)


class FFMRFSS:
    def __init__(self, FA, TE,T_recovery,nrep):
        print(T_recovery)
        print(nrep)
        """ build sequence """
        seqlen = len(TE)
        self.TR=np.array(TE)+1.24
        self.T_recovery=T_recovery
        self.nrep=nrep
        seq=[]
        for r in range(nrep):
            curr_seq = [epg.Wait(0.0001)]
            for i in range(seqlen):
                echo = [
                    epg.T(FA, 90),
                    epg.Wait(TE[i]),
                    epg.ADC,
                    epg.Wait(1.24),
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
            curr_seq=epg.modify(curr_seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)
            seq.extend(curr_seq)
        #seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        if not(calc_deriv):
            return np.asarray(epg.simulate(seq, **kwargs))
        else:
            return epg.simulate(seq,calc_deriv=calc_deriv, **kwargs)


rep=20

Treco = 0.0001
# other options
sequence_config["T_recovery"]=Treco
sequence_config["nrep"]=rep

sequence_config
seq = FFMRFSS(**sequence_config)


# generate signals
wT1 = dict_config["water_T1"]
fT1 = dict_config["fat_T1"]
wT2 = dict_config["water_T2"]
fT2 = dict_config["fat_T2"]
att = dict_config["B1_att"]
df = dict_config["delta_freqs"]
df = [- value / 1000 for value in df] # temp
# df = np.linspace(-0.1, 0.1, 101)


water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])




indices=[]
for i in range(1,water.ndim):
    indices.append(np.random.choice(water.shape[i]))
indices[0]=-1
indices[2]=0
plt.figure()
plt.title("Starting point readout as a function of rep : T1 {} B1 {} Df {}".format(wT1[indices[0]], att[indices[2]], df[indices[3]]))
plt.plot(np.real(water[::int(nb_allspokes),indices[0],indices[1],indices[2],indices[3]]))

#-> 6/10 reps for steady state

fig,ax=plt.subplots(2,1)
fig.suptitle("T1 {} B1 {} Df {}".format(wT1[indices[0]],att[indices[2]],df[indices[3]]))
ax[0].plot(np.real(water[:,indices[0],indices[1],indices[2],indices[3]].reshape(rep,nb_allspokes)).T)
ax[0].set_title("Real Part")
ax[1].plot(np.imag(water[:,indices[0],indices[1],indices[2],indices[3]].reshape(rep,nb_allspokes)).T)
ax[1].set_title("Imag Part")

plt.close("all")
num_sig=5
for j in range(num_sig):
    indices=[]
    for i in range(1,water.ndim):
        indices.append(np.random.choice(water.shape[i]))

    plt.figure()
    plt.title("Starting point readout as a function of rep : T1 {} B1 {} Df {}".format(wT1[indices[0]], att[indices[2]], df[indices[3]]))
    plt.plot(np.real(water[::(nb_allspokes),indices[0],indices[1],indices[2],indices[3]]))

    fig,ax=plt.subplots(2,1)
    fig.suptitle("T1 {} B1 {} Df {}".format(wT1[indices[0]],att[indices[2]],df[indices[3]]))
    ax[0].plot(np.real(water[:,indices[0],indices[1],indices[2],indices[3]].reshape(rep,nb_allspokes)).T)
    ax[0].set_title("Real Part")
    ax[1].plot(np.imag(water[:,indices[0],indices[1],indices[2],indices[3]].reshape(rep,nb_allspokes)).T)
    ax[1].set_title("Imag Part")




import pandas as pd
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import *
from image_series import *
from utils_mrf import *
from utils_simu import *

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


plt.close("all")


with open("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v5.json") as f:
    sequence_config_1 = json.load(f)

with open("./mrf_sequence_adjusted.json") as f:
    sequence_config_2 = json.load(f)

with open("./mrf_sequence_adjusted_760.json") as f:
    sequence_config_3 = json.load(f)

plt.figure()
plt.plot(np.array(sequence_config_2["B1"])*5,label="MRF T1-FF 1400 spokes")
#plt.plot(np.array(sequence_config_3["B1"])*5,label="MRF T1-FF 760 spokes")
plt.plot(np.array(sequence_config_1["B1"])*5,label="Optimized MRF T1-FF")
plt.legend()

plt.figure()
plt.plot(np.array(sequence_config_2["TE"]),label="MRF T1-FF 1400 spokes")
#plt.plot(np.array(sequence_config_3["TE"]),label="MRF T1-FF 760 spokes")
plt.plot(np.array(sequence_config_1["TE"]),label="Optimized T1-FF")
plt.legend()

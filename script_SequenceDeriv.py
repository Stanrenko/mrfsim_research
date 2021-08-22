
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
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

dictfile = "./mrf175_SimReco2_mid_point.dict"
dictfile = "./mrf175_SimReco2.dict"
#dictfile = "mrf175_CS.dict"

with open("./mrf_sequence.json") as f:
    sequence_config = json.load(f)

with open("./mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

seq = T1MRF(**sequence_config)

load_maps=False
save_maps = False

type="SquarePhantom"


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
water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]],calc_deriv=True)

derivs=water[1]


Fisher_information =0

for k in derivs.keys():

    plt.figure()
    derivatives = np.abs(np.array(derivs[k])[:, 0])
    lines = plt.plot(derivatives.reshape((len(derivatives), -1)))
    plt.title(f"MRF deriv {k}")
    plt.grid()
    plt.xlabel("iter")
    plt.ylabel("sensibility")
    plt.legend(lines)
    plt.show()
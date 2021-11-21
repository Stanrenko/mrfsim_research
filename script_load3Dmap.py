
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF

from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time

import pickle

filename="./data/InVivo/3D/20211105_TestCS_MRF/meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"
filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00043_FID42065_raFin_3D_tra_1x1x5mm_us2_vivo.dat"
filename="./data/InVivo/3D/20211119_EV_MRF/meas_MID00044_FID42066_raFin_3D_tra_1x1x5mm_us4_vivo.dat"

file_map = filename.split(".dat")[0] + "_volumes_MRF_map.pkl"
file = open(file_map, "rb")
all_maps = pickle.load(file)

iter=0
map_rebuilt=all_maps[iter][0]
mask=all_maps[iter][1]

keys_simu = list(map_rebuilt.keys())
values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
map_for_sim = dict(zip(keys_simu, values_simu))

map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
map_Python.buildParamMap()

plot_image_grid(list(map_for_sim["ff"][:]),nb_row_col=(8,6),cbar_mode="single",title="Fat fraction upper body")

plot_image_grid(list(map_for_sim["ff"][6:-6]),nb_row_col=(6,6),cbar_mode="single",title="Fat fraction upper body")


map_Python.plotParamMap("ff",sl=24)

map_Python.animParamMap("ff")
map_Python.animParamMap("wT1")
map_Python.animParamMap("attB1")
map_Python.animParamMap("df")



path = r"C:/Users/c.slioussarenko/PythonRepositories"
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")

from mutools import io

key="ff"
file_mha = filename.split(".dat")[0] + "_MRF_map_{}.mha".format(key)
io.write(file_mha,map_for_sim[key],tags={"spacing":[5,1,1]})




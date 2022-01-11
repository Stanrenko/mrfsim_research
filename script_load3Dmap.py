
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
filename="./data/InVivo/3D/20211122_EV_MRF/meas_MID00146_FID42269_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"
filename="./data/InVivo/3D/20211122_EV_MRF/meas_MID00149_FID42272_raFin_3D_tra_1x1x5mm_USx2.dat"
filename="./data/InVivo/3D/20211122_EV_MRF/meas_MID00147_FID42270_raFin_3D_tra_1x1x5mm_FULL_incoherent.dat"
filename="./data/InVivo/3D/20211122_EV_MRF/meas_MID00148_FID42271_raFin_3D_tra_1x1x5mm_FULL_high_res.dat"
filename="./data/InVivo/3D/20211129_BM/meas_MID00085_FID43316_raFin_3D_FULL_highRES_incoh.dat"
filename="./data/InVivo/3D/20211105_TestCS_MRF/meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro.dat"
filename="./data/InVivo/3D/20211105_TestCS_MRF/meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro_volumes_norm_vol.dat"
filename="./data/InVivo/3D/20211122_EV_MRF/meas_MID00146_FID42269_raFin_3D_tra_1x1x5mm_FULL_vitro_test.dat"
#filename="./data/InVivo/3D/20211209_AL_Tongue/meas_MID00258_FID45162_raFin_3D_tra_1x1x5mm_FULl.dat"
filename="./data/InVivo/3D/20211220_Phantom_MRF/meas_MID00026_FID47383_raFin_3D_tra_1x1x5mm_FULl.dat"
filename="./data/InVivo/3D/20220106/meas_MID00167_FID48477_raFin_3D_tra_1x1x5mm_FULL_new.dat"



file_map = filename.split(".dat")[0] + "_MRF_map.pkl"
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




from mutools import io

for key in ["ff","wT1","df","attB1"]:

    file_mha = filename.split(".dat")[0] + "_MRF_map_{}.mha".format(key)
    io.write(file_mha,map_for_sim[key],tags={"spacing":[15,4,4]})


folder =r"\\192.168.0.1\RMN_FILES"
file ="\meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro_mask_norm_vol.npy"

file_path = folder+file

mask = np.load(file_path)
animate_images(mask)

folder=r"\\192.168.0.1\RMN_FILES\0_Wip\New\1_Methodological_Developments\1_Methodologie_3T\&0_2021_MR_MyoMaps\3_Data\4_3D\Invivo\20211105_TestCS_MRF"
file="\meas_MID00042_FID40391_raFin_3D_tra_1x1x5mm_FULL_vitro_mask.npy"
file_path = folder+file

mask_base = np.load(file_path)

animate_images(mask_base)



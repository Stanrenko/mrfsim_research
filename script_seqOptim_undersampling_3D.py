



from utils_simu import *
from dictoptimizers import SimpleDictSearch
import json
from image_series import *
from trajectory import *
import pickle

with open("./mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

fat_amp = np.array(dict_config["fat_amp"])
fat_shift = -np.array(dict_config["fat_cshift"])

nspoke=8
spokes_count=760
ntimesteps=int(spokes_count/nspoke)
nb_channels=1
npoint = 512

image_size = (int(npoint/2), int(npoint/2))
nspoke=int(spokes_count/ntimesteps)
folder="./data/KneePhantom/Phantom1/"

file_dico=folder+"dicoMasks_Control_multislice_retained_13.pkl"
file_volume="volumes_Control_multislice_retained_13.npy"
file_map_gt=folder+"paramMap_Control_multislice_masked_13.pkl"


with open(file_dico,"rb") as file:
    dico_retained=pickle.load(file)

with open(file_map_gt,"rb") as file:
    dico_map_gt=pickle.load(file)

volumes_all=np.load(file_volume,allow_pickle=True)

#animate_images(mask_full)
keys_dico=list(dico_retained.keys())
params_mrf=np.array(keys_dico)

fileseq_basis="./mrf_sequence_adjusted.json"

with open(fileseq_basis, "r") as file:
    seq_config_base = json.load(file)

def cost_function_simul_breaks_random_FA_KneePhantom_3D(params):
    #global result
    global spokes_count

    global min_TR_delay
    global DFs
    global FFs
    global T1s
    global B1s
    global bound_min_FA
    global num_breaks_TE
    global num_params_FA

    #global bound_max_FA

    global lambda_FA
    global lambda_T1
    global lambda_time
    global lambda_FF
    global inversion

    global seq_config_base

    global useGPU
    global ntimesteps
    global params_mrf

    global volumes_all

    start_time=datetime.now()
    group_size = 8

    bound_max_FA=params[-1]
    params_for_curve=params[:-1]
    TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

    volumes_simu = np.zeros(volumes_all.shape[1:], dtype="complex64")

    for j in tqdm(range(params_mrf.shape[0])):
        param = params_mrf[j]
        ff = param[1]
        df = param[-1]
        b1 = param[-2]
        wT1 = param[0] / 1000

        s = simulate_gen_eq_signal(TR_, FA_, TE_, ff, df, wT1, 300 / 1000, b1, T_2w=40 / 1000,
                                                           T_2f=80 / 1000,
                                                           amp=fat_amp, shift=fat_shift, sigma=None,
                                                           group_size=group_size,
                                                           return_fat_water=False)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

        s = np.expand_dims(s.squeeze(), axis=1)

        volumes_simu += s * volumes_all[j]

    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
                                               return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T

    pca = True
    threshold_pca_bc = 10

    split = 100
    dict_optim_bc_cf = SimpleDictSearch(mask=None, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True)

    all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), volumes_simu)

    key = "wT1"
    map = all_maps[0][0][key][dico_map_gt["ff"]<0.7]*1000
    map_gt=dico_map_gt[key][dico_map_gt["ff"]<0.7]
    error = np.abs(map - map_gt)
    error_wT1 = np.mean(error/ map_gt)

    print("wT1 Cost : {}".format(error_wT1))

    std_wT1 = np.std(error/ map_gt)
    print("wT1 Std : {}".format(std_wT1))

    key = "ff"
    map = all_maps[0][0][key]
    map_gt = dico_map_gt[key]
    error = np.abs(map - map_gt)
    error_ff = np.mean(error)

    print("FF Cost : {}".format(error_ff))

    key = "df"
    map = all_maps[0][0][key]/1000
    map_gt = dico_map_gt[key]/1000
    error = np.abs(map - map_gt)
    error_df = np.mean(error)

    print("DF Cost : {}".format(error_df))

    key = "attB1"
    map = all_maps[0][0][key]
    map_gt = dico_map_gt[key]
    error = np.abs(map - map_gt)
    error_b1 = np.mean(error)

    print("B1 Cost : {}".format(error_b1))

    # num_breaks_TE=len(TE_breaks)
    FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
    # print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    print("Time Cost : {}".format(time_cost))
    #test_time=time_cost-TR_[-1]
    #print("Time Test : {}".format(test_time))

    result=lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_DF * error_df + lambda_B1 * error_b1 +lambda_FA * FA_cost + lambda_time * time_cost+lambda_stdT1 * std_wT1

    end_time=datetime.now()
    print(end_time-start_time)

    return result



class LogCost(object):
    def __init__(self,f):
        self.cost_values=[]
        self.it=1
        self.f=f

    def __call__(self,x,**kwargs):
        print("############# ITERATION {}###################".format(self.it))
        self.cost_values.append(self.f(x))
        self.it +=1

        with open("x_random_FA_US_3D_H6_log.pkl", "wb") as file:
            pickle.dump(x, file)

    def reset(self):
        self.cost_values = []
        self.it=1

#sigma2=0.02**2

spokes_count=760


min_TR_delay=1.87*10**-3
DFs=[-60,-30,0,30,60]
FFs=[0.]
B1s=[0.4,0.6,0.8,1]
T1s=np.array([1000,1100,1200,1300,1400,1500,1600,1800,2000])/1000
T1s=np.array([1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1600,1700,1800,2000])/1000

bound_min_TE=2.2/1000
bound_min_FA=5*np.pi/180
num_breaks_TE=3
#bound_max_FA=70*np.pi/180

H=4
num_params_FA=2*H
bounds=[(100,400)]*(num_breaks_TE)+[(0.0022,0.006)]*(num_breaks_TE+1)+[(0,1)]*num_params_FA+[(0,4)]+[(15*np.pi/180,70*np.pi/180)]

from scipy.optimize import LinearConstraint
A_TEbreaks=np.zeros((1,len(bounds)))
A_TEbreaks[0,:num_breaks_TE]=1
con1=LinearConstraint(A_TEbreaks,lb=-np.inf,ub=spokes_count-100)
constraints=(con1)

from scipy.optimize import differential_evolution
lambda_FA=0.
lambda_time=0.0
lambda_FF=2
lambda_T1=1
lambda_DF=4
lambda_B1=1
lambda_stdT1=0
inversion=True
useGPU=False

log_cost=LogCost(cost_function_simul_breaks_random_FA_KneePhantom_3D)

res=differential_evolution(cost_function_simul_breaks_random_FA_KneePhantom_3D,bounds=bounds,callback=log_cost,constraints=constraints,maxiter=1000)#,constraints=constraints)

import pickle
with open("res_simul_random_FA_US_all_params_3D.pkl","wb") as file:
    pickle.dump(res, file)

plt.figure()
plt.plot(log_cost.cost_values)

import pickle
with open("res_simul_random_FA_US.pkl","rb") as file:
    res=pickle.load(file)

cost_function_simul_breaks_random_FA_KneePhantom(res.x)

params=res.x
bound_max_FA=params[-1]
params_for_curve=params[:-1]
TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

plt.figure()
plt.plot(FA_[1:])


plt.figure()
plt.plot(TE_[1:])
res.x
generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_random_US_v2.json","./mrf_dictconf_SimReco2.json",FA_,TE_,3.73,1.87/1000,fileseq_basis="./mrf_sequence_adjusted.json")






import pickle
from utils_simu import *
with open("./optims/res_simul_random_20221028_125515.pkl","rb") as file:
    res=pickle.load(file)


with open("./optims/random_opt_config_20221028_125515.json","rb") as file:
    config=json.load(file)

#cost_function_simul_breaks_random(res.x)

inversion=config["inversion"]
min_TR_delay=config["min_TR_delay"]
spokes_count=config["spokes_count"]
bound_min_TE=config["bound_min_TE"]
bound_max_TE=config["bound_max_TE"]
bound_min_FA=config["bound_min_FA"]

params=res.x
bound_max_FA=params[-1]
params_for_curve=params[:-1]
TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,bound_min_TE,bound_max_TE,bound_min_FA,bound_max_FA,inversion)

plt.figure()
plt.plot(FA_[1:])


plt.figure()
plt.plot(TE_[1:])
res.x
generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v3.json","./mrf_dictconf_SimReco2_lightDFB1.json",FA_,TE_,4,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")





import pickle
from utils_simu import *
with open("./optims/res_simul_random_FA_20221031_144435.pkl","rb") as file:
    res=pickle.load(file)


with open("./optims/random_FA_opt_config_20221031_144435.json","rb") as file:
    config=json.load(file)

#cost_function_simul_breaks_random(res.x)

inversion=config["inversion"]
min_TR_delay=config["min_TR_delay"]
spokes_count=config["spokes_count"]
bound_min_TE=config["bound_min_TE"]
bound_max_TE=config["bound_max_TE"]
bound_min_FA=config["bound_min_FA"]
num_breaks_TE=config["num_breaks_TE"]
num_params_FA=config["num_params_FA"]




params=res.x
bound_max_FA=params[-1]
params_for_curve=params[:-1]
TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

plt.figure()
plt.plot(FA_[1:])


plt.figure()
plt.plot(TE_[1:])
res.x
generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_random_FA_v2.json","./mrf_dictconf_SimReco2_lightDFB1.json",FA_,TE_,4,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")




import pickle
from utils_simu import *
with open("x_random_FA_US_log.pkl","rb") as file:
    x=pickle.load(file)




#cost_function_simul_breaks_random(res.x)

inversion=True

spokes_count=760

min_TR_delay=1.87*10**-3
bound_min_TE=2.2/1000
bound_min_FA=5*np.pi/180
num_breaks_TE=3

H=4
num_params_FA=2*H


params=x
bound_max_FA=params[-1]
params_for_curve=params[:-1]
TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

# plt.figure()
# plt.plot(FA_[1:])
#
#
# plt.figure()
# plt.plot(TE_[1:])
# x
generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_random_FA_v1.json","./mrf_dictconf_SimReco2.json",FA_,TE_,3.95,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")




import pickle
from utils_simu import *
with open("./optims/res_simul_breaks_common_20221031_143619.pkl","rb") as file:
    res=pickle.load(file)


with open("./optims/breaks_common_opt_config_20221031_143619.json","rb") as file:
    config=json.load(file)

#cost_function_simul_breaks_random(res.x)

inversion=config["inversion"]
min_TR_delay=config["min_TR_delay"]
spokes_count=config["spokes_count"]
num_breaks_TE=config["num_breaks_TE"]
num_breaks_FA=config["num_breaks_FA"]

params=res.x
bound_max_FA=params[-1]
params_for_curve=params[:-1]
TR_,FA_,TE_=convert_params_to_sequence_breaks_common(res.x,min_TR_delay,num_breaks_TE,num_breaks_FA,spokes_count)
# #
plt.figure()
plt.plot(FA_[1:])


plt.figure()
plt.plot(TE_[1:])
res.x
generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v4.json","./mrf_dictconf_SimReco2_lightDFB1.json",FA_,TE_,3.8,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")




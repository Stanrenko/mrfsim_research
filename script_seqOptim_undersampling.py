from utils_simu import *
from dictoptimizers import SimpleDictSearch

#generate_epg_dico_T1MRFSS_from_sequence_file("mrf_sequence_adjusted.json","mrf_dictconf_SimReco2.json",4,sim_mode="mid_point",start=0,window=int(1400/50))

#TR_list,FA_list,TE_list=load_sequence_file("mrf_sequence_adjusted.json",3,1.87/1000)
#
# generate_epg_dico_T1MRFSS("mrf_sequence_adjusted_1_87.json","mrf_dictconf_Dico2_Invivo.json",FA_list,TE_list,4,1.87/1000)
#

#
#
# #https://cds.ismrm.org/protected/16MPresentations/videos/0429/index.html
#
# with open("./mrf_dictconf_SimReco2_light.json") as f:
#     dict_config = json.load(f)
#
# fat_amp = np.array(dict_config["fat_amp"])
# fat_shift = -np.array(dict_config["fat_cshift"])
#
#
# T1_w=1
# dTs=np.arange(-500,1000,100)*10**-3
# #dTs=np.array([500,500,1000])
# DFs=[-30,0,30]
# #DFs=[-60,-30,0,30,60]
# FFs=[0.,0.1,0.2,0.3,0.4,0.5,0.95]
# B1=[0.5,0.7,1]
# #DFs=[-60,-30,0,30,60]
# #FFs=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
# recovery=0
# sigma=0.02
# noise_size=100
# group_size=8
# noise_type="Absolute"
#
# recovery=0
# fileseq_1=r"./mrf_sequence_adjusted_optimized_M0_local_optim_correl_smooth.json"
# fileseq_1=r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter.json"
# fileseq_1=r"./mrf_sequence_adjusted.json"
# fileseq_1=r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_v2.json"
# recovery=0
# wT1_errors=[]
# ff_errors=[]
#
#
# fileseq_list=[
#     r"./mrf_sequence_adjusted.json",
#     r"./mrf_sequence_adjusted.json",
#     r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim.json",
#     r"./mrf_sequence_adjusted_optimized_M0_local_optim_correl.json",
#     r"./mrf_sequence_adjusted_optimized_M0_local_optim_correl_smooth.json",
#     r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter.json",
#     r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_v2.json"
# ]
#
# fileseq_list=[
#     r"./mrf_sequence_adjusted.json",
#     #r"./mrf_sequence_adjusted.json",
#     #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760.json",
#     #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized.json",
# r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF.json",
#     r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF.json",
#
#
#     r"./mrf_sequence_adjusted_760.json",
#     #"mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v5.json",
# #"mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v6.json",
# #"mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v7.json",
# #    "mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v2.json",
# #"mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v3.json",
# "mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v4.json",
# "mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v5.json",
# "mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_random_FA_v1.json",
# "mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_random_FA_v2.json",
# "mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_random_v6.json",
#     #"mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v3.json",
# #"mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v4.json"
#
#
#     #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_v2.json",
# #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_v3.json",
# #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_v6.json",
#
# ]
#
#
# recoveries=[4,3.9,3,3,3.9,4,4,4,4]
#
#
#
# fileseq_list=[
#     r"./mrf_sequence_adjusted.json",
#     r"./mrf_sequence_adjusted_760.json",
#
# "mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v5.json",
#     #"mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v3.json",
# #"mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v4.json"
#
#
#     #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_v2.json",
# #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_v3.json",
# #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_v6.json",
#
# ]
#
#
# recoveries=[4,3,4]
#
#
# df_results=pd.DataFrame(index=[f+"_"+str(recoveries[j]) for j,f in enumerate(fileseq_list)],columns=["Error rel wT1","Error abs FF","std wT1","std FF","TR"])
# min_TR_delay=1.87/1000
# l=None
# #plt.figure()
# labels=["MRF T1-FF 1400 Spokes","MRF T1-FF 760 Spokes","Optimized MRF T1-FF"]
# for j,fileseq_1 in enumerate(fileseq_list):
#     recovery=recoveries[j]
#     ind = fileseq_1 + "_" + str(recovery)
#
#     TR_list_1,FA_list_1,TE_list_1=load_sequence_file(fileseq_1,recovery,min_TR_delay)
#
#     s,s_w,s_f,keys=simulate_gen_eq_signal(TR_list_1,FA_list_1,TE_list_1,FFs,DFs,T1_w+dTs,300/1000,B1,T_2w=40/1000,T_2f=80/1000,amp=fat_amp,shift=fat_shift,sigma=sigma,noise_size=noise_size,group_size=group_size,return_fat_water=True,noise_type=noise_type)#,amp=np.array([1]),shift=np.array([-418]),sigma=None):
#
#     s_w=s_w.reshape(s_w.shape[0],-1).T
#     s_f=s_f.reshape(s_f.shape[0],-1).T
#     #keys=keys.reshape(-1,4)
#     #keys=[tuple(p) for p in keys]
#     s=s.reshape(s.shape[0],-1)
#
#     # plt.close("all")
#     # plt.figure()
#     # plt.plot(s_w[:len(DFs)].T)
#     # plt.figure()
#     # plt.plot(s_w[len(DFs):2*len(DFs)].T)
#
#     nb_signals=s.shape[-1]
#     mask=None
#     pca = True
#     threshold_pca_bc = 20
#
#
#     split=nb_signals+1
#     split=10
#     dict_optim_bc_cf =SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
#                                     threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
#                                     gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,return_matched_signals=True)
#
#
#     all_maps,matched_signals=dict_optim_bc_cf.search_patterns_test((s_w,s_f,keys),s)
#
#     if l is None:
#         l = np.random.choice(range(matched_signals.shape[-1]))
#
#     plt.figure()
#     plt.plot(matched_signals[:,l],label=labels[j])
#     plt.plot(s[:,l])
#
#     keys_all=list(product(keys,FFs))
#     keys_all=[(*rest, a) for rest,a in keys_all]
#     keys_all=np.array(keys_all)
#
#     key="wT1"
#     map=all_maps[0][0][key].reshape(-1,noise_size)
#     keys_all_current=np.array(list(keys_all[:,0])*noise_size).reshape(noise_size,-1).T
#     error=np.abs(map-keys_all_current)
#     error_wT1=np.mean(np.mean(error,axis=-1)/keys_all[:,0])
#     std_wT1=np.mean(np.std(error,axis=-1)/keys_all[:,0])
#
#
#     key="ff"
#     map=all_maps[0][0][key].reshape(-1,noise_size)
#     keys_all_current=np.array(list(keys_all[:,-1])*noise_size).reshape(noise_size,-1).T
#     error=np.abs(map-keys_all_current)
#     error_ff=np.mean(np.mean(error,axis=-1))
#     std_ff = np.mean(np.std(error,axis=-1))
#     df_results.loc[ind]=[error_wT1,error_ff,std_wT1,std_ff,np.sum(TR_list_1)]
# df_results
#








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
num=1
file="./data/KneePhantom/Phantom{}/paramMap_Control.mat".format(num)


m_=MapFromFile("Knee2D_Optim",file=file,rounding=True)
radial_traj=Radial(total_nspokes=spokes_count,npoint=npoint)

m_.buildParamMap()

fileseq_basis="./mrf_sequence_adjusted.json"

with open(fileseq_basis, "r") as file:
    seq_config_base = json.load(file)

def cost_function_simul_breaks_random_FA_KneePhantom(params):
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
    global m_
    global radial_traj
    global ntimesteps


    group_size = 8

    bound_max_FA=params[-1]
    params_for_curve=params[:-1]
    TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

    # with open(fileseq_basis,"r") as file:
    #     seq_config = json.load(file)
    #
    # seq_config_new = copy(seq_config_base)
    # seq_config_new["B1"] = list(np.array(FA_[1:]) * 180 / np.pi / 5)
    # seq_config_new["TR"] = list(np.array(TR_[1:]) * 10 ** 3)
    # seq_config_new["TE"] = list(np.array(TE_[1:]) * 10 ** 3)
    #
    # nrep = 2
    # rep = nrep - 1
    #
    # Treco = params[-2]*1000
    #
    # ##other options
    # seq_config_new["T_recovery"] = Treco
    # seq_config_new["nrep"] = nrep
    # seq_config_new["rep"] = rep
    #
    # seq = T1MRFSS(**seq_config_new)


    m_.build_ref_images_bloch(TR_,FA_,TE_)
    #images_series_1=copy(m_.images_series)
    # m_.build_ref_images(seq)
    # images_series_2=copy(m_.images_series)
    #
    # images_series_1=images_series_1[:,m_.mask>0]
    # images_series_2=images_series_2[:,m_.mask>0]
    #
    # j=np.random.choice(images_series_1.shape[-2])
    #
    # plt.figure()
    # plt.plot(np.real(images_series_1[:,j]))
    # plt.plot(np.real(images_series_2[:,j]))
    #
    # plt.figure()
    # plt.plot(np.imag(images_series_1[:, j]))
    # plt.plot(np.imag(images_series_2[:, j]))



    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
                                               return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T

    data = m_.generate_kdata(radial_traj, useGPU=useGPU)
    data = np.array(data)
    data = data.reshape(spokes_count, -1, npoint)
    b1_all_slices = np.ones(image_size)
    b1_all_slices = np.expand_dims(b1_all_slices, axis=0)
    volumes_all = simulate_radial_undersampled_images(data, radial_traj, image_size, density_adj=True, useGPU=useGPU,ntimesteps=ntimesteps)

    # keys=keys.reshape(-1,4)
    # keys=[tuple(p) for p in keys]
    #s = s.reshape(s.shape[0], -1)

    # plt.close("all")
    # plt.figure()
    # plt.plot(s_w[:len(DFs)].T)
    # plt.figure()
    # plt.plot(s_w[len(DFs):2*len(DFs)].T)

    nb_signals = s.shape[-1]
    mask = m_.mask
    pca = True
    threshold_pca_bc = 10

    split = 100
    dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True)

    all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), volumes_all)

    # j=np.random.choice(range(matched_signals.shape[-1]))
    # plt.figure()
    # plt.plot(matched_signals[:,j])
    # plt.plot(s[:,j])


    key = "wT1"
    map = all_maps[0][0][key][m_.paramMap["ff"]<0.7]*1000
    map_gt=m_.paramMap[key][m_.paramMap["ff"]<0.7]
    error = np.abs(map - map_gt)
    error_wT1 = np.mean(error/ map_gt)

    print("wT1 Cost : {}".format(error_wT1))

    std_wT1 = np.std(error/ map_gt)
    print("wT1 Std : {}".format(std_wT1))

    key = "ff"
    map = all_maps[0][0][key]
    map_gt = m_.paramMap[key]
    error = np.abs(map - map_gt)
    error_ff = np.mean(error)

    print("FF Cost : {}".format(error_ff))

    # num_breaks_TE=len(TE_breaks)
    FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
    # print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    print("Time Cost : {}".format(time_cost))
    #test_time=time_cost-TR_[-1]
    #print("Time Test : {}".format(test_time))

    result=lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_FA * FA_cost + lambda_time * time_cost+lambda_stdT1 * std_wT1


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

        with open("x_random_FA_US_log.pkl", "wb") as file:
            pickle.dump(x, file)

    def reset(self):
        self.cost_values = []
        self.it=1

#sigma2=0.02**2

spokes_count=760


min_TR_delay=1.87*10**-3
DFs=[-60,-30,0,30,60]
FFs=[0.]
B1s=[0.3,0.5,0.7,1]
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
lambda_stdT1=0
inversion=True
useGPU=False

log_cost=LogCost(cost_function_simul_breaks_random_FA_KneePhantom)

res=differential_evolution(cost_function_simul_breaks_random_FA_KneePhantom,bounds=bounds,callback=log_cost,constraints=constraints,maxiter=1000)#,constraints=constraints)

import pickle
with open("res_simul_random_FA_US.pkl","wb") as file:
    pickle.dump(res, file)

plt.figure()
plt.plot(log_cost.cost_values)

import pickle
with open("res_simul_random_FA_US.pkl","rb") as file:
    res=pickle.load(file)

cost_function_simul_breaks_random_FA(res.x)

params=res.x
bound_max_FA=params[-1]
params_for_curve=params[:-1]
TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

plt.figure()
plt.plot(FA_[1:])


plt.figure()
plt.plot(TE_[1:])
res.x
generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_random_v6.json","./mrf_dictconf_SimReco2_lightDFB1.json",FA_,TE_,4,1.87/1000,fileseq_basis="./mrf_sequence_adjusted.json")






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

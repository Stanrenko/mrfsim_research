
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
generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v1.json","./mrf_dictconf_SimReco2.json",FA_,TE_,3.95,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")

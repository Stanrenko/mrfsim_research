from utils_simu import *
from dictoptimizers import SimpleDictSearch

generate_epg_dico_T1MRFSS_from_sequence_file("mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF.json","mrf_dictconf_SimReco2.json",3)

TR_list,FA_list,TE_list=load_sequence_file("mrf_sequence_adjusted.json",3,1.87/1000)

generate_epg_dico_T1MRFSS("mrf_sequence_adjusted_1_87.json","mrf_dictconf_Dico2_Invivo.json",FA_list,TE_list,4,1.87/1000)




#https://cds.ismrm.org/protected/16MPresentations/videos/0429/index.html

with open("./mrf_dictconf_SimReco2_light.json") as f:
    dict_config = json.load(f)

fat_amp = np.array(dict_config["fat_amp"])
fat_shift = -np.array(dict_config["fat_cshift"])


T1_w=1
dTs=np.arange(-500,1000,100)*10**-3
#dTs=np.array([500,500,1000])
DFs=[-30,0,30]
#DFs=[-60,-30,0,30,60]
FFs=[0.,0.1,0.2,0.3,0.4,0.5]
#DFs=[-60,-30,0,30,60]
#FFs=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
recovery=0
sigma=0.04
noise_size=100
group_size=8
noise_type="Absolute"

recovery=0
fileseq_1=r"./mrf_sequence_adjusted_optimized_M0_local_optim_correl_smooth.json"
fileseq_1=r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter.json"
fileseq_1=r"./mrf_sequence_adjusted.json"
fileseq_1=r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_v2.json"
recovery=0
wT1_errors=[]
ff_errors=[]


fileseq_list=[
    r"./mrf_sequence_adjusted.json",
    r"./mrf_sequence_adjusted.json",
    r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim.json",
    r"./mrf_sequence_adjusted_optimized_M0_local_optim_correl.json",
    r"./mrf_sequence_adjusted_optimized_M0_local_optim_correl_smooth.json",
    r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter.json",
    r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_v2.json"
]

fileseq_list=[
    r"./mrf_sequence_adjusted.json",
    r"./mrf_sequence_adjusted.json",
    #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760.json",
    r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized.json",
    r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF.json",

    r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v2.json",
    r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v3.json"
    #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v2.json"
    #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF.json"

    #r"./mrf_sequence_adjusted_sp760_CRLB_Optim_v2.json",
    #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_CRLB_optim.json",
    #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_CRLB_optim_v2.json",
    #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_optim.json",
    #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_optim_FF.json",
    #r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_optim_FF_v2.json"
]


recoveries=[4]+[3]*(len(fileseq_list)-2)+[4]
df_results=pd.DataFrame(index=[f+"_"+str(recoveries[j]) for j,f in enumerate(fileseq_list)],columns=["Error rel wT1","Error abs FF","TR"])
min_TR_delay=1.94/1000

for j,fileseq_1 in enumerate(fileseq_list):
    recovery=recoveries[j]
    ind = fileseq_1 + "_" + str(recovery)

    TR_list_1,FA_list_1,TE_list_1=load_sequence_file(fileseq_1,recovery,min_TR_delay)

    s,s_w,s_f,keys=simulate_gen_eq_signal(TR_list_1,FA_list_1,TE_list_1,FFs,DFs,T1_w+dTs,300/1000,T_2w=40/1000,T_2f=80/1000,amp=fat_amp,shift=fat_shift,sigma=sigma,noise_size=noise_size,group_size=group_size,return_fat_water=True,noise_type=noise_type)#,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w=s_w.reshape(s_w.shape[0],-1).T
    s_f=s_f.reshape(s_f.shape[0],-1).T
    #keys=keys.reshape(-1,4)
    #keys=[tuple(p) for p in keys]
    s=s.reshape(s.shape[0],-1)

    # plt.close("all")
    # plt.figure()
    # plt.plot(s_w[:len(DFs)].T)
    # plt.figure()
    # plt.plot(s_w[len(DFs):2*len(DFs)].T)

    nb_signals=s.shape[-1]
    mask=None
    pca = True
    threshold_pca_bc = 20


    split=nb_signals+1
    dict_optim_bc_cf =SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                    threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
                                    gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,return_matched_signals=True)


    all_maps,matched_signals=dict_optim_bc_cf.search_patterns_test((s_w,s_f,keys),s)

    # j=np.random.choice(range(matched_signals.shape[-1]))
    # plt.figure()
    # plt.plot(matched_signals[:,j])
    # plt.plot(s[:,j])

    keys_all=list(product(keys,FFs))
    keys_all=[(*rest, a) for rest,a in keys_all]
    keys_all=np.array(keys_all)

    key="wT1"
    map=all_maps[0][0][key].reshape(-1,noise_size)
    keys_all_current=np.array(list(keys_all[:,0])*noise_size).reshape(noise_size,-1).T
    error=np.abs(map-keys_all_current)
    error_wT1=np.mean(np.mean(error,axis=-1)/keys_all[:,0])


    key="ff"
    map=all_maps[0][0][key].reshape(-1,noise_size)
    keys_all_current=np.array(list(keys_all[:,-1])*noise_size).reshape(noise_size,-1).T
    error=np.abs(map-keys_all_current)
    error_ff=np.mean(np.mean(error,axis=-1))
    df_results.loc[ind]=[error_wT1,error_ff,np.sum(TR_list_1)]

df_results

###Optimize FA schedule###############




from utils_simu import *
from dictoptimizers import SimpleDictSearch


#https://cds.ismrm.org/protected/16MPresentations/videos/0429/index.html

with open("./mrf_dictconf_SimReco2_light.json") as f:
    dict_config = json.load(f)

fat_amp = np.array(dict_config["fat_amp"])
fat_shift = -np.array(dict_config["fat_cshift"])


T1_w=1
dTs=np.arange(-500,1000,100)*10**-3
#dTs=np.array([500,500,1000])
DFs=[-30,0,30]
FFs=[0.,0.1,0.2,0.3,0.4,0.5]
recovery=0
sigma=0.02
noise_size=1000
group_size=8

recovery=0
fileseq_1=r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim.json"

TR_list, _, TE_list = load_sequence_file(fileseq_1, recovery)
num_sequence=10000

new_size=880

TE_new=np.zeros(new_size+1)
for i in range(new_size):
    TE_new[i+1]=TE_list[int(i*(len(TE_list)-1)/new_size)]

TR_new=np.zeros(new_size+1)
TR_new[0]=TR_list[0]
TR_new[1:]=np.max(TR_list[1:])

H=10
T=len(TR_new)-1

all_FA=[]
errors_wT1=[]
errors_FF=[]
for n in tqdm(range(num_sequence)):
    FA_list=generate_FA(T,H=H)
    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_new, FA_list, TE_new, FFs, DFs, T1_w + dTs, 300 / 1000,
                                               T_2w=40 / 1000, T_2f=80 / 1000, amp=fat_amp, shift=fat_shift,
                                               sigma=sigma, noise_size=noise_size, group_size=group_size,
                                               return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T
    s = s.reshape(s.shape[0], -1)

    nb_signals = s.shape[-1]
    mask = None
    pca = True
    threshold_pca_bc = 20

    split = nb_signals + 1
    dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True)

    all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), s)

    # j=np.random.choice(range(matched_signals.shape[-1]))
    # plt.figure()
    # plt.plot(matched_signals[:,j])
    # plt.plot(s[:,j])

    keys_all = list(product(keys, FFs))
    keys_all = [(*rest, a) for rest, a in keys_all]
    keys_all = np.array(keys_all)

    key = "wT1"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, 0]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_wT1 = np.mean(np.mean(error, axis=-1) / keys_all[:, 0])
    print(error_wT1)

    if error_wT1>0.3:
        continue

    errors_wT1.append(error_wT1)



    key = "ff"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, -1]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_ff = np.mean(np.mean(error, axis=-1))
    errors_FF.append(error_ff)

    all_FA.append(FA_list)


all_FA=np.array(all_FA)
errors_FF=np.array(errors_FF)
errors_wT1=np.array(errors_wT1)

np.save("all_FA.npy",all_FA)
np.save("errors_wT1.npy",all_FA)
np.save("errors_FF.npy",all_FA)


plt.figure()
plt.plot(np.sort(errors_wT1))

plt.figure()
plt.plot(np.sort(errors_FF))


ind_min=np.argsort(errors_wT1)[0]

print(errors_FF[ind_min])
print(errors_wT1[ind_min])

plt.figure()
plt.plot(all_FA[ind_min][1:]*180/np.pi)




from utils_simu import *
from dictoptimizers import SimpleDictSearch


#https://cds.ismrm.org/protected/16MPresentations/videos/0429/index.html

with open("./mrf_dictconf_SimReco2_light.json") as f:
    dict_config = json.load(f)

fat_amp = np.array(dict_config["fat_amp"])
fat_shift = -np.array(dict_config["fat_cshift"])


T1_w=1
dTs=np.arange(-500,1000,100)*10**-3
#dTs=np.array([500,500,1000])
DFs=[-30,0,30]
FFs=[0.,0.1,0.2,0.3,0.4,0.5]
#recovery=0
sigma=0.02
noise_size=1000
group_size=8

#recovery=4
fileseq_1=r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter.json"

TR_list, FA_list, TE_list = load_sequence_file(fileseq_1, 0)
#num_sequence=10000

sizes=np.arange(600,1401,80)
recoveries=[2,3,4]

#all_FA=[]
errors_wT1=[]
errors_FF=[]
for new_size in tqdm(sizes):
    for recovery in recoveries:
        #FA_list=generate_FA(T,H=H)
        TE_new = np.zeros(new_size + 1)
        TR_new = np.zeros(new_size + 1)
        FA_new=np.zeros(new_size + 1)
        for i in range(new_size):
            TE_new[i + 1] = TE_list[int((i+1) * (len(TE_list) - 1) / new_size)]
            TR_new[i + 1] = TR_list[int((i+1) * (len(TR_list) - 1) / new_size)]
            FA_new[i + 1] = FA_list[int((i+1) * (len(FA_list) - 1) / new_size)]

        TR_new[-1]=TR_new[-1]+recovery
        TR_new[0]=TR_list[0]
        FA_new[0]=np.pi

        s, s_w, s_f, keys = simulate_gen_eq_signal(TR_new, FA_new, TE_new, FFs, DFs, T1_w + dTs, 300 / 1000,
                                                   T_2w=40 / 1000, T_2f=80 / 1000, amp=fat_amp, shift=fat_shift,
                                                   sigma=sigma, noise_size=noise_size, group_size=group_size,
                                                   return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

        s_w = s_w.reshape(s_w.shape[0], -1).T
        s_f = s_f.reshape(s_f.shape[0], -1).T
        s = s.reshape(s.shape[0], -1)

        nb_signals = s.shape[-1]
        mask = None
        pca = True
        threshold_pca_bc = 20

        split = nb_signals + 1
        dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                            threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                            useGPU_simulation=False,
                                            gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                            return_matched_signals=True)

        all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), s)

        # j=np.random.choice(range(matched_signals.shape[-1]))
        # plt.figure()
        # plt.plot(matched_signals[:,j])
        # plt.plot(s[:,j])

        keys_all = list(product(keys, FFs))
        keys_all = [(*rest, a) for rest, a in keys_all]
        keys_all = np.array(keys_all)

        key = "wT1"
        map = all_maps[0][0][key].reshape(-1, noise_size)
        keys_all_current = np.array(list(keys_all[:, 0]) * noise_size).reshape(noise_size, -1).T
        error = np.abs(map - keys_all_current)
        error_wT1 = np.mean(np.mean(error, axis=-1) / keys_all[:, 0])
        print(error_wT1)

        #if error_wT1>0.3:
        #    continue

        errors_wT1.append(error_wT1)



        key = "ff"
        map = all_maps[0][0][key].reshape(-1, noise_size)
        keys_all_current = np.array(list(keys_all[:, -1]) * noise_size).reshape(noise_size, -1).T
        error = np.abs(map - keys_all_current)
        error_ff = np.mean(np.mean(error, axis=-1))
        errors_FF.append(error_ff)

        #all_FA.append(FA_list)


plt.close("all")
errors_wT1=np.array(errors_wT1)
errors_FF=np.array(errors_FF)

errors_wT1=errors_wT1.reshape(-1,len(recoveries))
errors_FF=errors_FF.reshape(-1,len(recoveries))

plt.figure()
plt.plot(sizes,errors_wT1[:,:])

plt.figure()
plt.plot(sizes,errors_wT1[:,-3:])

plt.figure()
plt.plot(errors_wT1.T)




plt.figure()
plt.plot(errors_FF[:,-2:])

np.save("{}_errors_wT1.npy".format(str.split(fileseq_1,".")[0]),errors_wT1)
np.save("{}_errors_FF.npy".format(str.split(fileseq_1,".")[0]),errors_FF)


new_size=sizes[-1]
recovery=4
TE_new = np.zeros(new_size + 1)
TR_new = np.zeros(new_size + 1)
FA_new=np.zeros(new_size + 1)
for i in range(new_size):
    TE_new[i + 1] = TE_list[int((i+1) * (len(TE_list) - 1) / new_size)]
    TR_new[i + 1] = TR_list[int((i+1) * (len(TR_list) - 1) / new_size)]
    FA_new[i + 1] = FA_list[int((i+1) * (len(FA_list) - 1) / new_size)]

TR_new[-1]=TR_new[-1]+recovery
TR_new[0]=TR_list[0]
FA_new[0]=np.pi

np.sum(TR_new)

plt.figure()
plt.plot(TR_new[:-1])

plt.figure()
plt.plot(FA_new[1:])

new_size = sizes[-1]
recovery =4
TE_new = np.zeros(new_size + 1)
TR_new = np.zeros(new_size + 1)
FA_new = np.zeros(new_size + 1)
for i in range(new_size):
    TE_new[i + 1] = TE_list[int((i + 1) * (len(TE_list) - 1) / new_size)]
    TR_new[i + 1] = TR_list[int((i + 1) * (len(TR_list) - 1) / new_size)]
    FA_new[i + 1] = FA_list[int((i + 1) * (len(FA_list) - 1) / new_size)]

#TR_new[-1] = TR_new[-1] + recovery
TR_new[0] = TR_list[0]
FA_new[0] = np.pi


np.sum(TR_new)

#write_seq_file(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_Reco3_sp{}.json".format(new_size),TR_new,FA_new,TE_new)
generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp{}.json".format(new_size),"./mrf_dictconf_SimReco2_light.json",TR_new,FA_new,TE_new,recovery)


plt.figure()
plt.plot(TR_new[:-1])

plt.figure()
plt.plot(FA_new[1:])


new_size = sizes[-1]
recovery = 1
TE_new = np.zeros(new_size + 1)
TR_new = np.zeros(new_size + 1)
FA_new = np.zeros(new_size + 1)
for i in range(new_size):
    TE_new[i + 1] = TE_list[int((i + 1) * (len(TE_list) - 1) / new_size)]
    TR_new[i + 1] = TR_list[int((i + 1) * (len(TR_list) - 1) / new_size)]
    FA_new[i + 1] = FA_list[int((i + 1) * (len(FA_list) - 1) / new_size)]

TR_new[-1] = TR_new[-1] + recovery
TR_new[0] = TR_list[0]
FA_new[0] = np.pi


np.sum(TR_new)

plt.figure()
plt.plot(TR_new[:-1])

plt.figure()
plt.plot(FA_new[1:])


########################################################################"

from utils_simu import *
from dictoptimizers import SimpleDictSearch

with open("./mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

fat_amp = np.array(dict_config["fat_amp"])
fat_shift = -np.array(dict_config["fat_cshift"])

def cost_function_simul(params):
    global spokes_count
    global min_TR_delay
    global recovery
    global TE_breaks
    global FA_breaks
    global DFs
    global FFs
    global T1s
    global lambda_FA
    global lambda_time
    global lambda_FF
    global fat_amp
    global fat_shift

    sigma = 0.03
    noise_size = 100
    group_size = 8

    list_deriv = ["ff", "wT1"]

    TR_, FA_, TE_ = convert_params_to_sequence(params, recovery, min_TR_delay, TE_breaks, FA_breaks, spokes_count)
    # print(FA_[:10])
    # print(params)

    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=sigma, noise_size=noise_size,
                                               group_size=group_size,
                                               return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T
    # keys=keys.reshape(-1,4)
    # keys=[tuple(p) for p in keys]
    s = s.reshape(s.shape[0], -1)

    # plt.close("all")
    # plt.figure()
    # plt.plot(s_w[:len(DFs)].T)
    # plt.figure()
    # plt.plot(s_w[len(DFs):2*len(DFs)].T)

    nb_signals = s.shape[-1]
    mask = None
    pca = True
    threshold_pca_bc = 20

    split = nb_signals + 1
    dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True)

    all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), s)

    # j=np.random.choice(range(matched_signals.shape[-1]))
    # plt.figure()
    # plt.plot(matched_signals[:,j])
    # plt.plot(s[:,j])

    keys_all = list(product(keys, FFs))
    keys_all = [(*rest, a) for rest, a in keys_all]
    keys_all = np.array(keys_all)

    key = "wT1"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, 0]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_wT1 = np.mean(np.mean(error, axis=-1) / keys_all[:, 0])

    print("wT1 Cost : {}".format(error_wT1))

    key = "ff"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, -1]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_ff = np.mean(np.mean(error, axis=-1))

    print("FF Cost : {}".format(error_ff))

    num_breaks_TE = len(TE_breaks)
    FA_cost = (np.linalg.norm(np.diff([0] + params[num_breaks_TE:])))
    print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    print("Time Cost : {}".format(time_cost))

    return error_wT1 + lambda_FF * error_ff + lambda_FA * FA_cost + lambda_time * time_cost


recovery=3
min_TR_delay=1.94/1000

TR_list,FA_list,TE_list=load_sequence_file("./mrf_sequence_adjusted.json",recovery,min_TR_delay)
TR_list,FA_list,TE_list=load_sequence_file("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF.json",recovery,min_TR_delay)
spokes_count=1400

plt.figure()
plt.plot(TE_list[1:])

plt.figure()
plt.plot(FA_list[1:])


TE_new = np.zeros(spokes_count + 1)
TR_new = np.zeros(spokes_count + 1)
FA_new=np.zeros(spokes_count + 1)
for i in range(spokes_count):
    TE_new[i + 1] = TE_list[int((i+1) * (len(TE_list) - 1) / spokes_count)]
    TR_new[i + 1] = TR_list[int((i+1) * (len(TR_list) - 1) / spokes_count)]
    FA_new[i + 1] = FA_list[int((i+1) * (len(FA_list) - 1) / spokes_count)]

if TE_new[1]==0:
    TE_new[1] =TE_new[2]

if FA_new[1] == 0:
    FA_new[1] = FA_new[2]

FA_breaks=[0]+list(np.argwhere(np.diff(np.array(FA_new[1:]))!=0).flatten())+[spokes_count]
TE_breaks=[0]+list(np.argwhere(np.diff(np.array(TE_new[1:]))!=0).flatten())+[spokes_count]

plt.close("all")

T1_w=1
dTs=np.arange(-500,1000,100)*10**-3
DFs=[-60,-30,0,30,60]
FFs=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
DFs=[-30,0,30]
FFs=[0.,0.1,0.2,0.3,0.4,0.5]
T1s=T1_w+dTs

params_0=convert_sequence_to_params(FA_list,TE_list)
bounds=[(0.0022,0.006)]*(len(TE_breaks)-1)+[(5*np.pi/180,70*np.pi/180)]*(len(FA_breaks)-1)


from scipy.optimize import differential_evolution
lambda_FA=0.
lambda_time=0.0
lambda_FF=2
res=differential_evolution(cost_function_simul,bounds)

print(cost_function_simul(res.x))
print(cost_function_simul(params_0))

TR_,FA_,TE_=convert_params_to_sequence(res.x,recovery,min_TR_delay,TE_breaks,FA_breaks,spokes_count)

plt.figure()
plt.plot(TE_[1:])

plt.figure()
plt.plot(FA_[1:])

generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_DE_Simu_FF.json","./mrf_dictconf_SimReco2_light.json",FA_,TE_,recovery,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")






from utils_simu import *
from dictoptimizers import SimpleDictSearch
import json

with open("./mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

fat_amp = np.array(dict_config["fat_amp"])
fat_shift = -np.array(dict_config["fat_cshift"])



TR_list,FA_list,TE_list=load_sequence_file("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized.json",3,1.94/1000)
spokes_count=680

TE_new = np.zeros(spokes_count + 1)
TR_new = np.zeros(spokes_count + 1)
FA_new=np.zeros(spokes_count + 1)
for i in range(spokes_count):
    TE_new[i + 1] = TE_list[int((i+1) * (len(TE_list) - 1) / spokes_count)]
    TR_new[i + 1] = TR_list[int((i+1) * (len(TR_list) - 1) / spokes_count)]
    FA_new[i + 1] = FA_list[int((i+1) * (len(FA_list) - 1) / spokes_count)]
FA_breaks=[0]+list(np.argwhere(np.diff(np.array(FA_new[1:]))!=0).flatten())+[spokes_count]
TE_breaks=[0]+list(np.argwhere(np.diff(np.array(TE_new[1:]))!=0).flatten())+[spokes_count]


def cost_function_simul_breaks(params):
    global spokes_count
    global min_TR_delay
    global recovery
    global num_breaks_TE
    global num_breaks_FA
    global DFs
    global FFs
    global T1s
    global lambda_FA
    global lambda_T1
    global lambda_time
    global lambda_FF
    global inversion

    sigma = 0.6
    noise_size = 100
    group_size = 8
    noise_type = "Relative"

    # print(params)

    list_deriv = ["ff", "wT1"]

    # print(params)
    # print(num_breaks_TE)
    # print(num_breaks_FA)
    TR_, FA_, TE_ = convert_params_to_sequence_breaks(params, min_TR_delay, num_breaks_TE, num_breaks_FA, spokes_count,
                                                      inversion)
    # print(FA_[:10])
    # print(params)

    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=sigma, noise_size=noise_size,
                                               noise_type=noise_type, group_size=group_size,
                                               return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T
    # keys=keys.reshape(-1,4)
    # keys=[tuple(p) for p in keys]
    s = s.reshape(s.shape[0], -1)

    # plt.close("all")
    # plt.figure()
    # plt.plot(s_w[:len(DFs)].T)
    # plt.figure()
    # plt.plot(s_w[len(DFs):2*len(DFs)].T)

    nb_signals = s.shape[-1]
    mask = None
    pca = True
    threshold_pca_bc = 20

    split = nb_signals + 1
    dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True)

    all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), s)

    # j=np.random.choice(range(matched_signals.shape[-1]))
    # plt.figure()
    # plt.plot(matched_signals[:,j])
    # plt.plot(s[:,j])

    keys_all = list(product(keys, FFs))
    keys_all = [(*rest, a) for rest, a in keys_all]
    keys_all = np.array(keys_all)

    key = "wT1"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, 0]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_wT1 = np.mean(np.mean(error, axis=-1) / keys_all[:, 0])

    print("wT1 Cost : {}".format(error_wT1))

    key = "ff"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, -1]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_ff = np.mean(np.mean(error, axis=-1))

    print("FF Cost : {}".format(error_ff))

    # num_breaks_TE=len(TE_breaks)
    FA_cost = (np.linalg.norm(np.diff([0] + params[num_breaks_TE + 1 + num_breaks_TE + num_breaks_FA:])))
    # print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    print("Time Cost : {}".format(time_cost))

    return lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_FA * FA_cost + lambda_time * time_cost


#sigma2=0.02**2
min_TR_delay=1.24*10**-3

T1_w=1
dTs=np.arange(-500,1000,100)*10**-3
DFs=[-60,-30,0,30,60]
FFs=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
DFs=[-30,0,30]
FFs=[0.,0.1,0.2,0.3,0.4,0.5]
T1s=T1_w+dTs

recovery=3

params_0,TE_breaks_,FA_breaks_=convert_sequence_to_params_breaks(FA_new,TE_new,3)
num_breaks_TE=len(TE_breaks_)-2
num_breaks_FA=len(FA_breaks_)-2

bounds=[(50,300)]*(num_breaks_TE)+[(50,300)]*(num_breaks_FA)+[(0.0022,0.006)]*(num_breaks_TE+1)+[(5*np.pi/180,70*np.pi/180)]*(num_breaks_FA+1)+[(0,4)]

from scipy.optimize import LinearConstraint
A_TEbreaks=np.zeros((1,len(params_0)))
A_TEbreaks[0,:num_breaks_TE]=1

A_FAbreaks=np.zeros((1,len(params_0)))
A_FAbreaks[0,num_breaks_TE:num_breaks_FA]=1

con1=LinearConstraint(A_TEbreaks,lb=-np.inf,ub=spokes_count-50)
con2=LinearConstraint(A_FAbreaks,lb=-np.inf,ub=spokes_count-50)
constraints=(con1,con2)

from scipy.optimize import differential_evolution
lambda_FA=0.
lambda_time=0.025
lambda_FF=2
inversion=False
lambda_T1=0
inversion=False
res=differential_evolution(cost_function_simul_breaks,bounds=bounds,constraints=constraints)

import pickle
with open("res_simul.pkl","wb") as file:
    pickle.dump(res, file)

import pickle
with open("res_simul.pkl","rb") as file:
    res=pickle.load(file)

cost_function_simul_breaks(res.x)

TR_,FA_,TE_=convert_params_to_sequence_breaks(res.x,min_TR_delay,num_breaks_TE,num_breaks_FA,spokes_count)

plt.figure()
plt.plot(FA_[1:])


plt.figure()
plt.plot(TE_[1:])

generate_epg_dico_T1MRFSS_NoInv(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_NoInv.json","./mrf_dictconf_SimReco2.json",FA_,TE_,0.,1.87/1000,fileseq_basis="./mrf_sequence_adjusted.json")
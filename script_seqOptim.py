from utils_simu import *

with open("./mrf_dictconf_SimReco2_light.json") as f:
    dict_config = json.load(f)

fat_amp = np.array(dict_config["fat_amp"])
fat_shift = -np.array(dict_config["fat_cshift"])


T1_w=1
dTs=np.arange(-500,1000,100)*10**-3
DFs=[-60,-40,0,30,60]
FFs=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
recovery=0
sigma2=0.02**2

fileseq_2=r"./mrf_sequence_adjusted.json"
TR_list_2,FA_list_2,TE_list_2=load_sequence_file(fileseq_2,recovery)

correl=get_correl_all(TR_list_2,FA_list_2,TE_list_2,DFs,FFs,T1_w+dTs,amp=fat_amp,shift=fat_shift)
metric_init=np.linalg.norm(np.eye(correl.shape[0])-correl)
print(metric_init)

TRtotref=np.sum(np.array(TR_list_2))
print(TRtotref)


recovery=0
fileseq_1=r"./mrf_sequence_adjusted_optimized_M0_local_optim_correl_smooth.json"
TR_list_1,FA_list_1,TE_list_1=load_sequence_file(fileseq_1,recovery)

s,dico_deriv=simulate_gen_eq_signal(TR_list_1,FA_list_1,TE_list_1,FFs,DFs,T1_w+dTs,300/1000,T_2w=40/1000,T_2f=80/1000,amp=fat_amp,shift=fat_shift,list_deriv=["wT1"])#,amp=np.array([1]),shift=np.array([-418]),sigma=None):
jac=convert_dico_to_jac(dico_deriv)
crlb=crlb_split(jac,sigma2=sigma2)
norm_init=np.linalg.norm(crlb[:,0])
print(norm_init)

magnitude_init=np.mean(calc_A_B_gen(FA_list_1,TR_list_1,T1_w+dTs)[1])

print(magnitude_init)



min_TR_delay=1.24*10**-3
smoothen=False

num_thresholds=2
num_thresholds_shuffles=100
additional_FA_thresholds=2
spokes_count=1400
min_thres_size=150
max_thres_size=700
min_thres_size_FA=150
max_thres_size_FA=600

thres_magnitude=0.5 #increase to constrain more
thres_crlb=0.7 #decrease to constrain more
thres_TRtot=1.1 #decrease to constrain more



all_FAs=np.array([5,10,15,20,30])*np.pi/180
FA_values_all=np.array(list(product(all_FAs,repeat=5)))


ind_thres_all = np.array(list(product(np.arange(min_thres_size,max_thres_size,50),repeat=2)))
ind_thres_all=np.cumsum(ind_thres_all,axis=-1)[(np.cumsum(ind_thres_all,axis=-1)[:,1]<=(spokes_count-min_thres_size))&(np.cumsum(ind_thres_all,axis=-1)[:,1]>=(spokes_count-max_thres_size))]


ind_FA_all = np.array(list(product(np.arange(min_thres_size_FA,max_thres_size_FA,50),repeat=2)))
ind_FA_all=np.cumsum(ind_FA_all,axis=-1)[(np.cumsum(ind_FA_all,axis=-1)[:,1]<=(spokes_count-min_thres_size_FA))&(np.cumsum(ind_FA_all,axis=-1)[:,1]>=(spokes_count-max_thres_size_FA))]

metric = []
crlbs = []
magnitudes = []
FA_all = []
TE_all = []

for ind in tqdm(ind_thres_all):
    ind = [1] + list(ind) + [spokes_count + 1]

    # print(ind)

    TE_ = np.zeros(spokes_count + 1)

    TE_values_all = list(np.unique(TE_list_2)[1:]) * int((num_thresholds + 1) / 2)

    for j in range(len(ind) - 1):
        TE_[ind[j]:ind[j + 1]] = TE_values_all[j]

    TE_ = list(TE_)

    TR_ = np.zeros(spokes_count + 1)
    TR_[0] = TR_list_2[0]
    TR_[1:] = np.array(TE_[1:]) + min_TR_delay
    TR_ = list(TR_)

    TRtot = np.sum(np.array(TR_))
    if (TRtotref * thres_TRtot) < TRtot:
        print("Sequence too long {} > {}".format(TRtot, TRtotref * thres_TRtot))
        continue

    ind_tot_all = []
    for ind_FA in ind_FA_all:
        ind_tot = np.sort(np.array(list(ind_FA) + list(ind)))
        # print(ind_tot)
        diff_FA = np.diff(np.array(ind_tot))
        if ((diff_FA > max_thres_size_FA).any() or (diff_FA < min_thres_size_FA).any()):
            continue
        else:
            ind_tot_all.append(ind_tot)

    ind_tot_all = np.unique(np.array(ind_tot_all), axis=0)

    for ind_tot in tqdm(ind_tot_all):
        for FA_values in (FA_values_all):
            FA_ = np.zeros(spokes_count + 1)
            FA_[0] = np.pi
            #ind_FA = (np.random.choice(np.arange(1, spokes_count), size=additional_FA_thresholds, replace=False))

            if np.min(np.diff(np.array(ind_tot))) < 50:
                continue
            for j in range(len(ind_tot) - 1):
                FA_[ind_tot[j]:ind_tot[j + 1]] = FA_values[j]

            if smoothen:
                FA_ = interp_FA(FA_, ind_tot)

            magnitude = np.mean(calc_A_B_gen(FA_, TR_, T1_w + dTs)[1])

            if magnitude < thres_magnitude * magnitude_init:
                #print("Magnitude too low {0:.3f} < {1:.3f}".format(magnitude, thres_magnitude * magnitude_init))
                continue

            s, dico_deriv = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1_w + dTs, 300 / 1000, T_2w=40 / 1000,
                                                   T_2f=80 / 1000, amp=fat_amp, shift=fat_shift, list_deriv=[
                    "wT1"])  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):
            jac = convert_dico_to_jac(dico_deriv)
            crlb = crlb_split(jac, sigma2=sigma2)
            norm = np.linalg.norm(crlb[:, 0])

            # print(norm)
            if norm > thres_crlb * norm_init:
                #print("CRLB norm too high {} > {}".format(norm, thres_crlb * norm_init))
                continue

            crlbs.append(norm)

            correl = get_correl_all(TR_, FA_, TE_, DFs, FFs, T1_w + dTs)

            FA_all.append(FA_)
            TE_all.append(TE_)

            metric.append(np.linalg.norm(np.eye(correl.shape[0]) - correl))
            # ind_all.append(ind)
            # ind_tot_all.append(ind_tot)
            magnitudes.append(magnitude)

metric = np.array(metric)
FA_all = np.array(FA_all)
TE_all = np.array(TE_all)
crlbs = np.array(crlbs)
magnitudes = np.array(magnitudes)


plt.figure(figsize=(15,10))
plt.title("Correl metric")
plt.plot(np.sort(metric))
plt.axhline(y=metric_init)


ind_min=np.argmin(metric)
plt.figure(figsize=(15,10))
plt.title("FA optim")
plt.plot(FA_all[ind_min][1:])

plt.figure(figsize=(15,10))
plt.title("TE optim")
plt.plot(TE_all[ind_min][1:])

print("Correl norm orig {0:.2f} - correl norm optimized {1:.2f}".format(metric_init,metric[ind_min]))
print("Magnitude orig {0:.2f} - Magnitude optimized {1:.2f}".format(magnitude_init,magnitudes[ind_min]))
print("CRLB orig {0:.2f} - CRLB optimized {1:.2f}".format(norm_init,crlbs[ind_min]))
print("TR total orig {0:.2f} - TR total optimized {1:.2f}".format(TRtotref,np.sum(np.array(TE_all[ind_min]+min_TR_delay))-min_TR_delay))


FA_=FA_all[ind_min]
TE_=TE_all[ind_min]
TR_=TE_
TR_[0]=8.32/1000
TR_[1:]=TR_[1:]+min_TR_delay

write_seq_file("./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_v2.json",list(TR_),list(FA_),list(TE_))
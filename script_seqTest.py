from utils_simu import *



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
B1=np.array([[[0.4,0.6,0.8,1]]])

T1s=T1_w+dTs

TR_list,FA_list,TE_list=load_sequence_file("mrf_sequence_adjusted.json",3,1.87/1000)


def calc_A_B_gen(FA_, TR_, T_1,B1):
    b_s = np.exp(-np.array(TR_).reshape(-1, 1) / T_1)
    b_s = np.expand_dims(b_s,axis=tuple(range(2,B1.ndim)))
    FA_init=FA_[0]
    FA_=np.array(FA_)
    FA_=np.expand_dims(FA_,axis=tuple(range(1,B1.ndim)))
    FA_=FA_*B1
    FA_[0]=FA_init
    a_s = np.cos(FA_)
    k_s = (a_s * b_s)[:len(TR_)]
    b_s_one_rep = b_s[:len(TR_)]
    A = np.prod(k_s, axis=0)
    cumprod_ks = np.cumprod(k_s[::-1], axis=0)[:-1]
    ones = np.ones((1,) + cumprod_ks.shape[1:])
    # B = np.sum(np.cumprod(np.array([1]+list(k_s))[:-1][::-1])[::-1]*(1-b_s_one_rep))
    B = np.sum(np.concatenate([ones, cumprod_ks])[::-1] * (1 - b_s_one_rep), axis=0)
    l = B / (1 - A)
    return A, l

calc_A_B_gen(FA_list,TR_list,T1s,B1)


def simulate_gen(u_0, TR_list, FA_list, nb_rep, T_1,B1):
    b_s = np.exp(-np.array(TR_list * nb_rep).reshape(-1, 1) / T_1)
    b_s = np.expand_dims(b_s, axis=tuple(range(2, B1.ndim)))
    FA_list=np.array(FA_list * nb_rep)
    FA_init = FA_list[0]
    FA_list = np.expand_dims(FA_list, axis=tuple(range(1, B1.ndim)))
    FA_list = FA_list * B1
    FA_list[0] = FA_init
    a_s = np.cos(FA_list)
    N = len(a_s)
    u_s = [u_0]
    u = u_0

    for j in range(N):
        u = b_s[j] * a_s[j] * u + (1 - b_s[j])
        u_s.append(u)
    u_s = np.array(u_s)
    return u_s


def simulate_gen_eq(TR_list, FA_list, T_1,B1):
    A, l = calc_A_B_gen(FA_list, TR_list, T_1,B1)
    u_i = simulate_gen(l, TR_list, FA_list, 1, T_1,B1)

    return u_i

test=simulate_gen_eq(TR_list,FA_list,T1s,B1)


DFs=np.array([[[[-30,0,30]]]])

def simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1, T_2,B1, amp=np.array([1]), shift=np.array([0])):
    u_i = simulate_gen_eq(TR_list, FA_list, T_1,B1)
    ax_expand = tuple(range(1, df.ndim))
    FA_init = FA_list[0]
    TEs = np.expand_dims(np.array(TE_list), axis=ax_expand)
    FAs = np.expand_dims(np.array(FA_list), axis=ax_expand)
    ax_expand_B1=tuple(range(B1.ndim, df.ndim))
    B1=np.expand_dims(B1,axis=ax_expand_B1)

    FAs = FAs * B1
    FAs[0] = FA_init

    chemical_shift = (np.exp(np.array(TE_list).reshape(-1, 1) * 2j * np.pi * shift.reshape(1, -1))) @ amp
    chemical_shift = np.expand_dims(chemical_shift, axis=ax_expand)
    E_2 = np.exp(TEs * (2j * np.pi * df - 1 / T_2)) * (chemical_shift)
    u = np.expand_dims(u_i[:-1], axis=-1)
    s_i = u * np.sin(np.array(FAs)) * E_2
    return s_i


s_i=simulate_gen_eq_transverse(TR_list, FA_list, TE_list, DFs, T1s, 40/1000,B1, amp=np.array([1]), shift=np.array([0]))

s_i=s_i.reshape(1401,-1)

plt.plot(s_i[:,10,:,1])
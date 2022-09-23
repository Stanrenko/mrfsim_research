import numpy as np
from numpy.random import normal
import pandas as pd
try:
    import matplotlib.pyplot as plt
except:
    pass
from itertools import combinations_with_replacement,product
import datetime
import json

from tqdm import tqdm
from mrfsim import *
from image_series import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
from movements import *
from dictoptimizers import *
import glob
from tqdm import tqdm
import pickle
from scipy.io import savemat

from copy import copy



class T1MRFSS:
    def __init__(self, FA, TI, TE, TR, B1,T_recovery,nrep,rep=None):
        print(T_recovery)
        print(nrep)
        """ build sequence """
        seqlen = len(TE)
        self.TR=TR
        self.inversion = epg.T(180, 0) # perfect inversion
        self.T_recovery=T_recovery
        self.nrep=nrep
        self.rep=rep
        seq=[]
        for r in range(nrep):
            curr_seq = [epg.Offset(TI)]
            for i in range(seqlen):
                echo = [
                    epg.T(FA * B1[i], 90),
                    epg.Wait(TE[i]),
                    epg.ADC,
                    epg.Wait(TR[i] - TE[i]),
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
        rep=self.rep
        for r in range(self.nrep):
            curr_seq=self._seq[r*self.len_rep:(r+1)*(self.len_rep)]
            curr_seq=[self.inversion, epg.modify(curr_seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
            seq.extend(curr_seq)
        #seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        if not(calc_deriv):
            result=np.asarray(epg.simulate(seq, **kwargs))
            if rep is None:#returning all repetitions
                return result
            else:#returning only the rep
                result = result.reshape((self.nrep, -1) + result.shape[1:])[rep]
                return result

        else:
            return epg.simulate(seq,calc_deriv=calc_deriv, **kwargs)


def crlb(J, H=None, W=None, sigma2=1, log=False):
    """ Cramer-Rao lower bound cost function """
    xp = np

    # J.shape: npoint x nparam x ...
    if np.iscomplexobj(J):
        J = xp.concatenate([J.real, J.imag], axis=-2)
    I = 1 / sigma2 * xp.einsum("...np,...nq->...pq", J, J)
    is_singular = np.linalg.cond(I) > 1e30
    I[is_singular] = np.nan
    lb = xp.linalg.inv(I)

    if W is not None:  # apply weights
        W = xp.asarray(W)[:, np.newaxis]
    else:
        W = 1

    cost = xp.trace(W * lb, axis1=-2, axis2=-1)
    if H is None:
        return cost if not log else np.log10(cost)

    # H is the derivative of J
    # H: npoint x nparam1 x nparam2 x ...
    if np.iscomplexobj(H):
        H = xp.concatenate([H.real, H.imag], axis=-3)
    grad = -2 * xp.einsum("...np,...pq,...nqr->...r", J, lb @ (W * lb), H)
    if not log:
        return cost, grad
    return np.log10(cost), grad / cost[..., np.newaxis] / np.log(10)


def crlb_split(J, W=None, sigma2=1):
    """ CRB for each variables in Jacobian """
    xp = np

    # J.shape: npoint x nparam x ...
    if np.iscomplexobj(J):
        J = xp.concatenate([J.real, J.imag], axis=-2)
    I = 1 / sigma2 * xp.einsum("...np,...nq->...pq", J, J)
    is_singular = np.linalg.cond(I) > 1e30
    I[is_singular] = np.nan
    lb = xp.linalg.inv(I)
    idiag = xp.arange(lb.shape[-1])
    crb = lb[..., idiag, idiag]
    if W is not None:  # apply weights
        crb *= xp.asarray(W)
    return crb


def simulate_gen(u_0, TR_list, FA_list, nb_rep, T_1):
    b_s = np.exp(-np.array(TR_list * nb_rep).reshape(-1, 1) / T_1)
    a_s = np.cos(np.array(FA_list * nb_rep).reshape(-1, 1))
    N = len(a_s)
    u_s = [u_0]
    u = u_0

    for j in range(N):
        u = b_s[j] * a_s[j] * u + (1 - b_s[j])
        u_s.append(u)
    u_s = np.array(u_s)
    return u_s


def calc_A_B_gen(FA_, TR_, T_1):
    b_s = np.exp(-np.array(TR_).reshape(-1, 1) / T_1)
    a_s = np.cos(np.array(FA_).reshape(-1, 1))
    k_s = (a_s * b_s)[:len(TR_)]
    b_s_one_rep = b_s[:len(TR_)]
    A = np.prod(k_s, axis=0)
    cumprod_ks = np.cumprod(k_s[::-1], axis=0)[:-1]
    ones = np.ones((1,) + cumprod_ks.shape[1:])
    # B = np.sum(np.cumprod(np.array([1]+list(k_s))[:-1][::-1])[::-1]*(1-b_s_one_rep))
    B = np.sum(np.concatenate([ones, cumprod_ks])[::-1] * (1 - b_s_one_rep), axis=0)
    l = B / (1 - A)
    return A, l


def simulate_gen_eq(TR_list, FA_list, T_1):
    A, l = calc_A_B_gen(FA_list, TR_list, T_1)
    u_i = simulate_gen(l, TR_list, FA_list, 1, T_1)

    return u_i


def simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1, T_2, amp=np.array([1]), shift=np.array([0])):
    u_i = simulate_gen_eq(TR_list, FA_list, T_1)
    ax_expand = tuple(range(1, df.ndim))
    TEs = np.expand_dims(np.array(TE_list), axis=ax_expand)
    FAs = np.expand_dims(np.array(FA_list), axis=ax_expand)
    chemical_shift = (np.exp(np.array(TE_list).reshape(-1, 1) * 2j * np.pi * shift.reshape(1, -1))) @ amp
    chemical_shift = np.expand_dims(chemical_shift, axis=ax_expand)
    E_2 = np.exp(TEs * (2j * np.pi * df - 1 / T_2)) * (chemical_shift)
    u = np.expand_dims(u_i[:-1], axis=-1)
    s_i = u * np.sin(np.array(FAs)) * E_2
    return s_i


def simulate_gen_eq_signal(TR_list, FA_list, TE_list, FF, df, T_1w, T_1f, T_2w=40 / 1000, T_2f=80 / 1000,
                           amp=np.array([1]), shift=np.array([-418]), sigma=None, list_deriv=None,noise_size=None,noise_type="Absolute",group_size=None,return_fat_water=False):
    T_1w = np.array(T_1w)
    T_1f = np.array(T_1f)
    df = np.array(df)
    FF = np.array(FF)

    if (np.array(T_1w).shape == ()):
        T_1w = np.array([T_1w])
    if (np.array(T_1f).shape == ()):
        T_1f = np.array([T_1f])
    if (np.array(df).shape == ()):
        df = np.array([df])
    if (np.array(FF).shape == ()):
        FF = np.array([FF])

    if not (T_1w.shape == (1,)):
        T_1w = np.squeeze(T_1w)
    if not (T_1f.shape == (1,)):
        T_1f = np.squeeze(T_1f)
    if not (df.shape == (1,)):
        df = np.squeeze(df)
    if not (FF.shape == (1,)):
        FF = np.squeeze(FF)

    keys = list(product(list(T_1w), list(T_1f), [1], list(df)))

    #keys=np.array(keys).reshape(len(T_1w),len(T_1f),1,len(df),4)

    T_1w = np.expand_dims(T_1w, axis=0)
    T_1f = np.expand_dims(T_1f, axis=0)
    df = np.expand_dims(df, axis=(0, 1))
    FF = np.expand_dims(FF, axis=(0, 1, 2))


    s_iw = simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1w, T_2w)[1:]
    s_iw = np.expand_dims(s_iw, axis=(2, -1))
    s_if = simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1f, T_2f, amp, shift)[1:]
    s_if = np.expand_dims(s_if, axis=(1, -1))
    s_iw, s_if = np.broadcast_arrays(s_iw, s_if)

    if group_size is not None:
        s_iw=np.array([np.mean(gp, axis=0) for gp in groupby(s_iw, group_size)])
        s_if = np.array([np.mean(gp, axis=0) for gp in groupby(s_if, group_size)])
        print(s_iw.shape)
        print(s_if.shape)


    s_i = FF * s_if + (1 - FF) * s_iw



    if sigma is not None:
        if noise_size is None:
            e_i = np.random.normal(size=s_i.shape) + 1j * np.random.normal(size=s_i.shape)
            # print(e_i.shape)
            # e_i*=np.abs(np.mean(s_i))/np.abs(e_i)/snr
            # e_i*=np.abs(s_i)/np.abs(e_i)/sigma
            if noise_type=="Absolute":
                e_i *= sigma
            elif noise_type=="Relative":
                e_i*=sigma*np.abs(s_i)
            else:
                raise ValueError("Unknown noise_type")

            s_i += e_i
        else:
            e_i = np.random.normal(size=s_i.shape+(noise_size,)) + 1j * np.random.normal(size=s_i.shape+(noise_size,))
            s_i = np.expand_dims(s_i, axis=-1)
            if noise_type == "Absolute":
                e_i *= sigma
            elif noise_type == "Relative":
                e_i *= sigma * np.abs(s_i)
            else:
                raise ValueError("Unknown noise_type")

            s_i=s_i+e_i

    if list_deriv is None:
        if return_fat_water:
            return s_i,s_iw,s_if,keys
        else:
            return s_i

    else:
        dico_calc_deriv = {}
        if "ff" in list_deriv:
            ds_ff = s_if - s_iw
            ds_ff = ds_ff[1:]
            reps = tuple(np.ones(s_i.ndim - 1).astype(int)) + (FF.shape[-1],)
            # print(reps)
            ds_ff = np.tile(ds_ff, reps)
            ds_ff = np.moveaxis(ds_ff, 0, -1)
            # print(ds_ff.shape)
            dico_calc_deriv["ff"] = ds_ff

        if "wT1" in list_deriv:
            dT1 = 10 ** -3
            s_iw_dT1 = simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1w + dT1, T_2w)[1:]
            s_iw_dT1 = np.expand_dims(s_iw_dT1, axis=(2, -1))
            ds_T1 = (1 - FF) * (s_iw_dT1 - s_iw) / dT1
            ds_T1 = ds_T1[1:]
            ds_T1 = np.moveaxis(ds_T1, 0, -1)
            dico_calc_deriv["wT1"] = ds_T1

        return s_i, dico_calc_deriv


def convert_dico_to_jac(dico_deriv):
    jac = np.stack(list(dico_deriv.values()), axis=-1)
    return jac


def get_correl_all(TR_, FA_, TE_, DFs, FFs, Ts, sigma=None, T_1f=300 / 1000, T_2w=40 / 1000, T_2f=80 / 1000,
                   amp=np.array([1]), shift=np.array([-418])):
    orig_signal = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, Ts, T_1f, T_2w, T_2f, amp, shift)
    orig_signal = orig_signal.reshape(orig_signal.shape[0], -1).T

    if sigma is not None:
        e_i_0 = np.random.normal(size=orig_signal.shape) + 1j * np.random.normal(size=orig_signal.shape)
        # print(e_i.shape)
        # e_i*=np.abs(np.mean(s_i))/np.abs(e_i)/snr
        # e_i*=np.abs(s_i)/np.abs(e_i)/sigma
        e_i_0 *= sigma / np.abs(e_i_0)
        signal = orig_signal + e_i_0

    else:
        signal = orig_signal

    cov_signal = np.abs(signal @ signal.conj().T)
    inverse_std_signal = np.diag(np.sqrt(1 / np.diag(cov_signal)))

    if sigma is not None:
        e_i = np.random.normal(size=orig_signal.shape) + 1j * np.random.normal(size=orig_signal.shape)
        # print(e_i.shape)
        # e_i*=np.abs(np.mean(s_i))/np.abs(e_i)/snr
        # e_i*=np.abs(s_i)/np.abs(e_i)/sigma
        e_i *= sigma / np.abs(e_i)
        signal_1 = orig_signal + e_i

        cov_signal_1 = np.abs(signal @ signal_1.conj().T)
        inverse_std_signal_1 = np.diag(np.sqrt(1 / np.diag(cov_signal_1)))



    else:
        cov_signal_1 = cov_signal
        inverse_std_signal_1 = inverse_std_signal

    corr_signal = inverse_std_signal @ cov_signal_1 @ inverse_std_signal_1

    return corr_signal


def load_sequence_file(fileseq,recovery,min_TR_delay):
    with open(fileseq, "r") as file:
        seq_config = json.load(file)

    TI = seq_config["TI"] * 10 ** -3
    TE = list(np.array(seq_config["TE"]) * 10 ** -3)
    TR = list(np.array(TE)+min_TR_delay)
    FA = seq_config["FA"]
    B1 = seq_config["B1"]
    # B1[:600]=[1.5]*600
    B1 = list(1 * np.array(B1))


    TR[-1] = TR[-1] + recovery

    TR_list = [TI] + TR
    TE_list = [0] + TE
    FA_list = [np.pi] + list(np.array(B1) * FA * np.pi / 180)
    return TR_list,FA_list,TE_list


def write_seq_file(fileseq,FA_list,TE_list,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json"):
    with open(fileseq_basis,"r") as file:
        seq_config = json.load(file)

    seq_config_new = seq_config
    seq_config_new["B1"] = list(np.array(FA_list[1:]) * 180 / np.pi / 5)
    seq_config_new["TR"] = list((np.array(TE_list[1:])+min_TR_delay) * 10 ** 3)
    seq_config_new["TE"] = list(np.array(TE_list[1:]) * 10 ** 3)


    with open(fileseq, "w") as file:
        json.dump(seq_config_new, file)



def interp_FA(FA_list, indices, minFA=5 * np.pi / 180):
    FA_ = np.array(FA_list)
    indices[0] = np.maximum(1, indices[0])
    for j, ind in enumerate(indices[:-1]):
        # print(indices[j])
        angle = FA_list[int((indices[j] + indices[j + 1]) / 2)]
        alpha = angle / ((indices[j + 1] - indices[j]) / 2) ** 2
        i = np.arange(indices[j], indices[j + 1])
        FA_[indices[j]:indices[j + 1]] = np.maximum(alpha * (i - indices[j]) * (indices[j + 1] - i), minFA)

    return list(FA_)


def generate_FA(T, H=10):
    FA_bound_min = np.random.choice(np.arange(5, 16)) * np.pi / 180
    FA_bound_max = np.random.choice(np.arange(30, 60)) * np.pi / 180

    rho = (np.random.uniform(size=H) * np.logspace(-0.5, -2.5, H)).reshape(-1, 1)
    phi = (np.random.uniform(size=H) * 2 * np.pi).reshape(-1, 1)

    t = np.arange(0, 2 * np.pi, 2 * np.pi / T).reshape(1, -1)

    FA_traj = np.sum(rho * np.sin(np.arange(1, H + 1).reshape(-1, 1) @ t + phi), axis=0)
    FA_traj_min = np.min(FA_traj)
    FA_traj_max = np.max(FA_traj)

    FA_traj = (FA_traj - FA_traj_min) / (FA_traj_max - FA_traj_min) * (FA_bound_max - FA_bound_min) + FA_bound_min

    FA_traj = [np.pi] + list(FA_traj)

    return FA_traj


def generate_epg_dico_T1MRFSS(fileseq,filedictconf,FA_list,TE_list,recovery,min_TR_delay,rep=2,overwrite=True,sim_mode="mean",fileseq_basis="./mrf_sequence_adjusted.json"):
    print("Generating sequence file {}".format(fileseq))
    write_seq_file(fileseq,FA_list,TE_list,min_TR_delay,fileseq_basis=fileseq_basis)
    generate_epg_dico_T1MRFSS_from_sequence_file(fileseq,filedictconf,recovery,rep,overwrite,sim_mode)


def generate_epg_dico_T1MRFSS_from_sequence_file(fileseq,filedictconf,recovery,rep=2,overwrite=True,sim_mode="mean"):
    prefix_dico=str.split(filedictconf,".json")[0]
    suffix_seq=str.split(fileseq,"_sequence")[1]
    suffix_seq=str.split(suffix_seq,".json")[0]
    dictfile=prefix_dico+suffix_seq+"_reco{}.dict".format(str(recovery))

    print("Generating dictionary {} from {}".format(dictfile,fileseq))

    with open(fileseq) as f:
        sequence_config = json.load(f)

    with open(filedictconf) as f:
        dict_config = json.load(f)


    # generate signals
    wT1 = dict_config["water_T1"]
    fT1 = dict_config["fat_T1"]
    wT2 = dict_config["water_T2"]
    fT2 = dict_config["fat_T2"]
    att = dict_config["B1_att"]
    df = dict_config["delta_freqs"]
    df = [- value / 1000 for value in df]  # temp
    # df = np.linspace(-0.1, 0.1, 101)

    TR_total = np.sum(sequence_config["TR"])
    print(TR_total)

    sequence_config["T_recovery"] = recovery*1000
    sequence_config["nrep"] = rep

    seq = T1MRFSS(**sequence_config)


    fat_amp = dict_config["fat_amp"]
    fat_cs = dict_config["fat_cshift"]
    fat_cs = [- value / 1000 for value in fat_cs]  # temp

    # other options

    window = dict_config["window_size"]

    # water
    printer("Generate water signals.")
    water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])
    water = water.reshape((rep, -1) + water.shape[1:])[-1]

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
    fat = fat.reshape((rep, -1) + fat.shape[1:])[-1]

    if sim_mode == "mean":
        fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
    elif sim_mode == "mid_point":
        fat = fat[(int(window / 2) - 1):-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    water = np.array(water)
    fat = np.array(fat)
    # join water and fat
    printer("Build dictionary.")
    keys = list(itertools.product(wT1, fT1, att, df))
    values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
    values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

    printer("Save dictionary.")
    mrfdict = dictsearch.Dictionary(keys, values)
    mrfdict.save(dictfile, overwrite=overwrite)

def convert_sequence_to_params(FA_, TE_):
    FA_breaks_ = list(np.argwhere(np.diff(np.array(FA_[1:])) != 0).flatten()) + [-1]
    TE_breaks_ = list(np.argwhere(np.diff(np.array(TE_[1:])) != 0).flatten()) + [-1]
    # print(FA_breaks)
    # print(TE_breaks)
    FA_values = [FA_[1:][i] for i in FA_breaks_]
    TE_values = [TE_[1:][i] for i in TE_breaks_]
    params_ = list(TE_values) + list(FA_values)

    return params_


def convert_params_to_sequence(params, recovery, min_TR_delay, TE_breaks, FA_breaks, spokes_count):
    num_breaks_TE = len(TE_breaks)
    TE_ = np.zeros(spokes_count + 1)
    for j in range(num_breaks_TE - 1):
        TE_[(TE_breaks[j] + 1):(TE_breaks[j + 1] + 1)] = params[j]

    num_breaks_FA = len(FA_breaks)
    FA_ = np.zeros(spokes_count + 1)
    for j in range(num_breaks_FA - 1):
        FA_[(FA_breaks[j] + 1):(FA_breaks[j + 1] + 1)] = params[j + num_breaks_TE - 1]

    FA_[0] = np.pi

    TR_ = np.zeros(spokes_count + 1)
    TR_[0] = 8.32 / 1000
    TR_[1:] = np.array(TE_[1:]) + min_TR_delay
    TR_[-1] = TR_[-1] + recovery
    TR_ = list(TR_)
    return TR_, FA_, TE_

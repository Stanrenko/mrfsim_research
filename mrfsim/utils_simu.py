import numpy as np
import json
try:
    import matplotlib.pyplot as plt
except:
    pass
from itertools import product

from epgpy.sequence import Sequence, Variable, operators
import epgpy as epg

#from image_series import
from mrfsim.utils_mrf import *
from mrfsim.movements import *


#from dictoptimizers import *

class T1MRF_generic:
    def __init__(self, SEQ_CONFIG):
        """ build sequence """
        self.seqlen = len(SEQ_CONFIG['TE'])
        self.READOUT = SEQ_CONFIG['readout']
        self.TI = SEQ_CONFIG['TI']
        self.TE = SEQ_CONFIG['TE']
        self.FA = SEQ_CONFIG['FA']*SEQ_CONFIG['B1']
        TR = [i + SEQ_CONFIG['dTR'] for i in self.TE]
        self.TR = TR
        self.inversion = epg.T(180, 0) 
        
        self.T_recovery=SEQ_CONFIG['T_recovery']
        # self.nrep=nrep
        # self.rep=rep
        seq=[]
        if SEQ_CONFIG['readout'] == 'FLASH_ideal': 
            self.spl = epg.SPOILER
            self.PHI = [50 * i * (i + 1) / 2 for i in range(len(self.TE))]
        elif SEQ_CONFIG['readout'] == 'FLASH': 
            self.spl = epg.S(1)
            self.PHI = [50 * i * (i + 1) / 2 for i in range(len(self.TE))]
        elif SEQ_CONFIG['readout'] == 'FISP': 
            self.spl = epg.S(1)
            self.PHI = np.array([0]*len(self.TE))
        elif SEQ_CONFIG['readout'] == 'pSSFP': 
            self.spl = epg.S(1)
            self.PHI = [1 * i * (i + 1) / 2 for i in range(len(self.TE))]
        elif SEQ_CONFIG['readout'] == 'pSSFP_generic': 
            self.spl = epg.S(1)
            self.PHI = [SEQ_CONFIG['PHI'][i] * i * (i + 1) / 2 for i in range(len(self.TE))]
        elif SEQ_CONFIG['readout'] == 'pSSFP_2': 
            self.spl = epg.S(1)
            self.PHI = np.array([1 * i * (i + 1) / 2 for i in range(int(np.round(len(self.TE)/2)))] + [(-1) * i * (i + 1) / 2 for i in range(int(np.round(len(self.TE)/2)))])            
        elif SEQ_CONFIG['readout'] == 'TrueFISP':
            self.spl = epg.NULL
            self.PHI = np.array([0, 180] * (len(self.TE) // 2) + [0] * (len(self.TE) % 2)) 

    def __call__(self, T1, T2, g, att, cs, frac, **kwargs):
        """ simulate sequence """
        self.init = [epg.PD(frac[j]) for j in range(len(frac))]
        self.rf = [epg.T(self.FA[i]*att, self.PHI[i]) for i in range(self.seqlen)] 
        self.adc = [epg.Adc(phase=-self.rf[i].phi) for i in range(self.seqlen)]
        self.rlx0 = epg.E(self.TI, T1, T2, duration=True) # inversion delay
        self.rlx1 = [[epg.E(self.TE[i], T1, T2, cs[j] + g, duration=True) for i in range(len(self.FA))]  for j in range(len(cs))] 
        self.rlx2 = [[epg.E(self.TR[i] - self.TE[i], T1, T2, cs[j] + g, duration=True) for i in range(len(self.FA))] for j in range(len(cs))] 
        self.rlx3 = epg.E(self.T_recovery, T1, T2, duration=True) # inversion delay
                
        seq = [[self.init[j]] + [self.inversion] + [self.rlx0] + [[self.rf[i], self.rlx1[j][i], self.adc[i], self.rlx2[j][i], self.spl] for i in range(self.seqlen)] for j in range(len(cs))] 
  
        result=np.asarray(epg.simulate(seq, disp=True, max_nstate=30))
        
        result = np.reshape(result, [len(cs),len(self.FA),*np.shape(result)[1:]])
        result = np.sum(np.asarray(frac)[:, np.newaxis, np.newaxis] * result , axis=0)
        
        return result

class T1MRFSS:
    def __init__(self, FA, TI, TE, TR, B1,T_recovery,nrep,rep=None):
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

    def __call__(self, T1, T2, g, att,**kwargs):
        """ simulate sequence """
        seq=[]
        rep=self.rep
        for r in range(self.nrep):
            curr_seq=self._seq[r*self.len_rep:(r+1)*(self.len_rep)]
            curr_seq=[self.inversion, epg.modify(curr_seq, T1=T1, T2=T2, att=att, g=g)]
            seq.extend(curr_seq)
        #seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        
        result=np.asarray(epg.simulate(seq, **kwargs))
        if rep is not None:
            result = result.reshape((self.nrep, -1) + result.shape[1:])[rep]
        return result


class T1MRFSS_ImperfectInv:
    def __init__(self, FA, TI, TE, TR, B1,T_recovery,nrep,rep=None):
        """ build sequence """
        seqlen = len(TE)
        self.TR=TR
        # self.inversion = epg.T(180, 0) # perfect inversion
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

    def __call__(self, T1, T2, g, att,att_inv,**kwargs):
        """ simulate sequence """
        seq=[]
        rep=self.rep
        for r in range(self.nrep):
            curr_seq=self._seq[r*self.len_rep:(r+1)*(self.len_rep)]
            curr_seq=[epg.T(180*att_inv, 0), epg.modify(curr_seq, T1=T1, T2=T2, att=att, g=g)]
            seq.extend(curr_seq)
        #seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        
        result=np.asarray(epg.simulate(seq, **kwargs))
        if rep is not None:
            result = result.reshape((self.nrep, -1) + result.shape[1:])[rep]
        return result

def modifier_inv(op, **kwargs):
    """default modifier to handle 'T1', 'T2', 'g' and 'att' keywords
    TODO: handle differential operators (options gradients and hessian)
    """
    # print(op)
    # if isinstance(op, operators.T):
    #     print(op.alpha)
    
    if isinstance(op, operators.T) and ((type(op.alpha)==int) or (type(op.alpha)==float)) and (op.alpha == 180):
        # add B1 attenuation
        att = kwargs.get("att")
        if att is None or np.allclose(att, 1):
            pass  # nothing to do
        else:
            # update T operator
            op = operators.T(op.alpha * att, op.phi, name=op.name, duration=op.duration)
            op.name += "#"
     
    return op



def t1_mfr_seq(SEQ_CONFIG):
    att = Variable("att")
    T1 = Variable("T1")
    T2 = Variable("T2")
    g = Variable("g")
    cs = Variable("cs")
    frac = Variable("frac")

    seqlen = len(SEQ_CONFIG['TE'])
    READOUT = SEQ_CONFIG['readout']
    TI = SEQ_CONFIG['TI']
    TE = SEQ_CONFIG['TE']
    FA = SEQ_CONFIG['FA']*SEQ_CONFIG['B1']
    TR = [i + SEQ_CONFIG['dTR'] for i in TE]
    T_recovery=SEQ_CONFIG['T_recovery']



    if SEQ_CONFIG['readout'] == 'FLASH_ideal': 
        spl = operators.SPOILER
        PHI = [50 * i * (i + 1) / 2 for i in range(len(TE))]
    elif SEQ_CONFIG['readout'] == 'FLASH': 
        spl = operators.S(1)
        PHI = [50 * i * (i + 1) / 2 for i in range(len(TE))]
    elif SEQ_CONFIG['readout'] == 'FISP': 
        spl = operators.S(1)
        PHI = np.array([0]*len(TE))
    elif SEQ_CONFIG['readout'] == 'pSSFP': 
        spl = operators.S(1)
        PHI = [1 * i * (i + 1) / 2 for i in range(len(TE))]
    elif SEQ_CONFIG['readout'] == 'pSSFP_2': 
        spl = operators.S(1)
        PHI = np.array([1 * i * (i + 1) / 2 for i in range(int(np.round(len(TE)/2)))] + [(-1) * i * (i + 1) / 2 for i in range(int(np.round(len(TE)/2)))])
    elif SEQ_CONFIG['readout'] == 'pSSFP_generic': 
        spl = operators.S(1)
        PHI = [SEQ_CONFIG['PHI'][i] * i * (i + 1) / 2 for i in range(len(TE))]
    elif SEQ_CONFIG['readout'] == 'TrueFISP':
        spl = operators.NULL
        PHI = np.array([0, 180] * (len(TE) // 2) + [0] * (len(TE) % 2)) 

    init = operators.PD(frac)
    inversion = operators.T(180, 0)
    rf = [operators.T(FA[i]*att, PHI[i]) for i in range(seqlen)] 
    adc = [operators.Adc(phase=-rf[i].phi, reduce=1) for i in range(seqlen)]
    # adc = [operators.Adc(phase=-rf[i].phi) for i in range(seqlen)]
    rlx0 = operators.E(TI, T1, T2, duration=True) 
    rlx1 = [operators.E(TE[i], T1, T2, cs + g, duration=True) for i in range(len(FA))] 
    rlx2 = [operators.E(TR[i] - TE[i], T1, T2, cs + g, duration=True) for i in range(len(FA))]
    # rlx3 = operators.E(T_recovery, T1, T2, duration=True)
            
    seq = Sequence([init] + [inversion] + [rlx0] + [[rf[i], rlx1[i], adc[i], rlx2[i], spl] for i in range(seqlen)])
    
    return seq

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

#
# def simulate_gen(u_0, TR_list, FA_list, nb_rep, T_1):
#     b_s = np.exp(-np.array(TR_list * nb_rep).reshape(-1, 1) / T_1)
#     a_s = np.cos(np.array(FA_list * nb_rep).reshape(-1, 1))
#     N = len(a_s)
#     u_s = [u_0]
#     u = u_0
#
#     for j in range(N):
#         u = b_s[j] * a_s[j] * u + (1 - b_s[j])
#         u_s.append(u)
#     u_s = np.array(u_s)
#     return u_s
#
#
# def calc_A_B_gen(FA_, TR_, T_1):
#     b_s = np.exp(-np.array(TR_).reshape(-1, 1) / T_1)
#     a_s = np.cos(np.array(FA_).reshape(-1, 1))
#     k_s = (a_s * b_s)[:len(TR_)]
#     b_s_one_rep = b_s[:len(TR_)]
#     A = np.prod(k_s, axis=0)
#     cumprod_ks = np.cumprod(k_s[::-1], axis=0)[:-1]
#     ones = np.ones((1,) + cumprod_ks.shape[1:])
#     # B = np.sum(np.cumprod(np.array([1]+list(k_s))[:-1][::-1])[::-1]*(1-b_s_one_rep))
#     B = np.sum(np.concatenate([ones, cumprod_ks])[::-1] * (1 - b_s_one_rep), axis=0)
#     l = B / (1 - A)
#     return A, l
#
#
# def simulate_gen_eq(TR_list, FA_list, T_1):
#     A, l = calc_A_B_gen(FA_list, TR_list, T_1)
#     u_i = simulate_gen(l, TR_list, FA_list, 1, T_1)
#
#     return u_i




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

# def simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1, T_2, amp=np.array([1]), shift=np.array([0])):
#     u_i = simulate_gen_eq(TR_list, FA_list, T_1)
#     ax_expand = tuple(range(1, df.ndim))
#     TEs = np.expand_dims(np.array(TE_list), axis=ax_expand)
#     FAs = np.expand_dims(np.array(FA_list), axis=ax_expand)
#     chemical_shift = (np.exp(np.array(TE_list).reshape(-1, 1) * 2j * np.pi * shift.reshape(1, -1))) @ amp
#     chemical_shift = np.expand_dims(chemical_shift, axis=ax_expand)
#     E_2 = np.exp(TEs * (2j * np.pi * df - 1 / T_2)) * (chemical_shift)
#     u = np.expand_dims(u_i[:-1], axis=-1)
#     s_i = u * np.sin(np.array(FAs)) * E_2
#     return s_i


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


def simulate_gen_eq_signal(TR_list, FA_list, TE_list, FF, df, T_1w, T_1f,B1, T_2w=40 / 1000, T_2f=80 / 1000,
                           amp=np.array([1]), shift=np.array([-418]), sigma=None, list_deriv=None,noise_size=None,noise_type="Absolute",group_size=None,return_fat_water=False,return_combined_signal=True):
    T_1w = np.array(T_1w)
    T_1f = np.array(T_1f)
    df = np.array(df)
    if return_combined_signal:
        FF = np.array(FF)
    B1 = np.array(B1)

    if (np.array(T_1w).shape == ()):
        T_1w = np.array([T_1w])
    if (np.array(T_1f).shape == ()):
        T_1f = np.array([T_1f])
    if (np.array(df).shape == ()):
        df = np.array([df])

    if return_combined_signal:
        if (np.array(FF).shape == ()):
            FF = np.array([FF])
    if (np.array(B1).shape == ()):
        B1 = np.array([B1])

    if not (T_1w.shape == (1,)):
        T_1w = np.squeeze(T_1w)
    if not (T_1f.shape == (1,)):
        T_1f = np.squeeze(T_1f)
    if not (df.shape == (1,)):
        df = np.squeeze(df)

    if return_combined_signal:
        if not (FF.shape == (1,)):
            FF = np.squeeze(FF)

    if not (B1.shape == (1,)):
        B1 = np.squeeze(B1)

    keys = list(product(list(T_1w), list(T_1f), list(B1), list(df)))

    #keys=np.array(keys).reshape(len(T_1w),len(T_1f),1,len(df),4)

    T_1w = np.expand_dims(T_1w, axis=0)
    T_1f = np.expand_dims(T_1f, axis=0)
    B1=np.expand_dims(B1, axis=(0, 1))
    df = np.expand_dims(df, axis=(0, 1,2))
    if return_combined_signal:
        FF = np.expand_dims(FF, axis=(0, 1, 2,3))


    s_iw = simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1w, T_2w,B1)[1:]
    s_iw = np.expand_dims(s_iw, axis=(2, -1))
    s_if = simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1f, T_2f,B1, amp, shift)[1:]
    s_if = np.expand_dims(s_if, axis=(1, -1))
    s_iw, s_if = np.broadcast_arrays(s_iw, s_if)

    if group_size is not None:
        s_iw=np.array([np.mean(gp, axis=0) for gp in groupby(s_iw, group_size)])
        s_if = np.array([np.mean(gp, axis=0) for gp in groupby(s_if, group_size)])


    if return_combined_signal:
        s_i = FF * s_if + (1 - FF) * s_iw
    else:
        s_i=None



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
            s_iw_dT1 = simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1w + dT1, T_2w,B1)[1:]
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


def write_seq_file(fileseq,FA_list,TE_list,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json",TI=None):
    with open(fileseq_basis,"r") as file:
        seq_config = json.load(file)

    seq_config_new = seq_config
    seq_config_new["B1"] = list(np.array(FA_list[1:]) * 180 / np.pi / 5)
    seq_config_new["TR"] = list((np.array(TE_list[1:])+min_TR_delay) * 10 ** 3)
    seq_config_new["TE"] = list(np.array(TE_list[1:]) * 10 ** 3)
    if TI is not None:
        seq_config_new["TI"]=TI

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



def create_new_seq(FA_list,TE_list,min_TR_delay,TI,FA_factor=5):
    seq_config_new={}
    seq_config_new["FA"]=FA_factor
    seq_config_new["TI"]=TI
    seq_config_new["TE"] = list(np.array(TE_list[1:]) * 10 ** 3)
    seq_config_new["TR"] = list((np.array(TE_list[1:])+min_TR_delay) * 10 ** 3)
    seq_config_new["B1"] = list(np.array(FA_list[1:]) * 180 / np.pi / 5)

    return seq_config_new

def generate_epg_dico_T1MRFSS_from_sequence(sequence_config,filedictconf,recovery,rep=2,overwrite=True,sim_mode="mean",start=None,window=None, dest=None,prefix_dico="dico"):
    if type(filedictconf)==str:
        with open(filedictconf) as f:
            dict_config = json.load(f)
    
    elif type(filedictconf)==dict:
        dict_config=filedictconf



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

    sequence_config["T_recovery"] = recovery*1000
    sequence_config["nrep"] = rep

    TR_delay=np.round(sequence_config["TR"][0]-sequence_config["TE"][0],2)

    seq = T1MRFSS(**sequence_config)


    fat_amp = np.array(dict_config["fat_amp"])
    fat_cs = dict_config["fat_cshift"]
    fat_cs = [- value / 1000 for value in fat_cs]  # temp

    # other options
    if window is None:
        window = dict_config["window_size"]


    if start is None:
        dictfile = prefix_dico  +"_TR{}_reco{}.dict".format(str(TR_delay),str(recovery))
    else:
        dictfile = prefix_dico + "_TR{}_reco{}_start{}.dict".format(str(TR_delay),str(recovery),start)

    if dest is not None:
        dictfile = str(pathlib.Path(dest) / pathlib.Path(dictfile).name)
    # print("Generating dictionary {}".format(dictfile))

    # water
    print("Generate water signals.")
    water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])
    water = water.reshape((rep, -1) + water.shape[1:])[-1]

    if sim_mode == "mean":
        water = [np.mean(gp, axis=0) for gp in groupby(water, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)

        water = water[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    # fat
    print("Generate fat signals.")
    # eval = "dot(signal, amps)"
    # args = {"amps": fat_amp}
    # merge df and fat_cs df to dict
    fatdf = [[cs + f for cs in fat_cs] for f in df]
    fat = seq(T1=[fT1], T2=fT2, att=[[att]], g=[[[fatdf]]])#, eval=eval, args=args)
    fat=fat @ fat_amp
    fat = fat.reshape((rep, -1) + fat.shape[1:])[-1]

    if sim_mode == "mean":
        fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)
        fat = fat[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    water = np.array(water)
    fat = np.array(fat)
    # join water and fat
    print("Build dictionary.")
    keys = list(itertools.product(wT1, fT1, att, df))
    values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
    values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

    # print("Save dictionary.")
    mrfdict = Dictionary(keys, values)
    # mrfdict.save(dictfile, overwrite=overwrite)
    hdr={"sequence_config":sequence_config,"dict_config":dict_config,"recovery":recovery,"initial_repetitions":rep,"window":window,"sim_mode":sim_mode,"param_names":("wT1","fT1","att","df")}
    return mrfdict,hdr,dictfile

def generate_epg_dico_T1MRF_generic_from_sequence(sequence_config,filedictconf,overwrite=True,sim_mode="mean",start=None,window=None, dest=None,prefix_dico="dico_MRF_generic"):

    if type(filedictconf)==str:
        with open(filedictconf) as f:
            dict_config = json.load(f)
    
    elif type(filedictconf)==dict:
        dict_config=filedictconf



    # generate signals
    wT1 = dict_config["water_T1"]
    fT1 = dict_config["fat_T1"]
    wT2 = dict_config["water_T2"]
    fT2 = dict_config["fat_T2"]
    att = dict_config["B1_att"]
    df = dict_config["delta_freqs"]
    df = [- value / 1000 for value in df]  # temp
    # df = np.linspace(-0.1, 0.1, 101)
    
    combinations = list(itertools.product(fT1, fT2, att, df))
    fat_T1, fat_T2, fat_att, fat_df = zip(*combinations)
    
    combinations = list(itertools.product(wT1, wT2, att, df))
    water_T1, water_T2, water_att, water_df = zip(*combinations)    

    TR_delay=sequence_config["dTR"]

    seq = T1MRF_generic(sequence_config)

    water_amp = [1]
    water_cs = [0]
    fat_amp = np.array(dict_config["fat_amp"])
    fat_cs = dict_config["fat_cshift"]
    fat_cs = [- value / 1000 for value in fat_cs]  # temp

    # other options
    if window is None:
        window = dict_config["window_size"]


    if start is None:
        dictfile = prefix_dico  +"_TR{}_reco{}.dict".format(str(TR_delay),str(sequence_config['T_recovery']))
    else:
        dictfile = prefix_dico + "_TR{}_reco{}_start{}.dict".format(str(TR_delay),str(sequence_config['T_recovery']),start)

    if dest is not None:
        dictfile = str(pathlib.Path(dest) / pathlib.Path(dictfile).name)
    # print("Generating dictionary {}".format(dictfile))


    print("Generate water signals.")
    water = seq(T1=np.array(water_T1), T2=np.array(water_T2), att=np.array(water_att), g=np.array(water_df), cs=water_cs, frac=water_amp)


    if sim_mode == "mean":
        water = [np.mean(gp, axis=0) for gp in groupby(water, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)

        water = water[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")
    
    water = np.reshape(water, (np.shape(water)[0], np.shape(wT1)[0],np.shape(wT2)[0], 1, 1, np.shape(att)[0], np.shape(df)[0] ))

    # fat
    print("Generate fat signals.")
    fat = seq(T1=np.array(fat_T1), T2=np.array(fat_T2), att=np.array(fat_att), g=np.array(fat_df), cs=fat_cs, frac=fat_amp)   
    
    if sim_mode == "mean":
        fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)
        fat = fat[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    fat = np.reshape(fat, (np.shape(fat)[0], 1, 1, np.shape(fT1)[0],np.shape(fT2)[0], np.shape(att)[0], np.shape(df)[0] ))
    
    water = np.array(water)
    fat = np.array(fat)
    # join water and fat
    print("Build dictionary.")
    keys = list(itertools.product(wT1, wT2, fT1, fT2, att, df))
    values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
    
    values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

    # print("Save dictionary.")
    mrfdict = Dictionary(keys, values)
    # mrfdict.save(dictfile, overwrite=overwrite)
    hdr={"sequence_config":sequence_config,"dict_config":dict_config,"recovery":sequence_config['T_recovery'],"window":window,"sim_mode":sim_mode,"param_names":("wT1","fT1","wT2","fT2","att","df")}
    return mrfdict,hdr,dictfile

def generate_epg_dico_T1MRF(sequence_config,filedictconf,overwrite=True,sim_mode="mean",useGPU=True, batch_size=None, start=None,window=None, dest=None,prefix_dico="dico_MRF_generic"):

    if type(filedictconf)==str:
        with open(filedictconf) as f:
            dict_config = json.load(f)
    
    elif type(filedictconf)==dict:
        dict_config=filedictconf


    
    seq = t1_mfr_seq(sequence_config)
    
    
    
    # generate signals
    wT1 = dict_config["water_T1"]
    fT1 = dict_config["fat_T1"]
    wT2 = dict_config["water_T2"]
    fT2 = dict_config["fat_T2"]
    att = dict_config["B1_att"]
    df = dict_config["delta_freqs"]
    df = [- value / 1000 for value in df]  # temp
    
    combinations = list(itertools.product(fT1, fT2, att, df))
    fat_T1, fat_T2, fat_att, fat_df = zip(*combinations)
    n_fat = len(fat_T1) 
    
    combinations = list(itertools.product(wT1, wT2, att, df))
    water_T1, water_T2, water_att, water_df = zip(*combinations)
    n_water = len(water_T1) 

    TR_delay=sequence_config["dTR"]

    water_amp = [1]
    water_cs = [0]
    fat_amp = np.array(dict_config["fat_amp"])
    fat_cs = dict_config["fat_cshift"]
    fat_cs = [- value / 1000 for value in fat_cs]  # temp

    # other options
    if window is None:
        window = dict_config["window_size"]


    if start is None:
        dictfile = prefix_dico  +"_TR{}_reco{}.dict".format(str(TR_delay),str(sequence_config['T_recovery']))
    else:
        dictfile = prefix_dico + "_TR{}_reco{}_start{}.dict".format(str(TR_delay),str(sequence_config['T_recovery']),start)

    if dest is not None:
        dictfile = str(pathlib.Path(dest) / pathlib.Path(dictfile).name)
    # print("Generating dictionary {}".format(dictfile))

    if useGPU:
        epg.set_array_module('cupy')
    else:
        epg.set_array_module('numpy')


    print("Generate water signals.")
    
    if batch_size is not None:
        n_batches_water = int(np.ceil(n_water / batch_size['water']))
        n_batches_fat = int(np.ceil(n_fat / batch_size['fat']))
        epg_opt = {"disp": True, 'max_nstate':30}
    else:
        n_batches_water = 1
        n_batches_fat = 1
        batch_size = {'water': n_water, 'fat': n_fat}
        epg_opt = {"disp": True, 'max_nstate':30}
    
    water = []
    for i in tqdm(range(n_batches_water)):
        start = i * batch_size['water']
        end = min((i + 1) * batch_size['water'], n_water)

        # Création du batch
        water_temp = seq(
            T1=np.array(water_T1[start:end]),
            T2=np.array(water_T2[start:end]),
            att=np.array(water_att[start:end]),
            g=np.array(water_df[start:end])[:, np.newaxis],
            cs=[water_cs],
            frac=[water_amp],
            options= epg_opt
        )
        water.append(water_temp)
    
    water = np.concatenate(water, axis=0)
    water = np.transpose(water, (1, 0))
    # water_temp = seq(T1=np.array(water_T1[idx]), T2=np.array(water_T2[idx]), att=np.array(water_att[idx]), g=np.array(water_df[idx])[:,np.newaxis], cs=[water_cs], frac=[water_amp], options={"disp":True})

    if sim_mode == "mean":
        water = [np.mean(gp, axis=0) for gp in groupby(water, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)

        water = water[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")
    
    
    water = np.array(water)
    water = np.reshape(water, (np.shape(water)[0], np.shape(wT1)[0],np.shape(wT2)[0], 1, 1, np.shape(att)[0], np.shape(df)[0] ))

    # fat
    print("Generate fat signals.")
    # fat_temp = seq(T1=np.array(fat_T1[idx]), T2=np.array(fat_T2[idx]), att=np.array(fat_att[idx]), g=np.array(fat_df[idx])[:,np.newaxis], cs=[fat_cs], frac=[fat_amp], options={"disp":True})

    fat = []
    for i in tqdm(range(n_batches_fat)):
        start = i * batch_size['fat']
        end = min((i + 1) * batch_size['fat'], n_fat)

        # Création du batch
        fat_temp = seq(
            T1=np.array(fat_T1[start:end]),
            T2=np.array(fat_T2[start:end]),
            att=np.array(fat_att[start:end]),
            g=np.array(fat_df[start:end])[:, np.newaxis],
            cs=[fat_cs],
            frac=[fat_amp],
            options= epg_opt
        )
        fat.append(fat_temp)
    
    fat = np.concatenate(fat, axis=0)
    fat = np.transpose(fat, (1, 0))
    
    if sim_mode == "mean":
        fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)
        fat = fat[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    fat = np.array(fat)
    fat = np.reshape(fat, (np.shape(fat)[0], 1, 1, np.shape(fT1)[0],np.shape(fT2)[0], np.shape(att)[0], np.shape(df)[0] ))
    
    # water = np.array(water)
    # fat = np.array(fat)
    # join water and fat
    print("Build dictionary.")
    keys = list(itertools.product(wT1, wT2, fT1, fT2, att, df))
    values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
    
    values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

    # print("Save dictionary.")
    mrfdict = Dictionary(keys, values)
    # mrfdict.save(dictfile, overwrite=overwrite)
    hdr={"sequence_config":sequence_config,"dict_config":dict_config,"recovery":sequence_config['T_recovery'],"window":window,"sim_mode":sim_mode,"param_names":("wT1","wT2","fT1","fT2","att","df")}
    return mrfdict,hdr,dictfile


def load_sequence_file(fileseq,recovery,min_TR_delay):

    if type(fileseq)==str:
        with open(fileseq, "r") as file:
            seq_config = json.load(file)
    elif type(fileseq)==dict:
        seq_config = fileseq

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


def convert_sequence_to_params_breaks(FA_, TE_, recovery):
    FA_breaks_ = list(np.argwhere(np.diff(np.array(FA_[1:])) != 0).flatten()) + [-1]
    TE_breaks_ = list(np.argwhere(np.diff(np.array(TE_[1:])) != 0).flatten()) + [-1]
    # print(FA_breaks)
    # print(TE_breaks)

    FA_values = [FA_[1:][i] for i in FA_breaks_]
    TE_values = [TE_[1:][i] for i in TE_breaks_]
    TE_breaks_ = [0] + TE_breaks_
    FA_breaks_ = [0] + FA_breaks_
    params_ = list(np.diff(TE_breaks_[:-1])) + list(np.diff(FA_breaks_[:-1])) + list(TE_values) + list(FA_values) + [
        recovery]

    return params_, TE_breaks_, FA_breaks_


def convert_params_to_sequence_breaks(params, min_TR_delay, num_breaks_TE, num_breaks_FA, spokes_count, inversion=True):
    TE_ = np.zeros(spokes_count + 1)
    # print(num_breaks_TE)
    # print(num_breaks_FA)
    TE_breaks = params[:num_breaks_TE].astype(int)
    FA_breaks = params[num_breaks_TE:(num_breaks_FA + num_breaks_TE)].astype(int)
    TE_breaks = [0] + list(np.cumsum(TE_breaks)) + [spokes_count]
    FA_breaks = [0] + list(np.cumsum(FA_breaks)) + [spokes_count]
    # print(TE_breaks)
    # print(FA_breaks)
    for j in range(num_breaks_TE + 1):
        TE_[(TE_breaks[j] + 1):(TE_breaks[j + 1] + 1)] = params[num_breaks_TE + num_breaks_FA + j]
    FA_ = np.zeros(spokes_count + 1)
    for j in range(num_breaks_FA + 1):
        FA_[(FA_breaks[j] + 1):(FA_breaks[j + 1] + 1)] = params[j + num_breaks_TE + 1 + num_breaks_TE + num_breaks_FA]

    FA_[0] = np.pi

    TR_ = np.zeros(spokes_count + 1)
    TR_[0] = 8.32 / 1000
    if not (inversion):
        FA_[0] = 0
        TR_[0] = 0
    TR_[1:] = np.array(TE_[1:]) + min_TR_delay
    TR_[-1] = TR_[-1] + params[-1]
    TR_ = list(TR_)
    return TR_, FA_, TE_


def convert_sequence_to_params_breaks_common(FA_, TE_, recovery):
    FA_breaks_ = list(np.argwhere(np.diff(np.array(FA_[1:])) != 0).flatten()) + [-1]
    TE_breaks_ = list(np.argwhere(np.diff(np.array(TE_[1:])) != 0).flatten()) + [-1]
    # print(FA_breaks)
    # print(TE_breaks)

    FA_values = [FA_[1:][i] for i in FA_breaks_]
    TE_values = [TE_[1:][i] for i in TE_breaks_]
    TE_breaks_ = [0] + TE_breaks_
    FA_breaks_ = [0] + FA_breaks_
    params_ = list(np.diff(FA_breaks_[:-1])) + list(TE_values) + list(FA_values) + [
        recovery]

    return params_, TE_breaks_, FA_breaks_


def convert_params_to_sequence_breaks_common(params, min_TR_delay, num_breaks_TE, num_breaks_FA, spokes_count, inversion=True):
    TE_ = np.zeros(spokes_count + 1)
    # print(num_breaks_TE)
    # print(num_breaks_FA)
    TE_breaks = params[:(num_breaks_FA + num_breaks_TE)].astype(int)
    FA_breaks = params[:(num_breaks_FA + num_breaks_TE)].astype(int)

    TE_breaks = [0] + list(np.cumsum(TE_breaks)[[0,2]]) + [spokes_count]
    FA_breaks = [0] + list(np.cumsum(FA_breaks)) + [spokes_count]
    #print(params)
    # print(TE_breaks)
    # print(FA_breaks)
    for j in range(num_breaks_TE + 1):
        TE_[(TE_breaks[j] + 1):(TE_breaks[j + 1] + 1)] = params[num_breaks_TE + num_breaks_FA + j]
    FA_ = np.zeros(spokes_count + 1)
    for j in range(len(FA_breaks)-1):
        FA_[(FA_breaks[j] + 1):(FA_breaks[j + 1] + 1)] = params[j + num_breaks_TE + 1 + num_breaks_TE + num_breaks_FA]

    FA_[0] = np.pi

    TR_ = np.zeros(spokes_count + 1)
    TR_[0] = 8.32 / 1000
    if not (inversion):
        FA_[0] = 0
        TR_[0] = 0
    TR_[1:] = np.array(TE_[1:]) + min_TR_delay
    TR_[-1] = TR_[-1] + params[-1]
    TR_ = list(TR_)
    return TR_, FA_, TE_

def generate_random_curve(T, params_rho,params_phi,bound_min,bound_max):
    #FA_bound_min = np.random.choice(np.arange(5, 16)) * np.pi / 180
    #FA_bound_max = np.random.choice(np.arange(30, 60)) * np.pi / 180
    H=len(params_rho)
    rho = (params_rho * np.logspace(-0.5, -2.5, H)).reshape(-1, 1)
    phi = (params_phi * 2 * np.pi).reshape(-1, 1)

    t = np.arange(0, 2 * np.pi-np.pi/T, 2 * np.pi / T).reshape(1, -1)

    traj = np.sum(rho * np.sin(np.arange(1, H + 1).reshape(-1, 1) @ t + phi), axis=0)
    traj_min = np.min(traj)
    traj_max = np.max(traj)

    traj = (traj - traj_min) / (traj_max - traj_min) * (bound_max - bound_min) + bound_min

    #traj = [np.pi] + list(FA_traj)

    return traj


def convert_params_to_sequence_breaks_random(params, min_TR_delay, spokes_count, bound_min_TE,bound_max_TE,bound_min_FA,bound_max_FA,inversion=True):
    nb_params=len(params)
    params_TE=params[:int((nb_params-1)/2)]
    params_FA = params[int((nb_params-1)/2):-1]
    nb_params_TE=int(len(params_TE)/2)
    nb_params_FA = int(len(params_FA) / 2)

    TE_=generate_random_curve(spokes_count, params_TE[:int(nb_params_TE)],params_TE[int(nb_params_TE):],bound_min_TE,bound_max_TE)
    FA_ = generate_random_curve(spokes_count, params_FA[:int(nb_params_FA)], params_FA[int(nb_params_FA):], bound_min_FA,
                                bound_max_FA)
    FA_=[np.pi]+list(FA_)
    TE_ = [0] + list(TE_)

    TR_ = np.zeros(spokes_count + 1)
    TR_[0] = 8.32 / 1000

    if not (inversion):
        FA_[0] = 0
        TR_[0] = 0

    TR_[1:] = np.array(TE_[1:]) + min_TR_delay
    TR_[-1] = TR_[-1] + params[-1]
    TR_ = list(TR_)
    return TR_, FA_, TE_

#@nb.njit
def convert_params_to_sequence_breaks_random_FA(params, min_TR_delay, spokes_count,num_breaks_TE,num_params_FA,
                                                 bound_min_FA, bound_max_FA, inversion=True):
    TE_ = np.zeros(spokes_count + 1)
    # print(num_breaks_TE)
    # print(num_breaks_FA)
    TE_breaks = params[:num_breaks_TE].astype(int)
    TE_breaks = [0] + list(np.cumsum(TE_breaks)) + [spokes_count]

    TE_breaks=np.unique([te for te in TE_breaks if te<=spokes_count])
    num_breaks_TE=len(TE_breaks)-2

    print(TE_breaks)
    # print(TE_breaks)
    # print(FA_breaks)
    for j in range(num_breaks_TE + 1):
        TE_[(TE_breaks[j] + 1):(TE_breaks[j + 1] + 1)] = params[num_breaks_TE + j]

    params_FA = params[(2*num_breaks_TE+1):(2*num_breaks_TE+1+num_params_FA)]
    nb_params_FA = int(len(params_FA) / 2)
    FA_ = generate_random_curve(spokes_count, params_FA[:int(nb_params_FA)], params_FA[int(nb_params_FA):],bound_min_FA,bound_max_FA)
    FA_ = [np.pi] + list(FA_)

    #TE_ = [0] + list(TE_)
    TE_=list(TE_)

    TR_ = np.zeros(spokes_count + 1)
    TR_[0] = 8.32 / 1000

    if not (inversion):
        FA_[0] = 0
        TR_[0] = 0

    TR_[1:] = np.array(TE_[1:]) + min_TR_delay
    TR_[-1] = TR_[-1] + params[-1]
    TR_ = list(TR_)
    return TR_, FA_, TE_


def convert_params_to_sequence_breaks_proportional_random_FA(params, min_TR_delay, spokes_count,num_breaks_TE,num_params_FA,
                                                 bound_min_FA, bound_max_FA, inversion=True):
    
    print("Generating Random Curve")
    TE_ = np.zeros(spokes_count + 1)
    # print(num_breaks_TE)
    # print(num_breaks_FA)
    print(params[:num_breaks_TE])
    TE_breaks = (spokes_count*(np.sort(params[:num_breaks_TE]))).astype(int)
    TE_breaks = [0] + list(TE_breaks) + [spokes_count]

    #TE_breaks=np.unique(TE_breaks)
    #num_breaks_TE_local=len(TE_breaks)-2
    
    print(TE_breaks)
    # print(TE_breaks)
    # print(FA_breaks)
    for j in range(num_breaks_TE + 1):
        TE_[(TE_breaks[j] + 1):(TE_breaks[j + 1] + 1)] = params[num_breaks_TE + j]

    params_FA = params[(2*num_breaks_TE+1):(2*num_breaks_TE+1+num_params_FA)]
    nb_params_FA = int(len(params_FA) / 2)
    FA_ = generate_random_curve(spokes_count, params_FA[:int(nb_params_FA)], params_FA[int(nb_params_FA):],bound_min_FA,bound_max_FA)
    FA_ = [np.pi] + list(FA_)

    #TE_ = [0] + list(TE_)
    TE_=list(TE_)

    TR_ = np.zeros(spokes_count + 1)
    TR_[0] = 8.32 / 1000

    if not (inversion):
        FA_[0] = 0
        TR_[0] = 0

    TR_[1:] = np.array(TE_[1:]) + min_TR_delay
    TR_[-1] = TR_[-1] + params[-1]
    TR_ = list(TR_)
    return TR_, FA_, TE_

def convert_params_to_sequence_breaks_FF_single_FA(params, min_TR_delay, spokes_count,num_breaks_TE,FA, inversion=False):
    TE_ = np.zeros(spokes_count + 1)
    # print(num_breaks_TE)
    # print(num_breaks_FA)
    TE_breaks = params[:num_breaks_TE].astype(int)
    TE_breaks = [0] + list(np.cumsum(TE_breaks)) + [spokes_count]
    # print(TE_breaks)
    # print(FA_breaks)
    for j in range(num_breaks_TE + 1):
        TE_[(TE_breaks[j] + 1):(TE_breaks[j + 1] + 1)] = params[num_breaks_TE + j]

    FA_ = [np.pi] + [FA]*spokes_count

    #TE_ = [0] + list(TE_)

    TR_ = np.zeros(spokes_count + 1)
    TR_[0] = 8.32 / 1000

    if not (inversion):
        FA_[0] = 0
        TR_[0] = 0

    TR_[1:] = np.array(TE_[1:]) + min_TR_delay
    TR_[-1] = TR_[-1] + [0.0]
    TR_ = list(TR_)
    return TR_, FA_, TE_


    
def save_maps(all_maps, file_seqParams, keys = ["ff","wT1","attB1","df"], dest=None):
    '''
    generate the map images in .mha format for all params and stores the optimisation results in a .pkl
    inputs:
    all_maps - tuple containing for all iterations 
            (maps - dictionary with parameter maps for all keys
             mask - numpy array
             cost map (OPTIONAL)
             phase map - numpy array (OPTIONAL)
             proton density map - numpy array (OPTIONAL)
             matched_signals - numpy array  (OPTIONAL))

    file_seqParams - file containing acquisition sequence parameters (.pkl)
    keys - parameters for which to generate the .mha
    
    outputs:
    '''

    if file_seqParams is not None:
        file = open(file_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        path, _ = os.path.split(file_seqParams)
        print(path)
    else:
        dico_seqParams = {}
        dico_seqParams["spacing"] = (1.0, 1.0, 5.0)
        dico_seqParams["origin"] = (0.0, 0.0, 0.0)
        dico_seqParams["orientation"] = 'transversal'
        dico_seqParams["offset"] = 0.0
        dico_seqParams["is3D"] = False
        path = dest
        

    file_map=os.path.join(path,"maps.pkl")
    with open(file_map,"wb") as file:
        pickle.dump(all_maps, file)

    for k in tqdm(keys) :
                
        map_rebuilt = all_maps[0][0]
        mask = all_maps[0][1]
        map_all_slices = makevol(map_rebuilt[k], mask > 0)            
        map_all_slices,geom=convertArrayToImageHelper(dico_seqParams,map_all_slices,apply_offset=True)
        # curr_volume=sitk.GetImageFromArray(map_all_slices)
        
        # curr_volume.SetSpacing(geom["spacing"])
        # curr_volume.SetOrigin(geom["origin"])
        
        # file_map=os.path.join(path,"{}_map.mha".format(k))
        # sitk.WriteImage(curr_volume,file_map,useCompression=True)
        vol = io.Volume(map_all_slices, **geom)
        io.write("{}_map.mha".format(k),vol)
        



def generate_dictionaries(sequence_file,reco,min_TR_delay,dictconf,dictconf_light,TI=8.32, dest=None,diconame="dico",is_build_phi=False,L0=6):
    '''
    Generates dictionaries from sequence and dico configuration files
    inputs:
    sequence_file - sequence acquistion parameters (dictionary)
    reco - waiting time at the end of each MRF repetition (seconds)
    min_TR_delay - waiting time from echo time to next RF pulse (ms)
    dictconf - dictionary parameter grid for full dictionary (dictionary)
    dictconf_light - dictionary parameter grid for light dictionary (dictionary)
    TI - inversion time (ms)
    build_phi - whether to build temporal basis phi (bool)
    L0 - number of temporal components (int)
    
    outputs:
    saves the dictionaries paths and headers in a .pkl file
    '''


    _,FA_list,TE_list=load_sequence_file(sequence_file,reco,min_TR_delay/1000)
    seq_config=create_new_seq(FA_list,TE_list,min_TR_delay/1000,TI)

    mrfdict,hdr,dictfile=generate_epg_dico_T1MRFSS_from_sequence(seq_config,dictconf,reco, dest=dest,prefix_dico="{}".format(diconame))
    mrfdict_light,hdr_light,dictfile_light=generate_epg_dico_T1MRFSS_from_sequence(seq_config,dictconf_light,reco, dest=dest,prefix_dico="{}_light".format(diconame))
    
    dico_full_with_hdr={"hdr":hdr,
                        "hdr_light":hdr_light,
                        "mrfdict":mrfdict,
                        "mrfdict_light":mrfdict_light}
    
    if is_build_phi:
        dico_full_with_hdr=add_temporal_basis(dico_full_with_hdr,L0)
            


    dico_full_name = str.split(dictfile,".dict")[0]+".pkl"
    with open(dico_full_name,"wb") as file:
        pickle.dump(dico_full_with_hdr,file)
    print("Generated dictionary {}".format(dico_full_name))

    return

def generate_dictionaries_mrf_generic(sequence_config,dictconf,dictconf_light,useGPU=True, batch_size = None, dest=None,diconame="dico",is_build_phi=False,L0=6):
    '''
    Generates dictionaries from sequence and dico configuration files
    inputs:
    sequence_file - sequence acquistion parameters (dictionary)
    reco - waiting time at the end of each MRF repetition (seconds)
    min_TR_delay - waiting time from echo time to next RF pulse (ms)
    dictconf - dictionary parameter grid for full dictionary (dictionary)
    dictconf_light - dictionary parameter grid for light dictionary (dictionary)
    TI - inversion time (ms)
    build_phi - whether to build temporal basis phi (bool)
    L0 - number of temporal components (int)
    
    outputs:
    saves the dictionaries paths and headers in a .pkl file
    '''
    mrfdict,hdr,dictfile=generate_epg_dico_T1MRF(sequence_config,dictconf,useGPU=useGPU, batch_size=batch_size, dest=dest,prefix_dico="{}".format(diconame))
    mrfdict_light,hdr_light,dictfile_light=generate_epg_dico_T1MRF(sequence_config,dictconf_light,useGPU=useGPU, batch_size=batch_size, dest=dest,prefix_dico="{}_light".format(diconame))
    
    dico_full_with_hdr={"hdr":hdr,
                        "hdr_light":hdr_light,
                        "mrfdict":mrfdict,
                        "mrfdict_light":mrfdict_light}
    
    if is_build_phi:
        dico_full_with_hdr=add_temporal_basis(dico_full_with_hdr,L0)
            


    dico_full_name = str.split(dictfile,".dict")[0]+".pkl"
    with open(dico_full_name,"wb") as file:
        pickle.dump(dico_full_with_hdr,file)
    print("Generated dictionary {}".format(dico_full_name))

    return
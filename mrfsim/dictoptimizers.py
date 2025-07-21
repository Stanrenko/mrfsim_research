import numpy as np
import itertools
from scipy import ndimage
from tqdm import tqdm
import twixtools
from datetime import datetime
from copy import copy
import os
import pickle
from mrfsim.Transformers import PCAComplex

try:
    import cupy as cp
except:
    print("Could not import cupy")

from .dictmodel import Dictionary

import dask.array as da


DEFAULT_CLUSTERING_WINDOWS={
    "wT1":400,
    "fT1":100,
    "df":0.03,
    "attB1":0.2
}

def combine_mrf_dict_components(mrfdict ,FF_list):
    '''
    Combine the water and fat components of the dictionary with FF_list to create a single dictionary with all the FFs
    '''

    ff = np.zeros(mrfdict.values.shape[:-1 ] +(len(FF_list),))
    ff_matrix =np.tile(np.array(FF_list) ,ff.shape[:-1 ] +(1,))

    water_signal =np.expand_dims(mrfdict.values[: ,: ,0] ,axis=-1 ) *(1-ff_matrix)
    fat_signal =np.expand_dims(mrfdict.values[: ,: ,1] ,axis=-1 ) *(ff_matrix)

    signal =water_signal +fat_signal

    signal_reshaped =np.moveaxis(signal ,-1 ,-2)
    signal_reshaped =signal_reshaped.reshape((-1 ,signal_reshaped.shape[-1]))

    keys_with_ff = list(itertools.product(mrfdict.keys, FF_list))
    keys_with_ff = [(*res, f) for res, f in keys_with_ff]

    return keys_with_ff,signal_reshaped


def build_phi(mrfdict,FFs=np.arange(0.1,1.09,0.1)):

    #mrfdict = dictsearch.Dictionary()
    print("Generating full dictionary")
    keys,values=combine_mrf_dict_components(mrfdict,FFs)

    
    print("Performing svd")
    u,s,vh = da.linalg.svd(da.asarray(values))

    print("SVD done")
    vh=np.array(vh)
    s=np.array(s)

    # phi=vh[:L0]
    #phi=vh

    return vh



def match_signals_v2(all_signals,keys,pca_water,pca_fat,array_water_unique,array_fat_unique,transformed_array_water_unique,transformed_array_fat_unique,var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates,verbose,split,useGPU_dictsearch,param_names=("wT1","fT1","attB1","df"),return_matched_signals=False):

    nb_signals = all_signals.shape[1]

    if remove_duplicates:
        all_signals, index_signals_unique = np.unique(all_signals, axis=1, return_inverse=True)
        nb_signals = all_signals.shape[1]

    print("There are {} unique signals to match along {} water and {} fat components".format(nb_signals,
                                                                                             array_water_unique.shape[
                                                                                                 0],
                                                                                             array_fat_unique.shape[
                                                                                                 0]))




    num_group = int(nb_signals / split) + 1

    #idx_max_all_unique = []
    #alpha_optim = []

    if not(useGPU_dictsearch):
        idx_max_all_unique = np.zeros(nb_signals,dtype="int64")
        alpha_optim = np.zeros(nb_signals)
    else:
        idx_max_all_unique = cp.zeros(nb_signals,dtype="int64")
        alpha_optim = cp.zeros(nb_signals)


    if return_matched_signals:
        phase_optim = []
        J_optim = []


    for j in tqdm(range(num_group)):
        j_signal = j * split
        j_signal_next = np.minimum((j + 1) * split, nb_signals)

        if j_signal==j_signal_next:
            continue

        if verbose:
            print("PCA transform")
            start = datetime.now()

        if not (useGPU_dictsearch):

            if pca:
                transformed_all_signals_water = np.transpose(
                    pca_water.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))
                transformed_all_signals_fat = np.transpose(
                    pca_fat.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))

                sig_ws_all_unique = np.matmul(transformed_array_water_unique,
                                              transformed_all_signals_water.conj())
                sig_fs_all_unique = np.matmul(transformed_array_fat_unique,
                                              transformed_all_signals_fat.conj())
            else:
                sig_ws_all_unique = np.matmul(array_water_unique, all_signals[:, j_signal:j_signal_next].conj())
                sig_fs_all_unique = np.matmul(array_fat_unique, all_signals[:, j_signal:j_signal_next].conj())


        else:

            if pca:

                transformed_all_signals_water = cp.transpose(
                    pca_water.transform(cp.transpose(cp.asarray(all_signals[:, j_signal:j_signal_next])))).get()
                transformed_all_signals_fat = cp.transpose(
                    pca_fat.transform(cp.transpose(cp.asarray(all_signals[:, j_signal:j_signal_next])))).get()

                sig_ws_all_unique = (cp.matmul(cp.asarray(transformed_array_water_unique),
                                               cp.asarray(transformed_all_signals_water).conj())).get()
                sig_fs_all_unique = (cp.matmul(cp.asarray(transformed_array_fat_unique),
                                               cp.asarray(transformed_all_signals_fat).conj())).get()
            else:

                sig_ws_all_unique = (cp.matmul(cp.asarray(array_water_unique),
                                               cp.asarray(all_signals)[:, j_signal:j_signal_next].conj())).get()
                sig_fs_all_unique = (cp.matmul(cp.asarray(array_fat_unique),
                                               cp.asarray(all_signals)[:, j_signal:j_signal_next].conj())).get()

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Extracting all sig_ws and sig_fs")
            start = datetime.now()

        if index_water_unique is not None:
            current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
            current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]

        else:
            current_sig_ws_for_phase=sig_ws_all_unique
            current_sig_fs_for_phase=sig_fs_all_unique

        if verbose:
            end = datetime.now()
            print(end - start)

        if not (useGPU_dictsearch):

            if verbose:
                print("Adjusting Phase")
                print("Calculating alpha optim and flooring")

            A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
            B = (
                        current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase

            a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
            b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
            c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

            discr = b ** 2 - 4 * a * c
            alpha1 = (-b + np.sqrt(discr)) / (2 * a)
            alpha2 = (-b - np.sqrt(discr)) / (2 * a)

            del a
            del b
            del c
            del discr

            current_alpha_all_unique = (1 * (alpha1 >= 0) & (alpha1 <= 1)) * alpha1 + (
                    1 - (1 * (alpha1 >= 0) & (alpha1 <= 1))) * alpha2

            if verbose:
                start = datetime.now()

            apha_more_0=(current_alpha_all_unique>=0)
            alpha_less_1=(current_alpha_all_unique<=1)
            alpha_out_bounds=(1*(apha_more_0))*(1*(alpha_less_1))==0

            J_0=np.abs(current_sig_ws_for_phase)/np.sqrt(var_w)

            J_1 = np.abs(current_sig_fs_for_phase) / np.sqrt(var_f)

            current_alpha_all_unique[alpha_out_bounds]=np.argmax(np.concatenate([J_0[alpha_out_bounds, None], J_1[alpha_out_bounds, None]], axis=-1), axis=-1).astype("float")


            if verbose:
                end = datetime.now()
                print(end - start)

            if verbose:
                print("Calculating cost for all signals")
            start = datetime.now()


            J_all = np.abs((
                             1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase) / np.sqrt(
                (
                        1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                        1 - current_alpha_all_unique) * sig_wf)

            end = datetime.now()

            all_J = np.stack([J_all, J_0, J_1], axis=0)

            ind_max_J = np.argmax(all_J, axis=0)

            del all_J


            J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_0 + (ind_max_J == 2) * J_1
            del J_0
            del J_1

            current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 0 + (
                        ind_max_J == 2) * 1

            idx_max_all_current = np.argmax(J_all, axis=0)
            current_alpha_all_unique_optim=current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])]
            idx_max_all_unique[j_signal:j_signal_next]=idx_max_all_current
            alpha_optim[j_signal:j_signal_next]=current_alpha_all_unique_optim


            if return_matched_signals:
                d = (
                            1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current, np.arange(J_all.shape[1])] + current_alpha_all_unique_optim * current_sig_fs_for_phase[idx_max_all_current, np.arange(J_all.shape[1])]
                phase_adj = -np.arctan(d.imag / d.real)
                cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real

                del d

                phase_adj = (phase_adj) * (
                        1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                    1 * (cond) > 0)

                del cond

            if return_matched_signals:
                J_all_optim=J_all[idx_max_all_current, np.arange(J_all.shape[1])]


            del J_all
            del current_alpha_all_unique



        else:
            if verbose:
                print("Calculating alpha optim and flooring")
                start = datetime.now()

            current_sig_ws_for_phase = cp.asarray(current_sig_ws_for_phase)
            current_sig_fs_for_phase = cp.asarray(current_sig_fs_for_phase)

            ### Testing direct phase solving
            A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
            B = (
                        current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase

            a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
            b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
            c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

            del A
            del B

            # del beta
            # del delta
            # del gamma
            # del nu

            discr = b ** 2 - 4 * a * c
            alpha1 = (-b + np.sqrt(discr)) / (2 * a)
            alpha2 = (-b - np.sqrt(discr)) / (2 * a)

            #################################################################################################################################""""
            del a
            del b
            del c
            del discr

            current_alpha_all_unique = (1 * (alpha1 >= 0) & (alpha1 <= 1)) * alpha1 + (
                    1 - (1 * (alpha1 >= 0) & (alpha1 <= 1))) * alpha2

            # current_alpha_all_unique_2 = (1 * (alpha2 >= 0) & (alpha2 <= 1)) * alpha2 + (
            #            1 - (1*(alpha2 >= 0) & (alpha2 <= 1))) * alpha1

            del alpha1
            del alpha2

            if verbose:
                end = datetime.now()
                print(end - start)

            if verbose:
                start = datetime.now()

            apha_more_0 = (current_alpha_all_unique >= 0)
            alpha_less_1 = (current_alpha_all_unique <= 1)
            alpha_out_bounds = (1 * (apha_more_0)) * (1 * (alpha_less_1)) == 0



            J_0 = cp.abs(current_sig_ws_for_phase) / cp.sqrt(var_w)
            J_1 = cp.abs(current_sig_fs_for_phase) / cp.sqrt(var_f)

            current_alpha_all_unique[alpha_out_bounds] = cp.argmax(
                cp.reshape(cp.concatenate([J_0[alpha_out_bounds], J_1[alpha_out_bounds]], axis=-1), (-1, 2)), axis=-1)

            if verbose:
                end = datetime.now()
                print(end - start)

            if verbose:
                print("Calculating cost for all signals")
                start = datetime.now()


            J_all = cp.abs((
                             1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase) / np.sqrt(
                (
                        1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                        1 - current_alpha_all_unique) * sig_wf)


            all_J = cp.stack([J_all, J_0, J_1], axis=0)

            ind_max_J = cp.argmax(all_J, axis=0)

            del all_J


            J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_0 + (ind_max_J == 2) * J_1
            del J_0
            del J_1

            current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 0 + (
                    ind_max_J == 2) * 1

            idx_max_all_current = cp.argmax(J_all, axis=0)
            current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])]

            idx_max_all_unique[j_signal:j_signal_next] = idx_max_all_current
            alpha_optim[j_signal:j_signal_next]=current_alpha_all_unique_optim

            

            if return_matched_signals:
                d = (
                            1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current, cp.arange(J_all.shape[1])] + current_alpha_all_unique_optim * current_sig_fs_for_phase[idx_max_all_current, cp.arange(J_all.shape[1])]
                phase_adj = -cp.arctan(d.imag / d.real)
                cond = cp.sin(phase_adj) * d.imag - cp.cos(phase_adj) * d.real

                del d

                phase_adj = (phase_adj) * (
                        1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                    1 * (cond) > 0)
                
                phase_adj=phase_adj.get()

                del cond

            del current_sig_ws_for_phase
            del current_sig_fs_for_phase


            if return_matched_signals:
                J_all_optim = J_all[idx_max_all_current, cp.arange(J_all.shape[1])]
                J_all_optim=J_all_optim.get()


            idx_max_all_current = idx_max_all_current.get()

            del J_all
            del current_alpha_all_unique


            if verbose:
                end = datetime.now()
                print(end - start)



        if verbose:
            print("Extracting index of pattern with max correl")
            start = datetime.now()

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Filling the lists with results for this loop")
            start = datetime.now()





        del current_alpha_all_unique_optim
        del idx_max_all_current


        if return_matched_signals:
            phase_optim.extend(phase_adj)
            J_optim.extend(J_all_optim)


        if verbose:
            end = datetime.now()
            print(end - start)


    if useGPU_dictsearch:
        idx_max_all_unique=idx_max_all_unique.get()
        alpha_optim=alpha_optim.get()

    if return_matched_signals:
        phase_optim = np.array(phase_optim)
        J_optim = np.array(J_optim)




    idx_max_all_unique=idx_max_all_unique.astype(int)
    params_all_unique = np.array(
        [keys[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique)])

    if remove_duplicates:
        params_all = params_all_unique[index_signals_unique]
    else:
        params_all = params_all_unique

    del params_all_unique

    map_rebuilt=dict(zip(param_names, np.transpose(params_all[:,:-1])))
    map_rebuilt["ff"]=params_all[:,-1]
    
    if return_matched_signals:
        matched_signals=array_water_unique[index_water_unique, :][idx_max_all_unique, :].T * (
                        1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][
                                                                    idx_max_all_unique, :].T * np.array(
                    alpha_optim).reshape(1, -1)
        matched_signals/=np.linalg.norm(matched_signals,axis=0)
        matched_signals *= J_optim*np.exp(1j*phase_optim)
        return map_rebuilt,None,None,matched_signals.squeeze()
    else:
        return map_rebuilt, None, None



def match_signals_v2_clustered_on_dico(all_signals_current,keys,pca_water,pca_fat,transformed_array_water_unique,transformed_array_fat_unique,var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique,useGPU_dictsearch,unique_keys,labels,split,param_names=("wT1","fT1","attB1","df"),clustering_windows=DEFAULT_CLUSTERING_WINDOWS,high_ff=False,return_cost=False):

    nb_clusters = unique_keys.shape[-1]

    nb_signals=all_signals_current.shape[-1]

    if not(useGPU_dictsearch):
        idx_max_all_unique_low_ff = np.zeros(nb_signals)
        alpha_optim_low_ff = np.zeros(nb_signals)
        if return_cost:
            J_optim = np.zeros(nb_signals)
            phase_optim=np.zeros(nb_signals)
    else:
        idx_max_all_unique_low_ff = cp.zeros(nb_signals,dtype="int64")
        alpha_optim_low_ff = cp.zeros(nb_signals)
        if return_cost:
            J_optim = cp.zeros(nb_signals)
            phase_optim=cp.zeros(nb_signals)


    if not (useGPU_dictsearch):
        labels = np.array(labels)
        for cl in tqdm(range(nb_clusters)):

            indices = np.argwhere(labels == cl)
            nb_signals_cluster=len(indices)
            num_group = int(nb_signals_cluster / split) + 1

            for i,param in enumerate(param_names):
                if param in clustering_windows.keys():
                    d_param=clustering_windows[param]
                    current_retained_keys=(keys[:, i] < unique_keys[:, cl][i] + d_param) & ((keys[:, i] > unique_keys[:, cl][i] - d_param))
                    if i==0:
                        retained_keys=current_retained_keys
                    else:
                        retained_keys=retained_keys & current_retained_keys
            
            retained_signals = np.argwhere(retained_keys).flatten()


            var_w = var_w_total[retained_signals]
            var_f = var_f_total[retained_signals]
            sig_wf = sig_wf_total[retained_signals]

            all_signals_cluster=all_signals_current[:, indices.flatten()]
            idx_max_all_unique_cluster = []
            alpha_optim_cluster = []
            if return_cost:
                J_optim_cluster = np.zeros(nb_signals_cluster)
                phase_optim_cluster = np.zeros(nb_signals_cluster)

            for j in range(num_group):
                j_signal = j * split
                j_signal_next = np.minimum((j + 1) * split, nb_signals_cluster)
                if j_signal==j_signal_next:
                    continue

                transformed_all_signals_water = np.transpose(
                    pca_water.transform(np.transpose(all_signals_cluster[:, j_signal:j_signal_next])))
                transformed_all_signals_fat = np.transpose(
                    pca_fat.transform(np.transpose(all_signals_cluster[:, j_signal:j_signal_next])))
                sig_ws_all_unique = np.matmul(transformed_array_water_unique,
                                              transformed_all_signals_water.conj())
                sig_fs_all_unique = np.matmul(transformed_array_fat_unique,
                                              transformed_all_signals_fat.conj())
                current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :][retained_signals]
                current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :][retained_signals]
                A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
                B = (
                            current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase
                a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
                b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag
                discr = b ** 2 - 4 * a * c
                alpha1 = (-b + np.sqrt(discr)) / (2 * a)
                alpha2 = (-b - np.sqrt(discr)) / (2 * a)
                del a
                del b
                del c
                del discr
                current_alpha_all_unique = (1 * (alpha1 >= 0) & (alpha1 <= 1)) * alpha1 + (
                        1 - (1 * (alpha1 >= 0) & (alpha1 <= 1))) * alpha2

                apha_more_0 = (current_alpha_all_unique >= 0)
                alpha_less_1 = (current_alpha_all_unique <= 1)
                alpha_out_bounds = (1 * (apha_more_0)) * (1 * (alpha_less_1)) == 0

                if not(high_ff):
                    J_0 = np.abs(current_sig_ws_for_phase) / np.sqrt(var_w)
                J_1 = np.abs(current_sig_fs_for_phase) / np.sqrt(var_f)

                if not(high_ff):
                    current_alpha_all_unique[alpha_out_bounds] = np.argmax(
                    np.concatenate([J_0[alpha_out_bounds, None], J_1[alpha_out_bounds, None]], axis=-1), axis=-1).astype(
                    "float")
                else:
                    current_alpha_all_unique[alpha_out_bounds] = 1

                J_all = np.abs((
                                       1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase) / np.sqrt(
                    (
                            1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                            1 - current_alpha_all_unique) * sig_wf)


                if not(high_ff):
                    all_J = np.stack([J_all, J_0, J_1], axis=0)
                else:
                    all_J = np.stack([J_all, J_1], axis=0)
                ind_max_J = np.argmax(all_J, axis=0)
                del all_J

                if not(high_ff):
                    J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_0 + (ind_max_J == 2) * J_1
                    del J_0
                    current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 0 + (
                            ind_max_J == 2) * 1
                else:
                    J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_1
                    current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 1
                del J_1

                idx_max_all_current_sig = np.argmax(J_all, axis=0)
                current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current_sig, np.arange(J_all.shape[1])]
                idx_max_all_unique_cluster.extend(idx_max_all_current_sig)
                alpha_optim_cluster.extend(current_alpha_all_unique_optim)

                if return_cost:
                    J_optim_cluster[j_signal:j_signal_next] = np.nan_to_num(J_all[idx_max_all_current_sig, np.arange(J_all.shape[1])] / np.linalg.norm(all_signals_cluster[:, j_signal:j_signal_next],axis=0))
                    d = (1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current_sig, np.arange(J_all.shape[1])] + current_alpha_all_unique_optim * \
                        current_sig_fs_for_phase[idx_max_all_current_sig, np.arange(J_all.shape[1])]
                    phase_adj = -np.arctan(d.imag / d.real)
                    cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real
                    del d
                    phase_adj = (phase_adj) * (
                            1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                        1 * (cond) > 0)
                    phase_optim_cluster[j_signal:j_signal_next]=np.nan_to_num(phase_adj)


            if return_cost:
                J_optim[indices.flatten()]=J_optim_cluster
                phase_optim[indices.flatten()] = phase_optim_cluster

            idx_max_all_unique_low_ff[indices.flatten()] = (retained_signals[idx_max_all_unique_cluster])
            alpha_optim_low_ff[indices.flatten()] = (alpha_optim_cluster)

    else:

        for cl in tqdm(range(nb_clusters)):

            indices = cp.argwhere(labels == cl)
            nb_signals_cluster=len(indices)
            num_group = int(nb_signals_cluster / split) + 1


            for i,param in enumerate(param_names):
                if param in clustering_windows.keys():
                    d_param=clustering_windows[param]
                    current_retained_keys=(keys[:, i] < unique_keys[:, cl][i] + d_param) & ((keys[:, i] > unique_keys[:, cl][i] - d_param))
                    if i==0:
                        retained_keys=current_retained_keys
                    else:
                        retained_keys=retained_keys & current_retained_keys
            
            retained_signals = cp.argwhere(retained_keys).flatten()

            var_w = var_w_total[retained_signals]
            var_f = var_f_total[retained_signals]
            sig_wf = sig_wf_total[retained_signals]

            all_signals_cluster=cp.asarray(all_signals_current[:, (indices.get()).flatten()])
            idx_max_all_unique_cluster = cp.zeros(nb_signals_cluster,dtype="int64")
            alpha_optim_cluster = cp.zeros(nb_signals_cluster)
            if return_cost:
                J_optim_cluster = cp.zeros(nb_signals_cluster)
                phase_optim_cluster = cp.zeros(nb_signals_cluster)

            for j in range(num_group):
                j_signal = j * split
                j_signal_next =cp.minimum((j + 1) * split, nb_signals_cluster)

                if j_signal==j_signal_next:
                    continue


                transformed_all_signals_water = cp.transpose(
                    pca_water.transform(cp.transpose(all_signals_cluster[:, j_signal:j_signal_next])))
                transformed_all_signals_fat = cp.transpose(
                    pca_fat.transform(cp.transpose(all_signals_cluster[:, j_signal:j_signal_next])))
                sig_ws_all_unique = cp.matmul(cp.asarray(transformed_array_water_unique),
                                              transformed_all_signals_water.conj())
                sig_fs_all_unique = cp.matmul(cp.asarray(transformed_array_fat_unique),
                                              transformed_all_signals_fat.conj())
                current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :][retained_signals]
                current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :][retained_signals]
                A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
                B = (
                            current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase
                a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
                b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag
                discr = b ** 2 - 4 * a * c
                alpha1 = (-b + cp.sqrt(discr)) / (2 * a)
                alpha2 = (-b - cp.sqrt(discr)) / (2 * a)
                del a
                del b
                del c
                del discr
                current_alpha_all_unique = (1 * (alpha1 >= 0) & (alpha1 <= 1)) * alpha1 + (
                        1 - (1 * (alpha1 >= 0) & (alpha1 <= 1))) * alpha2
                # current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)
                apha_more_0 = (current_alpha_all_unique >= 0)
                alpha_less_1 = (current_alpha_all_unique <= 1)
                alpha_out_bounds = (1 * (apha_more_0)) * (1 * (alpha_less_1)) == 0

                if not(high_ff):
                    J_0 = cp.abs(current_sig_ws_for_phase) / cp.sqrt(var_w)
                J_1 = cp.abs(current_sig_fs_for_phase) / cp.sqrt(var_f)

                if not(high_ff):
                    current_alpha_all_unique[alpha_out_bounds] = cp.argmax(
                cp.reshape(cp.concatenate([J_0[alpha_out_bounds], J_1[alpha_out_bounds]], axis=-1), (-1, 2)), axis=-1)
                else:
                    current_alpha_all_unique[alpha_out_bounds] = 1

                J_all = cp.abs((
                                       1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase) / cp.sqrt(
                    (
                            1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                            1 - current_alpha_all_unique) * sig_wf)


                if not(high_ff):
                    all_J = cp.stack([J_all, J_0, J_1], axis=0)
                else:
                    all_J = cp.stack([J_all, J_1], axis=0)
                ind_max_J = cp.argmax(all_J, axis=0)
                del all_J

                if not(high_ff):
                    J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_0 + (ind_max_J == 2) * J_1
                    del J_0
                    current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 0 + (
                            ind_max_J == 2) * 1
                else:
                    J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_1
                    current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 1
                del J_1

                idx_max_all_current_sig = cp.argmax(J_all, axis=0)
                current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current_sig, cp.arange(J_all.shape[1])]

                idx_max_all_unique_cluster[j_signal:j_signal_next]=idx_max_all_current_sig
                alpha_optim_cluster[j_signal:j_signal_next]=current_alpha_all_unique_optim

                if return_cost:
                    J_optim_cluster[j_signal:j_signal_next] = cp.nan_to_num(J_all[idx_max_all_current_sig, cp.arange(J_all.shape[1])] / cp.linalg.norm(all_signals_cluster[:, j_signal:j_signal_next],axis=0))
                    d = (1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current_sig, cp.arange(J_all.shape[1])] + current_alpha_all_unique_optim * \
                        current_sig_fs_for_phase[idx_max_all_current_sig, cp.arange(J_all.shape[1])]
                    phase_adj = -cp.arctan(d.imag / d.real)
                    cond = cp.sin(phase_adj) * d.imag - cp.cos(phase_adj) * d.real
                    del d
                    phase_adj = (phase_adj) * (
                            1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                        1 * (cond) > 0)
                    phase_optim_cluster[j_signal:j_signal_next]=cp.nan_to_num(phase_adj)



            idx_max_all_unique_low_ff[indices.flatten()] = (retained_signals[idx_max_all_unique_cluster])
            alpha_optim_low_ff[indices.flatten()] = (alpha_optim_cluster)
            if return_cost:
                J_optim[indices.flatten()]=J_optim_cluster
                phase_optim[indices.flatten()] = phase_optim_cluster



        idx_max_all_unique_low_ff=idx_max_all_unique_low_ff.get()
        alpha_optim_low_ff = alpha_optim_low_ff.get()
        if return_cost:
            J_optim=J_optim.get()
            phase_optim=phase_optim.get()


    if return_cost:
        return idx_max_all_unique_low_ff,alpha_optim_low_ff,J_optim,phase_optim

    return idx_max_all_unique_low_ff,alpha_optim_low_ff

class Optimizer(object):

    def __init__(self,mask=None,verbose=False,useGPU=False,**kwargs):
        self.paramDict=kwargs
        self.paramDict["useGPU"]=useGPU
        self.mask=mask
        self.verbose=verbose


    def search_patterns(self,dictfile,volumes,retained_timesteps=None):
        #takes as input dictionary pattern and an array of images or volumes and outputs parametric maps
        raise ValueError("search_patterns should be implemented in child")

class SimpleDictSearch(Optimizer):

    def __init__(self,split=500,pca=True,threshold_pca=15,useGPU_dictsearch=False,remove_duplicate_signals=False,threshold=None,return_matched_signals=False,volumes_type="raw",clustering_windows=DEFAULT_CLUSTERING_WINDOWS,force_pca=True,**kwargs):
        
        super().__init__(**kwargs)
        self.paramDict["split"] = split
        self.paramDict["pca"] = pca
        self.paramDict["threshold_pca"] = int(threshold_pca)
        self.paramDict["remove_duplicate_signals"] = remove_duplicate_signals
        self.paramDict["return_matched_signals"] = return_matched_signals


        self.paramDict["useGPU_dictsearch"]=useGPU_dictsearch
        self.paramDict["threshold"]=threshold

        if volumes_type not in ["singular", "raw"]:
            raise ValueError('volumes_type must be either "singular" or "raw".')
        
        self.paramDict["volumes_type"]=volumes_type

        if clustering_windows is None:
            clustering_windows = DEFAULT_CLUSTERING_WINDOWS
        self.paramDict["clustering_windows"]=clustering_windows

        self.paramDict["force_pca"]=force_pca


    def search_patterns_test_multi(self, dicofull_file, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        volumes_type=self.paramDict["volumes_type"]
        

        verbose = self.verbose
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        force_pca = self.paramDict["force_pca"]

        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]

        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        # if pca and (type()==str):
        #     pca_file = str.split(dictfile, ".dict")[0] + "_{}pca_simple.pkl".format(threshold_pca)
        #     pca_file_name = str.split(pca_file, "/")[-1]

        # if type(dictfile)==str:
        #     vars_file = str.split(dictfile, ".dict")[0] + "_vars_simple.pkl".format(threshold_pca)
        #     vars_file_name = str.split(vars_file, "/")[-1]
        #     path = str.split(os.path.realpath(__file__), "/utils_mrf.py")[0]

        if volumes.ndim > 2:
            all_signals = volumes[:, mask > 0]
        else:  # already masked
            all_signals = volumes

        ntimesteps=volumes.shape[0]

        all_signals=all_signals.astype("complex64")


        del volumes

        with open(dicofull_file, "rb") as file:
            dicofull = pickle.load(file)

        if "param_names" in dicofull["hdr_light"].keys():
            param_names=dicofull["hdr_light"]["param_names"]

        else:
            param_names=("wT1","fT1","attB1","df")

        if volumes_type == "raw":
            mrfdict = dicofull["mrfdict_light"]

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]

            del mrfdict
        elif volumes_type=="singular":  # otherwise dictfile contains (s_w,s_f,keys)
            array_water = dicofull["mrfdict_light_L0{}".format(threshold_pca)][0]
            array_fat = dicofull["mrfdict_light_L0{}".format(threshold_pca)][1]
            keys = dicofull["mrfdict_light_L0{}".format(threshold_pca)][2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        ntimesteps_dico=array_water.shape[-1]

        if not(ntimesteps_dico==ntimesteps):
            raise ValueError("The dictionary and the incoming signal did not have the same number of timesteps: ntimesteps_dico {} != ntimesteps_signal {}".format(ntimesteps_dico,ntimesteps))


        # array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        # array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        # if not(volumes_type=="raw")or("vars_light" not in dicofull.keys()) or ((pca) and ("pca_light_{}".format(threshold_pca) not in dicofull.keys())):

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)



        del array_water
        del array_fat

        if pca:
            if not(volumes_type=="raw") or ("pca_light_{}".format(threshold_pca) not in dicofull.keys()) or (force_pca):
                pca_water = PCAComplex(n_components_=threshold_pca)
                pca_fat = PCAComplex(n_components_=threshold_pca)

                pca_water.fit(array_water_unique)
                pca_fat.fit(array_fat_unique)

                transformed_array_water_unique = pca_water.transform(array_water_unique)
                transformed_array_fat_unique = pca_fat.transform(array_fat_unique)
                if volumes_type=="raw":
                    dicofull["pca_light_{}".format(threshold_pca)] = (pca_water, pca_fat, transformed_array_water_unique, transformed_array_fat_unique)
                    with open(dicofull_file, "wb") as file:
                        pickle.dump(dicofull, file)

            else:
                print("Loading pca")
                (pca_water, pca_fat, transformed_array_water_unique, transformed_array_fat_unique)=dicofull["pca_light_{}".format(threshold_pca)]
 
        else:
            pca_water = None
            pca_fat = None
            transformed_array_water_unique = None
            transformed_array_fat_unique = None

        if not(volumes_type=="raw") or ("vars_light" not in dicofull.keys()) or (force_pca):
            var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
            var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
            sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(),
                            axis=1).real

            var_w = var_w[index_water_unique]
            var_f = var_f[index_fat_unique]

            var_w = np.reshape(var_w, (-1, 1))
            var_f = np.reshape(var_f, (-1, 1))
            sig_wf = np.reshape(sig_wf, (-1, 1))
            if volumes_type == "raw":
                dicofull["vars_light"] = (var_w, var_f, sig_wf, index_water_unique, index_fat_unique)
                with open(dicofull_file, "wb") as file:
                    pickle.dump(dicofull, file)
                
        else:
            print("Loading var w / var f / sig wf")
            (var_w, var_f, sig_wf, index_water_unique, index_fat_unique)=dicofull["vars_light"] 

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(1))
        

        print("Calculating optimal fat fraction and best pattern per signal")
        if not (self.paramDict["return_matched_signals"]):
            map_rebuilt, J_optim, phase_optim = match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                 array_water_unique, array_fat_unique,
                                                                 transformed_array_water_unique,
                                                                 transformed_array_fat_unique, var_w, var_f,
                                                                 sig_wf, pca, index_water_unique,
                                                                 index_fat_unique, remove_duplicates, verbose,split, useGPU_dictsearch, param_names=param_names
                                                                 )
        else:
            map_rebuilt, J_optim, phase_optim, matched_signals = match_signals_v2(all_signals, keys, pca_water,
                                                                                  pca_fat,
                                                                                  array_water_unique,
                                                                                  array_fat_unique,
                                                                                  transformed_array_water_unique,
                                                                                  transformed_array_fat_unique,
                                                                                  var_w, var_f, sig_wf,
                                                                                  pca, index_water_unique,
                                                                                  index_fat_unique,
                                                                                  remove_duplicates, verbose, split,
                                                                                  useGPU_dictsearch, param_names=param_names,                                                                            
                                                                                  return_matched_signals=True)



        print("Maps built")


        values_results.append((map_rebuilt, mask))



        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)), matched_signals
        else:
            return dict(zip(keys_results, values_results))





    def search_patterns_test_multi_2_steps_dico(self, dicofull_file, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        volumes_type=self.paramDict["volumes_type"]

        if "clustering" not in self.paramDict:
            self.paramDict["clustering"]=True

        split = self.paramDict["split"]
        pca = self.paramDict["pca"]

        force_pca = self.paramDict["force_pca"]

        if volumes.ndim==5:
            ntimesteps=volumes.shape[1]
        else:
            ntimesteps=volumes.shape[0]

        threshold_pca = self.paramDict["threshold_pca"]
        
        threshold_pca=np.minimum(ntimesteps,threshold_pca)

        threshold_ff=self.paramDict["threshold_ff"]
        # dictfile_light=self.paramDict["dictfile_light"]

        if "return_cost" not in self.paramDict:
            self.paramDict["return_cost"]=False
        return_cost = self.paramDict["return_cost"]

        if "calculate_matched_signals" not in self.paramDict:
            self.paramDict["calculate_matched_signals"]=False
        calculate_matched_signals = self.paramDict["calculate_matched_signals"]

        if "return_matched_signals" not in self.paramDict:
            self.paramDict["return_matched_signals"]=False
        return_matched_signals = self.paramDict["return_matched_signals"]


        if calculate_matched_signals:
            return_cost=True

        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]

        clustering_windows=self.paramDict["clustering_windows"]

        # if pca and (type(dictfile)==dict):
        #     pca_file = str.split(dictfile, ".dict")[0] + "_{}pca.pkl".format(threshold_pca)
        #     pca_file_name = str.split(pca_file, "/")[-1]

        # if type(dictfile)==str:
        #     vars_file = str.split(dictfile, ".dict")[0] + "_vars.pkl".format(threshold_pca)
        #     vars_file_name=str.split(vars_file,"/")[-1]
        #     path=str.split(os.path.realpath(__file__),"/utils_mrf.py")[0]

        # print(path)
        # print(vars_file_name)
        if volumes.ndim > 2:
            
            all_signals = volumes[:, mask > 0]
            
        else:  # already masked
            all_signals = volumes

        all_signals=all_signals.astype("complex64")
        nb_signals=all_signals.shape[1]



        del volumes

        if os.path.isfile(dicofull_file):
            with open(dicofull_file, "rb") as file:
                dicofull = pickle.load(file)
        elif type(dicofull_file) == dict:        
            dicofull = dicofull_file
        else:
            raise ValueError("dicofull_file should be a path to a existing dictionary or a dictionary")  
        
        if "param_names" in dicofull["hdr"].keys():
            param_names=dicofull["hdr"]["param_names"]

        else:
            param_names=("wT1","fT1","attB1","df")

        if volumes_type == "raw":
            
            mrfdict = dicofull["mrfdict"]
            # mrfdict.load(dictfile, force=True)

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]
            keys=np.array(keys)

            del mrfdict
        elif volumes_type=="singular":  # otherwise dictfile contains {"mrfdict":(s_w,s_f,keys),"mrfdict_light":(s_w_light,s_f_light,keys_light)}
            array_water = dicofull["mrfdict_L0{}".format(threshold_pca)][0]
            array_fat = dicofull["mrfdict_L0{}".format(threshold_pca)][1]
            keys = dicofull["mrfdict_L0{}".format(threshold_pca)][2]
            keys=np.array(keys)

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]
        
        ntimesteps_dico=array_water.shape[-1]

        if not(ntimesteps_dico==ntimesteps):
            raise ValueError("The dictionary and the incoming signal did not have the same number of timesteps: ntimesteps_dico {} != ntimesteps_signal {}".format(ntimesteps_dico,ntimesteps))



        # if not(volumes_type=="raw")or("vars" not in dicofull.keys()) or ((pca) and ("pca_{}".format(threshold_pca) not in dicofull.keys())) or (calculate_matched_signals):

            # print("Calculating unique dico signals")
        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)


        if not(volumes_type=="raw") or ("vars" not in dicofull.keys()) or (force_pca):

            var_w_total = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
            var_f_total = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
            sig_wf_total = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(),
                                  axis=1).real
            var_w_total = var_w_total[index_water_unique]
            var_f_total = var_f_total[index_fat_unique]
            var_w_total = np.reshape(var_w_total, (-1, 1))
            var_f_total = np.reshape(var_f_total, (-1, 1))
            sig_wf_total = np.reshape(sig_wf_total, (-1, 1))
            
            if volumes_type=="raw":
                dicofull["vars"]=(var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique)
                with open(dicofull_file,"wb") as file:
                    pickle.dump(dicofull,file)
        else:
            print("Loading var w / var f / sig wf")
            
            (var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique)=dicofull["vars"]

        if pca:
            if not(volumes_type=="raw") or ("pca_{}".format(threshold_pca) not in dicofull.keys()) or (force_pca):
                pca_water = PCAComplex(n_components_=threshold_pca)
                pca_fat = PCAComplex(n_components_=threshold_pca)

                pca_water.fit(array_water_unique)
                pca_fat.fit(array_fat_unique)

                transformed_array_water_unique = pca_water.transform(array_water_unique)
                transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

                if volumes_type=="raw":
                    dicofull["pca_{}".format(threshold_pca)]=(pca_water,pca_fat,transformed_array_water_unique,transformed_array_fat_unique)    
                    with open(dicofull_file,"wb") as file:
                        pickle.dump(dicofull,file)
                    
            else:
                print("Loading pca")
                (pca_water, pca_fat, transformed_array_water_unique, transformed_array_fat_unique)=dicofull["pca_{}".format(threshold_pca)] 
        else:
            pca_water = None
            pca_fat = None
            transformed_array_water_unique = None
            transformed_array_fat_unique = None

        if useGPU_dictsearch:
            var_w_total = cp.asarray(var_w_total)
            var_f_total = cp.asarray(var_f_total)
            sig_wf_total = cp.asarray(sig_wf_total)
            keys=cp.asarray(keys)

        values_results = []
        keys_results = list(range(1))

        print("Calculating optimal fat fraction and best pattern per signal")

        if self.paramDict["clustering"]:
            #Trick to avoid returning matched signals in the coarse dictionary matching step
            return_matched_signals_backup=self.paramDict["return_matched_signals"]
            self.paramDict["return_matched_signals"]=False

            print("Preliminary dictionary matching for clustering")
            all_maps_bc_cf_light = self.search_patterns_test_multi(dicofull_file,all_signals)

            self.paramDict["return_matched_signals"] = return_matched_signals_backup

            ind_high_ff = np.argwhere(all_maps_bc_cf_light[0][0]["ff"] >= threshold_ff)
            ind_low_ff = np.argwhere(all_maps_bc_cf_light[0][0]["ff"] < threshold_ff)
            all_maps_low_ff = np.array([all_maps_bc_cf_light[0][0][k][ind_low_ff] for k in list(all_maps_bc_cf_light[0][0].keys())[:-1]]).squeeze()
            all_maps_high_ff = np.array([all_maps_bc_cf_light[0][0][k][ind_high_ff] for k in
                                         list(all_maps_bc_cf_light[0][0].keys())[:-1]]).squeeze()
            unique_keys, labels = np.unique(all_maps_low_ff, axis=-1, return_inverse=True)
            #nb_clusters = unique_keys.shape[-1]
            unique_keys_high_ff, labels_high_ff = np.unique(all_maps_high_ff, axis=-1, return_inverse=True)



            idx_max_all_unique = np.zeros(nb_signals)
            alpha_optim = np.zeros(nb_signals)
            if return_cost:
                J_optim = np.zeros(nb_signals)
                phase_optim = np.zeros(nb_signals)

            if useGPU_dictsearch:
                unique_keys=cp.asarray(unique_keys)
                labels = cp.asarray(labels)
                unique_keys_high_ff = cp.asarray(unique_keys_high_ff)
                labels_high_ff = cp.asarray(labels_high_ff)

            all_signals_low_ff = all_signals[:, ind_low_ff.flatten()]
            all_signals_high_ff = all_signals[:, ind_high_ff.flatten()]
            print(type(labels))


            if return_cost:
                idx_max_all_unique_low_ff, alpha_optim_low_ff,J_optim_low_ff,phase_optim_low_ff = match_signals_v2_clustered_on_dico(all_signals_low_ff,
                                                                                                                                     keys, pca_water,
                                                                                                                                     pca_fat,
                                                                                                                                     transformed_array_water_unique,
                                                                                                                                     transformed_array_fat_unique,
                                                                                                                                     var_w_total,
                                                                                                                                     var_f_total,
                                                                                                                                     sig_wf_total,
                                                                                                                                     index_water_unique,
                                                                                                                                     index_fat_unique,
                                                                                                                                     useGPU_dictsearch,
                                                                                                                                     unique_keys, labels,
                                                                                                                                     split, param_names,clustering_windows,False,return_cost=True)

            else:
                idx_max_all_unique_low_ff,alpha_optim_low_ff=match_signals_v2_clustered_on_dico(all_signals_low_ff, keys, pca_water, pca_fat, transformed_array_water_unique,
                                                                                                transformed_array_fat_unique, var_w_total, var_f_total, sig_wf_total,
                                                                                                index_water_unique, index_fat_unique, useGPU_dictsearch, unique_keys, labels,split,param_names,clustering_windows,clustering_windows,False)



            if return_cost:
                idx_max_all_unique_high_ff, alpha_optim_high_ff,J_optim_high_ff,phase_optim_high_ff = match_signals_v2_clustered_on_dico(
                    all_signals_high_ff, keys, pca_water, pca_fat, transformed_array_water_unique,
                    transformed_array_fat_unique, var_w_total, var_f_total, sig_wf_total,
                    index_water_unique, index_fat_unique, useGPU_dictsearch, unique_keys_high_ff, split,param_names,clustering_windows, True,return_cost=True)
            else:
                idx_max_all_unique_high_ff,alpha_optim_high_ff=match_signals_v2_clustered_on_dico(all_signals_high_ff, keys, pca_water, pca_fat, transformed_array_water_unique,
                                                                                                  transformed_array_fat_unique, var_w_total, var_f_total, sig_wf_total,
                                                                                                  index_water_unique, index_fat_unique, useGPU_dictsearch, unique_keys_high_ff, labels_high_ff,split,param_names,clustering_windows,True)



            idx_max_all_unique[ind_low_ff.flatten()] = idx_max_all_unique_low_ff
            idx_max_all_unique[ind_high_ff.flatten()] = idx_max_all_unique_high_ff

            alpha_optim[ind_low_ff.flatten()] = alpha_optim_low_ff
            alpha_optim[ind_high_ff.flatten()] = alpha_optim_high_ff

            if return_cost:
                J_optim[ind_low_ff.flatten()] = J_optim_low_ff
                J_optim[ind_high_ff.flatten()] = J_optim_high_ff

                phase_optim[ind_low_ff.flatten()] = phase_optim_low_ff
                phase_optim[ind_high_ff.flatten()] = phase_optim_high_ff
                matched_signals = array_water_unique[index_water_unique, :][idx_max_all_unique.astype(int), :].T * (
                        1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][
                                                                    idx_max_all_unique.astype(int),
                                                                    :].T * np.array(alpha_optim).reshape(1, -1)
                rho_optim= J_optim*np.linalg.norm(all_signals,axis=0)/np.linalg.norm(matched_signals, axis=0)

            if calculate_matched_signals:
                matched_signals=array_water_unique[index_water_unique, :][idx_max_all_unique.astype(int), :].T * (1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][idx_max_all_unique.astype(int), :].T * np.array(alpha_optim).reshape(1, -1)
                matched_signals *=np.linalg.norm(all_signals,axis=0)/np.linalg.norm(matched_signals, axis=0)
                matched_signals *= J_optim * np.exp(1j * phase_optim)



            if useGPU_dictsearch:
                keys=keys.get()

            keys_for_map = [tuple(k) for k in keys]

            params_all_unique = np.array(
                [keys_for_map[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique.astype(int))])
            map_rebuilt=dict(zip(param_names, np.transpose(params_all_unique[:,:-1])))
            map_rebuilt["ff"]=params_all_unique[:,-1]
            # map_rebuilt = {
            #     "wT1": params_all_unique[:, 0],
            #     "fT1": params_all_unique[:, 1],
            #     "attB1": params_all_unique[:, 2],
            #     "df": params_all_unique[:, 3],
            #     "ff": params_all_unique[:, 4]

            # }
            if return_cost:
                if not(return_matched_signals):
                    values_results.append((map_rebuilt, mask,J_optim,phase_optim,rho_optim))
                else:
                    values_results.append((map_rebuilt, mask,J_optim,phase_optim,rho_optim,matched_signals))
            else:
                values_results.append((map_rebuilt, mask))

        else:
            #Trick to avoid returning matched signals in the coarse dictionary matching step
            return_matched_signals_backup=self.paramDict["return_matched_signals"]



            if calculate_matched_signals:
                all_maps,matched_signals = self.search_patterns_test_multi(dicofull_file,all_signals)

            else:
                all_maps = self.search_patterns_test_multi(dicofull_file,all_signals)

            map_rebuilt=all_maps[0][0]
            mask=all_maps[0][1]

            if return_cost:
                if not(return_matched_signals):
                    values_results.append((map_rebuilt, mask,None,None))
                else:
                    values_results.append((map_rebuilt, mask,None,None,matched_signals))
            else:
                values_results.append((map_rebuilt, mask))

        print("Maps built")

        return dict(zip(keys_results, values_results))


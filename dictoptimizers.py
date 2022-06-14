import numpy as np
from scipy import ndimage
from scipy.ndimage import affine_transform
from utils_mrf import translation_breathing,build_mask_single_image,build_mask,simulate_radial_undersampled_images,read_mrf_dict,create_cuda_context,correct_mvt_kdata
from Transformers import PCAComplex
from mutools.optim.dictsearch import dictsearch
from tqdm import tqdm
from mrfsim import makevol
from image_series import MapFromDict
from datetime import datetime
from skimage.restoration import denoise_tv_chambolle
#import cupy as cp
try:
    #from pycuda.autoinit import _finish_up
    import cupy as cp
except:
    print("Could not import cupy")
    pass

try :
    from SPIJN import *
except:
    print("Could not import SPIJN")
    pass

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from numba import cuda
import gc
import pickle

class GaussianWeighting(object):
    def __init__(self, sig=np.pi/2):
        self.sig=sig

    def apply(self,traj):
        return np.exp(-np.linalg.norm(traj,axis=-1)**2/(2*self.sig**2))


def match_signals(all_signals,keys,pca_water,pca_fat,array_water_unique,array_fat_unique,transformed_array_water_unique,transformed_array_fat_unique,var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates,verbose,niter,split,useGPU_dictsearch,mask,tv_denoising_weight,log_phase=False):

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

    idx_max_all_unique = []
    alpha_optim = []

    if niter > 0:
        phase_optim = []
        J_optim = []

    elif log_phase:
        phase_optim = []

    for j in tqdm(range(num_group)):
        j_signal = j * split
        j_signal_next = np.minimum((j + 1) * split, nb_signals)

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

        current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
        current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]

        # current_sig_ws = current_sig_ws_for_phase.real
        # current_sig_fs = current_sig_fs_for_phase.real

        if verbose:
            end = datetime.now()
            print(end - start)

        if not (useGPU_dictsearch):
            # if adj_phase:
            if verbose:
                print("Adjusting Phase")
                print("Calculating alpha optim and flooring")

                ### Testing direct phase solving
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

            # current_alpha_all_unique_2 = (1 * (alpha2 >= 0) & (alpha2 <= 1)) * alpha2 + (
            #            1 - (1*(alpha2 >= 0) & (alpha2 <= 1))) * alpha1

            del alpha1
            del alpha2

            if verbose:
                start = datetime.now()



            #current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)

            apha_more_0=(current_alpha_all_unique>=0)
            alpha_less_1=(current_alpha_all_unique<=1)
            alpha_out_bounds=(1*(apha_more_0))*(1*(alpha_less_1))==0


            # phase_adj=np.angle((1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)



            d_oobounds_0 = current_sig_ws_for_phase[alpha_out_bounds]
            phase_adj_0 = -np.arctan(d_oobounds_0.imag / d_oobounds_0.real)
            cond = np.sin(phase_adj_0) * d_oobounds_0.imag - np.cos(phase_adj_0) * d_oobounds_0.real
            del d_oobounds_0

            phase_adj_0 = (phase_adj_0) * (
                    1 * (cond) <= 0) + (phase_adj_0 + np.pi) * (
                                1 * (cond) > 0)

            del cond

            current_sig_ws_0 = (current_sig_ws_for_phase[alpha_out_bounds] * np.exp(1j * phase_adj_0)).real
            J_0=current_sig_ws_0/np.sqrt(np.squeeze(var_w[np.argwhere(alpha_out_bounds)[:,0]]))

            d_oobounds_1 = current_sig_fs_for_phase[alpha_out_bounds]
            phase_adj_1 = -np.arctan(d_oobounds_1.imag / d_oobounds_1.real)
            cond = np.sin(phase_adj_1) * d_oobounds_1.imag - np.cos(phase_adj_1) * d_oobounds_1.real
            del d_oobounds_1

            phase_adj_1 = (phase_adj_1) * (
                    1 * (cond) <= 0) + (phase_adj_1 + np.pi) * (
                                  1 * (cond) > 0)

            del cond

            current_sig_fs_1 = (current_sig_fs_for_phase[alpha_out_bounds] * np.exp(1j * phase_adj_1)).real
            J_1 = current_sig_fs_1 / np.sqrt(np.squeeze(var_f[np.argwhere(alpha_out_bounds)[:,0]]))

            current_alpha_all_unique[alpha_out_bounds]=np.argmax(np.concatenate([J_0[:, None], J_1[:, None]], axis=-1), axis=-1).astype("float")


            d = (
                        1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase

            phase_adj = -np.arctan(d.imag / d.real)
            cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real
            del d

            phase_adj = (phase_adj) * (
                    1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                1 * (cond) > 0)

            del cond

            ##############################################################################################################################
            def J_alpha_pixel(alpha,phi, i, j):

                current_sig_ws = (current_sig_ws_for_phase[i,j] * np.exp(1j * phi)).real
                current_sig_fs = (current_sig_fs_for_phase[i,j] * np.exp(1j * phi)).real
                return ((
                         1 - alpha) * current_sig_ws + alpha * current_sig_fs) / np.sqrt(
                    (
                            1 - alpha) ** 2 * var_w[i] + alpha ** 2 * var_f[i] + 2 * alpha * (
                            1 - alpha) * sig_wf[i])

            phi = np.arange(-np.pi,np.pi,np.pi/20)
            alpha = np.arange(0.,1.01,0.01)
            alphav_np, phiv_np = np.meshgrid(alpha, phi, sparse=False, indexing='ij')

            i=0
            j=0

            s,t=current_sig_ws_for_phase.shape
            n,m = alphav_np.shape
            result_np=np.zeros(alphav_np.shape)


            i,j=np.unravel_index(np.random.choice(np.arange(s*t)),(s,t))

            for p in tqdm(range(n)):
                for q in range(m):
                    result_np[p,q]=J_alpha_pixel(alphav_np[p,q],phiv_np[p,q],i,j)


            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_surface(alphav_np, phiv_np, result_np,alpha=0.5)



            index_min_p,index_min_q = np.unravel_index(np.argmax(result_np), result_np.shape)
            alpha_min = alphav_np[index_min_p,index_min_q]
            phi_min = phiv_np[index_min_p, index_min_q]
            result_min = result_np[index_min_p, index_min_q]

            # alpha_ = (1 * (alpha1[i, j] >= 0) & (alpha1[i, j] <= 1)) * alpha1[i, j] + (
            #             1 - (1 * (alpha1[i, j] >= 0) & (alpha1[i, j] <= 1))) * alpha2[i, j]

            print("Max alpha on surface : {}".format(np.round(alpha_min,2)))
            #print("Alpha 1 : {}".format(np.round(alpha1[i,j],2)))
            #print("Alpha 2 : {}".format(np.round(alpha2[i,j],2)))
            print("Alpha calc : {}".format(np.round(current_alpha_all_unique[i,j], 2)))

            # phi_calc = -np.angle((
            #                               1 - alpha_) * current_sig_ws_for_phase[i, j] + alpha_ * current_sig_fs_for_phase[i, j])
            #
            # d = (1 - alpha_) * current_sig_ws_for_phase[i, j] + alpha_ * \
            #      current_sig_fs_for_phase[i, j]
            # phi_form = -np.arctan(d.imag / d.real)
            # phi_form = (phi_form) * (
            #             1 * (np.sin(phi_form) * d.imag - np.cos(phi_form) * d.real) <= 0) + (
            #                  np.mod(phi_form + np.pi, 2 * np.pi)) * (
            #                          1 * (np.sin(phi_form) * d.imag - np.cos(phi_form) * d.real) > 0)

            # phi_calc1 = -np.angle((
            #                              1 - alpha1[i,j]) * current_sig_ws_for_phase[i,j] + alpha1[i,j] * current_sig_fs_for_phase[i,j])
            # phi_calc2 = -np.angle((
            #                              1 - alpha2[i, j]) * current_sig_ws_for_phase[i, j] +  alpha2[i, j] *
            #                      current_sig_fs_for_phase[i, j])
            #
            # d1 = (1 - alpha1[i,j]) * current_sig_ws_for_phase[i,j] + alpha1[i,j] * current_sig_fs_for_phase[i,j]
            # phi_form_1 = -np.arctan(d1.imag/d1.real)
            # d2 = (1 - alpha2[i, j]) * current_sig_ws_for_phase[i, j] + alpha2[i, j] * current_sig_fs_for_phase[
            #     i, j]
            # phi_form_2 = -np.arctan(d2.imag/d2.real)
            #
            # phi_form_1 = (phi_form_1)*(1*(np.sin(phi_form_1)*d1.imag-np.cos(phi_form_1)*d1.real)<=0)+(np.mod(phi_form_1+np.pi,2*np.pi))*(1*(np.sin(phi_form_1)*d1.imag-np.cos(phi_form_1)*d1.real)>0)
            # phi_form_2 = (phi_form_2) * (
            #             1 * (np.sin(phi_form_2) * d2.imag - np.cos(phi_form_2) * d2.real) <= 0) + (
            #                  np.mod(phi_form_2 + np.pi, 2 * np.pi)) * (
            #                          1 * (np.sin(phi_form_2) * d2.imag - np.cos(phi_form_2) * d2.real) > 0)

            print("Max phi on surface : {}".format(np.round(phi_min, 2)))
            # print("Phi Ideal 1 : {}".format(np.round(phi_calc1, 2)))
            # print("Phi Ideal 2 : {}".format(np.round(phi_calc2, 2)))
            # print("Phi Formula 1 : {}".format(np.round(phi_form_1, 2)))
            # print("Phi Formula 2 : {}".format(np.round(phi_form_2, 2)))
            print("Phi optim: {}".format(np.round(phase_adj[i,j], 2)))

            print("Max correl on surface {}".format(np.round(result_min,2)))
            print("Retrieved correl on surface {}".format(np.round( J_alpha_pixel(current_alpha_all_unique[i,j], phase_adj[i,j], i, j)[0],2)))

            ax.plot(alpha_min,phi_min,result_min,marker="x")
            ax.plot(current_alpha_all_unique[i,j], phase_adj[i,j], J_alpha_pixel(current_alpha_all_unique[i,j], phase_adj[i,j], i, j)[0], marker="o")
            ax.set_title("Signal {},{}".format(i,j))
            # ax.plot(alpha1[i,j], phi_form_1, J_alpha_pixel(alpha1[i,j],phi_form_1,i,j)[0], marker="o")
            # ax.plot(alpha2[i,j], phi_form_2,
            #         J_alpha_pixel(alpha2[i, j], phi_form_2, i, j)[0], marker="o")
            #################################################################################################################################""""

            if verbose:
                end = datetime.now()
                print(end - start)

            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            if verbose:
                print("Calculating cost for all signals")
            start = datetime.now()

            current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
            current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real

            J_all = ((
                             1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                (
                        1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                        1 - current_alpha_all_unique) * sig_wf)

            end = datetime.now()

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

            current_alpha_all_unique = cp.minimum(cp.maximum(current_alpha_all_unique, 0.0), 1.0)

            # phase_adj = np.angle((
            #                                 1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)

            d = (
                        1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase
            phase_adj = -cp.arctan(d.imag / d.real)
            cond = cp.sin(phase_adj) * d.imag - cp.cos(phase_adj) * d.real <= 0

            del d

            phase_adj = phase_adj * (1 * (cond)) + (phase_adj + np.pi) * (1 * (1 - cond))

            del cond

            if verbose:
                end = datetime.now()
                print(end - start)

            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            if verbose:
                print("Calculating cost for all signals")
                start = datetime.now()

            current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
            current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real

            # del phase_adj
            del current_sig_ws_for_phase
            del current_sig_fs_for_phase

            J_all = ((
                             1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                (
                        1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                        1 - current_alpha_all_unique) * sig_wf)

            J_all = J_all.get()
            current_alpha_all_unique = current_alpha_all_unique.get()

            if niter > 0 or log_phase:
                phase_adj=phase_adj.get()


            del current_sig_fs
            del current_sig_ws

            if verbose:
                end = datetime.now()
                print(end - start)

        if verbose:
            print("Extracting index of pattern with max correl")
            start = datetime.now()

        idx_max_all_current = np.argmax(J_all, axis=0)
        # check_max_correl=np.max(J_all,axis=0)

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Filling the lists with results for this loop")
            start = datetime.now()

        idx_max_all_unique.extend(idx_max_all_current)
        alpha_optim.extend(current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])])

        if niter > 0:
            phase_optim.extend(phase_adj[idx_max_all_current, np.arange(J_all.shape[1])])
            J_optim.extend(J_all[idx_max_all_current, np.arange(J_all.shape[1])])

        elif log_phase:
            phase_optim.extend(phase_adj[idx_max_all_current, np.arange(J_all.shape[1])])

        del phase_adj

        if verbose:
            end = datetime.now()
            print(end - start)

    # idx_max_all_unique = np.argmax(J_all, axis=0)
    del J_all
    del current_alpha_all_unique

    if niter > 0:
        phase_optim = np.array(phase_optim)
        J_optim = np.array(J_optim)
    elif log_phase:
        phase_optim = np.array(phase_optim)



    # del sig_ws_all_unique
    # del sig_fs_all_unique

    params_all_unique = np.array(
        [keys[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique)])

    if remove_duplicates:
        params_all = params_all_unique[index_signals_unique]
    else:
        params_all = params_all_unique

    del params_all_unique



    map_rebuilt = {
        "wT1": params_all[:, 0],
        "fT1": params_all[:, 1],
        "attB1": params_all[:, 2],
        "df": params_all[:, 3],
        "ff": params_all[:, 4]

    }

    if tv_denoising_weight is not None:
        for i,k in enumerate(map_rebuilt.keys()):
            map_rebuilt[k]=denoise_tv_chambolle(makevol(map_rebuilt[k],mask>0),weight=tv_denoising_weight)[mask>0]
            if k!="ff":#Projection back on dictionary parameters
                print("Projection back to dictionary values for {} after denoising".format(k))
                curr_values = np.unique(np.array(keys)[:,i])
                map_rebuilt[k]=curr_values[np.argmin(np.abs(map_rebuilt[k].reshape(-1, 1) - curr_values), axis=-1)]



    if niter==0:
        if not(log_phase):
            return map_rebuilt,None,None
        else:
            return map_rebuilt, None, phase_optim
    else:
        return map_rebuilt,J_optim,phase_optim


class Optimizer(object):

    def __init__(self,log=False,mask=None,verbose=False,useGPU=False,**kwargs):
        self.paramDict=kwargs
        self.paramDict["log"]=log
        self.paramDict["useGPU"]=useGPU
        self.mask=mask
        self.verbose=verbose


    def search_patterns(self,dictfile,volumes,retained_timesteps=None):
        #takes as input dictionary pattern and an array of images or volumes and outputs parametric maps
        raise ValueError("search_patterns should be implemented in child")

class SimpleDictSearch(Optimizer):

    def __init__(self,niter=0,seq=None,trajectory=None,split=500,pca=True,threshold_pca=15,useGPU_dictsearch=False,useGPU_simulation=True,movement_correction=False,cond=None,remove_duplicate_signals=False,threshold=None,tv_denoising_weight=None,log_phase=False,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        super().__init__(**kwargs)
        self.paramDict["niter"]=niter
        self.paramDict["split"] = split
        self.paramDict["pca"] = pca
        self.paramDict["threshold_pca"] = threshold_pca
        self.paramDict["remove_duplicate_signals"] = remove_duplicate_signals
        #self.paramDict["useAdjPred"]=useAdjPred

        if niter>0:
            if seq is None:
                raise ValueError("When more than 0 iteration, one needs to supply a sequence in order to resimulate the image series")
            else:
                self.paramDict["sequence"]=seq
            if trajectory is None:
                raise ValueError("When more than 0 iteration, one needs to supply a kspace trajectory in order to resimulate the image series")
            else:
                self.paramDict["trajectory"]=trajectory

        self.paramDict["useGPU_dictsearch"]=useGPU_dictsearch
        self.paramDict["useGPU_simulation"] = useGPU_simulation
        self.paramDict["movement_correction"]=movement_correction
        self.paramDict["cond"]=cond
        self.paramDict["threshold"]=threshold
        self.paramDict["tv_denoising_weight"] = tv_denoising_weight
        self.paramDict["log_phase"] = log_phase

    def search_patterns(self,dictfile,volumes,retained_timesteps=None):


        if self.mask is None:
            mask = build_mask(volumes)
        else:
            mask = self.mask


        verbose=self.verbose
        niter=self.paramDict["niter"]
        split=self.paramDict["split"]
        pca=self.paramDict["pca"]
        threshold_pca=self.paramDict["threshold_pca"]
        #useAdjPred=self.paramDict["useAdjPred"]
        if niter>0:
            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode=self.paramDict["gen_mode"]
        log=self.paramDict["log"]
        useGPU_dictsearch=self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction=self.paramDict["movement_correction"]
        cond_mvt=self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        threshold = self.paramDict["threshold"]
        #adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        signals = volumes[:, mask > 0]

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/volumes0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, volumes)


        del volumes

        if niter > 0:
            signals0 = signals

        #norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        norm_signals=np.linalg.norm(signals, 2, axis=0)
        all_signals = signals/norm_signals

        mrfdict = dictsearch.Dictionary()
        mrfdict.load(dictfile, force=True)

        keys = mrfdict.keys
        array_water = mrfdict.values[:, :, 0]
        array_fat = mrfdict.values[:, :, 1]

        del mrfdict

        if retained_timesteps is not None:
            array_water=array_water[:,retained_timesteps]
            array_fat=array_fat[:,retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique)
            pca_fat.fit(array_fat_unique)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

        var_w = var_w[index_water_unique]
        var_f = var_f[index_fat_unique]

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):



            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))

            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            nb_signals = all_signals.shape[1]

            if remove_duplicates:
                all_signals, index_signals_unique = np.unique(all_signals, axis=1, return_inverse=True)
                nb_signals = all_signals.shape[1]



            print("There are {} unique signals to match along {} water and {} fat components".format(nb_signals,array_water_unique.shape[0],array_fat_unique.shape[0]))



            num_group = int(nb_signals / split) + 1

            idx_max_all_unique = []
            alpha_optim = []

            if niter>0:
                phase_optim=[]
                J_optim=[]

            for j in tqdm(range(num_group)):
                j_signal = j * split
                j_signal_next = np.minimum((j + 1) * split, nb_signals)

                if self.verbose:
                    print("PCA transform")
                    start = datetime.now()

                if not(useGPU_dictsearch):

                    if pca:
                        transformed_all_signals_water = np.transpose(pca_water.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))
                        transformed_all_signals_fat = np.transpose(pca_fat.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))

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
                        transformed_all_signals_fat = cp.transpose(pca_fat.transform(cp.transpose(cp.asarray(all_signals[:, j_signal:j_signal_next])))).get()

                        sig_ws_all_unique = (cp.matmul(cp.asarray(transformed_array_water_unique),
                                                      cp.asarray(transformed_all_signals_water).conj())).get()
                        sig_fs_all_unique = (cp.matmul(cp.asarray(transformed_array_fat_unique),
                                                      cp.asarray(transformed_all_signals_fat).conj())).get()
                    else:

                        sig_ws_all_unique = (cp.matmul(cp.asarray(array_water_unique),
                                                      cp.asarray(all_signals)[:, j_signal:j_signal_next].conj())).get()
                        sig_fs_all_unique = (cp.matmul(cp.asarray(array_fat_unique),
                                                      cp.asarray(all_signals)[:, j_signal:j_signal_next].conj())).get()


                if self.verbose:
                    end = datetime.now()
                    print(end - start)

                if self.verbose:
                    print("Extracting all sig_ws and sig_fs")
                    start = datetime.now()



                current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
                current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]

                #current_sig_ws = current_sig_ws_for_phase.real
                #current_sig_fs = current_sig_fs_for_phase.real

                if self.verbose:
                    end = datetime.now()
                    print(end-start)

                if not(useGPU_dictsearch):
                    #if adj_phase:
                    if self.verbose:
                        print("Adjusting Phase")
                        print("Calculating alpha optim and flooring")

                        ### Testing direct phase solving
                    A = sig_wf*current_sig_ws_for_phase-var_w*current_sig_fs_for_phase
                    B = (current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase

                    a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
                    b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                    c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

                    discr = b**2-4*a*c
                    alpha1 = (-b + np.sqrt(discr)) / (2 * a)
                    alpha2 = (-b - np.sqrt(discr)) / (2 * a)


                    del a
                    del b
                    del c
                    del discr

                    current_alpha_all_unique=(1 * (alpha1 >= 0) & (alpha1 <= 1)) * alpha1 + (1 - (1*(alpha1 >= 0) & (alpha1 <= 1))) * alpha2

                        #current_alpha_all_unique_2 = (1 * (alpha2 >= 0) & (alpha2 <= 1)) * alpha2 + (
                        #            1 - (1*(alpha2 >= 0) & (alpha2 <= 1))) * alpha1

                    del alpha1
                    del alpha2

                    if self.verbose:
                        start = datetime.now()

                    current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)

                        #phase_adj=np.angle((1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)


                    d = (1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase
                    phase_adj = -np.arctan(d.imag /d.real)
                    cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real
                    del d

                    phase_adj = (phase_adj) * (
                                    1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                                 1 * (cond) > 0)

                    del cond

                    if self.verbose:
                        end=datetime.now()
                        print(end-start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    if self.verbose:
                        print("Calculating cost for all signals")
                    start = datetime.now()

                    current_sig_ws = (current_sig_ws_for_phase*np.exp(1j*phase_adj)).real
                    current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real


                    J_all = ((
                                     1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                        (
                                1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                                1 - current_alpha_all_unique) * sig_wf)
                    end = datetime.now()

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
                    b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag +B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                    c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

                    del A
                    del B

                    #del beta
                    #del delta
                    #del gamma
                    #del nu

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

                    current_alpha_all_unique = cp.minimum(cp.maximum(current_alpha_all_unique, 0.0), 1.0)

                    #phase_adj = np.angle((
                    #                                 1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)


                    d = (1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase
                    phase_adj=-cp.arctan(d.imag / d.real)
                    cond = cp.sin(phase_adj)*d.imag-cp.cos(phase_adj)*d.real<=0

                    del d

                    phase_adj=phase_adj*(1*(cond))+(phase_adj+np.pi)*(1*(1-cond))

                    del cond


                    if verbose:
                        end = datetime.now()
                        print(end - start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    if verbose:
                        print("Calculating cost for all signals")
                        start = datetime.now()

                    current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
                    current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real



                    #del phase_adj
                    del current_sig_ws_for_phase
                    del current_sig_fs_for_phase

                    J_all = ((
                                     1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                        (
                                1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                                1 - current_alpha_all_unique) * sig_wf)

                    J_all = J_all.get()
                    current_alpha_all_unique=current_alpha_all_unique.get()

                    del current_sig_fs
                    del current_sig_ws


                    if verbose:
                        end = datetime.now()
                        print(end - start)


                if verbose:
                    print("Extracting index of pattern with max correl")
                    start = datetime.now()

                idx_max_all_current = np.argmax(J_all, axis=0)
                #check_max_correl=np.max(J_all,axis=0)

                if verbose:
                    end = datetime.now()
                    print(end-start)

                if verbose:
                    print("Filling the lists with results for this loop")
                    start = datetime.now()

                idx_max_all_unique.extend(idx_max_all_current)
                alpha_optim.extend(current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])])

                if niter>0:
                    phase_optim.extend(phase_adj[idx_max_all_current, np.arange(J_all.shape[1])])
                    J_optim.extend(J_all[idx_max_all_current, np.arange(J_all.shape[1])])

                del phase_adj

                if verbose:
                    end = datetime.now()
                    print(end - start)

            # idx_max_all_unique = np.argmax(J_all, axis=0)
            del J_all
            del current_alpha_all_unique

            if niter>0:
                phase_optim=np.array(phase_optim)
                J_optim = np.array(J_optim)

            print("Building the maps for iteration {}".format(i))

            # del sig_ws_all_unique
            # del sig_fs_all_unique

            params_all_unique = np.array(
                [keys[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique)])

            if remove_duplicates:
                params_all = params_all_unique[index_signals_unique]
            else:
                params_all = params_all_unique

            del params_all_unique

            map_rebuilt = {
                "wT1": params_all[:, 0],
                "fT1": params_all[:, 1],
                "attB1": params_all[:, 2],
                "df": params_all[:, 3],
                "ff": params_all[:, 4]

            }

            values_results.append((map_rebuilt, mask))

            if log:
                with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i):(map_rebuilt, mask)}, f)

            if useGPU_dictsearch:#Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()

                cp.cuda.set_allocator(None)
                # Disable memory pool for pinned memory (CPU).
                cp.cuda.set_pinned_memory_allocator(None)
                gc.collect()




            if i == niter:
                break

            print("Generating prediction volumes and undersampled images for iteration {}".format(i))
            # generate prediction volumes
            keys_simu = list(map_rebuilt.keys())
            values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
            map_for_sim = dict(zip(keys_simu, values_simu))

            # predict spokes
            images_pred = MapFromDict("RebuiltMapFromParams", paramMap=map_for_sim, rounding=True,gen_mode=gen_mode)
            images_pred.buildParamMap()

            del map_for_sim

            del keys_simu
            del values_simu

            nspoke = int(trajectory.paramDict["total_nspokes"] / self.paramDict["ntimesteps"])

            images_pred.build_ref_images(seq,norm=J_optim*norm_signals,phase=phase_optim)

            print("Normalizing image series")


            print("Filling images series with renormalized signals")

            kdatai = images_pred.generate_kdata(trajectory,useGPU=useGPU_simulation)

            if log:
                print("Saving Ideal Volumes for iteration {}".format(i))
                with open('./log/predvolumes_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, images_pred.images_series[::nspoke].astype(np.complex64))

            del images_pred

            if movement_correction:
                traj=trajectory.get_traj()
                kdatai, traj_retained_final, _ = correct_mvt_kdata(kdatai, trajectory, cond_mvt,self.paramDict["ntimesteps"],density_adj=True)

            kdatai = np.array(kdatai)
            #nans = np.nonzero(np.isnan(kdatai))
            nans = [np.nonzero(np.isnan(k))[0] for k in kdatai]
            nans_count = np.array([len(n) for n in nans]).sum()

            if nans_count>0:
                print("Warning : Nan Values replaced by zeros in rebuilt kdata")
                for i,k in enumerate(kdatai):
                    kdatai[i][nans[i]]=0.0

            if not(movement_correction):
                volumesi = simulate_radial_undersampled_images(kdatai,trajectory,mask.shape,useGPU=useGPU_simulation,density_adj=True)

            else:
                trajectory.traj_for_reconstruction=traj_retained_final
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, mask.shape,
                                                               useGPU=useGPU_simulation, density_adj=True,is_theta_z_adjusted=True)

            #volumesi/=(2*np.pi)

            nans_volumes = np.argwhere(np.isnan(volumesi))
            if len(nans_volumes) > 0:
                np.save('./log/kdatai.npy', kdatai)
                np.save('./log/volumesi.npy',volumesi)
                raise ValueError("Error : Nan Values in volumes")

            del kdatai

            #volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)


            if log:
                print("Saving correction volumes for iteration {}".format(i))
                with open('./log/volumes1_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(volumesi))

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:,mask>0]
            #normi= np.linalg.norm(signalsi, axis=0)
            #signalsi *= norm_signals/normi

            del volumesi


            signals += signals0 - signalsi

            if threshold is not None:
                signals_for_matching = signals * (1-(np.abs(signals)>threshold))
            else:
                signals_for_matching = signals

            norm_signals = np.linalg.norm(signals_for_matching,axis=0)


            all_signals =signals_for_matching/norm_signals



        if log:
            print(date_time)

        return dict(zip(keys_results, values_results))




    def search_patterns_MRF_denoising(self, dictfile, volumes, retained_timesteps=None):

        gw = self.paramDict["Weighting"]
        sig_0=gw.sig

        if self.mask is None:
            mask = build_mask(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        # adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        signals = volumes[:, mask > 0]

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/volumes0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, volumes)

        del volumes

        if niter > 0:
            signals0 = signals

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        norm_signals = np.linalg.norm(signals, 2, axis=0)
        all_signals = signals / norm_signals

        mrfdict = dictsearch.Dictionary()
        mrfdict.load(dictfile, force=True)

        keys = mrfdict.keys
        array_water = mrfdict.values[:, :, 0]
        array_fat = mrfdict.values[:, :, 1]

        del mrfdict

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique)
            pca_fat.fit(array_fat_unique)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

        var_w = var_w[index_water_unique]
        var_f = var_f[index_fat_unique]

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))

            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
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

            idx_max_all_unique = []
            alpha_optim = []

            if niter > 0:
                phase_optim = []
                J_optim=[]

            for j in tqdm(range(num_group)):
                j_signal = j * split
                j_signal_next = np.minimum((j + 1) * split, nb_signals)

                if self.verbose:
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

                if self.verbose:
                    end = datetime.now()
                    print(end - start)

                if self.verbose:
                    print("Extracting all sig_ws and sig_fs")
                    start = datetime.now()

                current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
                current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]


                if self.verbose:
                    end = datetime.now()
                    print(end - start)

                if not (useGPU_dictsearch):
                    # if adj_phase:
                    if self.verbose:
                        print("Adjusting Phase")
                        print("Calculating alpha optim and flooring")

                    A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
                    B = (
                                    current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase

                    a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
                    b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                    c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

                    # a = beta + delta
                    # b = gamma-delta+nu
                    # c=-gamma

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

                    if self.verbose:
                        start = datetime.now()

                    current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)

                    d = (
                                    1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase
                    phase_adj = -np.arctan(d.imag / d.real)
                    cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real
                    del d

                    phase_adj = (phase_adj) * (
                            1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                        1 * (cond) > 0)

                    del cond

                    if self.verbose:
                        end = datetime.now()
                        print(end - start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    if self.verbose:
                        print("Calculating cost for all signals")
                    start = datetime.now()

                    current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
                    current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real


                    J_all = ((
                                     1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                        (
                                1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                                1 - current_alpha_all_unique) * sig_wf)
                    end = datetime.now()

                else:
                    if verbose:
                        print("Calculating alpha optim and flooring")
                        start = datetime.now()

                    current_sig_ws_for_phase = cp.asarray(current_sig_ws_for_phase)
                    current_sig_fs_for_phase = cp.asarray(current_sig_fs_for_phase)


                    A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
                    B = (
                                current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase


                    a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
                    b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                    c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

                    del A
                    del B


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

                    current_alpha_all_unique = cp.minimum(cp.maximum(current_alpha_all_unique, 0.0), 1.0)

                    # phase_adj = np.angle((
                    #                                 1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)

                    d = (
                                    1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase
                    phase_adj = -cp.arctan(d.imag / d.real)
                    cond = cp.sin(phase_adj) * d.imag - cp.cos(phase_adj) * d.real <= 0

                    del d

                    phase_adj = phase_adj * (1 * (cond)) + (phase_adj + np.pi) * (1 * (1 - cond))

                    del cond

                    if verbose:
                        end = datetime.now()
                        print(end - start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    if verbose:
                        print("Calculating cost for all signals")
                        start = datetime.now()

                    current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
                    current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real

                    # del phase_adj
                    del current_sig_ws_for_phase
                    del current_sig_fs_for_phase

                    J_all = ((
                                     1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                        (
                                1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                                1 - current_alpha_all_unique) * sig_wf)

                    J_all = J_all.get()
                    current_alpha_all_unique = current_alpha_all_unique.get()

                    del current_sig_fs
                    del current_sig_ws

                    if verbose:
                        end = datetime.now()
                        print(end - start)

                if verbose:
                    print("Extracting index of pattern with max correl")
                    start = datetime.now()

                idx_max_all_current = np.argmax(J_all, axis=0)
                # check_max_correl=np.max(J_all,axis=0)

                if verbose:
                    end = datetime.now()
                    print(end - start)

                if verbose:
                    print("Filling the lists with results for this loop")
                    start = datetime.now()

                idx_max_all_unique.extend(idx_max_all_current)
                alpha_optim.extend(current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])])

                if niter > 0:
                    phase_optim.extend(phase_adj[idx_max_all_current, np.arange(J_all.shape[1])])
                    J_optim.extend(J_all[idx_max_all_current, np.arange(J_all.shape[1])])
                del phase_adj

                if verbose:
                    end = datetime.now()
                    print(end - start)

            # idx_max_all_unique = np.argmax(J_all, axis=0)
            del J_all
            del current_alpha_all_unique

            if niter > 0:
                phase_optim = np.array(phase_optim)
                J_optim = np.array(J_optim)
            print("Building the maps for iteration {}".format(i))

            # del sig_ws_all_unique
            # del sig_fs_all_unique

            params_all_unique = np.array(
                [keys[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique)])

            if remove_duplicates:
                params_all = params_all_unique[index_signals_unique]
            else:
                params_all = params_all_unique

            del params_all_unique

            map_rebuilt = {
                "wT1": params_all[:, 0],
                "fT1": params_all[:, 1],
                "attB1": params_all[:, 2],
                "df": params_all[:, 3],
                "ff": params_all[:, 4]

            }

            values_results.append((map_rebuilt, mask))

            if log:
                with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i): (map_rebuilt, mask)}, f)

            if useGPU_dictsearch:  # Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()

                cp.cuda.set_allocator(None)
                # Disable memory pool for pinned memory (CPU).
                cp.cuda.set_pinned_memory_allocator(None)
                gc.collect()

            if i == niter:
                break

            print("Generating prediction volumes and undersampled images for iteration {}".format(i))
            # generate prediction volumes
            keys_simu = list(map_rebuilt.keys())
            values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
            map_for_sim = dict(zip(keys_simu, values_simu))

            # predict spokes
            images_pred = MapFromDict("RebuiltMapFromParams", paramMap=map_for_sim, rounding=True, gen_mode=gen_mode)
            images_pred.buildParamMap()

            del map_for_sim

            del keys_simu
            del values_simu

            nspoke = int(trajectory.paramDict["total_nspokes"] / self.paramDict["ntimesteps"])

            images_pred.build_ref_images(seq, norm=norm_signals*J_optim,phase=np.array(phase_optim))

            kdatai = images_pred.generate_kdata(trajectory, useGPU=useGPU_simulation)

            gw.sig=gw.sig*(np.pi/sig_0)**(1/niter)
            traj=trajectory.get_traj()
            kdatai = [gw.apply(traj[j])*k for j,k in enumerate(kdatai)]

            if log:
                print("Saving Ideal Volumes for iteration {}".format(i))
                with open('./log/predvolumes_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, images_pred.images_series[::nspoke].astype(np.complex64))

            del images_pred

            if movement_correction:
                traj = trajectory.get_traj()
                kdatai, traj_retained_final, _ = correct_mvt_kdata(kdatai, trajectory, cond_mvt,
                                                                   self.paramDict["ntimesteps"], density_adj=True)

            kdatai = np.array(kdatai)
            # nans = np.nonzero(np.isnan(kdatai))
            nans = [np.nonzero(np.isnan(k))[0] for k in kdatai]
            nans_count = np.array([len(n) for n in nans]).sum()

            if nans_count > 0:
                print("Warning : Nan Values replaced by zeros in rebuilt kdata")
                for i, k in enumerate(kdatai):
                    kdatai[i][nans[i]] = 0.0

            if not (movement_correction):
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, mask.shape, useGPU=useGPU_simulation,
                                                               density_adj=True)

            else:
                trajectory.traj_for_reconstruction = traj_retained_final
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, mask.shape,
                                                               useGPU=useGPU_simulation, density_adj=True,
                                                               is_theta_z_adjusted=True)

            # volumesi/=(2*np.pi)

            nans_volumes = np.argwhere(np.isnan(volumesi))
            if len(nans_volumes) > 0:
                np.save('./log/kdatai.npy', kdatai)
                np.save('./log/volumesi.npy', volumesi)
                raise ValueError("Error : Nan Values in volumes")

            del kdatai

            # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

            if log:
                print("Saving correction volumes for iteration {}".format(i))
                with open('./log/volumes1_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(volumesi))

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signals = volumesi[:, mask > 0]
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            del volumesi

            # if useAdjPred:
            #     a = np.sum((volumesi * pred_volumesi.conj()).real) / np.sum(volumesi * volumesi.conj())
            #     volumes = [vol0 - (a * voli - predvoli) for vol0, voli, predvoli in
            #                zip(volumes, volumesi, pred_volumesi)]
            #
            # else:
            #     volumes = [vol + (vol0 - voli) for vol, vol0, voli in zip(volumes, volumes0, volumesi)]

            # signals = [s + (s0 - si) for s, s0, si in zip(signals, signals0, signalsi)]

            norm_signals = np.linalg.norm(signals, axis=0)

            all_signals = signals / norm_signals

        if log:
            print(date_time)

        return dict(zip(keys_results, values_results))

    def search_patterns_test(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        threshold = self.paramDict["threshold"]
        tv_denoising_weight = self.paramDict["tv_denoising_weight"]
        log_phase=self.paramDict["log_phase"]
        # adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        signals = volumes[:, mask > 0]

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/volumes0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, volumes)

        del volumes

        if niter > 0:
            signals0 = signals

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        norm_signals = np.linalg.norm(signals, 2, axis=0)
        all_signals = signals / norm_signals

        mrfdict = dictsearch.Dictionary()
        mrfdict.load(dictfile, force=True)

        keys = mrfdict.keys
        array_water = mrfdict.values[:, :, 0]
        array_fat = mrfdict.values[:, :, 1]

        del mrfdict

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique)
            pca_fat.fit(array_fat_unique)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        else:
            pca_water=None
            pca_fat=None
            transformed_array_water_unique=None
            transformed_array_fat_unique=None

        var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

        var_w = var_w[index_water_unique]
        var_f = var_f[index_fat_unique]

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):
            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            map_rebuilt,J_optim,phase_optim=match_signals(all_signals,keys, pca_water, pca_fat, array_water_unique, array_fat_unique,
                          transformed_array_water_unique, transformed_array_fat_unique, var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates, verbose,
                          niter, split, useGPU_dictsearch,mask,tv_denoising_weight,log_phase)


            print("Maps build for iteration {}".format(i))

            if not(log_phase):
                values_results.append((map_rebuilt, mask))

            else:
                values_results.append((map_rebuilt, mask,phase_optim))

            if log:
                with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i): (map_rebuilt, mask)}, f)

            if i == niter:
                break

            if i>0 and threshold is not None :
                all_signals=all_signals_unthresholded
                map_rebuilt,J_optim,phase_optim = match_signals(all_signals,keys, pca_water, pca_fat, array_water_unique, array_fat_unique,
                          transformed_array_water_unique, transformed_array_fat_unique, var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates, verbose,
                          niter, split, useGPU_dictsearch,mask,tv_denoising_weight)

            print("Generating prediction volumes and undersampled images for iteration {}".format(i))
            # generate prediction volumes
            keys_simu = list(map_rebuilt.keys())
            values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
            map_for_sim = dict(zip(keys_simu, values_simu))

            # predict spokes
            images_pred = MapFromDict("RebuiltMapFromParams", paramMap=map_for_sim, rounding=True, gen_mode=gen_mode)
            images_pred.buildParamMap()

            del map_for_sim

            del keys_simu
            del values_simu

            nspoke = int(trajectory.paramDict["total_nspokes"] / self.paramDict["ntimesteps"])

            images_pred.build_ref_images(seq, norm=J_optim * norm_signals, phase=phase_optim)

            print("Normalizing image series")

            print("Filling images series with renormalized signals")

            kdatai = images_pred.generate_kdata(trajectory, useGPU=useGPU_simulation)

            if log:
                print("Saving Ideal Volumes for iteration {}".format(i))
                with open('./log/predvolumes_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, images_pred.images_series[::nspoke].astype(np.complex64))

            del images_pred

            if movement_correction:
                traj = trajectory.get_traj()
                kdatai, traj_retained_final, _ = correct_mvt_kdata(kdatai, trajectory, cond_mvt,
                                                                   self.paramDict["ntimesteps"], density_adj=True)

            kdatai = np.array(kdatai)
            # nans = np.nonzero(np.isnan(kdatai))
            nans = [np.nonzero(np.isnan(k))[0] for k in kdatai]
            nans_count = np.array([len(n) for n in nans]).sum()

            if nans_count > 0:
                print("Warning : Nan Values replaced by zeros in rebuilt kdata")
                for i, k in enumerate(kdatai):
                    kdatai[i][nans[i]] = 0.0

            if not (movement_correction):
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, mask.shape, useGPU=useGPU_simulation,
                                                               density_adj=True)

            else:
                trajectory.traj_for_reconstruction = traj_retained_final
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, mask.shape,
                                                               useGPU=useGPU_simulation, density_adj=True,
                                                               is_theta_z_adjusted=True)

            # volumesi/=(2*np.pi)

            nans_volumes = np.argwhere(np.isnan(volumesi))
            if len(nans_volumes) > 0:
                np.save('./log/kdatai.npy', kdatai)
                np.save('./log/volumesi.npy', volumesi)
                raise ValueError("Error : Nan Values in volumes")

            del kdatai

            # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

            if log:
                print("Saving correction volumes for iteration {}".format(i))
                with open('./log/volumes1_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(volumesi))

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:, mask > 0]
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            del volumesi

            signals += signals0 - signalsi

            norm_signals = np.linalg.norm(signals, axis=0)
            all_signals_unthresholded = signals / norm_signals

            if threshold is not None:

                signals_for_map = signals * (1 - (np.abs(signals) > threshold))
                all_signals = signals_for_map/np.linalg.norm(signals_for_map,axis=0)

            else:

                all_signals=all_signals_unthresholded

        if log:
            print(date_time)

        return dict(zip(keys_results, values_results))


class ToyNN(Optimizer):

    def __init__(self,model,fitting_opt,model_opt={},input_scaler=StandardScaler(),output_scaler=MinMaxScaler(),niter=0,log=True,fitted=False,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        super().__init__(**kwargs)

        self.paramDict["model"]=model

        self.paramDict["input_scaler"]=input_scaler
        self.paramDict["output_scaler"] = output_scaler

        self.paramDict["fitting_opt"]=fitting_opt
        self.paramDict["is_fitted"] = fitted
        self.paramDict["model_opt"]=model_opt




    def search_patterns(self,dictfile,volumes,force_refit=False):
        model = self.paramDict["model"]
        fitting_opt=self.paramDict["fitting_opt"]

        if not(self.paramDict["is_fitted"]) or (force_refit):
            self.fit_and_set(dictfile)
            model=self.paramDict["model"]

        mask=self.mask
        all_signals = volumes[:, mask > 0]

        #all_signals=all_signals/np.expand_dims(np.linalg.norm(all_signals,axis=0),axis=0)

        real_signals=all_signals.real.T
        imag_signals=all_signals.imag.T

        #real_signals = real_signals / np.expand_dims(np.linalg.norm(real_signals, axis=-1), axis=-1)
        #imag_signals = imag_signals / np.expand_dims(np.linalg.norm(imag_signals, axis=-1), axis=-1)


        signals_for_model = np.concatenate((real_signals, imag_signals), axis=-1)
        #signals_for_model=signals_for_model/np.expand_dims(np.linalg.norm(signals_for_model,axis=-1),axis=-1)
        if self.paramDict["input_scaler"] is not None:
            signals_for_model=self.paramDict["input_scaler"].transform(signals_for_model)

        print(signals_for_model.shape)

        params_all = self.paramDict["output_scaler"].inverse_transform(model.predict(signals_for_model))

        map_rebuilt = {
            "wT1": params_all[:, 0],
            "fT1": params_all[:, 1],
            "attB1": params_all[:, 2],
            "df": params_all[:, 3],
            "ff": params_all[:, 4]

        }

        return {0:(map_rebuilt,mask)}




    def fit_and_set(self,dictfile):
        #For keras - shape is n_observations * n_features

        model = self.paramDict["model"]
        fitting_opt = self.paramDict["fitting_opt"]
        model_opt = self.paramDict["model_opt"]

        FF_list = list(np.arange(0., 1.05, 0.05))
        keys, signal = read_mrf_dict(dictfile, FF_list)

        Y_TF = np.array(keys)

        #signal=signal/np.expand_dims(np.linalg.norm(signal,axis=-1),axis=-1)

        real_signal = signal.real
        imag_signal = signal.imag

        #real_signal=real_signal/np.expand_dims(np.linalg.norm(real_signal,axis=-1),axis=-1)
        #imag_signal=imag_signal/np.expand_dims(np.linalg.norm(imag_signal,axis=-1),axis=-1)

        X_TF = np.concatenate((real_signal, imag_signal), axis=1)
        #X_TF = X_TF/np.expand_dims(np.linalg.norm(X_TF,axis=-1),axis=-1)

        input_scaler = self.paramDict["input_scaler"]
        output_scaler = self.paramDict["output_scaler"]

        if input_scaler is not None:
            X_TF = input_scaler.fit_transform(X_TF)
        Y_TF = output_scaler.fit_transform(Y_TF)

        self.paramDict["input_scaler"]=input_scaler
        self.paramDict["output_scaler"] = output_scaler

        print(X_TF.shape)
        print(Y_TF.shape)

        n_outputs = Y_TF.shape[1]

        print(n_outputs)

        final_model = model(n_outputs,**model_opt)

        history = final_model.fit(
            X_TF, Y_TF, **fitting_opt)

        self.paramDict["model"]=final_model
        self.paramDict["is_fitted"]=True




#
# class SPIJNSearch(Optimizer):
#
#     def __init__(self,num_comp=2,max_iter=10,ff_list=np.arange(0.,1.05,0.05),pca=True,threshold_pca=15,**kwargs):
#         #transf is a function that takes as input timesteps arrays and outputs shifts as output
#         super().__init__(**kwargs)
#
#         self.paramDict["num_comp"]=num_comp
#         self.paramDict["max_iter"] = max_iter
#         self.paramDict["ff_list"] = ff_list
#         self.paramDict["pca"] = pca
#         self.paramDict["threshold_pca"] = threshold_pca
#
# 
#
#     def search_patterns(self,dictfile,volumes):
#
#         num_comp=self.paramDict["num_comp"]
#         max_iter=self.paramDict["max_iter"]
#         ff_list=self.paramDict["ff_list"]
#         pca=self.paramDict["pca"]
#         threshold_pca=self.paramDict["threshold_pca"]
#
#         keys,values=read_mrf_dict(dictfile,ff_list)
#
#         mask=self.mask
#         all_signals = volumes[:, mask > 0]
#         #values=values.T
#
#
#         if pca:
#             pca_values = PCAComplex(n_components_=threshold_pca)
#
#             pca_values.fit(values)
#
#             print(
#                 "Components Retained {} out of {} timesteps".format(pca_signal.n_components_, values.shape[1]))
#
#             transformed_values= pca_values.transform(values)
#
#             transformed_all_signals = np.transpose(
#                 pca_values.transform(np.transpose(all_signals)))
#
#
#         else:
#             pca_values=None
#             transformed_values=values
#             transformed_all_signals=all_signals
#
#
#         transformed_values=values.T
#
#         print(transformed_values.T)
#
#         map_rebuilt = {
#             "wT1": params_all[:, 0],
#             "fT1": params_all[:, 1],
#             "attB1": params_all[:, 2],
#             "df": params_all[:, 3],
#             "ff": params_all[:, 4]
#
#         }
#
#         return {0:(map_rebuilt,mask)}
#

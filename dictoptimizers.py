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
try:
    #from pycuda.autoinit import _finish_up
    import cupy as cp
except:
    print("I was here")
    pass

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from numba import cuda
import gc

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

    def __init__(self,niter=0,seq=None,trajectory=None,split=500,pca=True,threshold_pca=15,useGPU_dictsearch=False,useGPU_simulation=True,movement_correction=False,cond=None,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        super().__init__(**kwargs)
        self.paramDict["niter"]=niter
        self.paramDict["split"] = split
        self.paramDict["pca"] = pca
        self.paramDict["threshold_pca"] = threshold_pca
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
        cond=self.paramDict["cond"]

        if movement_correction:
            if cond is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")


        volumes = volumes / np.linalg.norm(volumes, 2, axis=0)
        all_signals = volumes[:, mask > 0]
        volumes0 = volumes

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")

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
            all_signals_unique, index_signals_unique = np.unique(all_signals, axis=1, return_inverse=True)
            nb_signals_unique = all_signals_unique.shape[1]
            nb_signals = all_signals.shape[1]

            print("There are {} unique signals to match along {} water and {} fat components".format(nb_signals_unique,array_water_unique.shape[0],array_fat_unique.shape[0]))

            duplicate_signals = True
            if nb_signals_unique == nb_signals:
                print("No duplicate signals")
                duplicate_signals = False
                all_signals_unique = all_signals


            num_group = int(nb_signals_unique / split) + 1

            idx_max_all_unique = []
            alpha_optim = []

            for j in tqdm(range(num_group)):
                j_signal = j * split
                j_signal_next = np.minimum((j + 1) * split, nb_signals_unique)

                if self.verbose:
                    print("PCA transform")
                    start = datetime.now()

                if not(useGPU_dictsearch):

                    if pca:
                        transformed_all_signals_water = np.transpose(pca_water.transform(np.transpose(all_signals_unique)))
                        transformed_all_signals_fat = np.transpose(pca_fat.transform(np.transpose(all_signals_unique)))

                        sig_ws_all_unique = np.matmul(transformed_array_water_unique,
                                                      transformed_all_signals_water[:, j_signal:j_signal_next].conj())
                        sig_fs_all_unique = np.matmul(transformed_array_fat_unique,
                                                      transformed_all_signals_fat[:, j_signal:j_signal_next].conj())
                    else:
                        sig_ws_all_unique = np.matmul(array_water_unique, all_signals_unique[:, j_signal:j_signal_next].conj())
                        sig_fs_all_unique = np.matmul(array_fat_unique, all_signals_unique[:, j_signal:j_signal_next].conj())


                else:


                    if pca:

                        transformed_all_signals_water = cp.transpose(
                            pca_water.transform(cp.transpose(cp.asarray(all_signals_unique)))).get()
                        transformed_all_signals_fat = cp.transpose(pca_fat.transform(cp.transpose(cp.asarray(all_signals_unique)))).get()

                        sig_ws_all_unique = (cp.matmul(cp.asarray(transformed_array_water_unique),
                                                      cp.asarray(transformed_all_signals_water)[:,
                                                      j_signal:j_signal_next].conj())).get()
                        sig_fs_all_unique = (cp.matmul(cp.asarray(transformed_array_fat_unique),
                                                      cp.asarray(transformed_all_signals_fat)[:,
                                                      j_signal:j_signal_next].conj())).get()
                    else:

                        sig_ws_all_unique = (cp.matmul(cp.asarray(array_water_unique),
                                                      cp.asarray(all_signals_unique)[:, j_signal:j_signal_next].conj())).get()
                        sig_fs_all_unique = (cp.matmul(cp.asarray(array_fat_unique),
                                                      cp.asarray(all_signals_unique)[:, j_signal:j_signal_next].conj())).get()


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
                    print("Calculating alpha optim and flooring")
                    # start = datetime.now()
                    # current_alpha_all_unique = (sig_wf * current_sig_ws - var_w * current_sig_fs) / (
                    #         (current_sig_ws + current_sig_fs) * sig_wf - var_w * current_sig_fs - var_f * current_sig_ws)
                    # end=datetime.now()
                    # print(end-start)

                    ### Testing direct phase solving
                    A = sig_wf*current_sig_ws_for_phase-var_w*current_sig_fs_for_phase
                    B = (current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase
                    beta = B.real*current_sig_fs_for_phase.real-B.imag*current_sig_fs_for_phase.imag
                    delta = B.imag*current_sig_ws_for_phase.imag - B.real*current_sig_ws_for_phase.real
                    gamma=A.real*current_sig_ws_for_phase.real - A.imag*current_sig_ws_for_phase.imag
                    nu = A.imag*current_sig_fs_for_phase.imag-A.real*current_sig_fs_for_phase.real

                    a = beta + delta
                    b = gamma-delta+nu
                    c=-gamma

                    del beta
                    del delta
                    del gamma
                    del nu


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

                    start = datetime.now()
                    current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)

                    phase_adj=np.angle((1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)


                    end=datetime.now()
                    print(end-start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    print("Calculating cost for all signals")
                    start = datetime.now()
                    current_sig_ws = (current_sig_ws_for_phase*np.exp(-1j*phase_adj)).real
                    current_sig_fs = (current_sig_fs_for_phase * np.exp(-1j * phase_adj)).real

                    del phase_adj
                    del current_sig_ws_for_phase
                    del current_sig_fs_for_phase

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

                    current_sig_ws=cp.asarray(current_sig_ws)
                    current_sig_fs=cp.asarray(current_sig_fs)

                    current_alpha_all_unique = (sig_wf * current_sig_ws - var_w * current_sig_fs) / (
                            (current_sig_ws + current_sig_fs) * sig_wf - var_w * current_sig_fs - var_f * current_sig_ws)

                    if verbose:
                        end = datetime.now()
                        print(end - start)

                    if verbose:
                        start = datetime.now()
                    current_alpha_all_unique = cp.minimum(cp.maximum(current_alpha_all_unique, 0.0), 1.0)

                    phase_adj = np.angle((
                                                     1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)

                    if verbose:
                        end = datetime.now()
                        print(end - start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    if verbose:
                        print("Calculating cost for all signals")
                        start = datetime.now()

                    current_sig_ws = (current_sig_ws_for_phase * np.exp(-1j * phase_adj)).real
                    current_sig_fs = (current_sig_fs_for_phase * np.exp(-1j * phase_adj)).real

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

                if verbose:
                    end = datetime.now()
                    print(end-start)

                if verbose:
                    print("Filling the lists with results for this loop")
                    start = datetime.now()

                idx_max_all_unique.extend(idx_max_all_current)
                alpha_optim.extend(current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])])

                if verbose:
                    end = datetime.now()
                    print(end - start)

            # idx_max_all_unique = np.argmax(J_all, axis=0)
            del J_all
            del current_alpha_all_unique

            print("Building the maps for iteration {}".format(i))

            # del sig_ws_all_unique
            # del sig_fs_all_unique

            params_all_unique = np.array(
                [keys[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique)])

            if duplicate_signals:
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

            images_pred.build_ref_images(seq)
            #images_pred.build_timeline(seq)
            #pred_volumesi = images_pred.images_series

            # map_all_on_mask = np.stack(list(images_pred.paramMap.values())[:-1], axis=-1)
            # map_ff_on_mask = images_pred.paramMap["ff"]
            #
            # mrfdict = dictsearch.Dictionary()
            # mrfdict.load(dictfile, force=True)
            #
            #
            # images_series = np.zeros(self.image_size + (values.shape[-2],), dtype=np.complex_)
            # # water_series = images_series.copy()
            # # fat_series = images_series.copy()
            #
            # print("Building image series")
            # images_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0] * (1 - map_ff_on_mask[i]) + mrfdict[tuple(
            #     pixel_params)][:, 1] * (map_ff_on_mask[i]) for (i, pixel_params) in enumerate(map_all_on_mask)])
            # # water_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0]  for (i, pixel_params) in enumerate(map_all_on_mask)])
            # # fat_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 1]  for (i, pixel_params) in enumerate(map_all_on_mask)])
            #
            # images_series[self.mask > 0, :] = images_in_mask

            #volumesi = images_pred.simulate_radial_undersampled_images(trajectory, density_adj=True)
            kdatai = images_pred.generate_kdata(trajectory,useGPU=useGPU_simulation)

            if movement_correction:
                traj=trajectory.get_traj()
                kdatai, traj_retained_final, _ = correct_mvt_kdata(kdatai, traj, cond,trajectory.paramDict["ntimesteps"])

            kdatai = np.array(kdatai)
            #nans = np.nonzero(np.isnan(kdatai))
            nans = [np.nonzero(np.isnan(k))[0] for k in kdatai]
            nans_count = np.array([len(n) for n in nans]).sum()

            if nans_count>0:
                print("Warning : Nan Values replaced by zeros in rebuilt kdata")
                for i,k in enumerate(kdatai):
                    kdatai[i][nans[i]]=0.0

            if not(movement_correction):
                volumesi = simulate_radial_undersampled_images(kdatai,trajectory,images_pred.image_size,useGPU=useGPU_simulation,density_adj=True)

            else:
                trajectory.traj_for_reconstruction=traj_retained_final
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, images_pred.image_size,
                                                               useGPU=useGPU_simulation, density_adj=True,is_theta_z_adjusted=True)

            nans_volumes = np.argwhere(np.isnan(volumesi))
            if len(nans_volumes) > 0:
                np.save('./log/kdatai.npy', kdatai)
                np.save('./log/volumesi.npy',volumesi)
                raise ValueError("Error : Nan Values in volumes")

            volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)


            if log:
                print("Saving correction volumes for iteration {}".format(i))
                with open('./log/volumes0_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(volumes0))
                with open('./log/volumes1_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(volumesi))
                with open('./log/predvolumes_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(images_pred.images_series))


            del images_pred
            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            # if useAdjPred:
            #     a = np.sum((volumesi * pred_volumesi.conj()).real) / np.sum(volumesi * volumesi.conj())
            #     volumes = [vol0 - (a * voli - predvoli) for vol0, voli, predvoli in
            #                zip(volumes, volumesi, pred_volumesi)]
            #
            # else:
            #     volumes = [vol + (vol0 - voli) for vol, vol0, voli in zip(volumes, volumes0, volumesi)]

            volumes = [vol + (vol0 - voli) for vol, vol0, voli in zip(volumes, volumes0, volumesi)]

            del volumesi
            del kdatai

            all_signals = np.array(volumes)[:, mask > 0]


        if log:
            print(date_time)

        return dict(zip(keys_results, values_results))


class ToyNN(Optimizer):

    def __init__(self,model,model_opt,input_scaler=StandardScaler(),output_scaler=MinMaxScaler(),niter=0,pca=False,threshold_pca=15,log=True,fitted=False,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        super().__init__(**kwargs)

        self.paramDict["model"]=model
        self.paramDict["input_scaler"]=input_scaler
        self.paramDict["output_scaler"] = output_scaler

        self.paramDict["model_opt"]=model_opt
        self.paramDict["pca"] = pca
        self.paramDict["threshold_pca"] = threshold_pca
        self.paramDict["is_fitted"] = fitted



    def search_patterns(self,dictfile,volumes,force_refit=False):
        model = self.paramDict["model"]
        model_opt=self.paramDict["model_opt"]

        if not(self.paramDict["is_fitted"]) or (force_refit):
            self.fit_and_set(dictfile,model,model_opt)
            model=self.paramDict["model"]

        mask=self.mask
        all_signals = volumes[:, mask > 0]
        real_signals=all_signals.real.T
        imag_signals=all_signals.imag.T


        signals_for_model = np.concatenate((real_signals, imag_signals), axis=-1)

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




    def fit_and_set(self,dictfile,model,model_opt):
        #For keras - shape is n_observations * n_features
        FF_list = list(np.arange(0., 1.05, 0.05))
        keys, signal = read_mrf_dict(dictfile, FF_list)

        Y_TF = np.array(keys)
        real_signal = signal.real
        imag_signal = signal.imag

        X_TF = np.concatenate((real_signal, imag_signal), axis=1)

        input_scaler = StandardScaler()
        output_scaler = MinMaxScaler()

        X_TF = input_scaler.fit_transform(X_TF)
        Y_TF = output_scaler.fit_transform(Y_TF)

        self.paramDict["input_scaler"]=input_scaler
        self.paramDict["output_scaler"] = output_scaler

        print(X_TF.shape)
        print(Y_TF.shape)

        n_outputs = Y_TF.shape[1]

        print(n_outputs)

        final_model = model(n_outputs)

        history = final_model.fit(
            X_TF, Y_TF, **model_opt)

        self.paramDict["model"]=final_model
        self.paramDict["is_fitted"]=True




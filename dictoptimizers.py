import numpy as np
from scipy import ndimage
from scipy.ndimage import affine_transform
from utils_mrf import translation_breathing,build_mask_single_image,build_mask,simulate_radial_undersampled_images,read_mrf_dict
from Transformers import PCAComplex
from mutools.optim.dictsearch import dictsearch
from tqdm import tqdm
from mrfsim import makevol
from image_series import MapFromDict
from datetime import datetime
try:
    import cupy as cp
except:
    pass

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from numba import cuda


class Optimizer(object):

    def __init__(self,log=False,mask=None,verbose=False,useGPU=False,**kwargs):
        self.paramDict=kwargs
        self.paramDict["log"]=log
        self.paramDict["useGPU"]=useGPU
        self.mask=mask
        self.verbose=verbose


    def search_patterns(self,volumes):
        #takes as input dictionary pattern and an array of images or volumes and outputs parametric maps
        raise ValueError("search_patterns should be implemented in child")

class SimpleDictSearch(Optimizer):

    def __init__(self,niter=0,seq=None,trajectory=None,split=500,pca=True,threshold_pca=15,log=True,useAdjPred=False,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        super().__init__(**kwargs)
        self.paramDict["niter"]=niter
        self.paramDict["split"] = split
        self.paramDict["pca"] = pca
        self.paramDict["threshold_pca"] = threshold_pca
        self.paramDict["useAdjPred"]=useAdjPred

        if niter>0:
            if seq is None:
                raise ValueError("When more than 0 iteration, one needs to supply a sequence in order to resimulate the image series")
            else:
                self.paramDict["sequence"]=seq
            if trajectory is None:
                raise ValueError("When more than 0 iteration, one needs to supply a kspace trajectory in order to resimulate the image series")
            else:
                self.paramDict["trajectory"]=trajectory


    def search_patterns(self,dictfile,volumes):

        if self.mask is None:
            mask = build_mask(volumes)
        else:
            mask = self.mask


        verbose=self.verbose
        niter=self.paramDict["niter"]
        split=self.paramDict["split"]
        pca=self.paramDict["pca"]
        threshold_pca=self.paramDict["threshold_pca"]
        useAdjPred=self.paramDict["useAdjPred"]
        if niter>0:
            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
        log=self.paramDict["log"]
        useGPU=self.paramDict["useGPU"]

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

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]
        nb_patterns = array_water.shape[0]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components=threshold_pca)
            pca_fat = PCAComplex(n_components=threshold_pca)

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

                if not(useGPU):

                    if pca:
                        transformed_all_signals_water = np.transpose(pca_water.transform(np.transpose(all_signals_unique)))
                        transformed_all_signals_fat = np.transpose(pca_fat.transform(np.transpose(all_signals_unique)))

                        sig_ws_all_unique = np.matmul(transformed_array_water_unique,
                                                      transformed_all_signals_water[:, j_signal:j_signal_next].conj()).real
                        sig_fs_all_unique = np.matmul(transformed_array_fat_unique,
                                                      transformed_all_signals_fat[:, j_signal:j_signal_next].conj()).real
                    else:
                        sig_ws_all_unique = np.matmul(array_water_unique, all_signals_unique[:, j_signal:j_signal_next].conj()).real
                        sig_fs_all_unique = np.matmul(array_fat_unique, all_signals_unique[:, j_signal:j_signal_next].conj()).real


                else:

                    print("Using GPU")
                    if pca:

                        transformed_all_signals_water = cp.transpose(
                            pca_water.transform(cp.transpose(cp.asarray(all_signals_unique)))).get()
                        transformed_all_signals_fat = cp.transpose(pca_fat.transform(cp.transpose(cp.asarray(all_signals_unique)))).get()

                        sig_ws_all_unique = (cp.matmul(cp.asarray(transformed_array_water_unique),
                                                      cp.asarray(transformed_all_signals_water)[:,
                                                      j_signal:j_signal_next].conj()).real).get()
                        sig_fs_all_unique = (cp.matmul(cp.asarray(transformed_array_fat_unique),
                                                      cp.asarray(transformed_all_signals_fat)[:,
                                                      j_signal:j_signal_next].conj()).real).get()
                    else:

                        sig_ws_all_unique = (cp.matmul(cp.asarray(array_water_unique),
                                                      cp.asarray(all_signals_unique)[:, j_signal:j_signal_next].conj()).real).get()
                        sig_fs_all_unique = (cp.matmul(cp.asarray(array_fat_unique),
                                                      cp.asarray(all_signals_unique)[:, j_signal:j_signal_next].conj()).real).get()


                if self.verbose:
                    end = datetime.now()
                    print(end - start)

                if self.verbose:
                    print("Extracting all sig_ws and sig_fs")
                    start = datetime.now()

                current_sig_ws = sig_ws_all_unique[index_water_unique, :]
                current_sig_fs = sig_fs_all_unique[index_fat_unique, :]

                if self.verbose:
                    end = datetime.now()
                    print(end-start)

                if not(useGPU):
                    print("Calculating alpha optim and flooring")
                    start = datetime.now()
                    current_alpha_all_unique = (sig_wf * current_sig_ws - var_w * current_sig_fs) / (
                            (current_sig_ws + current_sig_fs) * sig_wf - var_w * current_sig_fs - var_f * current_sig_ws)
                    end=datetime.now()
                    print(end-start)

                    start = datetime.now()
                    current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)
                    end=datetime.now()
                    print(end-start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    print("Calculating cost for all signals")
                    start = datetime.now()
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

                    sig_wf=cp.asarray(sig_wf)
                    current_sig_ws=cp.asarray(current_sig_ws)
                    current_sig_fs=cp.asarray(current_sig_fs)
                    var_w = cp.asarray(var_w)
                    var_f = cp.asarray(var_f)

                    current_alpha_all_unique = (sig_wf * current_sig_ws - var_w * current_sig_fs) / (
                            (current_sig_ws + current_sig_fs) * sig_wf - var_w * current_sig_fs - var_f * current_sig_ws)

                    if verbose:
                        end = datetime.now()
                        print(end - start)

                    if verbose:
                        start = datetime.now()
                    current_alpha_all_unique = cp.minimum(cp.maximum(current_alpha_all_unique, 0.0), 1.0)

                    if verbose:
                        end = datetime.now()
                        print(end - start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    if verbose:
                        print("Calculating cost for all signals")
                        start = datetime.now()
                    J_all = ((
                                     1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                        (
                                1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                                1 - current_alpha_all_unique) * sig_wf)

                    J_all = J_all.get()
                    current_alpha_all_unique=current_alpha_all_unique.get()
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

            print("Building the maps for iteration {}".format(i))

            # del sig_ws_all_unique
            # del sig_fs_all_unique

            params_all_unique = np.array(
                [keys[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique)])

            if duplicate_signals:
                params_all = params_all_unique[index_signals_unique]
            else:
                params_all = params_all_unique

            map_rebuilt = {
                "wT1": params_all[:, 0],
                "fT1": params_all[:, 1],
                "attB1": params_all[:, 2],
                "df": params_all[:, 3],
                "ff": params_all[:, 4]

            }

            values_results.append((map_rebuilt, mask))

            if useGPU:#Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()

            if i == niter:
                break

            print("Generating prediction volumes and undersampled images for iteration {}".format(i))
            # generate prediction volumes
            keys_simu = list(map_rebuilt.keys())
            values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
            map_for_sim = dict(zip(keys_simu, values_simu))

            # predict spokes
            images_pred = MapFromDict("RebuiltMapFromParams", paramMap=map_for_sim, rounding=False)
            images_pred.buildParamMap()
            images_pred.build_ref_images(seq)
            pred_volumesi = images_pred.images_series

            #volumesi = images_pred.simulate_radial_undersampled_images(trajectory, density_adj=True)
            kdatai = images_pred.generate_kdata(trajectory,useGPU=useGPU)
            volumesi = simulate_radial_undersampled_images(kdatai,trajectory,images_pred.image_size,useGPU=useGPU,density_adj=True)
            volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)


            if log:
                print("Saving correction volumes for iteration {}".format(i))
                with open('./log/volumes0_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(volumes0))
                with open('./log/volumes1_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(volumesi))
                with open('./log/predvolumes_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(pred_volumesi))

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            if useAdjPred:
                a = np.sum((volumesi * pred_volumesi.conj()).real) / np.sum(volumesi * volumesi.conj())
                volumes = [vol0 - (a * voli - predvoli) for vol0, voli, predvoli in
                           zip(volumes, volumesi, pred_volumesi)]

            else:
                volumes = [vol + (vol0 - voli) for vol, vol0, voli in zip(volumes, volumes0, volumesi)]

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




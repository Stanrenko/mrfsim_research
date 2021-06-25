try:
    import matplotlib
    matplotlib.use('TkAgg')
except:
    pass

from scipy import ndimage
from scipy.optimize import minimize
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd
from utils_mrf import create_random_map,voronoi_volumes,normalize_image_series,build_mask
from mutools.optim.dictsearch import dictsearch
import itertools
from mrfsim import groupby,makevol,load_data,loadmat
import numpy as np
import finufft
from tqdm import tqdm
from Transformers import PCAComplex
DEFAULT_wT2 = 80
DEFAULT_fT1 = 350
DEFAULT_fT2 = 40

DEFAULT_ROUNDING_wT1=0
DEFAULT_ROUNDING_wT2=0
DEFAULT_ROUNDING_fT1=0
DEFAULT_ROUNDING_fT2=0
DEFAULT_ROUNDING_df=3 #df in kHz but chemical shifts generally order of magnitude in Hz
DEFAULT_ROUNDING_attB1=2
DEFAULT_ROUNDING_ff=2

DEFAULT_IMAGE_SIZE =(256,256)

def dump_function(func):
    """
    Decorator to print function call details.

    This includes parameters names and effective values.
    """

    def wrapper(*args, **kwargs):

        print(f"{func.__module__}.{func.__qualname__} ")
        return func(*args, **kwargs)

def wrapper_rounding(func):
    def wrapper(self, *args, **kwargs):
        print("Building Param Map")
        func(self,*args,**kwargs)
        if self.paramDict["rounding"]:
            print("Warning : Values in the initial map are being rounded")
            for paramName in self.paramMap.keys():
                self.roundParam(paramName,self.paramDict["rounding_"+paramName])

        if "dict_overrides" in self.paramDict:
            print("Warning : Overriding map values for params {}" .format(self.paramDict["dict_overrides"].keys()))
            for paramName in self.paramDict["dict_overrides"].keys():
                self.paramMap[paramName]=np.ones(self.paramMap[paramName].shape)*self.paramDict["dict_overrides"][paramName]

    return wrapper


class ImageSeries(object):

    def __init__(self,name,dict_config={},**kwargs):
        self.name=name
        self.dict_config=dict_config
        self.paramDict = kwargs
        if "image_size" not in self.paramDict:
            self.paramDict["image_size"]=DEFAULT_IMAGE_SIZE

        self.image_size=self.paramDict["image_size"]
        self.images_series=None
        self.cached_images_series=None

        self.mask =np.ones(self.image_size)
        self.paramMap=None

        if "rounding" not in self.paramDict:
            self.paramDict["rounding"]=False
        else:
            if self.paramDict["rounding"]:
                if "rounding_wT1" not in self.paramDict:
                    self.paramDict["rounding_wT1"] = DEFAULT_ROUNDING_wT1
                if "rounding_wT2" not in self.paramDict:
                    self.paramDict["rounding_wT2"] = DEFAULT_ROUNDING_wT2
                if "rounding_fT1" not in self.paramDict:
                    self.paramDict["rounding_fT1"] = DEFAULT_ROUNDING_fT1
                if "rounding_fT2" not in self.paramDict:
                    self.paramDict["rounding_fT2"] = DEFAULT_ROUNDING_fT2
                if "rounding_ff" not in self.paramDict:
                    self.paramDict["rounding_ff"] = DEFAULT_ROUNDING_ff
                if "rounding_attB1" not in self.paramDict:
                    self.paramDict["rounding_attB1"] = DEFAULT_ROUNDING_attB1
                if "rounding_df" not in self.paramDict:
                    self.paramDict["rounding_df"] = DEFAULT_ROUNDING_df


        self.fat_amp=[0.0586, 0.0109, 0.0618, 0.1412, 0.66, 0.0673]
        fat_cs = [-101.1, 208.3, 281.0, 305.7, 395.6, 446.2]
        self.fat_cs = [- value / 1000 for value in fat_cs]  # temp


    def build_ref_images(self,seq,window=8):
        print("Building Ref Images")
        if self.paramMap is None:
            return ValueError("buildparamMap should be called prior to image simulation")

        list_keys = ["wT1","wT2","fT1","fT2","attB1","df","ff"]
        for k in list_keys:
            if k not in self.paramMap:
                raise ValueError("key {} should be in the paramMap".format(k))

        map_all_on_mask = np.stack(list(self.paramMap.values())[:-1], axis=-1)
        map_ff_on_mask = self.paramMap["ff"]

        params_all = np.reshape(map_all_on_mask, (-1, 6))
        params_unique = np.unique(params_all, axis=0)

        wT1_in_map = np.unique(params_unique[:, 0])
        wT2_in_map = np.unique(params_unique[:, 1])
        fT1_in_map = np.unique(params_unique[:, 2])
        fT2_in_map = np.unique(params_unique[:, 3])
        attB1_in_map = np.unique(params_unique[:, 4])
        df_in_map = np.unique(params_unique[:, 5])

        # Simulating the image sequence

        # water
        print("Simulating Water Signal")
        water = seq(T1=wT1_in_map, T2=wT2_in_map, att=[[attB1_in_map]], g=[[[df_in_map]]])
        water = [np.mean(gp, axis=0) for gp in groupby(water, window)]

        # fat
        print("Simulating Fat Signal")
        eval = "dot(signal, amps)"
        args = {"amps": self.fat_amp}
        # merge df and fat_cs df to dict
        fatdf_in_map = [[cs + f for cs in self.fat_cs] for f in df_in_map]
        fat = seq(T1=[fT1_in_map], T2=fT2_in_map, att=[[attB1_in_map]], g=[[[fatdf_in_map]]], eval=eval, args=args)
        fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]

        # building the time axis
        TE_list = seq.TE
        t = np.cumsum([np.sum(dt, axis=0) for dt in groupby(np.array(TE_list), window)])

        # join water and fat
        print("Build dictionary.")
        keys = list(itertools.product(wT1_in_map, wT2_in_map, fT1_in_map, fT2_in_map, attB1_in_map, df_in_map))
        values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
        values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)
        mrfdict = dictsearch.Dictionary(keys, values)

        images_series = np.zeros(self.image_size + (values.shape[-2],), dtype=np.complex_)
        #water_series = images_series.copy()
        #fat_series = images_series.copy()

        images_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0] * (1 - map_ff_on_mask[i]) + mrfdict[tuple(
            pixel_params)][:, 1] * (map_ff_on_mask[i]) for (i, pixel_params) in enumerate(map_all_on_mask)])
        #water_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0]  for (i, pixel_params) in enumerate(map_all_on_mask)])
        #fat_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 1]  for (i, pixel_params) in enumerate(map_all_on_mask)])

        images_series[self.mask > 0, :] = images_in_mask
        #water_series[self.mask > 0, :] = water_in_mask
        #fat_series[self.mask > 0, :] = fat_in_mask

        images_series = np.moveaxis(images_series, -1, 0)
        #water_series = np.moveaxis(water_series, -1, 0)
        #fat_series = np.moveaxis(fat_series, -1, 0)

        #images_series=normalize_image_series(images_series)
        self.images_series=images_series

        #self.water_series = water_series
        #self.fat_series=fat_series

        self.cached_images_series=images_series
        self.t=t


    def simulate_radial_undersampled_images(self,traj,nspoke=8,density_adj=True,npoint=None):

        size = self.image_size
        images_series=self.images_series
        #images_series =normalize_image_series(self.images_series)

        kdata = [
            finufft.nufft2d2(t.real, t.imag, p)
            for t, p in zip(traj, images_series)
        ]

        dtheta = 1/nspoke
        kdata = np.array(kdata)/npoint*dtheta

        #kdata /= np.sum(np.abs(kdata) ** 2) ** 0.5 / len(kdata)


        if density_adj:
            if npoint is None:
                raise ValueError("Should supply number of point on spoke for density compensation")
            density = np.abs(np.linspace(-1, 1, npoint))
            kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata]

        #kdata = (normalize_image_series(np.array(kdata)))

        images_series_rebuilt = [
            finufft.nufft2d1(t.real, t.imag, s, size)
            for t, s in zip(traj, kdata)
        ]

        #images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

        return np.array(images_series_rebuilt)


    def simulate_undersampled_images(self,traj,density_adj=True):

        size = self.image_size

        kdata = [
            finufft.nufft2d2(t.real, t.imag, p)
            for t, p in zip(traj, self.images_series)
        ]

        #kdata /= np.sum(np.abs(kdata) ** 2) ** 0.5 / len(kdata)

        if density_adj:
            density = np.zeros(kdata.shape)
            for i in tqdm(range(kdata.shape[0])):
                vol = voronoi_volumes(np.transpose(np.array([traj[i].real,traj[i].imag])))[0]

                density[i]
            density = [voronoi_volumes(np.transpose(np.array([t.real,t.imag])))[0] for t in traj]
            kdata = [k*density[i] for i,k in enumerate(kdata)]/(2*np.pi)**2

        images_series_rebuilt = [
            finufft.nufft2d1(t.real, t.imag, s, size)
            for t, s in zip(traj, kdata)
        ]

        return np.array(images_series_rebuilt)

    def generate_kdata(self,traj):
        kdata = [
            finufft.nufft2d2(t.real, t.imag, p)
            for t, p in zip(traj, self.images_series)
        ]

        return kdata

    def rotate_images(self,angles_t):
        # angles_t function returning rotation angle as a function of t
        angles = [angles_t(t) for t in self.t]
        self.images_series= np.array([ndimage.rotate(self.images_series[i,:,:], angles[i], reshape=False) for i in range(self.images_series.shape[0])])

    def translate_images(self,shifts_t,round=True):
        # shifts_t function returning tuple with x,y shift as a function of t
        if round:
            shifts = [np.round(shifts_t(t))for t in self.t]
        else:
            shifts = [shifts_t(t) for t in self.t]

        self.images_series = np.array([affine_transform(self.images_series[i, :, :], ((1.0, 0.0), (0.0, 1.0)),offset=list(-shifts[i]),order=1,mode="nearest") for i in
                                      range(self.images_series.shape[0])])

        #orig = self.images_series[0,:,:]
        #trans = affine_transform(self.images_series[0, :, :], ((1.0, 0.0), (0.0, 1.0)),offset=list(np.round(shifts[0])),order=3,mode="nearest")


    def change_resolution(self,compression_factor=2):
        print("WARNING : Compression is irreversible")
        kept_indices=int(compression_factor)
        self.images_series = self.images_series[:,::kept_indices,::kept_indices]
        self.cached_images_series=self.images_series
        new_mask=self.mask[::kept_indices,::kept_indices]

        for param in self.paramMap.keys():
            values_on_mask=self.paramMap[param]
            values=makevol(values_on_mask,self.mask>0)
            values=values[::kept_indices,::kept_indices]
            new_values_on_mask=values[new_mask>0]
            self.paramMap[param]=new_values_on_mask

        self.mask=new_mask
        self.image_size = self.images_series.shape[1:]


    def buildParamMap(self,mask=None):
        raise ValueError("should be implemented in child")

    def plotParamMap(self,key=None,figsize=(5,5),fontsize=5):
        if key is None:
            keys=list(self.paramMap.keys())
            fig,axes=plt.subplots(1,len(keys),figsize=(len(keys)*figsize[0],figsize[1]))
            for i,k in enumerate(keys):
                im=axes[i].imshow(makevol(self.paramMap[k],(self.mask>0)))
                axes[i].set_title("{} Map".format(k))
                axes[i].tick_params(axis='x', labelsize=fontsize)
                axes[i].tick_params(axis='y', labelsize=fontsize)
                cbar=fig.colorbar(im, ax=axes[i],fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=fontsize)
        else:
            fig,ax=plt.subplots(figsize=figsize)

            im=ax.imshow(makevol(self.paramMap[key],(self.mask>0)))
            ax.set_title("{} Map".format(key))
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            cbar=fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=fontsize)



        plt.show()

    def roundParam(self,paramName,decimals=0):
        self.paramMap[paramName]=np.round(self.paramMap[paramName],decimals=decimals)

    def reset_image_series(self):
        self.images_series=self.cached_images_series

    def compare_patterns(self,pixel_number):

        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.plot(np.real(self.cached_images_series[:,pixel_number[0],pixel_number[1]]),label="original image - real part")
        ax1.plot(np.real(self.images_series[:, pixel_number[0],pixel_number[1]]), label="transformed image - real part")
        ax1.legend()
        ax2.plot(np.imag(self.cached_images_series[:, pixel_number[0], pixel_number[1]]),
                 label="original image - imaginary part")
        ax2.plot(np.imag(self.images_series[:, pixel_number[0], pixel_number[1]]),
                 label="transformed - imaginary part")
        ax2.legend()

        ax3.plot(np.abs(self.cached_images_series[:, pixel_number[0], pixel_number[1]]),
                 label="original image - norm")
        ax3.plot(np.abs(self.images_series[:, pixel_number[0], pixel_number[1]]),
                 label="transformed - norm")
        ax3.legend()

        plt.show()

    def dictSearchMemoryOptimIterative(self, dictfile, seq, traj, npoint, niter=1, pca=True, threshold_pca=0.999999,
                                       split=2000,log=False,mode="Undersampled",useAdjPred=False,path=None,simulate_undersampling=True,true_mask=False):

        nspoke = int(traj.shape[1] / npoint)
        if simulate_undersampling:
            volumes0 = self.simulate_radial_undersampled_images(traj,density_adj=True,npoint=npoint,nspoke=nspoke)
        else:
            volumes0 = self.images_series
            print("No undersampling artifact as working with true images : forcing number of iterations to 0")
            niter = 0

        if true_mask:
            mask =self.mask
        else:
            mask=None

        res = dictSearchMemoryOptimIterative(dictfile,volumes0,seq,traj,npoint,niter,pca,threshold_pca,split,log,useAdjPred,mask)

        return res


class RandomMap(ImageSeries):

    def __init__(self,name,dict_config,**kwargs):
        super().__init__(name,dict_config,**kwargs)

        self.region_size=self.paramDict["region_size"]

        if "mask_reduction_factor" not in self.paramDict:
            self.paramDict["mask_reduction_factor"] =0.0# mask_reduction_factors*total_pixels will be cropped on all edges of the image

        mask_red = self.paramDict["mask_reduction_factor"]
        mask = np.zeros(self.image_size)
        mask[int(self.image_size[0] *mask_red):int(self.image_size[0]*(1-mask_red)), int(self.image_size[1] *mask_red):int(self.image_size[1]*(1-mask_red))] = 1.0
        self.mask=mask

    @wrapper_rounding
    def buildParamMap(self,mask=None):
        #print("Building Param Map")
        if mask is None:
            mask=self.mask
        else:
            self.mask=mask

        wT1 = self.dict_config["water_T1"]
        fT1 = self.dict_config["fat_T1"]
        wT2 = self.dict_config["water_T2"]
        fT2 = self.dict_config["fat_T2"]
        att = self.dict_config["B1_att"]
        df = self.dict_config["delta_freqs"]
        df = [- value / 1000 for value in df]
        ff = self.dict_config["ff"]

        map_wT1 = create_random_map(wT1, self.region_size, self.image_size, mask)
        map_wT2 = create_random_map([wT2], self.region_size, self.image_size, mask)
        map_fT1 = create_random_map(fT1, self.region_size, self.image_size, mask)
        map_fT2 = create_random_map([fT2], self.region_size, self.image_size, mask)
        map_attB1 = create_random_map(att, self.region_size, self.image_size, mask)
        map_df = create_random_map(df, self.region_size, self.image_size, mask)
        map_ff = create_random_map(ff, self.region_size, self.image_size, mask)
        map_all = np.stack((map_wT1, map_wT2, map_fT1, map_fT2, map_attB1, map_df, map_ff), axis=-1)
        map_all_on_mask = map_all[mask > 0]


        self.paramMap = {
            "wT1": map_all_on_mask[:, 0],
            "wT2": map_all_on_mask[:, 1],
            "fT1": map_all_on_mask[:, 2],
            "fT2": map_all_on_mask[:, 3],
            "attB1": map_all_on_mask[:, 4],
            "df": map_all_on_mask[:, 5],
            "ff": map_all_on_mask[:, 6]

        }



class MapFromFile(ImageSeries):

    def __init__(self, name, **kwargs):
        super().__init__(name, {}, **kwargs)



        if "file" not in self.paramDict:
            raise ValueError("file key value argument containing param map file path should be given for MapFromFile")

        if "default_wT2" not in self.paramDict:
            self.paramDict["default_wT2"]=DEFAULT_wT2
        if "default_fT2" not in self.paramDict:
            self.paramDict["default_fT2"]=DEFAULT_fT2
        if "default_fT1" not in self.paramDict:
            self.paramDict["default_fT1"]=DEFAULT_fT1

    @wrapper_rounding
    def buildParamMap(self,mask=None):

        if mask is not None:
            raise ValueError("mask automatically built from wT1 map for file load for now")

        matobj = loadmat(self.paramDict["file"])["paramMap"]
        map_wT1 = matobj["T1"][0, 0]
        map_df = matobj["Df"][0, 0]
        map_attB1 = matobj["B1"][0, 0]
        map_ff = matobj["FF"][0, 0]

        self.image_size=map_wT1.shape

        mask = np.zeros(self.image_size)
        mask[map_wT1>0]=1.0
        self.mask=mask

        map_wT2 = mask*self.paramDict["default_wT2"]
        map_fT1 = mask*self.paramDict["default_fT1"]
        map_fT2 = mask*self.paramDict["default_fT2"]

        map_all = np.stack((map_wT1, map_wT2, map_fT1, map_fT2, map_attB1, map_df, map_ff), axis=-1)
        map_all_on_mask = map_all[mask > 0]

        self.paramMap = {
            "wT1": map_all_on_mask[:, 0],
            "wT2": map_all_on_mask[:, 1],
            "fT1": map_all_on_mask[:, 2],
            "fT2": map_all_on_mask[:, 3],
            "attB1": map_all_on_mask[:, 4],
            "df": -map_all_on_mask[:, 5]/1000,
            "ff": map_all_on_mask[:, 6]
        }


class MapFromDict(ImageSeries):

    def __init__(self, name, **kwargs):
        super().__init__(name, {}, **kwargs)



        if "paramMap" not in self.paramDict:
            raise ValueError("paramMap key value argument containing param map file path should be given for MapFromFile")

        if "default_wT2" not in self.paramDict:
            self.paramDict["default_wT2"]=DEFAULT_wT2
        if "default_fT2" not in self.paramDict:
            self.paramDict["default_fT2"]=DEFAULT_fT2
        if "default_fT1" not in self.paramDict:
            self.paramDict["default_fT1"]=DEFAULT_fT1

    @wrapper_rounding
    def buildParamMap(self,mask=None):

        if mask is not None:
            raise ValueError("mask automatically built from wT1 map for map load for now")

        paramMap = self.paramDict["paramMap"]

        map_wT1=paramMap["wT1"]
        self.image_size=map_wT1.shape

        mask = np.zeros(self.image_size)
        mask[map_wT1>0]=1.0

        self.mask=mask

        map_wT2 = mask*self.paramDict["default_wT2"]
        map_fT1 = mask*self.paramDict["default_fT1"]
        map_fT2 = mask*self.paramDict["default_fT2"]

        map_all = np.stack((paramMap["wT1"], map_wT2, map_fT1, map_fT2, paramMap["attB1"], paramMap["df"], paramMap["ff"]), axis=-1)
        map_all_on_mask = map_all[mask > 0]

        self.paramMap = {
            "wT1": map_all_on_mask[:, 0],
            "wT2": map_all_on_mask[:, 1],
            "fT1": map_all_on_mask[:, 2],
            "fT2": map_all_on_mask[:, 3],
            "attB1": map_all_on_mask[:, 4],
            "df": map_all_on_mask[:, 5],
            "ff": map_all_on_mask[:, 6]
        }




class MapFromMatching(ImageSeries):
    def __init__(self, name, **kwargs):
        super().__init__(name, {}, **kwargs)
        if "search_function" not in self.paramDict:
            raise ValueError("You should define a search_function for building a map from matching with existing patterns")
        if "kdata" not in self.paramDict :
            raise ValueError(
                "You should define a kdata for building a map from matching with existing patterns")
        if "traj" not in self.paramDict :
            raise ValueError(
                "You should define a traj for building a map from matching with existing patterns")

    def buildParamMap(self,mask=None):
        pass


def dictSearchMemoryOptimIterative(dictfile, volumes, seq, traj, npoint, niter=1, pca=True, threshold_pca=0.999999,
                                   split=2000, log=False, useAdjPred=False,init_mask=None):

    if init_mask is None:
        mask=build_mask(volumes)
    else:
        mask=init_mask

    all_signals=np.array(volumes)[:,mask>0]
    volumes0 = volumes

    if log:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")

    nspoke = int(traj.shape[1] / npoint)

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

        print("Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
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

        duplicate_signals = True
        if nb_signals_unique == nb_signals:
            print("No duplicate signals")
            duplicate_signals = False
            all_signals_unique = all_signals

        if pca:
            transformed_all_signals_water = np.transpose(pca_water.transform(np.transpose(all_signals_unique)))
            transformed_all_signals_fat = np.transpose(pca_fat.transform(np.transpose(all_signals_unique)))

            sig_ws_all_unique = np.matmul(transformed_array_water_unique,
                                          transformed_all_signals_water[:, :].conj()).real
            sig_fs_all_unique = np.matmul(transformed_array_fat_unique,
                                          transformed_all_signals_fat[:, :].conj()).real
        else:
            sig_ws_all_unique = np.matmul(array_water_unique, all_signals_unique[:, :].conj()).real
            sig_fs_all_unique = np.matmul(array_fat_unique, all_signals_unique[:, :].conj()).real

        alpha_all_unique = np.zeros((nb_patterns, nb_signals_unique))
        # J_all = np.zeros(alpha_all_unique.shape)

        num_group = int(nb_signals_unique / split) + 1

        idx_max_all_unique = []
        alpha_optim = []

        for j in tqdm(range(num_group)):
            j_signal = j * split
            j_signal_next = np.minimum((j + 1) * split, nb_signals_unique)
            current_sig_ws = sig_ws_all_unique[index_water_unique, j_signal:j_signal_next]
            current_sig_fs = sig_fs_all_unique[index_fat_unique, j_signal:j_signal_next]
            current_alpha_all_unique = (sig_wf * current_sig_ws - var_w * current_sig_fs) / (
                    (current_sig_ws + current_sig_fs) * sig_wf - var_w * current_sig_fs - var_f * current_sig_ws)
            current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)
            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            J_all = ((
                             1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                (
                        1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                        1 - current_alpha_all_unique) * sig_wf)

            idx_max_all_current = np.argmax(J_all, axis=0)
            idx_max_all_unique.extend(idx_max_all_current)
            alpha_optim.extend(current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])])

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

        values_results.append((map_rebuilt,mask))

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
        volumesi = images_pred.simulate_radial_undersampled_images(traj, nspoke=nspoke, npoint=npoint, density_adj=True)

        # def fourier_scaling_cost(a,vol1,vol2,mask):
        #    return np.sum(np.abs(vol1 - a * vol2)[:, mask > 0])

        # res=minimize(lambda x:fourier_scaling_cost(x,pred_volumesi,volumesi,mask),1)

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
            volumes = [vol0 - (a * voli - predvoli) for vol0, voli, predvoli in zip(volumes0, volumesi, pred_volumesi)]

        else:
            volumes = [vol0 - (voli - vol0) for vol0, voli in zip(volumes0, volumesi)]

        if init_mask is None:
            mask=build_mask(volumes)
        all_signals = np.array(volumes)[:, mask > 0]

    if log:
        print(date_time)

    return dict(zip(keys_results, values_results))


def dictSearchMemoryOptimIterativeExternalFile(dictfile, path,seq,shape,nspoke, niter=1, pca=True, threshold_pca=0.999999,
                                   split=2000, log=False, useAdjPred=False):
    print(f"Load input data")
    kdata, traj = load_data(path)

    # density compensation
    npoint = traj.shape[1]
    density=np.abs(np.linspace(-1,1,npoint))

    traj = np.reshape(groupby(traj, nspoke), (-1, npoint * nspoke))
    kdata = np.reshape(groupby(kdata * density ** 0.5, nspoke), (-1, npoint * nspoke))
    kdata /= np.sum(np.abs(kdata) ** 2) ** 0.5 / len(kdata)
    volumes = [
        finufft.nufft2d1(t.real, t.imag, s, shape)
        for t, s in zip(traj, kdata)
    ]

    res=dictSearchMemoryOptimIterative(dictfile,volumes,seq,traj,npoint,niter,pca,threshold_pca, split, log, useAdjPred)

    return res

try:
    import matplotlib
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt

from utils_mrf import create_random_map
from mutools.optim.dictsearch import dictsearch
import itertools
from mrfsim import groupby,makevol,loadmat
import numpy as np
import finufft

DEFAULT_wT2 = 80
DEFAULT_fT1 = 350
DEFAULT_fT2 = 40


def dump_function(func):
    """
    Decorator to print function call details.

    This includes parameters names and effective values.
    """

    def wrapper(*args, **kwargs):

        print(f"{func.__module__}.{func.__qualname__} ")
        return func(*args, **kwargs)



class ImageSeries(object):

    def __init__(self,name,dict_config={},**kwargs):
        self.name=name
        self.dict_config=dict_config
        self.paramDict = kwargs
        self.image_size=self.paramDict["image_size"]
        self.mask =np.ones(self.image_size)
        self.paramMap=None

        self.fat_amp=[0.0586, 0.0109, 0.0618, 0.1412, 0.66, 0.0673]
        fat_cs = [0.1011, -0.2083, -0.281, -0.30569999999999997, -0.3956, -0.4462]
        self.fat_cs = [- value / 1000 for value in fat_cs]  # temp


    def build_ref_images(self,seq,window):
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

        # join water and fat
        print("Build dictionary.")
        keys = list(itertools.product(wT1_in_map, wT2_in_map, fT1_in_map, fT2_in_map, attB1_in_map, df_in_map))
        values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
        values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)
        mrfdict = dictsearch.Dictionary(keys, values)

        images_series = np.zeros(self.image_size + (values.shape[-2],), dtype=np.complex_)
        images_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0] * (1 - map_ff_on_mask[i]) + mrfdict[tuple(
            pixel_params)][:, 1] * (map_ff_on_mask[i]) for (i, pixel_params) in enumerate(map_all_on_mask)])
        images_series[self.mask > 0, :] = images_in_mask

        images_series = np.moveaxis(images_series, -1, 0)
        self.images_series=images_series


    def simulate_undersampled_images(self,traj):

        kdata = [
            finufft.nufft2d2(t.real, t.imag, p)
            for t, p in zip(traj, self.images_series)
        ]

        kdata /= np.sum(np.abs(kdata) ** 2) ** 0.5 / len(kdata)
        images_series_rebuilt = [
            finufft.nufft2d1(t.real, t.imag, s, self.image_size)
            for t, s in zip(traj, kdata)
        ]

        return images_series_rebuilt

    def rotate_images(self,R):
        pass

    def translate_images(self,T):
        pass

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

class RandomMap(ImageSeries):

    def __init__(self,name,dict_config,**kwargs):
        super().__init__(name,dict_config,**kwargs)

        self.paramDict=kwargs

        self.region_size=self.paramDict["region_size"]

        if "mask_reduction_factor" not in self.paramDict:
            self.paramDict["mask_reduction_factor"] =0.0# mask_reduction_factors*total_pixels will be cropped on all edges of the image

        mask_red = self.paramDict["mask_reduction_factor"]
        mask = np.zeros(self.image_size)
        mask[int(self.image_size[0] *mask_red):int(self.image_size[0]*(1-mask_red)), int(self.image_size[1] *mask_red):int(self.image_size[1]*(1-mask_red))] = 1.0
        self.mask=mask


    def buildParamMap(self,mask=None):
        print("Building Param Map")
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

        self.paramDict = kwargs

        if "file" not in self.paramDict:
            raise ValueError("file key value argument containing param map file path should be given for MapFromFile")

        if "default_wT2" not in self.paramDict:
            self.paramDict["default_wT2"]=DEFAULT_wT2
        if "default_fT2" not in self.paramDict:
            self.paramDict["default_fT2"]=DEFAULT_fT2
        if "default_fT1" not in self.paramDict:
            self.paramDict["default_fT1"]=DEFAULT_fT1


    def buildParamMap(self,mask=None):

        if mask is not None:
            raise ValueError("mask automatically built from wT1 map for file load for now")

        matobj = loadmat(self.paramDict["file"])["paramMap"]
        map_wT1 = matobj["T1"][0, 0]
        map_df = matobj["Df"][0, 0]
        map_attB1 = matobj["B1"][0, 0]
        map_ff = matobj["FF"][0, 0]

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
            "df": map_all_on_mask[:, 5],
            "ff": map_all_on_mask[:, 6]

        }












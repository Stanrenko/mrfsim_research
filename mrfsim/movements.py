import numpy as np
from scipy import ndimage
from scipy.ndimage import affine_transform
from utils_mrf import translation_breathing
try:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as cupy_affine_transform
except:
    pass
class Movement(object):

    def __init__(self,**kwargs):
        self.paramDict=kwargs

    def apply(self,timesteps_images):
        #timesteps_images is a n_timesteps * 2 dataframe - first column being the time step, second column being the image at that timestep
        #"Timesteps" / "Images"
        raise ValueError("apply should be implemented in child")

class Translation(Movement):

    def __init__(self,transf,round=True,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        super().__init__(**kwargs)
        self.paramDict["transformation"]=transf
        self.paramDict["round"]=round

    def apply(self,timesteps_images,useGPU=False):
        transf_timesteps_images=timesteps_images.copy()
        transf = self.paramDict["transformation"]

        t = np.array(timesteps_images.Timesteps).reshape(-1,1)

        if self.paramDict["round"]:
            shifts = np.around(transf(t))
        else:
            shifts = transf(t)

        dim = len(timesteps_images.Images.iloc[0].shape)

        if not (np.array(shifts).shape[1] == dim):
            raise ValueError("The transform dimension is not the same as the image space dimension")

        if not(useGPU):
            affine_matrix = tuple([tuple(a) for a in np.eye(dim)])
            transf_timesteps_images["Images"]=timesteps_images.apply(lambda row:affine_transform(row.Images,affine_matrix, offset=list(-shifts[row.name]), order=1, mode="nearest"),axis=1)

        else:
            affine_matrix=np.eye(dim)
            matrix = cp.asarray(affine_matrix)
            #transformed = [cupy_affine_transform(image, matrix, offset=list(-ts[i]), order=1, mode="nearest") for i in
            #           range(len(ts))]
            transf_timesteps_images["Images"] = timesteps_images.apply(
                lambda row: cupy_affine_transform(cp.asarray(row.Images), matrix, offset=list(-shifts[row.name]), order=1,
                                             mode="nearest").get(), axis=1)
        return transf_timesteps_images

class TranslationBreathing(Translation):
    def __init__(self,direction,T=4000,frac_exp=0.7,round=True,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        transf = lambda t:translation_breathing(t,direction,T=T,frac_expiration=frac_exp)
        super().__init__(transf,round,**kwargs)

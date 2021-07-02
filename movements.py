import numpy as np
from scipy import ndimage
from scipy.ndimage import affine_transform
from utils_mrf import translation_breathing

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

    def apply(self,timesteps_images):
        transf_timesteps_images=timesteps_images.copy()
        transf = self.paramDict["transformation"]

        if self.paramDict["round"]:
            shifts = timesteps_images.Timesteps.apply(lambda t:np.round(transf(t)))
        else:
            shifts = timesteps_images.Timesteps.apply(lambda t:transf(t))

        shifts = np.array(list(shifts.values))
        dim = len(timesteps_images.Images.iloc[0].shape)

        if not (np.array(shifts).shape[1] == dim):
            raise ValueError("The transform dimension is not the same as the image space dimension")
        affine_matrix = tuple([tuple(a) for a in np.eye(dim)])

        transf_timesteps_images["Images"]=timesteps_images.apply(lambda row:affine_transform(row.Images,affine_matrix, offset=list(-shifts[row.name]), order=1, mode="nearest"),axis=1)
        return transf_timesteps_images

class TranslationBreathing(Translation):
    def __init__(self,direction,T=4000,frac_exp=0.7,round=True,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        transf = lambda t:translation_breathing(t,direction,T=T,frac_expiration=frac_exp)
        super().__init__(transf,round,**kwargs)

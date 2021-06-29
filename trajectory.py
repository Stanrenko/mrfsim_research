import numpy as np
from utils_mrf import radial_golden_angle_traj,radial_golden_angle_traj_3D
from mrfsim import groupby

class Trajectory(object):

    def __init__(self,**kwargs):
        self.paramDict=kwargs
        self.traj = None

    def get_traj(self):
        #Returns and stores the trajectory array of ntimesteps * total number of points * ndim
        raise ValueError("get_traj should be implemented in child")

class Radial(Trajectory):

    def __init__(self,ntimesteps=175,nspoke=8,npoint=512,**kwargs):
        super().__init__(**kwargs)

        self.paramDict["ntimesteps"]=ntimesteps
        self.paramDict["nspoke"] = nspoke
        self.paramDict["npoint"] = npoint

    def get_traj(self):
        if self.traj is None:
            nspoke = self.paramDict["nspoke"]
            npoint = self.paramDict["npoint"]
            total_nspoke = nspoke *self.paramDict["ntimesteps"]
            all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
            traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))
            traj=np.stack([traj.real,traj.imag],axis=-1)
            self.traj=traj

        return self.traj


class Radial3D(Trajectory):

    def __init__(self,ntimesteps=175,nspoke=8,npoint=512,undersampling_factor=4,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["ntimesteps"] = ntimesteps
        self.paramDict["nspoke"] = nspoke
        self.paramDict["npoint"] = npoint
        self.paramDict["undersampling_factor"] = undersampling_factor

    def get_traj(self):
        if self.traj is None:
            nspoke = self.paramDict["nspoke"]
            npoint = self.paramDict["npoint"]
            total_nspoke = nspoke * self.paramDict["ntimesteps"]
            nb_slices=self.paramDict["nb_slices"]
            undersampling_factor=self.paramDict["undersampling_factor"]
            self.traj=radial_golden_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor)

        return self.traj
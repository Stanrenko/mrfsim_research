import numpy as np
from utils_mrf import radial_golden_angle_traj,radial_golden_angle_traj_3D,spiral_golden_angle_traj,spiral_golden_angle_traj_v2,radial_golden_angle_traj_random_3D
from mrfsim import groupby


class Trajectory(object):

    def __init__(self,**kwargs):
        self.paramDict=kwargs
        self.traj = None
        self.traj_for_reconstruction=None

    def get_traj(self):
        #Returns and stores the trajectory array of ntimesteps * total number of points * ndim
        raise ValueError("get_traj should be implemented in child")

    def get_traj_for_reconstruction(self):
        if self.traj_for_reconstruction is None:
            traj = self.get_traj()
            timesteps = self.paramDict["ntimesteps"]
            self.traj_for_reconstruction=traj.reshape(timesteps,-1,traj.shape[-1])
        return self.traj_for_reconstruction

class Radial(Trajectory):

    def __init__(self,ntimesteps=175,nspoke=8,npoint=512,**kwargs):
        super().__init__(**kwargs)

        self.paramDict["ntimesteps"]=ntimesteps
        self.paramDict["nspoke"] = nspoke
        self.paramDict["npoint"] = npoint
        self.paramDict["nb_rep"]=1

    def get_traj(self):
        if self.traj is None:
            nspoke = self.paramDict["nspoke"]
            npoint = self.paramDict["npoint"]
            total_nspoke = nspoke *self.paramDict["ntimesteps"]
            all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
            #traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))
            traj = all_spokes
            traj=np.stack([traj.real,traj.imag],axis=-1)
            self.traj=traj

        return self.traj



class Radial3D(Trajectory):

    def __init__(self,ntimesteps=175,nspoke=8,npoint=512,undersampling_factor=4,is_random=False,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["ntimesteps"] = ntimesteps
        self.paramDict["nspoke"] = nspoke
        self.paramDict["npoint"] = npoint
        self.paramDict["undersampling_factor"] = undersampling_factor
        self.paramDict["nb_rep"]=int(self.paramDict["nb_slices"]/self.paramDict["undersampling_factor"])
        self.paramDict["random"]=is_random

    def get_traj(self):
        if self.traj is None:
            nspoke = self.paramDict["nspoke"]
            npoint = self.paramDict["npoint"]
            total_nspoke = nspoke * self.paramDict["ntimesteps"]
            nb_slices=self.paramDict["nb_slices"]
            undersampling_factor=self.paramDict["undersampling_factor"]
            if self.paramDict["random"]:
                if "frac_center" in self.paramDict:
                    self.traj = radial_golden_angle_traj_random_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor,self.paramDict["frac_center"])
                else:
                    self.traj = radial_golden_angle_traj_random_3D(total_nspoke, npoint, nspoke, nb_slices,
                                                                   undersampling_factor)
            else:
                self.traj=radial_golden_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor)

        return self.traj




class VariableSpiral(Trajectory):

    def __init__(self,ntimesteps=175,nspiral=8,max_gradient=80 * 10 ** -3,max_slew=200,alpha=128,npoint=256,ninterleaves=1,spatial_us=32,temporal_us=0.1,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["ntimesteps"] = ntimesteps
        self.paramDict["nspiral"] = nspiral #corresponds to the number of spirals by timestep
        self.paramDict["max_gradient"] = max_gradient
        self.paramDict["max_slew"] = max_slew
        self.paramDict["alpha"] = alpha
        self.paramDict["npoint"] = npoint
        self.paramDict["ninterleaves"] = ninterleaves
        self.paramDict["spatial_us"] = spatial_us
        self.paramDict["temporal_us"] = temporal_us

    def get_traj(self):
        if self.traj is None:
            gm = self.paramDict["max_gradient"]
            sm = self.paramDict["max_slew"]
            alpha = self.paramDict["alpha"]
            N = self.paramDict["npoint"]
            fov = N / (2*np.pi)
            ninterleaves = self.paramDict["ninterleaves"]
            R = self.paramDict["spatial_us"]
            f_sampling = self.paramDict["temporal_us"]

            nspiral = self.paramDict["nspiral"]
            ntimesteps = self.paramDict["ntimesteps"]
            total_spirals =ntimesteps*nspiral
            #all_spirals = spiral_golden_angle_traj(total_spirals, fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
            all_spirals=spiral_golden_angle_traj_v2(total_spirals,nspiral, fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
            #traj = np.reshape(groupby(all_spirals, nspiral), (ntimesteps, -1))
            traj=np.stack([all_spirals.real, all_spirals.imag], axis=-1)
            self.traj = traj

        return self.traj



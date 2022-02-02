import freud
import matplotlib.cm
import numpy as np
#import plato.draw.fresnel
import rowan

#backend = plato.draw.fresnel
# For interactive scenes:
# import plato.draw.pythreejs
# backend = plato.draw.pythreejs


#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat

nb_allspokes=1400
undersampling_factor=1
npoint=32
nb_slices=16
incoherent=True
mode="old"

try:
    del radial_traj
except:
    pass

radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=undersampling_factor,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode=mode)

traj=radial_traj.get_traj_for_reconstruction(timesteps=175)

i=0

curr_traj=traj[i]


box=freud.box.Box.cube(2*np.pi+0.00001)

voro = freud.locality.Voronoi()
voro.compute(system=(box, positions))



volumes_reshaped=np.array(voro.volumes).reshape(-1,nb_slices,npoint)


plt.close("all")
for j in range(nb_slices):
    plt.figure()
    plt.plot(volumes_reshaped[:,j,1:799].T)
    plt.title(j)

plt.figure()
plt.plot(volumes_reshaped.reshape(-1,npoint)[2:798].T)


curr_traj=traj[i]
curr_traj_reshaped=np.array(curr_traj).reshape(-1,nb_slices,npoint,3)

positions=curr_traj_reshaped[:,0,:,:2].reshape(-1,2)
plt.figure()
plt.scatter(x=positions[:,0],y=positions[:,1])

box=freud.box.Box.square(2*np.pi+0.00001)

positions = np.hstack((positions, np.zeros((positions.shape[0], 1))))
voro = freud.locality.Voronoi()
voro.compute(system=(box, positions))


def draw_voronoi(box, points, cells, nlist=None, color_by_sides=False):
    ax = plt.gca()
    # Draw Voronoi cells
    patches = [plt.Polygon(cell[:, :2]) for cell in cells]
    patch_collection = matplotlib.collections.PatchCollection(patches, edgecolors='black', alpha=0.4)
    cmap = plt.cm.Set1

    if color_by_sides:
        colors = [len(cell) for cell in voro.polytopes]
    else:
        colors = np.random.permutation(np.arange(len(patches)))

    cmap = plt.cm.get_cmap('Set1', np.unique(colors).size)
    bounds = np.array(range(min(colors), max(colors)+2))

    patch_collection.set_array(np.array(colors))
    patch_collection.set_cmap(cmap)
    patch_collection.set_clim(bounds[0], bounds[-1])
    ax.add_collection(patch_collection)

    # Draw points
    plt.scatter(points[:,0], points[:,1], c=colors)
    plt.title('Voronoi Diagram')
    plt.xlim((-box.Lx/2, box.Lx/2))
    plt.ylim((-box.Ly/2, box.Ly/2))

    # Set equal aspect and draw box
    ax.set_aspect('equal', 'datalim')
    box_patch = plt.Rectangle([-box.Lx/2, -box.Ly/2], box.Lx, box.Ly, alpha=1, fill=None)
    ax.add_patch(box_patch)

    # Draw neighbor lines
    if nlist is not None:
        bonds = np.asarray([points[j] - points[i] for i, j in zip(nlist.index_i, nlist.index_j)])
        box.wrap(bonds)
        line_data = np.asarray([[points[nlist.index_i[i]],
                                 points[nlist.index_i[i]]+bonds[i]] for i in range(len(nlist.index_i))])
        line_data = line_data[:, :, :2]
        line_collection = matplotlib.collections.LineCollection(line_data, alpha=0.3)
        ax.add_collection(line_collection)

    # Show colorbar for number of sides
    if color_by_sides:
        cb = plt.colorbar(patch_collection, ax=ax, ticks=bounds, boundaries=bounds)
        cb.set_ticks(cb.formatter.locs + 0.5)
        cb.set_ticklabels((cb.formatter.locs - 0.5).astype('int'))
        cb.set_label("Number of sides", fontsize=12)
    plt.show()
plt.close("all")
draw_voronoi(box,positions,voro.polytopes)
volumes_reshaped=np.array(voro.volumes).reshape(-1,nb_slices,npoint)

plt.close("all")
for j in range(nb_slices):
    plt.figure()
    plt.plot(volumes_reshaped[:,j,1:799].T)
    plt.title(j)

plt.figure()
plt.plot(volumes_reshaped.reshape(-1,npoint)[2:798].T)

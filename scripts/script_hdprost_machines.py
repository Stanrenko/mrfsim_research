
import machines as ma
from machines import Toolbox
import numpy as np
from pathlib import Path
import logging
import pickle
import time
import sys

sys.path.insert(0, '/home/sconstantin/PythonRepositories/mrfsim_research')

from mrfsim.reco_prost_gpu_v3 import llr_hosvd_allbins, llr_hosvd_regularization
from scipy.ndimage import zoom
from copy import copy

logging.basicConfig(level=logging.INFO)


# =============================================================================
# ── Helpers ───────────────────────────────────────────────────────────────────
# =============================================================================

def downsample_3D(vol, factor):
    if factor is None or factor == 1:
        return vol
    return zoom(vol, (1 / factor, 1 / factor, 1 / factor), order=1)


def gamma_transform_3D(volume, gamma):
    v = np.abs(volume)
    vmin, vmax = v.min(), v.max()
    if vmax == vmin:
        return np.zeros_like(v)
    return ((v - vmin) / (vmax - vmin)) ** gamma


def make_serializable(d):
    """Recursively convert a dict to JSON/pickle-safe types."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = v
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


# =============================================================================
# ── Custom file handler ───────────────────────────────────────────────────────
# =============================================================================

def save_function(dirname, data):
    dirname = Path(dirname)
    np.save(dirname / "denoised_volumes.npy", data["data"])
    with open(dirname / "metadata.pkl", "wb") as fp:
        pickle.dump(data["metadata"], fp)


handler = ma.file_handler(save=save_function)


# =============================================================================
# ── Machine: all bins (with optional joint bin dimension) ─────────────────────
# =============================================================================

@ma.machine()
@ma.output("HDPROST", handler=handler)
@ma.parameter("filename",          str,                                          default=None,     description="Path to .npy file with singular volumes (nb_bins, L, nx, ny, nz)")
@ma.parameter("factor",            int,                                          default=1,        description="Spatial downsampling factor (1 = no downsampling)")
@ma.parameter("patch_size",        int,                                          default=7,        description="Spatial patch size s — patch shape (s, s, s)")
@ma.parameter("bins_patch_size",   int,                                          default=None,     description="Bin patch size bp — if set, process bp adjacent bins jointly")
@ma.parameter("search_radius",     int,                                          default=20,       description="Max voxel distance between patch centres (spatial only)")
@ma.parameter("max_patches",       int,                                          default=30,       description="Max number of similar patches per group (K)")
@ma.parameter("sliding_window",    int,                                          default=3,        description="Step between reference patch centres (W)")
@ma.parameter("rank_selection",    str,                                          default="variance", description="Rank selection method: 'fixed' or 'variance'")
@ma.parameter("variance_threshold",float,                                        default=0.90,     description="Cumulative variance threshold for rank selection (0–1)")
@ma.parameter("gamma",             float,                                        default=0.7,      description="Gamma correction applied after denoising (None to skip)")
@ma.parameter("search_backend",    ma.Choice(['kdtree', 'faiss_cpu', 'faiss_gpu']), default="faiss_gpu", description="Similarity search backend")
def denoise_HDPROST(
    filename, factor, gamma,
    patch_size, bins_patch_size,
    search_radius, max_patches, sliding_window,
    rank_selection, variance_threshold,
    search_backend,
):
    volumes = np.load(filename)
    if volumes.ndim == 4:
        logging.info("Single-bin data — adding bin axis")
        volumes = volumes[None, ...]

    nb_bins, L0, nx, ny, nz = volumes.shape
    logging.info("Loaded volumes: shape=%s dtype=%s", volumes.shape, volumes.dtype)

    # ---- optional downsampling ----
    if factor > 1:
        nx2, ny2, nz2 = nx // factor, ny // factor, nz // factor
        ds = np.zeros((nb_bins, L0, nx2, ny2, nz2), dtype=volumes.dtype)
        for b in range(nb_bins):
            for l in range(L0):
                ds[b, l] = downsample_3D(volumes[b, l], factor)
        volumes = ds
        del ds

    # ---- denoising ----
    t0 = time.perf_counter()
    denoised = llr_hosvd_allbins(
        volumes,
        patch_size=patch_size,
        bins_patch_size=bins_patch_size,   # None → per-bin; int → joint
        search_radius=search_radius,
        max_patches=max_patches,
        sliding_window=sliding_window,
        n_components=None,                 # let variance_threshold decide
        rank_selection=rank_selection,
        variance_threshold=variance_threshold,
        search_backend=search_backend,
    )
    runtime_s = time.perf_counter() - t0
    logging.info("Denoising completed in %.1f s (%.1f min)", runtime_s, runtime_s / 60)

    # ---- optional gamma correction ----
    if gamma is not None and gamma != 1.0:
        nb_bins2, L02 = denoised.shape[:2]
        gamma_adj = np.zeros_like(denoised)
        for b in range(nb_bins2):
            for l in range(L02):
                gamma_adj[b, l] = gamma_transform_3D(denoised[b, l], gamma)
        denoised = gamma_adj

    metadata = make_serializable({
        "filename":          filename,
        "factor":            factor,
        "patch_size":        patch_size,
        "bins_patch_size":   bins_patch_size,
        "search_radius":     search_radius,
        "max_patches":       max_patches,
        "sliding_window":    sliding_window,
        "rank_selection":    rank_selection,
        "variance_threshold":variance_threshold,
        "gamma":             gamma,
        "search_backend":    search_backend,
        "input_shape":       str(volumes.shape),
        "output_shape":      str(denoised.shape),
        "runtime_s":         round(runtime_s, 2),
        "runtime_min":       round(runtime_s / 60, 2),
    })

    return {"metadata": metadata, "data": denoised}


# =============================================================================
# ── Machine: single bin ───────────────────────────────────────────────────────
# =============================================================================

@ma.machine()
@ma.output("HDPROST_singlebin", handler=handler)
@ma.parameter("filename",          str,                                          default=None,     description="Path to .npy file with singular volumes (nb_bins, L, nx, ny, nz)")
@ma.parameter("bin",               int,                                          default=0,        description="Bin index to denoise")
@ma.parameter("factor",            int,                                          default=1,        description="Spatial downsampling factor (1 = no downsampling)")
@ma.parameter("patch_size",        int,                                          default=7,        description="Spatial patch size s")
@ma.parameter("search_radius",     int,                                          default=20,       description="Max voxel distance between patch centres")
@ma.parameter("max_patches",       int,                                          default=30,       description="Max number of similar patches per group (K)")
@ma.parameter("sliding_window",    int,                                          default=3,        description="Step between reference patch centres (W)")
@ma.parameter("rank_selection",    str,                                          default="variance", description="Rank selection method: 'fixed' or 'variance'")
@ma.parameter("variance_threshold",float,                                        default=0.90,     description="Cumulative variance threshold for rank selection (0–1)")
@ma.parameter("gamma",             float,                                        default=0.7,      description="Gamma correction applied after denoising (None to skip)")
@ma.parameter("search_backend",    ma.Choice(['kdtree', 'faiss_cpu', 'faiss_gpu']), default="faiss_gpu", description="Similarity search backend")
def denoise_HDPROST_singlebin(
    filename, bin, factor, gamma,
    patch_size,
    search_radius, max_patches, sliding_window,
    rank_selection, variance_threshold,
    search_backend,
):
    all_volumes = np.load(filename)
    if all_volumes.ndim == 4:
        logging.info("Single-bin data — adding bin axis")
        all_volumes = all_volumes[None, ...]

    print("Starting denoising for bin %d from file '%s'", bin, filename)
    nb_bins, L0, nx, ny, nz = all_volumes.shape
    if bin >= nb_bins:
        raise ValueError(f"bin={bin} out of range for nb_bins={nb_bins}")

    volume = all_volumes[bin].copy()
    del all_volumes
    logging.info("Processing bin %d — shape=%s dtype=%s", bin, volume.shape, volume.dtype)

    # ---- optional downsampling ----
    if factor > 1:
        nx2, ny2, nz2 = nx // factor, ny // factor, nz // factor
        ds = np.zeros((L0, nx2, ny2, nz2), dtype=volume.dtype)
        for l in range(L0):
            ds[l] = downsample_3D(volume[l], factor)
        volume = ds
        del ds

    # ---- denoising ----
    t0 = time.perf_counter()
    denoised = llr_hosvd_regularization(
        volume,
        patch_size=patch_size,
        search_radius=search_radius,
        max_patches=max_patches,
        sliding_window=sliding_window,
        n_components=None,
        rank_selection=rank_selection,
        variance_threshold=variance_threshold,
        search_backend=search_backend,
    )
    runtime_s = time.perf_counter() - t0
    logging.info("Denoising completed in %.1f s (%.1f min)", runtime_s, runtime_s / 60)

    # ---- optional gamma correction ----
    if gamma is not None and gamma != 1.0:
        gamma_adj = np.zeros_like(denoised)
        for l in range(L0):
            gamma_adj[l] = gamma_transform_3D(denoised[l], gamma)
        denoised = gamma_adj

    metadata = make_serializable({
        "filename":          filename,
        "bin":               bin,
        "factor":            factor,
        "patch_size":        patch_size,
        "search_radius":     search_radius,
        "max_patches":       max_patches,
        "sliding_window":    sliding_window,
        "rank_selection":    rank_selection,
        "variance_threshold":variance_threshold,
        "gamma":             gamma,
        "search_backend":    search_backend,
        "input_shape":       str(volume.shape),
        "output_shape":      str(denoised.shape),
        "runtime_s":         round(runtime_s, 2),
        "runtime_min":       round(runtime_s / 60, 2),
    })

    return {"metadata": metadata, "data": denoised}


# =============================================================================
# ── Toolbox ───────────────────────────────────────────────────────────────────
# =============================================================================

toolbox = Toolbox(
    "script_hdprost_machines",
    description="HDPROST LLR-HOSVD denoising on MRF singular volumes",
)
toolbox.add_program("denoise_HDPROST",           denoise_HDPROST)
toolbox.add_program("denoise_HDPROST_singlebin", denoise_HDPROST_singlebin)

if __name__ == "__main__":
    toolbox.cli()

# import machines as ma
# from machines import Toolbox
# import numpy as np
# from pathlib import Path

# import sys
# sys.path.insert(0, '/home/sconstantin/PythonRepositories/mrfsim_research')


# from mrfsim.reco_prost_gpu_v2 import *
# from scipy.ndimage import zoom
# from datetime import datetime


# # from mrfsim.utils_mrf import gamma_transform_3D


# import logging
# logging.basicConfig(level=logging.INFO)

# from copy import copy

# def downsample_3D(vol, factor):
#     if factor is None or factor == 1:
#         return vol
#     return zoom(vol, (1/factor, 1/factor, 1/factor), order=1)

# def gamma_transform_3D(volume,gamma):

#     target=copy(volume)
#     target=((np.abs(volume)-np.min(np.abs(volume)))/(np.max(np.abs(volume))-np.min(np.abs(volume))))**gamma
    
#     return target


# def make_serializable(d):
#     out = {}
#     for k, v in d.items():
#         if isinstance(v, (int, float, str, bool, list, tuple, type(None))):
#             out[k] = v
#         else:
#             out[k] = str(v)
#     return out


# # custom file handler
# def save_function(dirname, data):
#     import pathlib
#     import pickle

#     metadata=data["metadata"]
#     volumes=data["data"]
    

#     dirname = pathlib.Path(dirname)
#     filename="denoised_volumes.npy"
#     fileheader="metadata.pkl"

#     np.save(dirname / filename,volumes)
    
#     with open(dirname / fileheader, "wb") as fp:
#         pickle.dump(metadata, fp)



# handler = ma.file_handler(save=save_function)



# @ma.machine()
# @ma.output("HDPROST", handler=handler)
# @ma.parameter("filename", str, default=None, description="Singular volumes for all-bins")
# @ma.parameter("factor", int, default=1, description="Downsampling index")
# @ma.parameter("patch_size", int, default=7, description="Patch size")
# @ma.parameter("bins_patch_size", int, default=3, description="Patch size temporal dimension")
# @ma.parameter("search_radius", int, default=20, description="Search radius")
# @ma.parameter("max_patches", int, default=30, description="Gamma intensity correction")
# @ma.parameter("sliding_window", int, default=3, description="Gamma intensity correction")
# @ma.parameter("n_components", int, default=None, description="Number of components kept in the SVD")
# @ma.parameter("rank_selection", str, default="variance", description="Method for rank selection")
# @ma.parameter("variance_threshold", float, default=0.9, description="Variance threshold")
# @ma.parameter("gamma", float, default=0.7, description="Gamma intensity correction")
# @ma.parameter("search_backend", ma.Choice(['kdtree', 'faiss_cpu','faiss_gpu']), default="kdtree", description="Gamma intensity correction")
# def denoise_HDPROST(filename,factor,gamma,patch_size,bins_patch_size,search_radius,max_patches,sliding_window,n_components,rank_selection,variance_threshold,search_backend):
#     # twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock"],
#                                 #    optional_additional_arrays=["SliceThickness"])
#     volumes_allbins=np.load(filename)
#     if volumes_allbins.ndim==4:
#         print("Single bin data")
#         volumes_allbins=volumes_allbins[None,...]
    

#     nbins,L0,nx,ny,nz=volumes_allbins.shape
#     volumes_allbins_ds=np.zeros(shape=(nbins,L0,nx//factor,ny//factor,nz//factor),dtype=volumes_allbins.dtype)

#     if factor > 1:
#         for gr in range(nbins):
#             for l in range(L0):
#                 volumes_allbins_ds[gr,l]=downsample_3D(volumes_allbins[gr,l],factor=factor)
#         volumes_allbins=volumes_allbins_ds
#         del volumes_allbins_ds

#     denoised_allbins_full=llr_hosvd_allbins(
#         volumes_allbins, patch_size=patch_size,bins_patch_size=bins_patch_size,search_radius=search_radius, max_patches=max_patches, sliding_window=sliding_window,
#         n_components=n_components, rank_selection=rank_selection, variance_threshold=variance_threshold,search_backend=search_backend
#     )

#     if gamma is not None:

#         denoised_allbins_gamma_adj=np.zeros_like(denoised_allbins_full)

#         for gr in range(nbins):
#             for l in range(L0):
#                 denoised_allbins_gamma_adj[gr,l]=gamma_transform_3D(denoised_allbins_full[gr,l],gamma)

#         denoised_allbins_full=denoised_allbins_gamma_adj
#         del denoised_allbins_gamma_adj


#     metadata = locals().copy()

#         # remove large/unwanted entries
#     metadata.pop("volumes_allbins", None)
#     metadata.pop("denoised_allbins_full", None)
    
#     return {
#         "metadata":metadata,
#         "data":denoised_allbins_full
#     }


# @ma.machine()
# @ma.output("HDPROST_singlebin", handler=handler)
# @ma.parameter("filename", str, default=None, description="Singular volumes for all-bins")
# @ma.parameter("bin", int, default=0, description="Bin number")
# @ma.parameter("factor", int, default=1, description="Downsampling index")
# @ma.parameter("patch_size", int, default=7, description="Patch size")
# @ma.parameter("search_radius", int, default=20, description="Search radius")
# @ma.parameter("max_patches", int, default=30, description="Gamma intensity correction")
# @ma.parameter("sliding_window", int, default=3, description="Gamma intensity correction")
# @ma.parameter("n_components", int, default=None, description="Number of components kept in the SVD")
# @ma.parameter("rank_selection", str, default="variance", description="Method for rank selection")
# @ma.parameter("variance_threshold", float, default=0.9, description="Variance threshold")
# @ma.parameter("gamma", float, default=0.7, description="Gamma intensity correction")
# @ma.parameter("search_backend", ma.Choice(['kdtree', 'faiss_cpu','faiss_gpu']), default="kdtree", description="Gamma intensity correction")
# def denoise_HDPROST_singlebin(filename,bin,factor,gamma,patch_size,search_radius,max_patches,sliding_window,n_components,rank_selection,variance_threshold,search_backend):
#     # twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock"],

#     volumes_allbins=np.load(filename)
#     if volumes_allbins.ndim==4:
#         print("Single bin data")
#         volumes_allbins=volumes_allbins[None,...]
    

#     nbins,L0,nx,ny,nz=volumes_allbins.shape
#     volumes=volumes_allbins[bin]
#     del volumes_allbins
#     volumes_ds=np.zeros(shape=(L0,nx//factor,ny//factor,nz//factor),dtype=volumes.dtype)

    
#     if factor > 1:
#         for l in range(L0):
#             volumes_ds[l]=downsample_3D(volumes[l],factor=factor)
#         volumes=volumes_ds
#         del volumes_ds

#     denoised=llr_hosvd_regularization(
#         volumes, patch_size=patch_size, search_radius=search_radius, max_patches=max_patches, sliding_window=sliding_window,
#         n_components=n_components, rank_selection=rank_selection, variance_threshold=variance_threshold,search_backend=search_backend
#     )

#     if gamma is not None:

#         denoised_gamma_adj=np.zeros_like(denoised)

#         for l in range(L0):
#             denoised_gamma_adj[l]=gamma_transform_3D(denoised[l],gamma)

#         denoised=denoised_gamma_adj
#         del denoised_gamma_adj

#     metadata = locals().copy()

#         # remove large/unwanted entries
#     metadata.pop("volumes", None)
#     metadata.pop("denoised", None)
    
#     return {
#         "metadata":metadata,
#         "data":denoised
#     }



# toolbox = Toolbox("script_hdprost_machines", description="Python HDPROST denoising on MRF singular volumes")

# toolbox.add_program("denoise_HDPROST", denoise_HDPROST)
# toolbox.add_program("denoise_HDPROST_singlebin", denoise_HDPROST_singlebin)

# if __name__ == "__main__":
#     toolbox.cli()


"""
ADMM with HD-PROST (LLR-HOSVD) denoising for iterative MRF reconstruction.

Supports:
  - volumes shape (L0, nz, ny, nx)          — single bin, ndim=4
  - volumes shape (nb_bins, L0, nz, ny, nx) — multi-bin,  ndim=5

When ndim=4 a dummy bin axis is added internally and removed on output.

ADMM splitting:
    min_x  (1/2)||AHA x - b||^2  +  lambda * R(z)
    s.t.   x = z

    x-update : x <- x - mu * ( (AHA(x) - b) + (x - z + u) )
    z-update : z <- prox_{lam}(x + u)   [HOSVD soft-SV thresholding]
    u-update : u <- u + x - z
"""

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tuple3(v, name="param"):
    """Normalise int or length-3 sequence to a (z, y, x) tuple."""
    if isinstance(v, (int, np.integer)):
        return (int(v), int(v), int(v))
    v = tuple(int(x) for x in v)
    if len(v) != 3:
        raise ValueError(f"{name} must be an int or length-3 tuple, got {v}")
    return v


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def _extract_patches_3d(vol, patch_size, step=1):
    """
    Extract overlapping 3-D patches from (C, nz, ny, nx).

    patch_size : int or (sz, sy, sx)
    step       : int or (sz, sy, sx)
    """
    C, nz, ny, nx = vol.shape
    psz, psy, psx = _to_tuple3(patch_size, "patch_size")
    stz, sty, stx = _to_tuple3(step,       "step")

    sz = min(psz, nz)
    sy = min(psy, ny)
    sx = min(psx, nx)

    zs = np.arange(0, nz - sz + 1, stz)
    ys = np.arange(0, ny - sy + 1, sty)
    xs = np.arange(0, nx - sx + 1, stx)
    gz, gy, gx = len(zs), len(ys), len(xs)

    zi = zs[:, None, None, None, None, None] + np.arange(sz)[None, None, None, :, None, None]
    yi = ys[None, :, None, None, None, None] + np.arange(sy)[None, None, None, None, :, None]
    xi = xs[None, None, :, None, None, None] + np.arange(sx)[None, None, None, None, None, :]

    patches_full = vol[:, zi, yi, xi]
    patches_full = patches_full.transpose(1, 2, 3, 0, 4, 5, 6)
    patches_flat = patches_full.reshape(gz * gy * gx, C, sz, sy, sx)

    cz, cy, cx = np.meshgrid(zs, ys, xs, indexing='ij')
    centers = np.stack([cz.ravel(), cy.ravel(), cx.ravel()], axis=1)

    return patches_flat, centers, (gz, gy, gx)


# ---------------------------------------------------------------------------
# Undersampling operator dispatcher
# ---------------------------------------------------------------------------

def _apply_undersampling(vol, radial_traj, b1, weights, dens_adj, incoherent):
    from mrfsim.utils_simu import undersampling_operator_singular, undersampling_operator_singular_new
    """
    vol : (L0, nz, ny, nx)  ->  AHA(vol) same shape.
    Uses undersampling_operator_singular[_new] from calling scope (globals).
    """
    if not incoherent:
        return undersampling_operator_singular_new(
            vol, radial_traj, b1, weights=weights, density_adj=dens_adj)
    else:
        return undersampling_operator_singular(
            vol, radial_traj, b1, weights=weights, density_adj=dens_adj)


# ---------------------------------------------------------------------------
# HOSVD soft singular value thresholding
# ---------------------------------------------------------------------------

def _truncated_hosvd_soft(tensor, lam):
    """
    HOSVD with soft thresholding of singular values per mode.
    Exact proximal operator of the multilinear nuclear norm.

    tensor : (C, N, K)  complex   C = nb_bins * L0
    lam    : float in [0,1] — fraction of leading SV used as threshold
             lam=0 -> identity,  lam=1 -> zero output
    """
    if any(d == 0 for d in tensor.shape):
        return np.zeros_like(tensor)

    Us = []
    for mode in range(3):
        unfolding = np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)
        U, sv, _  = np.linalg.svd(unfolding, full_matrices=False)
        scale     = np.maximum(1.0 - lam * sv[0] / (sv + 1e-30), 0.0)
        Us.append(U * scale[None, :])

    # forward: core = tensor x_0 U_0^H x_1 U_1^H x_2 U_2^H
    core = tensor.copy()
    for mode, U in enumerate(Us):
        unf    = np.moveaxis(core, mode, 0).reshape(core.shape[mode], -1)
        core_m = U.conj().T @ unf
        other  = tuple(core.shape[m] for m in range(core.ndim) if m != mode)
        core   = np.moveaxis(core_m.reshape((core_m.shape[0],) + other), 0, mode)

    # backward: recon = core x_0 U_0 x_1 U_1 x_2 U_2
    recon = core.copy()
    for mode, U in enumerate(Us):
        unf   = np.moveaxis(recon, mode, 0).reshape(recon.shape[mode], -1)
        rec_m = U @ unf
        other = tuple(recon.shape[m] for m in range(recon.ndim) if m != mode)
        recon = np.moveaxis(rec_m.reshape((rec_m.shape[0],) + other), 0, mode)

    return recon.astype(tensor.dtype)


# ---------------------------------------------------------------------------
# HD-PROST proximal step — multi-bin aware
# ---------------------------------------------------------------------------

def _prox_hd_prost(v, lam,
                   patch_size=4, search_radius=10, max_patches=20,
                   sliding_window=2, bins_patch_size=1,
                   search_backend='faiss_gpu', faiss_gpu_id=0,
                   mask=None):
    """
    prox_{lam * ||.||_*}(v) via patch-based HOSVD soft-SV thresholding.

    Parameters
    ----------
    v               : (L0, nz, ny, nx)          ndim=4 — single bin
                      (nb_bins, L0, nz, ny, nx) ndim=5 — multi-bin
    lam             : float in [0,1]
    patch_size      : int or (sz, sy, sx)
    search_radius   : int or (rz, ry, rx) or -1 (global search)
    max_patches     : int
    sliding_window  : int or (sz, sy, sx)
    bins_patch_size : int  number of consecutive bins in each HOSVD group
                      1 = process each bin independently (default)
    search_backend  : 'kdtree' | 'faiss_cpu' | 'faiss_gpu'
    faiss_gpu_id    : int
    mask            : (nz, ny, nx) bool or None

    Returns denoised volume of same shape as input.
    """
    from mrfsim.reco_prost_gpu_v3 import (_build_groups,
                                          _dilate_mask_for_patches,
                                          _filter_centers_by_mask)

    # normalise to 5D: (nb_bins, L0, nz, ny, nx)
    squeezed = (v.ndim == 4)
    if squeezed:
        v = v[None, ...]

    nb_bins, L0, nz, ny, nx = v.shape
    size = (nz, ny, nx)

    ps = _to_tuple3(patch_size,     "patch_size")
    sw = _to_tuple3(sliding_window, "sliding_window")
    sz = min(ps[0], nz)
    sy = min(ps[1], ny)
    sx = min(ps[2], nx)

    # search_radius: int / tuple / -1
    global_search = (search_radius == -1)
    if not global_search:
        sr_tuple  = _to_tuple3(search_radius, "search_radius")
        sr_scalar = max(sr_tuple)
    else:
        sr_tuple  = None
        sr_scalar = -1

    bp        = max(1, min(int(bins_patch_size), nb_bins))
    n_bin_pos = nb_bins - bp + 1

    accum  = np.zeros_like(v)
    weight = np.zeros((nb_bins,) + size, dtype=np.float32)

    for b0 in range(n_bin_pos):
        C         = bp * L0
        vol_joint = v[b0:b0+bp].reshape(C, nz, ny, nx)

        patches, centers, _ = _extract_patches_3d(vol_joint, ps, step=sw)

        if mask is not None:
            dilated = _dilate_mask_for_patches(mask, max(ps))
            keep    = _filter_centers_by_mask(centers, dilated)
            patches = patches[keep]
            centers = centers[keep]

        n_patches   = patches.shape[0]
        N_actual    = sz * sy * sx
        patches_vec = patches.reshape(n_patches, C * N_actual)

        groups, k_actual = _build_groups(
            patches_vec, centers, sr_scalar, max_patches,
            backend=search_backend, faiss_gpu_id=faiss_gpu_id)

        # per-axis radius post-filtering when tuple search_radius given
        if sr_tuple is not None and not global_search:
            centers_f = centers.astype(np.float32)
            sr_arr    = np.array(sr_tuple, dtype=np.float32)
            for ref_idx in range(n_patches):
                K   = k_actual[ref_idx]
                sel = groups[ref_idx, :K]
                sel = sel[sel >= 0]
                if len(sel) == 0:
                    continue
                diff  = np.abs(centers_f[sel] - centers_f[ref_idx])
                valid = np.all(diff <= sr_arr, axis=1)
                valid_sel = sel[valid]
                K_new = max(1, len(valid_sel))
                groups[ref_idx, :K_new] = valid_sel[:K_new]
                if K_new < K:
                    groups[ref_idx, K_new:K] = -1
                k_actual[ref_idx] = K_new

        accum_joint  = np.zeros((C, nz, ny, nx), dtype=np.complex64)
        weight_joint = np.zeros((nz, ny, nx),    dtype=np.float32)

        for ref_idx in tqdm(range(n_patches),
                            desc=f"HOSVD patches bin_pos={b0}", leave=False):
            K        = k_actual[ref_idx]
            selected = groups[ref_idx, :K]
            selected = selected[selected >= 0]
            K        = len(selected)
            if K == 0:
                continue

            group      = patches[selected].reshape(K, C, N_actual)
            tensor     = group.transpose(1, 2, 0)          # (C, N_actual, K)
            tensor_den = _truncated_hosvd_soft(tensor, lam)
            group_den  = tensor_den.transpose(2, 0, 1).reshape(K, C, sz, sy, sx)

            for k_idx, patch_idx in enumerate(selected):
                pz, py, px = (int(c) for c in centers[patch_idx])
                accum_joint[:, pz:pz+sz, py:py+sy, px:px+sx] += group_den[k_idx]
                weight_joint[pz:pz+sz, py:py+sy, px:px+sx]   += 1.0

        weight_joint   = np.where(weight_joint == 0, 1.0, weight_joint)
        denoised_joint = (accum_joint / weight_joint[None, :]).reshape(bp, L0, nz, ny, nx)

        for bi in range(bp):
            accum[b0+bi]  += denoised_joint[bi]
            weight[b0+bi] += 1.0

    weight = np.where(weight == 0, 1.0, weight)
    result = accum / weight[:, None, :, :, :]

    if mask is not None:
        result[:, :, ~mask] = 0.0

    print(f"  _prox_hd_prost  |result|={np.linalg.norm(result):.3e}  "
          f"|v - result|={np.linalg.norm(v - result):.3e}")

    return result[0] if squeezed else result


# ---------------------------------------------------------------------------
# ADMM loop
# ---------------------------------------------------------------------------

def admm_hd_prost(volumes0, weights,
                  radial_traj, b1, dens_adj, incoherent,
                  mu=1.0, lam=0.1, niter=20, n_inner=1,
                  patch_size=4, search_radius=10, max_patches=20,
                  sliding_window=2, bins_patch_size=1,
                  search_backend='faiss_gpu', faiss_gpu_id=0,
                  mask=None):
    """
    ADMM reconstruction with HD-PROST proximal denoising.

    Parameters
    ----------
    volumes0        : (L0, nz, ny, nx)          single bin   ndim=4
                      (nb_bins, L0, nz, ny, nx) multi-bin    ndim=5
    weights         : single bin: array or 1
                      multi-bin:  list of per-bin weights
    radial_traj     : RadialTrajectory object
    b1              : (nb_channels, nz, ny, nx)
    dens_adj        : bool
    incoherent      : bool
    mu              : float   gradient step  (< 1/||AHA||)
    lam             : float   soft-SV threshold in [0,1]
    niter           : int     ADMM outer iterations
    n_inner         : int     gradient steps per x-update
    patch_size      : int or (sz, sy, sx)
    search_radius   : int or (rz, ry, rx) or -1 (global)
    max_patches     : int
    sliding_window  : int or (sz, sy, sx)
    bins_patch_size : int   joint bin patch size (1 = independent bins)
    search_backend  : str
    faiss_gpu_id    : int
    mask            : (nz, ny, nx) bool or None

    Returns
    -------
    z                : same shape as volumes0
    vol_denoised_log : list of z snapshots
    """
    volumes0 = volumes0.astype(np.complex64)

    # normalise to 5D
    squeezed = (volumes0.ndim == 4)
    if squeezed:
        volumes0 = volumes0[None, ...]
        if not isinstance(weights, list):
            weights = [weights]

    nb_bins = volumes0.shape[0]

    x = volumes0.copy()
    z = x.copy()
    u = np.zeros_like(x)

    vol_denoised_log = [z.copy()]

    for i in tqdm(range(niter), desc="ADMM iterations"):
        print(f"\n=== ADMM iteration {i} ===")

        # ---- x-update ----
        for inner in range(n_inner):
            grad_data = np.zeros_like(x)
            for gr in range(nb_bins):
                w_gr = weights[gr] if isinstance(weights, list) else weights
                Ax   = _apply_undersampling(
                           x[gr], radial_traj, b1, w_gr, dens_adj, incoherent)
                Ax   = Ax.reshape(volumes0[gr].shape)
                grad_data[gr] = Ax - volumes0[gr]

            grad_aug = grad_data + (x - z + u)
            x        = x - mu * grad_aug

            print(f"  x-update inner {inner}  "
                  f"|grad_data|={np.linalg.norm(grad_data):.3e}  "
                  f"|x-z|={np.linalg.norm(x - z):.3e}")

        # ---- z-update ----
        print("  z-update: HD-PROST soft-SV denoising ...")
        z_prev = z.copy()
        z_out  = _prox_hd_prost(
                     x + u, lam=lam,
                     patch_size=patch_size,
                     search_radius=search_radius,
                     max_patches=max_patches,
                     sliding_window=sliding_window,
                     bins_patch_size=bins_patch_size,
                     search_backend=search_backend,
                     faiss_gpu_id=faiss_gpu_id,
                     mask=mask)

        # ensure 5D
        z = z_out[None, ...] if z_out.ndim == 4 else z_out

        # ---- u-update ----
        u = u + x - z

        primal_res = float(np.linalg.norm(x - z))
        dual_res   = float(np.linalg.norm(z - z_prev))
        print(f"  |x|={np.linalg.norm(x):.3e}  |z|={np.linalg.norm(z):.3e}  "
              f"|u|={np.linalg.norm(u):.3e}")
        print(f"  |x+u|={np.linalg.norm(x+u):.3e}  "
              f"prox_change={np.linalg.norm(z - z_prev):.3e}")
        print(f"  primal={primal_res:.3e}  dual={dual_res:.3e}")

        vol_denoised_log.append(z.copy())

        if i % 5 == 0:
            np.save(f"volumes_denoised_admm_it{i}.npy",
                    z[0] if squeezed else z)

    z_out   = z[0]  if squeezed else z
    log_out = [s[0] if squeezed else s for s in vol_denoised_log]

    return z_out, log_out

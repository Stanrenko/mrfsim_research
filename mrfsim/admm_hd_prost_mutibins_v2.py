"""
ADMM with HD-PROST (LLR-HOSVD) denoising for iterative MRF reconstruction.

Weights convention
------------------
weights is the full respiratory binning mask applied to k-space.

  ndim=4 input  (L0, nz, ny, nx)          : single bin, weights = 1 (no binning)
  ndim=5 input  (nb_bins, L0, nz, ny, nx) : multi-bin,
                 weights shape (nb_bins, 1, nb_segments, nb_part, 1)
                 weights[gr] shape (1, nb_segments, nb_part, 1) passed to operator

The operator handles the reshape internally.  No weights_list, no squeezing.

ADMM splitting:
    min_x  (1/2) sum_gr ||AHA_gr x_gr - b_gr||^2  +  lambda * R(z)
    s.t.   x = z

    x-update : x <- x - mu * ( grad_data + (x - z + u) )
    z-update : z <- prox_{lam}(x + u)
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
        raise ValueError(f"{name} must be int or length-3 tuple, got {v}")
    return v


def _get_bin_weight(weights, gr):
    """
    Return the weight array for bin gr.

    weights = 1 (int)                          -> return 1
    weights shape (nb_bins, 1, ns, np, 1)      -> return weights[gr]
                                                  shape (1, ns, np, 1)
    """
    if isinstance(weights, (int, type(None))):
        return weights
    return weights[gr]      # (1, nb_segments, nb_part, 1)


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
    stz, sty, stx = _to_tuple3(step, "step")

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

def _apply_undersampling(vol, radial_traj, b1, weight_gr, dens_adj, incoherent,eps=1e-3):
    """
    Apply AHA to vol (L0, nz, ny, nx).

    weight_gr : 1  (single bin, no binning)
             or (1, nb_segments, nb_part, 1)  (one bin's weight mask)
    """
    from mrfsim.utils_mrf import (undersampling_operator_singular,
                                  undersampling_operator_singular_new)
    if not incoherent:
        return undersampling_operator_singular_new(
            vol, radial_traj, b1,
            weights=weight_gr, density_adj=dens_adj)
    else:
        return undersampling_operator_singular(
            vol, radial_traj, b1,
            weights=weight_gr, density_adj=dens_adj,eps=eps)


# ---------------------------------------------------------------------------
# Spatial normalisation — preserves relative L0/bin amplitudes
# ---------------------------------------------------------------------------

def compute_spatial_norm(v5d, eps=1e-10):
    """
    Per-voxel L2 norm across (bins, L0) axes.

    v5d : (nb_bins, L0, nz, ny, nx)
    returns : (nz, ny, nx) float32
    """
    return (np.sqrt(np.sum(np.abs(v5d)**2, axis=(0, 1)))
            .astype(np.float32) + eps)


def _normalize(v5d, norm_map):
    """v5d / norm_map,  norm_map broadcast over (bins, L0)."""
    return v5d / norm_map[None, None, :, :, :]


def _denormalize(v5d, norm_map):
    """v5d * norm_map,  norm_map broadcast over (bins, L0)."""
    return v5d * norm_map[None, None, :, :, :]


# # ---------------------------------------------------------------------------
# # HOSVD soft singular value thresholding
# # ---------------------------------------------------------------------------

# def _truncated_hosvd_soft(tensor, lam):
#     """
#     HOSVD with soft thresholding of singular values per mode.
#     Exact proximal operator of the multilinear nuclear norm.

#     tensor : (C, N, K)  complex   C = bins_patch_size * L0
#     lam    : float in [0,1]  fraction of leading SV used as threshold
#              lam=0 -> identity (|out|/|in| = 1.0)
#              lam=1 -> zero output
#     """
#     if any(d == 0 for d in tensor.shape):
#         return np.zeros_like(tensor)

#     Us = []
#     for mode in range(3):
#         unfolding = np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)
#         U, sv, _  = np.linalg.svd(unfolding, full_matrices=False)
#         scale     = np.maximum(1.0 - lam * sv[0] / (sv + 1e-30), 0.0)
#         Us.append(U * scale[None, :])

#     # forward: core = tensor x_0 U_0^H x_1 U_1^H x_2 U_2^H
#     core = tensor.copy()
#     for mode, U in enumerate(Us):
#         unf    = np.moveaxis(core, mode, 0).reshape(core.shape[mode], -1)
#         core_m = U.conj().T @ unf
#         other  = tuple(core.shape[m] for m in range(core.ndim) if m != mode)
#         core   = np.moveaxis(core_m.reshape((core_m.shape[0],) + other), 0, mode)

#     # backward: recon = core x_0 U_0 x_1 U_1 x_2 U_2
#     recon = core.copy()
#     for mode, U in enumerate(Us):
#         unf   = np.moveaxis(recon, mode, 0).reshape(recon.shape[mode], -1)
#         rec_m = U @ unf
#         other = tuple(recon.shape[m] for m in range(recon.ndim) if m != mode)
#         recon = np.moveaxis(rec_m.reshape((rec_m.shape[0],) + other), 0, mode)

#     return recon.astype(tensor.dtype)




# # ---------------------------------------------------------------------------
# # HD-PROST proximal step
# # ---------------------------------------------------------------------------

# def _prox_hd_prost(v, lam,
#                    patch_size=4, search_radius=10, max_patches=20,
#                    sliding_window=2, bins_patch_size=1,
#                    search_backend='faiss_gpu', faiss_gpu_id=0,
#                    mask=None, normalize=True):
#     """
#     prox_{lam * ||.||_*}(v) via patch-based HOSVD soft-SV thresholding.

#     Parameters
#     ----------
#     v               : (L0, nz, ny, nx)          ndim=4 — single bin
#                       (nb_bins, L0, nz, ny, nx) ndim=5 — multi-bin
#     lam             : float in [0,1]
#     patch_size      : int or (sz, sy, sx)
#     search_radius   : int or (rz, ry, rx) or -1 (global search)
#     max_patches     : int
#     sliding_window  : int or (sz, sy, sx)
#     bins_patch_size : int  consecutive bins processed jointly (1=independent)
#     search_backend  : 'kdtree' | 'faiss_cpu' | 'faiss_gpu'
#     faiss_gpu_id    : int
#     mask            : (nz, ny, nx) bool or None
#     normalize       : bool  spatially normalise before denoising (recommended)

#     Returns denoised volume of same shape as input.
#     """
#     from mrfsim.reco_prost_gpu_v3 import (_build_groups,
#                                           _dilate_mask_for_patches,
#                                           _filter_centers_by_mask)

#     # normalise to 5D
#     squeezed = (v.ndim == 4)
#     if squeezed:
#         v = v[None, ...]                          # (1, L0, nz, ny, nx)

#     nb_bins, L0, nz, ny, nx = v.shape
#     size = (nz, ny, nx)
#     v    = v.astype(np.complex64)

#     # spatial normalisation — uniform across L0/bins
#     if normalize:
#         norm_map = compute_spatial_norm(v)        # (nz, ny, nx)
#         v_work   = _normalize(v, norm_map)
#     else:
#         v_work   = v
#         norm_map = None

#     ps = _to_tuple3(patch_size,     "patch_size")
#     sw = _to_tuple3(sliding_window, "sliding_window")
#     sz = min(ps[0], nz)
#     sy = min(ps[1], ny)
#     sx = min(ps[2], nx)

#     global_search = (search_radius == -1)
#     if not global_search:
#         sr_tuple  = _to_tuple3(search_radius, "search_radius")
#         sr_scalar = max(sr_tuple)
#     else:
#         sr_tuple  = None
#         sr_scalar = -1

#     bp        = max(1, min(int(bins_patch_size), nb_bins))
#     n_bin_pos = nb_bins - bp + 1

#     accum  = np.zeros_like(v_work)
#     weight = np.zeros((nb_bins,) + size, dtype=np.float32)

#     for b0 in range(n_bin_pos):
#         C         = bp * L0
#         vol_joint = v_work[b0:b0+bp].reshape(C, nz, ny, nx)

#         patches, centers, _ = _extract_patches_3d(vol_joint, ps, step=sw)

#         if mask is not None:
#             dilated = _dilate_mask_for_patches(mask, max(ps))
#             keep    = _filter_centers_by_mask(centers, dilated)
#             patches = patches[keep]
#             centers = centers[keep]

#         n_patches   = patches.shape[0]
#         N_actual    = sz * sy * sx
#         patches_vec = patches.reshape(n_patches, C * N_actual)

#         groups, k_actual = _build_groups(
#             patches_vec, centers, sr_scalar, max_patches,
#             backend=search_backend, faiss_gpu_id=faiss_gpu_id)

#         # per-axis radius post-filtering
#         if sr_tuple is not None and not global_search:
#             centers_f = centers.astype(np.float32)
#             sr_arr    = np.array(sr_tuple, dtype=np.float32)
#             for ref_idx in range(n_patches):
#                 K   = k_actual[ref_idx]
#                 sel = groups[ref_idx, :K]
#                 sel = sel[sel >= 0]
#                 if len(sel) == 0:
#                     continue
#                 diff      = np.abs(centers_f[sel] - centers_f[ref_idx])
#                 valid     = np.all(diff <= sr_arr, axis=1)
#                 valid_sel = sel[valid]
#                 K_new     = max(1, len(valid_sel))
#                 groups[ref_idx, :K_new] = valid_sel[:K_new]
#                 if K_new < K:
#                     groups[ref_idx, K_new:K] = -1
#                 k_actual[ref_idx] = K_new

#         accum_joint  = np.zeros((C, nz, ny, nx), dtype=np.complex64)
#         weight_joint = np.zeros((nz, ny, nx),    dtype=np.float32)

#         for ref_idx in tqdm(range(n_patches),
#                             desc=f"HOSVD patches bin_pos={b0}", leave=False):
#             K        = k_actual[ref_idx]
#             selected = groups[ref_idx, :K]
#             selected = selected[selected >= 0]
#             K        = len(selected)
#             if K == 0:
#                 continue

#             group      = patches[selected].reshape(K, C, N_actual)
#             tensor     = group.transpose(1, 2, 0)      # (C, N_actual, K)
#             tensor_den = _truncated_hosvd_soft(tensor, lam)
#             group_den  = tensor_den.transpose(2, 0, 1).reshape(K, C, sz, sy, sx)

#             for k_idx, patch_idx in enumerate(selected):
#                 pz, py, px = (int(c) for c in centers[patch_idx])
#                 accum_joint[:, pz:pz+sz, py:py+sy, px:px+sx] += group_den[k_idx]
#                 weight_joint[pz:pz+sz, py:py+sy, px:px+sx]   += 1.0

#         weight_joint   = np.where(weight_joint == 0, 1.0, weight_joint)
#         denoised_joint = (accum_joint / weight_joint[None, :]
#                           ).reshape(bp, L0, nz, ny, nx)
#         for bi in range(bp):
#             accum[b0+bi]  += denoised_joint[bi]
#             weight[b0+bi] += 1.0

#     weight = np.where(weight == 0, 1.0, weight)
#     result = accum / weight[:, None, :, :, :]

#     # denormalise
#     if normalize:
#         result = _denormalize(result, norm_map)

#     if mask is not None:
#         result[:, :, ~mask] = 0.0

#     print(f"  _prox_hd_prost  |result|={np.linalg.norm(result):.3e}  "
#           f"|v - result|={np.linalg.norm(v - result):.3e}")

#     return result[0] if squeezed else result


# ---------------------------------------------------------------------------
# HOSVD soft singular value thresholding — CPU (single tensor)
# ---------------------------------------------------------------------------

def _truncated_hosvd_soft(tensor, lam):
    """
    HOSVD with soft thresholding — CPU, single tensor (C, N, K).
    lam=0 -> identity, lam=1 -> zero output.
    """
    if any(d == 0 for d in tensor.shape):
        return np.zeros_like(tensor)

    Us = []
    for mode in range(3):
        unfolding = np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)
        U, sv, _  = np.linalg.svd(unfolding, full_matrices=False)
        scale     = np.maximum(1.0 - lam * sv[0] / (sv + 1e-30), 0.0)
        keep      = scale > 0
        if not np.any(keep):
            keep[0] = True
        Us.append(U[:, keep] * scale[None, keep])

    core = tensor.copy()
    for mode, U in enumerate(Us):
        core = np.tensordot(U.conj().T, core, axes=([1], [mode]))
        core = np.moveaxis(core, 0, mode)

    recon = core.copy()
    for mode, U in enumerate(Us):
        recon = np.tensordot(U, recon, axes=([1], [mode]))
        recon = np.moveaxis(recon, 0, mode)

    return recon.astype(tensor.dtype)


# ---------------------------------------------------------------------------
# HOSVD soft singular value thresholding — GPU batched (all patches at once)
# ---------------------------------------------------------------------------

def _truncated_hosvd_soft_gpu_batched(tensors, lam, device='cuda'):
    """
    Batched HOSVD with soft thresholding on GPU.

    tensors : (n_patches, C, N, K)  numpy complex64
              all patches processed in a single GPU SVD call per mode
    lam     : float in [0,1]
    device  : torch device

    Returns : (n_patches, C, N, K)  numpy complex64
    """
    import torch

    dev = torch.device(device)
    t   = torch.from_numpy(tensors.astype(np.complex64)).to(dev)
    # t shape: (B, C, N, K)  where B = n_patches

    Us = []
    # mode 0 -> axis 1 (C), mode 1 -> axis 2 (N), mode 2 -> axis 3 (K)
    for mode in range(3):
        dim   = mode + 1                              # tensor axis
        other = [d for d in range(1, 4) if d != dim]
        perm  = [0, dim] + other                      # (B, dim, rest...)
        inv_perm = [0] * 4
        for i, p in enumerate(perm):
            inv_perm[p] = i

        unf = t.permute(perm).reshape(t.shape[0], t.shape[dim], -1)
        # unf: (B, dim_size, rest)

        U, sv, _ = torch.linalg.svd(unf, full_matrices=False)
        # U: (B, dim_size, r),  sv: (B, r)

        sv0   = sv[:, :1].clamp(min=1e-30)           # (B, 1)
        scale = torch.clamp(1.0 - lam * sv0 / (sv + 1e-30), min=0.0)
        # (B, r) — zero columns where scale=0
        keep_mask = scale > 0                          # (B, r)

        # absorb scale into U columns, zero out columns where scale=0
        U_scaled = U * scale.unsqueeze(1)             # (B, dim_size, r)

        Us.append((U_scaled, keep_mask, dim, other, perm, inv_perm))

    # ---- forward: core = t x_0 U_0^H x_1 U_1^H x_2 U_2^H ----
    core = t.clone()
    for U_scaled, keep_mask, dim, other, perm, inv_perm in Us:
        shape  = core.shape
        unf    = core.permute(perm).reshape(shape[0], shape[dim], -1)
        core_m = torch.bmm(U_scaled.conj().transpose(1, 2), unf)
        # core_m: (B, r, rest)
        r         = core_m.shape[1]
        new_shape = [shape[0], r] + [shape[d] for d in range(1, 4) if d != dim]
        core      = core_m.reshape(new_shape).permute(inv_perm)

    # ---- backward: recon = core x_0 U_0 x_1 U_1 x_2 U_2 ----
    recon = core.clone()
    for U_scaled, keep_mask, dim, other, perm, inv_perm in Us:
        shape  = recon.shape
        unf    = recon.permute(perm).reshape(shape[0], shape[dim], -1)
        rec_m  = torch.bmm(U_scaled, unf)
        # rec_m: (B, orig_dim_size, rest)
        orig_dim = U_scaled.shape[1]
        new_shape = [shape[0], orig_dim] + [shape[d] for d in range(1, 4) if d != dim]
        recon     = rec_m.reshape(new_shape).permute(inv_perm)

    return recon.cpu().numpy().astype(np.complex64)


# ---------------------------------------------------------------------------
# HD-PROST proximal step — with optional GPU HOSVD
# ---------------------------------------------------------------------------

def _prox_hd_prost(v, lam,
                   patch_size=4, search_radius=10, max_patches=20,
                   sliding_window=2, bins_patch_size=1,
                   search_backend='faiss_gpu', faiss_gpu_id=0,
                   mask=None, normalize=True,
                   use_torch=True, torch_device='cuda',
                   torch_batch_size=1024):
    """
    prox_{lam * ||.||_*}(v) via patch-based HOSVD soft-SV thresholding.

    use_torch      : bool  use GPU-batched SVD via PyTorch (recommended)
    torch_device   : str   'cuda' or 'cuda:0' etc.
    torch_batch_size : int patches per GPU batch (tune to fit VRAM)
    """
    from mrfsim.reco_prost_gpu_v3 import (_build_groups,
                                          _dilate_mask_for_patches,
                                          _filter_centers_by_mask)

    squeezed = (v.ndim == 4)
    if squeezed:
        v = v[None, ...]

    nb_bins, L0, nz, ny, nx = v.shape
    size = (nz, ny, nx)
    v    = v.astype(np.complex64)

    if normalize:
        norm_map = compute_spatial_norm(v)
        v_work   = _normalize(v, norm_map)
    else:
        v_work   = v
        norm_map = None

    ps = _to_tuple3(patch_size,     "patch_size")
    sw = _to_tuple3(sliding_window, "sliding_window")
    sz = min(ps[0], nz)
    sy = min(ps[1], ny)
    sx = min(ps[2], nx)

    global_search = (search_radius == -1)
    if not global_search:
        sr_tuple  = _to_tuple3(search_radius, "search_radius")
        sr_scalar = max(sr_tuple)
    else:
        sr_tuple  = None
        sr_scalar = -1

    bp        = max(1, min(int(bins_patch_size), nb_bins))
    n_bin_pos = nb_bins - bp + 1

    accum  = np.zeros_like(v_work)
    weight = np.zeros((nb_bins,) + size, dtype=np.float32)

    for b0 in range(n_bin_pos):
        C         = bp * L0
        vol_joint = v_work[b0:b0+bp].reshape(C, nz, ny, nx)

        patches, centers, _ = _extract_patches_3d(vol_joint, ps, step=sw)

        if mask is not None:
            dilated = _dilate_mask_for_patches(mask, max(ps))
            keep    = _filter_centers_by_mask(centers, dilated)
            patches = patches[keep]
            centers = centers[keep]

        n_patches   = patches.shape[0]
        N_actual    = sz * sy * sx
        print("N_actual:", N_actual)
        print("n_patches:", n_patches)
        print("patches shape:", patches.shape)


        # ---- group building: use patches_mean (faster, mirrors llr_hosvd_allbins) ----
        patches_mean = patches.reshape(n_patches, bp, L0 * N_actual).mean(axis=1)
        print("patches_mean shape:", patches_mean.shape)

        groups, k_actual = _build_groups(
            patches_mean, centers, sr_scalar, max_patches,
            backend=search_backend, faiss_gpu_id=faiss_gpu_id)

        
        print("Groups shape:", groups.shape)
        print("k_actual shape:", k_actual.shape)

        if sr_tuple is not None and not global_search:
            centers_f = centers.astype(np.float32)
            sr_arr    = np.array(sr_tuple, dtype=np.float32)
            for ref_idx in range(n_patches):
                K   = k_actual[ref_idx]
                sel = groups[ref_idx, :K]
                sel = sel[sel >= 0]
                if len(sel) == 0:
                    continue
                diff      = np.abs(centers_f[sel] - centers_f[ref_idx])
                valid     = np.all(diff <= sr_arr, axis=1)
                valid_sel = sel[valid]
                K_new     = max(1, len(valid_sel))
                groups[ref_idx, :K_new] = valid_sel[:K_new]
                if K_new < K:
                    groups[ref_idx, K_new:K] = -1
                k_actual[ref_idx] = K_new

        accum_joint  = np.zeros((C, nz, ny, nx), dtype=np.complex64)
        weight_joint = np.zeros((nz, ny, nx),    dtype=np.float32)

        if use_torch:
            import torch
            dev = torch.device(torch_device)

            for start in tqdm(range(0, n_patches, torch_batch_size),
                            desc=f"HOSVD GPU bin_pos={b0}", leave=False):
                end = min(start + torch_batch_size, n_patches)
                n_b = end - start

                # ---- build batch tensor on-the-fly ----
                # (n_b, C, N_actual, max_patches) — only current batch in RAM
                batch_tensors = np.zeros((n_b, C, N_actual, max_patches),
                                        dtype=np.complex64)
                batch_valid_K = np.zeros(n_b, dtype=np.int32)
                batch_selected = []   # store selected indices for scatter-back

                for i, ref_idx in enumerate(range(start, end)):
                    K        = k_actual[ref_idx]
                    selected = groups[ref_idx, :K]
                    selected = selected[selected >= 0]
                    K        = len(selected)
                    batch_selected.append(selected)
                    if K == 0:
                        continue
                    group    = patches[selected].reshape(K, C, N_actual)
                    tensor   = group.transpose(1, 2, 0)   # (C, N_actual, K)
                    batch_tensors[i, :, :, :K] = tensor
                    batch_valid_K[i]           = K

                # ---- GPU batched SVD ----
                # print("Batch tensors shape:", batch_tensors.shape)
                batch_den = _truncated_hosvd_soft_gpu_batched(
                    batch_tensors, lam, device=torch_device)
                # batch_den: (n_b, C, N_actual, max_patches)

                # print("Batch den shape:", batch_den.shape)

                # ---- scatter back immediately ----
                for i, ref_idx in enumerate(range(start, end)):
                    K        = batch_valid_K[i]
                    selected = batch_selected[i]
                    if K == 0:
                        continue
                    group_den = batch_den[i].transpose(2, 0, 1)[:K].reshape(
                        K, C, sz, sy, sx)
                    for k_idx, patch_idx in enumerate(selected):
                        pz, py, px = (int(c) for c in centers[patch_idx])
                        accum_joint[:, pz:pz+sz, py:py+sy, px:px+sx] += group_den[k_idx]
                        weight_joint[pz:pz+sz, py:py+sy, px:px+sx]   += 1.0

                # free batch memory immediately
                del batch_tensors, batch_den

        else:
            # ---- CPU path: per-patch loop ----
            for ref_idx in tqdm(range(n_patches),
                                desc=f"HOSVD bin_pos={b0}", leave=False):
                K        = k_actual[ref_idx]
                selected = groups[ref_idx, :K]
                selected = selected[selected >= 0]
                K        = len(selected)
                if K == 0:
                    continue

                group      = patches[selected].reshape(K, C, N_actual)
                tensor     = group.transpose(1, 2, 0)
                tensor_den = _truncated_hosvd_soft(tensor, lam)
                group_den  = tensor_den.transpose(2, 0, 1).reshape(K, C, sz, sy, sx)

                for k_idx, patch_idx in enumerate(selected):
                    pz, py, px = (int(c) for c in centers[patch_idx])
                    accum_joint[:, pz:pz+sz, py:py+sy, px:px+sx] += group_den[k_idx]
                    weight_joint[pz:pz+sz, py:py+sy, px:px+sx]   += 1.0

        weight_joint   = np.where(weight_joint == 0, 1.0, weight_joint)
        denoised_joint = (accum_joint / weight_joint[None, :]
                          ).reshape(bp, L0, nz, ny, nx)
        for bi in range(bp):
            accum[b0+bi]  += denoised_joint[bi]
            weight[b0+bi] += 1.0

    weight = np.where(weight == 0, 1.0, weight)
    result = accum / weight[:, None, :, :, :]

    if normalize:
        result = _denormalize(result, norm_map)
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
                  mask=None, normalize=True, torch_batch_size=1024, torch_device='cuda', use_torch=True):
    """
    ADMM reconstruction with HD-PROST proximal denoising.

    Parameters
    ----------
    volumes0    : (L0, nz, ny, nx)           single bin  ndim=4
                  (nb_bins, L0, nz, ny, nx)  multi-bin   ndim=5
    weights     : 1  (single bin, no binning)
                  (nb_bins, 1, nb_segments, nb_part, 1)  (multi-bin)
                  weights[gr] = (1, nb_segments, nb_part, 1) passed to operator
    radial_traj : RadialTrajectory object
    b1          : (nb_channels, nz, ny, nx)
    dens_adj    : bool
    incoherent  : bool
    mu          : float   gradient step  (< 1/||AHA||)
    lam         : float   soft-SV threshold in [0,1]
    niter       : int     ADMM outer iterations
    n_inner     : int     gradient steps per x-update
    patch_size  : int or (sz, sy, sx)
    search_radius : int or (rz, ry, rx) or -1 (global)
    max_patches : int
    sliding_window : int or (sz, sy, sx)
    bins_patch_size : int  joint bin patch size (1=independent)
    search_backend : str
    faiss_gpu_id : int
    mask        : (nz, ny, nx) bool or None
    normalize   : bool  spatial normalisation inside prox (recommended True)

    Returns
    -------
    z                : same shape as volumes0
    vol_denoised_log : list of z snapshots per iteration
    """
    volumes0 = volumes0.astype(np.complex64)

    # normalise to 5D internally
    squeezed = (volumes0.ndim == 4)
    if squeezed:
        volumes0 = volumes0[None, ...]    # (1, L0, nz, ny, nx)
        # single bin -> weights = 1 (no binning mask needed)
        weights  = 1

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
                w_gr = _get_bin_weight(weights, gr)
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
        z_new  = _prox_hd_prost(
                     x + u, lam=lam,
                     patch_size=patch_size,
                     search_radius=search_radius,
                     max_patches=max_patches,
                     sliding_window=sliding_window,
                     bins_patch_size=bins_patch_size,
                     search_backend=search_backend,
                     faiss_gpu_id=faiss_gpu_id,
                     mask=mask,
                     normalize=normalize,torch_batch_size=torch_batch_size,torch_device=torch_device,use_torch=use_torch)

        # ensure 5D
        z = z_new[None, ...] if z_new.ndim == 4 else z_new

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

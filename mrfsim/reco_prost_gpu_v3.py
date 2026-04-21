"""
Locally Low-Rank (LLR) regularization for MR Fingerprinting using HOSVD.

For each bin, processes a volume of shape (L, nx, ny, nz) where L is the number
of singular volumes. For each spatial patch of size s x s x s, finds K similar
patches within radius R, forms a tensor of shape (L, N, K) where N = s**3,
denoises via truncated HOSVD, and averages back.

Supports complex-valued data throughout.

Component selection
-------------------
Two strategies are available via the `rank_selection` argument:

  'fixed'    (default)
      Retain exactly n_components[mode] singular vectors per mode.
      n_components must be a tuple of 3 ints.

  'variance'
      Retain the minimum number of singular vectors whose cumulative explained
      variance (sum of squared singular values) reaches `variance_threshold`
      (float in (0, 1]), independently per mode and per tensor.
      n_components is then used as an *upper bound* (hard cap) per mode.
      Set n_components=None to apply no cap.

Mask support  (mask argument / compute_mask)
--------------------------------------------
Both llr_hosvd_regularization and llr_hosvd_allbins accept an optional
boolean mask of shape (nx, ny, nz).  When provided:

  - The mask is dilated by patch_size so that patches whose footprint
    overlaps the mask boundary are still included.
  - Only patch centres that fall within the dilated mask are processed;
    all others are skipped entirely — they never enter the group-building
    or HOSVD steps, giving a proportional speedup.
  - Voxels outside the original mask are set to zero in the output.

compute_mask(volumes, threshold_factor, dilation_iter) builds the mask
automatically from the first singular volume (L=0) of each bin, taking
the union across bins.

Joint bin processing  (bins_patch_size argument)
-------------------------------------------------
Setting bins_patch_size=bp (int) enables joint processing of bp adjacent
bins.  Each patch then has shape (bp*L, s, s, s) and HOSVD exploits cross-
bin correlations.  A single shared mask (union across bins) is used.

Similarity search backends  (search_backend argument)
------------------------------------------------------
  'kdtree'    scipy cKDTree radius search + L2 ranking (default).
  'faiss_cpu' faiss.IndexFlatL2 — batched exact K-NN on CPU.
  'faiss_gpu' faiss.GpuIndexFlatL2 — same search on GPU.

search_radius argument
----------------------
  int   Spatially constrained search: only patches whose centres lie within
        search_radius voxels (Chebyshev/max-norm) of the reference are
        considered.  Applied after FAISS for the faiss_* backends.
  None  Global search: the K most similar patches are selected from the
        entire volume regardless of spatial distance.  Only meaningful with
        the faiss_cpu or faiss_gpu backends (kdtree always requires a
        radius).  Useful when similar tissue types repeat across the volume
        (e.g. fat, muscle) and B0/B1 variation is small.

GPU path  (llr_hosvd_regularization_gpu / llr_hosvd_allbins_gpu)
-----------------------------------------------------------------
HOSVD and scatter-add run on GPU via PyTorch (torch.linalg.svd with MAGMA).
search_backend='faiss_gpu' is recommended alongside the GPU path.
Mask support is available in the GPU path as well.
"""

import logging
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# =============================================================================
# ── Shared helpers ────────────────────────────────────────────────────────────
# =============================================================================

def _extract_patches_3d(vol, patch_size, step=1):
    """Extract overlapping 3-D patches from (C, nx, ny, nz).

    If patch_size exceeds a spatial dimension (e.g. nz=1 for 2-D data),
    the patch size is clamped to that dimension automatically so the same
    function works for both 2-D slabs and full 3-D volumes.
    """
    C, nx, ny, nz = vol.shape
    sx = min(patch_size, nx)
    sy = min(patch_size, ny)
    sz = min(patch_size, nz)
    xs = np.arange(0, nx - sx + 1, step)
    ys = np.arange(0, ny - sy + 1, step)
    zs = np.arange(0, nz - sz + 1, step)
    gx, gy, gz = len(xs), len(ys), len(zs)
    xi = xs[:, None, None, None, None, None] + np.arange(sx)[None, None, None, :, None, None]
    yi = ys[None, :, None, None, None, None] + np.arange(sy)[None, None, None, None, :, None]
    zi = zs[None, None, :, None, None, None] + np.arange(sz)[None, None, None, None, None, :]
    patches_full = vol[:, xi, yi, zi]
    patches_full = patches_full.transpose(1, 2, 3, 0, 4, 5, 6)
    patches_flat = patches_full.reshape(gx * gy * gz, C, sx, sy, sz)
    cx, cy, cz = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=1)
    return patches_flat, centers, (gx, gy, gz)


def _rank_from_variance(singular_values, threshold, max_rank=None):
    sv2 = singular_values ** 2
    total = sv2.sum()
    if total == 0:
        return 1
    cumvar = np.cumsum(sv2) / total
    hits = np.where(cumvar >= threshold)[0]
    r = int(hits[0]) + 1 if len(hits) else len(singular_values)
    if max_rank is not None:
        r = min(r, max_rank)
    return max(r, 1)


def _truncated_hosvd(tensor, n_components, rank_selection='fixed',
                     variance_threshold=0.95):
    if any(d == 0 for d in tensor.shape):
        return np.zeros_like(tensor)
    shape = tensor.shape
    Us = []
    for mode in range(3):
        unfolding = np.moveaxis(tensor, mode, 0).reshape(shape[mode], -1)
        U, sv, _ = np.linalg.svd(unfolding, full_matrices=False)
        max_possible = U.shape[1]
        if rank_selection == 'fixed':
            cap = n_components[mode] if n_components is not None else max_possible
            r = max(1, min(int(cap), max_possible))
        elif rank_selection == 'variance':
            cap = (n_components[mode]
                   if (n_components is not None and n_components[mode] is not None)
                   else None)
            effective_cap = min(cap, max_possible) if cap is not None else max_possible
            r = _rank_from_variance(sv, variance_threshold, max_rank=effective_cap)
        else:
            raise ValueError(
                f"rank_selection must be 'fixed' or 'variance', got {rank_selection!r}")
        Us.append(U[:, :r])
    core = tensor.copy()
    for mode, U in enumerate(Us):
        core = np.tensordot(U.conj().T, core, axes=([1], [mode]))
        core = np.moveaxis(core, 0, mode)
    recon = core
    for mode, U in enumerate(Us):
        recon = np.tensordot(U, recon, axes=([1], [mode]))
        recon = np.moveaxis(recon, 0, mode)
    return recon


# =============================================================================
# ── Mask helpers ──────────────────────────────────────────────────────────────
# =============================================================================

def compute_mask(volumes, threshold_factor=0.05, dilation_iters=3,
                 rough_patch_size=7, rough_sliding_window=8,
                 rough_search_radius=15, rough_max_patches=20,
                 search_backend='kdtree', faiss_gpu_id=0):
    """
    Compute a binary foreground mask using a fast rough LLR denoising pass.

    Directly thresholding noisy MRF singular volumes is unreliable — the mask
    will have holes inside tissue and false positives in background noise.
    Instead, a quick LLR denoising pass is applied first using aggressive
    (coarse) spatial parameters.  All L singular volumes are used, as they
    are key to correct denoising — speed comes only from the large
    sliding_window which reduces the number of reference patches processed.

    Steps
    -----
    1. Run llr_hosvd_allbins on the full (nb_bins, L, nx, ny, nz) volumes
       with large sliding_window (coarse spatial coverage, fast).
    2. Threshold the smoothed magnitude of the first singular volume (L=0)
       per bin at threshold_factor x 99th-percentile.
    3. Take the union of per-bin threshold masks.
    4. Apply binary dilation to pad the boundary by dilation_iters voxels.

    Parameters
    ----------
    volumes : ndarray (nb_bins, L, nx, ny, nz) or (L, nx, ny, nz), complex
    threshold_factor : float, default 0.05
    dilation_iters : int, default 3
    rough_patch_size : int, default 7
    rough_sliding_window : int, default 8
    rough_search_radius : int, default 15
    rough_max_patches : int, default 20
    search_backend : str, default 'kdtree'
    faiss_gpu_id : int, default 0

    Returns
    -------
    mask : bool ndarray (nx, ny, nz)
    """
    from scipy.ndimage import binary_dilation

    if volumes.ndim == 4:
        volumes = volumes[None, ...]

    nb_bins       = volumes.shape[0]
    spatial_shape = volumes.shape[2:]

    logging.info("Computing mask: rough LLR denoising of all L channels "
                 "(nb_bins=%d, L=%d, sliding_window=%d) ...",
                 nb_bins, volumes.shape[1], rough_sliding_window)

    smoothed = llr_hosvd_allbins(
        volumes,
        patch_size=rough_patch_size,
        search_radius=rough_search_radius,
        max_patches=rough_max_patches,
        sliding_window=rough_sliding_window,
        n_components=None,
        rank_selection='variance',
        variance_threshold=0.90,
        search_backend=search_backend,
        faiss_gpu_id=faiss_gpu_id,
        bins_patch_size=None,
        mask=None,
        use_mask=False,
    )

    mask = np.zeros(spatial_shape, dtype=bool)
    for b in range(nb_bins):
        mag = np.abs(smoothed[b, 0])
        thr = threshold_factor * np.percentile(mag, 99)
        mask |= (mag > thr)

    if dilation_iters > 0:
        struct = np.ones((3, 3, 3), dtype=bool)
        for _ in range(dilation_iters):
            mask = binary_dilation(mask, structure=struct)

    logging.info("Mask: %d / %d voxels (%.1f%%)",
                 mask.sum(), mask.size, 100 * mask.mean())
    return mask


def _dilate_mask_for_patches(mask, patch_size):
    from scipy.ndimage import binary_dilation
    r = patch_size // 2
    if r == 0:
        return mask
    struct = np.ones((2 * r + 1,) * 3, dtype=bool)
    return binary_dilation(mask, structure=struct)


def _filter_centers_by_mask(centers, dilated_mask):
    return dilated_mask[centers[:, 0], centers[:, 1], centers[:, 2]]


# =============================================================================
# ── Similarity search ─────────────────────────────────────────────────────────
# =============================================================================

def _build_patch_index_cpu(vol_shape, patch_size, step):
    """Pre-compute flat (P, N) voxel index arrays for GPU gather.
    Patch size is clamped per axis so 2-D volumes are handled transparently.
    """
    _, nx, ny, nz = vol_shape
    sx = min(patch_size, nx)
    sy = min(patch_size, ny)
    sz = min(patch_size, nz)
    xs = np.arange(0, nx - sx + 1, step, dtype=np.int32)
    ys = np.arange(0, ny - sy + 1, step, dtype=np.int32)
    zs = np.arange(0, nz - sz + 1, step, dtype=np.int32)
    oi, oj, ok = np.meshgrid(np.arange(sx, dtype=np.int32),
                              np.arange(sy, dtype=np.int32),
                              np.arange(sz, dtype=np.int32), indexing='ij')
    offsets = np.stack([oi.ravel(), oj.ravel(), ok.ravel()], axis=1)
    cx, cy, cz = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=1).astype(np.int32)
    ix = (centers[:, 0:1] + offsets[:, 0]).astype(np.int32)
    iy = (centers[:, 1:2] + offsets[:, 1]).astype(np.int32)
    iz = (centers[:, 2:3] + offsets[:, 2]).astype(np.int32)
    return centers, ix, iy, iz


def _build_groups_kdtree(patches_vec, centers, search_radius, max_patches):
    """cKDTree radius search + L2 ranking. search_radius must be an int."""
    n_patches = patches_vec.shape[0]
    groups    = np.full((n_patches, max_patches), -1, dtype=np.int32)
    k_actual  = np.zeros(n_patches, dtype=np.int32)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centers)
        use_tree = True
    except ImportError:
        use_tree = False
    for ref_idx in range(n_patches):
        if use_tree:
            cands = np.array(
                tree.query_ball_point(centers[ref_idx], r=search_radius),
                dtype=np.int32)
        else:
            diff  = np.abs(centers - centers[ref_idx])
            cands = np.where(np.all(diff <= search_radius, axis=1))[0].astype(np.int32)
        diffs = patches_vec[cands] - patches_vec[ref_idx]
        l2    = (diffs.real ** 2 + diffs.imag ** 2).sum(axis=1)
        order = np.argsort(l2)
        K     = min(max_patches, len(order))
        groups[ref_idx, :K] = cands[order[:K]]
        k_actual[ref_idx]   = K
    return groups, k_actual


def _build_groups_faiss(patches_vec, centers, search_radius, max_patches,
                        use_gpu=False, gpu_id=0):
    """
    FAISS exact K-NN search.

    search_radius : int or -1
        int  — spatially constrained: after FAISS search, keep only
               neighbours whose patch centres lie within search_radius
               voxels (Chebyshev norm).
        None — global search: keep the K nearest neighbours by patch-
               content L2 distance regardless of spatial position.
               n_query is set to exactly max_patches in this case since
               no post-filtering will discard any results.
    """
    import faiss

    n_patches  = patches_vec.shape[0]
    global_search = search_radius == -1

    feats = np.concatenate(
        [patches_vec.real, patches_vec.imag], axis=1
    ).astype(np.float32)
    dim = feats.shape[1]

    # For global search we query exactly max_patches neighbours — no spatial
    # filtering will discard any, so no need for the 8x buffer.
    # For spatially constrained search we query more to absorb filter losses.
    if global_search:
        n_query = min(n_patches, max_patches)
    else:
        n_query = min(n_patches, max(max_patches * 8, 128))

    if use_gpu:
        res = faiss.StandardGpuResources()
        res.setTempMemory(8 * 1024 * 1024 * 1024)
        res.setPinnedMemory(512 * 1024 * 1024)
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = gpu_id
        index = faiss.GpuIndexFlatL2(res, dim, cfg)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(feats)
    _, I = index.search(feats, n_query)     # (P, n_query), sorted by L2 asc

    if global_search:
        # No spatial filter — all returned neighbours are valid.
        # Guarantee self at position 0 (FAISS almost always returns it first
        # at distance 0, but we enforce it explicitly for correctness).
        groups   = np.full((n_patches, max_patches), -1, dtype=np.int32)
        k_actual = np.full(n_patches, min(n_query, max_patches), dtype=np.int32)
        groups[:, :k_actual[0]] = I[:, :k_actual[0]].astype(np.int32)
        groups[:, 0] = np.arange(n_patches, dtype=np.int32)
        return groups, k_actual

    # Spatially constrained search — vectorised Chebyshev filter
    centers_f    = centers.astype(np.float32)
    I_safe       = np.where(I < 0, 0, I)
    neighbor_c   = centers_f[I_safe]
    ref_c        = centers_f[:, np.newaxis, :]
    spatial_dist = np.abs(neighbor_c - ref_c).max(axis=2)
    valid        = (I >= 0) & (spatial_dist <= search_radius)
    self_cols    = (I == np.arange(n_patches)[:, np.newaxis]).argmax(axis=1)
    valid[np.arange(n_patches), self_cols] = True

    order    = np.argsort(~valid, axis=1, kind='stable')
    I_sorted = np.take_along_axis(I, order, axis=1)
    k_actual = valid.sum(axis=1).clip(1, max_patches).astype(np.int32)

    groups = np.full((n_patches, max_patches), -1, dtype=np.int32)
    groups[:, :max_patches] = np.where(
        I_sorted[:, :max_patches] >= 0,
        I_sorted[:, :max_patches], -1)
    groups[:, 0] = np.arange(n_patches, dtype=np.int32)

    return groups, k_actual


def _build_groups(patches_vec, centers, search_radius, max_patches,
                  backend='kdtree', faiss_gpu_id=0):
    """
    Dispatch similarity search to the requested backend.

    Parameters
    ----------
    search_radius : int or None
        int  — spatially constrained search (all backends).
        None — global search, faiss_cpu or faiss_gpu only.
               Raises ValueError if backend='kdtree'.
    """
    if search_radius==-1 and backend == 'kdtree':
        raise ValueError(
            "search_radius=-1 (global search) requires backend='faiss_cpu' "
            "or 'faiss_gpu'. kdtree always needs a finite radius.")

    if backend == 'kdtree':
        print("Building groups with cKDTree radius search (CPU)...")
        return _build_groups_kdtree(patches_vec, centers, search_radius, max_patches)
    elif backend == 'faiss_cpu':
        print("Building groups with FAISS exact K-NN search (CPU)...")
        return _build_groups_faiss(patches_vec, centers, search_radius, max_patches,
                                   use_gpu=False)
    elif backend == 'faiss_gpu':
        print("Building groups with FAISS exact K-NN search (GPU)...")
        return _build_groups_faiss(patches_vec, centers, search_radius, max_patches,
                                   use_gpu=True, gpu_id=faiss_gpu_id)
    else:
        raise ValueError(
            f"search_backend must be 'kdtree', 'faiss_cpu', or 'faiss_gpu', "
            f"got {backend!r}")


# =============================================================================
# ── CPU HOSVD — single volume ─────────────────────────────────────────────────
# =============================================================================

def llr_hosvd_regularization(
    volume,
    patch_size=4,
    search_radius=10,
    max_patches=20,
    sliding_window=2,
    n_components=None,
    rank_selection='fixed',
    variance_threshold=0.95,
    search_backend='kdtree',
    faiss_gpu_id=0,
    mask=None,
):
    """
    LLR-HOSVD regularization for one (L, nx, ny, nz) volume — CPU.

    Parameters
    ----------
    volume : ndarray (L, nx, ny, nz), complex
    patch_size, max_patches, sliding_window : int
    search_radius : int or None
        int  — spatially constrained patch search.
        None — global patch search (faiss_cpu or faiss_gpu only).
    n_components : tuple(3) or None
    rank_selection : {'fixed', 'variance'}
    variance_threshold : float in (0, 1]
    search_backend : {'kdtree', 'faiss_cpu', 'faiss_gpu'}
    faiss_gpu_id : int
    mask : bool ndarray (nx, ny, nz) or None

    Returns
    -------
    denoised : ndarray (L, nx, ny, nz), complex
    """
    L, nx, ny, nz = volume.shape
    s = patch_size
    N = s ** 3
    if rank_selection == 'fixed' and n_components is None:
        n_components = (L, max(1, N // 4), max(1, max_patches // 2))

    patches, centers, _ = _extract_patches_3d(volume, s, step=sliding_window)
    n_patches_all = patches.shape[0]
    _, _, sx, sy, sz = patches.shape

    if mask is not None:
        dilated = _dilate_mask_for_patches(mask, patch_size)
        keep    = _filter_centers_by_mask(centers, dilated)
        patches = patches[keep]
        centers = centers[keep]
        logging.info("Mask: keeping %d / %d patches (%.0f%%)",
                     keep.sum(), n_patches_all, 100 * keep.mean())

    n_patches   = patches.shape[0]
    N_actual    = sx * sy * sz
    patches_vec = patches.reshape(n_patches, L * N_actual)

    accum  = np.zeros_like(volume)
    weight = np.zeros((nx, ny, nz), dtype=np.float32)

    groups, k_actual = _build_groups(
        patches_vec, centers, search_radius, max_patches,
        backend=search_backend, faiss_gpu_id=faiss_gpu_id)

    for ref_idx in tqdm(range(n_patches)):
        K = k_actual[ref_idx]
        if K == 0:
            continue
        selected = groups[ref_idx, :K]
        group    = patches[selected].reshape(K, L, N_actual)
        tensor   = group.transpose(1, 2, 0)

        if n_components is not None:
            k_cap = max(1, min(n_components[2], K) if n_components[2] is not None else K)
            nc = (max(1, n_components[0]), max(1, n_components[1]), k_cap)
        else:
            nc = None

        tensor_den = _truncated_hosvd(tensor, nc, rank_selection, variance_threshold)
        group_den  = tensor_den.transpose(2, 0, 1).reshape(K, L, sx, sy, sz)

        for k_idx, patch_idx in enumerate(selected):
            px, py, pz = (int(v) for v in centers[patch_idx])
            accum[:, px:px+sx, py:py+sy, pz:pz+sz] += group_den[k_idx]
            weight[px:px+sx, py:py+sy, pz:pz+sz]   += 1.0

    weight   = np.where(weight == 0, 1.0, weight)
    denoised = accum / weight[None, :, :, :]

    if mask is not None:
        denoised[:, ~mask] = 0.0

    return denoised


# =============================================================================
# ── CPU HOSVD — all bins ──────────────────────────────────────────────────────
# =============================================================================

def llr_hosvd_allbins(
    volumes,
    patch_size=4,
    search_radius=10,
    max_patches=20,
    sliding_window=2,
    n_components=None,
    rank_selection='fixed',
    variance_threshold=0.95,
    search_backend='kdtree',
    faiss_gpu_id=0,
    bins_patch_size=None,
    mask=None,
    use_mask=False,
    mask_threshold_factor=0.05,
    mask_dilation_iters=3,
):
    """
    LLR-HOSVD regularization for all bins of (nb_bins, L, nx, ny, nz) — CPU.

    Parameters
    ----------
    volumes : ndarray (nb_bins, L, nx, ny, nz), complex
    patch_size, max_patches, sliding_window : int
    search_radius : int or None
        int  — spatially constrained patch search.
        None — global patch search (faiss_cpu or faiss_gpu only).
    n_components : tuple(3) or None
    rank_selection : {'fixed', 'variance'}
    variance_threshold : float in (0, 1]
    search_backend : {'kdtree', 'faiss_cpu', 'faiss_gpu'}
    faiss_gpu_id : int
    bins_patch_size : int or None
    mask : bool ndarray (nx, ny, nz) or None
    use_mask : bool, default False
    mask_threshold_factor : float, default 0.05
    mask_dilation_iters : int, default 3

    Returns
    -------
    denoised : ndarray (nb_bins, L, nx, ny, nz), complex
    """
    nb_bins, L, nx, ny, nz = volumes.shape

    active_mask = None
    if use_mask:
        if mask is not None:
            active_mask = mask.astype(bool)
            logging.info("Using provided mask (%d voxels, %.1f%%)",
                         active_mask.sum(), 100 * active_mask.mean())
        else:
            logging.info("Computing mask from first singular volumes ...")
            active_mask = compute_mask(
                volumes,
                threshold_factor=mask_threshold_factor,
                dilation_iters=mask_dilation_iters)

    if bins_patch_size is None:
        denoised = np.empty_like(volumes)
        for b in range(nb_bins):
            tag = (f"threshold={variance_threshold}"
                   if rank_selection == 'variance'
                   else f"n_components={n_components}")
            sr_tag   = "global" if search_radius is None else search_radius
            mask_tag = f", mask={active_mask is not None}" if use_mask else ""
            print(f"  LLR-HOSVD bin {b+1}/{nb_bins}  "
                  f"[{rank_selection}, {tag}, search={search_backend}, "
                  f"radius={sr_tag}{mask_tag}] ...")
            denoised[b] = llr_hosvd_regularization(
                volumes[b],
                patch_size=patch_size,
                search_radius=search_radius,
                max_patches=max_patches,
                sliding_window=sliding_window,
                n_components=n_components,
                rank_selection=rank_selection,
                variance_threshold=variance_threshold,
                search_backend=search_backend,
                faiss_gpu_id=faiss_gpu_id,
                mask=active_mask,
            )
        return denoised

    # ---- joint bin path --------------------------------------------------
    bp = int(bins_patch_size)
    if bp < 1 or bp > nb_bins:
        raise ValueError(
            f"bins_patch_size must be in [1, nb_bins={nb_bins}], got {bp}")

    s         = patch_size
    C         = bp * L
    n_bin_pos = nb_bins - bp + 1

    tag    = (f"threshold={variance_threshold}"
              if rank_selection == 'variance' else f"n_components={n_components}")
    sr_tag = "global" if search_radius is None else search_radius
    mask_tag = f", mask={active_mask is not None}" if use_mask else ""
    print(f"  LLR-HOSVD joint bins  [bp={bp}, {rank_selection}, {tag}, "
          f"search={search_backend}, radius={sr_tag}{mask_tag}] ...")

    if active_mask is not None:
        dilated_mask = _dilate_mask_for_patches(active_mask, patch_size)
    else:
        dilated_mask = None

    accum  = np.zeros_like(volumes)
    weight = np.zeros((nb_bins, nx, ny, nz), dtype=np.float32)

    for b0 in range(n_bin_pos):
        vol_joint = volumes[b0:b0+bp].reshape(C, nx, ny, nz)

        patches, centers, _ = _extract_patches_3d(vol_joint, s, step=sliding_window)
        n_patches_all = patches.shape[0]
        _, _, sx, sy, sz = patches.shape
        N_actual = sx * sy * sz

        if dilated_mask is not None:
            keep    = _filter_centers_by_mask(centers, dilated_mask)
            patches = patches[keep]
            centers = centers[keep]
            logging.info("  bin pos %d: keeping %d / %d patches (%.0f%%)",
                         b0+1, keep.sum(), n_patches_all, 100 * keep.mean())

        n_patches    = patches.shape[0]
        patches_mean = patches.reshape(n_patches, bp, L * N_actual).mean(axis=1)

        groups, k_actual = _build_groups(
            patches_mean, centers, search_radius, max_patches,
            backend=search_backend, faiss_gpu_id=faiss_gpu_id)

        for ref_idx in range(n_patches):
            K = k_actual[ref_idx]
            if K == 0:
                continue
            selected = groups[ref_idx, :K]
            group    = patches[selected].reshape(K, C, N_actual)
            tensor   = group.transpose(1, 2, 0)

            if n_components is not None:
                k_cap = max(1, min(n_components[2], K)
                            if n_components[2] is not None else K)
                nc = (max(1, n_components[0]), max(1, n_components[1]), k_cap)
            else:
                nc = None

            tensor_den = _truncated_hosvd(tensor, nc, rank_selection, variance_threshold)
            group_den  = tensor_den.transpose(2, 0, 1).reshape(K, bp, L, sx, sy, sz)

            for k_idx, patch_idx in enumerate(selected):
                px, py, pz = (int(v) for v in centers[patch_idx])
                for bi in range(bp):
                    accum[b0+bi, :, px:px+sx, py:py+sy, pz:pz+sz] += group_den[k_idx, bi]
                    weight[b0+bi, px:px+sx, py:py+sy, pz:pz+sz]   += 1.0

        logging.info("  bin position %d/%d done", b0+1, n_bin_pos)

    weight   = np.where(weight == 0, 1.0, weight)
    denoised = accum / weight[:, None, :, :, :]

    if active_mask is not None:
        denoised[:, :, ~active_mask] = 0.0

    return denoised


# =============================================================================
# ── GPU HOSVD ─────────────────────────────────────────────────────────────────
# =============================================================================

def _batched_hosvd_torch(tensors, n_components, rank_selection, variance_threshold):
    import torch
    n_batch   = tensors.shape[0]
    orig_dims = [tensors.shape[1], tensors.shape[2], tensors.shape[3]]
    Us        = []

    for mode in range(3):
        unf     = torch.moveaxis(tensors, mode + 1, 1)
        unf     = unf.reshape(n_batch, orig_dims[mode], -1)
        U, S, _ = torch.linalg.svd(unf, full_matrices=False)
        k_max   = S.shape[1]

        if rank_selection == 'fixed':
            cap = n_components[mode] if n_components is not None else k_max
            r   = max(1, min(int(cap), k_max))
            U_t = U[:, :, :r]
        else:
            cap     = (n_components[mode]
                       if (n_components is not None and n_components[mode] is not None)
                       else None)
            eff_cap = min(int(cap), k_max) if cap is not None else k_max
            S2      = S ** 2
            totals  = S2.sum(dim=1, keepdim=True)
            safe_t  = torch.where(totals == 0, torch.ones_like(totals), totals)
            cumvar  = torch.cumsum(S2 / safe_t, dim=1)
            exceeded = cumvar >= variance_threshold
            any_ex   = exceeded.any(dim=1)
            raw_rank = exceeded.int().argmax(dim=1) + 1
            ranks    = torch.where(any_ex, raw_rank,
                                    torch.full((n_batch,), k_max,
                                               dtype=torch.int32,
                                               device=tensors.device))
            ranks    = ranks.clamp(1, eff_cap)
            col_idx  = torch.arange(k_max, device=tensors.device)
            mask_t   = col_idx[None, :] < ranks[:, None]
            U_t      = U * mask_t[:, None, :].to(U.dtype)

        Us.append(U_t)

    d    = list(orig_dims)
    core = tensors.clone()
    for mode, U_t in enumerate(Us):
        other   = [m for m in range(3) if m != mode]
        unf     = torch.moveaxis(core, mode + 1, 1).reshape(n_batch, d[mode], -1)
        r_m     = U_t.shape[2]
        new_unf = torch.matmul(U_t.conj().transpose(1, 2), unf)
        core    = new_unf.reshape(n_batch, r_m, d[other[0]], d[other[1]])
        core    = torch.moveaxis(core, [1, 2, 3], [mode+1, other[0]+1, other[1]+1])
        d[mode] = r_m

    recon = core
    for mode, U_t in enumerate(Us):
        other      = [m for m in range(3) if m != mode]
        dim_m      = recon.shape[mode + 1]
        unf        = torch.moveaxis(recon, mode + 1, 1).reshape(n_batch, dim_m, -1)
        new_unf    = torch.matmul(U_t, unf)
        orig_dim_m = U_t.shape[1]
        d_o0       = recon.shape[other[0] + 1]
        d_o1       = recon.shape[other[1] + 1]
        recon      = new_unf.reshape(n_batch, orig_dim_m, d_o0, d_o1)
        recon      = torch.moveaxis(recon, [1, 2, 3], [mode+1, other[0]+1, other[1]+1])

    return recon


def llr_hosvd_regularization_gpu(
    volume,
    patch_size=4,
    search_radius=10,
    max_patches=20,
    sliding_window=2,
    n_components=None,
    rank_selection='fixed',
    variance_threshold=0.95,
    hosvd_batch_size=256,
    device=None,
    search_backend='faiss_gpu',
    faiss_gpu_id=0,
    mask=None,
):
    """
    GPU-accelerated LLR-HOSVD for one (C, nx, ny, nz) volume.

    Parameters
    ----------
    volume : ndarray (C, nx, ny, nz), complex
    patch_size, max_patches, sliding_window : int
    search_radius : int or None
        int  — spatially constrained patch search.
        None — global patch search (faiss_cpu or faiss_gpu only).
    n_components : tuple(3) or None
    rank_selection : {'fixed', 'variance'}
    variance_threshold : float in (0, 1]
    hosvd_batch_size : int
    device : str or torch.device or None
    search_backend : {'kdtree', 'faiss_cpu', 'faiss_gpu'}
    faiss_gpu_id : int
    mask : bool ndarray (nx, ny, nz) or None

    Returns
    -------
    denoised : ndarray (C, nx, ny, nz), same dtype as input
    """
    import torch

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    C, nx, ny, nz = volume.shape
    s = patch_size
    N = s ** 3

    if rank_selection == 'fixed' and n_components is None:
        n_components = (C, max(1, N // 4), max(1, max_patches // 2))

    centers, ix_cpu, iy_cpu, iz_cpu = _build_patch_index_cpu(
        volume.shape, patch_size, sliding_window)
    n_patches_all = centers.shape[0]

    if mask is not None:
        dilated = _dilate_mask_for_patches(mask, patch_size)
        keep    = _filter_centers_by_mask(centers, dilated)
        centers = centers[keep]
        ix_cpu  = ix_cpu[keep]
        iy_cpu  = iy_cpu[keep]
        iz_cpu  = iz_cpu[keep]
        logging.info("[GPU] Mask: keeping %d / %d patches (%.0f%%)",
                     keep.sum(), n_patches_all, 100 * keep.mean())

    n_patches = centers.shape[0]

    patches_vec_cpu = volume[:, ix_cpu, iy_cpu, iz_cpu]
    patches_vec_cpu = patches_vec_cpu.transpose(1, 0, 2).reshape(n_patches, C * N)

    groups_cpu, k_actual_cpu = _build_groups(
        patches_vec_cpu, centers, search_radius, max_patches,
        backend=search_backend, faiss_gpu_id=faiss_gpu_id)

    vol_t      = torch.from_numpy(volume).to(device)
    ix_t       = torch.from_numpy(ix_cpu).to(device)
    iy_t       = torch.from_numpy(iy_cpu).to(device)
    iz_t       = torch.from_numpy(iz_cpu).to(device)
    groups_t   = torch.from_numpy(groups_cpu).to(device)
    k_actual_t = torch.from_numpy(k_actual_cpu).to(device)

    nxyz      = nx * ny * nz
    flat_idx  = ix_t * (ny * nz) + iy_t * nz + iz_t
    patches_t = vol_t[:, ix_t, iy_t, iz_t].permute(1, 0, 2)

    accum_r = torch.zeros(C, nxyz, dtype=torch.float32, device=device)
    accum_i = torch.zeros(C, nxyz, dtype=torch.float32, device=device)
    weight  = torch.zeros(nxyz,    dtype=torch.float32, device=device)

    for start in range(0, n_patches, hosvd_batch_size):
        end = min(start + hosvd_batch_size, n_patches)
        n_b = end - start

        g_b  = groups_t[start:end]
        ka_b = k_actual_t[start:end]

        valid_k       = g_b >= 0
        safe_g        = g_b.clamp(min=0)
        group_patches = patches_t[safe_g]
        group_patches = group_patches * valid_k[:, :, None, None]
        tensors_b     = group_patches.permute(0, 2, 3, 1)

        if n_components is not None:
            k_cap = max(1, min(n_components[2], max_patches)
                        if n_components[2] is not None else max_patches)
            nc_b = (max(1, n_components[0]), max(1, n_components[1]), k_cap)
        else:
            nc_b = None

        recon_b = _batched_hosvd_torch(
            tensors_b, nc_b, rank_selection, variance_threshold)

        for k in range(max_patches):
            valid = ka_b > k
            if not valid.any():
                break
            member_idx = g_b[:, k].clamp(min=0)
            flat_m     = flat_idx[member_idx]
            patch_den  = recon_b[:, :, :, k]
            patch_den  = torch.where(valid[:, None, None],
                                      patch_den, torch.zeros_like(patch_den))
            flat_m_r = flat_m.reshape(-1)
            for c in range(C):
                flat_v = patch_den[:, c, :].reshape(-1)
                accum_r[c].index_add_(0, flat_m_r, flat_v.real.float())
                accum_i[c].index_add_(0, flat_m_r, flat_v.imag.float())
            flat_w = valid.float().unsqueeze(1).expand(-1, N).reshape(-1)
            weight.index_add_(0, flat_m_r, flat_w)

    weight   = weight.clamp(min=1.0)
    accum_r /= weight[None, :]
    accum_i /= weight[None, :]
    denoised = torch.complex(accum_r, accum_i).reshape(C, nx, ny, nz)
    result   = denoised.cpu().numpy().astype(volume.dtype)

    if mask is not None:
        result[:, ~mask] = 0.0

    return result


def llr_hosvd_allbins_gpu(
    volumes,
    patch_size=4,
    search_radius=10,
    max_patches=20,
    sliding_window=2,
    n_components=None,
    rank_selection='fixed',
    variance_threshold=0.95,
    hosvd_batch_size=256,
    device=None,
    search_backend='faiss_gpu',
    faiss_gpu_id=0,
    bins_patch_size=None,
    mask=None,
    use_mask=False,
    mask_threshold_factor=0.05,
    mask_dilation_iters=3,
):
    """
    GPU-accelerated LLR-HOSVD for all bins of (nb_bins, L, nx, ny, nz).

    Parameters
    ----------
    volumes : ndarray (nb_bins, L, nx, ny, nz), complex
    patch_size, max_patches, sliding_window : int
    search_radius : int or None
        int  — spatially constrained patch search.
        None — global patch search (faiss_cpu or faiss_gpu only).
    n_components : tuple(3) or None
    rank_selection : {'fixed', 'variance'}
    variance_threshold : float in (0, 1]
    hosvd_batch_size : int
    device : str or torch.device or None
    search_backend : {'kdtree', 'faiss_cpu', 'faiss_gpu'}
    faiss_gpu_id : int
    bins_patch_size : int or None
    mask : bool ndarray (nx, ny, nz) or None
    use_mask : bool, default False
    mask_threshold_factor : float, default 0.05
    mask_dilation_iters : int, default 3

    Returns
    -------
    denoised : ndarray (nb_bins, L, nx, ny, nz), same dtype as volumes
    """
    import torch

    nb_bins, L, nx, ny, nz = volumes.shape

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    active_mask = None
    if use_mask:
        if mask is not None:
            active_mask = mask.astype(bool)
            logging.info("Using provided mask (%d voxels, %.1f%%)",
                         active_mask.sum(), 100 * active_mask.mean())
        else:
            logging.info("Computing mask from first singular volumes ...")
            active_mask = compute_mask(
                volumes,
                threshold_factor=mask_threshold_factor,
                dilation_iters=mask_dilation_iters)

    tag    = (f"threshold={variance_threshold}"
              if rank_selection == 'variance' else f"n_components={n_components}")
    sr_tag = "global" if search_radius is None else search_radius
    mask_tag = f", mask=True" if active_mask is not None else ""

    if bins_patch_size is None:
        denoised = np.empty_like(volumes)
        for b in range(nb_bins):
            print(f"  LLR-HOSVD (GPU/{device}) bin {b+1}/{nb_bins}  "
                  f"[{rank_selection}, {tag}, search={search_backend}, "
                  f"radius={sr_tag}, batch={hosvd_batch_size}{mask_tag}] ...")
            denoised[b] = llr_hosvd_regularization_gpu(
                volumes[b],
                patch_size=patch_size,
                search_radius=search_radius,
                max_patches=max_patches,
                sliding_window=sliding_window,
                n_components=n_components,
                rank_selection=rank_selection,
                variance_threshold=variance_threshold,
                hosvd_batch_size=hosvd_batch_size,
                device=device,
                search_backend=search_backend,
                faiss_gpu_id=faiss_gpu_id,
                mask=active_mask,
            )
        return denoised

    bp = int(bins_patch_size)
    if bp < 1 or bp > nb_bins:
        raise ValueError(
            f"bins_patch_size must be in [1, nb_bins={nb_bins}], got {bp}")

    C         = bp * L
    n_bin_pos = nb_bins - bp + 1

    print(f"  LLR-HOSVD (GPU/{device}) joint bins  "
          f"[bp={bp}, C={C}, {rank_selection}, {tag}, "
          f"search={search_backend}, radius={sr_tag}, "
          f"batch={hosvd_batch_size}{mask_tag}] ...")

    accum_r = np.zeros((nb_bins, L, nx * ny * nz), dtype=np.float32)
    accum_i = np.zeros((nb_bins, L, nx * ny * nz), dtype=np.float32)
    weight  = np.zeros((nb_bins, nx * ny * nz),    dtype=np.float32)

    for b0 in range(n_bin_pos):
        logging.info("[GPU] Joint bin position %d/%d", b0+1, n_bin_pos)
        vol_joint = volumes[b0:b0+bp].reshape(C, nx, ny, nz)

        denoised_joint = llr_hosvd_regularization_gpu(
            vol_joint,
            patch_size=patch_size,
            search_radius=search_radius,
            max_patches=max_patches,
            sliding_window=sliding_window,
            n_components=n_components,
            rank_selection=rank_selection,
            variance_threshold=variance_threshold,
            hosvd_batch_size=hosvd_batch_size,
            device=device,
            search_backend=search_backend,
            faiss_gpu_id=faiss_gpu_id,
            mask=active_mask,
        )
        denoised_joint = denoised_joint.reshape(bp, L, nx * ny * nz)

        for bi in range(bp):
            accum_r[b0+bi] += denoised_joint[bi].real
            accum_i[b0+bi] += denoised_joint[bi].imag
            weight[b0+bi]  += 1.0

    weight   = np.where(weight == 0, 1.0, weight)
    denoised = ((accum_r + 1j * accum_i) / weight[:, None, :]
                ).reshape(nb_bins, L, nx, ny, nz).astype(volumes.dtype)

    if active_mask is not None:
        denoised[:, :, ~active_mask] = 0.0

    return denoised


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    nb_bins, L, nx, ny, nz = 5, 6, 16, 16, 16
    vol = (np.random.randn(nb_bins, L, nx, ny, nz)
           + 1j * np.random.randn(nb_bins, L, nx, ny, nz)).astype(np.complex64)
    vol[:, :, 4:12, 4:12, 4:12] *= 10

    print("=== per-bin, radius=5 (kdtree) ===")
    den1 = llr_hosvd_allbins(
        vol, patch_size=3, search_radius=5, max_patches=8, sliding_window=3,
        n_components=(6, 4, 4), rank_selection='fixed', search_backend='kdtree')
    assert den1.shape == vol.shape
    print("Output:", den1.shape, den1.dtype)

    print("\n=== per-bin, radius=None (global, faiss_cpu) ===")
    den2 = llr_hosvd_allbins(
        vol, patch_size=3, search_radius=None, max_patches=8, sliding_window=3,
        n_components=(6, 4, 4), rank_selection='fixed', search_backend='faiss_cpu')
    assert den2.shape == vol.shape
    print("Output:", den2.shape, den2.dtype)

    print("\n=== joint bins bp=3, radius=None (global) ===")
    den3 = llr_hosvd_allbins(
        vol, patch_size=3, search_radius=None, max_patches=8, sliding_window=3,
        n_components=(18, 4, 4), rank_selection='fixed', search_backend='faiss_cpu',
        bins_patch_size=3)
    assert den3.shape == vol.shape
    print("Output:", den3.shape, den3.dtype)

    print("\n=== search_radius=None with kdtree should raise ===")
    try:
        llr_hosvd_allbins(vol, patch_size=3, search_radius=None, max_patches=8,
                          sliding_window=3, search_backend='kdtree')
        assert False, "should have raised"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print("\nAll smoke tests passed.")
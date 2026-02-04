import numpy as np
def rowwise_minmax_normalize(dist, fill_value=-1.0):
    T, N, _ = dist.shape
    finite = np.isfinite(dist)
    dist_nan = np.where(finite, dist, np.nan)

    mins = np.nanmin(dist_nan, axis=2)
    maxs = np.nanmax(dist_nan, axis=2)

    ranges = maxs - mins
    ranges[ranges == 0] = 1

    norm = (dist - mins[:, :, None]) / ranges[:, :, None]
    norm[~finite] = fill_value
    return norm
def minmax_norm_po(data):
    global_min = data.min()
    global_max = data.max()
    return (data - global_min) / (global_max - global_min)
def minmax_norm_po_local(data):
    min_vals = data.min(axis=1, keepdims=True)
    max_vals = data.max(axis=1, keepdims=True)
    return (data - min_vals) / (max_vals - min_vals)

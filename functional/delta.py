import numpy as np


def delta_hyperbolicity(d: np.ndarray):
    # TODO: make our own implementation
    p = 0
    row = d[p, :][np.newaxis, :]
    col = d[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - d)
    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)

    return np.max(maxmin - XY_p)

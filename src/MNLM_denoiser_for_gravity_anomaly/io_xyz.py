"""Load and save regularly sampled gravity XYZ files.

Unlike transform_in / transform_out (kept as MATLAB-faithful ports),
this module infers the grid from unique coordinates and writes
columns as (X, Y, value) with no swap.
"""

from pathlib import Path

import numpy as np


def load_xyz_grid(path):
    """Load an XYZ file into 2D grids.

    Points are assumed to be listed on a regular grid. The fast axis is
    inferred from which coordinate changes between the first two rows.

    Parameters
    ----------
    path : str or Path
        ASCII file with at least three columns: X, Y, value.

    Returns
    -------
    values, xx, yy : 2D arrays
        Value grid and matching X/Y coordinate grids. For the synthetic
        files in this repo the layout is (n_x, n_y) with X slow, Y fast —
        the same as transform_in(..., 121, 121).

    Raises
    ------
    ValueError
        If the file has fewer than three columns or the unique X/Y
        counts do not fill a regular grid.
    """
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path} must have at least 3 columns (X, Y, value)")
    x, y, v = data[:, 0], data[:, 1], data[:, 2]
    ux = np.unique(x)
    uy = np.unique(y)
    n_x, n_y = ux.size, uy.size
    if n_x * n_y != v.size:
        raise ValueError(
            f"{path}: {v.size} points do not fill a {n_x} x {n_y} unique-X/Y grid"
        )

    y_is_fast = abs(y[1] - y[0]) >= abs(x[1] - x[0])
    if y_is_fast:
        values = v.reshape(n_x, n_y)
        xx = x.reshape(n_x, n_y)
        yy = y.reshape(n_x, n_y)
    else:
        values = v.reshape(n_y, n_x)
        xx = x.reshape(n_y, n_x)
        yy = y.reshape(n_y, n_x)
    return values, xx, yy


def save_xyz(path, xx, yy, values):
    """Write a grid as an ASCII XYZ file: columns X, Y, value.

    Parameters
    ----------
    path : str or Path
        Destination file. Parent directories are created if needed.
    xx, yy, values : array_like
        Coordinate and value grids; all are raveled in C order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.column_stack((np.ravel(xx), np.ravel(yy), np.ravel(values)))
    np.savetxt(path, xyz, fmt="%16.7e")

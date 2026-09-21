"""Python equivalent of transform_in.m

Reshape an XYZ point list (columns X, Y, value) into 2D grids.
"""

import numpy as np


def transform_in(Data, y, x):
    """Convert an (N*M) x 3 XYZ array into 2D matrices.

    Parameters
    ----------
    Data : 2D array, shape (N*M, 3)
        Columns are X, Y, value; points listed row-block by row-block.
    y : int
        Number of rows (blocks) in the output grids (N).
    x : int
        Number of points per row / block length (M).

    Returns
    -------
    DATA, xx, yy : 2D arrays, shape (N, M)
        Value grid and the corresponding X and Y coordinate grids.
        This matches MATLAB ``transform_in``: X varies slowly (rows),
        Y varies fast (columns).
    """
    Data = np.asarray(Data, dtype=float)
    data = Data[:, 2]
    X = Data[:, 0]
    Y = Data[:, 1]
    M = x
    N = y
    # MATLAB: DATA(k,:) = data(1+(k-1)*M : k*M) for k = 1..N
    DATA = data[:N * M].reshape(N, M)
    xx = X[:N * M].reshape(N, M)
    yy = Y[:N * M].reshape(N, M)
    return DATA, xx, yy

"""Python equivalent of transform_out.m

Flatten a 2D grid back into an XYZ point list (columns X, Y, value).
"""

import numpy as np


def transform_out(data, y, x):
    """Convert a 2D grid into an (N*M) x 3 XYZ array.

    Parameters
    ----------
    data : 2D array, shape (N, M)
        The value grid.
    y : 1D array, length N
        Coordinate assigned per row block (one value repeated M times).
    x : 1D array, length M
        Coordinates tiled within each row block.

    Returns
    -------
    CC : 2D array, shape (N*M, 3)
        Columns ``[XX, YY, BB]`` where ``BB`` is the flattened grid.
        The MATLAB driver then swaps columns 1 and 2 so the written
        file is ``(X, Y, value)`` on this square survey.
    """
    data = np.asarray(data, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    M = len(x)
    N = len(y)
    # MATLAB loop: BB(1+(k-1)*M : k*M) = data(k,:); XX(...) = x; YY(...) = y(k)
    BB = data[:N, :M].reshape(N * M)
    XX = np.tile(x, N)
    YY = np.repeat(y, M)
    CC = np.column_stack((XX, YY, BB))
    return CC

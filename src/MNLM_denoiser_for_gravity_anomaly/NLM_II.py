"""Python equivalent of NLM_II.m

Non-local Means filter Accelerated by Integral Image
(Modified Non-local Means, MNLM).
"""

import numpy as np


def NLM_II(N, Ds, ds, h):
    """Denoise a 2D grid with the Modified Non-Local Means filter.

    Parameters
    ----------
    N : 2D array
        The original (noisy) data.
    Ds : int
        Half-width of the searching window (window size is 2*Ds+1).
    ds : int
        Half-width of the comparing window / patch (patch size is 2*ds+1).
    h : float
        Filter parameter. A bigger h filters more information about
        texture, which can lead to ambiguity of the processed result.

    Returns
    -------
    N_processed : 2D array
        The denoised data, same shape as N.

    Raises
    ------
    ValueError
        If ``h`` is zero (weights are ``exp(-S / h^2)``).
    """
    # Initialization:
    inp = np.asarray(N, dtype=float)
    h = float(h)
    if h == 0.0:
        raise ValueError("h must be nonzero (weights use exp(-S / h^2))")
    Ds = int(Ds)
    ds = int(ds)
    m, n = inp.shape
    d = 2 * ds + 1  # size of the comparing window.
    # D = 2*Ds + 1  # size of the searching window (unused, kept for reference).

    # Enlarge the size of the input data N from m x n to
    # (m+2Ds+2ds+2) x (n+2Ds+2ds+2), the new matrix is called N_enlarged
    # (equivalent of padarray(..., 'symmetric', 'both')):
    pad = ds + Ds + 1
    N_enlarged = np.pad(inp, pad, mode="symmetric")

    # Assign initial values to the matrices needed:
    N_processing = np.zeros((m, n))
    N_f = np.zeros((m, n))
    maxweight = np.zeros((m, n))
    # MATLAB: N_enlarged(1+Ds:Ds+m+2*ds+1, 1+Ds:Ds+n+2*ds+1)
    Static_patch = N_enlarged[Ds:Ds + m + 2 * ds + 1, Ds:Ds + n + 2 * ds + 1]

    # Main loop over every shift (p, q) in the searching window:
    for p in range(-Ds, Ds + 1):
        for q in range(-Ds, Ds + 1):
            # Skip the current calculation when (p,q) equals (0,0):
            if p == 0 and q == 0:
                continue
            # Calculate the integral image:
            Moving_patch = N_enlarged[Ds + p:Ds + m + 2 * ds + 1 + p,
                                      Ds + q:Ds + n + 2 * ds + 1 + q]
            difference = (Static_patch - Moving_patch) ** 2
            Integral_image = np.cumsum(np.cumsum(difference, axis=0), axis=1)
            # Calculate the (unweighted) Euclidean distance using the
            # four-corner rule of the integral image:
            S = (Integral_image[2 * ds + 1:m + 2 * ds + 1, 2 * ds + 1:n + 2 * ds + 1]
                 + Integral_image[0:m, 0:n]
                 - Integral_image[0:m, 2 * ds + 1:n + 2 * ds + 1]
                 - Integral_image[2 * ds + 1:m + 2 * ds + 1, 0:n])
            S = S / (d ** 2)
            # Calculate the weight:
            weight = np.exp(-S / (h * h))
            N_processing += weight * Moving_patch[ds + 1:ds + 1 + m, ds + 1:ds + 1 + n]
            N_f += weight
            maxweight = np.maximum(maxweight, weight)

    # The center pixel gets the maximum weight seen from any other patch:
    N_processing += maxweight * Static_patch[ds + 1:ds + 1 + m, ds + 1:ds + 1 + n]
    N_f += maxweight
    # Calculate the final result. If a pixel somehow collected no weight,
    # fall back to the original (noisy) value.
    N_processed = np.divide(N_processing, N_f, out=inp.copy(), where=N_f != 0)
    return N_processed

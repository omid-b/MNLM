"""Denoisers with a shared contract: denoise(noisy_grid, **params) -> grid.

Every function takes the *noisy* field. MNLM is the paper method; SVD,
DCT, and wavelet are comparison baselines applied to the same input.
"""

import numpy as np
import pywt
from scipy.fft import dctn, idctn

from NLM_II import NLM_II


def denoise_mnlm(noisy, Ds=8, ds=5, h=1.0):
    """Modified Non-Local Means (Ai et al. integral-image MNLM).

    Parameters
    ----------
    noisy : array_like
        Noisy 2D gravity grid.
    Ds : int, optional
        Search-window half-width (window size ``2*Ds+1``).
    ds : int, optional
        Patch half-width (patch size ``2*ds+1``).
    h : float, optional
        Filter strength; weights are ``exp(-S / h^2)``.

    Returns
    -------
    ndarray
        Denoised grid, same shape as ``noisy``.
    """
    return NLM_II(noisy, int(Ds), int(ds), float(h))


def denoise_svd(noisy, n_keep=None, sigma=None, thresh_scale=1.0):
    """SVD denoising.

    If ``sigma`` is given, soft-threshold the singular values at
    ``thresh_scale * sigma * (sqrt(m) + sqrt(n))`` — the largest singular
    value of an m x n matrix of iid noise with std sigma is about
    ``sigma * (sqrt(m) + sqrt(n))``, so this shrinks noise-level modes
    continuously as thresh_scale grows.

    Otherwise fall back to keeping the largest ``n_keep`` singular values.

    Parameters
    ----------
    noisy : array_like
        Noisy 2D gravity grid.
    n_keep : int, optional
        Number of leading singular values to keep when ``sigma`` is None.
    sigma : float, optional
        Assumed noise std. When set, rank truncation is replaced by
        soft-thresholding.
    thresh_scale : float, optional
        Multiplier on the noise-floor threshold (larger → more shrinkage).

    Returns
    -------
    ndarray
        Denoised grid, same shape as ``noisy``.
    """
    noisy = np.asarray(noisy, dtype=float)
    u, s, vt = np.linalg.svd(noisy, full_matrices=False)
    if sigma is not None:
        m, n = noisy.shape
        t = float(thresh_scale) * float(sigma) * (np.sqrt(m) + np.sqrt(n))
        s0 = np.maximum(s - t, 0.0)
        if not np.any(s0 > 0):
            s0[0] = s[0]  # never return a zero field: keep the mean/trend mode
    else:
        k = int(max(1, min(n_keep if n_keep is not None else 10, s.size)))
        s0 = s.copy()
        s0[k:] = 0.0
    return (u * s0) @ vt


def denoise_dct(noisy, k_keep=None, sigma=None, thresh_scale=1.0):
    """2D-DCT denoising.

    If ``sigma`` is given, soft-threshold the DCT coefficients at
    ``thresh_scale * sigma * sqrt(2 ln N)`` (universal threshold; the
    orthonormal DCT of iid noise has iid coefficients with std sigma).
    The DC coefficient is kept so the field mean is preserved.

    Otherwise fall back to keeping the lowest-frequency
    ``k_keep x k_keep`` block.

    Parameters
    ----------
    noisy : array_like
        Noisy 2D gravity grid.
    k_keep : int, optional
        Side length of the kept low-frequency block when ``sigma`` is None.
    sigma : float, optional
        Assumed noise std. When set, the block cutoff is replaced by
        coefficient soft-thresholding.
    thresh_scale : float, optional
        Multiplier on the universal threshold.

    Returns
    -------
    ndarray
        Denoised grid, same shape as ``noisy``.
    """
    noisy = np.asarray(noisy, dtype=float)
    spectrum = dctn(noisy, type=2, norm="ortho")
    if sigma is not None:
        t = float(thresh_scale) * float(sigma) * np.sqrt(2.0 * np.log(noisy.size))
        dc = spectrum[0, 0]
        kept = np.sign(spectrum) * np.maximum(np.abs(spectrum) - t, 0.0)
        kept[0, 0] = dc
    else:
        k = int(max(1, min(k_keep if k_keep is not None else 10, min(spectrum.shape))))
        kept = np.zeros_like(spectrum)
        kept[:k, :k] = spectrum[:k, :k]
    return idctn(kept, type=2, norm="ortho")


def estimate_sigma(noisy, wavelet="db1"):
    """Robust noise-std estimate: MAD of the finest wavelet HH detail.

    Parameters
    ----------
    noisy : array_like
        Noisy 2D grid.
    wavelet : str, optional
        Wavelet family used for the one-level decomposition.

    Returns
    -------
    float
        Estimated noise standard deviation.
    """
    noisy = np.asarray(noisy, dtype=float)
    coeffs = pywt.wavedec2(noisy, wavelet, level=1)
    hh = coeffs[-1][-1]
    return float(np.median(np.abs(hh)) / 0.6745)


def _bayes_threshold(detail, sigma):
    """BayesShrink threshold for one wavelet detail band.

    ``thr = sigma^2 / sigma_x``, where ``sigma_x`` is the estimated
    signal std in that band. If the band looks like pure noise the
    threshold is set above the largest coefficient so the band is
    zeroed.

    Parameters
    ----------
    detail : array_like
        One wavelet detail subband.
    sigma : float
        Assumed noise standard deviation.

    Returns
    -------
    float
        Soft-threshold to apply to ``detail``.
    """
    variance = float(np.mean(detail ** 2))
    signal_var = max(variance - sigma ** 2, 0.0)
    if signal_var <= 0.0:
        return float(np.max(np.abs(detail)) + 1.0)
    return float(sigma ** 2 / np.sqrt(signal_var))


def denoise_wavelet(
    noisy,
    wavelet="sym4",
    level=3,
    method="bayes",
    thresh_scale=1.0,
    mode="soft",
    sigma=None,
):
    """Wavelet denoising with BayesShrink or VisuShrink on the details.

    ``method='approx'`` keeps only the approximation (the MATLAB script's
    aggressive low-pass). Prefer ``bayes`` or ``visu`` for a real comparison.

    Parameters
    ----------
    noisy : array_like
        Noisy 2D gravity grid.
    wavelet : str, optional
        Wavelet family (default ``'sym4'``).
    level : int, optional
        Decomposition depth, clipped to the maximum the grid allows.
    method : {'bayes', 'visu', 'approx'}, optional
        Per-band BayesShrink, universal VisuShrink, or approximation only.
    thresh_scale : float, optional
        Multiplier on the computed threshold.
    mode : str, optional
        ``pywt.threshold`` mode (``'soft'`` or ``'hard'``).
    sigma : float, optional
        Assumed noise std. Estimated from the HH detail if omitted.

    Returns
    -------
    ndarray
        Denoised grid, cropped to the input shape.
    """
    noisy = np.asarray(noisy, dtype=float)
    max_level = pywt.dwt_max_level(min(noisy.shape), pywt.Wavelet(wavelet).dec_len)
    level = int(max(1, min(level, max_level)))
    coeffs = pywt.wavedec2(noisy, wavelet, level=level)
    if method == "approx":
        new_coeffs = [coeffs[0]] + [
            tuple(np.zeros_like(band) for band in details) for details in coeffs[1:]
        ]
    else:
        if sigma is None:
            sigma = estimate_sigma(noisy)
        n = noisy.size
        visu_thr = sigma * np.sqrt(2.0 * np.log(n))
        new_coeffs = [coeffs[0]]
        for details in coeffs[1:]:
            bands = []
            for band in details:
                if method == "visu":
                    thr = visu_thr * thresh_scale
                else:
                    thr = _bayes_threshold(band, sigma) * thresh_scale
                bands.append(pywt.threshold(band, thr, mode=mode))
            new_coeffs.append(tuple(bands))
    out = pywt.waverec2(new_coeffs, wavelet)
    return out[: noisy.shape[0], : noisy.shape[1]]

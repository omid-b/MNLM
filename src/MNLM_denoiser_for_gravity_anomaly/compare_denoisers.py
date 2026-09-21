"""Fair comparison of MNLM, SVD, wavelet, and DCT on the same noisy grid.

The first argument is the assumed noise level as a percent of the
anomaly RMS (the demeaned field). Each method is run on a small
parameter grid and we keep the weakest denoising whose residual RMS
reaches that sigma (Morozov discrepancy).

    python3 compare_denoisers.py 10

Parameter choice uses only the noisy grid and the assumed noise
percent (deployable). The clean field is used only to score and plot
the final maps, not to pick knobs.

Prints the summary table and opens the comparison plot; nothing is
written to disk. Use --quiet to hide per-setting progress, or
--no-show to skip the plot window.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from denoisers import denoise_dct, denoise_mnlm, denoise_svd, denoise_wavelet
from io_xyz import load_xyz_grid

HERE = Path(__file__).resolve().parent


def rmse(a, b):
    """Root-mean-square error between two arrays of the same shape."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    """Mean absolute error between two arrays of the same shape."""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def max_abs(a, b):
    """Largest absolute pointwise difference between two arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def rms(a):
    """Root-mean-square of an array (L2 norm divided by sqrt of length)."""
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def assumed_sigma(noisy, percent, ref="anomaly"):
    """Convert a user-stated noise percent into a sigma in data units.

    Parameters
    ----------
    noisy : array_like
        The noisy gravity grid.
    percent : float
        Assumed noise level, e.g. 10 for 10 percent. Must be positive.
    ref : {'anomaly', 'rms', 'mean_abs'}, optional
        Scale the percent is of. ``anomaly`` (default) is the RMS of the
        demeaned field; ``rms`` is the raw-field RMS; ``mean_abs`` is the
        mean absolute value.

    Returns
    -------
    float
        Assumed noise standard deviation in the same units as ``noisy``.
    """
    noisy = np.asarray(noisy, dtype=float)
    if percent <= 0:
        raise ValueError("noise percent must be positive")
    if ref == "mean_abs":
        scale = float(np.mean(np.abs(noisy)))
    elif ref == "rms":
        scale = rms(noisy)
    else:
        scale = rms(noisy - noisy.mean())
    return (percent / 100.0) * scale


def pick_discrepancy(scored, sigma):
    """Select the weakest denoising whose residual RMS reaches ``sigma``.

    This is the Morozov discrepancy principle: walk the grid from little
    to more denoising and stop at the first residual that is at least the
    assumed noise level. If no setting reaches ``sigma``, the strongest
    (largest residual) setting is returned.

    Parameters
    ----------
    scored : list of dict
        Each item must have a ``resid_rms`` key.
    sigma : float
        Target residual RMS.

    Returns
    -------
    dict
        The chosen entry from ``scored``.
    """
    reached = [s for s in scored if s["resid_rms"] >= sigma]
    if reached:
        return min(reached, key=lambda r: r["resid_rms"])
    return max(scored, key=lambda r: r["resid_rms"])


def gradient_corr(true, est):
    """Mean Pearson correlation of x- and y-gradients (structure score).

    Parameters
    ----------
    true, est : array_like
        Clean and estimated 2D fields.

    Returns
    -------
    float
        Average of the x-gradient and y-gradient correlations. 1 means
        the estimated field keeps the clean field's edges; 0 means none.
    """
    gx_t = np.diff(true, axis=1)
    gx_e = np.diff(est, axis=1)
    gy_t = np.diff(true, axis=0)
    gy_e = np.diff(est, axis=0)

    def _corr(u, v):
        """Pearson correlation of two flattened, demeaned arrays."""
        u = u.ravel() - u.mean()
        v = v.ravel() - v.mean()
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom == 0:
            return 0.0
        return float(np.dot(u, v) / denom)

    return 0.5 * (_corr(gx_t, gx_e) + _corr(gy_t, gy_e))


def metrics(name, params, noisy, true, denoised, elapsed):
    """Score one denoised result against the noisy and clean fields.

    Parameters
    ----------
    name : str
        Method label (e.g. ``'MNLM'``).
    params : dict
        Knob values that produced ``denoised``.
    noisy, true, denoised : array_like
        Input, clean reference, and reconstructed grids.
    elapsed : float
        Runtime in seconds for that single denoiser call.

    Returns
    -------
    dict
        Scalar scores plus ``output``, ``residual`` (``noisy - denoised``)
        and ``error`` (``true - denoised``) grids for plotting.
    """
    residual = noisy - denoised
    error = true - denoised
    return {
        "method": name,
        "params": params,
        "rmse": rmse(denoised, true),
        "mae": mae(denoised, true),
        "max_abs": max_abs(denoised, true),
        "residual_mean": float(residual.mean()),
        "residual_rms": rms(residual),
        "grad_corr": gradient_corr(true, denoised),
        "seconds": elapsed,
        "output": denoised,
        "residual": residual,
        "error": error,
    }


def _param_str(params):
    """Format a parameter dict as a compact ``k=v, k=v`` string."""
    return ", ".join(
        f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.4g}"
        for k, v in params.items()
    )


def run_grid(name, noisy, true, sigma, param_list, denoise_fn, verbose=True):
    """Run one method over its knob grid and stop at the assumed noise.

    Parameters
    ----------
    name : str
        Method label for console output.
    noisy, true : array_like
        Noisy input and clean reference (the latter is used only to score).
    sigma : float
        Assumed noise std; residual RMS is required to reach this value.
    param_list : list of dict
        Keyword arguments passed one-by-one to ``denoise_fn``.
    denoise_fn : callable
        ``denoise_fn(noisy, **params) -> grid``.
    verbose : bool, optional
        If True, print every setting and the chosen stop.

    Returns
    -------
    dict
        ``metrics`` record for the discrepancy-principle pick.
    """
    n = len(param_list)
    print(f"  {name}: searching {n} settings (stop when residual RMS >= {sigma:.5g})")
    scored = []
    for i, params in enumerate(param_list, start=1):
        t0 = time.perf_counter()
        out = denoise_fn(noisy, **params)
        elapsed = time.perf_counter() - t0
        score = {
            "params": params,
            "output": out,
            "resid_rms": rms(noisy - out),
            "seconds": elapsed,
        }
        scored.append(score)
        if verbose:
            print(
                f"    [{i:3d}/{n}] {_param_str(params):<48}  "
                f"residual_rms={score['resid_rms']:.5g}  ({elapsed:.3f}s)"
            )
    assumed = pick_discrepancy(scored, sigma)
    if verbose:
        print(
            f"    -> stop: {_param_str(assumed['params'])}  "
            f"residual_rms={assumed['resid_rms']:.5g} "
            f"(target sigma={sigma:.5g})"
        )
    return metrics(
        name, assumed["params"], noisy, true,
        assumed["output"], assumed["seconds"],
    )


def param_grids(sigma):
    """Build the sigma-scaled search grids for MNLM, SVD, wavelet, and DCT.

    Every knob is a multiple of ``sigma`` so the discrepancy stop can land
    near the assumed noise rather than jumping in coarse rank/block steps.

    Parameters
    ----------
    sigma : float
        Assumed noise standard deviation.

    Returns
    -------
    dict
        ``{name: (list_of_param_dicts, denoise_fn)}`` for each method.
    """
    thresh_scales = (
        0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8,
        1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0,
    )
    return {
        "MNLM": (
            [
                {"Ds": int(Ds), "ds": 5, "h": float(scale * sigma)}
                for Ds in range(2, 20, 2)
                for scale in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
            ],
            denoise_mnlm,
        ),
        "SVD": (
            [{"sigma": sigma, "thresh_scale": s} for s in thresh_scales],
            denoise_svd,
        ),
        "Wavelet": (
            [
                {
                    "level": level,
                    "method": method,
                    "thresh_scale": scale,
                    "sigma": sigma,
                }
                for level in (1, 2, 3, 4)
                for method in ("bayes", "visu")
                for scale in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
            ],
            denoise_wavelet,
        ),
        "DCT": (
            [{"sigma": sigma, "thresh_scale": s} for s in thresh_scales],
            denoise_dct,
        ),
    }


def print_table(rows):
    """Print a fixed-width metrics table for the chosen denoisers."""
    headers = [
        "method", "params", "rmse", "mae", "max_abs",
        "residual_mean", "residual_rms", "grad_corr", "seconds",
    ]
    widths = {h: max(len(h), 12) for h in headers}
    widths["params"] = 56
    widths["method"] = 10
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for row in rows:
        cells = []
        for h in headers:
            val = row[h]
            if h == "params":
                text = _param_str(val)
            elif isinstance(val, float):
                text = f"{val:.6g}"
            else:
                text = str(val)
            cells.append(text[: widths[h]].ljust(widths[h]))
        print("  ".join(cells))


def plot_regime(noisy, true, xx, yy, rows, sigma):
    """Draw the side-by-side comparison figure (not written to disk).

    Rows are noisy / residual / error-vs-clean. Columns are the input
    (or true noise / clean field) plus one column per method. Color
    scales are shared across each row.

    Parameters
    ----------
    noisy, true : array_like
        Noisy and clean 2D grids.
    xx, yy : array_like
        Coordinate grids matching ``noisy``.
    rows : list of dict
        ``metrics`` records, one per method.
    sigma : float
        Assumed noise std, shown in the figure title.
    """
    n = len(rows)
    fig, axes = plt.subplots(3, n + 1, figsize=(3.2 * (n + 1), 8.5), facecolor="white")

    den_lim = np.percentile(true, [1, 99])
    res_lim = max(abs(r["residual"]).max() for r in rows)
    err_lim = max(abs(r["error"]).max() for r in rows)

    def _map(ax, z, title, vmin, vmax, cmap="jet"):
        """Filled-contour one panel onto ``ax`` with a shared color scale."""
        if vmax <= vmin:
            vmax = vmin + 1e-12
        levels = np.linspace(vmin, vmax, 21)
        cs = ax.contourf(xx, yy, z, levels=levels, cmap=cmap, extend="both")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)

    _map(axes[0, 0], noisy, "noisy input", den_lim[0], den_lim[1])
    _map(axes[1, 0], noisy - true, "true noise", -res_lim, res_lim)
    _map(axes[2, 0], true, "clean field", den_lim[0], den_lim[1])

    for j, row in enumerate(rows, start=1):
        _map(axes[0, j], row["output"], f"{row['method']} denoised", den_lim[0], den_lim[1])
        _map(axes[1, j], row["residual"], f"{row['method']} residual", -res_lim, res_lim)
        _map(axes[2, j], row["error"], f"{row['method']} error vs clean", -err_lim, err_lim)

    fig.suptitle(
        f"Assumed-noise comparison (shared color scale per row)  (target σ={sigma:.4g})",
        fontsize=12,
    )
    fig.tight_layout()


def main(argv=None):
    """Load the synthetic pair, pick knobs from the assumed noise, and plot.

    Parameters
    ----------
    argv : list of str, optional
        Command-line tokens; defaults to ``sys.argv[1:]``.

    Returns
    -------
    list of dict
        One ``metrics`` record per denoiser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "noise_percent",
        type=float,
        help="Assumed noise as a percent of the field (e.g. 10 for 10%%)",
    )
    parser.add_argument(
        "--noise-ref",
        choices=("anomaly", "rms", "mean_abs"),
        default="anomaly",
        help="What the percent is of (default: RMS of the demeaned anomaly)",
    )
    parser.add_argument(
        "--clean",
        type=Path,
        default=HERE / "synthetic.xyz",
        help="Noise-free XYZ grid",
    )
    parser.add_argument(
        "--noisy",
        type=Path,
        default=HERE / "10precent.xyz",
        help="Noisy XYZ grid",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the plot window (print the table only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the summary table (no per-setting progress)",
    )
    args = parser.parse_args(argv)
    verbose = not args.quiet

    if verbose:
        print(f"Loading clean grid: {args.clean}")
        print(f"Loading noisy grid: {args.noisy}")
    true, xx, yy = load_xyz_grid(args.clean)
    noisy, _, _ = load_xyz_grid(args.noisy)
    if true.shape != noisy.shape:
        raise ValueError(f"grid shape mismatch: clean {true.shape} vs noisy {noisy.shape}")
    if verbose:
        print(f"Grid shape: {noisy.shape[0]} x {noisy.shape[1]}")

    sigma = assumed_sigma(noisy, args.noise_percent, args.noise_ref)
    print(
        f"Assumed noise: {args.noise_percent:g}% of noisy {args.noise_ref} "
        f"-> sigma={sigma:.6g}"
    )
    print(f"Reference RMSE of the untreated noisy grid: {rmse(noisy, true):.6g}")

    rows = []
    grids = param_grids(sigma)
    for name, (params, fn) in grids.items():
        row = run_grid(name, noisy, true, sigma, params, fn, verbose=verbose)
        rows.append(row)
        if not verbose:
            print(
                f"    {name}: {_param_str(row['params'])}  "
                f"residual_rms={row['residual_rms']:.5g}"
            )

    print()
    print_table(rows)

    if args.no_show:
        return rows
    if verbose:
        print("Building comparison figure...")
    plot_regime(noisy, true, xx, yy, rows, sigma=sigma)
    print("Opening the comparison plot (close the window to finish)...")
    plt.show()
    return rows


if __name__ == "__main__":
    main()

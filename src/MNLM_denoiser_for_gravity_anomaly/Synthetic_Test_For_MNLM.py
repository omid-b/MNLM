"""Python equivalent of Synthetic_Test_For_MNLM.m

Author of the original MATLAB code: Hanbing Ai
E-mail: AHB_ECUT@163.com / 1724178612@qq.com
Date: 2022.11.13.

This file reproduces the original MATLAB experiment, including its
unfair baselines (SVD / wavelet / DCT applied to the clean field).
For a fair comparison on the noisy grid, run compare_denoisers.py.

Run:  python3 Synthetic_Test_For_MNLM.py
Outputs default to results/matlab_repro/ so committed MATLAB text
files in this folder are not overwritten.
"""

import argparse
from pathlib import Path

import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

from NLM_II import NLM_II
from transform_in import transform_in
from transform_out import transform_out

HERE = Path(__file__).resolve().parent


def main(argv=None):
    """Reproduce the original MATLAB synthetic experiment.

    Loads ``synthetic.xyz`` and ``10precent.xyz``, oracle-tunes MNLM
    against the clean field, runs the MATLAB-faithful SVD / wavelet /
    DCT baselines on the *clean* field, writes twelve ASCII result
    files, and optionally shows the four contour figures.

    Parameters
    ----------
    argv : list of str, optional
        Command-line tokens; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "matlab_repro",
        help="Where to write the twelve ASCII result files",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # %% Read the raw data
    DATA_1 = np.loadtxt(HERE / "synthetic.xyz")
    DATA_2 = np.loadtxt(HERE / "10precent.xyz")
    #
    D_1, X, Y = transform_in(DATA_1, 121, 121)
    D_2, XX, YY = transform_in(DATA_2, 121, 121)
    # %%
    INPUT = D_2  # Noise-corrupted data
    TRUE = D_1   # Noise-free data

    # %% Main program for MNLM to denoise
    ds = 5  # block size for calculating the weight
    Ds = np.arange(2, 20, 2)  # search block: 2,4,...,18
    sigma = 0.0001 * INPUT.mean()
    h = np.arange(100, 1100, 100) * sigma  # 100*sigma ... 1000*sigma
    #
    N_processed = np.zeros((len(Ds), len(h)))
    for i in range(len(Ds)):
        for j in range(len(h)):
            N_processed[i, j] = np.sqrt(np.mean((D_1 - NLM_II(INPUT, Ds[i], ds, h[j])) ** 2))
    #
    # MATLAB's find() scans column-major and the script keeps R(1),C(1);
    # flatten in Fortran (column-major) order to break ties identically:
    R, C = np.unravel_index(np.argmin(N_processed.flatten(order="F")),
                            N_processed.shape, order="F")
    #
    OUTPUT_MNLM = NLM_II(INPUT, Ds[R], ds, h[C])
    Noise_component_MNLM = INPUT - OUTPUT_MNLM
    Difference_MNLM = D_1 - OUTPUT_MNLM
    #
    RMS_1 = N_processed.min()
    print(f"Best MNLM parameters: Ds={Ds[R]}, h={h[C]:.6g}, RMSE={RMS_1:.6g}")
    #
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="white")
    cs = axes[0, 0].contourf(X, Y, Noise_component_MNLM, cmap="jet")
    fig.colorbar(cs, ax=axes[0, 0])
    axes[0, 0].set_title("Noise component", fontsize=12)
    cs = axes[0, 1].contourf(X, Y, OUTPUT_MNLM, cmap="jet")
    fig.colorbar(cs, ax=axes[0, 1])
    axes[0, 1].set_title("Denoised result", fontsize=12)
    cs = axes[1, 0].contourf(X, Y, Difference_MNLM, cmap="jet")
    fig.colorbar(cs, ax=axes[1, 0])
    axes[1, 0].set_title("Difference between denoised result and noise-free data", fontsize=10)
    cs = axes[1, 1].contourf(X, Y, INPUT, cmap="jet")
    fig.colorbar(cs, ax=axes[1, 1])
    axes[1, 1].set_title("Noise corrupted result", fontsize=12)
    fig.suptitle("MNLM")

    # %% SVD
    u, s, vt = np.linalg.svd(D_1)
    #
    s0 = s.copy()
    s0[np.abs(s) < 580] = 0.0
    #
    OUTPUT_SVD = (u * s0) @ vt
    Noise_component_SVD = INPUT - OUTPUT_SVD
    Difference_SVD = D_1 - OUTPUT_SVD
    #
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="white")
    cs = axes[0, 0].contourf(X, Y, Noise_component_SVD, cmap="jet")
    fig.colorbar(cs, ax=axes[0, 0])
    axes[0, 0].set_title("Noise component", fontsize=12)
    cs = axes[0, 1].contourf(X, Y, OUTPUT_SVD, cmap="jet")
    fig.colorbar(cs, ax=axes[0, 1])
    axes[0, 1].set_title("Denoised result", fontsize=12)
    cs = axes[1, 0].contourf(X, Y, Difference_SVD, cmap="jet")
    fig.colorbar(cs, ax=axes[1, 0])
    axes[1, 0].set_title("Difference between denoised result and noise-free data", fontsize=10)
    axes[1, 1].axis("off")
    fig.suptitle("SVD")

    # %% Wavelet
    # MATLAB: [c,s] = wavedec2(D_1,3,'sym4'); OUTPUT_wavelet = wrcoef2('a',c,s,'sym4');
    # i.e. keep only the level-3 approximation and reconstruct.
    coeffs = pywt.wavedec2(D_1, "sym4", level=3)
    coeffs_approx_only = [coeffs[0]] + [
        tuple(np.zeros_like(d) for d in details) for details in coeffs[1:]
    ]
    OUTPUT_wavelet = pywt.waverec2(coeffs_approx_only, "sym4")
    OUTPUT_wavelet = OUTPUT_wavelet[: D_1.shape[0], : D_1.shape[1]]
    Noise_component_wavelet = INPUT - OUTPUT_wavelet
    Difference_wavelet = D_1 - OUTPUT_wavelet
    #
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="white")
    cs = axes[0, 0].contourf(X, Y, Noise_component_wavelet, cmap="jet")
    fig.colorbar(cs, ax=axes[0, 0])
    axes[0, 0].set_title("Noise component", fontsize=12)
    cs = axes[0, 1].contourf(X, Y, OUTPUT_wavelet, cmap="jet")
    fig.colorbar(cs, ax=axes[0, 1])
    axes[0, 1].set_title("Denoised result", fontsize=12)
    cs = axes[1, 0].contourf(X, Y, Difference_wavelet, cmap="jet")
    fig.colorbar(cs, ax=axes[1, 0])
    axes[1, 0].set_title("Difference between denoised result and noise-free data", fontsize=10)
    axes[1, 1].axis("off")
    fig.suptitle("Wavelet")

    # %% DCT
    GRAVITY = dctn(D_1, type=2, norm="ortho")
    m, n = GRAVITY.shape
    #
    QC1 = np.zeros((m, n))
    QC1[:10, :10] = GRAVITY[:10, :10]
    #
    OUTPUT_DCT = idctn(QC1, type=2, norm="ortho")
    Noise_component_DCT = INPUT - OUTPUT_DCT
    Difference_DCT = D_1 - OUTPUT_DCT
    #
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="white")
    cs = axes[0, 0].contourf(X, Y, Noise_component_DCT, cmap="jet")
    fig.colorbar(cs, ax=axes[0, 0])
    axes[0, 0].set_title("Noise component", fontsize=12)
    cs = axes[0, 1].contourf(X, Y, OUTPUT_DCT, cmap="jet")
    fig.colorbar(cs, ax=axes[0, 1])
    axes[0, 1].set_title("Denoised result", fontsize=12)
    cs = axes[1, 0].contourf(X, Y, Difference_DCT, cmap="jet")
    fig.colorbar(cs, ax=axes[1, 0])
    axes[1, 0].set_title("Difference between denoised result and noise-free data", fontsize=10)
    axes[1, 1].axis("off")
    fig.suptitle("DCT")

    # %% Convert grids back to XYZ lists (with the same column swap as the MATLAB script)
    def _to_xyz(grid):
        """Flatten a grid to XYZ and swap X/Y to match the MATLAB script."""
        cc = transform_out(grid, Y[0, :], X[:, 0])
        # MATLAB swaps columns 1 and 2 after transform_out:
        return np.column_stack((cc[:, 1], cc[:, 0], cc[:, 2]))

    DT_MNLM = _to_xyz(OUTPUT_MNLM)
    DT_SVD = _to_xyz(OUTPUT_SVD)
    DT_wavelet = _to_xyz(OUTPUT_wavelet)
    DT_DCT = _to_xyz(OUTPUT_DCT)
    # %%
    DT_Noise_component_MNLM = _to_xyz(Noise_component_MNLM)
    DT_Noise_component_SVD = _to_xyz(Noise_component_SVD)
    DT_Noise_component_wavelet = _to_xyz(Noise_component_wavelet)
    DT_Noise_component_DCT = _to_xyz(Noise_component_DCT)
    # %%
    DT_Difference_MNLM = _to_xyz(Difference_MNLM)
    DT_Difference_SVD = _to_xyz(Difference_SVD)
    DT_Difference_wavelet = _to_xyz(Difference_wavelet)
    DT_Difference_DCT = _to_xyz(Difference_DCT)

    # %% Save results (equivalent of MATLAB's `save -ascii`)
    _FMT = "%16.7e"
    np.savetxt(out_dir / "denoised_result_SyntheticCase_MNLM.txt", DT_MNLM, fmt=_FMT)
    np.savetxt(out_dir / "denoised_result_SyntheticCase_SVD.txt", DT_SVD, fmt=_FMT)
    np.savetxt(out_dir / "denoised_result_SyntheticCase_wavelet.txt", DT_wavelet, fmt=_FMT)
    np.savetxt(out_dir / "denoised_result_SyntheticCase_DCT.txt", DT_DCT, fmt=_FMT)
    #
    np.savetxt(out_dir / "Noise_component_SyntheticCase_MNLM.txt", DT_Noise_component_MNLM, fmt=_FMT)
    np.savetxt(out_dir / "Noise_component_SyntheticCase_SVD.txt", DT_Noise_component_SVD, fmt=_FMT)
    np.savetxt(out_dir / "Noise_component_SyntheticCase_wavelet.txt", DT_Noise_component_wavelet, fmt=_FMT)
    np.savetxt(out_dir / "Noise_component_SyntheticCase_DCT.txt", DT_Noise_component_DCT, fmt=_FMT)
    #
    np.savetxt(out_dir / "Difference_SyntheticCase_MNLM.txt", DT_Difference_MNLM, fmt=_FMT)
    np.savetxt(out_dir / "Difference_SyntheticCase_SVD.txt", DT_Difference_SVD, fmt=_FMT)
    np.savetxt(out_dir / "Difference_SyntheticCase_wavelet.txt", DT_Difference_wavelet, fmt=_FMT)
    np.savetxt(out_dir / "Difference_SyntheticCase_DCT.txt", DT_Difference_DCT, fmt=_FMT)

    print(f"Wrote MATLAB-repro outputs to {out_dir}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

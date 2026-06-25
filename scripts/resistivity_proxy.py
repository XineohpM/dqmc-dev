#!/usr/bin/env python3
""" 
resistivity_proxy.py
phoenixm@stanford.edu

Compute two proxy estimates of the dc resistivity from the imaginary-time
current-current correlator (q=0, xx component), extracted as JNJN_xx_perbin.npy.

Proxy 1:
    rho1(T) = pi * T^2 / Lambda(beta/2)
Proxy 2:
    rho2(T) = Lambda''(beta/2) / (2 * pi * Lambda(beta/2)^2)

Reference:
    10.1126/science.aau7063

USAGE EXAMPLES:
python3 /home/users/phoenixm/scripts/resistivity_proxy.py \
        --path /scratch/users/phoenixm/dqmc_runs/6x6_tp0_nflux0/n0.6_resistivity \
        --sym \
        --items \
            "T0.05_beta20_U-6/mu-0.462713,,0.05" \
            "T0.1_beta10_U-6/mu-0.463656,,0.1" \
            "T0.125_beta8_U-6/mu-0.464428,,0.125" \
            "T0.166667_beta6_U-6/mu-0.466893,,0.166667" \
            "T0.2_beta5_U-6/mu-0.470034,,0.2" \
            "T0.222222_beta4.5_U-6/mu-0.472615,,0.222222" \
            "T0.25_beta4_U-6/mu-0.47691,,0.25" \
            "T0.285714_beta3.5_U-6/mu-0.484274,,0.285714" \
            "T0.333333_beta3_U-6/mu-0.496175,,0.333333" \
            "T0.4_beta2.5_U-6/mu-0.515708,,0.4" \
            "T0.5_beta2_U-6/mu-0.5489,,0.5" \
            "T0.666667_beta1.5_U-6/mu-0.610407,,0.666667" \
            "T1_beta1_U-6/mu-0.755037,,1.0" \
            "T2_beta0.5_U-6/mu-1.3016,,2.0" \
            "T4_beta0.25_U-6/mu-2.68617,,4.0" \
            "T8_beta0.125_U-6/mu-5.85011,,8.0" \
        --output_path /scratch/users/phoenixm/dqmc_runs/6x6_tp0_nflux0/n0.6_resistivity/proxies_03062026_sym

python3 /home/users/phoenixm/scripts/resistivity_proxy.py \
        --path /scratch/users/phoenixm/dqmc_runs/6x6_tp0_nflux0/n0.8_resistivity \
        --sym \
        --items \
            "T0.05_beta20_U-6/mu-0.216806,,0.05" \
            "T0.1_beta10_U-6/mu-0.218041,,0.1" \
            "T0.125_beta8_U-6/mu-0.219898,,0.125" \
            "T0.166667_beta6_U-6/mu-0.223798,,0.166667" \
            "T0.2_beta5_U-6/mu-0.226645,,0.2" \
            "T0.222222_beta4.5_U-6/mu-0.228323,,0.222222" \
            "T0.25_beta4_U-6/mu-0.230101,,0.25" \
            "T0.285714_beta3.5_U-6/mu-0.232509,,0.285714" \
            "T0.333333_beta3_U-6/mu-0.237124,,0.333333" \
            "T0.4_beta2.5_U-6/mu-0.246334,,0.4" \
            "T0.5_beta2_U-6/mu-0.263233,,0.5" \
            "T0.666667_beta1.5_U-6/mu-0.294278,,0.666667" \
            "T1_beta1_U-6/mu-0.364676,,1.0" \
            "T2_beta0.5_U-6/mu-0.624785,,2.0" \
            "T4_beta0.25_U-6/mu-1.28167,,4.0" \
            "T8_beta0.125_U-6/mu-2.78836,,8.0" \
        --output_path /scratch/users/phoenixm/dqmc_runs/6x6_tp0_nflux0/n0.8_resistivity/proxies_03062026_sym

This produces:
    proxies_03062026_sym_rho1.npz, proxies_03062026_sym_rho1.csv
    proxies_03062026_sym_rho2.npz, proxies_03062026_sym_rho2.csv
"""

from __future__ import annotations

import os, argparse
import numpy as np


NORMALIZATION_NOTE = (
    "Absolute proxy normalization assumes JNJN_xx_perbin.npy is the per-site q=0 "
    "correlator and has passed the sum-rule check in scripts/check_sum_rule.py."
)


def _as_checked_real(arr: np.ndarray, name: str, imag_tol: float) -> tuple[np.ndarray, float]:
    """Return a real array after checking that any imaginary part is negligible."""
    arr = np.asarray(arr)
    imag_max = 0.0
    if np.iscomplexobj(arr):
        imag_max = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
        real_scale = float(np.max(np.abs(arr.real))) if arr.size else 0.0
        tol = max(float(imag_tol), float(imag_tol) * real_scale)
        if imag_max > tol:
            raise ValueError(
                f"{name} has non-negligible imaginary part: max |imag|={imag_max:g}, "
                f"tolerance={tol:g}"
            )
        arr = arr.real
    return np.asarray(arr, dtype=float), imag_max


def _load_tau(subpath: str):
    """
    Docstring for _load_tau
    
    Args:
    subpath (str): Pathname that contains tau.npy for a certain temperature.

    Returns:
    tau (L,): Imaginary time array.
    dt (float): Imaginary time stepsize.
    L (int): Imaginary time slices.
    beta (float): Inverse temperature.

    """
    taufile = os.path.join(subpath, "tau.npy")
    if not os.path.exists(taufile): 
        raise FileNotFoundError(f"missing {taufile}")
    tau_raw = np.load(taufile)
    tau, imag_max = _as_checked_real(tau_raw, taufile, imag_tol=0.0)
    if imag_max != 0.0:
        raise ValueError(f"{taufile} should be real, got max |imag|={imag_max:g}")
    if tau.ndim != 1 or tau.size < 3:
        raise ValueError(f"bad tau shape: {tau.shape}")
    if not np.all(np.isfinite(tau)):
        raise ValueError(f"{taufile} contains non-finite values")
    L = tau.size
    dt = float(tau[1] - tau[0])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"bad tau step in {taufile}: dt={dt}")
    if not np.allclose(np.diff(tau), dt, rtol=1e-8, atol=1e-12):
        raise ValueError(f"{taufile} is not uniformly spaced")
    # Convention: tau = 0, dt, ..., (L-1)dt  (no endpoint); beta = L*dt
    beta = float(dt * L)
    return tau, dt, beta


def _load_corr(subpath: str, prefix: str, imag_tol: float):
    """
    Docstring for _load_corr
    
    Args:
    subpath (str): Pathname that contains tau.npy for a certain temperature.

    Returns:

    """
    jjfile_name = prefix + "JNJN_xx_perbin.npy"
    jjfile_perbin = os.path.join(subpath, jjfile_name)
    if not os.path.exists(jjfile_perbin):
        raise FileNotFoundError(
            f"Missing {jjfile_perbin}. This script computes mean from per-bin data and does not read JNJN_xx_mean.npy."
        )

    corr_perbin_raw = np.load(jjfile_perbin)
    corr_perbin, imag_max = _as_checked_real(corr_perbin_raw, jjfile_perbin, imag_tol)
    if corr_perbin.ndim != 2:
        raise ValueError(f"bad perbin shape in {jjfile_perbin}: {corr_perbin.shape}")
    if corr_perbin.shape[0] < 1:
        raise ValueError(f"{jjfile_perbin} has no bins")
    if not np.all(np.isfinite(corr_perbin)):
        raise ValueError(f"{jjfile_perbin} contains non-finite values")

    corr_mean = corr_perbin.mean(axis=0)
    return corr_mean, corr_perbin, imag_max


def _validate_tau_matches_corr(tau: np.ndarray, corr_len: int, dpath: str) -> None:
    if tau.size != corr_len:
        raise ValueError(
            f"tau/correlator length mismatch in {dpath}: tau has L={tau.size}, "
            f"correlator has L={corr_len}. tau.npy must use the no-endpoint grid."
        )
    if not np.isclose(tau[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"tau grid in {dpath} should start at 0, got tau[0]={tau[0]:g}")


def _symmetrize_about_beta_over_2(corr: np.ndarray) -> np.ndarray:
    """Symmetrize corr(τ) about β/2: corr_sym(τ)=0.5*(corr(τ)+corr(β-τ)).

    Assumes corr is sampled on τ = 0, dt, ..., (L-1)dt.
    """
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 1:
        raise ValueError(f"corr must be 1D for symmetrization, got shape {corr.shape}")
    # For the no-endpoint grid (tau=i*dt, i=0..L-1, beta=L*dt), the point corresponding
    # to beta - tau_i maps to index (-i) mod L.
    idx_partner = (-np.arange(corr.size)) % corr.size
    return 0.5 * (corr + corr[idx_partner])

def Lambda_xx_beta_over_2(corr: np.ndarray):
    """
    Docstring for Lambda_xx_beta_over_2
    
    Args:
    
    Returns:

    """
    L = corr.size
    mid = L // 2

    if not np.all(np.isfinite(corr)):
        raise ValueError("corr contains non-finite values")


    if L % 2 == 0:
        return float(corr[mid]), mid
    return float(0.5 * (corr[mid] + corr[mid + 1])), mid

def Lambda_2nd_deriv(corr: np.ndarray, dt: float, window: int = 3, return_fit: bool = False):
    """
    Docstring for Lambda_2nd_deriv
    
    Args:

    Returns:

    """
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 1:
        raise ValueError(f"corr must be 1D, got shape {corr.shape}")
    if not np.all(np.isfinite(corr)):
        raise ValueError("corr contains non-finite values")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"bad dt={dt}")

    L = corr.size
    if L < 5:
        raise ValueError(f"L should be at least 5 for quadratic fit, got L={L}")

    # Center at tau = beta/2.
    beta = dt * L
    tau0 = 0.5 * beta

    # Use a symmetric window of points around tau0.
    if int(window) < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    w = int(window)
    i_center = tau0 / dt
    i0 = int(np.floor(i_center)) - w
    i1 = int(np.ceil(i_center)) + w
    i0 = max(0, i0)
    i1 = min(L - 1, i1)

    idx = np.arange(i0, i1 + 1, dtype=int)
    if idx.size < 3:
        raise ValueError("Not enough points for quadratic fit")

    tau = idx * dt
    x = tau - tau0
    y = corr[idx]

    # Fit y(x) = a x^2 + b x + c, then Lambda''(tau0) = 2a.
    a, b, c = np.polyfit(x, y, deg=2)
    d2 = float(2.0 * a)
    if not return_fit:
        return d2
    yfit = a * x * x + b * x + c
    rms = float(np.sqrt(np.mean((y - yfit) ** 2)))
    return d2, int(idx.size), rms


def _bootstrap_proxies(
    T: float,
    corr_perbin: np.ndarray,
    dt: float,
    nboot: int,
    seed: int,
    deriv_window: int,
) -> tuple[float, float, float, float, float, float, int]:
    if nboot <= 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    nbin = corr_perbin.shape[0]
    rng = np.random.default_rng(seed)
    rho1_vals = np.empty(nboot, dtype=float)
    rho2_vals = np.empty(nboot, dtype=float)
    ngood = 0

    for _ in range(nboot):
        sample_idx = rng.integers(0, nbin, size=nbin)
        corr_b = corr_perbin[sample_idx].mean(axis=0)
        try:
            lam_mid_b, _ = Lambda_xx_beta_over_2(corr_b)
            lam2_mid_b = Lambda_2nd_deriv(corr_b, dt, window=deriv_window)
            if not (
                np.isfinite(lam_mid_b)
                and np.isfinite(lam2_mid_b)
                and abs(lam_mid_b) > 1e-14
            ):
                continue
            rho1_vals[ngood] = rho1(T, lam_mid_b)
            rho2_vals[ngood] = rho2(lam_mid_b, lam2_mid_b)
            ngood += 1
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue

    if ngood == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    rho1_vals = rho1_vals[:ngood]
    rho2_vals = rho2_vals[:ngood]
    rho1_p16, rho1_p84 = np.percentile(rho1_vals, [16, 84])
    rho2_p16, rho2_p84 = np.percentile(rho2_vals, [16, 84])
    if ngood >= 2:
        rho1_stderr = float(np.std(rho1_vals, ddof=1))
        rho2_stderr = float(np.std(rho2_vals, ddof=1))
    else:
        rho1_stderr = np.nan
        rho2_stderr = np.nan
    return (
        float(rho1_p16),
        float(rho1_p84),
        float(rho2_p16),
        float(rho2_p84),
        rho1_stderr,
        rho2_stderr,
        int(ngood),
    )
    
def rho1(T: float, corr: float):
    """
    Proxy 1, rho1(T) = pi * T^2 / Lambda(beta/2)
    
    Args:
    T (float): Temperature
    corr (float): Current-current correlation at beta/2

    Returns:
    
    """
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"bad T={T}")
    if not np.isfinite(corr):
        raise ValueError("Lambda(beta/2) is non-finite")
    if corr == 0.0:
        raise ZeroDivisionError("Lambda(beta/2) is 0")

    return np.pi * T * T / corr

def rho2(corr: float, d2corr: float):
    """
    Proxy 2, rho2(T) = Lambda''(beta/2) / (2 * pi * Lambda(beta/2)^2)
    
    Args:
    corr (float): Current-current correlation at beta/2
    d2corr (float): Second derivation of current-current correlation at beta/2

    Returns:

    """
    if not np.isfinite(corr):
        raise ValueError("Lambda(beta/2) is non-finite")
    if corr == 0.0:
        raise ZeroDivisionError("Lambda(beta/2) is 0")
    if not np.isfinite(d2corr):
        raise ValueError(f"Lambda''(beta/2) is non-finite")

    return d2corr/ 2 / np.pi / corr / corr
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        required=True,
        help="Base directory containing T_* or beta_* subdirectories.",
    )
    p.add_argument("--items", nargs="+", required=True,
                    help=("List of items: each is 'relpath,prefix,T'. Example: "
                          "'T_0.2/maxent_out,U-6_T0.2_jjxx_,0.2'"))
    # kept for forward compatibility (not used in proxy-only workflow)
    p.add_argument(
        "--output_path",
        required=True,
        help="Output file path (.npz) or output directory. Will be created if needed.",
    )
    p.add_argument("--sym", action="store_true", help="If set, the imaginary-time" \
                   " correlation function would be symmetrized about β/2.")
    p.add_argument("--nboot", type=int, default=1000,
                   help="Number of bootstrap resamples for proxy uncertainties. Use 0 to disable.")
    p.add_argument("--seed", type=int, default=12345,
                   help="Random seed for bootstrap resampling.")
    p.add_argument("--deriv_window", type=int, default=3,
                   help="Half-window in tau steps for quadratic Lambda'' fit around beta/2.")
    p.add_argument("--imag_tol", type=float, default=1e-10,
                   help="Allowed absolute/relative imaginary-part tolerance for input arrays.")

    args = p.parse_args()

    base = os.path.expanduser(args.path)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Base path not found or not a directory: {base}")

    # Decide output file location.
    # --output_path can be either:
    #   (i) a directory, OR
    #   (ii) a file prefix (no extension), OR
    #   (iii) a full .npz path (treated as prefix by stripping .npz).
    out = os.path.expanduser(args.output_path)
    if out.endswith(".npz"):
        out_prefix = os.path.splitext(out)[0]
        out_dir = os.path.dirname(out_prefix) or "."
    else:
        # if it's an existing directory OR ends with a path separator, treat as directory
        if os.path.isdir(out) or out.endswith(os.sep):
            out_dir = out
            out_prefix = os.path.join(out_dir, "proxies")
        else:
            # treat as a prefix
            out_prefix = out
            out_dir = os.path.dirname(out_prefix) or "."

    os.makedirs(out_dir, exist_ok=True)

    rows = []
    warnings = []

    for item in args.items:
        parts = [s.strip() for s in item.split(",", 2)]
        if len(parts) != 3:
            raise ValueError(f"Bad --items entry {item!r}; expected 'relpath,prefix,T'")
        rel, pfx, Tstr = parts
        T = float(Tstr)
        dpath = os.path.join(base, rel)

        corr_mean, corr_perbin, imag_max = _load_corr(dpath, pfx, args.imag_tol)
        if imag_max > 0:
            warnings.append(
                f"[WARN] {dpath}: input correlator had small imaginary part "
                f"max |imag|={imag_max:g}; using real part."
            )

        tau, dt, beta = _load_tau(dpath)
        _validate_tau_matches_corr(tau, corr_mean.size, dpath)

        if args.sym:
            corr_mean = _symmetrize_about_beta_over_2(corr_mean)
            if corr_perbin is not None:
                L = corr_perbin.shape[1]
                idx_partner = (-np.arange(L)) % L
                corr_perbin = 0.5 * (corr_perbin + corr_perbin[:, idx_partner])

        if beta > 0:
            T_from_beta = 1.0 / beta
            if abs(T - T_from_beta) / max(T, T_from_beta, 1e-12) > 0.02:
                warnings.append(
                    f"[WARN] {dpath}: T(from name)={T:g} differs from 1/beta={T_from_beta:g} (beta={beta:g})."
                )

        lam_mid, mid = Lambda_xx_beta_over_2(corr_mean)
        lam2_mid, fit_npts, fit_rms = Lambda_2nd_deriv(
            corr_mean, dt, window=args.deriv_window, return_fit=True
        )

        # Mean-curve proxies (kept as auxiliary outputs)
        rho1_mean = float(rho1(T, lam_mid))
        rho2_mean = float(rho2(lam_mid, lam2_mid))

        rho1_p16 = rho1_p84 = rho2_p16 = rho2_p84 = np.nan
        rho1_stderr = rho2_stderr = np.nan
        ngood_boot = 0
        nbin = 0

        if corr_perbin is not None:
            nbin, L = corr_perbin.shape
            if L != corr_mean.size:
                raise ValueError(
                    f"perbin L mismatch: perbin {L} vs mean {corr_mean.size} in {dpath}"
                )

            if args.nboot > 0:
                boot_seed = int(args.seed) + len(rows)
                (
                    rho1_p16,
                    rho1_p84,
                    rho2_p16,
                    rho2_p84,
                    rho1_stderr,
                    rho2_stderr,
                    ngood_boot,
                ) = _bootstrap_proxies(
                    T,
                    corr_perbin,
                    dt,
                    args.nboot,
                    boot_seed,
                    args.deriv_window,
                )
                if ngood_boot < args.nboot:
                    warnings.append(
                        f"[WARN] {dpath}: kept {ngood_boot} out of {args.nboot} "
                        "bootstrap samples for uncertainty estimates"
                    )
            else:
                warnings.append(f"[WARN] {dpath}: bootstrap disabled; uncertainty estimates are NaN")

        rows.append(
            (
                float(T),
                float(beta),
                float(dt),
                float(lam_mid),
                float(lam2_mid),
                float(rho1_mean),
                float(rho2_mean),
                float(rho1_p16),
                float(rho1_p84),
                float(rho2_p16),
                float(rho2_p84),
                float(rho1_stderr),
                float(rho2_stderr),
                int(nbin),
                int(ngood_boot),
                int(fit_npts),
                float(fit_rms),
                dpath,
            )
        )

    if not rows:
        msg = "No valid folders processed.\n"
        if warnings:
            msg += "\n".join(warnings[:20])
        raise RuntimeError(msg)

    # Sort by temperature
    rows = sorted(rows, key=lambda r: r[0])

    # Convert to arrays
    T_arr = np.array([r[0] for r in rows], dtype=float)
    beta_arr = np.array([r[1] for r in rows], dtype=float)
    dt_arr = np.array([r[2] for r in rows], dtype=float)
    lam_mid_arr = np.array([r[3] for r in rows], dtype=float)
    lam2_mid_arr = np.array([r[4] for r in rows], dtype=float)
    rho1_mean_arr = np.array([r[5] for r in rows], dtype=float)
    rho2_mean_arr = np.array([r[6] for r in rows], dtype=float)
    rho1_p16_arr = np.array([r[7] for r in rows], dtype=float)
    rho1_p84_arr = np.array([r[8] for r in rows], dtype=float)
    rho2_p16_arr = np.array([r[9] for r in rows], dtype=float)
    rho2_p84_arr = np.array([r[10] for r in rows], dtype=float)
    rho1_stderr_arr = np.array([r[11] for r in rows], dtype=float)
    rho2_stderr_arr = np.array([r[12] for r in rows], dtype=float)
    nbin_arr = np.array([r[13] for r in rows], dtype=int)
    ngood_boot_arr = np.array([r[14] for r in rows], dtype=int)
    fit_npts_arr = np.array([r[15] for r in rows], dtype=int)
    fit_rms_arr = np.array([r[16] for r in rows], dtype=float)
    folder_arr = np.array([r[17] for r in rows], dtype=object)

    # Write proxy-1 outputs
    out_npz_rho1 = out_prefix + "_rho1.npz"
    np.savez(
        out_npz_rho1,
        T=T_arr,
        beta=beta_arr,
        dt=dt_arr,
        Lambda_mid=lam_mid_arr,
        rho_mean=rho1_mean_arr,
        rho_p16=rho1_p16_arr,
        rho_p84=rho1_p84_arr,
        rho_stderr=rho1_stderr_arr,
        nbin=nbin_arr,
        nboot=np.full_like(nbin_arr, args.nboot),
        ngood_boot=ngood_boot_arr,
        Lambda2_fit_npts=fit_npts_arr,
        Lambda2_fit_rms=fit_rms_arr,
        normalization_note=np.array(NORMALIZATION_NOTE),
        folder=folder_arr,
    )

    out_csv_rho1 = os.path.splitext(out_npz_rho1)[0] + ".csv"
    with open(out_csv_rho1, "w") as f:
        f.write(
            "T,beta,dt,Lambda_mid,rho_mean,rho_p16,rho_p84,rho_stderr,nbin,nboot,ngood_boot,Lambda2_fit_npts,Lambda2_fit_rms,folder\n"
        )
        for i in range(T_arr.size):
            f.write(
                f"{T_arr[i]},{beta_arr[i]},{dt_arr[i]},{lam_mid_arr[i]},{rho1_mean_arr[i]},{rho1_p16_arr[i]},{rho1_p84_arr[i]},{rho1_stderr_arr[i]},{nbin_arr[i]},{args.nboot},{ngood_boot_arr[i]},{fit_npts_arr[i]},{fit_rms_arr[i]},{repr(folder_arr[i])}\n"
            )

    # Write proxy-2 outputs
    out_npz_rho2 = out_prefix + "_rho2.npz"
    np.savez(
        out_npz_rho2,
        T=T_arr,
        beta=beta_arr,
        dt=dt_arr,
        Lambda_mid=lam_mid_arr,
        Lambda2_mid=lam2_mid_arr,
        rho_mean=rho2_mean_arr,
        rho_p16=rho2_p16_arr,
        rho_p84=rho2_p84_arr,
        rho_stderr=rho2_stderr_arr,
        nbin=nbin_arr,
        nboot=np.full_like(nbin_arr, args.nboot),
        ngood_boot=ngood_boot_arr,
        Lambda2_fit_npts=fit_npts_arr,
        Lambda2_fit_rms=fit_rms_arr,
        normalization_note=np.array(NORMALIZATION_NOTE),
        folder=folder_arr,
    )

    out_csv_rho2 = os.path.splitext(out_npz_rho2)[0] + ".csv"
    with open(out_csv_rho2, "w") as f:
        f.write(
            "T,beta,dt,Lambda_mid,Lambda2_mid,rho_mean,rho_p16,rho_p84,rho_stderr,nbin,nboot,ngood_boot,Lambda2_fit_npts,Lambda2_fit_rms,folder\n"
        )
        for i in range(T_arr.size):
            f.write(
                f"{T_arr[i]},{beta_arr[i]},{dt_arr[i]},{lam_mid_arr[i]},{lam2_mid_arr[i]},{rho2_mean_arr[i]},{rho2_p16_arr[i]},{rho2_p84_arr[i]},{rho2_stderr_arr[i]},{nbin_arr[i]},{args.nboot},{ngood_boot_arr[i]},{fit_npts_arr[i]},{fit_rms_arr[i]},{repr(folder_arr[i])}\n"
            )

    print("Wrote", out_npz_rho1)
    print("Wrote", out_csv_rho1)
    print("Wrote", out_npz_rho2)
    print("Wrote", out_csv_rho2)
    print(f"Processed {len(rows)} temperature points under {base}")
    print("NOTE:", NORMALIZATION_NOTE)
    if warnings:
        print("\nWarnings (first 20):")
        for w in warnings[:20]:
            print(w)



if __name__ == "__main__":
    main()



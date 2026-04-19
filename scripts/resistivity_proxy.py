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

import os, re, argparse
import numpy as np


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
    tau = np.load(taufile).astype(float)
    if tau.ndim != 1 or tau.size < 3:
        raise ValueError(f"bad tau shape: {tau.shape}")
    L = tau.size
    dt = float(tau[1] - tau[0])
    # Convention: tau = 0, dt, ..., (L-1)dt  (no endpoint); beta = L*dt
    beta = float(dt * L)
    return tau, dt, beta


def _load_corr(subpath: str, prefix: str):
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

    corr_perbin = np.load(jjfile_perbin)
    if corr_perbin.ndim != 2:
        raise ValueError(f"bad perbin shape in {jjfile_perbin}: {corr_perbin.shape}")

    corr_mean = corr_perbin.mean(axis=0).astype(float)
    return corr_mean, corr_perbin.astype(float)


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

def Lambda_2nd_deriv(corr: np.ndarray, dt: float):
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
    w = 3  # half-window in time steps; uses ~ (2w+1) points when available
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
    return float(2.0 * a)
    
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
        rel, pfx, Tstr = [s.strip() for s in item.split(",")]
        T = float(Tstr)
        dpath = os.path.join(base, rel)

        corr_mean, corr_perbin = _load_corr(dpath, pfx)
        if args.sym:
            corr_mean = _symmetrize_about_beta_over_2(corr_mean)
            if corr_perbin is not None:
                L = corr_perbin.shape[1]
                idx_partner = (-np.arange(L)) % L
                corr_perbin = 0.5 * (corr_perbin + corr_perbin[:, idx_partner])

        tau, dt, beta = _load_tau(dpath)

        if beta > 0:
            T_from_beta = 1.0 / beta
            if abs(T - T_from_beta) / max(T, T_from_beta, 1e-12) > 0.02:
                warnings.append(
                    f"[WARN] {dpath}: T(from name)={T:g} differs from 1/beta={T_from_beta:g} (beta={beta:g})."
                )

        lam_mid, mid = Lambda_xx_beta_over_2(corr_mean)
        lam2_mid = Lambda_2nd_deriv(corr_mean, dt)

        # Mean-curve proxies (kept as auxiliary outputs)
        rho1_mean = float(rho1(T, lam_mid))
        rho2_mean = float(rho2(lam_mid, lam2_mid))

        # Optional: per-bin uncertainty estimates (recommended)
        rho1_p16 = rho1_p84 = rho2_p16 = rho2_p84 = np.nan
        rho1_stderr = rho2_stderr = np.nan
        nbin = 0

        if corr_perbin is not None:
            nbin, L = corr_perbin.shape
            if L != corr_mean.size:
                raise ValueError(
                    f"perbin L mismatch: perbin {L} vs mean {corr_mean.size} in {dpath}"
                )

            # Lambda(beta/2) per bin
            lam_mid_b = np.array([
                Lambda_xx_beta_over_2(corr_perbin[i, :])[0] for i in range(nbin)
            ], dtype=float)

            # Lambda''(beta/2) per bin
            lam2_mid_b = np.array([
                Lambda_2nd_deriv(corr_perbin[i, :], dt) for i in range(nbin)
            ], dtype=float)

            # Filter out pathological bins
            good = np.isfinite(lam_mid_b) & np.isfinite(lam2_mid_b) & (np.abs(lam_mid_b) > 1e-14)
            ngood = int(np.sum(good))
            if ngood < nbin:
                warnings.append(f"[WARN] {dpath}: dropped {nbin - ngood} out of {nbin} bins due to non-finite or tiny Lambda(beta/2)")
            if ngood >= 1:
                rho1_b = (np.pi * T * T) / lam_mid_b[good]
                rho2_b = lam2_mid_b[good] / (2.0 * np.pi * lam_mid_b[good] * lam_mid_b[good])
                rho1_med = float(np.median(rho1_b))
                rho2_med = float(np.median(rho2_b))
                # Basic uncertainty summaries (percentiles and standard error)
                rho1_p16, rho1_p84 = np.percentile(rho1_b, [16, 84])
                rho2_p16, rho2_p84 = np.percentile(rho2_b, [16, 84])
                if ngood >= 2:
                    rho1_stderr = float(np.std(rho1_b, ddof=1) / np.sqrt(ngood))
                    rho2_stderr = float(np.std(rho2_b, ddof=1) / np.sqrt(ngood))
                else:
                    rho1_stderr = np.nan
                    rho2_stderr = np.nan
            else:
                warnings.append(f"[WARN] {dpath}: no valid bins for uncertainty estimates")

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
    folder_arr = np.array([r[14] for r in rows], dtype=object)

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
        folder=folder_arr,
    )

    out_csv_rho1 = os.path.splitext(out_npz_rho1)[0] + ".csv"
    with open(out_csv_rho1, "w") as f:
        f.write(
            "T,beta,dt,Lambda_mid,rho_mean,rho_p16,rho_p84,rho_stderr,nbin,folder\n"
        )
        for i in range(T_arr.size):
            f.write(
                f"{T_arr[i]},{beta_arr[i]},{dt_arr[i]},{lam_mid_arr[i]},{rho1_mean_arr[i]},{rho1_p16_arr[i]},{rho1_p84_arr[i]},{rho1_stderr_arr[i]},{nbin_arr[i]},{repr(folder_arr[i])}\n"
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
        folder=folder_arr,
    )

    out_csv_rho2 = os.path.splitext(out_npz_rho2)[0] + ".csv"
    with open(out_csv_rho2, "w") as f:
        f.write(
            "T,beta,dt,Lambda_mid,Lambda2_mid,rho_mean,rho_p16,rho_p84,rho_stderr,nbin,folder\n"
        )
        for i in range(T_arr.size):
            f.write(
                f"{T_arr[i]},{beta_arr[i]},{dt_arr[i]},{lam_mid_arr[i]},{lam2_mid_arr[i]},{rho2_mean_arr[i]},{rho2_p16_arr[i]},{rho2_p84_arr[i]},{rho2_stderr_arr[i]},{nbin_arr[i]},{repr(folder_arr[i])}\n"
            )

    print("Wrote", out_npz_rho1)
    print("Wrote", out_csv_rho1)
    print("Wrote", out_npz_rho2)
    print("Wrote", out_csv_rho2)
    print(f"Processed {len(rows)} temperature points under {base}")
    if warnings:
        print("\nWarnings (first 20):")
        for w in warnings[:20]:
            print(w)



if __name__ == "__main__":
    main()






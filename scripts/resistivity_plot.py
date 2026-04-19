#!/usr/bin/env python3
'''
resistivity_plot.py
phoenixm@stanford.edu

Usage examples:
(1) MaxEnt only (nearest-omega DC), no proxies
    python3 /home/users/phoenixm/scripts/resistivity_plot.py \
        --base /scratch/users/phoenixm/dqmc_runs/6x6_tp0_nflux0/n0.6_resistivity \
        --maxent_subdir maxent_out \
        --dc_method nearest \
        --divide_pi \
        --show_proxy none \
        --items \
            "T0.05_beta20_U-6/mu-0.462713,0.05" \
            "T0.1_beta10_U-6/mu-0.463656,0.1" \
            "T0.125_beta8_U-6/mu-0.464428,0.125" \
            "T0.166667_beta6_U-6/mu-0.466893,0.166667" \
        --out /scratch/users/phoenixm/dqmc_runs/6x6_tp0_nflux0/n0.6_resistivity/rho_maxent_nearest.png

(2) MaxEnt + proxies, Drude DC, with zoom inset and WITHOUT grey connectors
    python3 /home/users/phoenixm/scripts/resistivity_plot.py \
        --base /scratch/users/phoenixm/dqmc_runs/6x6_tp0_nflux0/n0.6_resistivity \
        --maxent_subdir maxent_out \
        --dc_method drude \
        --divide_pi \
        --show_proxy all \
        --proxy1 /scratch/users/phoenixm/dqmc_runs/6x6_tp0_nflux0/n0.6_resistivity/proxies_rho1.npz \
        --proxy2 /scratch/users/phoenixm/dqmc_runs/6x6_tp0_nflux0/n0.6_resistivity/proxies_rho2.npz \
        --x_range 0.2 1.0 \
        --items \
            "T0.05_beta20_U-6/mu-0.462713,0.05" \
            "T0.1_beta10_U-6/mu-0.463656,0.1" \
            "T0.125_beta8_U-6/mu-0.464428,0.125" \
            "T0.166667_beta6_U-6/mu-0.466893,0.166667" \
        --out /scratch/users/phoenixm/dqmc_runs/6x6_tp0_nflux0/n0.6_resistivity/rho_all_drude.png
'''

import os, argparse, re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.optimize import curve_fit

def _drude_model(w, sigma_reg, D, gamma):
    """
    Drude + constant background model for Re sigma(omega).

    sigma(w) = sigma_reg + (D * gamma) / (w^2 + gamma^2)

    Here gamma = 1/tau is the scattering rate. DC limit is:
        sigma_dc = sigma_reg + D/gamma.
    """
    w = np.asarray(w, dtype=float)
    return sigma_reg + (D * gamma) / (w * w + gamma * gamma)


def drude_fit_sigma_dc(omega: np.ndarray, s_all: np.ndarray, n_fit_points: int):
    """
    Fit Drude peak to the lowest-frequency window and return DC sigma.

    This function performs an independent nonlinear fit for EACH spectrum sample (row of s_all)
    using the lowest `n_fit_points` positive frequencies in `omega`, then computes
        sigma_dc = sigma_reg + D/gamma
    per sample. Failed fits are dropped.

    Args:
        omega: (Nw,) frequency grid. Must be positive and increasing (monotonic).
        s_all: (Nsamp, Nw) MaxEnt/bootstrap samples of Re sigma(omega).
        n_fit_points: number of lowest-frequency points to use in the fit.

    Returns:
        sigma_dc_good: (N_good,) array of fitted sigma_dc values (finite, positive).
        n_good: number of successful fits.
        n_tot: total number of samples.

    Raises:
        ValueError: if shapes are inconsistent or n_fit_points is invalid.
    """
    omega = np.asarray(omega, dtype=float)
    s_all = np.asarray(s_all, dtype=float)

    if omega.ndim != 1 or s_all.ndim != 2 or s_all.shape[1] != omega.size:
        raise ValueError(f"bad shapes: omega {omega.shape}, s_all {s_all.shape}")
    if n_fit_points < 3 or n_fit_points > omega.size:
        raise ValueError(f"n_fit_points must be in [3, {omega.size}], got {n_fit_points}")

    # Require strictly positive omega
    if np.min(omega) <= 0:
        raise ValueError("drude_fit_sigma_dc() expects positive omega grid")
    # Ensure omega is strictly increasing so omega[:n_fit_points] is the lowest-frequency window.
    if omega.size >= 2 and not np.all(np.diff(omega) > 0):
        raise ValueError("omega grid must be strictly increasing (monotonic) for Drude fitting")
    w = omega[:n_fit_points]

    # Parameter bounds: sigma_reg>=0, D>=0, gamma>0
    lb = (0.0, 0.0, 1e-12)
    ub = (np.inf, np.inf, np.inf)

    sigma_dc_list = []
    n_tot = int(s_all.shape[0])

    # Precompute a conservative scale for initial guesses.
    w_max = float(w[-1])
    gamma0_base = max(w_max / 2.0, 1e-6)

    for k in range(n_tot):
        y = s_all[k, :n_fit_points]
        if not np.all(np.isfinite(y)):
            continue

        # Initial guesses
        # Background starts near the minimum in the fitting window
        sigma_reg0 = float(np.min(y))
        # A crude dc estimate from the lowest available frequency.
        sigma_dc0 = float(y[0])
        gamma0 = gamma0_base
        D0 = max((sigma_dc0 - sigma_reg0) * gamma0, 0.0)
        p0 = (sigma_reg0, D0, gamma0)

        try:
            popt, _ = curve_fit(
                _drude_model,
                w,
                y,
                p0=p0,
                bounds=(lb, ub),
                maxfev=20000,
            )
            sigma_reg, D, gamma = [float(x) for x in popt]
            if not (np.isfinite(sigma_reg) and np.isfinite(D) and np.isfinite(gamma) and gamma > 0):
                continue

            sigma_dc = sigma_reg + D / gamma
            if np.isfinite(sigma_dc) and sigma_dc > 0:
                sigma_dc_list.append(float(sigma_dc))
        except Exception:
            # Drop failed fits.
            continue

    sigma_dc_good = np.asarray(sigma_dc_list, dtype=float)
    return sigma_dc_good, int(sigma_dc_good.size), n_tot


def convergence_test_select_npoints(omega: np.ndarray, s_all: np.ndarray):
    """
    Select the number of low-frequency points for Drude fitting via a simple convergence test.

    Strategy:
      - Build a sequence of candidate n_fit_points values.
      - For each candidate, fit Drude to the MEDIAN spectrum (across samples) in that window.
      - Define sigma_dc(n) from the fitted parameters.
      - Choose the smallest n such that sigma_dc(n), sigma_dc(n+1), sigma_dc(n+2) are stable
        within a relative tolerance.

    Notes:
        This is a pragmatic stability test, not a rigorous model selection criterion.
        If no candidate passes, we fall back to the largest candidate.

    Returns:
        n_best: chosen number of points.
        diag: dict with arrays for diagnostics (candidates, sigma_dc_est, rel_deltas).
    """
    omega = np.asarray(omega, dtype=float)
    s_all = np.asarray(s_all, dtype=float)

    if omega.ndim != 1 or s_all.ndim != 2 or s_all.shape[1] != omega.size:
        raise ValueError(f"bad shapes: omega {omega.shape}, s_all {s_all.shape}")
    if np.min(omega) <= 0:
        raise ValueError("Expects strictly positive omega grid")
    # Ensure omega is strictly increasing so omega[:n] always refers to the lowest-frequency points.
    if omega.size >= 2 and not np.all(np.diff(omega) > 0):
        raise ValueError("omega grid must be strictly increasing (monotonic) for convergence test")

    Nw = int(omega.size)
    # Candidate window sizes: 4..min(20, Nw) (step 1)
    nmax = min(20, Nw)
    candidates = np.arange(4, nmax + 1, dtype=int)

    # Robust representative spectrum to avoid per-sample fitting cost during selection
    y_med = np.median(s_all, axis=0)

    sigma_dc_est = np.full(candidates.size, np.nan, dtype=float)

    # Fit the median spectrum using the existing per-sample Drude fitting routine.
    # We pass the median spectrum as a single "sample" (Nsamp=1) to reuse the same fitter.
    s_med = y_med[None, :]

    for i, n in enumerate(candidates):
        try:
            sigma_dc_good, n_good, _ = drude_fit_sigma_dc(omega, s_med, n)
            if n_good >= 1:
                sigma_dc_est[i] = float(sigma_dc_good[0])
        except Exception:
            continue

    # Relative changes between successive candidates.
    rel_deltas = np.full(candidates.size - 1, np.nan, dtype=float)
    for i in range(candidates.size - 1):
        a = sigma_dc_est[i]
        b = sigma_dc_est[i + 1]
        if np.isfinite(a) and np.isfinite(b) and b != 0:
            rel_deltas[i] = abs(a - b) / abs(b)

    # Convergence criterion: three consecutive sigma_dc within tol.
    tol = 0.02  # 2% relative stability
    n_best = int(candidates[-1])
    for i in range(candidates.size - 2):
        a = sigma_dc_est[i]
        b = sigma_dc_est[i + 1]
        c = sigma_dc_est[i + 2]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            continue
        if b == 0 or c == 0:
            continue
        r1 = abs(a - b) / abs(b)
        r2 = abs(b - c) / abs(c)
        if r1 < tol and r2 < tol:
            n_best = int(candidates[i])
            break

    diag = {
        "candidates": candidates,
        "sigma_dc_est": sigma_dc_est,
        "rel_deltas": rel_deltas,
        "tol": tol,
    }
    return n_best, diag


def load_dc_sigma(dpath: str, prefix: str, divide_pi: bool=True, dc_method: str="nearest"):
    """
    Load DC sigma samples (bootstrap / MaxEnt samples).

    Behavior:
      - If omega grid contains 0: use sigma(0).
      - If omega grid is strictly positive:
          * dc_method='nearest': use sigma at the frequency closest to 0 (fast).
          * dc_method='drude'  : use convergence test + Drude fit to extrapolate sigma_dc (slow).

    Returns:
      good: array of DC sigma samples (finite, nonzero)
      n_good: number of good samples
      n_tot: total number of samples in the original s_all
    """

    sp = os.path.join(dpath, prefix + "s_all.npy")
    wp = os.path.join(dpath, prefix + "omega.npy")
    if not (os.path.exists(sp) and os.path.exists(wp)):
        raise FileNotFoundError(f"Missing {sp} or {wp}")

    omega = np.load(wp)  # (Nw,)
    s_all = np.load(sp)  # (Nsamp, Nw)
    if omega.ndim != 1 or s_all.ndim != 2 or s_all.shape[1] != omega.size:
        raise ValueError(f"bad shapes: omega {omega.shape}, s_all {s_all.shape}")

    # Defensive check: routines assume omega is sorted ascending.
    if omega.size >= 2 and not np.all(np.diff(omega) > 0):
        raise ValueError("omega grid must be strictly increasing (monotonic)")

    omega_abs_min_idx = int(np.argmin(np.abs(omega)))

    # Case 1: omega grid contains exactly 0 -> take sigma(0).
    if np.isclose(float(omega[omega_abs_min_idx]), 0.0, atol=1e-12):
        dc_sigma = s_all[:, omega_abs_min_idx]

    # Case 2: omega grid is strictly positive -> choose method.
    elif float(omega[omega_abs_min_idx]) > 0:
        if dc_method == "nearest":
            dc_sigma = s_all[:, omega_abs_min_idx]
        elif dc_method == "drude":
            n_best, diag = convergence_test_select_npoints(omega, s_all)
            dc_sigma, n_good_fit, _ = drude_fit_sigma_dc(omega, s_all, n_best)
            if n_good_fit == 0:
                raise RuntimeError(
                    f"All Drude fits failed in {dpath} (prefix={prefix}); "
                    f"n_best={n_best}, diag_tol={diag.get('tol', None)}"
                )
        else:
            raise ValueError(f"Unknown dc_method: {dc_method}")

    else:
        raise ValueError("Unsupported omega grid (contains negative values but no omega=0 point).")

    # Keep only finite, nonzero samples (rho = 1/sigma)
    good_mask = np.isfinite(dc_sigma) & (dc_sigma != 0.0)
    good = dc_sigma[good_mask]
    n_good, n_tot = int(good.shape[0]), int(s_all.shape[0])

    if n_good == 0:
        raise RuntimeError(f"All bootstrap spectra are invalid in {dpath} (prefix={prefix})")

    if divide_pi:
        good = good / np.pi

    return good, n_good, n_tot

def sigma_to_rho(dc_sigma: np.ndarray):
    """
    Convert DC conductivity samples to resistivity statistics.

    Args:
        dc_sigma: (ngood,) array of sigma_dc samples.

    Returns:
        rho_mean: mean of rho samples.
        yerr: shape (2,) array with [rho - p16, p84 - rho].
        rho_p16, rho_p84: percentiles.
        rho_stderr: standard error of the mean of rho samples (NaN if ngood<2).
        nsamp: number of samples used.
    """

    sigma = np.asarray(dc_sigma, dtype=float)
    sigma = sigma[np.isfinite(sigma) & (sigma != 0.0)]
    ngood = int(sigma.size)
    if ngood < 2: 
        raise ValueError(f"Number of valid DC sigma samples (got {ngood}) is not enough.")

    rho = 1.0 / sigma
    # TODO: mean or median?
    rho_mean = float(np.mean(rho))
    rho_p16, rho_p84 = np.percentile(rho, [16, 84])
    rho_stderr = float(np.std(rho, ddof=1) / np.sqrt(ngood))
    yerr = np.array([rho_stderr, rho_stderr], dtype=float)
    return rho_mean, yerr, float(rho_p16), float(rho_p84), rho_stderr, ngood

def load_proxy(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    for k in ("T", "rho_mean"):
        if k not in d:
            raise KeyError(f"{npz_path} missing key: {k}")
        
    T = d["T"].astype(float)
    rho = d["rho_mean"].astype(float)

    if T.shape != rho.shape:
        raise ValueError(f"shape mismatch in {npz_path}: T {T.shape} vs rho {rho.shape}")
    
    rho_p16 = d.get("rho_p16", np.full_like(rho, np.nan)).astype(float)
    rho_p84 = d.get("rho_p84", np.full_like(rho, np.nan)).astype(float)
    rho_stderr = d.get("rho_stderr", np.full_like(rho, np.nan)).astype(float)
    yerr = np.vstack([rho_stderr, rho_stderr])
    return T, rho, yerr, rho_stderr

def parse_T_from_folder(name: str) -> float:
    m = re.search(r"(?:^|_)T_?([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", name)
    if not m:
        raise ValueError(f"cannot parse T from folder name: {name}")
    return float(m.group(1))

def maxent_prefix(dpath: str, pfx: str) -> str:
    if not os.path.isdir(dpath):
        raise FileNotFoundError(f"MaxEnt directory not found: {dpath}")

    pfx = "" if pfx is None else str(pfx)
    suffix = "omega.npy" if pfx == "" else f"{pfx}omega.npy"
    cand = [fn for fn in os.listdir(dpath) if fn.endswith(suffix)]

    if not cand:
        raise FileNotFoundError(
            f"No MaxEnt omega file matching '*{suffix}' in {dpath}"
        )

    if len(cand) != 1:
        cand_sorted = sorted(cand)
        raise RuntimeError(
            f"Ambiguous MaxEnt omega files in {dpath} matching '*{suffix}': {cand_sorted}"
        )

    fn = cand[0]
    return fn[:-len("omega.npy")]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base run directory containing T_* subfolders")
    p.add_argument("--maxent_prefix", default="", help="Prefix (fixed part) used in MaxEnt filenames when searching for '*<prefix>omega.npy'. Leave empty to auto-detect based on files ending with 'omega.npy'.")
    p.add_argument("--maxent_subdir", default="maxent_out", help="Subdir under each T_* that contains MaxEnt outputs")
    p.add_argument(
        "--show_proxy",
        choices=["none", "1", "2", "all"],
        default="none",
        help="Which proxy curve(s) to show: none, 1, 2, or all.",
    )
    p.add_argument("--proxy1", help="NPZ for proxy-1 outputs")
    p.add_argument("--proxy2", help="NPZ for proxy-2 outputs")
    p.add_argument("--divide_pi", action="store_true", help="Divide DC sigma by pi before inverting")
    p.add_argument("--convergence_plots", action="store_true", 
                   help="Show the Drude fitting convergenst test plot for each T or not")
    p.add_argument("--dc_method", choices=["nearest", "drude"], required=True, 
                   help="nearest: use sigma at the frequency closest to 0 (fast); drude: use Drude fit to extrapolate sigma_dc (slow).")
    p.add_argument("--out", default="", help="If set, save figure to this path")
    p.add_argument(
        "--x_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="If provided, add an inset zoomed to [XMIN, XMAX] on the main plot (main plot unchanged) and also write a separate zoom figure.",
    )
    p.add_argument("--items", nargs="+", required=True,
                    help=("List of relative paths and temperatures. Example: 'T0.05_beta20_U-6/mu-0.462713/,0.05'"))    
    p.add_argument("--linear_fit", action="store_true",
        help=("If set, perform an iterative linear fit of MaxEnt resistivity rho(T) for T>=2. "),
    )
    p.add_argument("--highT_slope", type=float, default=None, help="If set, plot a dashed high-T limit line.")
    
    args = p.parse_args()

    # Load proxies (as requested)
    do_p1 = args.show_proxy in ("1", "all")
    do_p2 = args.show_proxy in ("2", "all")

    if do_p1:
        if not args.proxy1:
            raise ValueError("--proxy1 must be provided when --show_proxy is '1' or 'all'")
        T1, rho1, yerr1, rho1_stderr = load_proxy(args.proxy1)
    if do_p2:
        if not args.proxy2:
            raise ValueError("--proxy2 must be provided when --show_proxy is '2' or 'all'")
        T2, rho2, yerr2, rho2_stderr = load_proxy(args.proxy2)

    # Load MaxEnt dc rho per temperature folder
    Ts_me, rho_me, yerr_me = [], [], []
    nsamp_me = []
    
    path = args.base
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Base path not found or not a directory: {path}")

    for item in args.items:
        relpath, Tstr = [s.strip() for s in item.split(",")]
        T = float(Tstr)

        dpath = os.path.join(path, relpath, args.maxent_subdir)
        if not os.path.isdir(dpath):
            continue

        prefix = maxent_prefix(dpath, args.maxent_prefix)

        if args.convergence_plots and args.dc_method == "drude":
            # --- Drude convergence diagnostic plot (saved under --base) ---
            # Only meaningful when omega grid is strictly positive
            try:
                wp = os.path.join(dpath, prefix + "omega.npy")
                sp = os.path.join(dpath, prefix + "s_all.npy")
                if os.path.exists(wp) and os.path.exists(sp):
                    omega_dbg = np.load(wp)
                    s_all_dbg = np.load(sp)
                    omega_dbg = np.asarray(omega_dbg, dtype=float)
                    if omega_dbg.ndim == 1 and np.min(omega_dbg) > 0:
                        n_best_dbg, diag_dbg = convergence_test_select_npoints(omega_dbg, s_all_dbg)

                        cand = np.asarray(diag_dbg.get("candidates"), dtype=float)
                        sig = np.asarray(diag_dbg.get("sigma_dc_est"), dtype=float)

                        # Save a simple figure showing sigma_dc(n) vs n and the chosen n_best.
                        plt.figure()
                        plt.plot(cand, sig, marker='o', linestyle='-')
                        plt.axvline(float(n_best_dbg), linestyle='--')
                        plt.xlabel("Number of low-frequency points used in Drude fit")
                        plt.ylabel(r"Estimated $\\sigma_{dc}$ from median spectrum")
                        plt.title(f"Drude Peak Fit Convergence Test T={T}")
                        plt.tight_layout()

                        # Write into --base (one file per temperature)
                        T_str = ("%g" % T)
                        out_png = os.path.join(args.base, f"drude_convergence_T{T_str}.png")
                        plt.savefig(out_png, dpi=200)
                        plt.close()
            except Exception:
                # Diagnostics should never stop the main pipeline.
                pass
            # --- end diagnostic plot ---

        try:
            sigma_dc_samples, n_good, n_tot = load_dc_sigma(dpath, prefix,
                                                            divide_pi=args.divide_pi,
                                                            dc_method=args.dc_method)
            rho_mean, yerr, rho_p16, rho_p84, rho_stderr, nsamp = sigma_to_rho(sigma_dc_samples)
        except Exception as e:
            print(f"[SKIP MaxEnt] {relpath}: {type(e).__name__}: {e}")
            continue

        Ts_me.append(T)
        rho_me.append(rho_mean)
        yerr_me.append(yerr)
        nsamp_me.append(nsamp)

    Ts_me = np.array(Ts_me, dtype=float)
    rho_me = np.array(rho_me, dtype=float)
    if len(yerr_me):
        yerr_me = np.array(yerr_me, dtype=float)
        if yerr_me.ndim != 2 or yerr_me.shape[1] != 2:
            raise ValueError(f"bad yerr_me shape after collection: {yerr_me.shape}")
        yerr_me = yerr_me.T  # (2, N)
    else:
        yerr_me = np.zeros((2, 0), dtype=float)

    # sort MaxEnt points
    if Ts_me.size:
        idx = np.argsort(Ts_me)
        Ts_me, rho_me, yerr_me = Ts_me[idx], rho_me[idx], yerr_me[:, idx]

        if args.out and Ts_me.size:
            out_dir = os.path.dirname(args.out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            root, _ = os.path.splitext(args.out)
            np.save(f"{root}_T.npy", Ts_me)
            np.save(f"{root}_rho_maxent.npy", rho_me)
            print("Wrote", f"{root}_T.npy")
            print("Wrote", f"{root}_rho_maxent.npy")

    def _linear_fit_slope_r2(x: np.ndarray, y: np.ndarray):
        """Return (slope, intercept, r2) for y = slope*x + intercept."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 2:
            raise ValueError("need at least two points for linear fit")
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        # If y is constant, treat as perfect fit only if residual is also ~0
        if ss_tot == 0.0:
            r2 = 1.0 if ss_res == 0.0 else 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot
        return float(slope), float(intercept), float(r2)

    # Optional: iterative linear fit on the MaxEnt curve for T>=2
    if args.linear_fit:
        if Ts_me.size < 2:
            print("[linear_fit] not enough MaxEnt points to fit")
        else:
            mask = (Ts_me >= 2.0) & np.isfinite(Ts_me) & np.isfinite(rho_me)
            x = Ts_me[mask]
            y = rho_me[mask]
            # Ensure increasing x
            order = np.argsort(x)
            x = x[order]
            y = y[order]

            r2_thr = 0.99
            while x.size >= 2:
                try:
                    slope, intercept, r2 = _linear_fit_slope_r2(x, y)
                except Exception as e:
                    print(f"[linear_fit] failed: {type(e).__name__}: {e}")
                    break

                if r2 > r2_thr:
                    print(f"[linear_fit] T>= {x[0]:g} (N={x.size}) slope={slope:.6g} R^2={r2:.6f}")
                    break

                # Drop the lowest-T point and refit
                x = x[1:]
                y = y[1:]

    def _ylim_for_xrange(xmin: float, xmax: float):
        """Compute reasonable y-limits (including error bars) for data within [xmin, xmax]."""
        y_lows = []
        y_highs = []

        # MaxEnt
        if Ts_me.size:
            m = (Ts_me >= xmin) & (Ts_me <= xmax)
            if np.any(m):
                y = rho_me[m]
                if yerr_me.size:
                    yl = y - yerr_me[0, m]
                    yh = y + yerr_me[1, m]
                else:
                    yl = y
                    yh = y
                y_lows.append(np.min(yl))
                y_highs.append(np.max(yh))

        # Proxy-1
        if do_p1:
            m = (T1 >= xmin) & (T1 <= xmax)
            if np.any(m):
                y = rho1[m]
                yl = y - yerr1[0, m]
                yh = y + yerr1[1, m]
                y_lows.append(np.min(yl))
                y_highs.append(np.max(yh))

        # Proxy-2
        if do_p2:
            m = (T2 >= xmin) & (T2 <= xmax)
            if np.any(m):
                y = rho2[m]
                yl = y - yerr2[0, m]
                yh = y + yerr2[1, m]
                y_lows.append(np.min(yl))
                y_highs.append(np.max(yh))

        if not y_lows:
            return None

        y0 = float(np.min(y_lows))
        y1 = float(np.max(y_highs))
        if not (np.isfinite(y0) and np.isfinite(y1)):
            return None
        if y0 == y1:
            pad = 1.0 if y0 == 0 else 0.05 * abs(y0)
            return y0 - pad, y1 + pad
        pad = 0.05 * (y1 - y0)
        return y0 - pad, y1 + pad

    # Plot
    fig, ax = plt.subplots()
    maxent_color = None
    Tmax_me = None
    if Ts_me.size:
        eb_me = ax.errorbar(Ts_me, rho_me, yerr=yerr_me, fmt="o-", capsize=3,
                            label="MaxEnt DC resistivity")
        try:
            maxent_color = eb_me.lines[0].get_color()
        except Exception:
            maxent_color = None
        Tmax_me = float(np.max(Ts_me))
    if do_p1:
        m1 = (T1 <= 1.0)
        ax.errorbar(T1[m1], rho1[m1], yerr=yerr1[:, m1], fmt="s--", capsize=3,
                    label=r"$\rho_{\text{proxy}} = \pi T^2/\Lambda(\beta/2)$")
    if do_p2:
        m2 = (T2 <= 1.0)
        ax.errorbar(T2[m2], rho2[m2], yerr=yerr2[:, m2], fmt="d--", capsize=3,
                    label=r"$\rho_{\text{proxy}} = \Lambda^{\prime \prime}(\beta/2)/(2\pi\Lambda(\beta/2)^2)$")

    # Plot high-T slope if requested
    if args.highT_slope is not None and Tmax_me is not None:
        slope = float(args.highT_slope)
        x0 = Tmax_me + 1.0
        x1 = Tmax_me + 3.0
        xs = np.array([x0, x1], dtype=float)
        ys = slope * xs  # intercept = 0
        ax.plot(xs, ys, "--", color=maxent_color, label="high-T limit")
        ax.set_xlim(-0.05, x1)

    if args.highT_slope is None:
        ax.set_xlim(left=-0.05)

    ax.set_xlabel("T")
    ax.set_ylabel(r"$\rho$")
    ax.legend()
    ax.grid()
    fig.tight_layout()

    if args.x_range is not None:
        xmin, xmax = float(args.x_range[0]), float(args.x_range[1])
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        yl = _ylim_for_xrange(xmin, xmax)
        if yl is not None:
            axins = inset_axes(ax, width="45%", height="45%", loc="upper right")
            axins.set_xlim(xmin, xmax)
            axins.set_ylim(yl[0], yl[1])

            if Ts_me.size:
                axins.errorbar(Ts_me, rho_me, yerr=yerr_me, fmt="o-", capsize=2)
            if do_p1:
                m1 = (T1 <= 1.0)
                axins.errorbar(T1[m1], rho1[m1], yerr=yerr1[:, m1], fmt="s--", capsize=2)
            if do_p2:
                m2 = (T2 <= 1.0)
                axins.errorbar(T2[m2], rho2[m2], yerr=yerr2[:, m2], fmt="d--", capsize=2)

            axins.grid(True)

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(args.out, dpi=200)
        print("Wrote", args.out)
        plt.close(fig)

        if args.x_range is not None:
            xmin, xmax = float(args.x_range[0]), float(args.x_range[1])
            if xmin > xmax:
                xmin, xmax = xmax, xmin

            yl = _ylim_for_xrange(xmin, xmax)
            if yl is not None:
                figz, axz = plt.subplots()
                if Ts_me.size:
                    axz.errorbar(Ts_me, rho_me, yerr=yerr_me, fmt="o-", capsize=3,
                                 label="MaxEnt DC resistivity")
                if do_p1:
                    m1 = (T1 <= 1.0)
                    axz.errorbar(T1[m1], rho1[m1], yerr=yerr1[:, m1], fmt="s--", capsize=3,
                                 label=r"$\rho_{\text{proxy}} = \pi T^2/\Lambda(\beta/2)$")
                if do_p2:
                    m2 = (T2 <= 1.0)
                    axz.errorbar(T2[m2], rho2[m2], yerr=yerr2[:, m2], fmt="d--", capsize=3,
                                 label=r"$\rho_{\text{proxy}} = \Lambda^{\prime \prime}(\beta/2)/(2\pi\Lambda(\beta/2)^2)$")

                axz.set_xlim(xmin, xmax)
                axz.set_ylim(yl[0], yl[1])
                axz.set_xlabel("T")
                axz.set_ylabel(r"$\rho$")
                axz.legend()
                axz.grid()
                figz.tight_layout()

                root, ext = os.path.splitext(args.out)
                if not ext:
                    ext = ".png"
                zoom_out = f"{root}_x{xmin:g}-{xmax:g}{ext}"
                figz.savefig(zoom_out, dpi=200)
                plt.close(figz)
                print("Wrote", zoom_out)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

utilpath = Path(__file__).resolve().parents[1] / "util"
sys.path.insert(0, str(utilpath))

import util


def parse_T_beta_U(t_dir: Path):
    m = re.search(
        r"T([0-9eE+\-.]+)_beta([0-9eE+\-.]+)_U([+\-]?[0-9eE.]+)",
        t_dir.name,
    )
    if not m:
        return np.nan, np.nan, np.nan
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def parse_mu_from_dirname(mu_dir: Path):
    m = re.search(r"mu([+\-]?[0-9eE.]+)", mu_dir.name)
    if not m:
        return np.nan
    return float(m.group(1))


def with_trailing_slash(p: Path):
    s = str(p)
    return s if s.endswith(os.sep) else s + os.sep


def get_mu_from_firstfile(mu_dir: Path):
    mu_dir_s = with_trailing_slash(mu_dir)
    try:
        mu = util.load_firstfile(mu_dir_s, "metadata/mu")[0]
        return float(mu)
    except Exception:
        return parse_mu_from_dirname(mu_dir)


def get_nsite_from_firstfile(mu_dir: Path):
    mu_dir_s = with_trailing_slash(mu_dir)
    d0 = util.load_firstfile(mu_dir_s, "meas_eqlt/density")[0]
    d0 = np.asarray(d0)
    if d0.ndim == 0:
        raise ValueError(f"Cannot infer Nsite from scalar density dataset in {mu_dir}")
    return int(d0.shape[0])


def load_mu_density_bins(mu_dir: Path):
    """
    Load per-bin density for one mu directory.

    Returns a dict containing:
        mu
        n_mean
        n_err
        n_bins_raw: per-bin per-site density, shape (n_bins_valid,)
        sign
        nsite

    The mean/error for n(mu) is computed through util.jackknife(sign, N),
    consistent with get_n_from_best_mu.py.
    """
    mu_dir_s = with_trailing_slash(mu_dir)

    n_sample, sign, density = util.load(
        mu_dir_s,
        "meas_eqlt/n_sample",
        "meas_eqlt/sign",
        "meas_eqlt/density",
    )

    n_sample = np.asarray(n_sample).reshape(-1)
    sign = np.asarray(sign).reshape(-1)
    density = np.asarray(density)

    if density.shape[0] != n_sample.shape[0]:
        nb = min(density.shape[0], n_sample.shape[0], sign.shape[0])
        density = density[:nb]
        n_sample = n_sample[:nb]
        sign = sign[:nb]

    nsite = get_nsite_from_firstfile(mu_dir)
    mask = n_sample == np.nanmax(n_sample)
    dsum = density.reshape(density.shape[0], -1).sum(axis=1)

    valid = (
        mask
        & np.isfinite(sign)
        & np.isfinite(dsum)
        & (sign != 0)
    )

    sign = sign[valid]
    dsum = dsum[valid]

    mu = get_mu_from_firstfile(mu_dir)

    if sign.size < 3:
        return {
            "mu": mu,
            "n_mean": np.nan,
            "n_err": np.nan,
            "n_bins_raw": np.array([], dtype=float),
            "sign": np.array([], dtype=float),
            "nsite": nsite,
            "mu_dir": mu_dir,
        }

    # util.jackknife returns mean and stderr for sign-weighted observable.
    N_mean, N_err = util.jackknife(sign, dsum)

    n_mean = N_mean / nsite
    n_err = N_err / nsite

    # Raw per-bin density used for the final jackknife over kappa.
    # For attractive Hubbard sign should be essentially +1, so this is the
    # natural raw bin observable for the n(mu) slope.
    n_bins_raw = dsum / nsite

    return {
        "mu": mu,
        "n_mean": n_mean,
        "n_err": n_err,
        "n_bins_raw": n_bins_raw,
        "sign": sign,
        "nsite": nsite,
        "mu_dir": mu_dir,
    }


def choose_local_window(mu, n_mean, n_err, filling, window, min_points, range_tol):
    """
    Choose a local contiguous window in mu around the target filling.
    This supports endpoint estimates such as n=1.0 near mu=0.
    """
    finite = np.isfinite(mu) & np.isfinite(n_mean)
    if finite.sum() < min_points:
        return None

    mu_f = mu[finite]
    n_f = n_mean[finite]
    err_f = n_err[finite]

    n_min = np.nanmin(n_f)
    n_max = np.nanmax(n_f)

    err_scale = 0.0
    if np.isfinite(err_f).any():
        err_scale = np.nanmax(err_f[np.isfinite(err_f)])

    tol = max(range_tol, 5.0 * err_scale, 1e-10)

    if filling < n_min - tol or filling > n_max + tol:
        return None

    # Clamp only for selecting the nearest point. The requested filling is
    # still used in the reported mu_at_n.
    filling_for_index = min(max(filling, n_min), n_max)

    center = int(np.nanargmin(np.abs(n_f - filling_for_index)))

    w = max(int(window), int(min_points))
    w = min(w, len(mu_f))

    start = center - w // 2
    end = start + w

    if start < 0:
        start = 0
        end = w
    if end > len(mu_f):
        end = len(mu_f)
        start = end - w

    idx_f = np.arange(len(mu))[finite]
    return idx_f[start:end]


def compute_kappa_for_T_dir(t_dir: Path, filling: float, window: int, min_points: int, range_tol: float):
    """
    For one T*_beta*_U*/ directory:
      1. load all mu*/ density bins;
      2. build n(mu);
      3. choose local window around target filling;
      4. compute dn/dmu by local linear fit;
      5. compute jackknife error from per-bin local slopes using util.jackknife.
    """
    T, beta, U = parse_T_beta_U(t_dir)
    mu_dirs = sorted([p for p in t_dir.glob("mu*/") if p.is_dir()])

    if len(mu_dirs) < min_points:
        return {
            "T": T,
            "beta": beta,
            "U": U,
            "mu_at_n": np.nan,
            "kappa": np.nan,
            "kappa_err": np.nan,
            "n_mu_points": len(mu_dirs),
            "status": "too_few_mu_dirs",
        }

    data = []
    for mu_dir in mu_dirs:
        try:
            item = load_mu_density_bins(mu_dir)
            if np.isfinite(item["mu"]) and np.isfinite(item["n_mean"]) and item["n_bins_raw"].size >= 3:
                data.append(item)
        except Exception as e:
            print(f"[WARN] failed to load {mu_dir}: {e}", file=sys.stderr)

    if len(data) < min_points:
        return {
            "T": T,
            "beta": beta,
            "U": U,
            "mu_at_n": np.nan,
            "kappa": np.nan,
            "kappa_err": np.nan,
            "n_mu_points": len(data),
            "status": "too_few_valid_mu_dirs",
        }

    data.sort(key=lambda d: d["mu"])

    mu = np.array([d["mu"] for d in data], dtype=float)
    n_mean = np.array([d["n_mean"] for d in data], dtype=float)
    n_err = np.array([d["n_err"] for d in data], dtype=float)

    # Remove duplicate mu values if present.
    unique_mu, unique_idx = np.unique(mu, return_index=True)
    if len(unique_mu) != len(mu):
        data = [data[i] for i in sorted(unique_idx)]
        mu = np.array([d["mu"] for d in data], dtype=float)
        n_mean = np.array([d["n_mean"] for d in data], dtype=float)
        n_err = np.array([d["n_err"] for d in data], dtype=float)

    idx = choose_local_window(
        mu=mu,
        n_mean=n_mean,
        n_err=n_err,
        filling=filling,
        window=window,
        min_points=min_points,
        range_tol=range_tol,
    )

    if idx is None or len(idx) < min_points:
        return {
            "T": T,
            "beta": beta,
            "U": U,
            "mu_at_n": np.nan,
            "kappa": np.nan,
            "kappa_err": np.nan,
            "n_mu_points": len(data),
            "status": "target_filling_not_bracketed_or_too_few_points",
        }

    mu_sel = mu[idx]
    n_sel = n_mean[idx]

    if len(np.unique(mu_sel)) < min_points:
        return {
            "T": T,
            "beta": beta,
            "U": U,
            "mu_at_n": np.nan,
            "kappa": np.nan,
            "kappa_err": np.nan,
            "n_mu_points": len(data),
            "status": "duplicate_mu_in_window",
        }

    # Mean-curve local linear fit:
    #     n(mu) = a * mu + b
    # compressibility kappa = dn/dmu = a
    a_mean, b_mean = np.polyfit(mu_sel, n_sel, deg=1)

    if not np.isfinite(a_mean) or abs(a_mean) < 1e-14:
        mu_at_n = np.nan
    else:
        mu_at_n = (filling - b_mean) / a_mean

    # Build per-bin local slopes. Truncate to common valid bin count across selected mu dirs.
    selected_data = [data[i] for i in idx]
    min_bins = min(d["n_bins_raw"].size for d in selected_data)

    if min_bins < 3:
        return {
            "T": T,
            "beta": beta,
            "U": U,
            "mu_at_n": mu_at_n,
            "kappa": np.nan,
            "kappa_err": np.nan,
            "n_mu_points": len(data),
            "status": "too_few_bins_for_jackknife",
        }

    slopes = []
    for ib in range(min_bins):
        y = np.array([d["n_bins_raw"][ib] for d in selected_data], dtype=float)
        if not np.isfinite(y).all():
            continue
        try:
            a_bin, _ = np.polyfit(mu_sel, y, deg=1)
            if np.isfinite(a_bin):
                slopes.append(a_bin)
        except Exception:
            continue

    slopes = np.asarray(slopes, dtype=float)

    if slopes.size < 3:
        return {
            "T": T,
            "beta": beta,
            "U": U,
            "mu_at_n": mu_at_n,
            "kappa": np.nan,
            "kappa_err": np.nan,
            "n_mu_points": len(data),
            "status": "too_few_finite_slopes",
        }

    # Final jackknife error analysis through util.jackknife.
    # With unit weights this is the ordinary jackknife over per-bin slope estimates.
    unit_weight = np.ones_like(slopes)
    kappa, kappa_err = util.jackknife(unit_weight, slopes)

    return {
        "T": T,
        "beta": beta,
        "U": U,
        "mu_at_n": mu_at_n,
        "kappa": kappa,
        "kappa_err": kappa_err,
        "n_mu_points": len(data),
        "status": "ok",
    }


def filling_tag(filling: float):
    return f"n{filling:.6f}".replace(".", "p").replace("-", "m")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute charge compressibility kappa_c = dn/dmu at a target filling "
            "from DQMC n(mu) data under T*_beta*_U*/mu*/ directories."
        )
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Directory containing T*_beta*_U*/mu*/ subdirectories.",
    )
    parser.add_argument(
        "--filling",
        required=True,
        type=float,
        help="Target per-site filling n, e.g. 0.7, 0.8, 0.9, 1.0.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of local mu points used for the linear slope fit. Default: 5.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum number of mu points required for the local fit. Default: 3.",
    )
    parser.add_argument(
        "--range-tol",
        type=float,
        default=1e-3,
        help=(
            "Tolerance for allowing target filling near the edge of simulated n range. "
            "Useful for n=1.0 when max(n) is slightly below 1 due to noise. Default: 1e-3."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Output prefix. Default: charge_compressibility_<filling_tag> "
            "saved under --path."
        ),
    )
    args = parser.parse_args()

    base = Path(args.path).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Path not found or not a directory: {base}")

    t_dirs = sorted([p for p in base.glob("T*_beta*_U*/") if p.is_dir()])
    if not t_dirs:
        raise FileNotFoundError(f"No T*_beta*_U*/ directories found under {base}")

    tag = filling_tag(args.filling)
    if args.output_prefix is None:
        out_prefix = base / f"charge_compressibility_{tag}"
    else:
        out_prefix = Path(args.output_prefix).expanduser()
        if not out_prefix.is_absolute():
            out_prefix = base / out_prefix

    results = []
    for t_dir in t_dirs:
        res = compute_kappa_for_T_dir(
            t_dir=t_dir,
            filling=args.filling,
            window=args.window,
            min_points=args.min_points,
            range_tol=args.range_tol,
        )
        results.append(res)

        print(
            f"[{res['status']}] "
            f"T={res['T']:.12g} beta={res['beta']:.12g} U={res['U']:.12g} "
            f"mu_at_n={res['mu_at_n']:.12g} "
            f"kappa={res['kappa']:.12g} err={res['kappa_err']:.12g}"
        )

    # Sort by T
    results.sort(key=lambda r: r["T"] if np.isfinite(r["T"]) else np.inf)

    T_arr = np.array([r["T"] for r in results], dtype=float)
    beta_arr = np.array([r["beta"] for r in results], dtype=float)
    U_arr = np.array([r["U"] for r in results], dtype=float)
    mu_at_n_arr = np.array([r["mu_at_n"] for r in results], dtype=float)
    kappa_arr = np.array([r["kappa"] for r in results], dtype=float)
    kappa_err_arr = np.array([r["kappa_err"] for r in results], dtype=float)
    n_mu_points_arr = np.array([r["n_mu_points"] for r in results], dtype=int)
    status_arr = np.array([r["status"] for r in results], dtype=object)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(out_prefix) + "_T.npy", T_arr)
    np.save(str(out_prefix) + "_beta.npy", beta_arr)
    np.save(str(out_prefix) + "_U.npy", U_arr)
    np.save(str(out_prefix) + "_mu_at_n.npy", mu_at_n_arr)
    np.save(str(out_prefix) + "_kappa.npy", kappa_arr)
    np.save(str(out_prefix) + "_kappa_err.npy", kappa_err_arr)
    np.save(str(out_prefix) + "_n_mu_points.npy", n_mu_points_arr)
    np.save(str(out_prefix) + "_status.npy", status_arr)

    print()
    print(f"[OK] saved arrays with prefix:")
    print(f"  {out_prefix}")
    print()
    print("Saved:")
    print(f"  {out_prefix}_T.npy")
    print(f"  {out_prefix}_beta.npy")
    print(f"  {out_prefix}_U.npy")
    print(f"  {out_prefix}_mu_at_n.npy")
    print(f"  {out_prefix}_kappa.npy")
    print(f"  {out_prefix}_kappa_err.npy")
    print(f"  {out_prefix}_n_mu_points.npy")
    print(f"  {out_prefix}_status.npy")


if __name__ == "__main__":
    main()
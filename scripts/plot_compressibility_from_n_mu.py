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
import paired_bootstrap


def _validate_jackknife_denominators(sign):
    sign = np.asarray(sign)
    total = np.sum(sign)
    total_abs = np.sum(np.abs(sign))
    rtol = paired_bootstrap.DEFAULT_DENOMINATOR_RTOL
    if np.abs(total) <= rtol * total_abs:
        raise ValueError("total accumulated sign is too close to zero")
    leave_one_out = total - sign
    leave_one_out_abs = total_abs - np.abs(sign)
    if np.any(np.abs(leave_one_out) <= rtol * leave_one_out_abs):
        bad = np.flatnonzero(
            np.abs(leave_one_out) <= rtol * leave_one_out_abs
        )
        raise ValueError(
            "jackknife leave-one-out accumulated sign is too close to zero "
            f"when omitting bin(s) {bad[:10].tolist()}"
        )


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
        density_numerator: raw signed per-site density accumulator,
            shape (n_bins_valid, 1)
        sign
        n_sample
        nsite

    The mean/error for n(mu) is computed from raw accumulators through
    util.jackknife_noniid(n_sample, sign, density_numerator).
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
        & np.isfinite(n_sample)
        & (n_sample > 0)
        & np.isfinite(sign)
        & np.isfinite(dsum)
    )

    n_sample = n_sample[valid]
    sign = sign[valid]
    dsum = dsum[valid]

    mu = get_mu_from_firstfile(mu_dir)

    if sign.size < 3:
        return {
            "mu": mu,
            "n_mean": np.nan,
            "n_err": np.nan,
            "density_numerator": np.empty((0, 1), dtype=float),
            "sign": np.array([], dtype=float),
            "n_sample": np.array([], dtype=float),
            "nsite": nsite,
            "mu_dir": mu_dir,
        }

    density_numerator = (dsum / nsite)[:, None]

    # Completed bins have equal n_sample, but retain it in the estimator so
    # the raw paired-bin contract remains explicit.
    _validate_jackknife_denominators(sign)
    n_mean, n_err = util.jackknife_noniid(
        n_sample,
        sign,
        density_numerator,
    )

    return {
        "mu": mu,
        "n_mean": float(np.asarray(n_mean).reshape(-1)[0]),
        "n_err": float(np.asarray(n_err).reshape(-1)[0]),
        "density_numerator": density_numerator,
        "sign": sign,
        "n_sample": n_sample,
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


def compute_kappa_for_T_dir(
    t_dir: Path,
    filling: float,
    window: int,
    min_points: int,
    range_tol: float,
    nboot: int,
    seed: int,
    bootstrap_block_size: int,
):
    """
    For one T*_beta*_U*/ directory:
      1. load all mu*/ density bins;
      2. build n(mu);
      3. choose local window around target filling;
      4. compute dn/dmu by local linear fit;
      5. independently paired-bootstrap each mu dataset and fit every
         resampled n(mu) curve.
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
            if (
                np.isfinite(item["mu"])
                and np.isfinite(item["n_mean"])
                and item["density_numerator"].shape[0] >= 3
            ):
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

    # Each mu directory is an independent simulation.  Bootstrap its paired
    # (density numerator, sign, n_sample) bins independently; do not align bin
    # indices across different mu values.
    selected_data = [data[i] for i in idx]
    rng = np.random.default_rng(seed)
    n_boot = np.empty((nboot, len(selected_data)), dtype=float)
    for imu, item in enumerate(selected_data):
        nbin = item["density_numerator"].shape[0]
        indices = paired_bootstrap.bootstrap_indices(
            nbin,
            nboot,
            block_size=bootstrap_block_size,
            rng=rng,
        )
        estimates = paired_bootstrap.bootstrap_ratio_of_sums(
            item["density_numerator"],
            item["sign"],
            indices,
        )
        n_boot[:, imu] = np.asarray(estimates).reshape(nboot)

    mu_centered = mu_sel - np.mean(mu_sel)
    slope_denominator = np.sum(mu_centered ** 2)
    slopes = (n_boot @ mu_centered) / slope_denominator
    finite_slopes = slopes[np.isfinite(slopes)]

    if finite_slopes.size < 2:
        return {
            "T": T,
            "beta": beta,
            "U": U,
            "mu_at_n": mu_at_n,
            "kappa": np.nan,
            "kappa_err": np.nan,
            "kappa_p16": np.nan,
            "kappa_p84": np.nan,
            "n_mu_points": len(data),
            "status": "too_few_finite_bootstrap_slopes",
        }

    kappa = float(a_mean)
    kappa_err = float(np.std(finite_slopes, ddof=1))
    kappa_p16, kappa_p84 = np.percentile(finite_slopes, [16.0, 84.0])

    return {
        "T": T,
        "beta": beta,
        "U": U,
        "mu_at_n": mu_at_n,
        "kappa": kappa,
        "kappa_err": kappa_err,
        "kappa_p16": float(kappa_p16),
        "kappa_p84": float(kappa_p84),
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
        "--nboot",
        type=int,
        default=1000,
        help="Number of independent paired-bootstrap slope samples. Default: 1000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for paired bootstrap. Default: 12345.",
    )
    parser.add_argument(
        "--bootstrap-block-size",
        type=int,
        default=1,
        help="Circular paired-bootstrap block size within each mu dataset. Default: 1.",
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

    if args.nboot < 2:
        raise ValueError("--nboot must be at least 2")
    if args.bootstrap_block_size < 1:
        raise ValueError("--bootstrap-block-size must be at least 1")

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
            nboot=args.nboot,
            seed=args.seed + len(results),
            bootstrap_block_size=args.bootstrap_block_size,
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
    kappa_p16_arr = np.array(
        [r.get("kappa_p16", np.nan) for r in results],
        dtype=float,
    )
    kappa_p84_arr = np.array(
        [r.get("kappa_p84", np.nan) for r in results],
        dtype=float,
    )
    n_mu_points_arr = np.array([r["n_mu_points"] for r in results], dtype=int)
    status_arr = np.array([r["status"] for r in results], dtype=object)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(out_prefix) + "_T.npy", T_arr)
    np.save(str(out_prefix) + "_beta.npy", beta_arr)
    np.save(str(out_prefix) + "_U.npy", U_arr)
    np.save(str(out_prefix) + "_mu_at_n.npy", mu_at_n_arr)
    np.save(str(out_prefix) + "_kappa.npy", kappa_arr)
    np.save(str(out_prefix) + "_kappa_err.npy", kappa_err_arr)
    np.save(str(out_prefix) + "_kappa_p16.npy", kappa_p16_arr)
    np.save(str(out_prefix) + "_kappa_p84.npy", kappa_p84_arr)
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
    print(f"  {out_prefix}_kappa_p16.npy")
    print(f"  {out_prefix}_kappa_p84.npy")
    print(f"  {out_prefix}_n_mu_points.npy")
    print(f"  {out_prefix}_status.npy")


if __name__ == "__main__":
    main()

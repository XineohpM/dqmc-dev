#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import glob
from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))
import util
import paired_bootstrap


def ensure_trailing_slash(path):
    path = str(path)
    return path if path.endswith("/") else path + "/"


def get_basic_meas(path: str):
    path = ensure_trailing_slash(path)
    n_sample, sign, beta, density, double_occ, g00 = util.load(
        path,
        "meas_eqlt/n_sample",
        "meas_eqlt/sign",
        "metadata/beta",
        "meas_eqlt/density",
        "meas_eqlt/double_occ",
        "meas_eqlt/g00",
    )

    out = {
        "sign_phase_raw_per_sample": dataset_to_bin_series(
            sign,
            n_sample=n_sample,
        ),
        "density_raw_per_sample": dataset_to_bin_series(
            density,
            n_sample=n_sample,
        ),
        "double_occ_raw_per_sample": dataset_to_bin_series(
            double_occ,
            n_sample=n_sample,
        ),
        "g00_raw_per_sample": dataset_to_bin_series(
            g00,
            n_sample=n_sample,
        ),
    }

    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    if beta_arr.size > 0:
        out["beta"] = float(beta_arr[0])
    else:
        out["beta"] = np.nan

    return out

def safe_name(s):
    return s.strip("/").replace("/", "__").replace(" ", "_")

def estimate_tau_int(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan, np.nan

    x = x - np.mean(x)
    var = np.dot(x, x) / n
    if var <= 0:
        return 0.5, n

    acf = np.correlate(x, x, mode="full")[n-1:] / (var * np.arange(n, 0, -1))

    # Initial positive sequence cutoff
    cutoff = 1
    for t in range(1, n):
        if acf[t] <= 0:
            cutoff = t
            break
    else:
        cutoff = n

    tau_int = 0.5 + np.sum(acf[1:cutoff])
    tau_int = max(tau_int, 0.5)
    neff = n / (2.0 * tau_int)
    return tau_int, neff

def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))

def dataset_to_bin_series(arr, n_sample=None):
    arr = np.asarray(arr)

    if arr.size == 0 or arr.ndim == 0:
        return None

    arr = np.real(arr)
    if arr.shape[0] < 4:
        return None

    if arr.ndim == 1:
        y = arr.astype(float)
    else:
        flat = arr.reshape(arr.shape[0], -1).astype(float)
        finite_count = np.count_nonzero(np.isfinite(flat), axis=1)
        y = np.full(arr.shape[0], np.nan, dtype=float)
        np.divide(
            np.nansum(flat, axis=1),
            finite_count,
            out=y,
            where=finite_count > 0,
        )

    if n_sample is not None:
        n_sample = np.asarray(n_sample, dtype=float).reshape(-1)
        if n_sample.shape != y.shape:
            raise ValueError(
                f"n_sample shape {n_sample.shape} does not match "
                f"bin series shape {y.shape}"
            )
        if not np.all(np.isfinite(n_sample)) or np.any(n_sample <= 0):
            raise ValueError("n_sample must be finite and positive")
        y = y / n_sample

    if len(y) < 4:
        return None
    return y

def analyze_series(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 4:
        return None

    n1 = n // 2
    first = y[:n1]
    second = y[n1:]

    m_all = np.mean(y)
    e_all = sem(y)
    m_first = np.mean(first)
    m_second = np.mean(second)
    e_first = sem(first)
    e_second = sem(second)

    denom = np.sqrt(e_first**2 + e_second**2)
    z_halves = abs(m_first - m_second) / denom if denom > 0 else np.nan

    x = np.arange(n, dtype=float)
    x0 = x - x.mean()
    y0 = y - y.mean()
    slope = np.dot(x0, y0) / np.dot(x0, x0)
    residual = y - (y.mean() + slope * x0)
    if n > 2:
        s2 = np.sum(residual**2) / (n - 2)
        slope_err = np.sqrt(s2 / np.dot(x0, x0))
        slope_z = abs(slope / slope_err) if slope_err > 0 else np.nan
    else:
        slope_z = np.nan

    tau_int, neff = estimate_tau_int(y)

    means_after_drop = {}
    for frac in [0.0, 0.1, 0.2, 0.3, 0.5]:
        k = int(round(frac * n))
        if k < n:
            means_after_drop[frac] = np.mean(y[k:])
        else:
            means_after_drop[frac] = np.nan

    return {
        "n": n,
        "mean": m_all,
        "sem_naive": e_all,
        "first_mean": m_first,
        "second_mean": m_second,
        "first_sem": e_first,
        "second_sem": e_second,
        "z_halves": z_halves,
        "slope": slope,
        "slope_z": slope_z,
        "tau_int_bins": tau_int,
        "neff": neff,
        **{f"mean_drop_{int(frac*100)}pct": val for frac, val in means_after_drop.items()},
    }

def plot_series(y, title, png_path):
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.arange(n)

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", markersize=2, linewidth=1)
    plt.xlabel("bin index")
    plt.ylabel("diagnostic value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

def plot_running_mean(y, title, png_path):
    y = np.asarray(y, dtype=float)
    n = len(y)

    ks = np.arange(n)
    rm = np.array([np.mean(y[k:]) for k in ks])

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(ks, rm, linewidth=1.5)
    plt.xlabel("number of initial measured bins discarded")
    plt.ylabel("mean diagnostic value after discard")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def load_optional_h5_obs(path, obs_name):
    path = ensure_trailing_slash(path)
    try:
        n_sample, arr = util.load(
            path,
            "meas_eqlt/n_sample",
            f"meas_eqlt/{obs_name}",
        )
    except Exception as e:
        print(f"[WARN] failed to read meas_eqlt/{obs_name} under {path}: {e}")
        return None
    return dataset_to_bin_series(arr, n_sample=n_sample)


def prefix_ratio_of_sums_series(numerator, sign):
    numerator = np.asarray(numerator)
    sign = np.asarray(sign)
    if numerator.ndim != 2:
        raise ValueError(
            f"numerator must have shape (Nbin, L), got {numerator.shape}"
        )
    if sign.shape != (numerator.shape[0],):
        raise ValueError(
            f"sign shape {sign.shape} does not match Nbin={numerator.shape[0]}"
        )

    cumulative_numerator = np.cumsum(numerator, axis=0)
    cumulative_sign = np.cumsum(sign)
    cumulative_absolute_sign = np.cumsum(np.abs(sign))
    bad = np.abs(cumulative_sign) <= (
        paired_bootstrap.DEFAULT_DENOMINATOR_RTOL
        * cumulative_absolute_sign
    )

    ratios = np.full(
        cumulative_numerator.shape,
        np.nan,
        dtype=np.result_type(numerator.dtype, sign.dtype, np.float64),
    )
    good = ~bad
    ratios[good] = (
        cumulative_numerator[good].T / cumulative_sign[good]
    ).T
    return dataset_to_bin_series(ratios), np.flatnonzero(bad)


def load_optional_paired_diagnostics(path, filename):
    bundle_path = Path(path) / filename
    if not bundle_path.exists():
        print(f"[WARN] missing {bundle_path}")
        return {}
    try:
        bundle = paired_bootstrap.load_paired_bundle(bundle_path)
    except Exception as e:
        print(f"[WARN] failed to read {bundle_path}: {e}")
        return {}

    stem = bundle_path.stem
    prefix_ratio, bad_prefixes = prefix_ratio_of_sums_series(
        bundle.numerator,
        bundle.sign,
    )
    if bad_prefixes.size:
        print(
            f"[WARN] {bundle_path}: prefix accumulated sign/phase is too "
            f"close to zero at bin index/indices "
            f"{bad_prefixes[:10].tolist()}; recording NaN for those prefixes"
        )

    return {
        f"{stem}__raw_numerator_per_sample": dataset_to_bin_series(
            bundle.numerator,
            n_sample=bundle.n_sample,
        ),
        f"{stem}__sign_phase_per_sample": dataset_to_bin_series(
            bundle.sign,
            n_sample=bundle.n_sample,
        ),
        f"{stem}__prefix_ratio_of_sums": prefix_ratio,
    }


def analyze_and_plot_observable(path, rel, obs_name, y, file_out, summary_rows):
    if y is None:
        return

    stats = analyze_series(y)
    if stats is None:
        return

    obs_tag = safe_name(obs_name)
    plot_series(
        y,
        f"{rel} | {obs_name}",
        file_out / f"{obs_tag}_timeseries.png",
    )
    plot_running_mean(
        y,
        f"{rel} | {obs_name}",
        file_out / f"{obs_tag}_running_mean.png",
    )

    row = {
        "path": str(rel),
        "obs": obs_name,
        **stats,
    }
    summary_rows.append(row)

    print(
        f"{rel} {obs_name}: "
        f"N={stats['n']} "
        f"mean={stats['mean']:.8g} "
        f"z_halves={stats['z_halves']:.3g} "
        f"slope_z={stats['slope_z']:.3g} "
        f"tau_int={stats['tau_int_bins']:.3g} "
        f"Neff={stats['neff']:.3g}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=str, help="Root directory containing DQMC run subdirectories.")
    p.add_argument(
        "--glob",
        type=str,
        required=True,
        help="Subdirectory glob relative to root. Use '.' to process root itself.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="warmup_check",
        help="Directory where plots and warmup_summary.tsv will be written.",
    )
    p.add_argument("--kk", action="store_true", help="Also check meas_eqlt/kk.")
    p.add_argument("--kv", action="store_true", help="Also check meas_eqlt/kv.")
    p.add_argument("--vv", action="store_true", help="Also check meas_eqlt/vv.")
    p.add_argument("--nn", action="store_true", help="Also check meas_eqlt/nn.")
    p.add_argument(
        "--JNJN",
        action="store_true",
        help="Also diagnose JNJN_xx_paired.npz in each matched subdirectory.",
    )
    p.add_argument(
        "--g1p",
        action="store_true",
        help=("Also diagnose 1_particle_local_g_paired.npz in each matched "
              "subdirectory."),
    )
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.glob == ".":
        run_paths = [root]
    else:
        run_paths = sorted(p for p in root.glob(args.glob) if p.is_dir())

    if not run_paths:
        raise SystemExit(f"No directories matched root={root} glob={args.glob}")

    summary_rows = []
    print(f"Found {len(run_paths)} matched run directories under {root}")

    optional_h5_obs = []
    for name, enabled in [
        ("kk", args.kk),
        ("kv", args.kv),
        ("vv", args.vv),
        ("nn", args.nn),
    ]:
        if enabled:
            optional_h5_obs.append(name)

    for path in run_paths:
        rel = path.relative_to(root) if path != root else Path(".")
        file_tag = safe_name(str(rel)) if str(rel) != "." else "root"
        file_out = outdir / file_tag
        file_out.mkdir(parents=True, exist_ok=True)

        try:
            basic = get_basic_meas(path)
        except Exception as e:
            print(f"[WARN] failed to read basic measurements under {path}: {e}")
            basic = {}

        for obs_name in [
            "sign_phase_raw_per_sample",
            "density_raw_per_sample",
            "double_occ_raw_per_sample",
            "g00_raw_per_sample",
        ]:
            analyze_and_plot_observable(
                path,
                rel,
                obs_name,
                basic.get(obs_name),
                file_out,
                summary_rows,
            )

        for obs_name in optional_h5_obs:
            y = load_optional_h5_obs(path, obs_name)
            analyze_and_plot_observable(
                path,
                rel,
                f"{obs_name}_raw_per_sample",
                y,
                file_out,
                summary_rows,
            )

        if args.JNJN:
            diagnostics = load_optional_paired_diagnostics(
                path,
                "JNJN_xx_paired.npz",
            )
            for obs_name, y in diagnostics.items():
                analyze_and_plot_observable(
                    path,
                    rel,
                    obs_name,
                    y,
                    file_out,
                    summary_rows,
                )

        if args.g1p:
            diagnostics = load_optional_paired_diagnostics(
                path,
                "1_particle_local_g_paired.npz",
            )
            for obs_name, y in diagnostics.items():
                analyze_and_plot_observable(
                    path,
                    rel,
                    obs_name,
                    y,
                    file_out,
                    summary_rows,
                )

    if not summary_rows:
        raise SystemExit("No usable per-bin observables found.")

    keys = list(summary_rows[0].keys())
    tsv = outdir / "warmup_summary.tsv"
    with open(tsv, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for row in summary_rows:
            fh.write("\t".join(str(row.get(k, "")) for k in keys) + "\n")

    print()
    print(f"Summary written to: {tsv}")
    print(f"Plots written under: {outdir}")
    print()
    print("Interpretation guide:")
    print("  *_raw_per_sample: raw signed accumulator divided by n_sample;")
    print("                    it is a warmup diagnostic, not a physical mean")
    print("  *__prefix_ratio_of_sums: cumulative physical estimator with")
    print("                           near-zero sign/phase prefixes set to NaN")
    print("  z_halves > 2     : first half and second half differ noticeably")
    print("  slope_z > 2      : visible monotonic drift is likely")
    print("  Neff small       : autocorrelation is strong; error bars may be too optimistic")
    print("  running_mean plot: should become insensitive to discarding early measured bins")


if __name__ == "__main__":
    main()

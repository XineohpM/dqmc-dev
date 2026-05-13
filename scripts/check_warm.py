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


def ensure_trailing_slash(path):
    path = str(path)
    return path if path.endswith("/") else path + "/"


def get_basic_meas(path: str):
    path = ensure_trailing_slash(path)
    sign, beta, density, double_occ, g00 = util.load(
        path,
        "meas_eqlt/sign",
        "metadata/beta",
        "meas_eqlt/density",
        "meas_eqlt/double_occ",
        "meas_eqlt/g00",
    )

    out = {
        "sign": dataset_to_bin_series(sign),
        "density": dataset_to_bin_series(density),
        "double_occ": dataset_to_bin_series(double_occ),
        "g00": dataset_to_bin_series(g00),
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

def dataset_to_bin_series(arr, nbin=None):
    arr = np.asarray(arr)

    if arr.size == 0 or arr.ndim == 0:
        return None

    arr = np.real(arr)
    if arr.shape[0] < 4:
        return None

    if arr.ndim == 1:
        y = arr.astype(float)
    else:
        y = np.nanmean(arr.reshape(arr.shape[0], -1), axis=1).astype(float)

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
    plt.ylabel("per-bin value")
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
    plt.ylabel("mean after discard")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def load_optional_h5_obs(path, obs_name):
    path = ensure_trailing_slash(path)
    try:
        arr, = util.load(path, f"meas_eqlt/{obs_name}")
    except Exception as e:
        print(f"[WARN] failed to read meas_eqlt/{obs_name} under {path}: {e}")
        return None
    return dataset_to_bin_series(arr)


def load_optional_npy_series(path, filename):
    path = Path(path)
    npy_path = path / filename
    if not npy_path.exists():
        print(f"[WARN] missing {npy_path}")
        return None
    try:
        arr = np.load(npy_path)
    except Exception as e:
        print(f"[WARN] failed to read {npy_path}: {e}")
        return None
    return dataset_to_bin_series(arr)


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
        help="Also check JNJN_xx_perbin.npy in each matched subdirectory.",
    )
    p.add_argument(
        "--g1p",
        action="store_true",
        help="Also check 1_particle_local_g_all.npy in each matched subdirectory.",
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
            continue

        for obs_name in ["sign", "density", "double_occ", "g00"]:
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
                obs_name,
                y,
                file_out,
                summary_rows,
            )

        if args.JNJN:
            y = load_optional_npy_series(path, "JNJN_xx_perbin.npy")
            analyze_and_plot_observable(
                path,
                rel,
                "JNJN_xx_perbin",
                y,
                file_out,
                summary_rows,
            )

        if args.g1p:
            y = load_optional_npy_series(path, "1_particle_local_g_all.npy")
            analyze_and_plot_observable(
                path,
                rel,
                "1_particle_local_g_all",
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
    print("  z_halves > 2     : first half and second half differ noticeably")
    print("  slope_z > 2      : visible monotonic drift is likely")
    print("  Neff small       : autocorrelation is strong; error bars may be too optimistic")
    print("  running_mean plot: should become insensitive to discarding early measured bins")


if __name__ == "__main__":
    main()
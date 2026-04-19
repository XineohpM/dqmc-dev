#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_best_mu_vs_T.py

Read one or more "best mu" log files produced by get_mu.sh/get_mu.py and plot mu*(T).

Outputs (saved in current working directory):
  - <out_prefix>_overlay.png                 (all curves overlaid)
  - <out_prefix>_<stem_of_file>.png          (one per input file)

Usage:
  python3 plot_best_mu_vs_T.py /path/to/n0.6_best_mu_*.txt /path/to/n0.8_best_mu_*.txt
  python3 plot_best_mu_vs_T.py --out_prefix muT --smooth cubic file1.txt file2.txt

Smoothing:
  --smooth none|linear|cubic|spline
  If SciPy is unavailable, cubic/spline fall back to linear interpolation.

Notes on parsing:
  - Expects blocks like:
        [[...]]
        [-0.2263074]
        /scratch/.../T0.222222_beta4.5_U-6
    "incomplete" warnings are ignored.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# Optional SciPy for nicer smoothing
try:
    from scipy.interpolate import interp1d, UnivariateSpline
except Exception:
    interp1d = None
    UnivariateSpline = None


# -------- regex patterns --------
# directory line contains /T{T}_beta{beta}_U{U}
_T_DIR_RE = re.compile(r"/T([0-9eE+\-\.]+)_beta([0-9eE+\-\.]+)_U([0-9eE+\-\.]+)")
# mu_star line printed by get_mu.py: [ -0.2263 ]
_MU_LINE_RE = re.compile(r"^\s*\[\s*([-+0-9eE\.]+)\s*\]\s*$")

# header lines
_GEOM_RE = re.compile(r"^\s*geometry\s*=\s*(\S+)\s*$", re.IGNORECASE)
_TARGET_RE = re.compile(r"target\s*n\s*=\s*([-+0-9eE\.]+)", re.IGNORECASE)
_TP_RE = re.compile(r"^\s*tp\s*=\s*([-+0-9eE\.]+)\s*$", re.IGNORECASE)

# infer from path/name if missing
_NFLUX_IN_NAME_RE = re.compile(r"_nflux([-+0-9eE\.]+)")
_TP_IN_NAME_RE = re.compile(r"_tp([-+0-9eE\.]+)")
_NDIR_IN_PATH_RE = re.compile(r"/n([0-9]+(?:\.[0-9]+)?)/")


@dataclass
class SeriesMeta:
    filepath: Path
    geometry: Optional[str] = None
    target_n: Optional[float] = None
    tp: Optional[float] = None
    nflux: Optional[float] = None


@dataclass
class MuTSeries:
    meta: SeriesMeta
    T: np.ndarray
    mu: np.ndarray


def infer_from_path(p: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    s = str(p)
    nflux = None
    tp = None
    target_n = None

    m = _NFLUX_IN_NAME_RE.search(s)
    if m:
        try:
            nflux = float(m.group(1))
        except ValueError:
            pass

    m = _TP_IN_NAME_RE.search(s)
    if m:
        try:
            tp = float(m.group(1))
        except ValueError:
            pass

    m = _NDIR_IN_PATH_RE.search(s)
    if m:
        try:
            target_n = float(m.group(1))
        except ValueError:
            pass

    return target_n, tp, nflux


def parse_best_mu_file(path: Path) -> MuTSeries:
    meta = SeriesMeta(filepath=path)

    # fallback inference
    n_guess, tp_guess, nflux_guess = infer_from_path(path)
    meta.target_n = n_guess
    meta.tp = tp_guess
    meta.nflux = nflux_guess

    Ts: List[float] = []
    mus: List[float] = []

    last_mu_star: Optional[float] = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()

            # header
            if meta.geometry is None:
                mg = _GEOM_RE.match(s)
                if mg:
                    meta.geometry = mg.group(1)

            mt = _TARGET_RE.search(s)
            if mt:
                try:
                    meta.target_n = float(mt.group(1))
                except ValueError:
                    pass

            mp = _TP_RE.match(s)
            if mp:
                try:
                    meta.tp = float(mp.group(1))
                except ValueError:
                    pass

            # mu* line
            mmu = _MU_LINE_RE.match(s)
            if mmu:
                try:
                    last_mu_star = float(mmu.group(1))
                except ValueError:
                    last_mu_star = None
                continue

            # temperature directory line
            mdir = _T_DIR_RE.search(s)
            if mdir and last_mu_star is not None:
                try:
                    Tval = float(mdir.group(1))
                except ValueError:
                    last_mu_star = None
                    continue
                Ts.append(Tval)
                mus.append(last_mu_star)
                last_mu_star = None

    if not Ts:
        raise RuntimeError(f"No (T, mu*) pairs found in file: {path}")

    T = np.array(Ts, dtype=float)
    mu = np.array(mus, dtype=float)

    # sort by T
    order = np.argsort(T)
    T = T[order]
    mu = mu[order]

    return MuTSeries(meta=meta, T=T, mu=mu)


def make_label(meta: SeriesMeta) -> str:
    n_str = "?" if meta.target_n is None else f"{meta.target_n:g}"
    nflux_str = "?" if meta.nflux is None else f"{meta.nflux:g}"
    tp_str = "?" if meta.tp is None else f"{meta.tp:g}"
    return rf"$\langle n \rangle={n_str}$, $n_{{\rm flux}}={nflux_str}$, $t'={tp_str}$"


def smooth_curve(T: np.ndarray, mu: np.ndarray, smooth: str, n_grid: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (T_grid, mu_grid) for a smooth connecting curve."""
    if T.size < 2:
        return T, mu

    T_grid = np.linspace(T.min(), T.max(), n_grid)
    smooth = smooth.lower()

    if smooth in ("none", "linear"):
        mu_grid = np.interp(T_grid, T, mu)
        return T_grid, mu_grid

    # cubic interpolation (needs SciPy)
    if smooth == "cubic" and interp1d is not None:
        if T.size >= 4:
            f = interp1d(T, mu, kind="cubic", fill_value="extrapolate")
            return T_grid, f(T_grid)
        # fallback
        return T_grid, np.interp(T_grid, T, mu)

    # spline (needs SciPy)
    if smooth == "spline" and UnivariateSpline is not None:
        if T.size >= 4:
            spl = UnivariateSpline(T, mu, s=0)
            return T_grid, spl(T_grid)
        return T_grid, np.interp(T_grid, T, mu)

    # fallback
    return T_grid, np.interp(T_grid, T, mu)


def plot_one(series: MuTSeries, out_png: Path, smooth: str, n_grid: int) -> None:
    plt.figure()
    (pts_line,) = plt.plot(series.T, series.mu, marker="o", linestyle="None", label="data")
    color = pts_line.get_color()
    Tg, mug = smooth_curve(series.T, series.mu, smooth=smooth, n_grid=n_grid)
    plt.plot(Tg, mug, linestyle="-", color=color, label=f"{smooth} curve" if smooth != "none" else "connect")

    plt.xlabel("T")
    plt.ylabel(r"$\mu^*$")
    plt.title(make_label(series.meta))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_overlay(series_list: List[MuTSeries], out_png: Path, smooth: str, n_grid: int) -> None:
    plt.figure()
    for s in series_list:
        lbl = make_label(s.meta)
        (pts_line,) = plt.plot(s.T, s.mu, marker="o", linestyle="None", label=lbl)
        color = pts_line.get_color()
        Tg, mug = smooth_curve(s.T, s.mu, smooth=smooth, n_grid=n_grid)
        plt.plot(Tg, mug, linestyle="-", color=color)
    plt.xlabel("T")
    plt.ylabel(r"$\mu^*$")
    #plt.title(r"Best $\mu^*(T)$ (overlay)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot mu*(T) from one or more best-mu log files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("files", nargs="+", help="One or more best_mu*.txt files.")
    ap.add_argument("--out_prefix", default="muT", help="Prefix for output PNG files.")
    ap.add_argument("--smooth", choices=["none", "linear", "cubic", "spline"], default="cubic",
                    help="Smoothing method for connecting curve.")
    ap.add_argument("--n_grid", type=int, default=300, help="Number of grid points for smooth curve.")
    return ap.parse_args()


def main() -> None:
    args = parse_cli()
    paths = [Path(p).expanduser().resolve() for p in args.files]

    series_list: List[MuTSeries] = [parse_best_mu_file(p) for p in paths]

    # overlay
    overlay_png = Path(f"{args.out_prefix}_overlay.png")
    plot_overlay(series_list, overlay_png, smooth=args.smooth, n_grid=args.n_grid)

    # individual
    for s in series_list:
        out_png = Path(f"{args.out_prefix}_{s.meta.filepath.stem}.png")
        plot_one(s, out_png, smooth=args.smooth, n_grid=args.n_grid)

    print(f"Wrote {overlay_png}")
    for s in series_list:
        print(f"Wrote {args.out_prefix}_{s.meta.filepath.stem}.png")


if __name__ == "__main__":
    main()
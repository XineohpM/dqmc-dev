#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import h5py
except ImportError as e:
    raise SystemExit("Need h5py. Activate your dqmc conda env first.") from e


def n_sample_max(h5_path: Path):
    """Return (ok, ns_max). ok=False if dataset missing or unreadable."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "meas_eqlt/n_sample" not in f:
                return False, np.nan
            ns = f["meas_eqlt/n_sample"][...]
            # ns can be vector or matrix; take global max
            return True, float(np.nanmax(ns))
    except Exception:
        return False, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        help="Root directory to scan",
    )
    ap.add_argument("--out", default="h5_completion_report.tsv")
    ap.add_argument("--glob", default="n*/T*_beta*_U*/mu*/*.h5")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = sorted(root.glob(args.glob))

    if not files:
        raise SystemExit(f"No .h5 files found under {root} with glob {args.glob}")

    # First pass: compute n_sample.max per file, group by mu directory
    per_dir = defaultdict(list)  # mu_dir -> list of (file, ok, nsmax)
    for fp in files:
        ok, nsmax = n_sample_max(fp)
        per_dir[fp.parent].append((fp, ok, nsmax))

    # Second pass: for each mu_dir, define "dir_max" = max nsmax among readable files
    lines = []
    lines.append("mu_dir\tfile\tok_n_sample\tn_sample_max\tdir_max\tis_complete_in_dir\n")

    n_total = 0
    n_incomplete = 0
    n_bad = 0

    for mu_dir, items in sorted(per_dir.items(), key=lambda kv: str(kv[0])):
        # max over ok items
        ok_vals = [ns for _, ok, ns in items if ok and np.isfinite(ns)]
        dir_max = max(ok_vals) if ok_vals else np.nan

        for fp, ok, nsmax in items:
            n_total += 1
            if not ok:
                n_bad += 1
                is_complete = False
            else:
                # complete if nsmax equals dir_max (within tiny tolerance)
                is_complete = np.isfinite(dir_max) and abs(nsmax - dir_max) <= 1e-9
                if not is_complete:
                    n_incomplete += 1

            lines.append(
                f"{mu_dir}\t{fp}\t{int(ok)}\t{nsmax:.12g}\t{dir_max:.12g}\t{int(is_complete)}\n"
            )

    Path(args.out).write_text("".join(lines), encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"Total files: {n_total}")
    print(f"Missing/unreadable n_sample: {n_bad}")
    print(f"Incomplete (nsmax < dir_max within same mu_dir): {n_incomplete}")


if __name__ == "__main__":
    main()
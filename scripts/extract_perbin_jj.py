#!/usr/bin/env python3
"""
extract_perbin_jj.py
phoenixm@stanford.edu

Extract imaginary-time current-current correlator from DQMC HDF5 outputs.

In `jqjq.py`, `get_component(path, "jj")` returns q=0 correlator *NOT divided by sign*.
`get_sign(path)` returns accumulated sign SUM(phase) (not divided by n_sample).

These per-bin arrays are directly suitable for bootstrap/jackknife and MaxEnt/SAC workflows.

Usage examples:
  1) Electrical current correlator (JNJN) for xx component (default):
     python3 extract_perbin_jj.py \
       --path /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_conductivity/T_0.2/ \
       --output_type electric \
       --component xx

  2) Save all tensor components (xx, yy, xy, yx) of JNJN:
     python3 extract_perbin_jj.py \
       --path /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_conductivity/T_0.2/ \
       --output_type JNJN \
       --component all

  3) Thermal/heat-current correlators (outputs JQJQ, JQJN, JNJQ, and also JNJN) for xx:
     python3 extract_perbin_jj.py \
       --path /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_conductivity/T_0.2/ \
       --output_type thermal \
       --component xx

  4) Output everything (JNJN, JQJQ, JQJN, JNJQ) and all tensor components, with a filename prefix:
     python3 extract_perbin_jj.py \
       --path /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_conductivity/T_0.2/ \
       --output_type all \
       --component all \
       --prefix U-6_T0.2_

  5) Per-site normalization (requires Nx, Ny; Norb optional) and custom output directory:
     python3 extract_perbin_jj.py \
       --path /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_conductivity/T_0.2/ \
       --output_type electric \
       --component xx \
       --per_site --Nx 6 --Ny 6 --Norb 1 \
       --outdir /scratch/users/phoenixm/dqmc_runs/postproc/U-6_T0.2/

Notes:
  - The path must point to the directory that directly contains the *.h5 files.
  - If util.py uses glob(path + "*.h5"), ensure the directory path ends with a trailing '/'.
"""

from __future__ import annotations
import glob
import argparse
import os, sys
import numpy as np
from pathlib import Path
utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))
import jqjq


COMP_TO_IDX = {"xx": 0, "yy": 1, "xy": 2, "yx": 3}


def _ensure_outdir(outdir: str) -> None:
    """
    Ensure the output direction (outdir) exists. If not, create the output direction.
    
    Args:
        outdir (str): The output direction.
    """
    os.makedirs(outdir, exist_ok=True)


def _as_1d(a) -> np.ndarray:
    """
    Converts input a into an 1D array.
    """
    return np.asarray(a).reshape(-1)


def _load_sign_and_ns_completed(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load accumulated sign and n_sample for the *completed* bins.

    Mirrors jqjq.get_sign() logic so ordering matches jqjq.get_component().

    Returns:
      (sign_accum, n_sample) with shape (Nbin_completed,)
    """
    try:
        ns, s = jqjq.util.load(path, "meas_uneqlt/n_sample", "meas_uneqlt/sign")
    except Exception as e:
        raise RuntimeError(
            "Failed to load meas_uneqlt/n_sample and meas_uneqlt/sign via jqjq.util.load()."
        ) from e

    ns = _as_1d(ns)
    s = _as_1d(s)
    if ns.size != s.size:
        raise ValueError(f"n_sample and sign size mismatch: {ns.size} vs {s.size}")

    # Keep only completed bins (same mask as jqjq.get_sign).
    mask = ns == ns.max()
    return s[mask].astype(np.complex128 if np.iscomplexobj(s) else float), ns[mask].astype(float)


def _load_dt(path: str) -> float | None:
    """
    Try to load params/dt from the first available HDF5 file; return first value if available.

    Notes:
      - jqjq.util.load_firstfile returns a tuple even for a single key.
      - params/dt is expected to be a scalar or a 1-element array; we defensively handle both.
    """
    try:
        (dt_raw,) = jqjq.util.load_firstfile(path, "params/dt")
    except Exception:
        return None

    dt_arr = _as_1d(dt_raw)
    if dt_arr.size == 0:
        return None

    try:
        return float(dt_arr[0])
    except Exception:
        return None


def _reweight_by_sign_perbin(data: np.ndarray, sign_accum: np.ndarray) -> np.ndarray:
    """
    Apply correct bin-by-bin reweighting: Obin = (Σ sign*Obin) / (Σ sign).

    data: shape (Nbin, L)
    sign_accum: shape (Nbin,)

    Returns sign-reweighted data with same shape.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be (Nbin, L), got {data.shape}")
    Nbin = data.shape[0]
    if sign_accum.shape != (Nbin,):
        raise ValueError(f"sign_accum must be (Nbin,), got {sign_accum.shape}, expected ({Nbin},)")

    # Guard against division by zero (or nearly zero for complex phase-sum).
    bad = np.where(np.abs(sign_accum) == 0)[0] # contains indicies of bins with sign_accum == 0
    if bad.size != 0:
        raise ValueError(f"Found {bad.size} bins with zero accumulated sign; cannot reweight. First indices: {bad[:10]}")

    return data / sign_accum.reshape(Nbin, 1)


def _stats_over_bins(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, covariance, stderr over bins for each tau.

    data: shape (Nbin, L)
    Returns: (mean_tau (L,), cov_tau (L,L), stderr (L,))

    Note: For complex data, covariance will be complex; downstream MaxEnt usually
    expects real inputs. We will warn and proceed; users can take real/imag parts.
    """
    mean_tau = data.mean(axis=0)
    cov_tau = np.cov(data, rowvar=False)
    Nbin = data.shape[0]
    stderr = np.sqrt(np.diag(cov_tau) / max(1, Nbin))
    return mean_tau, cov_tau, stderr


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract current-current correlator per bin and basic stats for bootstrap/jackknife/MaxEnt."
    )
    p.add_argument("--path", required=True, help="Directory containing DQMC hdf5 files for one temperature/beta.")
    p.add_argument("--output_type", required=True, default="all",
                   choices=["JNJN", "JQJQ", "JNJQ", "JQJN", "electric", "thermal", "all"],
                   help="Output current-current correlator type, choose between \"JNJN\", \"JQJQ\", \"JNJQ\", \"JQJN\", \"electric\", \"thermal\", \"all\".")
    p.add_argument(
        "--component",
        default="xx",
        choices=["xx", "yy", "xy", "yx", "all"],
        help="Tensor component to save (default: xx).",
    )
    p.add_argument("--outdir", default=None, help="Output directory (default: same as --path).")
    p.add_argument("--prefix", default="", help="Prefix for output filenames (default: empty).")
    p.add_argument(
        "--per_site",
        action="store_true",
        help=(
            "Optional: divide correlator by Nsite (Nx*Ny*Norb). "
            "Default is OFF because many implementations already average over symmetry/volume."
        ),
    )
    p.add_argument("--Nx", type=int, default=None, help="Nx (required if --per_site).")
    p.add_argument("--Ny", type=int, default=None, help="Ny (required if --per_site).")
    p.add_argument("--Norb", type=int, default=1, help="Norb (default: 1).")

    args = p.parse_args()

    # Normalize paths.
    # NOTE: dqmc-dev/util/util.py currently uses glob(path + "*.h5"),
    # so `path` MUST end with a path separator to match files in a directory.
    path = os.path.expanduser(args.path)
    if os.path.isdir(path) and not path.endswith(os.sep):
        path += os.sep

    outdir = os.path.expanduser(args.outdir) if args.outdir else path
    _ensure_outdir(outdir)

    # Preflight: ensure there are .h5 files directly under `path`.
    h5_list = sorted(glob.glob(path + "*.h5"))
    if not h5_list:
        raise FileNotFoundError(
            f"No .h5 files found under: {path}\n"
            f"Tried pattern: {path + '*.h5'}\n"
            f"If your files are in a nested folder, pass the directory that directly contains the .h5 files."
        )

    # 1) Load jj and project to physical tensor components.
    jj_q0 = jqjq.get_component(path, "jj")
    jj_4 = np.asarray(jqjq.electrical_sum(path, jj_q0))  # (4, Nbin, L), np.stack((jj_xx, jj_yy, jj_xy, jj_yx), axis=0)

    if jj_4.ndim != 3 or jj_4.shape[0] != 4:
        raise ValueError(f"Unexpected jj_4 shape {jj_4.shape}; expected (4, Nbin, L)")

    _, Nbin, L = jj_4.shape

    if args.output_type == "JNJN" or args.output_type == "electric":  # returns JNJN only
        JNJN = (-1) * jj_4
    else:  # returns JNJN, JQJQ, JQJN, JNJQ
        names = ["j2j2", "jj2", "j2j", "jnj2", "j2jn", "jjn", "jnj", "jnjn", "jj"]
        q0_corrs = {}
        for name in names:
            try:
                q0_corrs[name] = jqjq.get_component(path, name)
            except KeyError as e:
                # Missing dataset required for thermal current correlator
                # Show which key is missing and one example file for debugging
                raise RuntimeError(f"Missing meas_uneqlt/{name} under {path}*.h5") from e
        if q0_corrs is None:
            raise RuntimeError(f"No files matching: {path}.*h5.")
        if len(q0_corrs) != len(names):
            raise RuntimeError(f"Unexpected util.load return length {len(q0_corrs)} for keys {names}.")
        
        q0_tuple = tuple(q0_corrs[name] for name in names)
        all_dict = jqjq.thermal_sum(path, q0_tuple)
        JNJN = all_dict["JNJN"]
        JQJQ = all_dict["JQJQ"]
        JQJN = all_dict["JQJN"]
        JNJQ = all_dict["JNJQ"]


    # 2) Load sign and n_sample for completed bins (must align with jj_4 bins).
    sign_accum, n_sample = _load_sign_and_ns_completed(path)

    if sign_accum.shape[0] != Nbin:
        raise ValueError(
            f"Bin count mismatch: jj has Nbin={Nbin} but sign has {sign_accum.shape[0]}. "
            "This usually indicates inconsistent masking or mixed/incomplete files in the directory."
        )

    mean_sign_perbin = sign_accum / n_sample

    # 3) Optional per-site normalization.
    if args.per_site:
        if args.Nx is None or args.Ny is None:
            raise ValueError("--per_site requires --Nx and --Ny")
        Nsite = int(args.Nx) * int(args.Ny) * int(args.Norb)
        if Nsite <= 0:
            raise ValueError("Invalid Nsite computed from Nx, Ny, Norb")
        jj_4 = jj_4 / float(Nsite)
        JNJN = JNJN / float(Nsite)
        if args.output_type != "JNJN" and args.output_type != "electric":
            JQJQ = JQJQ / float(Nsite)
            JQJN = JQJN / float(Nsite)
            JNJQ = JNJQ / float(Nsite)

    # 4) Correct per-bin sign reweighting.
    # Use the shared helper to guard against zero accumulated sign.
    JNJN_rw = np.stack([
        _reweight_by_sign_perbin(JNJN[a], sign_accum) for a in range(4)
    ], axis=0)
    if args.output_type != "JNJN" and args.output_type != "electric":
        JQJQ_rw = np.stack([
            _reweight_by_sign_perbin(JQJQ[a], sign_accum) for a in range(4)
        ], axis=0)
        JQJN_rw = np.stack([
            _reweight_by_sign_perbin(JQJN[a], sign_accum) for a in range(4)
        ], axis=0)
        JNJQ_rw = np.stack([
            _reweight_by_sign_perbin(JNJQ[a], sign_accum) for a in range(4)
        ], axis=0)

    # 5) Save imaginary time grid if dt is available.
    dt = _load_dt(path)
    if dt is not None:
        tau = np.arange(L, dtype=float) * dt
        np.save(os.path.join(outdir, f"{args.prefix}tau.npy"), tau)

    # 6) Save sign diagnostics (useful for future sign-problem cases).
    np.save(os.path.join(outdir, f"{args.prefix}sign_accum_perbin.npy"), sign_accum)
    np.save(os.path.join(outdir, f"{args.prefix}n_sample_perbin.npy"), n_sample)
    np.save(os.path.join(outdir, f"{args.prefix}mean_sign_perbin.npy"), mean_sign_perbin)

    # Print a concise summary.
    ms = np.mean(mean_sign_perbin)
    ms_min = np.min(mean_sign_perbin)
    ms_max = np.max(mean_sign_perbin)
    print(f"path: {path}  (found {len(h5_list)} .h5 files)")
    print(f"jj_4 shape: {jj_4.shape}  (4, Nbin, L) with Nbin={Nbin}, L={L}")
    if np.iscomplexobj(mean_sign_perbin):
        print("NOTE: mean sign is complex (sign problem).")
    elif float(ms) < 0.1:
        print("WARNING: mean sign is small; statistics may be very noisy.")
    else: 
        print(f"mean sign per bin: mean={ms:.6g}, min={ms_min:.6g}, max={ms_max:.6g}")

    to_save = {}
    if args.output_type == "JNJN" or args.output_type == "electric":
        to_save["JNJN"] = JNJN_rw
    elif args.output_type == "JQJQ":
        to_save["JQJQ"] = JQJQ_rw
    elif args.output_type == "JQJN":
        to_save["JQJN"] = JQJN_rw
    elif args.output_type == "JNJQ":
        to_save["JNJQ"] = JNJQ_rw
    elif args.output_type == "thermal" or args.output_type == "all":
        to_save["JNJN"] = JNJN_rw
        to_save["JQJQ"] = JQJQ_rw
        to_save["JQJN"] = JQJN_rw
        to_save["JNJQ"] = JNJQ_rw


    def save_one(key: str, comp: str) -> None:
        idx = COMP_TO_IDX[comp]
        data_all_comp = np.asarray(to_save[key])
        data = np.asarray(data_all_comp[idx])  # (Nbin, L)

        mean_tau, cov_tau, stderr = _stats_over_bins(data)

        np.save(os.path.join(outdir, f"{args.prefix}{key}_{comp}_perbin.npy"), data)
        np.save(os.path.join(outdir, f"{args.prefix}{key}_{comp}_mean.npy"), mean_tau)
        np.save(os.path.join(outdir, f"{args.prefix}{key}_{comp}_cov.npy"), cov_tau)
        np.save(os.path.join(outdir, f"{args.prefix}{key}_{comp}_stderr.npy"), stderr)

    keys = [k for k in ("JNJN","JQJQ","JQJN","JNJQ") if k in to_save]
    if args.component == "all":
        for comp in ("xx", "yy", "xy", "yx"):
            for key in keys:
                save_one(key, comp)
        print(f"Saved all files to: {outdir}")
    else:
        for key in keys:
            save_one(key, args.component)
            print(f"Saved {key}_{args.component} outputs to: {outdir}")


if __name__ == "__main__":
    main()
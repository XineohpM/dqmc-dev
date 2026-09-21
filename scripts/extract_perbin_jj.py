#!/usr/bin/env python3
"""
extract_perbin_jj.py
phoenixm@stanford.edu

Extract per-site imaginary-time current-current correlator from DQMC HDF5 outputs.

In `jqjq.py`, `get_component(path, "jj")` returns the q=0 raw numerator,
not divided by accumulated sign. This script saves self-contained paired
bundles so downstream bootstrap/MaxEnt/proxy code always has the aligned
numerator, sign, n_sample, tau grid, and metadata.

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
import util
import paired_bootstrap


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


def _scalar(value, name: str, file: str):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"Expected scalar {name} in {file}, got shape {array.shape}"
        )
    return array.reshape(-1)[0]


def _load_run_metadata(files: list[str]) -> dict:
    keys = (
        "metadata/beta",
        "params/dt",
        "params/L",
        "metadata/Nx",
        "metadata/Ny",
        "metadata/U",
        "metadata/mu",
        "metadata/nflux",
        "metadata/t'",
    )
    reference = None
    for file in files:
        values = jqjq.util.load_file(file, *keys)
        current = {
            "beta": float(_scalar(values[0], keys[0], file)),
            "dt": float(_scalar(values[1], keys[1], file)),
            "L": int(_scalar(values[2], keys[2], file)),
            "Nx": int(_scalar(values[3], keys[3], file)),
            "Ny": int(_scalar(values[4], keys[4], file)),
            "U": float(_scalar(values[5], keys[5], file)),
            "mu": float(_scalar(values[6], keys[6], file)),
            "nflux": int(_scalar(values[7], keys[7], file)),
            "tp": float(_scalar(values[8], keys[8], file)),
        }
        if not np.isclose(
            current["beta"] / current["dt"],
            current["L"],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"metadata/beta divided by params/dt does not match "
                f"params/L in {file}"
            )
        if reference is None:
            reference = current
            continue
        for key in ("L", "Nx", "Ny", "nflux"):
            if current[key] != reference[key]:
                raise ValueError(
                    f"Inconsistent {key} in {file}: "
                    f"{current[key]} != {reference[key]}"
                )
        for key in ("beta", "dt", "U", "mu", "tp"):
            if not np.isclose(
                current[key],
                reference[key],
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Inconsistent {key} in {file}: "
                    f"{current[key]} != {reference[key]}"
                )
    return reference


def _load_sign_and_ns_completed(
    path: str,
    files: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load accumulated sign and n_sample for the *completed* bins.

    Mirrors jqjq.get_sign() logic so ordering matches jqjq.get_component().

    Returns:
      (sign_accum, n_sample, source_files), each with first dimension
      Nbin_completed.
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
    if ns.size != len(files):
        raise ValueError(
            f"HDF5/sign row mismatch: {len(files)} files vs {ns.size} rows"
        )
    if not np.all(np.isfinite(ns)):
        raise ValueError("n_sample contains non-finite values")
    if np.max(ns) <= 0:
        raise ValueError("n_sample contains no completed positive-size bin")

    # Keep only completed bins (same mask as jqjq.get_sign).
    mask = ns == ns.max()
    source_files = np.asarray([os.path.basename(file) for file in files])
    sign_accum = s[mask].astype(
        np.complex128 if np.iscomplexobj(s) else float
    )
    n_sample = ns[mask].astype(float)
    source_files = source_files[mask]
    if not np.all(np.isfinite(sign_accum)):
        raise ValueError("completed-bin sign contains non-finite values")
    return sign_accum, n_sample, source_files


def _jackknife_stats(
    numerator: np.ndarray,
    sign_accum: np.ndarray,
    n_sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the ratio-of-sums mean and non-i.i.d. jackknife stderr."""

    if np.iscomplexobj(numerator) or np.iscomplexobj(sign_accum):
        jk_real = util.jackknife_noniid(
            n_sample,
            sign_accum,
            numerator,
            f=lambda ns, s, a: (a.T / s.T).T.real,
        )
        jk_imag = util.jackknife_noniid(
            n_sample,
            sign_accum,
            numerator,
            f=lambda ns, s, a: (a.T / s.T).T.imag,
        )
        mean_tau = jk_real[0] + 1j * jk_imag[0]
        stderr = np.hypot(jk_real[1], jk_imag[1])
    else:
        mean_tau, stderr = util.jackknife_noniid(
            n_sample,
            sign_accum,
            numerator,
        )

    return np.asarray(mean_tau), np.asarray(stderr)


def _sign_stats(
    n_sample: np.ndarray,
    sign_accum: np.ndarray,
) -> tuple[complex, float]:
    if np.iscomplexobj(sign_accum):
        jk_real = util.jackknife_noniid(
            n_sample,
            sign_accum,
            f=lambda ns, s: (s / ns).real,
        )
        jk_imag = util.jackknife_noniid(
            n_sample,
            sign_accum,
            f=lambda ns, s: (s / ns).imag,
        )
        mean = complex(jk_real[0], jk_imag[0])
        stderr = float(np.hypot(jk_real[1], jk_imag[1]))
        return mean, stderr
    mean, stderr = util.jackknife_noniid(
        n_sample,
        sign_accum,
        f=lambda ns, s: (s / ns).real,
    )
    return float(mean), float(stderr)


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
    metadata = _load_run_metadata(h5_list)

    # Load jj and project to physical tensor components.
    jj_q0 = jqjq.get_component(path, "jj")
    jj_4 = np.asarray(jqjq.electrical_sum(path, jj_q0))  # (4, Nbin, L), np.stack((jj_xx, jj_yy, jj_xy, jj_yx), axis=0)

    if jj_4.ndim != 3 or jj_4.shape[0] != 4:
        raise ValueError(f"Unexpected jj_4 shape {jj_4.shape}; expected (4, Nbin, L)")

    _, Nbin, L = jj_4.shape
    if L != metadata["L"]:
        raise ValueError(
            f"Correlator L={L} does not match params/L={metadata['L']}"
        )

    if args.output_type == "JNJN" or args.output_type == "electric":  # returns JNJN only
        JNJN = (-1) * jj_4
        source_transform = "jqjq.electrical_sum(meas_uneqlt/jj)"
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
        source_transform = "jqjq.thermal_sum"


    # Load sign and n_sample for completed bins (must align with jj_4 bins).
    sign_accum, n_sample, source_files = _load_sign_and_ns_completed(
        path,
        h5_list,
    )

    if sign_accum.shape[0] != Nbin:
        raise ValueError(
            f"Bin count mismatch: jj has Nbin={Nbin} but sign has {sign_accum.shape[0]}. "
            "This usually indicates inconsistent masking or mixed/incomplete files in the directory."
        )

    mean_sign, sign_stderr = _sign_stats(n_sample, sign_accum)
    tau = np.arange(L, dtype=float) * metadata["dt"]

    to_save = {}
    if args.output_type == "JNJN" or args.output_type == "electric":
        to_save["JNJN"] = JNJN
    elif args.output_type == "JQJQ":
        to_save["JQJQ"] = JQJQ
    elif args.output_type == "JQJN":
        to_save["JQJN"] = JQJN
    elif args.output_type == "JNJQ":
        to_save["JNJQ"] = JNJQ
    elif args.output_type == "thermal" or args.output_type == "all":
        to_save["JNJN"] = JNJN
        to_save["JQJQ"] = JQJQ
        to_save["JQJN"] = JQJN
        to_save["JNJQ"] = JNJQ

    common_metadata = {
        "format_version": paired_bootstrap.FORMAT_VERSION,
        "beta": metadata["beta"],
        "dt": metadata["dt"],
        "L": metadata["L"],
        "Nx": metadata["Nx"],
        "Ny": metadata["Ny"],
        "U": metadata["U"],
        "mu": metadata["mu"],
        "nflux": metadata["nflux"],
        "tp": metadata["tp"],
        "normalization": "per_site_q0",
        "mean_sign": mean_sign,
        "sign_stderr": sign_stderr,
        "source_transform": source_transform,
    }
    saved_files = []

    def save_one(key: str, comp: str) -> None:
        idx = COMP_TO_IDX[comp]
        data_all_comp = np.asarray(to_save[key])
        numerator = np.asarray(data_all_comp[idx])  # (Nbin, L)

        mean_tau, stderr = _jackknife_stats(
            numerator,
            sign_accum,
            n_sample,
        )
        bundle_metadata = dict(common_metadata)
        bundle_metadata.update(
            {
                "observable": key,
                "component": comp,
            }
        )
        bundle = paired_bootstrap.validate_paired_bundle(
            numerator=numerator,
            sign=sign_accum,
            n_sample=n_sample,
            tau=tau,
            mean=mean_tau,
            stderr=stderr,
            metadata=bundle_metadata,
            source_files=source_files,
        )
        output_file = os.path.join(
            outdir,
            f"{args.prefix}{key}_{comp}_paired.npz",
        )
        paired_bootstrap.save_paired_bundle(
            output_file,
            bundle,
            overwrite=True,
        )
        saved_files.append(output_file)

    keys = [k for k in ("JNJN","JQJQ","JQJN","JNJQ") if k in to_save]
    if args.component == "all":
        for comp in ("xx", "yy", "xy", "yx"):
            for key in keys:
                save_one(key, comp)
    else:
        for key in keys:
            save_one(key, args.component)
    print(
        f"Processed {Nbin} completed bins from {len(h5_list)} files; "
        f"average sign = {mean_sign:.12g} +/- {sign_stderr:.12g}"
    )
    print(f"Saved {len(saved_files)} paired bundles to: {outdir}")


if __name__ == "__main__":
    main()

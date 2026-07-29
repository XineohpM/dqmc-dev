#!/usr/bin/env python3
"""Extract the sign-reweighted local moment from equal-time DQMC bins.

For each completed bin b, the HDF5 datasets are raw signed accumulators:

    A_n,b = sum_i sign_i * n_i
    A_D,b = sum_i sign_i * (n_up n_down)_i
    S_b   = sum_i sign_i

The local-moment numerator is A_m2,b = A_n,b - 2 A_D,b and the physical
estimate is sum_b(A_m2,b) / sum_b(S_b).  Mean and uncertainty are evaluated
with util.jackknife_noniid using n_sample as the bin-size argument.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np

utilpath = Path(__file__).resolve().parents[1] / "util"
sys.path.insert(0, str(utilpath))

import util


PATH_NS = "meas_eqlt/n_sample"
PATH_SIGN = "meas_eqlt/sign"
PATH_DO = "meas_eqlt/double_occ"
PATH_N = "meas_eqlt/density"
PATH_NU = "meas_eqlt/density_u"
PATH_ND = "meas_eqlt/density_d"
PATH_BETA = "metadata/beta"
PATH_U = "metadata/U"
DENOMINATOR_RTOL = 1e-12


def _scalar(value: Any, name: str, filename: str) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"{filename}:{name} must be scalar, got shape {array.shape}"
        )
    result = array.reshape(-1)[0]
    if not np.isfinite(result):
        raise ValueError(f"{filename}:{name} is non-finite")
    return result.item()


def _mean_accumulator(
    value: Any,
    name: str,
    filename: str,
) -> np.number:
    """Reduce a scalar or site/orbital array to one raw spatial mean."""

    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{filename}:{name} is empty")
    if array.dtype.kind not in "biufc":
        raise ValueError(
            f"{filename}:{name} must be numeric, got dtype {array.dtype}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{filename}:{name} contains non-finite values")
    return np.mean(array)


def _read_bin(filename: str) -> Dict[str, Any]:
    with h5py.File(filename, "r") as h5:
        required = (PATH_NS, PATH_SIGN, PATH_DO, PATH_N, PATH_BETA, PATH_U)
        missing = [key for key in required if key not in h5]
        if missing:
            raise KeyError(
                f"{filename} is missing required dataset(s): {missing}"
            )

        n_sample = int(_scalar(h5[PATH_NS][...], PATH_NS, filename))
        if n_sample <= 0:
            raise ValueError(
                f"{filename}:{PATH_NS} must be positive, got {n_sample}"
            )

        sign = _scalar(h5[PATH_SIGN][...], PATH_SIGN, filename)
        density_array = np.asarray(h5[PATH_N][...])
        double_occ_array = np.asarray(h5[PATH_DO][...])
        if density_array.size != double_occ_array.size:
            raise ValueError(
                f"{filename}: density/double_occ size mismatch: "
                f"{density_array.size} vs {double_occ_array.size}"
            )

        density = _mean_accumulator(density_array, PATH_N, filename)
        double_occ = _mean_accumulator(
            double_occ_array,
            PATH_DO,
            filename,
        )

        has_nu = PATH_NU in h5
        has_nd = PATH_ND in h5
        if has_nu != has_nd:
            raise KeyError(
                f"{filename} must contain both {PATH_NU} and {PATH_ND}, "
                "or neither"
            )

        density_u: Optional[np.number] = None
        density_d: Optional[np.number] = None
        if has_nu:
            density_u_array = np.asarray(h5[PATH_NU][...])
            density_d_array = np.asarray(h5[PATH_ND][...])
            if (
                density_u_array.size != density_array.size
                or density_d_array.size != density_array.size
            ):
                raise ValueError(
                    f"{filename}: density_u/density_d sizes must match density"
                )
            density_u = _mean_accumulator(
                density_u_array,
                PATH_NU,
                filename,
            )
            density_d = _mean_accumulator(
                density_d_array,
                PATH_ND,
                filename,
            )

        return {
            "filename": filename,
            "n_sample": n_sample,
            "sign": sign,
            "density": density,
            "double_occ": double_occ,
            "density_u": density_u,
            "density_d": density_d,
            "beta": float(_scalar(h5[PATH_BETA][...], PATH_BETA, filename)),
            "U": float(_scalar(h5[PATH_U][...], PATH_U, filename)),
        }


def _validate_jackknife_denominators(sign: np.ndarray) -> None:
    sign = np.asarray(sign)
    total = np.sum(sign)
    total_abs = np.sum(np.abs(sign))
    if np.abs(total) <= DENOMINATOR_RTOL * total_abs:
        raise ValueError(
            "total accumulated sign/phase is too close to zero"
        )

    leave_one_out = total - sign
    leave_one_out_abs = total_abs - np.abs(sign)
    bad = np.abs(leave_one_out) <= (
        DENOMINATOR_RTOL * leave_one_out_abs
    )
    if np.any(bad):
        indices = np.flatnonzero(bad)[:10].tolist()
        raise ValueError(
            "jackknife leave-one-out accumulated sign/phase is too close "
            f"to zero when omitting bin(s) {indices}"
        )


def _jackknife_stats(
    n_sample: np.ndarray,
    sign: np.ndarray,
    numerator: np.ndarray,
):
    _validate_jackknife_denominators(sign)
    if np.iscomplexobj(sign) or np.iscomplexobj(numerator):
        jk_real = util.jackknife_noniid(
            n_sample,
            sign,
            numerator,
            f=lambda ns, s, a: (a / s).real,
        )
        jk_imag = util.jackknife_noniid(
            n_sample,
            sign,
            numerator,
            f=lambda ns, s, a: (a / s).imag,
        )
        mean = complex(jk_real[0], jk_imag[0])
        stderr = float(np.hypot(jk_real[1], jk_imag[1]))
        return mean, stderr

    mean, stderr = util.jackknife_noniid(
        n_sample,
        sign,
        numerator,
    )
    return float(mean), float(stderr)


def _check_metadata(
    records,
    temperature: float,
    expected_U: float,
) -> None:
    beta0 = records[0]["beta"]
    U0 = records[0]["U"]
    for record in records[1:]:
        if not np.isclose(
            record["beta"],
            beta0,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"inconsistent beta in {record['filename']}: "
                f"{record['beta']} != {beta0}"
            )
        if not np.isclose(
            record["U"],
            U0,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"inconsistent U in {record['filename']}: "
                f"{record['U']} != {U0}"
            )

    if not np.isclose(
        beta0 * temperature,
        1.0,
        rtol=1e-5,
        atol=1e-10,
    ):
        raise ValueError(
            f"temperature/beta mismatch: T={temperature:g}, beta={beta0:g}"
        )
    if not np.isclose(U0, expected_U, rtol=1e-12, atol=1e-12):
        raise ValueError(
            f"command-line U={expected_U:g} does not match HDF5 U={U0:g}"
        )


def analyze_temperature_dir(
    directory: str,
    temperature: float,
    expected_U: float,
    h5_glob: str,
    tol_density: float,
    tol_spin: float,
    strict_checks: bool,
) -> Dict[str, Any]:
    filenames = sorted(glob.glob(os.path.join(directory, h5_glob)))
    if not filenames:
        raise FileNotFoundError(
            f"No h5 files matching {h5_glob!r} under {directory}"
        )

    records = []
    failed = 0
    for filename in filenames:
        try:
            records.append(_read_bin(filename))
        except Exception as exc:
            failed += 1
            print(f"[WARN] failed to read {filename}: {exc}", file=sys.stderr)

    if not records:
        raise RuntimeError(
            f"{directory}: all HDF5 files failed validation "
            f"(failed={failed})"
        )

    _check_metadata(records, temperature, expected_U)

    n_sample_all = np.asarray(
        [record["n_sample"] for record in records],
        dtype=float,
    )
    completed = n_sample_all == np.max(n_sample_all)
    records = [
        record for record, keep in zip(records, completed) if keep
    ]
    if len(records) < 2:
        raise ValueError(
            f"{directory}: need at least two completed bins, got "
            f"{len(records)}"
        )

    n_sample = n_sample_all[completed]
    sign = np.asarray([record["sign"] for record in records])
    density = np.asarray([record["density"] for record in records])
    double_occ = np.asarray(
        [record["double_occ"] for record in records]
    )
    mz2_numerator = density - 2.0 * double_occ

    mz2_mean, mz2_stderr = _jackknife_stats(
        n_sample,
        sign,
        mz2_numerator,
    )
    density_mean, _ = _jackknife_stats(n_sample, sign, density)
    double_occ_mean, _ = _jackknife_stats(
        n_sample,
        sign,
        double_occ,
    )
    avg_sign = np.sum(sign) / np.sum(n_sample)

    spin_schema = [
        record["density_u"] is not None and record["density_d"] is not None
        for record in records
    ]
    density_mismatch = False
    spin_mismatch = False
    density_delta = np.nan
    spin_delta = np.nan
    if all(spin_schema):
        density_u = np.asarray(
            [record["density_u"] for record in records]
        )
        density_d = np.asarray(
            [record["density_d"] for record in records]
        )
        density_u_mean, _ = _jackknife_stats(
            n_sample,
            sign,
            density_u,
        )
        density_d_mean, _ = _jackknife_stats(
            n_sample,
            sign,
            density_d,
        )
        density_delta = density_u_mean + density_d_mean - density_mean
        spin_delta = density_u_mean - density_d_mean
        density_mismatch = np.abs(density_delta) > float(tol_density)
        spin_mismatch = np.abs(spin_delta) > float(tol_spin)
        if strict_checks and (density_mismatch or spin_mismatch):
            raise ValueError(
                f"{directory}: sign-reweighted density checks failed: "
                f"|nu+nd-n|={np.abs(density_delta):g}, "
                f"|nu-nd|={np.abs(spin_delta):g}"
            )
    elif any(spin_schema):
        print(
            f"[WARN] {directory}: density_u/density_d are not available "
            "for every completed bin; skipping spin-density checks",
            file=sys.stderr,
        )

    return {
        "temperature": float(temperature),
        "mz2_mean": mz2_mean,
        "mz2_stderr": float(mz2_stderr),
        "density_mean": density_mean,
        "double_occ_mean": double_occ_mean,
        "avg_sign": avg_sign,
        "nbin": len(records),
        "n_incomplete": int(np.count_nonzero(~completed)),
        "n_failed": int(failed),
        "density_mismatch": bool(density_mismatch),
        "spin_mismatch": bool(spin_mismatch),
        "density_delta": density_delta,
        "spin_delta": spin_delta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sign-reweighted <m_z^2>(T) from DQMC HDF5 bins."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing T_* subfolders.",
    )
    parser.add_argument(
        "--U",
        type=float,
        required=True,
        help=(
            "On-site interaction U. It is checked against HDF5 metadata and "
            "used for the atomic-limit reference curve."
        ),
    )
    parser.add_argument(
        "--h5_glob",
        default="*.h5",
        help="Glob for HDF5 files inside each T_* folder.",
    )
    parser.add_argument(
        "--out_prefix",
        default="local_moment",
        help="Prefix for output .npy/.png files.",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip T_* folders with no usable HDF5 bins instead of failing.",
    )
    parser.add_argument(
        "--tol_density",
        type=float,
        default=1e-6,
        help=(
            "Tolerance for the sign-reweighted "
            "|<n_up>+<n_down>-<n>| check."
        ),
    )
    parser.add_argument(
        "--tol_spin",
        type=float,
        default=1e-6,
        help="Tolerance for the sign-reweighted |<n_up>-<n_down>| check.",
    )
    parser.add_argument(
        "--strict_checks",
        action="store_true",
        help="Raise if a sign-reweighted density consistency check fails.",
    )
    parser.add_argument(
        "--imag_tol",
        type=float,
        default=1e-10,
        help=(
            "Imaginary-part tolerance for plotting. Complex estimates are "
            "always preserved in the saved .npy output."
        ),
    )
    return parser.parse_args()


def _parse_temperature(directory: str) -> float:
    name = os.path.basename(os.path.normpath(directory))
    if not name.startswith("T_"):
        raise ValueError(f"cannot parse temperature from directory {directory}")
    temperature = float(name[2:])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"invalid temperature in directory {directory}")
    return temperature


def main() -> None:
    args = parse_args()
    root = os.path.expanduser(args.root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Root path not found or not a directory: {root}"
        )
    if args.tol_density < 0 or args.tol_spin < 0 or args.imag_tol < 0:
        raise ValueError("all tolerances must be non-negative")

    t_dirs = sorted(glob.glob(os.path.join(root, "T_*")))
    t_dirs = [directory for directory in t_dirs if os.path.isdir(directory)]
    if not t_dirs:
        raise FileNotFoundError(f"No T_* subdirectories found under: {root}")

    results = []
    for directory in t_dirs:
        try:
            temperature = _parse_temperature(directory)
            result = analyze_temperature_dir(
                directory=directory,
                temperature=temperature,
                expected_U=args.U,
                h5_glob=args.h5_glob,
                tol_density=args.tol_density,
                tol_spin=args.tol_spin,
                strict_checks=args.strict_checks,
            )
        except Exception as exc:
            if args.skip_missing:
                print(f"[SKIP] {directory}: {exc}", file=sys.stderr)
                continue
            raise

        results.append(result)
        print(
            f"[OK] T_{temperature:g}: completed={result['nbin']} "
            f"incomplete={result['n_incomplete']} "
            f"failed={result['n_failed']} "
            f"<mz2>={result['mz2_mean']!s} "
            f"err={result['mz2_stderr']:.6g} "
            f"avg_sign={result['avg_sign']!s} "
            f"density_mismatch={result['density_mismatch']} "
            f"spin_mismatch={result['spin_mismatch']}"
        )

    if not results:
        raise RuntimeError("No temperature directories were processed")

    results.sort(key=lambda item: item["temperature"])
    temperature = np.asarray(
        [item["temperature"] for item in results],
        dtype=float,
    )
    mz2 = np.asarray([item["mz2_mean"] for item in results])
    mz2_stderr = np.asarray(
        [item["mz2_stderr"] for item in results],
        dtype=float,
    )
    avg_sign = np.asarray([item["avg_sign"] for item in results])
    nbin = np.asarray([item["nbin"] for item in results], dtype=int)

    np.save(os.path.join(root, f"{args.out_prefix}_T.npy"), temperature)
    np.save(os.path.join(root, f"{args.out_prefix}_mz2.npy"), mz2)
    np.save(
        os.path.join(root, f"{args.out_prefix}_mz2_err.npy"),
        mz2_stderr,
    )
    np.save(
        os.path.join(root, f"{args.out_prefix}_avg_sign.npy"),
        avg_sign,
    )
    np.save(os.path.join(root, f"{args.out_prefix}_nbin.npy"), nbin)

    mz2_real = np.real(mz2)
    if np.iscomplexobj(mz2):
        imag_max = float(np.max(np.abs(np.imag(mz2))))
        real_scale = float(np.max(np.abs(mz2_real)))
        tolerance = max(args.imag_tol, args.imag_tol * real_scale)
        if imag_max > tolerance:
            print(
                "[WARN] local moment has a non-negligible imaginary part "
                f"(max |imag|={imag_max:g}); saved complex values and "
                "plotted only the real part",
                file=sys.stderr,
            )

    figure, axis = plt.subplots()
    axis.errorbar(
        temperature,
        mz2_real,
        yerr=mz2_stderr,
        fmt="o",
        capsize=2,
        label="DQMC (real part)",
    )

    if temperature.size >= 2:
        log_temperature = np.log10(temperature)
        dense_log_temperature = np.linspace(
            log_temperature.min(),
            log_temperature.max(),
            400,
        )
        dense_temperature = 10 ** dense_log_temperature
        dense_mz2 = np.interp(
            dense_log_temperature,
            log_temperature,
            mz2_real,
        )
        axis.plot(
            dense_temperature,
            dense_mz2,
            linewidth=2.0,
            alpha=0.45,
            color="#4DA3FF",
        )
    else:
        dense_temperature = temperature

    exponent = np.clip(
        -args.U / (2.0 * dense_temperature),
        -700.0,
        700.0,
    )
    atomic_mz2 = 1.0 / (np.exp(exponent) + 1.0)
    axis.plot(
        dense_temperature,
        atomic_mz2,
        linestyle="--",
        linewidth=1.8,
        color="orange",
        alpha=0.8,
        label=r"atomic limit $t=0$",
    )

    axis.set_xscale("log")
    axis.set_xlabel("T")
    axis.set_ylabel(r"$\langle m_z^2 \rangle$")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    figure.tight_layout()
    output_png = os.path.join(
        root,
        f"{args.out_prefix}_mz2_vs_T.png",
    )
    figure.savefig(output_png, dpi=160)
    plt.close(figure)
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()

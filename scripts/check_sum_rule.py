#!/usr/bin/env python3
"""Check the optical sum rule using sign-aware paired bootstrap estimates."""

import argparse
import glob
import os

import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import paired_bootstrap


def _scalar(value, name, path):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"Expected scalar {name} in {path}, got shape {array.shape}"
        )
    return array.reshape(-1)[0]


def _as_real(values, name, imag_tol):
    values = np.asarray(values)
    if np.iscomplexobj(values):
        scale = max(1.0, float(np.max(np.abs(values.real))))
        tolerance = float(imag_tol) * scale
        max_imag = float(np.max(np.abs(values.imag)))
        if max_imag > tolerance:
            raise ValueError(
                f"{name} has non-negligible imaginary part: "
                f"{max_imag:g} > {tolerance:g}"
            )
        values = values.real
    return np.asarray(values, dtype=float)


def half_interval_norm(corr, dt, beta):
    """Match ``maxent.Preprocess(..., boson, sym=True)['norm']``."""

    corr = np.asarray(corr)
    if corr.ndim != 1:
        raise ValueError(f"correlator must be 1D, got shape {corr.shape}")
    L = corr.shape[0]
    mean_with_endpoint = np.concatenate((corr, corr[:1]))
    tau = np.arange(L + 1, dtype=float) * dt
    spline = InterpolatedUnivariateSpline(
        tau,
        mean_with_endpoint,
        ext=2,
        check_finite=True,
    )
    return float(spline.integral(0, beta / 2))


def _kinetic_numerator(g00, nx, ny, tp):
    g00 = np.asarray(g00).reshape((nx, ny), order="F")
    t1 = 2 * (
        g00[0, 1]
        + g00[1, 0]
        + g00[0, ny - 1]
        + g00[nx - 1, 0]
    )
    t2 = 2 * tp * (
        g00[1, 1]
        + g00[1, ny - 1]
        + g00[nx - 1, 1]
        + g00[nx - 1, ny - 1]
    )
    return t1 + t2


def load_kinetic_paired(directory):
    """Load completed equal-time kinetic numerator/sign rows."""

    files = sorted(glob.glob(os.path.join(directory, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {directory}")

    numerators = []
    signs = []
    n_samples = []
    source_files = []
    reference = None
    for path in files:
        with h5py.File(path, "r") as handle:
            current = {
                "Nx": int(_scalar(handle["metadata/Nx"][()], "Nx", path)),
                "Ny": int(_scalar(handle["metadata/Ny"][()], "Ny", path)),
                "tp": float(
                    _scalar(handle["metadata/t'"][()], "t'", path)
                ),
                "beta": float(
                    _scalar(handle["metadata/beta"][()], "beta", path)
                ),
            }
            sign = _scalar(
                handle["meas_eqlt/sign"][()],
                "meas_eqlt/sign",
                path,
            )
            n_sample = float(
                _scalar(
                    handle["meas_eqlt/n_sample"][()],
                    "meas_eqlt/n_sample",
                    path,
                )
            )
            numerator = _kinetic_numerator(
                handle["meas_eqlt/g00"][()],
                current["Nx"],
                current["Ny"],
                current["tp"],
            )

        if reference is None:
            reference = current
        else:
            for key in ("Nx", "Ny"):
                if current[key] != reference[key]:
                    raise ValueError(f"Inconsistent {key} in {path}")
            for key in ("tp", "beta"):
                if not np.isclose(
                    current[key],
                    reference[key],
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise ValueError(f"Inconsistent {key} in {path}")
        numerators.append(numerator)
        signs.append(sign)
        n_samples.append(n_sample)
        source_files.append(os.path.basename(path))

    numerators = np.asarray(numerators)
    signs = np.asarray(signs)
    n_samples = np.asarray(n_samples, dtype=float)
    source_files = np.asarray(source_files)
    if not np.all(np.isfinite(n_samples)) or np.max(n_samples) <= 0:
        raise ValueError("Invalid equal-time n_sample values")
    completed = n_samples == np.max(n_samples)
    numerators = numerators[completed]
    signs = signs[completed]
    n_samples = n_samples[completed]
    source_files = source_files[completed]
    if not np.all(np.isfinite(numerators)):
        raise ValueError("Equal-time kinetic numerator contains non-finite data")
    if not np.all(np.isfinite(signs)):
        raise ValueError("Equal-time sign contains non-finite data")
    return numerators, signs, n_samples, source_files, reference


def _bootstrap_sum_rule(
    bundle,
    kinetic_numerator,
    kinetic_sign,
    *,
    nboot,
    seed,
    block_size,
    aligned,
    dt,
    beta,
    imag_tol,
):
    corr_indices = paired_bootstrap.bootstrap_indices(
        bundle.nbin,
        nboot,
        block_size=block_size,
        seed=seed,
    )
    corr_estimates = paired_bootstrap.bootstrap_ratio_of_sums(
        bundle.numerator,
        bundle.sign,
        corr_indices,
    )
    if aligned:
        kinetic_indices = corr_indices
    else:
        kinetic_indices = paired_bootstrap.bootstrap_indices(
            kinetic_sign.size,
            nboot,
            block_size=block_size,
            seed=seed + 1,
        )
    kinetic_estimates = paired_bootstrap.bootstrap_ratio_of_sums(
        kinetic_numerator[:, None],
        kinetic_sign,
        kinetic_indices,
    )[:, 0]

    corr_estimates = _as_real(
        corr_estimates,
        "bootstrap correlator",
        imag_tol,
    )
    kinetic_estimates = _as_real(
        kinetic_estimates,
        "bootstrap kinetic energy",
        imag_tol,
    )
    norm_estimates = np.asarray(
        [4 * half_interval_norm(row, dt, beta) for row in corr_estimates]
    )
    target_estimates = -kinetic_estimates
    difference_estimates = norm_estimates - target_estimates
    return norm_estimates, target_estimates, difference_estimates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        help="Parent directory containing run subfolders.",
    )
    parser.add_argument(
        "--relpath_list",
        nargs="+",
        required=True,
        help="Relative run paths.",
    )
    parser.add_argument(
        "--correlator_name",
        default="JNJN_xx_paired.npz",
        help="Paired current-current correlator bundle filename.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for the central-value sum-rule check.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of paired-bootstrap replicates; use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--bootstrap_block_size",
        type=int,
        default=1,
        help="Circular bootstrap block size in source bins.",
    )
    parser.add_argument("--imag_tol", type=float, default=1e-10)
    args = parser.parse_args()

    base = os.path.expanduser(args.path)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Base path not found: {base}")
    if args.bootstrap < 0 or args.bootstrap == 1:
        raise ValueError("--bootstrap must be 0 or an integer >= 2")

    def temperature(relpath):
        bundle = paired_bootstrap.load_paired_bundle(
            os.path.join(base, relpath, args.correlator_name)
        )
        return 1.0 / float(bundle.metadata["beta"])

    for run_index, relpath in enumerate(
        sorted(args.relpath_list, key=temperature)
    ):
        directory = os.path.join(base, relpath)
        bundle_path = os.path.join(directory, args.correlator_name)
        bundle = paired_bootstrap.load_paired_bundle(bundle_path)
        if (
            bundle.metadata["observable"] != "JNJN"
            or bundle.metadata.get("component") != "xx"
        ):
            raise ValueError(
                f"{bundle_path} must contain JNJN xx correlator data"
            )
        dt = float(bundle.metadata["dt"])
        beta = float(bundle.metadata["beta"])
        corr_mean = _as_real(bundle.mean, "correlator mean", args.imag_tol)
        norm4 = 4 * half_interval_norm(corr_mean, dt, beta)

        (
            kinetic_numerator,
            kinetic_sign,
            kinetic_n_sample,
            kinetic_sources,
            kinetic_metadata,
        ) = load_kinetic_paired(directory)
        if not np.isclose(
            beta,
            kinetic_metadata["beta"],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Equal-time and unequal-time beta do not match")
        kinetic = paired_bootstrap.ratio_of_sums(
            kinetic_numerator[:, None],
            kinetic_sign,
        )[0]
        kinetic = float(
            _as_real(kinetic, "kinetic energy", args.imag_tol)
        )
        target = -kinetic
        relative_difference = (norm4 - target) / target

        aligned = (
            bundle.source_files is not None
            and np.array_equal(bundle.source_files, kinetic_sources)
        )
        alignment = "paired-by-source_files" if aligned else "independent"
        if not aligned:
            print(
                "WARNING: unequal-time and equal-time completed source files "
                "are not identical; using independent bootstrap streams"
            )

        norm_error = target_error = difference_error = np.nan
        z_score = np.nan
        if args.bootstrap > 0:
            norm_samples, target_samples, difference_samples = (
                _bootstrap_sum_rule(
                    bundle,
                    kinetic_numerator,
                    kinetic_sign,
                    nboot=args.bootstrap,
                    seed=int(args.seed) + 2 * run_index,
                    block_size=args.bootstrap_block_size,
                    aligned=aligned,
                    dt=dt,
                    beta=beta,
                    imag_tol=args.imag_tol,
                )
            )
            norm_error = float(np.std(norm_samples, ddof=1))
            target_error = float(np.std(target_samples, ddof=1))
            difference_error = float(np.std(difference_samples, ddof=1))
            if difference_error > 0:
                z_score = abs(norm4 - target) / difference_error

        uneqlt_mean_sign = bundle.sign.sum() / bundle.n_sample.sum()
        eqlt_mean_sign = kinetic_sign.sum() / kinetic_n_sample.sum()
        uneqlt_mean_sign = _as_real(
            uneqlt_mean_sign,
            "unequal-time mean sign",
            args.imag_tol,
        ).item()
        eqlt_mean_sign = _as_real(
            eqlt_mean_sign,
            "equal-time mean sign",
            args.imag_tol,
        ).item()

        print("T = ", 1.0 / beta)
        print("dt = ", dt, "(paired bundle)")
        print("corr shape = ", bundle.numerator.shape)
        print("kinetic bins = ", kinetic_sign.size)
        print("bootstrap alignment = ", alignment)
        print("unequal-time mean sign = ", uneqlt_mean_sign)
        print("equal-time mean sign = ", eqlt_mean_sign)
        print("norm of correlator = ", norm4)
        print("norm bootstrap stderr = ", norm_error)
        print("kinetic energy target = ", target)
        print("kinetic target bootstrap stderr = ", target_error)
        print("relative difference = ", relative_difference)
        print("difference bootstrap stderr = ", difference_error)
        print("difference / bootstrap error = ", z_score)
        if np.isclose(norm4, target, rtol=args.rtol, atol=0):
            print("norm of correlator = kinetic energy")
        else:
            print("norm of correlator and kinetic energy are not close")
        print("k/norm = ", target / norm4)
        print(" ")


if __name__ == "__main__":
    main()

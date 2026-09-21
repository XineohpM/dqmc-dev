#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot JNJN_xx directly from a self-contained paired bundle."""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import paired_bootstrap


def _find_unique(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")
    if len(hits) > 1:
        raise RuntimeError(
            f"Multiple files matched pattern: {pattern}\n"
            + "\n".join(hits)
        )
    return hits[0]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        help="Directory containing a *JNJN_xx_paired.npz bundle.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional bundle filename/path; otherwise require one unique match.",
    )
    parser.add_argument(
        "--imag_tol",
        type=float,
        default=1e-10,
        help="Relative tolerance for a numerical imaginary part.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Save the figure to this path instead of displaying it.",
    )
    args = parser.parse_args()

    base = os.path.abspath(os.path.expanduser(args.path))
    if not os.path.isdir(base):
        raise NotADirectoryError(f"--path is not a directory: {base}")
    if args.bundle is None:
        bundle_path = _find_unique(
            os.path.join(base, "*JNJN_xx_paired.npz")
        )
    else:
        bundle_path = os.path.expanduser(args.bundle)
        if not os.path.isabs(bundle_path):
            bundle_path = os.path.join(base, bundle_path)

    bundle = paired_bootstrap.load_paired_bundle(bundle_path)
    if (
        bundle.metadata["observable"] != "JNJN"
        or bundle.metadata.get("component") != "xx"
    ):
        raise ValueError(
            "Bundle must contain observable='JNJN', component='xx'"
        )

    mean = _as_real(bundle.mean, "bundle mean", args.imag_tol)
    stderr = np.asarray(bundle.stderr, dtype=float)
    tau = np.asarray(bundle.tau, dtype=float)
    beta = float(bundle.metadata["beta"])
    dt = float(bundle.metadata["dt"])
    L = bundle.ntau
    mid = L // 2
    lambda_mid = (
        mean[mid]
        if L % 2 == 0
        else 0.5 * (mean[mid] + mean[mid + 1])
    )
    print("Lambda(beta/2)=", float(lambda_mid))

    plt.figure()
    plt.plot(tau, mean, lw=2.0, label="jackknife mean JNJN")
    plt.fill_between(
        tau,
        mean - stderr,
        mean + stderr,
        alpha=0.25,
        label="jackknife stderr",
    )
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\langle J_x(\tau)\,J_x(0)\rangle$")
    plt.title(
        f"JNJN_xx (nbin={bundle.nbin}, L={L}, dt={dt:g}, beta={beta:g})"
    )
    plt.legend(loc="best")
    plt.tight_layout()

    if args.out:
        output = os.path.abspath(os.path.expanduser(args.out))
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output, dpi=200)
        print(f"[OK] Saved plot -> {output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

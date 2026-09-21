'''
extract_1_particle_local_g.py
phoenixm@stanford.edu

Extract imaginary-time single particle local Green's function from DQMC HDF5 outputs.

Usage examples:

'''
import glob
import argparse
import os, sys
import numpy as np
from pathlib import Path

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))

import util
import paired_bootstrap


def _scalar(value, name: str, file: str):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"Expected scalar {name} in {file}, got shape {array.shape}"
        )
    return array.reshape(-1)[0]


def _check_metadata(reference: dict, current: dict, file: str) -> None:
    for key in ("dt", "beta", "U", "mu"):
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
    for key in ("L", "Nx", "Ny"):
        if current[key] != reference[key]:
            raise ValueError(
                f"Inconsistent {key} in {file}: "
                f"{current[key]} != {reference[key]}"
            )


def _validate_jackknife_denominators(sign_accum: np.ndarray) -> None:
    sign_accum = np.asarray(sign_accum)
    total = sign_accum.sum()
    absolute_weight = np.abs(sign_accum).sum()
    tolerance = (
        paired_bootstrap.DEFAULT_DENOMINATOR_RTOL * absolute_weight
    )
    if np.abs(total) <= tolerance:
        raise ValueError(
            "Total accumulated sign/phase is too close to zero for a stable ratio"
        )

    if sign_accum.size <= 1:
        return
    leave_one_out = total - sign_accum
    leave_one_out_weight = np.maximum(
        absolute_weight - np.abs(sign_accum),
        0.0,
    )
    bad = np.abs(leave_one_out) <= (
        paired_bootstrap.DEFAULT_DENOMINATOR_RTOL * leave_one_out_weight
    )
    if np.any(bad):
        indices = np.flatnonzero(bad)[:10].tolist()
        raise ValueError(
            "Jackknife leave-one-out accumulated sign/phase is too close "
            f"to zero when omitting bin(s) {indices}"
        )


def _jackknife_stats(
    numerator: np.ndarray,
    sign_accum: np.ndarray,
    n_sample: np.ndarray,
):
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
        mean = jk_real[0] + 1j * jk_imag[0]
        stderr = np.hypot(jk_real[1], jk_imag[1])
    else:
        mean, stderr = util.jackknife_noniid(
            n_sample,
            sign_accum,
            numerator,
        )
    return np.asarray(mean), np.asarray(stderr)


def _sign_stats(n_sample: np.ndarray, sign_accum: np.ndarray):
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


def get_meas(file: str):
    gu, gd, g, beta, sign, Nx, Ny, nsamp, dt_file, U, mu = util.load_file(
        file,
        "meas_uneqlt/gt0_u",
        "meas_uneqlt/gt0_d",
        "meas_uneqlt/gt0",
        "metadata/beta",
        "meas_uneqlt/sign",
        "metadata/Nx",
        "metadata/Ny",
        "meas_uneqlt/n_sample",
        "params/dt",
        "metadata/U",
        "metadata/mu",
    )
    sign = _scalar(sign, "meas_uneqlt/sign", file)
    nsamp = int(_scalar(nsamp, "meas_uneqlt/n_sample", file))
    beta = float(_scalar(beta, "metadata/beta", file))
    Nx = int(_scalar(Nx, "metadata/Nx", file))
    Ny = int(_scalar(Ny, "metadata/Ny", file))
    dt_file = float(_scalar(dt_file, "params/dt", file))
    U = float(_scalar(U, "metadata/U", file))
    mu = float(_scalar(mu, "metadata/mu", file))
    N = Nx * Ny
    L_float = beta / dt_file
    L = int(round(L_float))
    if not np.isclose(L_float, L, rtol=1e-12, atol=1e-12):
        raise ValueError(
            f"metadata/beta divided by params/dt is not an integer in {file}: "
            f"{beta} / {dt_file} = {L_float}"
        )
    try:
        gu = np.asarray(gu).reshape(L, N)
        gd = np.asarray(gd).reshape(L, N)
        g = np.asarray(g).reshape(L, N)
    except Exception as e:
        raise ValueError(
            f"Failed to reshape gt0 datasets to (L={L}, N={N}) for file {file}."
        ) from e
    metadata = {
        "beta": beta,
        "dt": dt_file,
        "L": L,
        "Nx": Nx,
        "Ny": Ny,
        "U": U,
        "mu": mu,
    }
    return gu, gd, g, sign, nsamp, metadata

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Directory containing HDF5 bin files")
    p.add_argument("--output_path",
                   help="Directory for output, will be created if needed")
    p.add_argument("--out_prefix", default="1_particle_local_", help="Output file prefix")
    args = p.parse_args()
    path = args.path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Base path not found or not a directory: {path}")
    output_path = args.output_path if args.output_path is not None else path
    os.makedirs(output_path, exist_ok=True)

    bins_g = []
    bins_gu = []
    bins_gd = []
    bins_sign = []
    bins_nsamp = []
    bins_source_files = []
    metadata = None

    files = sorted(glob.glob(os.path.join(path, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {path}")
    for file in files:
        gu, gd, g, sign, nsamp, file_metadata = get_meas(file)
        if metadata is None:
            metadata = file_metadata
        else:
            _check_metadata(metadata, file_metadata, file)
        gu_loc = gu[:, 0]
        gd_loc = gd[:, 0]
        g_loc = 0.5 * (gu_loc + gd_loc)
        bins_g.append(g_loc)
        bins_gu.append(gu_loc)
        bins_gd.append(gd_loc)
        bins_sign.append(sign)
        bins_nsamp.append(nsamp)
        bins_source_files.append(os.path.basename(file))
    bins_g = np.asarray(bins_g)
    bins_gu = np.asarray(bins_gu)
    bins_gd = np.asarray(bins_gd)
    bins_sign = np.asarray(bins_sign)
    bins_nsamp = np.asarray(bins_nsamp, dtype=float)
    bins_source_files = np.asarray(bins_source_files)

    if bins_g.size == 0 or bins_gu.size == 0 or bins_gd.size == 0:
        raise ValueError(f"No valid bins found in {path}")
    if np.any(~np.isfinite(bins_nsamp)):
        raise ValueError(f"Non-finite n_sample encountered in {path}")
    if np.max(bins_nsamp) <= 0:
        raise ValueError(f"Non-positive n_sample encountered in {path}")

    completed = bins_nsamp == np.max(bins_nsamp)
    dropped_files = bins_source_files[~completed]
    bins_g = bins_g[completed]
    bins_gu = bins_gu[completed]
    bins_gd = bins_gd[completed]
    bins_sign = bins_sign[completed]
    bins_nsamp = bins_nsamp[completed]
    bins_source_files = bins_source_files[completed]

    if np.any(~np.isfinite(bins_g)):
        raise ValueError(f"Non-finite g encountered in completed bins in {path}")
    if np.any(~np.isfinite(bins_gu)):
        raise ValueError(f"Non-finite gu encountered in completed bins in {path}")
    if np.any(~np.isfinite(bins_gd)):
        raise ValueError(f"Non-finite gd encountered in completed bins in {path}")
    if np.any(~np.isfinite(bins_sign)):
        raise ValueError(f"Non-finite sign encountered in completed bins in {path}")
    if np.any(bins_nsamp <= 0):
        raise ValueError(f"Non-positive n_sample encountered in completed bins in {path}")
    _validate_jackknife_denominators(bins_sign)
    sign_mean, sign_err = _sign_stats(bins_nsamp, bins_sign)
    g_mean, g_err = _jackknife_stats(bins_g, bins_sign, bins_nsamp)
    gu_mean, gu_err = _jackknife_stats(bins_gu, bins_sign, bins_nsamp)
    gd_mean, gd_err = _jackknife_stats(bins_gd, bins_sign, bins_nsamp)
    g_mean = g_mean.reshape(-1)
    g_err = g_err.reshape(-1)
    gu_mean = gu_mean.reshape(-1)
    gu_err = gu_err.reshape(-1)
    gd_mean = gd_mean.reshape(-1)
    gd_err = gd_err.reshape(-1)

    L = metadata["L"]
    tau = np.arange(L, dtype=float) * metadata["dt"]

    common_metadata = {
        "format_version": paired_bootstrap.FORMAT_VERSION,
        "beta": metadata["beta"],
        "dt": metadata["dt"],
        "L": metadata["L"],
        "Nx": metadata["Nx"],
        "Ny": metadata["Ny"],
        "U": metadata["U"],
        "mu": metadata["mu"],
        "normalization": "local_displacement_zero",
        "mean_sign": sign_mean,
        "sign_stderr": sign_err,
    }
    bundle_specs = (
        (
            "g",
            bins_g,
            g_mean,
            g_err,
            "spin_average",
            "0.5*(meas_uneqlt/gt0_u+meas_uneqlt/gt0_d)",
        ),
        (
            "gu",
            bins_gu,
            gu_mean,
            gu_err,
            "up",
            "meas_uneqlt/gt0_u",
        ),
        (
            "gd",
            bins_gd,
            gd_mean,
            gd_err,
            "down",
            "meas_uneqlt/gt0_d",
        ),
    )
    saved_files = []
    for observable, numerator, mean, stderr, spin, source_dataset in bundle_specs:
        bundle_metadata = dict(common_metadata)
        bundle_metadata.update(
            {
                "observable": observable,
                "spin": spin,
                "source_dataset": source_dataset,
            }
        )
        bundle = paired_bootstrap.validate_paired_bundle(
            numerator=numerator,
            sign=bins_sign,
            n_sample=bins_nsamp,
            tau=tau,
            mean=mean,
            stderr=stderr,
            metadata=bundle_metadata,
            source_files=bins_source_files,
        )
        output_file = os.path.join(
            output_path,
            f"{args.out_prefix}{observable}_paired.npz",
        )
        paired_bootstrap.save_paired_bundle(
            output_file,
            bundle,
            overwrite=True,
        )
        saved_files.append(output_file)

    print(
        f"Processed {bins_g.shape[0]} completed bins from "
        f"{len(files)} files in {path}"
    )
    if dropped_files.size:
        print(
            f"Dropped {dropped_files.size} incomplete bins: "
            f"{', '.join(dropped_files[:10])}"
        )
    print(f"Average sign = {sign_mean:.12g} +/- {sign_err:.12g}")
    print(f"Saved {len(saved_files)} paired bundles to {output_path}")

if __name__ == "__main__":
    main()

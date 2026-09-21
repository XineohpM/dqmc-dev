from glob import glob
from pathlib import Path
import sys, os, re
import numpy as np
import argparse

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))

import util
import paired_bootstrap


def _validate_jackknife_denominators(sign):
    sign = np.asarray(sign)
    total = np.sum(sign)
    total_abs = np.sum(np.abs(sign))
    rtol = paired_bootstrap.DEFAULT_DENOMINATOR_RTOL
    if np.abs(total) <= rtol * total_abs:
        raise ValueError("total accumulated sign/phase is too close to zero")

    leave_one_out = total - sign
    leave_one_out_abs = total_abs - np.abs(sign)
    bad = np.abs(leave_one_out) <= rtol * leave_one_out_abs
    if np.any(bad):
        indices = np.flatnonzero(bad)[:10].tolist()
        raise ValueError(
            "jackknife leave-one-out accumulated sign/phase is too close "
            f"to zero when omitting bin(s) {indices}"
        )

def get_mu_n(path):
    '''From all files in path, get chemical potential and filling info
    Args:
        path
    Returns:
        tuple (mu, density, density_err)
    '''
    n_sample, sign, density = \
        util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign",
                        "meas_eqlt/density")

    n_sample = np.asarray(n_sample).reshape(-1)
    sign = np.asarray(sign).reshape(-1)
    density = np.asarray(density)
    if density.ndim == 0:
        raise ValueError("density must have a bin dimension")
    if not (
        density.shape[0] == sign.size == n_sample.size
    ):
        raise ValueError(
            "n_sample/sign/density bin count mismatch: "
            f"{n_sample.size}/{sign.size}/{density.shape[0]}"
        )

    dsum = density.reshape(density.shape[0], -1).sum(axis=1)
    valid = (
        np.isfinite(n_sample)
        & (n_sample > 0)
        & np.isfinite(sign)
        & np.isfinite(dsum)
    )
    if not np.any(valid):
        return util.load_firstfile(path, "metadata/mu")[0], np.nan, np.nan

    n_sample = n_sample[valid]
    sign = sign[valid]
    dsum = dsum[valid]

    completed = n_sample == np.max(n_sample)
    if not completed.all():
        print(
            f"{path} incomplete: "
            f"{completed.sum()}/{len(completed)} valid bins completed"
        )
    n_sample = n_sample[completed]
    sign = sign[completed]
    dsum = dsum[completed]

    if sign.size < 3:
        return util.load_firstfile(path, "metadata/mu")[0], np.nan, np.nan

    # A completed bin with zero accumulated sign remains part of both sums.
    # Reject only unstable full-sample or leave-one-out denominators.
    _validate_jackknife_denominators(sign)
    nj = util.jackknife_noniid(n_sample, sign, dsum)

    return util.load_firstfile(path, "metadata/mu")[0], nj[0], nj[1]

def infer_target_n_from_path(p: str) -> float:
    m = re.search(r"/n(0\.\d+)(?:/|$)", p)
    return float(m.group(1)) if m else np.nan


def infer_T_beta_U_from_path(p: str):
    m = re.search(r"/T([0-9eE+\-\.]+)_beta([0-9eE+\-\.]+)_U([0-9eE+\-\.]+)(?:/|$)", p)
    if not m:
        return np.nan, np.nan, np.nan
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def get_nsite_from_firstfile(mu_dir: str) -> int:
    d0 = util.load_firstfile(mu_dir, "meas_eqlt/density")[0]
    # d0 should be shape (Nsite,) or (Nsite, something). We want first dimension.
    if hasattr(d0, "shape"):
        return int(d0.shape[0])
    # fallback
    return int(len(d0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, 
                   help="Base directory")
    p.add_argument("--output_path",  
                   help="Directory for output, will be created if needed")
    p.add_argument("--glob", default="n*/T*_beta*_U*/mu*/")
    args = p.parse_args()
    base = Path(os.path.expanduser(args.path))
    if not base.is_dir():
        raise FileNotFoundError(f"Base path not found or not a directory: {base}")
    
    # Resolve output location
    if args.output_path is None:
        out_dir = "."
        out_prefix = os.path.join(out_dir, "n_from_mu")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out = os.path.expanduser(args.output_path)
        # If user passes a directory (existing or endswith /), write into it.
        if os.path.isdir(out) or out.endswith(os.sep):
            out_dir = out
            os.makedirs(out_dir, exist_ok=True)
            out_prefix = os.path.join(out_dir, "n_from_mu")
        else:
            # treat as a prefix
            out_prefix = out
            out_dir = os.path.dirname(out_prefix) or "."
            os.makedirs(out_dir, exist_ok=True)

    # Discover all mu directories
    mu_dirs = sorted(base.glob(args.glob))
    #pattern = os.path.join(base, "n*", "T*_beta*_U*", "mu*")
    #mu_dirs = sorted(glob(pattern))
    if not mu_dirs:
        raise FileNotFoundError(f"No mu directories found under {base}")

    rows = []
    n_bad = 0

    for mu_dir in mu_dirs:
        mu_dir_str = str(mu_dir)
        mu_dir_h5 = mu_dir_str if mu_dir_str.endswith(os.sep) else (mu_dir_str + os.sep)
        # Infer target n and (T,beta,U) from path
        n_target = infer_target_n_from_path(mu_dir_str)
        T, beta, U = infer_T_beta_U_from_path(mu_dir_str)

        # Determine Nsite for converting N -> <n>
        try:
            nsite = get_nsite_from_firstfile(mu_dir_h5)
        except Exception:
            nsite = np.nan

        # Compute (mu, N_mean, N_err) using existing helper
        try:
            mu, N_mean, N_err = get_mu_n(mu_dir_h5)
        except Exception as e:
            n_bad += 1
            mu, N_mean, N_err = np.nan, np.nan, np.nan

        # Convert to per-site filling <n> if possible
        if np.isfinite(N_mean) and np.isfinite(nsite) and nsite > 0:
            n_mean = N_mean / nsite
        else:
            n_mean = np.nan
        if np.isfinite(N_err) and np.isfinite(nsite) and nsite > 0:
            n_err = N_err / nsite
        else:
            n_err = np.nan

        # Deviation from target
        dn = n_mean - n_target if np.isfinite(n_mean) and np.isfinite(n_target) else np.nan

        rows.append((
            n_target, T, beta, U, mu, n_mean, n_err, dn, int(nsite) if np.isfinite(nsite) else -1, mu_dir_str
        ))

    # Sort rows: by n_target, then T, then mu
    rows.sort(key=lambda r: (r[0], r[1], r[4]))

    # Write TSV
    out_tsv = out_prefix + ".tsv"
    header = "n_target\tT\tbeta\tU\tmu\tn\tn_err\tdelta_n\tNsite\tmu_dir\n"
    with open(out_tsv, "w") as f:
        f.write(header)
        for (n_target, T, beta, U, mu, n_mean, n_err, dn, nsite, mu_dir) in rows:
            f.write(
                f"{n_target:.12g}\t{T:.12g}\t{beta:.12g}\t{U:.12g}\t{mu:.12g}\t"
                f"{n_mean:.12g}\t{n_err:.12g}\t{dn:.12g}\t{nsite}\t{mu_dir}\n"
            )

    # Print brief summary
    n_total = len(rows)
    n_nan = sum((not np.isfinite(r[5])) for r in rows)
    print(f"[OK] wrote {out_tsv}")
    print(f"Total mu dirs: {n_total}")
    print(f"Rows with NaN n: {n_nan}")
    if n_bad:
        print(f"Rows failed to process (exceptions): {n_bad}")


# Entry point
if __name__ == "__main__":
    main()

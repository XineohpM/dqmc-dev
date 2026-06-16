import os, sys, glob, re
from pathlib import Path
import numpy as np
import argparse
import data_analysis as da
from scipy.interpolate import InterpolatedUnivariateSpline
utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))
import util

def load_dt(dir):
    try:
        dt, = util.load_firstfile(dir, "params/dt")
        return float(dt), "params/dt"
    except Exception:
        pass

    genlog = os.path.join(dir, "gen.log")
    with open(genlog, "r") as f:
        text = f.read()
    m = re.search(r"(?m)^dt\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", text)
    if m is None:
        raise ValueError(f"Could not find dt in params/dt or {genlog}")
    return float(m.group(1)), "gen.log"

def half_interval_norm(corr, dt, beta):
    """Match maxent.Preprocess(..., op_type='boson', sym=True)['norm']."""
    corr = np.asarray(corr)
    if corr.ndim == 1:
        mean = corr
    elif corr.ndim == 2:
        mean = corr.mean(axis=0)
    else:
        raise ValueError(f"correlator must be 1D or 2D, got shape {corr.shape}")

    L = mean.shape[0]
    mean_with_endpoint = np.concatenate((mean, mean[:1]))
    tau = np.arange(L + 1) * dt
    spl = InterpolatedUnivariateSpline(tau, mean_with_endpoint, ext=2, check_finite=True)
    return float(spl.integral(0, beta / 2))

def bootstrap_norm(corr, dt, beta, nboot, seed):
    rng = np.random.default_rng(seed)
    nbin = corr.shape[0]
    vals = np.empty(nboot, dtype=float)
    for i in range(nboot):
        resample = rng.integers(0, nbin, size=nbin)
        vals[i] = 4 * half_interval_norm(corr[resample], dt, beta)
    return vals

def load_optional_array(dir, name):
    file = os.path.join(dir, name)
    if os.path.exists(file):
        return np.load(file, allow_pickle=False)
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True,
                   help="Parent directory containing subfolders.")
    p.add_argument("--relpath_list", nargs="+", required=True,
                   help="List of relative paths.")
    p.add_argument("--correlator_name", default="JNJN_xx_perbin.npy",
                   help="Filename of the imaginary time current-current correlator.")
    p.add_argument("--rtol", type=float, default=1e-2,
                   help="Relative tolerance for the central-value sum-rule check.")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Number of bootstrap resamples for the correlator integral. Use 0 to disable.")
    p.add_argument("--seed", type=int, default=12345,
                   help="Random seed for bootstrap resampling.")
    args = p.parse_args()
    path = args.path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Base path not found or not a directory: {path}")

    def get_temperature(relpath):
        dir = os.path.join(path, relpath, "")
        beta, = util.load_firstfile(dir, "metadata/beta")
        return 1.0 / beta
    
    for relpath in sorted(args.relpath_list, key=get_temperature):
        dir = os.path.join(path, relpath, "")
        dt, dt_source = load_dt(dir)
        beta, = util.load_firstfile(dir, "metadata/beta")
        k, k_err = da.eqlt_meas_1(dir, ["kinetic"])
        k = k["kinetic"].real
        k_err = k_err["kinetic"].real
        corr = np.load(os.path.join(dir, args.correlator_name), allow_pickle=False)
        if corr.ndim != 2:
            raise ValueError(f"{args.correlator_name} must have shape (Nbin, L), got {corr.shape}")
        if np.iscomplexobj(corr):
            imag_max = np.max(np.abs(corr.imag))
            corr = corr.real
        else:
            imag_max = 0.0

        nbin, L = corr.shape
        beta_from_corr = L * dt
        norm4 = 4 * half_interval_norm(corr, dt, beta)
        rel_diff = (norm4 - (-k)) / (-k)

        boot_msg = "disabled"
        z_msg = "n/a"
        if args.bootstrap > 0:
            vals = bootstrap_norm(corr, dt, beta, args.bootstrap, args.seed)
            norm4_err = vals.std(ddof=1)
            boot_msg = f"{norm4:.10g} +/- {norm4_err:.3g}"
            combined_err = np.hypot(norm4_err, k_err)
            if combined_err > 0:
                z_msg = f"{abs(norm4 - (-k)) / combined_err:.2f}"

        mean_sign = load_optional_array(dir, "mean_sign_perbin.npy")
        sign_msg = "not found"
        if mean_sign is not None:
            sign_msg = (
                f"mean={np.mean(mean_sign):.6g}, "
                f"min={np.min(mean_sign):.6g}, max={np.max(mean_sign):.6g}"
            )

        print("T = ", 1.0/beta)
        print("dt = ", dt, f"({dt_source})")
        print("corr shape = ", corr.shape)
        print("beta from metadata = ", beta)
        print("L * dt = ", beta_from_corr)
        if not np.isclose(beta_from_corr, beta, rtol=1e-12, atol=1e-12):
            print("WARNING: L * dt does not match beta")
        if imag_max > 0:
            print("max imaginary part in correlator = ", imag_max)
        print("mean_sign_perbin = ", sign_msg)
        print("norm of correlator = ", norm4)
        print("norm bootstrap = ", boot_msg)
        print("kinetic energy = ", -k)
        print("kinetic energy error = ", k_err)
        print("relative difference = ", rel_diff)
        print("difference / combined error = ", z_msg)
        if np.isclose(norm4, -k, rtol=args.rtol, atol=0): print("norm of correlator = kinetic energy")
        else: print("norm of correlator and kinetic energy are not close")
        print("k/norm = ", -k/norm4)
        print(" ")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /home/users/phoenixm/scripts/specific_heat_bootstrap.py
import argparse, os
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt

def load_E(dirpath, fname="E_perfile.npy"):
    p = Path(dirpath)/fname
    if not p.exists():
        raise FileNotFoundError(str(p))
    arr = np.load(p)
    # drop NaNs and report
    mask = ~np.isnan(arr)
    return arr[mask]

def bootstrap_c_vs_T(dirs, nboot=1000, per_site=True, Nsites=None, seed=12345):
    rng = np.random.default_rng(seed)
    # read E arrays; require same ordering of dirs passed
    Ts = []
    E_lists = []
    for d in dirs:
        # assume directory name contains T or use metadata file
        meta_p = Path(d)
        # try to read metadata beta->T
        T = None
        # look for any file *.h5 to read metadata if needed (skip for speed)
        # assume dir basename like T_<value>
        try:
            bname = meta_p.name
            if bname.startswith("T_"):
                T = float(bname.split("_",1)[1])
        except Exception:
            T = None
        if T is None:
            raise SystemExit("Directory name must be T_<Tvalue> or modify script to read metadata")
        Earr = load_E(d)
        Ts.append(T)
        E_lists.append(Earr)
    # sort by T ascending (important for derivative)
    order = np.argsort(Ts)
    Ts = np.array(Ts)[order]
    E_lists = [E_lists[i] for i in order]

    # compute mean per-dir and stderr (from sample std / sqrt(Nfiles))
    E_mean = np.array([arr.mean() for arr in E_lists])
    E_se = np.array([arr.std(ddof=1)/max(1, len(arr))**0.5 for arr in E_lists])

    # do bootstrap over files: for each bootstrap sample, resample with replacement in each dir
    # produce E_mean_boot[iboot, i_dir]
    n_dirs = len(E_lists)
    E_mean_boot = np.zeros((nboot, n_dirs))
    for ib in range(nboot):
        for i, arr in enumerate(E_lists):
            if len(arr) == 0:
                E_mean_boot[ib, i] = np.nan
            else:
                sample = rng.choice(arr, size=len(arr), replace=True)
                E_mean_boot[ib, i] = sample.mean()
    # compute derivative dE/dT for each bootstrap sample using central differences
    # use np.gradient for nonuniform T
    c_boot = np.gradient(E_mean_boot, Ts, axis=1)  # shape (nboot, n_dirs)
    # convert to per-site if requested
    if per_site:
        if Nsites is None:
            raise SystemExit("Nsites must be provided to get per-site c")
        c_boot = c_boot / float(Nsites)

    # summarize
    c_mean = np.nanmean(c_boot, axis=0)
    c_p16 = np.nanpercentile(c_boot, 16, axis=0)
    c_p84 = np.nanpercentile(c_boot, 84, axis=0)
    # also return E_mean and E uncertainties
    return {
        "T": Ts,
        "E_mean": E_mean,
        "E_se": E_se,
        "c_mean": c_mean,
        "c_p16": c_p16,
        "c_p84": c_p84,
        "c_boot": c_boot
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Absolute path to base folder containing T_* subdirs")
    p.add_argument("--nboot", type=int, default=1000)
    p.add_argument("--Nsites", type=int, required=True, help="Total number of sites (e.g. 6*6=36)")
    p.add_argument("--outprefix", default="specific_heat", help="prefix for saved outputs")
    args = p.parse_args()

    root = Path(args.root)
    # find T_* dirs
    dirs = sorted([str(x) for x in root.iterdir() if x.is_dir() and x.name.startswith("T_")])
    if not dirs:
        raise SystemExit("No T_ directories found in " + str(root))
    res = bootstrap_c_vs_T(dirs, nboot=args.nboot, per_site=True, Nsites=args.Nsites)

    # save arrays
    np.save(root / (args.outprefix + "_T.npy"), res["T"])
    np.save(root / (args.outprefix + "_E_mean.npy"), res["E_mean"])
    np.save(root / (args.outprefix + "_E_se.npy"), res["E_se"])
    np.save(root / (args.outprefix + "_c_mean.npy"), res["c_mean"])
    np.save(root / (args.outprefix + "_c_p16.npy"), res["c_p16"])
    np.save(root / (args.outprefix + "_c_p84.npy"), res["c_p84"])
    np.save(root / (args.outprefix + "_c_boot.npy"), res["c_boot"])
    print("Saved results to", root)

    # quick plot
    import matplotlib.pyplot as plt
    T = res["T"]
    c = res["c_mean"]
    p16 = res["c_p16"]
    p84 = res["c_p84"]
    plt.figure(figsize=(6,4))
    plt.fill_between(T, p16, p84, alpha=0.3, label='16-84%')
    plt.plot(T, c, '-o', label='c(T)')
    plt.xscale('log')
    plt.xlabel('T')
    plt.ylabel('c per site')
    plt.legend()
    plt.tight_layout()
    outfig = root / (args.outprefix + "_c_vs_T.png")
    plt.savefig(outfig, dpi=150)
    print("Wrote plot:", outfig)

if __name__ == "__main__":
    main()
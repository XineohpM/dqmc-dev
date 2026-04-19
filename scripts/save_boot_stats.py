#!/usr/bin/env python3
# save_boot_stats.py
import numpy as np, os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--boot", required=True, help="path to bootstrap .npy (Nboot,L)")
parser.add_argument("--outdir", required=True, help="directory to save mean/cov/pct")
parser.add_argument("--reg", type=float, default=1e-8, help="regularization add to cov diagonal (if needed)")
args = parser.parse_args()

boot = np.load(args.boot)    # shape (Nboot, L)
if boot.ndim != 2:
    raise SystemExit("bootstrap array must be 2D (Nboot,L)")

Nboot, L = boot.shape
mean = np.mean(boot, axis=0)
pct16 = np.percentile(boot, 16, axis=0)
pct84 = np.percentile(boot, 84, axis=0)
# covariance across bootstrap samples: shape (L,L)
cov = np.cov(boot, rowvar=False)   # unbiased sample cov by default (Nboot-1 denom)

# ensure symmetric
cov = 0.5*(cov + cov.T)

# check eigenvals
eig = np.linalg.eigvalsh(cov)
min_eig = eig.min()
print("Nboot,L =", Nboot, L, "cov eig min =", min_eig)

# regularize if necessary
if min_eig <= 0:
    eps = max(args.reg, abs(min_eig)*1e-6, 1e-12)
    print("Regularizing cov by adding eps on diagonal:", eps)
    cov += np.eye(L)*eps

# save
os.makedirs(args.outdir, exist_ok=True)
np.save(os.path.join(args.outdir,"G_mean.npy"), mean)
np.save(os.path.join(args.outdir,"G_pct16.npy"), pct16)
np.save(os.path.join(args.outdir,"G_pct84.npy"), pct84)
np.save(os.path.join(args.outdir,"G_cov.npy"), cov)
print("Saved G_mean.npy, G_pct16.npy, G_pct84.npy, G_cov.npy in", args.outdir)
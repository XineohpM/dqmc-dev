#!/usr/bin/env python3
import os, sys, argparse
import numpy as np


def block_data(arr: np.ndarray, B: int) -> np.ndarray:
    """Blocking: average every B consecutive rows -> (M, L)"""
    if B <= 1: 
        return arr.copy()
    Nbin, L = arr.shape
    M = Nbin // B
    if M < 1:
        raise ValueError(f"B={B} too large for Nbin={Nbin}")
    arr_trunc = arr[: M * B]
    return arr_trunc.reshape(M, B, L).mean(axis=1)


# --------- Helper: autocorrelation and block size estimation ----------
def estimate_tau_int(x: np.ndarray) -> float:
    """Estimate integrated autocorrelation time tau_int for a 1D series x[b].

    Uses a simple initial-positive-sequence cutoff: sum ACF until it first becomes negative.
    Returns tau_int = 0.5 + sum_{k=1}^{k*} acf[k].
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    n = x.size
    if n < 3:
        return 0.0

    x = x - x.mean()
    var = x.var()
    if not np.isfinite(var) or var <= 0:
        return 0.0

    # Biased ACF via correlate; normalize so acf[0] = 1
    acf = np.correlate(x, x, mode='full')[n-1:] / (var * n)

    # Initial-positive-sequence cutoff
    cut = 1
    for k in range(1, n):
        if not np.isfinite(acf[k]) or acf[k] < 0:
            cut = k
            break
    tau_int = 0.5 + float(np.sum(acf[1:cut]))
    return max(0.0, tau_int)


def choose_block_size(perbin: np.ndarray, tau_idx: int, mult: float, min_M: int) -> tuple[int, float]:
    """Choose blocking size B from per-bin data by estimating tau_int at one tau index.

    Returns (B, tau_int).
    """
    Nbin, L = perbin.shape
    tau_idx = int(tau_idx)
    if tau_idx < 0:
        tau_idx = L + tau_idx
    tau_idx = max(0, min(L - 1, tau_idx))

    x = perbin[:, tau_idx]
    tau_int = estimate_tau_int(x)

    B = int(np.ceil(mult * tau_int))
    B = max(1, B)

    # Ensure we keep at least min_M effective samples
    if min_M is not None and min_M > 0:
        B_max = max(1, Nbin // int(min_M))
        B = min(B, B_max)
        B = max(1, B)

    return B, tau_int

def main():
    ap = argparse.ArgumentParser(
        description="Make bootstrap samples of G(tau) mean from per-bin data"
    )
    ap.add_argument("--dir",  required=True, help="directory containing the per-bin file")
    ap.add_argument("--file", required=True, help="filename of per-bin npy, e.g. jj_xx_perbin.npy")
    ap.add_argument("--nboot", type=int, default=1000, help="number of bootstrap replicates (default: 1000)")
    ap.add_argument("--block", type=int, default=1,
                    help="blocking size B (default: 1 = no blocking). Ignored if --auto_block is set.")
    ap.add_argument("--auto_block", action="store_true",
                    help=("If set, estimate an integrated autocorrelation time from the per-bin data and "
                          "choose a blocking size B automatically."))
    ap.add_argument("--tau_idx", type=int, default=None,
                    help=("Tau index used for autocorrelation estimate when --auto_block is set. "
                          "Default: L//2. You can pass negative indices (e.g. -1)."))
    ap.add_argument("--block_mult", type=float, default=2.0,
                    help=("When --auto_block is set, choose B = ceil(block_mult * tau_int). "
                          "Default: 2.0."))
    ap.add_argument("--min_M", type=int, default=20,
                    help=("When --auto_block is set, enforce at least this many effective samples M=Nbin//B. "
                          "Default: 20."))
    ap.add_argument("--seed",  type=int, default=2025, help="rng seed (default: 2025)")
    ap.add_argument("--outprefix", default="G_boot", help="output prefix (default: G_boot)")
    args = ap.parse_args()

    # Expand ~ and $HOME
    dirp = os.path.expanduser(os.path.expandvars(args.dir))
    fn   = os.path.join(dirp, args.file)

    if not os.path.isfile(fn):
        print("ERROR: per-bin file not found:", fn, file=sys.stderr)
        sys.exit(2)

    perbin = np.load(fn)               # shape (Nbin, L)
    if perbin.ndim != 2:
        print("ERROR: per-bin array must be 2D (Nbin, L). Got", perbin.shape, file=sys.stderr)
        sys.exit(2)

    Nbin, L = perbin.shape
    print(f"Loaded: {fn}  shape=(Nbin={Nbin}, L={L})")

    # Blocking (manual or auto)
    if args.auto_block:
        tau_idx = (L // 2) if (args.tau_idx is None) else int(args.tau_idx)
        B, tau_int = choose_block_size(perbin, tau_idx=tau_idx, mult=float(args.block_mult), min_M=int(args.min_M))
        print(f"Auto-block: tau_idx={tau_idx}, estimated tau_int={tau_int:.6g} -> chosen B={B}")
    else:
        B = int(args.block)

    data = block_data(perbin, B)       # shape (M, L)
    M, L2 = data.shape
    assert L2 == L
    print(f"After blocking: B={B}, effective samples M={M}")

    # Bootstrap
    Nboot = int(args.nboot)
    rng   = np.random.default_rng(args.seed)
    idx   = rng.integers(0, M, size=(Nboot, M))    # (Nboot, M)
    # Vectorized mean over axis=1 for each replicate
    boots = data[idx].mean(axis=1)                 # (Nboot, L)

    # Save
    outname = f"{args.outprefix}_N{Nboot}_B{B}_seed{args.seed}.npy"
    outpath = os.path.join(dirp, outname)
    np.save(outpath, boots)
    print("Saved:", outpath, "shape", boots.shape)

    # Quick summary
    mean   = boots.mean(axis=0)
    lo     = np.percentile(boots, 16, axis=0)
    hi     = np.percentile(boots, 84, axis=0)
    print("Preview: mean[:5] =", np.round(mean[:5], 6))
    print("Preview: 16-84% band widths (first 5 taus) =", np.round((hi - lo)[:5], 6))

if __name__ == "__main__":
    main()

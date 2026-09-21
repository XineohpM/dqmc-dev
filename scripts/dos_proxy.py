from __future__ import annotations

import os, argparse
import numpy as np
import paired_bootstrap

def _as_checked_real(arr: np.ndarray, name: str, imag_tol: float) -> tuple[np.ndarray, float]:
    """Return a real array after checking that any imaginary part is negligible."""
    arr = np.asarray(arr)
    imag_max = 0.0
    if np.iscomplexobj(arr):
        imag_max = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
        real_scale = float(np.max(np.abs(arr.real))) if arr.size else 0.0
        tol = max(float(imag_tol), float(imag_tol) * real_scale)
        if imag_max > tol:
            raise ValueError(
                f"{name} has non-negligible imaginary part: max |imag|={imag_max:g}, "
                f"tolerance={tol:g}"
            )
        arr = arr.real
    return np.asarray(arr, dtype=float), imag_max

def _load_corr(subpath: str, prefix: str, imag_tol: float):
    bundle_path = os.path.join(
        subpath,
        prefix + "1_particle_local_g_paired.npz",
    )
    bundle = paired_bootstrap.load_paired_bundle(bundle_path)
    if bundle.metadata["observable"] != "g":
        raise ValueError(
            f"{bundle_path} contains observable "
            f"{bundle.metadata['observable']!r}, expected 'g'"
        )
    corr_mean, imag_max = _as_checked_real(
        bundle.mean,
        f"{bundle_path}:mean",
        imag_tol,
    )
    return bundle, corr_mean, imag_max

def _validate_tau_matches_corr(tau: np.ndarray, corr_len: int, dpath: str) -> None:
    if tau.size != corr_len:
        raise ValueError(
            f"tau/correlator length mismatch in {dpath}: tau has L={tau.size}, "
            f"correlator has L={corr_len}. Bundle tau must use the no-endpoint grid."
        )
    if not np.isclose(tau[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"tau grid in {dpath} should start at 0, got tau[0]={tau[0]:g}")
    

def G_beta_over_2(corr: np.ndarray):
    L = corr.size
    mid = L // 2

    if not np.all(np.isfinite(corr)):
        raise ValueError("corr contains non-finite values")

    if L % 2 == 0:
        return float(corr[mid]), mid
    return float(0.5 * (corr[mid] + corr[mid + 1])), mid

def _bootstrap_proxies(
    T: float,
    numerator: np.ndarray,
    sign: np.ndarray,
    nboot: int,
    seed: int,
    imag_tol: float,
    block_size: int = 1,
) -> tuple[float, int]:
    if nboot <= 0:
        return (np.nan, 0)

    nbin = numerator.shape[0]
    indices = paired_bootstrap.bootstrap_indices(
        nbin,
        nboot,
        block_size=block_size,
        seed=seed,
    )
    estimates = paired_bootstrap.bootstrap_ratio_of_sums(
        numerator,
        sign,
        indices,
    )
    estimates, _ = _as_checked_real(
        estimates,
        "paired-bootstrap Green's functions",
        imag_tol,
    )
    dos_vals = np.asarray(
        [dos(T, G_beta_over_2(corr)[0]) for corr in estimates],
        dtype=float,
    )
    ngood = int(np.count_nonzero(np.isfinite(dos_vals)))
    dos_vals = dos_vals[np.isfinite(dos_vals)]
    if ngood == 0:
        return (np.nan, 0)
    dos_stderr = (
        float(np.std(dos_vals, ddof=1)) if ngood >= 2 else np.nan
    )
    return (
        dos_stderr,
        int(ngood),
    )

def dos(T: float, corr: float):
    """
    Proxy for DOS(omega = 0), dos(T) = G(beta/2)/T/pi
    """
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"bad T={T}")
    if not np.isfinite(corr):
        raise ValueError("G(beta/2) is non-finite")

    return corr/T/np.pi

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        required=True,
        help="Base directory containing T_* or beta_* subdirectories.",
    )
    p.add_argument("--items", nargs="+", required=True,
                    help=("List of items: each is 'relpath,prefix,T'. Example: "
                          "'T_0.2,,0.2'. Prefix is prepended to "
                          "1_particle_local_g_paired.npz."))
    # kept for forward compatibility (not used in proxy-only workflow)
    p.add_argument(
        "--output_path",
        required=True,
        help="Output file path (.npz) or output directory. Will be created if needed.",
    )
    p.add_argument("--nboot", type=int, default=1000,
                   help="Number of bootstrap resamples for proxy uncertainties. Use 0 to disable.")
    p.add_argument("--seed", type=int, default=12345,
                   help="Random seed for bootstrap resampling.")
    p.add_argument(
        "--bootstrap_block_size",
        type=int,
        default=1,
        help="Circular paired-bootstrap block size in source bins.",
    )
    p.add_argument("--imag_tol", type=float, default=1e-10,
                   help="Allowed absolute/relative imaginary-part tolerance for input arrays.")

    args = p.parse_args()

    base = os.path.expanduser(args.path)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Base path not found or not a directory: {base}")

    # Decide output file location.
    # --output_path can be either:
    #   (i) a directory, OR
    #   (ii) a file prefix (no extension), OR
    #   (iii) a full .npz path (treated as prefix by stripping .npz).
    out = os.path.expanduser(args.output_path)
    if out.endswith(".npz"):
        out_prefix = os.path.splitext(out)[0]
        out_dir = os.path.dirname(out_prefix) or "."
    else:
        # if it's an existing directory OR ends with a path separator, treat as directory
        if os.path.isdir(out) or out.endswith(os.sep):
            out_dir = out
            out_prefix = os.path.join(out_dir, "proxies")
        else:
            # treat as a prefix
            out_prefix = out
            out_dir = os.path.dirname(out_prefix) or "."

    os.makedirs(out_dir, exist_ok=True)

    rows = []
    warnings = []

    for item in args.items:
        parts = [s.strip() for s in item.split(",", 2)]
        if len(parts) != 3:
            raise ValueError(f"Bad --items entry {item!r}; expected 'relpath,prefix,T'")
        rel, pfx, Tstr = parts
        T = float(Tstr)
        dpath = os.path.join(base, rel)

        bundle, corr_mean, imag_max = _load_corr(
            dpath,
            pfx,
            args.imag_tol,
        )
        if imag_max > 0:
            warnings.append(
                f"[WARN] {dpath}: input correlator had small imaginary part "
                f"max |imag|={imag_max:g}; using real part."
            )

        tau = bundle.tau
        dt = float(bundle.metadata["dt"])
        beta = float(bundle.metadata["beta"])
        _validate_tau_matches_corr(tau, corr_mean.size, dpath)

        if beta > 0:
            T_from_beta = 1.0 / beta
            if abs(T - T_from_beta) / max(T, T_from_beta, 1e-12) > 0.02:
                warnings.append(
                    f"[WARN] {dpath}: T(from name)={T:g} differs from 1/beta={T_from_beta:g} (beta={beta:g})."
                )

        g_mid, mid = G_beta_over_2(corr_mean)

        # Mean-curve proxies (kept as auxiliary outputs)
        dos_mean = float(dos(T, g_mid))

        dos_stderr = np.nan
        ngood_boot = 0
        nbin = bundle.nbin

        if args.nboot > 0:
            boot_seed = int(args.seed) + len(rows)
            (
                dos_stderr,
                ngood_boot,
            ) = _bootstrap_proxies(
                T,
                bundle.numerator,
                bundle.sign,
                args.nboot,
                boot_seed,
                args.imag_tol,
                args.bootstrap_block_size,
            )
        else:
            warnings.append(
                f"[WARN] {dpath}: bootstrap disabled; "
                "uncertainty estimates are NaN"
            )

        rows.append(
            (
                float(T),
                float(beta),
                float(dt),
                float(g_mid),
                float(dos_mean),
                float(dos_stderr),
                int(nbin),
                int(ngood_boot),
                dpath,
            )
        )

    if not rows:
        msg = "No valid folders processed.\n"
        if warnings:
            msg += "\n".join(warnings[:20])
        raise RuntimeError(msg)

    # Sort by temperature
    rows = sorted(rows, key=lambda r: r[0])

    # Convert to arrays
    T_arr = np.array([r[0] for r in rows], dtype=float)
    beta_arr = np.array([r[1] for r in rows], dtype=float)
    dt_arr = np.array([r[2] for r in rows], dtype=float)
    g_mid_arr = np.array([r[3] for r in rows], dtype=float)
    dos_mean_arr = np.array([r[4] for r in rows], dtype=float)
    dos_stderr_arr = np.array([r[5] for r in rows], dtype=float)
    nbin_arr = np.array([r[6] for r in rows], dtype=int)
    ngood_boot_arr = np.array([r[7] for r in rows], dtype=int)
    folder_arr = np.array([r[8] for r in rows], dtype=object)

    # Write proxy-1 outputs
    out_npz_dos = out_prefix + "_dos_0freq.npz"
    np.savez(
        out_npz_dos,
        T=T_arr,
        beta=beta_arr,
        dt=dt_arr,
        G_mid=g_mid_arr,
        dos_mean=dos_mean_arr,
        dos_stderr=dos_stderr_arr,
        nbin=nbin_arr,
        nboot=np.full_like(nbin_arr, args.nboot),
        bootstrap_block_size=np.full_like(
            nbin_arr,
            args.bootstrap_block_size,
        ),
        ngood_boot=ngood_boot_arr,
        folder=folder_arr,
    )

    out_csv_dos = os.path.splitext(out_npz_dos)[0] + ".csv"
    with open(out_csv_dos, "w") as f:
        f.write(
            "T,beta,dt,G_mid,dos_mean,dos_stderr,"
            "nbin,nboot,bootstrap_block_size,ngood_boot,folder\n"
        )
        for i in range(T_arr.size):
            f.write(
                f"{T_arr[i]},{beta_arr[i]},{dt_arr[i]},{g_mid_arr[i]},"
                f"{dos_mean_arr[i]},{dos_stderr_arr[i]},"
                f"{nbin_arr[i]},{args.nboot},{args.bootstrap_block_size},"
                f"{ngood_boot_arr[i]},{repr(folder_arr[i])}\n"
            )

    print("Wrote", out_npz_dos)
    print("Wrote", out_csv_dos)
    print(f"Processed {len(rows)} temperature points under {base}")
    if warnings:
        print("\nWarnings (first 20):")
        for w in warnings[:20]:
            print(w)

if __name__ == "__main__":
    main()

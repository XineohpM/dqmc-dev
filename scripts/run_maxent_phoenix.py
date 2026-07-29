#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sign-aware CLI wrapper for :mod:`maxent`.

The input is a self-contained ``*_paired.npz`` bundle from a correlator
extractor.  The wrapper performs paired bootstrap sign reweighting, adapts the
resulting covariance to ``maxent.py``'s row interface in memory, and then runs
the requested number of outer spectrum bootstraps.

Example::

    python run_maxent_phoenix.py \
        --data_path /path/to/run \
        --data_file JNJN_xx_paired.npz \
        --omega_max 12 --n_omega 200 \
        --cov_bs 1000 --bootstrap_block_size 1 \
        --bs 50 --rnd_seed 12345 \
        --op_type boson --sym \
        --output_path /path/to/run/maxent_out

The output remains ``A_mean.npy``, ``s_all.npy``, ``omega.npy``,
``domega.npy``, and ``metadata.npy``.  Metadata distinguishes source bins,
MaxEnt covariance rows, and outer spectrum bootstrap samples.
"""
import maxent
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
import traceback
import paired_bootstrap


def _as_real_maxent_data(values, name):
        """Reject a physically significant imaginary part before MaxEnt."""

        values = np.asarray(values)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
        if np.iscomplexobj(values):
            scale = max(1.0, float(np.max(np.abs(values.real))))
            tolerance = 1e-12 + 1e-10 * scale
            max_imag = float(np.max(np.abs(values.imag)))
            if max_imag > tolerance:
                raise ValueError(
                    f"{name} has a non-negligible imaginary part "
                    f"(max |Im|={max_imag:.6g}, tolerance={tolerance:.6g}); "
                    "maxent.py only supports real correlators"
                )
            values = values.real
        return np.asarray(values, dtype=float)


def _validate_bundle_grid(bundle, bundle_path):
        metadata = bundle.metadata
        if "L" not in metadata:
            raise ValueError(f"Paired bundle {bundle_path} is missing metadata 'L'")
        L = int(metadata["L"])
        beta = float(metadata["beta"])
        dt = float(metadata["dt"])
        if L != bundle.ntau:
            raise ValueError(
                f"Paired bundle {bundle_path} has L={L} but "
                f"numerator has {bundle.ntau} tau points"
            )
        if not np.isclose(beta, dt * L, rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"Paired bundle {bundle_path} has inconsistent beta/dt/L: "
                f"{beta} != {dt}*{L}"
            )
        expected_tau = np.arange(L, dtype=float) * dt
        if not np.allclose(
            bundle.tau,
            expected_tau,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Paired bundle {bundle_path} tau does not equal arange(L)*dt"
            )


def _validate_append_alignment(bundle, append_bundle, append_path):
        if append_bundle.ntau != 1:
            raise ValueError(
                f"Append paired bundle {append_path} must contain one column, "
                f"got {append_bundle.ntau}"
            )
        if append_bundle.nbin != bundle.nbin:
            raise ValueError(
                f"Append paired bundle {append_path} has "
                f"{append_bundle.nbin} bins; expected {bundle.nbin}"
            )
        if not np.array_equal(append_bundle.n_sample, bundle.n_sample):
            raise ValueError("Append and correlator n_sample arrays are not aligned")
        if not np.allclose(
            append_bundle.sign,
            bundle.sign,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Append and correlator sign arrays are not aligned")
        if not np.isclose(
            float(append_bundle.metadata["beta"]),
            float(bundle.metadata["beta"]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Append and correlator beta metadata do not match")
        if not np.isclose(
            float(append_bundle.metadata["dt"]),
            float(bundle.metadata["dt"]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Append and correlator dt metadata do not match")
        if (
            bundle.source_files is None
            and append_bundle.source_files is not None
        ) or (
            bundle.source_files is not None
            and append_bundle.source_files is None
        ):
            raise ValueError(
                "Append and correlator bundles must both contain source_files "
                "or both omit it"
            )
        if (
            bundle.source_files is not None
            and not np.array_equal(
                append_bundle.source_files,
                bundle.source_files,
            )
        ):
            raise ValueError(
                "Append and correlator source_files arrays are not aligned"
            )


def prepare_paired_maxent_input(
        bundle_path,
        covariance_bootstrap_samples,
        *,
        rng,
        bootstrap_block_size=1,
        append_bundle_path=None,
):
        """Load paired data and construct MaxEnt covariance surrogate rows."""

        bundle = paired_bootstrap.load_paired_bundle(bundle_path)
        _validate_bundle_grid(bundle, bundle_path)
        if bundle.nbin < 2:
            raise ValueError("MaxEnt paired bootstrap requires at least two bins")
        if (
            not isinstance(
                covariance_bootstrap_samples,
                (int, np.integer),
            )
            or covariance_bootstrap_samples < 2
        ):
            raise ValueError(
                "covariance_bootstrap_samples must be an integer >= 2"
            )

        indices = paired_bootstrap.bootstrap_indices(
            bundle.nbin,
            covariance_bootstrap_samples,
            block_size=bootstrap_block_size,
            rng=rng,
        )
        estimates = paired_bootstrap.bootstrap_ratio_of_sums(
            bundle.numerator,
            bundle.sign,
            indices,
        )
        chi = paired_bootstrap.bootstrap_covariance_rows(
            estimates,
            bundle.mean,
        )
        chi = _as_real_maxent_data(chi, "correlator surrogate rows")

        append = None
        if append_bundle_path is not None:
            append_bundle = paired_bootstrap.load_paired_bundle(
                append_bundle_path
            )
            _validate_append_alignment(
                bundle,
                append_bundle,
                append_bundle_path,
            )
            append_estimates = paired_bootstrap.bootstrap_ratio_of_sums(
                append_bundle.numerator,
                append_bundle.sign,
                indices,
            )
            append = paired_bootstrap.bootstrap_covariance_rows(
                append_estimates,
                append_bundle.mean,
            )
            append = _as_real_maxent_data(
                append,
                "append surrogate rows",
            )

        metadata = {
            "dt": float(bundle.metadata["dt"]),
            "beta": float(bundle.metadata["beta"]),
            "L": int(bundle.metadata["L"]),
            "source_nbin": int(bundle.nbin),
            "maxent_nrow": int(chi.shape[0]),
            "covariance_bootstrap_samples": int(
                covariance_bootstrap_samples
            ),
            "bootstrap_block_size": int(bootstrap_block_size),
            "sign_reweighting": "paired_bootstrap_ratio_of_sums",
            "covariance_adapter": "center_plus_sqrt_R_delta",
            "source_bundle": os.path.abspath(os.fspath(bundle_path)),
        }
        for key in ("observable", "component", "spin", "normalization"):
            if key in bundle.metadata:
                metadata[key] = bundle.metadata[key]
        if append_bundle_path is not None:
            metadata["append_bundle"] = os.path.abspath(
                os.fspath(append_bundle_path)
            )
        return {
            "chi": chi,
            "append": append,
            "metadata": metadata,
            "indices": indices,
        }


def _preprocess_maxent_rows(
        rows,
        dt,
        beta,
        *,
        grid_info,
        op_type,
        sym,
        model_arr,
        append,
):
        """Preprocess rows while preserving the fermionic endpoint pairing.

        ``maxent.Preprocess`` constructs the fermionic tau=beta endpoint from
        an independently resampled tau=0 column.  That destroys the row-wise
        covariance carried by the surrogate rows.  Restore the exact
        same-row sum rule G(beta) = 1 - G(0) in the wrapper without changing
        the sign-unaware maxent implementation.
        """

        pre = maxent.Preprocess(
            rows,
            dt,
            beta,
            grid_info=grid_info,
            op_type=op_type,
            sym=sym,
            model_arr=model_arr,
            append=append,
        )
        if op_type == "fermion":
            lhs = np.array(pre["lhs"], copy=True)
            expected_shape = (rows.shape[0], rows.shape[1] + 1)
            if lhs.shape != expected_shape:
                raise ValueError(
                    "Fermionic preprocessing returned lhs shape "
                    f"{lhs.shape}; expected {expected_shape}"
                )
            lhs[:, -1] = 1.0 - lhs[:, 0]
            pre["lhs"] = lhs
        return pre


#Adapted from Emily's run_maxent.py code
def perform_maxent(chi,  omega_grid, metadata, 
                   append=None, alpha_arr=np.logspace(1,9,1+20*(9-1)),
                   bs=1, anneal_arr = None, checks=False, printout=False, op_type='boson', sym=True, 
                   rng=None,
                   **mkwargs):
        """Performs MaxEnt on correlations of the form O(tau)O^{dagger}. Wrapper for maxent module. 
        Args: 
            chi: (Nrow,L) covariance-surrogate rows produced from a paired bundle
            omega_grid: tuple containing (omega, domega) arrays for the frequency grid
            metadata: dictionary with metadata including "dt", "beta", and "L"
        Keyword Args:
            bs: number of bootstrap samples to perform
            append: (Nbin,1) array to append as the tau=beta component of G
            anneal_arr: initial model array to use in maxent. If None, uses flat model.
            checks: if True, plots bootstrap results for visual inspection
            op_type: 'boson' or 'fermion' kernel type
            sym: if True, uses bosonic kernel symmetrized about omega = 0
            mkwargs: additional keyword arguments to pass to maxent.MaxEnt function

        Returns:
            dict with keys:
                "A": (N_omega,) bootstrap-mean of A(omega_i) * domega_i  (weights; sum ≈ 1).
                "s": (bs, N_omega) per-bootstrap physical spectrum; s[k] is from the k-th resample.
                     If bs == 1, shape is (1, N_omega); use s.squeeze(0) to get (N_omega,).
        """
        
        dt = metadata["dt"]
        beta = metadata["beta"]
        chi = _as_real_maxent_data(chi, "MaxEnt input rows")
        if chi.ndim != 2:
            raise ValueError(
                f"MaxEnt input rows must be 2D, got shape {chi.shape}"
            )
        nbin, L = chi.shape
        if nbin < 2:
            raise ValueError("MaxEnt covariance requires at least two rows")
        if int(metadata["L"]) != L:
            raise ValueError(
                f"metadata L={metadata['L']} does not match input L={L}"
            )
        if not isinstance(bs, (int, np.integer)) or bs < 1:
            raise ValueError("bs must be an integer >= 1")
        if rng is None:
            rng = np.random.default_rng()

        if op_type == "boson":
            if sym and (append is None):
                append = np.zeros((nbin,1), dtype=float)
            elif (not sym) and (append is None):
                raise ValueError("Must provide append array at tau=beta with correct symmetries for nonsymmetric bosonic kernel")
        if append is not None:
            append = _as_real_maxent_data(append, "append rows")
            if append.shape != (nbin, 1):
                raise ValueError(
                    f"append must have shape {(nbin, 1)}, got {append.shape}"
                )

        # drop last row/column for bosonic non-sym kernel after preprocessing for maxent
        drop = True if not sym and op_type == "boson" else False 
        omega, domega = omega_grid
        nw = omega.shape[0] 

        # default to flat model 
        if anneal_arr is None:
            anneal_arr = maxent.model_flat(domega)
        
        s_bs = np.full((bs,nw),np.nan,dtype=float)
        A_bs = np.full((bs,nw),np.nan,dtype=float)

        for i in tqdm(range(bs)): # progress bar looping over bootstraps
            try:
                resample = rng.integers(nbin, size=nbin)
                append_resampled = None if append is None else append[resample]

                # preprocess data for maxent
                pre = _preprocess_maxent_rows(
                    chi[resample],
                    dt,
                    beta,
                    grid_info=(omega, domega),
                    op_type=op_type,
                    sym=sym,
                    model_arr=anneal_arr,
                    append=append_resampled,
                )
                
                # drop extra datapoint if nonsymmetric bosonic kernel
                if drop:
                    pre["tau"] = pre["tau"][:-1]
                    pre["lhs"] = pre["lhs"][:,:-1]
                    pre["K"] = pre["K"][:-1,:]

                #  best estimate of A(omega_i) *    domega_i
                A = maxent.MaxEnt(pre, printout=printout, alpha_arr=alpha_arr, **mkwargs)
                s = (A/domega)*pre["norm"]*np.pi 
                A_bs[i,:] = A
                s_bs[i,:] = s
            except:
                # Maxent failed, fill with NaNs
                A_bs[i,:] = np.NaN
                s_bs[i,:] = np.NaN
                traceback.print_exc()

        if checks: 
            L = metadata["L"]
            plt.figure()
            plt.ylabel(r"$L(\omega)$ bootstrap")
            plt.plot(omega,s_bs.T,lw=1,color='k')

            plt.figure()
            plt.ylabel("raw maxent output bootstrap")
            plt.plot(omega, A_bs.T,lw=1,color='k')

            plt.figure()
            plt.ylabel("imaginary time data reproduction bootstrap")
            #note: errorbar is += 1 std error of mean
            plt.errorbar(np.arange(L)*dt,chi.mean(0),\
                yerr = np.std(chi, axis=0,ddof=1)/np.sqrt(nbin),fmt='s',label="data")
            for i in range(bs):
                plt.plot(pre["tau"], pre["K"] @ A_bs[i,:] * pre["norm"],lw=1,color='k')
            plt.legend(loc='best')
            plt.show()

        return { "A": np.nanmean(A_bs,axis=0), "s": s_bs}

def build_grid(op_type: str, sym: bool, n_omega: int,
               omega_max: float, grid: str, a: float =1.0,
               b: float =1.0):
        """
        Build (omega, domega) frequency grid using maxent.gen_grid()
        Args:
            op_type: "boson" / "fermion"
            sym: symmetric or not
            n_omega: number of points on the frequency axis
            omega_max: maximum frequency
            grid:
                - "linear" : omega = x
                - "sinh"   : omega = a*sinh(b*x)
        Returns:
            omega, domega
        """
        if grid == "linear":
            #Symmetric bosonic case: [0, omega_max]
            if op_type == "boson" and sym:
                return maxent.gen_grid(int(n_omega), 0.0, float(omega_max), lambda x: x)
            #Nonsymmetric case: [-omega_max, omega_max]
            else:
                return maxent.gen_grid(int(n_omega), -float(omega_max), float(omega_max), lambda x: x)
            
        elif grid == "sinh":
            assert a > 0 and b > 0
            #omega_max = a*sinh(b*x_max), x_max = (1/b)*arcsinh(omega_max/a)
            x_max = float(np.arcsinh(float(omega_max) / float(a)) / float(b))
            if op_type == "boson" and sym:
                return maxent.gen_grid(int(n_omega), 0.0, float(x_max), lambda x: a*np.sinh(b*x))
            else:
                return maxent.gen_grid(int(n_omega), -float(x_max), float(x_max), lambda x: a*np.sinh(b*x))
            
        else: raise ValueError(f"Unknown grid type: {grid}, grid type should be either \"linear\" or \"sinh\"")

def _parse_args():
        p = argparse.ArgumentParser(description="CLI wrapper of MaxEnt")
        p.add_argument("--data_path", type=str, required=True,
                    help="Path containing the imaginary time DQMC data.")
        p.add_argument("--data_file", type = str, required=True,
                    help="Filename of the self-contained *_paired.npz correlator bundle.")
        p.add_argument("--omega_max", type=float, required=True,
                    help="Maximum frequency.")
        p.add_argument("--n_omega", type=int, required=True,
                    help="Number of points on the frequency axis.")
        p.add_argument("--bs", type=int, required=True,
                    help="Number of outer MaxEnt spectrum bootstrap samples.")
        p.add_argument(
                    "--cov_bs",
                    type=int,
                    default=None,
                    help="Number of paired-bootstrap estimates used to construct MaxEnt covariance rows (default: --bs).")
        p.add_argument(
                    "--bootstrap_block_size",
                    type=int,
                    default=1,
                    help="Circular paired-bootstrap block size in source bins (default: 1).")
        p.add_argument("--op_type", choices=["boson", "fermion"], default="boson",
                    help="Kernel/operator type, choose between \"boson\" and \"fermion\".")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--sym",    dest="sym", action="store_true",
                    help="Use symmetric bosonic kernel (ignored for fermion).")
        g.add_argument("--nonsym", dest="sym", action="store_false",
                    help="Disable symmetric bosonic kernel.")
        p.set_defaults(sym=False)
        p.add_argument("--append", type=str,
                    help="Path to the aligned one-column paired bundle for tau=beta in the nonsymmetric bosonic case.")
        p.add_argument("--model", type=str,
                    help="Path to default model, which should be a (N_omega, 1) array.")
        p.add_argument("--method", choices=["classic", "bryan", "BT"], default="BT",
                    help="Alpha selection method, defualt BT.")
        p.add_argument("--alpha_min", type=float, default=1,
                    help="Base-10 exponent for the alpha scan lower bound; alpha_min=10**alpha_min.")
        p.add_argument("--alpha_max", type=float, default=9,
                    help="Base-10 exponent for the alpha scan upper bound; alpha_max=10**alpha_max.")
        p.add_argument("--alpha_pts", type=int, default=161,
                    help="Number of log-spaced alpha points between 10**alpha_min and 10**alpha_max.")
        p.add_argument("--output_path", type=str,
                    help="Path to write output files.")
        p.add_argument("--output_prefix", type=str,
                    help="Common filename prefix of the output files.")
        p.add_argument("--grid", choices=["linear", "sinh"], default="linear",
                    help="Frequency grid type, choose between \"linear\" and \"sinh\".")
        p.add_argument("--a", type=float, default=1.0,
                    help="Coefficient a in omega = a*sinh(b*x).")
        p.add_argument("--b", type=float, default=1.0,
                    help="Coefficient b in omega = a*sinh(b*x).")
        p.add_argument("--checks", action="store_true",
                    help="Plot bootstrap reconstructions and diagnostics.")
        p.add_argument("--printout", action="store_true",
                    help="Print bootstrap diagnostics.")
        p.add_argument("--rnd_seed", type=int,
                    help="Seed for bootstrap resampling RNG.")
        return p.parse_args()

def main():
        args = _parse_args()

        op_type = args.op_type
        sym = args.sym
        if op_type == "fermion" and sym:
            print("[INFO] --sym is ignored for fermion.", file=sys.stderr)
            sym = False

        input_path = os.path.join(args.data_path, args.data_file)
        append_path = None
        if (op_type == "boson") and (not sym):
            if not args.append:
                raise ValueError(
                    "--append paired bundle is required for the "
                    "nonsymmetric bosonic case."
                )
            append_path = (
                args.append
                if os.path.isabs(args.append)
                else os.path.join(args.data_path, args.append)
            )
        elif args.append:
            raise ValueError(
                "--append is only valid for a nonsymmetric bosonic kernel"
            )

        seed_sequence = np.random.SeedSequence(args.rnd_seed)
        covariance_seed, spectrum_seed = seed_sequence.spawn(2)
        covariance_rng = np.random.default_rng(covariance_seed)
        spectrum_rng = np.random.default_rng(spectrum_seed)
        if args.rnd_seed is not None:
            # maxent.Preprocess currently uses NumPy's legacy global RNG for
            # the fermionic tau=beta endpoint.
            np.random.seed(int(args.rnd_seed))

        cov_bs = int(args.bs) if args.cov_bs is None else int(args.cov_bs)
        prepared = prepare_paired_maxent_input(
            input_path,
            cov_bs,
            rng=covariance_rng,
            bootstrap_block_size=int(args.bootstrap_block_size),
            append_bundle_path=append_path,
        )
        chi = prepared["chi"]
        append = prepared["append"]
        metadata = prepared["metadata"]
        metadata.update(
            {
                "bootstrap_seed": args.rnd_seed,
                "spectrum_bootstrap_samples": int(args.bs),
            }
        )

        grid = args.grid
        if grid == "linear":
            omega, domega = build_grid(op_type, sym, int(args.n_omega), float(args.omega_max), grid)
        elif grid == "sinh":
            omega, domega = build_grid(op_type, sym, int(args.n_omega), float(args.omega_max), grid, float(args.a), float(args.b))
        else: raise ValueError(f"Unknown grid type: {grid}, grid type should be either \"linear\" or \"sinh\"")

        if args.alpha_pts < 2:
            raise ValueError("--alpha_pts must be at least 2.")
        if args.alpha_max <= args.alpha_min:
            raise ValueError("--alpha_max must be greater than --alpha_min.")
            
        alpha_arr = np.logspace(args.alpha_min, args.alpha_max, args.alpha_pts)

        anneal_model = None
        if args.model: 
            anneal_model = np.load(args.model, allow_pickle=False)
            if anneal_model.ndim != 1 or anneal_model.shape[0] != omega.shape[0]:
                raise ValueError(f"--model length {anneal_model.shape} must match N_omega={omega.shape[0]}.")
        
        # Collect optional MaxEnt kwargs
        mkwargs = {"method": args.method}

        # Run
        results = perform_maxent(
            chi=chi,
            omega_grid=(omega, domega),
            metadata=metadata,
            append=append,
            alpha_arr=alpha_arr,
            bs=int(args.bs),
            anneal_arr=anneal_model,
            checks=args.checks,
            printout=args.printout,
            op_type=op_type,
            sym=sym,
            rng=spectrum_rng,
            **mkwargs
        )

        # Save
        if args.output_path:
            out_dir = args.output_path
            os.makedirs(out_dir, exist_ok=True)
            prefix = args.output_prefix or ""
            outputs = {
                "A_mean": results["A"],
                "s_all": results["s"],
                "omega": omega,
                "domega": domega,
                "metadata": np.array(metadata, dtype=object),
            }
            for name, arr in outputs.items():
                fname = f"{prefix}{name}.npy"
                fpath = os.path.join(out_dir, fname)
                # Note: metadata will be saved as an object array; loading it requires allow_pickle=True
                np.save(fpath, arr)
                print(f"[OK] Saved {name} -> {fpath}")
        else: print("[INFO] No --output_path provided; results not saved.", file=sys.stderr)

if __name__ == "__main__": main()

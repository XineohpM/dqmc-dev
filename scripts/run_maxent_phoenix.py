#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_maxent_phoenix.py: A lightweight CLI wrapper of MaxEnt so you can pass data/params from the shell.

Usage examples:

1) Quick start — Bosonic, symmetric kernel, linear grid (positive frequencies)
    python run_maxent_phoenix.py \
        --data_path /path/to/run \
        --data_file jj_xx.npy \
        --beta 5.0 --dt 0.05 \
        --omega_max 12.0 --n_omega 200 \
        --bs 50 --rnd_seed 12345 \
        --op_type boson --sym \
        --grid linear \
        --output_path /path/to/outdir/ --output_prefix boson_sym_

2) Bosonic, symmetric kernel, sinh grid
    python run_maxent_phoenix.py \
        --data_path /path/to/run \
        --data_file jj_xx.npy \
        --beta 5.0 --dt 0.05 \
        --omega_max 12.0 --n_omega 200 \
        --bs 50 --rnd_seed 12345 \
        --op_type boson --sym \
        --grid sinh --a 0.4 --b 2.5 \
        --output_path /path/to/outdir/ --output_prefix boson_sym_sinh_

3) Bosonic, nonsymmetric kernel
    python run_maxent_phoenix.py \
        --data_path /path/to/run \
        --data_file jj_xx.npy \
        --beta 5.0 --dt 0.05 \
        --omega_max 12.0 --n_omega 200 \
        --bs 50 --rnd_seed 12345 \
        --op_type boson --nonsym \
        --append /path/to/chi_beta_column.npy \
        --grid linear \
        --output_path /path/to/outdir/ --output_prefix boson_nonsym_

4) Fermionic kernel (sym ignored; symmetric frequency grid)
    python run_maxent_phoenix.py \
        --data_path /path/to/run \
        --data_file jj_xx.npy \
        --beta 5.0 --dt 0.05 \
        --omega_max 12.0 --n_omega 200 \
        --bs 1 --rnd_seed 42 \
        --op_type fermion \
        --grid linear \
        --output_path /path/to/outdir/ --output_prefix fermion_

5) HPC cluster (SLURM sbatch) example:
    sbatch --partition=owners --cpus-per-task=2 --mem=64G --time=10:00:00 --requeue \
            --mail-type=FAIL,END --mail-user=you@stanford.edu \
            --export=ALL,OMP_NUM_THREADS=2 \
            --chdir=/home/users/you/dqmc_runs \
            --output=slurm-maxent-%j.out \
            --wrap="source ~/miniconda3/etc/profile.d/conda.sh && conda activate dqmc && \
                    python3 /home/users/you/scripts/run_maxent_phoenix.py \
                    --data_path /home/users/you/dqmc_runs \
                    --data_file jj_xx.npy \
                    --beta 5.0 --dt 0.05 \
                    --omega_max 12.0 --n_omega 100 \
                    --bs 200 --rnd_seed 12345 \
                    --op_type boson --sym \
                    --grid linear \
                    --output_path /home/users/you/dqmc_runs/maxent_out \
                    --output_prefix maxent_"

Outputs:
    The script writes separate .npy files into --output_path (created if missing), using --output_prefix:
    <prefix>A_mean.npy   # bootstrap mean of A(ω_i)·Δω_i, shape (N_ω,)
    <prefix>s_all.npy    # all bootstrap spectra, shape (bs, N_ω)
    <prefix>omega.npy    # frequency grid points, shape (N_ω,)
    <prefix>domega.npy   # frequency bin widths, shape (N_ω,)
    <prefix>metadata.npy # dict with dt, beta, L, nbin  (load with allow_pickle=True)
"""
import maxent
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
import traceback

#Adapted from Emily's run_maxent.py code
def perform_maxent(chi,  omega_grid, metadata, 
                   append=None,
                   bs=1, anneal_arr = None, checks=False, op_type='boson', sym=True, 
                   **mkwargs):
        """Performs MaxEnt on correlations of the form O(tau)O^{dagger}. Wrapper for maxent module. 
        Args: 
            chi: (Nbin,L) lhs of the imaginary time data to invert
            omega_grid: tuple containing (omega, domega) arrays for the frequency grid
            metadata: dictionary with metadata including "dt", "beta", "L", "nbin"
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
        nbin = metadata["nbin"]

        if op_type == "boson":
            if sym and (append is None):
                append = np.zeros((nbin,1), dtype=float)
            elif (not sym) and (append is None):
                raise ValueError("Must provide append array at tau=beta with correct symmetries for nonsymmetric bosonic kernel")

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
                resample = np.random.randint(nbin,size=nbin) #sample with replacement
                append_resampled = None if append is None else append[resample]

                # preprocess data for maxent
                pre = maxent.Preprocess(chi[resample], dt, beta, grid_info = (omega,domega),
                                        op_type = op_type, sym=sym, model_arr = anneal_arr, append=append_resampled)
                
                # drop extra datapoint if nonsymmetric bosonic kernel
                if drop:
                    pre["tau"] = pre["tau"][:-1]
                    pre["lhs"] = pre["lhs"][:,:-1]
                    pre["K"] = pre["K"][:-1,:]

                #  best estimate of A(omega_i) *    domega_i
                A = maxent.MaxEnt(pre, **mkwargs)
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
                    help="Filname of the imaginary time DQMC data, which should be a (N_bin, L) matrix.")
        p.add_argument("--beta", type=float, required=True,
                    help="Inverse temperature beta = 1/T.")
        p.add_argument("--dt", type=float, required=True,
                    help="Imaginary time step dt = beta/L.")
        p.add_argument("--omega_max", type=float, required=True,
                    help="Maximum frequency.")
        p.add_argument("--n_omega", type=int, required=True,
                    help="Number of points on the frequency axis.")
        p.add_argument("--bs", type=int, required=True,
                    help="Number of bootstrap samples to perform.")
        p.add_argument("--op_type", choices=["boson", "fermion"], default="boson",
                    help="Kernel/operator type, choose between \"boson\" and \"fermion\".")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--sym",    dest="sym", action="store_true",
                    help="Use symmetric bosonic kernel (ignored for fermion).")
        g.add_argument("--nonsym", dest="sym", action="store_false",
                    help="Disable symmetric bosonic kernel.")
        p.set_defaults(sym=False)
        p.add_argument("--append", type=str,
                    help="Path to tau=beta column for nonsymmetric bosonic kernel, which should be a (N_bin, 1) array.")
        p.add_argument("--model", type=str,
                    help="Path to default model, which should be a (N_omega, 1) array.")
        p.add_argument("--method", choices=["classic", "bryan", "BT"], default="BT",
                    help="Alpha selection method, defualt BT.")
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
        p.add_argument("--rnd_seed", type=int,
                    help="Seed for bootstrap resampling RNG.")
        return p.parse_args()

def main():
        args = _parse_args()
        
        if args.rnd_seed is not None: np.random.seed(int(args.rnd_seed))

        input_path = os.path.join(args.data_path, args.data_file)
        chi = np.load(input_path, allow_pickle=False)
        if chi.ndim != 2: raise ValueError(f"--chi must be 2D (N_bin, L) matrix. Got shape {chi.shape}.")

        nbin, L = chi.shape
        beta = float(args.beta)
        dt = float(args.dt)
        if not np.isclose(beta, dt*L):
            print(f"[WARN] beta != dt*L ({beta} vs {dt}*{L}={dt*L}). Proceeding anyway.", file=sys.stderr)
        metadata = {"dt": dt, "beta": beta, "L": L, "nbin": int(nbin)}

        op_type = args.op_type
        sym = args.sym
        if op_type == "fermion" and sym:
            print("[INFO] --sym is ignored for fermion.", file=sys.stderr)
            sym = False
        
        grid = args.grid
        if grid == "linear":
            omega, domega = build_grid(op_type, sym, int(args.n_omega), float(args.omega_max), grid)
        elif grid == "sinh":
            omega, domega = build_grid(op_type, sym, int(args.n_omega), float(args.omega_max), grid, float(args.a), float(args.b))
        else: raise ValueError(f"Unknown grid type: {grid}, grid type should be either \"linear\" or \"sinh\"")

        anneal_model = None
        if args.model: 
            anneal_model = np.load(args.model, allow_pickle=False)
            if anneal_model.ndim != 1 or anneal_model.shape[0] != omega.shape[0]:
                raise ValueError(f"--model length {anneal_model.shape} must match N_omega={omega.shape[0]}.")
        
        append = None
        if (op_type == "boson") and (not sym):
            if not args.append:
                raise ValueError("--append is required for nonsymmetric bosonic case.")
            append = np.load(args.append, allow_pickle=False)
            append = np.asarray(append)
            if append.ndim == 1: append = append.reshape(-1, 1)
            if append.shape != (nbin, 1): raise ValueError(f"--append must have shape (N_bin, 1), got {append.shape}, expected {(nbin, 1)}.")
        
        # Collect optional MaxEnt kwargs
        mkwargs = {"method": args.method}

        # Run
        results = perform_maxent(
            chi=chi,
            omega_grid=(omega, domega),
            metadata=metadata,
            append=append,
            bs=int(args.bs),
            anneal_arr=anneal_model,
            checks=args.checks,
            op_type=op_type,
            sym=sym,
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
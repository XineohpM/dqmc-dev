"""
run_maxent_anneal.py: A lightweight CLI wrapper for temperature-annealed MaxEnt, so you can pass a sequence of runs/data from the shell and use the previous temperature's spectrum as the next temperature's default model.

Usage examples:

1) Quick start — Bosonic, symmetric kernel, linear grid, annealing from high T to low T
    python run_maxent_anneal.py \
        --base /path/to/dqmc_runs \
        --items T_0.5,0.5 T_0.4,0.4 T_0.333333,0.333333 \
        --data_file jj_xx.npy \
        --omega_max 12.0 --n_omega 200 \
        --bs 50 --rnd_seed 12345 \
        --op_type boson --sym \
        --grid linear \
        --output_relpath maxent_out --output_prefix anneal_

2) Bosonic, symmetric kernel, sinh grid
    python run_maxent_anneal.py \
        --base /path/to/dqmc_runs \
        --items T_0.5,0.5 T_0.4,0.4 T_0.333333,0.333333 \
        --data_file jj_xx.npy \
        --omega_max 12.0 --n_omega 200 \
        --bs 50 --rnd_seed 12345 \
        --op_type boson --sym \
        --grid sinh --a 0.4 --b 2.5 \
        --output_relpath maxent_out --output_prefix anneal_sinh_

3) HPC cluster (SLURM sbatch) example:
    sbatch --partition=owners --cpus-per-task=2 --mem=64G --time=48:00:00 --requeue \
            --mail-type=FAIL,END --mail-user=phoenixm@stanford.edu \
            --export=ALL,OMP_NUM_THREADS=2 \
            --chdir=/scratch/users/phoenixm/dqmc_runs/U-6_n6x6_halffilling_anneal_03172026/ \
            --output=slurm-maxent-anneal-%j.out \
            --wrap='source ~/miniconda3/etc/profile.d/conda.sh && conda activate dqmc && \
                    python3 /home/users/phoenixm/scripts/run_maxent_anneal.py \
                    --base /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_halffilling_anneal_03172026/ \
                    --highT_model 0.031390990449147387,0.03122084162436454,0.030885620037054733,0.03039512976604581,0.029763250715573755,0.029007061379163555,0.028145835922722406,0.02720001803387754,0.02619026174937288,0.025136606739444844,0.02405782828654851,0.02297097590452262,0.02189109303259006,0.020831095356144615,0.01980177717472035,0.018811912819351663,0.017868421836172675,0.016976570867890603,0.016140190496558727,0.015361890752875154,0.014643263920911203,0.013985067358574,0.013387382224958702,0.012849746309921592,0.012371260728354762,0.01195067123127143,0.011586425459513092,0.011276707773301038,0.011019453464626234,0.010812344315142908,0.01065278769966539,0.010537881839011096,0.010464370441686688,0.010428590880800176,0.010426421228128405,0.010453232846924038,0.01050385667839537,0.010572572587179432,0.010653131789930602,0.010738822022031412,0.010822583226328434,0.010897177807017598,0.010955413788373678,0.010990411907537682,0.01099589966767323,0.010966508120152374,0.010898042350426278,0.010787695872744677,0.010634183273989618,0.010437774271015731,0.010200224440625042,0.009924610925072644,0.009615092817277357,0.00927662356374281,0.008914645490234855,0.008534794568893018,0.008142637923902114,0.007743458972980542,0.007342097202678149,0.006942842686871465,0.006549380338325266,0.006164775751687326,0.005791493168302923,0.005431436158822659,0.005086002601818153,0.004756147000577332,0.00444244478858743,0.0041451548008807956,0.0038642774065445738,0.003599606855983276,0.0033507771940094783,0.0031173016543356567,0.0028986058232018668,0.00269405508277894,0.0025029769581964307,0.0023246790288562534,0.0021584630514194813,0.0020036358982539965,0.0018595178553718415,0.0017254487576640475,0.0016007923728543654,0.0014849393828567118,0.0013773092542107002,0.001277351238888693,0.0011845447031677738,0.0010983989451369177,0.001018452630209994,0.0009442729480727896,0.0008754545731134506,0.0008116184929035074,0.0007524107550972587,0.000697501171655102,0.0006465820100966148,0.0005993666941457587,0.0005555885303011631,0.0005149994722629554,0.0004773689315375165,0.0004424826397281163,0.00041014156584381873,0.00038016089029189624 \
                    --items \
                        T_8,8.0 \
                        T_7.00001,7.0 \
                        T_5.99999,6.0 \
                        T_5,5.0 \
                        T_4,4.0 \
                        T_3,3.0 \
                        T_2,2.0 \
                        T_1,1.0 \
                        T_0.666667,0.666667 \
                        T_0.5,0.5 \
                        T_0.4,0.4 \
                        T_0.333333,0.333333 \
                        T_0.285714,0.285714 \
                        T_0.25,0.25 \
                        T_0.222222,0.222222 \
                        T_0.2,0.2 \
                    --data_file JNJN_xx_perbin.npy \
                    --omega_max 12.0 --n_omega 100 \
                    --bs 200 --rnd_seed 12345 \
                    --op_type boson --sym \
                    --grid linear \
                    --output_relpath maxent_out \
                    --output_prefix anneal_'

Outputs:
    The script writes separate .npy files into <item_dir>/--output_relpath (created if missing), using --output_prefix:
    <prefix>A_mean.npy   # bootstrap mean of A(ω_i)·Δω_i, shape (N_ω,)
    <prefix>s_all.npy    # all bootstrap spectra, shape (bs, N_ω)
    <prefix>omega.npy    # frequency grid points, shape (N_ω,)
    <prefix>domega.npy   # frequency bin widths, shape (N_ω,)
    <prefix>metadata.npy # dict with dt, beta, L, nbin (load with allow_pickle=True)
"""
import os, argparse, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import maxent
import run_maxent_phoenix
from pathlib import Path

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))

import util

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                    help="Base directory that contains the temperature subdirectories listed in --items.")
    p.add_argument("--items", nargs="+", required=True,
                    help=("List of items. Each item is either 'relpath,T' or "
                        "'relpath,T,alpha_min,alpha_max' or "
                        "'relpath,T,alpha_min,alpha_max,alpha_pts'."))
    p.add_argument("--data_file", type = str, required=True,
                    help="Filname of the imaginary time correlator, (N_bin, L)")
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
                help="Path to tau=beta column for nonsymmetric bosonic kernel, (N_bin, 1)")
    model_g = p.add_mutually_exclusive_group()
    model_g.add_argument("--highT_model", type=str,
                help="Comma-separated spectral func array for annealing the highest T.")
    model_g.add_argument("--lowT_gap", type=float,
                help="If set, use gapped model for the lowest T and do inverse annealing." \
                "Input the width of the gap.")
    p.add_argument("--method", choices=["classic", "bryan", "BT"], default="BT",
                help="Alpha selection method, defualt BT.")
    p.add_argument("--output_relpath", type=str,
                help="Relative path to write output files.")
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
    
    args = p.parse_args()
    path = args.base
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Base path not found or not a directory: {path}")

    T_arr = []
    dpath_arr = []
    #dt_arr = []
    alpha_min_arr = []
    alpha_max_arr = []
    alpha_pts_arr = []

    for item in args.items:
        parts = item.split(",")
        if len(parts) == 2:
            rel, Tstr = parts
            alpha_min = 1
            alpha_max = 9
            alpha_pts = 161
        elif len(parts) == 4:
            rel, Tstr, amin_str, amax_str = parts
            alpha_min = float(amin_str)
            alpha_max = float(amax_str)
            alpha_pts = 161
        elif len(parts) == 5:
            rel, Tstr, amin_str, amax_str, apts_str = parts
            alpha_min = float(amin_str)
            alpha_max = float(amax_str)
            alpha_pts = int(apts_str)
        else:
            raise ValueError(
                "Each --items entry must be 'relpath,T' or " \
                "'relpath,T,alpha_min,alpha_max' or " \
                "'relpath,T,alpha_min,alpha_max,alpha_pts'."
            )

        T = float(Tstr)
        dpath = os.path.join(args.base, rel)
        files = sorted(glob.glob(os.path.join(dpath, "*.h5")))
        if not files:
            raise FileNotFoundError(f"No .h5 files found in {dpath}")

        beta_raw = util.load_file(files[0], "metadata/beta")
        beta = float(np.asarray(beta_raw[0]).reshape(-1)[0])
        if np.abs(T - 1/beta) >= 5e-5:
            raise ValueError(f"Input T {T} does not match T {1/beta} read from hdf5.")
        
        T_arr.append(T)
        dpath_arr.append(dpath)
        alpha_min_arr.append(alpha_min)
        alpha_max_arr.append(alpha_max)
        alpha_pts_arr.append(alpha_pts)
        #dt_arr.append(dt)
        if args.highT_model is not None:
            idx = sorted(range(len(T_arr)), key=lambda i: T_arr[i], reverse=True)
            T_arr    = [T_arr[i] for i in idx]
            dpath_arr = [dpath_arr[i] for i in idx]
            alpha_min_arr = [alpha_min_arr[i] for i in idx]
            alpha_max_arr = [alpha_max_arr[i] for i in idx]
            alpha_pts_arr = [alpha_pts_arr[i] for i in idx]
            #dt_arr   = [dt_arr[i] for i in idx]
        elif args.lowT_gap is not None:
            idx = sorted(range(len(T_arr)), key=lambda i: T_arr[i], reverse=False)
            T_arr    = [T_arr[i] for i in idx]
            dpath_arr = [dpath_arr[i] for i in idx]
            alpha_min_arr = [alpha_min_arr[i] for i in idx]
            alpha_max_arr = [alpha_max_arr[i] for i in idx]
            alpha_pts_arr = [alpha_pts_arr[i] for i in idx]
            #dt_arr   = [dt_arr[i] for i in idx]
        else:
            print("Model not given. Using flat model for annealing from high T to low T.")
            idx = sorted(range(len(T_arr)), key=lambda i: T_arr[i], reverse=True)
            T_arr    = [T_arr[i] for i in idx]
            dpath_arr = [dpath_arr[i] for i in idx]
            alpha_min_arr = [alpha_min_arr[i] for i in idx]
            alpha_max_arr = [alpha_max_arr[i] for i in idx]
            alpha_pts_arr = [alpha_pts_arr[i] for i in idx]
            #dt_arr   = [dt_arr[i] for i in idx]

    op_type = args.op_type
    sym = args.sym
    if op_type == "fermion" and sym:
        print("[INFO] --sym is ignored for fermion.", file=sys.stderr)
        sym = False
    
    grid = args.grid
    if grid == "linear":
        omega, domega = run_maxent_phoenix.build_grid(op_type, sym, int(args.n_omega), float(args.omega_max), grid)
    elif grid == "sinh":
        omega, domega = run_maxent_phoenix.build_grid(op_type, sym, int(args.n_omega), float(args.omega_max), grid, float(args.a), float(args.b))
    else: raise ValueError(f"Unknown grid type: {grid}, grid type should be either \"linear\" or \"sinh\"")

    append = args.append

    mkwargs = {"method": args.method}
    if args.rnd_seed is not None:
        np.random.seed(int(args.rnd_seed))
    
    prev_A = None
    for i in range(len(T_arr)):
        T = T_arr[i]
        #dt = dt_arr[i]
        dpath = dpath_arr[i]
        if alpha_pts_arr[i] < 2:
            raise ValueError("alpha_pts must be at least 2.")
        if alpha_max_arr[i] <= alpha_min_arr[i]:
            raise ValueError("alpha_max must be greater than alpha_min.")
        alpha_arr = np.logspace(alpha_min_arr[i], alpha_max_arr[i], alpha_pts_arr[i])
        if i == 0:
            if args.highT_model is not None:
                model = np.array([float(x) for x in args.highT_model.split(",") if x.strip()], dtype=float)
                if model.shape != omega.shape:
                    raise ValueError(
                        f"--highT_model length {model.shape[0]} does not match omega grid length {omega.shape[0]}."
                    )
            elif args.lowT_gap is not None:
                gap = float(args.lowT_gap)
                power = 3.0
                m_cont = 1.0 - np.exp(-(omega/gap)**power)
                model = m_cont * domega
                model /= model.sum()
            else:
                model = None
        else:
            if prev_A is None:
                raise RuntimeError("Previous temperature MaxEnt output A is missing; cannot anneal to the next temperature.")
            model = prev_A

        corr = np.load(os.path.join(dpath, args.data_file), allow_pickle=False)
        if corr.ndim != 2: raise ValueError(f"Correlator must be 2D (N_bin, L) matrix. Got shape {corr.shape}.")
        nbin, L = corr.shape
        beta = 1/T
        dt = beta/L
        metadata = {"dt": dt, "beta": beta, "L": L, "nbin": int(nbin)}

        results = run_maxent_phoenix.perform_maxent(
            chi=corr,
            omega_grid=(omega, domega),
            metadata=metadata,
            append=append,
            alpha_arr=alpha_arr,
            bs=int(args.bs),
            anneal_arr=model,
            checks=args.checks,
            printout=args.printout,
            op_type=op_type,
            sym=sym,
            **mkwargs
        )
        prev_A = np.array(results["A"], copy=True)

        if args.output_relpath:
            out_dir = os.path.join(dpath, args.output_relpath)
            os.makedirs(out_dir, exist_ok=True)
            pfx = args.output_prefix or ""
            outputs = {
                "A_mean": results["A"],
                "s_all": results["s"],
                "omega": omega,
                "domega": domega,
                "metadata": np.array(metadata, dtype=object),
            }
            for name, arr in outputs.items():
                fname = f"{pfx}{name}.npy"
                fpath = os.path.join(out_dir, fname)
                np.save(fpath, arr)
                print(f"[OK] Saved {name} -> {fpath}")
        else: print("[INFO] No --output_relpath provided; results not saved.", file=sys.stderr)

if __name__ == "__main__": main()
        


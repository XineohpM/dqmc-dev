#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_beta_scan.py 
phoenixm@stanford.edu
Usage example:
    1) Log beta axis, U=10, 6*6, energy correlation
    python3 gen_beta_scan.py \
        --beta_axis_mode log --beta_min 0.01 --beta_max 10 --n_beta 20 \
        --dt_tgt 0.05 --L_min 40 \
        --n_matmul 5 --period-eqlt 5 --period-uneqlt 2 --dt_policy closest \
        --generator_path /home/users/phoenixm/dqmc-dev/util/gen_1band_unified_hub.py \
        --output_path  /scratch/users/phoenixm/dqmc_runs_U10_C \
        --output_prefix C_U10_ \
        --geometry square --Nx 6 --Ny 6 --U 10 --hs_channel auto --tp 0 --tpp 0 --nflux 0 --bc 1 --mu 0 \
        --Nfiles 100 --n_sweep_warm 200 --n_sweep_meas 4000 --trans_sym 1 \
        --meas_energy_corr 1
    2) Beta list, dry-run
    python3 gen_beta_scan.py \
        --beta_list "10,6.95193,4.83293,3.35982,2.33572,1.62378" \
        --dt_tgt 0.05 --L_min 40 \
        --n_matmul 5 --period-eqlt 5 --period-uneqlt 2 --dt_policy closest \
        --generator_path /home/users/phoenixm/dqmc-dev/util/gen_1band_unified_hub.py \
        --output_path /scratch/users/phoenixm/dqmc_runs_try \
        --output_prefix C_U10_ \
        --geometry square --Nx 6 --Ny 6 --U 10 --hs_channel auto \
        --Nfiles 10 --n_sweep_warm 200 --n_sweep_meas 1000 --trans_sym 1 \
        --meas_energy_corr 1 \
        --dry_run
"""
import argparse, math, os, sys, shlex, subprocess, textwrap
import numpy as np
from pathlib import Path

#Least common multiple
def lcm(a: int, b: int): return abs(a*b)//math.gcd(a, b)

def parse_args():
    p = argparse.ArgumentParser(description="Beta scan: compute (L,dt) with constraints and call generator per beta",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ---- beta specification ----
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--beta_axis_mode", choices=["log", "linear"], default="linear",
                   help="Generate beta axis (beta_min, beta_max), choose from \"linear\" or \"log\".")
    g.add_argument("--beta_list", type=str,
                   help="Comma-separated beta list, e.g. '10,6.05,4.83'.")
    p.add_argument("--beta_min", type=float,
                   help="Minimum value of beta.")
    p.add_argument("--beta_max", type=float,
                   help="Maximum value of beta.")
    p.add_argument("--n_beta", type=int,
                   help="Number of betas.")
    # ---- dt & L ----
    p.add_argument("--dt_tgt", type=float, default=0.05, 
                   help="Target value of dtau.")
    p.add_argument("--L_min", type=int, default=40,
                   help="Minimum value of L.")
    p.add_argument("--period-eqlt", type=int, default=8, 
                   help="Period of equal-time measurements in units of single-site updates.")
    p.add_argument("--period-uneqlt", type=int, default=0,
                   help="Period of unequal-time measurements in units of full H-S sweeps, 0 means disabled.")
    p.add_argument("--dt_policy", choices=["closest","ceiling"], default="closest",
                   help="Closest: choose multiple of B closest to dt_tgt; Ceiling: choose smallest L multiple with dt<=dt_tgt.")

    # ---- generator & filesystem ----
    p.add_argument("--generator_path", required=True,
                   help="Path to generator script (e.g., /path/to/gen_1band_unified_hub.py).")
    p.add_argument("--output_path", required=True,
                   help="Directory to place output per-beta subfolders.")
    p.add_argument("--output_prefix", default="_",
                   help="Prefix for generated .h5 files; final prefix is <output_prefix>T<...>_")
    p.add_argument("--dry_run", action="store_true", 
                   help="Only print commands, do not execute.")

    # ---- physics ----
    p.add_argument("--geometry", default="square", choices=["square","triangular","honeycomb","kagome"],
                   help="Lattice geometry, choose from \"square\", \"triangular\", \"honeycomb\", \"kagome\".")
    p.add_argument("--Nx", type=int, default=4,
                   help="Number of lattice sites along x direction.")
    p.add_argument("--Ny", type=int, default=4,
                   help="Number of lattice sites along y direction.")
    p.add_argument("--U", type=float, default=6.0,
                   help="On-site Hubbard repulsion strength.")
    p.add_argument("--hs_channel", choices=["auto", "spin", "density"], default="auto",
                   help=("HS decoupling channel passed to generator: 'spin' (n_up-n_dn), "
                         "'density' (n_up+n_dn-1), or 'auto' (spin for U>=0, density for U<0)."))
    p.add_argument("--tp", type=float, default=0.0,
                   help="Next nearest hopping integral.")
    p.add_argument("--tpp", type=float, default=0.0,
                   help="Third nearest hopping integral.")
    p.add_argument("--nflux", type=int, default=0,
                   help="Number of flux threading the cluster.")
    p.add_argument("--bc", type=int, default=1, 
                   help="Boundary conditions, 1 for periodic, 2 for open.")
    p.add_argument("--mu", type=float, default=0.0,
                   help="Chemical potential.")
    p.add_argument("--h", type=float, default=0.0, metavar="X",
        help="Zeeman field strength. Down electrons feel net (mu+h) chemical potential")
    p.add_argument("--twistx", type=float, default=0.0, metavar="X",
        help="Twist phase per bond along x. Equivalent to total twist Nx * twistx on boundary.")
    p.add_argument("--twisty", type=float, default=0.0, metavar="X",
        help="Twist phase per bond along y. Equivalent to total twist Ny * twisty on boundary.")
    
    # ---- expensive measurements ----
    p.add_argument("--n_sweep_warm", type=int, default=200,
                   help="Number of warmup sweeps.")
    p.add_argument("--n_sweep_meas", type=int, default=4000,
                   help="Number of measurement sweeps.")
    p.add_argument("--meas_energy_corr", type=int, default=0,
                   help="Whether to measure energy-energy correlations.")
    p.add_argument("--meas_bond_corr", type=int, default=0,
                   help="Whether to measure bond-bond correlations (current, kinetic energy, bond singlets).")
    p.add_argument("--meas_nematic_corr", type=int, default=0,
                   help="Whether to measure spin and charge nematic correlations.")
    p.add_argument("--meas_thermal", type=int, default=0,
                   help="Whether to measure extra jnj(2) type correlations for themal conductivity.")
    p.add_argument("--meas_2bond_corr", type=int, default=0,
                   help="Whether to measure extra jj(2) type correlations for themal conductivity.")
    p.add_argument("--meas_chiral", type=int, default=0,
                   help="Whether to measure scalar spin chirality.")
    p.add_argument("--meas_local_JQ", type=int, default=0,
                   help="Whether to measure local JQ for energy magnetization contribution to thermal Hall.")
    p.add_argument("--meas_gen_suscept", type=int, default=0,
                   help="Whether to measure generalized susceptibility.")
    p.add_argument("--meas_pair_bb_only", type=int, default=0,
                   help="Whether to, among expensive measurements, to only measure bond singlet pair correlators in order to save on storage.")

    # ---- generator ----
    p.add_argument("--Nfiles", type=int, default=100,
                   help="Number of simulation files to generate.")
    p.add_argument("--trans_sym", type=int, default=1,
                   help="Whether to apply translational symmetry to compress measurement data.")
    p.add_argument("--printout", type=int, default=1)
    p.add_argument("--overwrite", type=int, default=0)
    p.add_argument("--n_delay", type=int, default=16)
    p.add_argument("--n_matmul", type=int, default=8, 
                   help="Half the maximum number of direct matrix multiplications before applying a QR decomposition.")
    p.add_argument("--checkpoint_every", type=int, default=10000)

    return p.parse_args()

def build_beta_axis(args):
    if args.beta_list: 
        betas = [float(x) for x in args.beta_list.split(",") if x.strip()]
        return np.array(betas, dtype=float)
    if args.beta_axis_mode is None:
        raise SystemExit("Either --beta-list or --beta-mode with min/max/num required.")
    if any(param is None for param in (args.beta_min, args.beta_max, args.n_beta)):
        raise SystemExit("--beta_min/--beta_max/--n_beta required for --beta_axis_mode.")
    n_beta = args.n_beta
    if args.beta_axis_mode == "log":
        log10_beta_min = np.log10(args.beta_min)
        log10_beta_max = np.log10(args.beta_max)
        betas = np.logspace(log10_beta_min, log10_beta_max, num=n_beta)
    else:
        betas = np.linspace(args.beta_min, args.beta_max, num=args.n_beta)
    return betas

def lcm_block(args):
    B = 1
    vals = [args.n_matmul, args.period_eqlt,
            args.period_uneqlt if args.period_uneqlt>0 else 1, 2]
    for v in vals:
        B = lcm(B, v)
    return B

def Trotter_check(dt, args):
    """Return True if the Trotter step dt is sufficiently small.

    We use the magnitude |U| because the error estimate depends on the interaction scale,
    not on the sign of U. If U==0, the constraint is vacuous.

    Heuristic: (dtau^2) * |U| * t \lesssim 1/8 (with t scaled to 1 in our units).
    """
    absU = abs(args.U)
    if absU == 0.0:
        return True
    return dt**2 <= 1.0 / (8.0 * absU)

def choose_L_dt(beta, B, args):
    L0 = max(int(math.ceil(beta/args.dt_tgt)), args.L_min)
    if args.dt_policy == "closest":
        k = max(1, L0//B)
        cands = [k*B, (k+1)*B]
        cands = [c for c in cands if c >= args.L_min]
        if not cands:
            L = (k+1)*B
        else:
            L = min(cands, key=lambda L_: abs(beta/L_ - args.dt_tgt))
    else:  # ceiling: ensure dt <= dt_tgt
        k = int(math.ceil(L0 / B))
        L = max(args.L_min, k*B)
    dt = beta / L
    if not Trotter_check(dt, args):
        absU = abs(args.U)
        # If absU==0, Trotter_check would have returned True; keep a guard anyway.
        if absU == 0.0:
            return int(L), float(dt)
        dt_max = math.sqrt(1.0 / (8.0 * absU))
        dvd = int(math.ceil(dt / dt_max))
        dt = dt / dvd
        L = L * dvd
    return int(L), float(dt)

def main():
    args = parse_args()


    betas = build_beta_axis(args)
    B = lcm_block(args)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    generator_path = Path(args.generator_path)
    if not generator_path.exists(): raise SystemExit(f"Generator not found: {generator_path}.")

    summary_rows = []
    for beta in sorted(betas):
        L, dt = choose_L_dt(beta, B, args)
        T = 1.0/beta
        sub = output_path / f"T_{T:.6g}"
        sub.mkdir(parents=True, exist_ok=True)
        prefix = f"{args.output_prefix}T{T:.6g}_"

        #build command
        cmd = [
            sys.executable, str(generator_path),
            "--prefix", prefix,
            "--geometry", args.geometry,
            "--Nx", str(args.Nx), "--Ny", str(args.Ny),
            "--tp", str(args.tp), "--tpp", str(args.tpp),
            "--nflux", str(args.nflux),
            "--U", str(args.U),
            "--hs_channel", str(args.hs_channel),
            "--bc", str(args.bc), "--mu", str(args.mu),
            "--dt", f"{dt:.12g}", "--L", str(L),
            "--Nfiles", str(args.Nfiles),
            "--n_sweep_warm", str(args.n_sweep_warm),
            "--n_sweep_meas", str(args.n_sweep_meas),
            "--period_eqlt", str(args.period_eqlt),
            "--trans_sym", str(args.trans_sym),
            "--printout", str(args.printout),
            "--overwrite", str(args.overwrite),
            "--n_delay", str(args.n_delay),
            "--h", f"{args.h:.12g}",
            "--twistx", f"{args.twistx:.12g}",
            "--twisty", f"{args.twisty:.12g}",
            "--n_matmul", str(args.n_matmul),
            "--checkpoint_every", str(args.checkpoint_every),
        ]
        
        if args.period_uneqlt > 0: cmd += ["--period_uneqlt", str(args.period_uneqlt)]
        if args.meas_energy_corr: cmd += ["--meas_energy_corr", str(args.meas_energy_corr)]
        if args.meas_bond_corr: cmd += ["--meas_bond_corr", str(args.meas_bond_corr)]
        if args.meas_nematic_corr: cmd += ["--meas_nematic_corr", str(args.meas_nematic_corr)]
        if args.meas_thermal: cmd += ["--meas_thermal", str(args.meas_thermal)]
        if args.meas_2bond_corr: cmd += ["--meas_2bond_corr", str(args.meas_2bond_corr)]
        if args.meas_chiral: cmd += ["--meas_chiral", str(args.meas_chiral)]
        if args.meas_local_JQ: cmd += ["--meas_local_JQ", str(args.meas_local_JQ)]
        if args.meas_gen_suscept: cmd += ["--meas_gen_suscept", str(args.meas_gen_suscept)]
        if args.meas_pair_bb_only: cmd += ["--meas_pair_bb_only", str(args.meas_pair_bb_only)]

        log = sub / "gen.log"
        cwd = os.getcwd() #save cwd as string
        try:
            os.chdir(sub) 
            print(f"[GEN] T={T:.6g} beta={beta:.6g}  L={L} dt={dt:.6g}  B={B}")
            print("      ", " ".join(shlex.quote(x) for x in cmd))
            if not args.dry_run:
                with open(log, "w") as fo:
                    proc = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT, check=True)
                # move files whose name starts with prefix into sub dir if generator writes to cwd
                for f in Path(".").glob(f"{prefix}*.h5"):
                    # already in sub dir (cwd=sub), skip
                    pass
            summary_rows.append((T, beta, L, dt, str(sub), prefix))
        finally:
            os.chdir(cwd)

    # write summary
    sum_path = output_path / "summary.tsv"
    summary_exists = sum_path.exists() and sum_path.stat().st_size > 0
    mode = "a" if summary_exists else "w"
    with open(sum_path, mode) as f:
        if not summary_exists:
            f.write("#B=%d dt_policy=%s dt_tgt=%g L_min=%d\n" % (B, args.dt_policy, args.dt_tgt, args.L_min))
            f.write("T\tbeta\tL\tdt\tdir\tprefix\n")
        for T,beta,L,dt,sub,prefix in sorted(summary_rows, key=lambda r:r[0]):
            f.write(f"{T:.12g}\t{beta:.12g}\t{L}\t{dt:.12g}\t{sub}\t{prefix}\n")
    action = "appended to" if summary_exists else "wrote"
    print(f"[OK] {action} {sum_path}")
    print("[DONE]")

if __name__ == "__main__":
    main()
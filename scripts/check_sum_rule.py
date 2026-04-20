import os, sys, glob, re
from pathlib import Path
import numpy as np
import argparse
import maxent
import data_analysis as da
utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))
import util

def load_dt_from_genlog(dir):
    genlog = os.path.join(dir, "gen.log")
    with open(genlog, "r") as f:
        text = f.read()
    m = re.search(r"(?m)^dt\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", text)
    if m is None:
        raise ValueError(f"Could not find dt in {genlog}")
    return float(m.group(1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True,
                   help="Parent directory containing subfolders.")
    p.add_argument("--relpath_list", nargs="+", required=True,
                   help="List of relative paths.")
    p.add_argument("--correlator_name", default="JNJN_xx_perbin.npy",
                   help="Filename of the imaginary time current-current correlator.")
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
        dt = load_dt_from_genlog(dir)
        beta, = util.load_firstfile(dir, "metadata/beta")
        k, k_err = da.eqlt_meas_1(dir, ["kinetic"])
        k = k["kinetic"].real
        k_err = k_err["kinetic"].real
        corr = np.load(os.path.join(dir, args.correlator_name), allow_pickle=False)

        # Generate a random grid, does not affect norm of corr
        omega, domega = maxent.gen_grid(
            nw=200,
            x_min=0,
            x_max=5,
            w_x=lambda x: x
        )

        pre = maxent.Preprocess(
            G=corr,
            dt=dt,
            beta=beta,
            grid_info=(omega, domega),
            op_type="boson",
            sym=True
        )
        norm = pre["norm"]
        print("T = ", 1.0/beta)
        print("dt = ", dt)
        print("norm of correlator = ", norm*4)
        print("kinetic energy = ", -k)
        print("kinetic energy error = ", k_err)
        if np.isclose(norm*4, -k, rtol=1e-02, atol=0): print("norm of correlator = kinetic energy")
        else: print("norm of correlator and kinetic energy are not close")
        print("k/norm = ", -k/norm/4)
        print(" ")

if __name__ == "__main__":
    main()

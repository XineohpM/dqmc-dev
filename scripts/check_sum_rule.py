import os, sys, glob
from pathlib import Path
import numpy as np
import argparse
import maxent
import data_analysis as da
utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))
import util

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
    
    for relpath in args.relpath_list:
        dir = os.path.join(path, relpath, "")
        dt, beta = util.load_firstfile(dir, "metadata/dt", "metadata/beta")
        k, k_err = da.eqlt_meas_1(dir, ["kinetic"])
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

        print("norm of correlator = ", norm)
        print("kinetic energy = ", k)
        if np.isclose(norm, k): print("norm of correlator = kinetic energy")
        else: print("norm of correlator and kinetic energy are not close")
        print("norm/k = ", norm/k)
        print("k/norm = ", k/norm)

if __name__ == "__main__":
    main()


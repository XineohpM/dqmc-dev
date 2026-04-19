#!/usr/bin/env python3
# /home/users/phoenixm/scripts/extract_energy_perfile.py
import argparse, os
from pathlib import Path
import numpy as np
import h5py

def energy_from_h5(path, U):
    """Try to compute energy per site from an h5 file object path (Path or str).
    Strategy:
      1) If dataset 'meas_eqlt/energy' exists, use its last completed value.
      2) Else try kinetic 'meas_eqlt/kk' + potential U * double_occ ('meas_eqlt/double_occ').
         Note: signs follow file content; we will not flip signs automatically.
    Returns tuple (E_total, note) where E_total is energy **total** (not per-site) if possible.
    """
    with h5py.File(path,'r') as fh:
        # prefer explicit energy dataset if present
        if 'meas_eqlt/energy' in fh:
            arr = fh['meas_eqlt/energy'][...]
            # arr may be per-bin or scalar; take mean over last completed bin if array
            E = float(np.mean(arr)) if arr.size>1 else float(arr)
            return E, "from meas_eqlt/energy"
        # fallback: kinetic + potential
        have_kk = 'meas_eqlt/kk' in fh
        have_dd = 'meas_eqlt/double_occ' in fh
        if have_kk and have_dd:
            kk = fh['meas_eqlt/kk'][...]
            dd = fh['meas_eqlt/double_occ'][...]
            # take mean if arrays
            kk_m = float(np.mean(kk)) if kk.size>1 else float(kk)
            dd_m = float(np.mean(dd)) if dd.size>1 else float(dd)
            E = kk_m + U * dd_m
            return E, "from kk + U*double_occ"
        # if only double occupancy present, use potential only (less ideal)
        if have_dd:
            dd = fh['meas_eqlt/double_occ'][...]
            dd_m = float(np.mean(dd)) if dd.size>1 else float(dd)
            E = U * dd_m
            return E, "from U*double_occ (no kk)"
        raise KeyError("No usable energy fields (meas_eqlt/energy or kk & double_occ) in file")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Absolute path to T directory (e.g. /scratch/.../beta0p25)")
    p.add_argument("--U", required=False, type=float, help="U; if not given try to read from metadata in files")
    p.add_argument("--out", default="E_perfile.npy", help="output filename in same dir (default E_perfile.npy)")
    args = p.parse_args()

    d = Path(args.dir)
    if not d.exists(): raise SystemExit(f"Dir not found: {d}")
    files = sorted([p for p in d.glob("*.h5")])
    if not files:
        raise SystemExit("No .h5 files found in " + str(d))
    # try to get U from first file if not provided
    U = args.U
    if U is None:
        try:
            with h5py.File(files[0],'r') as fh:
                if 'metadata/U' in fh:
                    U = float(fh['metadata/U'][()])
        except Exception:
            pass
    if U is None:
        raise SystemExit("U not provided and not found in metadata; provide --U")

    E_list = []
    notes = []
    bad = []
    for f in files:
        try:
            E, note = energy_from_h5(f, U)
            E_list.append(float(E))
            notes.append(note)
        except Exception as e:
            bad.append((str(f), str(e)))
            E_list.append(np.nan)
            notes.append("ERR:"+str(e))

    outp = d / args.out
    np.save(outp, np.array(E_list))
    # save meta info
    meta = {
        "n_files": len(files),
        "files": [str(x.name) for x in files],
        "notes": notes
    }
    np.save(d / (args.out + ".meta.npy"), meta)
    print(f"Wrote {outp} (N={len(E_list)}). {len(bad)} files failed.")
    if bad:
        print("Failures (first 10):", bad[:10])

if __name__ == "__main__":
    main()
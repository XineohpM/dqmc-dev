#!/usr/bin/env python3
"""
push.py

Append absolute paths of HDF5 files into a stack file (one path per line).
This script expands glob patterns itself (via glob()), so you should quote patterns.

USAGE
-----
python3 push.py STACKFILE PATTERN [PATTERN2 ...]
python3 push.py STACKFILE a.h5 b.h5 ...

EXAMPLES
--------
# (1) From a run directory that contains T_* subfolders:
cd /scratch/users/phoenixm/dqmc_runs/U-2_n6x6_C
python3 /home/users/phoenixm/dqmc-dev/util/push.py stack_energy 'T_*/*.h5'

# (2) Append only a subset (e.g. a prefix):
python3 /home/users/phoenixm/dqmc-dev/util/push.py stack_energy 'T_0.1/C_U-2_*.h5' 'T_0.2/C_U-2_*.h5'

# NOTE
# - This script appends to STACKFILE. Re-running with the same PATTERN will duplicate entries.
# - If you want a fresh stackfile, delete it first: rm -f stack_energy
"""

import os
import sys
from glob import glob

def main(argv):
    if len(argv) < 3:
        print("usage: {} stackfile a.h5 b.h5 ...".format(argv[0]))
        return
    stack = argv[1]
    #          src    dest (created)
    os.symlink(stack, stack + "~")
    with open(stack, "a") as f:
        for x in argv[2:]:
            files = sorted(glob(x))
            if len(files) == 0:
                print("No files matching:"+x)
            else:
                for ff in files:
                    print(os.path.abspath(ff), file=f)
    os.remove(stack + "~")

if __name__ == "__main__":
    main(sys.argv)

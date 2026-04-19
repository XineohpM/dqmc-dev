#!/bin/bash
#SBATCH --job-name=a
#SBATCH --time=60:00:00
#SBATCH --partition=simes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20



time unbuffer ./a 6.0 0.4 8 | tee checkU6tp0

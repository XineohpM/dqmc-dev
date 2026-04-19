#!/bin/bash
#SBATCH --job-name=a
#SBATCH --time=60:00:00
#SBATCH --partition=simes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20



time unbuffer ./a 8.0 0.5 9 | tee U8tp0_9_n1
time unbuffer ./a 10.0 0.5 9 | tee U10tp0_9_n1
time unbuffer ./a 12.0 0.5 9 | tee U12tp0_9_n1

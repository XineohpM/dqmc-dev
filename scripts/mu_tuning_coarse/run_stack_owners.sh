#!/bin/bash
#SBATCH --job-name=8x8_mutune
#SBATCH --time=48:00:00
#SBATCH --partition=owners
#SBATCH --requeue #in case of preemption, requeue
#SBATCH --array=1-100
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --qos=normal #or: high_p
#SBATCH --mail-type=FAIL 
#SBATCH --mail-user=emilyzh@stanford.edu

# dqmc util folder location
# OUTPUT_DIR=/scratch/users/emilyzh/mu_tuning_gen/nflux1/n1.0/
OUTPUT_DIR=$1
# cd $SCRATCH/data_dump

echo "Date                 = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Acting Directory = $EXEC_DIR"
echo "Save Directory = $OUTPUT_DIR"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_CPUS_PER_TASK"

module list #double check module list

#this is where I put my DQMC executables
cd ${HOME}/dqmc-dev/build
#cd /home/users/emilyzh/dqmc-dev_commitide9f2680/dqmc-dev/build

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK #workaround for annoying slurm 22.05 upgrade
srun --export=ALL,MKL_DEBUG_CPU_TYPE=5 --cpus-per-task=$SLURM_CPUS_PER_TASK ./dqmc_stack -t 172400 ${OUTPUT_DIR}stack

#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=simes
#SBATCH --requeue #in case of preemption, requeue
#SBATCH --array=1-1000
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --qos=normal #or: high_p
#SBATCH --mail-type=FAIL 
#SBATCH --mail-user=phoenixm@stanford.edu

# dqmc util folder location
# If a directory is provided as the first argument, use it; otherwise use the OUTPUT_DIR environment variable if set;
# if neither is provided, fall back to a sensible default on $SCRATCH.
if [ -n "$1" ]; then
  OUTPUT_DIR="$1"
elif [ -n "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$OUTPUT_DIR"
else
  OUTPUT_DIR="$SCRATCH/dqmc_runs/U6_tp0_N6x6"
fi

case "$OUTPUT_DIR" in
  */) ;;
  *) OUTPUT_DIR="${OUTPUT_DIR}/" ;;
esac

# Create the directory if it doesn't exist and verify writability
mkdir -p "$OUTPUT_DIR"
if [ ! -w "$OUTPUT_DIR" ]; then
  echo "ERROR: cannot write to $OUTPUT_DIR"
  exit 1
fi

# --- stackfile selection ---
# Prefer STACKFILE env var; if not set, fall back to OUTPUT_DIR/stack.
if [ -n "$STACKFILE" ]; then
  STACKFILE="$STACKFILE"
else
  STACKFILE="${OUTPUT_DIR%/}/stack"
fi

# If relative path, resolve against current working dir
case "$STACKFILE" in
  /*) ;;
  *)  STACKFILE="$(pwd)/$STACKFILE" ;;
esac

if [ ! -f "$STACKFILE" ]; then
  echo "ERROR: STACKFILE not found: $STACKFILE"
  exit 1
fi
if [ ! -w "$STACKFILE" ]; then
  echo "ERROR: STACKFILE not writable (dqmc_stack needs to edit it): $STACKFILE"
  exit 1
fi
# --- end stackfile selection ---

EXEC_DIR="${EXEC_DIR:-${HOME}/dqmc-dev/build}"
# --- choose DQMC binary: real vs complex ---
# Priority: explicit $DQMC_BIN > $DQMC_VARIANT (cplx/real) > $DQMC_COMPLEX (1/0) > default real
DQMC_BIN="${DQMC_BIN:-}"
if [ -z "$DQMC_BIN" ]; then
  case "${DQMC_VARIANT:-}" in
    cplx|complex|CPLX|COMPLEX) DQMC_BIN="dqmc_stack_cplx" ;;
    real|REAL|"")             DQMC_BIN="dqmc_stack" ;;
    *)                         DQMC_BIN="dqmc_stack" ;;
  esac
fi
if [ -z "$DQMC_BIN" ] && [ "${DQMC_COMPLEX:-0}" = "1" ]; then
  DQMC_BIN="dqmc_stack_cplx"
fi
DQMC_BIN="${DQMC_BIN:-dqmc_stack}"
# --- end selection ---
if [ ! -x "${EXEC_DIR}/${DQMC_BIN}" ]; then
  if [ -x "${SCRATCH}/dqmc_runs/myrun/${DQMC_BIN}" ]; then
    EXEC_DIR="${SCRATCH}/dqmc_runs/myrun"
  else
    echo "ERROR: ${DQMC_BIN} not found or not executable in ${EXEC_DIR} nor ${SCRATCH}/dqmc_runs/myrun"
    echo "Please set EXEC_DIR to the directory containing ${DQMC_BIN} (export EXEC_DIR=/path/to/dir)"
    exit 1
  fi
fi

echo "Date                 = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Acting Directory = $EXEC_DIR"
echo "Save Directory = $OUTPUT_DIR"
echo "Stackfile        = $STACKFILE"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_CPUS_PER_TASK"

module load hdf5/1.10.2 icc/2019 imkl/2019 python/3.9.0
module list #double check module list

#this is where I put my DQMC executables
cd "${EXEC_DIR}" || { echo "ERROR: failed to cd to ${EXEC_DIR}"; exit 1; }
echo "Using EXEC_DIR=${EXEC_DIR}; dqmc_bin = ${EXEC_DIR}/${DQMC_BIN}"
module list #double check module list
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK #workaround for annoying slurm 22.05 upgrade
srun --export=ALL,MKL_DEBUG_CPU_TYPE=5 --cpus-per-task=$SLURM_CPUS_PER_TASK ./${DQMC_BIN} -t 172400 "$STACKFILE"
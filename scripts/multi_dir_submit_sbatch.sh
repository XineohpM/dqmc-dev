#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/users/phoenixm/dqmc_runs/U-6_n6x6_resistivity"
STACK_NAME="stack"

RUN_SH="/home/users/phoenixm/scripts/run_stack_owners.sh"
DQMC_BIN="dqmc_stack_negU"

EXEC_DIR="/scratch/users/phoenixm/dqmc_runs/myrun"

PARTITION="owners"
CPUS=2
MEM="32G"
TIME="48:00:00"

# OMP
OMP_NUM_THREADS=2

shopt -s nullglob

T_DIRS=("$ROOT"/T_*)
if (( ${#T_DIRS[@]} == 0 )); then
  echo "ERROR: No T_* directories found under: $ROOT" >&2
  exit 2
fi

echo "Found ${#T_DIRS[@]} temperature dirs under $ROOT"

n_submit=0
n_skip=0
n_fail=0

for d in "${T_DIRS[@]}"; do
  [[ -d "$d" ]] || continue

  stackfile="$d/$STACK_NAME"
  if [[ ! -s "$stackfile" ]]; then
    echo "[SKIP] $d  (missing or empty $STACK_NAME)"
    ((n_skip+=1))
    continue
  fi

  mkdir -p "$d/slurm_logs"

  echo "[SUBMIT] $d  stack=$(realpath "$stackfile")"

  if sbatch \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --requeue \
    --mail-type=FAIL,END \
    --mail-user=phoenixm@stanford.edu \
    --chdir="$d" \
    --output="$d/slurm_logs/slurm-stack_%j.out" \
    --export=ALL,OMP_NUM_THREADS="$OMP_NUM_THREADS",DQMC_BIN="$DQMC_BIN",EXEC_DIR="$EXEC_DIR",OUTPUT_DIR="$d",STACKFILE="$stackfile" \
    "$RUN_SH"; then
    ((n_submit+=1))
  else
    echo "[FAIL] submit failed for $d" >&2
    ((n_fail+=1))
  fi
done

echo "Done. submitted=$n_submit  skipped=$n_skip  failed=$n_fail"
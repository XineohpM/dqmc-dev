#!/usr/bin/env bash
set -uo pipefail

ROOT="/scratch/users/phoenixm/dqmc_runs/U-6_n6x6_resistivity"
PUSH="/home/users/phoenixm/dqmc-dev/util/push.py"
STACK_NAME="stack"

if [[ ! -f "$PUSH" ]]; then
  echo "ERROR: push.py not found: $PUSH" >&2
  exit 1
fi

shopt -s nullglob

# Discover T_* directories
T_DIRS=("$ROOT"/T_*)
if (( ${#T_DIRS[@]} == 0 )); then
  echo "ERROR: No T_* directories found under: $ROOT" >&2
  exit 2
fi
echo "Found ${#T_DIRS[@]} T_* directories under: $ROOT"

n_ok=0
n_skip=0
n_fail=0

for d in "${T_DIRS[@]}"; do
  [[ -d "$d" ]] || continue

  echo "[DIR]  $d"
  files=( "$d"/*.h5 )

  if (( ${#files[@]} == 0 )); then
    echo "[SKIP] $d  (no .h5)"
    ((n_skip++))
    continue
  fi

  stack="$d/$STACK_NAME"

  rm -f "$stack" "$stack~"
  : > "$stack"

  if ! python3 "$PUSH" "$stack" "$d/*.h5"; then
    echo "[FAIL] $d  (push.py failed)" >&2
    ((n_fail++))
    continue
  fi

  echo "[OK]   $d  -> $(realpath "$stack")  (N=${#files[@]})"
  ((n_ok++))
done

echo "Done. OK=$n_ok  SKIP=$n_skip  FAIL=$n_fail"
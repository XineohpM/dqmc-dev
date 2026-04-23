#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import defaultdict
import re
import subprocess


def log_indicates_complete(h5_path: Path):
    """Return (ok_log, is_complete, total_sweeps, last_reported_sweeps) based on the sibling .h5.log file.

    Complete means the log contains a line of the form
    'N/N sweeps completed' with equal integers N, and later contains both
    'saving data to disk' and 'sim_data_save() succeeded'.
    """
    log_path = Path(str(h5_path) + ".log")
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False, False, "", ""

    sweep_pat = re.compile(r"(\d+)\s*/\s*(\d+)\s+sweeps completed")
    matches = list(sweep_pat.finditer(text))
    if not matches:
        return True, False, "", ""

    last_reported_sweeps = matches[-1].group(1)
    total_sweeps = matches[-1].group(2)

    for m in matches:
        if m.group(1) != m.group(2):
            continue
        tail = text[m.end():]
        if "saving data to disk" in tail and "sim_data_save() succeeded" in tail:
            return True, True, m.group(2), matches[-1].group(1)

    return True, False, total_sweeps, last_reported_sweeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="Root directory to scan",
    )
    ap.add_argument("--glob", default="n*/T*_beta*_U*/mu*/*.h5")
    ap.add_argument("--push_stack", type=str, required=False)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = sorted(root.glob(args.glob))

    if not files:
        raise SystemExit(f"No .h5 files found under {root} with glob {args.glob}")

    # First pass: inspect sibling .h5.log per file, group by mu directory
    per_dir = defaultdict(list)
    for fp in files:
        ok_log, is_complete, total_sweeps, last_reported_sweeps = log_indicates_complete(fp)
        per_dir[fp.parent].append((fp, ok_log, is_complete, total_sweeps, last_reported_sweeps))

    # Second pass: report completion directly from the sibling .h5.log
    lines = []
    lines.append("dir\tfile\tok_log\tis_complete\ttotal_sweeps\tlast_reported_sweeps\n")

    n_total = 0
    n_incomplete = 0
    n_bad = 0

    stack = None
    incomplete_files = []
    if args.push_stack is not None:
        stack = root / args.push_stack

    for subdir, items in sorted(per_dir.items(), key=lambda kv: str(kv[0])):
        for fp, ok_log, is_complete, total_sweeps, last_reported_sweeps in items:
            n_total += 1
            if not ok_log:
                n_bad += 1
                is_complete = False
                incomplete_files.append(fp)
            elif not is_complete:
                n_incomplete += 1
                incomplete_files.append(fp)

            lines.append(
                f"{subdir}\t{fp}\t{int(ok_log)}\t{int(is_complete)}\t{total_sweeps}\t{last_reported_sweeps}\n"
            )

    output = root / "h5_completion_report.tsv"
    Path(output).write_text("".join(lines), encoding="utf-8")

    print(f"Wrote {output}")
    print(f"Total files: {n_total}")
    print(f"Missing/unreadable .h5.log: {n_bad}")
    print(f"Incomplete (log does not show finished sweeps + successful save): {n_incomplete}")
    if args.push_stack is not None:
        subprocess.run(
            ["python3", "/home/users/phoenixm/dqmc-dev/util/push.py", str(stack), *[str(fp.relative_to(root)) for fp in incomplete_files]],
            cwd=str(root),
            check=True,
        )
        print(f"Incomplete files pushed back to {stack}")


if __name__ == "__main__":
    main()
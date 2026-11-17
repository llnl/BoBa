#!/usr/bin/env bash
set -euo pipefail

# Usage: ./examples/benchmarks/matrix_norms/benchmark_matrix_norms.bash [output_directory]
# Set BOBA_MATRIX_NORMS_MAX_ELEMENTS to change the largest matrix in the sweep.
# Set BOBA_MATRIX_NORMS_REPETITIONS to change the number of timed calls per norm.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

output_dir="${1:-results/matrix_norms}"
max_elements="${BOBA_MATRIX_NORMS_MAX_ELEMENTS:-100000000}"
repetitions="${BOBA_MATRIX_NORMS_REPETITIONS:-10}"
jobs="${BOBA_MATRIX_NORMS_JOBS:-20}"

for value in "$max_elements" "$repetitions" "$jobs"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Expected positive integer, got: %s\n' "$value" >&2
    exit 1
  fi
done

mkdir -p "$output_dir/logs" "$output_dir/.matplotlib"
printf 'rows,cols,elements,reason\n' > "$output_dir/skipped_shapes.csv"

# Existing dependency files may still name the source path used before the move.
make -o examples/benchmarks/test_matrix_norms.cpp test_matrix_norms BOBA_CUDA=1 -j "$jobs"
exe="$(make -s --no-print-directory BOBA_CUDA=1 \
  --eval 'print-matrix-norms-exe: ; @echo test_matrix_norms$(NAME_FLAG).out' \
  print-matrix-norms-exe)"
if [[ ! -x "$exe" ]]; then
  printf 'Benchmark executable not found: %s\n' "$exe" >&2
  exit 1
fi

sizes=(1 10 100 1000 10000 100000 1000000 10000000)
run_logs=()
for rows in "${sizes[@]}"; do
  for cols in "${sizes[@]}"; do
    elements=$((rows * cols))
    if (( elements > max_elements )); then
      printf '%s,%s,%s,exceeds_max_elements\n' "$rows" "$cols" "$elements" >> "$output_dir/skipped_shapes.csv"
      printf 'Skipping rows=%s cols=%s (%s elements)\n' "$rows" "$cols" "$elements"
      continue
    fi

    printf 'Running rows=%s cols=%s (%s elements)\n' "$rows" "$cols" "$elements"
    run_log="$output_dir/logs/rows_${rows}_cols_${cols}.log"
    if ! "./$exe" --rows "$rows" --cols "$cols" --timing --repetitions "$repetitions" > "$run_log" 2>&1; then
      cat "$run_log" >&2
      exit 1
    fi
    sed -n '/^timing /p' "$run_log"
    run_logs+=("$run_log")
  done
done

MPLBACKEND=Agg MPLCONFIGDIR="$output_dir/.matplotlib" \
  python3 - "$output_dir" "$repetitions" "${run_logs[@]}" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext


output_dir = Path(sys.argv[1])
repetitions = int(sys.argv[2])
log_paths = [Path(path) for path in sys.argv[3:]]
timing = re.compile(
    r"^timing backend=(\S+) rows=(\d+) cols=(\d+) norm=(\S+) "
    r"mean_us=([\d.]+) repetitions=(\d+)\b"
)
shape = re.compile(r"rows_(\d+)_cols_(\d+)\.log")
norms = {
    "matrix_one": ("Matrix 1-norm", "matrix_one_norm.png"),
    "matrix_inf": ("Matrix infinity-norm", "matrix_inf_norm.png"),
    "frobenius": ("Frobenius norm", "frobenius_norm.png"),
}
naive_norms = {
    "matrix_one": "matrix_one_force_naive",
    "matrix_inf": "matrix_inf_force_naive",
    "frobenius": "frobenius_force_naive",
}
timed_norms = tuple(
    name for norm in norms for name in (norm, naive_norms.get(norm)) if name is not None
)
records = []

for path in log_paths:
    match = shape.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected log filename: {path}")
    expected_rows, expected_cols = map(int, match.groups())
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = timing.match(line)
        if match is None:
            continue
        backend, rows, cols, norm, mean_us, count = match.groups()
        if norm not in timed_norms:
            continue
        rows, cols, count = int(rows), int(cols), int(count)
        mean_us = float(mean_us)
        if (rows, cols) != (expected_rows, expected_cols) or count != repetitions:
            raise ValueError(f"Wrong shape or repetition count in {path}: {line}")
        if norm in found or not math.isfinite(mean_us) or mean_us <= 0:
            raise ValueError(f"Duplicate or invalid timing in {path}: {line}")
        found[norm] = (rows, cols, rows * cols, backend, norm, mean_us, count)
    if set(found) != set(timed_norms):
        raise ValueError(f"Missing norm timings in {path}: {set(timed_norms) - set(found)}")
    records.extend(found[norm] for norm in timed_norms)

if not records:
    raise ValueError("No benchmark timings found")

csv_path = output_dir / "matrix_norms_timings.csv"
with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("rows", "cols", "elements", "backend", "norm", "mean_us", "repetitions"))
    writer.writerows(records)

times = [record[5] for record in records]
elements = [record[2] for record in records]
columns = [record[1] for record in records]
y_limits = (min(times) / 1.2, max(times) * 1.2)
x_limits = (min(columns) / 1.2, max(columns) * 1.2)
color_norm = LogNorm(vmin=min(elements) / 1.2, vmax=max(elements) * 1.2)

for norm, (title, filename) in norms.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    for series, marker, label in ((norm, "o", "Default"),
                                  (naive_norms.get(norm), "^", "Naive Atomics")):
        if series is None:
            continue
        points = [record for record in records if record[4] == series]
        scatter = ax.scatter(
            [record[1] for record in points],
            [record[5] for record in points],
            c=[record[2] for record in points],
            cmap="viridis",
            norm=color_norm,
            marker=marker,
            label=label,
            s=110 if marker == "o" else 55,
            edgecolors="black",
            linewidths=0.3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Number of columns")
    ax.set_ylabel("Mean time per call (µs)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    colorbar.set_label("Total matrix elements (rows × columns)")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)

print(f"Timing CSV: {csv_path}")
for _, filename in norms.values():
    print(f"Plot: {output_dir / filename}")
print(f"Skipped shapes: {output_dir / 'skipped_shapes.csv'}")
PY

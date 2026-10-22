#!/usr/bin/env bash
set -euo pipefail

campaign_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${campaign_directory}/../.." && pwd)"

python3 "${repository_root}/examples/benchmarks/plot_benchmarks.py" \
  "${campaign_directory}" \
  "$@"

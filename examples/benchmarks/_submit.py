# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "util"))

from machines import get_machine


def positive_integer(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed_value


def default_executable(name):
    _, _, backend = get_machine()
    return REPOSITORY_ROOT / f"{name}_{backend}.out"


def make_parser(description, executable_name, output_directory):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--executable",
        type=Path,
        default=default_executable(executable_name),
        help="Benchmark executable to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "benchmark_jobs" / output_directory,
        help="Directory for Slurm output files.",
    )
    parser.add_argument("--time", default="02:00:00", help="Slurm time limit.")
    parser.add_argument("--partition", help="Optional Slurm partition.")
    parser.add_argument("--account", help="Optional Slurm account.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting them.",
    )
    return parser


def submit_jobs(arguments, jobs):
    executable = arguments.executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Build the benchmark first; executable not found: {executable}")

    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    for job_name, benchmark_arguments in jobs:
        output_file = output_directory / f"{job_name}.out"
        benchmark_command = [str(executable), *benchmark_arguments]
        submit_command = [
            "sbatch",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            f"--time={arguments.time}",
            "--open-mode=truncate",
            f"--job-name={job_name}",
            f"--output={output_file}",
            f"--chdir={REPOSITORY_ROOT}",
        ]
        if arguments.partition:
            submit_command.append(f"--partition={arguments.partition}")
        if arguments.account:
            submit_command.append(f"--account={arguments.account}")
        submit_command.extend(["--wrap", shlex.join(benchmark_command)])

        print(shlex.join(submit_command), flush=True)
        if not arguments.dry_run:
            subprocess.run(submit_command, check=True)

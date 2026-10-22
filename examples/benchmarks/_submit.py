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
    parser.add_argument("--time", help="Slurm time limit.")
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

    machine, _, _ = get_machine()
    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    for job_name, benchmark_arguments in jobs:
        result_name = f"{job_name}_gpu" if machine == "matrix" else job_name
        output_file = output_directory / f"{result_name}.out"
        benchmark_command = [str(executable), *benchmark_arguments]
        partition = arguments.partition
        if partition is None and machine == "dane":
            partition = "pdebug"
        time_limit = arguments.time
        if time_limit is None:
            time_limit = "01:00:00" if partition == "pdebug" else "02:00:00"
        submit_command = [
            "sbatch",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            f"--time={time_limit}",
            "--open-mode=truncate",
            f"--job-name={result_name}",
            f"--output={output_file}",
            f"--chdir={REPOSITORY_ROOT}",
        ]
        if partition:
            submit_command.append(f"--partition={partition}")
        if arguments.account:
            submit_command.append(f"--account={arguments.account}")
        if machine == "matrix":
            submit_command.extend(["--gpus", "1"])
        submit_command.extend(["--wrap", shlex.join(benchmark_command)])

        print(shlex.join(submit_command), flush=True)
        if not arguments.dry_run:
            subprocess.run(submit_command, check=True)

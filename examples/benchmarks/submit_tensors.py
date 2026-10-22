#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from _submit import make_parser, positive_integer, submit_jobs


def main():
    parser = make_parser(
        "Submit representative tensor permutation and contraction benchmarks.",
        "benchmark_tensors",
        "tensors",
    )
    parser.add_argument("--mode-size", type=positive_integer, default=50)
    parser.add_argument("--repetitions", type=positive_integer, default=10)
    parser.add_argument("--precision", type=int, default=2, choices=(1, 2))
    arguments = parser.parse_args()

    kernels = (
        ("permutation_4d", 2, 5),
        ("contraction_single", 9, 2),
        ("contraction_double", 11, 4),
        ("contraction_triple", 13, 3),
    )
    jobs = []
    for label, kernel, largest_mode_coefficient in kernels:
        target_mode_size = arguments.mode_size
        if label.startswith("contraction_"):
            target_mode_size *= 2
        size_factor = max(1, target_mode_size // largest_mode_coefficient)
        jobs.append(
            (
                f"tensor_{label}",
                [
                    "--which",
                    str(kernel),
                    "--size_factor",
                    str(size_factor),
                    "--precision",
                    str(arguments.precision),
                    "--repetitions",
                    str(arguments.repetitions),
                ],
            )
        )
    submit_jobs(arguments, jobs)


if __name__ == "__main__":
    main()

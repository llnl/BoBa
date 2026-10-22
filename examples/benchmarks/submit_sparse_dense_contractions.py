#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from _submit import make_parser, positive_integer, submit_jobs


def main():
    parser = make_parser(
        "Submit mixed sparse/dense and sparse/sparse contraction benchmarks across sparsity levels.",
        "benchmark_sparse_dense_contractions",
        "sparse_dense_contractions",
    )
    parser.add_argument("--mode-size", type=positive_integer, default=100)
    parser.add_argument("--repetitions", type=positive_integer, default=10)
    arguments = parser.parse_args()

    sparsities = (
        ("0p01", 10000),
        ("0p1", 1000),
        ("1", 100),
        ("10", 10),
    )
    jobs = []
    for percent_label, sparsity_period in sparsities:
        jobs.append(
            (
                f"sparse_dense_{percent_label}_percent",
                [
                    "--mode-size",
                    str(arguments.mode_size),
                    "--sparsity-period",
                    str(sparsity_period),
                    "--repetitions",
                    str(arguments.repetitions),
                ],
            )
        )
    submit_jobs(arguments, jobs)


if __name__ == "__main__":
    main()

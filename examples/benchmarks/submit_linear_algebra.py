#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from _submit import make_parser, positive_integer, submit_jobs


def main():
    parser = make_parser(
        "Submit the large linear-algebra benchmark matrix shapes.",
        "benchmark_linear_algebra",
        "linear_algebra",
    )
    parser.add_argument("--repetitions", type=positive_integer, default=10)
    arguments = parser.parse_args()

    shapes = ((1000, 10000), (10000, 1000), (3000, 3000))
    jobs = []
    for rows, columns in shapes:
        jobs.append(
            (
                f"linear_algebra_{rows}x{columns}",
                [
                    "--rows",
                    str(rows),
                    "--cols",
                    str(columns),
                    "--maximum",
                    str(rows * columns),
                    "--repetitions",
                    str(arguments.repetitions),
                ],
            )
        )
    submit_jobs(arguments, jobs)


if __name__ == "__main__":
    main()

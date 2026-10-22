#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import re
import socket
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import yaml


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "boba-matplotlib"))

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


LINEAR_HEADER = re.compile(
    r"^(?P<backend>\S+) (?P<precision>\S+) (?P<matrix>\S+) "
    r"(?P<rows>\d+)x(?P<columns>\d+) (?P<kernel>\S+)$"
)
ELAPSED = re.compile(r"^ elapsed (?P<time>[0-9.eE+-]+) (?P<unit>\S+)$")
REPETITIONS = re.compile(r"^\s*--repetitions (?P<count>\d+)$")
TENSOR_PREFIXES = (
    "Permutation;",
    "Reduction;",
    "Reduce single;",
    "Reduce double;",
    "Contraction;",
    "Double_contraction;",
    "Triple_contraction;",
)
MATRIX_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "ones",
            "identity",
            "hilbert",
            "vandermonde",
            "laplacian",
            "derivative",
            "low_rank_vandermonde",
        )
    )
}
KERNEL_ORDER = {
    name: index
    for index, name in enumerate(
        ("qr", "svd", "svdx", "lu", "lu_full", "cholesky", "randomized_svd")
    )
}
TENSOR_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "permutation_4d",
            "contraction_single",
            "contraction_double",
            "contraction_triple",
        )
    )
}
LINEAR_SHAPES = ("2000x200", "200x2000", "600x600")
SERIES_ORDER = (
    "cpu-naive",
    "cpu-tpl",
    "gpu-naive",
    "gpu-tpl",
    "cpu",
    "gpu",
)
SERIES_STYLE = {
    "cpu-naive": ("#7B3294", "CPU naive"),
    "cpu-tpl": ("#377EB8", "CPU TPL"),
    "gpu-naive": ("#006D2C", "GPU naive"),
    "gpu-tpl": ("#76B900", "GPU TPL"),
    "cpu": ("#377EB8", "CPU"),
    "gpu": ("#76B900", "GPU"),
}
SPEEDUP_SERIES = (
    ("cpu-naive", "gpu-naive", "#7B3294", "Naive"),
    ("cpu-tpl", "gpu-tpl", "#377EB8", "TPL"),
    ("cpu", "gpu", "#377EB8", "CPU/GPU"),
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot paired CPU/GPU BoBa benchmark results."
    )
    parser.add_argument(
        "campaign",
        nargs="?",
        type=Path,
        default=Path("benchmark_jobs/2026-10-21_cpu"),
        help="Campaign directory containing the three benchmark subdirectories.",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--runtime-yaml",
        type=Path,
        help="Path for parsed runtimes; defaults to runtimes_<hostname>.yaml in the campaign directory.",
    )
    return parser.parse_args()


def result_variant(path):
    return "gpu" if path.stem.endswith("_gpu") else "cpu"


def time_in_milliseconds(value, unit):
    normalized = unit.strip("()")
    if normalized in ("us", "microseconds"):
        return value / 1000.0
    if normalized in ("ms", "milliseconds"):
        return value
    if normalized in ("s", "seconds"):
        return value * 1000.0
    raise ValueError(f"Unsupported timing unit: {unit}")


def measured_average(values, repetitions, warmups, samples):
    expected_repetitions = warmups + samples
    if repetitions != expected_repetitions or len(values) < expected_repetitions:
        return None
    if len(values) % repetitions != 0:
        return None
    values_per_repetition = len(values) // repetitions
    measured_values = values[warmups * values_per_repetition :]
    return fmean(measured_values)


def parse_linear_algebra(directory, warmups, samples, warnings):
    data = defaultdict(dict)
    order = {}
    shape_order = {shape: index for index, shape in enumerate(LINEAR_SHAPES)}

    for path in sorted(directory.glob("linear_algebra_*.out")):
        result_shape = path.stem.removeprefix("linear_algebra_").removesuffix("_gpu")
        if result_shape not in shape_order:
            continue
        variant = result_variant(path)
        repetitions = None
        current = None
        values = defaultdict(list)
        failed = set()

        for raw_line in path.read_text(errors="replace").splitlines():
            repetition_match = REPETITIONS.match(raw_line)
            if repetition_match:
                repetitions = int(repetition_match.group("count"))

            header_match = LINEAR_HEADER.match(raw_line)
            if header_match:
                fields = header_match.groupdict()
                current = (
                    fields["rows"],
                    fields["columns"],
                    fields["matrix"],
                    fields["kernel"],
                )
                continue

            if current is not None and "? fail" in raw_line:
                failed.add(current)

            elapsed_match = ELAPSED.match(raw_line)
            if current is not None and elapsed_match:
                values[current].append(
                    time_in_milliseconds(
                        float(elapsed_match.group("time")), elapsed_match.group("unit")
                    )
                )

        if repetitions is None:
            warnings.append(f"{path}: no repetition count")
            continue

        incomplete = 0
        for key, timings in values.items():
            if key in failed:
                continue
            average = measured_average(timings, repetitions, warmups, samples)
            if average is None:
                incomplete += 1
                continue
            rows, columns, matrix, kernel = key
            if kernel not in KERNEL_ORDER:
                continue
            shape = f"{rows}x{columns}"
            category = f"{rows}x{columns}\n{matrix}\n{kernel}"
            data[category][variant] = average
            order[category] = (
                shape_order.get(shape, 99),
                MATRIX_ORDER.get(matrix, 99),
                KERNEL_ORDER.get(kernel, 99),
            )

        if failed:
            warnings.append(f"{path}: excluded {len(failed)} validation-failing series")
        if incomplete:
            warnings.append(f"{path}: omitted {incomplete} incomplete timing series")

    return data, sorted(data, key=lambda category: order[category])


def parse_tensor_file(path, warmups, samples, warnings):
    repetitions = None
    implementation = None
    values = defaultdict(list)
    validation_failed = False

    for raw_line in path.read_text(errors="replace").splitlines():
        repetition_match = REPETITIONS.match(raw_line)
        if repetition_match:
            repetitions = int(repetition_match.group("count"))
        if "Testing BoBa Fallback schemes" in raw_line:
            implementation = "fallback"
        elif "Testing third party schemes" in raw_line:
            implementation = "third-party"
        elif "? fail" in raw_line:
            validation_failed = True
        elif implementation is not None and raw_line.startswith(TENSOR_PREFIXES):
            fields = [field.strip() for field in raw_line.split(";")]
            if len(fields) != 5:
                continue
            operation, _, shape, duration, unit = fields
            values[(implementation, operation, shape)].append(
                time_in_milliseconds(float(duration), unit)
            )

    if validation_failed:
        warnings.append(f"{path}: excluded because a validation check failed")
        return {}
    if repetitions is None:
        warnings.append(f"{path}: no repetition count")
        return {}

    averages = {}
    incomplete = 0
    for key, timings in values.items():
        average = measured_average(timings, repetitions, warmups, samples)
        if average is None:
            incomplete += 1
        else:
            averages[key] = average
    if incomplete:
        warnings.append(f"{path}: omitted {incomplete} incomplete timing series")
    return averages


def parse_tensors(directory, warmups, samples, warnings):
    data = defaultdict(dict)
    order = {}

    for path in sorted(directory.glob("tensor_*.out")):
        variant = result_variant(path)
        label = path.stem.removeprefix("tensor_").removesuffix("_gpu")
        if label not in TENSOR_ORDER:
            continue
        for (implementation, _, _), average in parse_tensor_file(
            path, warmups, samples, warnings
        ).items():
            category = label.replace("_", " ")
            series = f"{variant}-{'naive' if implementation == 'fallback' else 'tpl'}"
            data[category][series] = average
            order[category] = (
                TENSOR_ORDER.get(label, 99),
            )

    return data, sorted(data, key=lambda category: order[category])


def parse_key_values(line):
    values = {}
    for field in line.split()[1:]:
        key, value = field.split("=", maxsplit=1)
        values[key] = value
    return values


def parse_sparse_dense(directory, warmups, samples, warnings):
    data = defaultdict(dict)
    order = {}
    operation_order = {"dense-sparse": 0, "sparse-dense": 1, "sparse-sparse": 2}
    dense_dense_category = "dense-dense"

    for path in sorted(directory.glob("sparse_dense_*.out")):
        variant = result_variant(path)
        records = [
            parse_key_values(line)
            for line in path.read_text(errors="replace").splitlines()
            if line.startswith("MIXED_TIMING ")
        ]
        if not records:
            warnings.append(f"{path}: no benchmark timing record")
            continue
        for record in records:
            if (
                int(record.get("warmup_repetitions", -1)) != warmups
                or int(record.get("measured_repetitions", -1)) != samples
            ):
                warnings.append(f"{path}: rerun required for the {warmups}+{samples} sampling rule")
                continue
            sparsity = float(record["sparsity_percent"])
            data[dense_dense_category][variant] = float(record["dense_dense_ms"])
            order[dense_dense_category] = (float("inf"), 0)
            for operation, field in (
                ("dense-sparse", "dense_sparse_ms"),
                ("sparse-dense", "sparse_dense_ms"),
                ("sparse-sparse", "sparse_sparse_ms"),
            ):
                if field not in record:
                    continue
                category = f"{sparsity:g}%\n{operation}"
                data[category][variant] = float(record[field])
                order[category] = (sparsity, operation_order[operation])

    return data, sorted(data, key=lambda category: order[category])


def annotate_empty(axis, message):
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=12,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def runtime_records(data):
    return [
        {
            "category": category,
            "series": series_name,
            "runtime_ms": runtime,
        }
        for category, series in data.items()
        for series_name, runtime in series.items()
    ]


def plot_study(title, data, categories, output_path, time_axis_limits=None):
    width = max(12.0, 0.42 * len(categories))
    figure, (time_axis, speedup_axis) = plt.subplots(
        2, 1, figsize=(width, 9), sharex=bool(categories), constrained_layout=True
    )
    figure.suptitle(title, fontsize=16)

    if categories:
        x_positions = list(range(len(categories)))
        series = [
            name
            for name in SERIES_ORDER
            if any(name in data[category] for category in categories)
        ]
        bar_width = 0.18 if len(series) > 2 else 0.38
        center = (len(series) - 1) / 2
        for index, name in enumerate(series):
            color, label = SERIES_STYLE[name]
            offset = (index - center) * bar_width
            present = [
                (category_index + offset, data[category][name])
                for category_index, category in enumerate(categories)
                if name in data[category]
            ]
            if present:
                time_axis.bar(
                    [item[0] for item in present],
                    [item[1] for item in present],
                    width=bar_width,
                    color=color,
                    label=label,
                )
        time_axis.set_yscale("log")
        if time_axis_limits is not None:
            time_axis.set_ylim(*time_axis_limits)
        time_axis.set_ylabel("Mean time (ms), log scale")
        time_axis.grid(axis="y", which="both", alpha=0.25)
        if time_axis.containers:
            time_axis.legend()

        speedup_axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        plotted_speedups = False
        for cpu_series, gpu_series, color, label in SPEEDUP_SERIES:
            paired = [
                (index, data[category][cpu_series] / data[category][gpu_series])
                for index, category in enumerate(categories)
                if cpu_series in data[category] and gpu_series in data[category]
            ]
            if not paired:
                continue
            speedup_axis.plot(
                [item[0] for item in paired],
                [item[1] for item in paired],
                color=color,
                marker="o",
                linewidth=1.5,
                label=label,
            )
            plotted_speedups = True
        if plotted_speedups:
            speedup_axis.set_ylim(bottom=0.0)
            speedup_axis.grid(axis="y", alpha=0.25)
            speedup_axis.legend()
        else:
            annotate_empty(speedup_axis, "Waiting for matching CPU and GPU results")
            speedup_axis.set_xticks(x_positions, categories, rotation=65, ha="right")
        speedup_axis.set_ylabel("Speedup (CPU time / GPU time)")
        speedup_axis.set_xticks(x_positions, categories, rotation=65, ha="right")
    else:
        annotate_empty(time_axis, "No complete, valid results yet")
        annotate_empty(speedup_axis, "Waiting for matching CPU and GPU results")

    time_axis.set_title("A. Mean of repetitions 6–10")
    speedup_axis.set_title("B. GPU speedup")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    cpu_count = sum(
        sum(name.startswith("cpu") for name in data[category]) for category in categories
    )
    gpu_count = sum(
        sum(name.startswith("gpu") for name in data[category]) for category in categories
    )
    pair_count = sum(
        any(
            cpu_series in data[category] and gpu_series in data[category]
            for cpu_series, gpu_series, _, _ in SPEEDUP_SERIES
        )
        for category in categories
    )
    print(
        f"Wrote {output_path} "
        f"({cpu_count} CPU series, {gpu_count} GPU series, {pair_count} pairs)"
    )


def main():
    arguments = parse_arguments()
    if arguments.warmups < 0 or arguments.samples <= 0:
        raise ValueError("warmups must be nonnegative and samples must be positive")

    campaign = arguments.campaign.resolve()
    warnings = []
    linear_data, linear_categories = parse_linear_algebra(
        campaign / "linear_algebra",
        arguments.warmups,
        arguments.samples,
        warnings,
    )
    for shape in LINEAR_SHAPES:
        shape_prefix = f"{shape}\n"
        for kernel in KERNEL_ORDER:
            kernel_suffix = f"\n{kernel}"
            selected_categories = [
                category
                for category in linear_categories
                if category.startswith(shape_prefix) and category.endswith(kernel_suffix)
            ]
            if not selected_categories:
                continue
            matrix_categories = [
                category.removeprefix(shape_prefix).removesuffix(kernel_suffix)
                for category in selected_categories
            ]
            kernel_data = {
                matrix: linear_data[category]
                for matrix, category in zip(matrix_categories, selected_categories)
            }
            plot_study(
                f"Linear algebra: {shape}, {kernel}",
                kernel_data,
                matrix_categories,
                campaign / f"linear_algebra_{shape}_{kernel}.png",
                time_axis_limits=(1.0, 300.0),
            )

    runtime_data = {"linear_algebra": runtime_records(linear_data)}
    studies = (
        (
            "tensors",
            "Tensor kernels",
            parse_tensors,
            campaign / "tensors",
            campaign / "tensors.png",
        ),
        (
            "sparse_dense_contractions",
            "Sparse/dense contractions",
            parse_sparse_dense,
            campaign / "sparse_dense_contractions",
            campaign / "sparse_dense_contractions.png",
        ),
    )

    for name, title, parser, directory, output_path in studies:
        data, categories = parser(
            directory, arguments.warmups, arguments.samples, warnings
        )
        plot_study(title, data, categories, output_path)
        runtime_data[name] = runtime_records(data)

    hostname = socket.gethostname()
    safe_hostname = re.sub(r"[^A-Za-z0-9_.-]", "_", hostname)
    runtime_yaml = arguments.runtime_yaml
    if runtime_yaml is None:
        runtime_yaml = campaign / f"runtimes_{safe_hostname}.yaml"
    else:
        runtime_yaml = runtime_yaml.resolve()
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(
        yaml.safe_dump(
            {
                "hostname": hostname,
                "campaign": str(campaign),
                "warmups": arguments.warmups,
                "samples": arguments.samples,
                "units": "milliseconds",
                "runtimes": runtime_data,
            },
            sort_keys=False,
        )
    )
    print(f"Wrote {runtime_yaml}")

    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()

```
python3 examples/benchmarks/submit_sparse_dense_contractions.py \
  --mode-size 30 \
  --repetitions 10 \
  --output-dir benchmark_jobs/2026-10-21_cpu/sparse_dense_contractions

python3 examples/benchmarks/submit_tensors.py \
  --mode-size 60 \
  --precision 2 \
  --repetitions 10 \
  --output-dir benchmark_jobs/2026-10-21_cpu/tensors

python3 examples/benchmarks/submit_linear_algebra.py \
  --repetitions 10 \
  --output-dir benchmark_jobs/2026-10-21_cpu/linear_algebra
  ```

# Running the BoBa benchmarks

Run these commands from the repository root. The submission scripts create one
Slurm job per study point and write stdout and stderr to a dedicated `.out`
file. Each kernel is run 10 times by default.

## Build

Build the benchmarks for the backend configured on the current machine:

```sh
make -j 20 benchmarks
```

Do not build the performance runs with `BOBA_CI=1`. That configuration reduces
the repetition count and makes the sparse/dense executable run its small CI
checks instead of the large benchmark.

By default, each submission script selects the corresponding executable in the
repository root, such as `benchmark_tensors_cpu.out`. Use `--executable` to
select a different build explicitly.

## Choose an output directory

Use a new directory for every machine, backend, build configuration, or source
revision. Slurm opens each output file in truncate mode, so reusing a directory
overwrites the previous study results.

For example:

```sh
mkdir -p benchmark_jobs/2026-10-21_cpu
git rev-parse HEAD > benchmark_jobs/2026-10-21_cpu/git_commit.txt
```

The examples below use this campaign directory. Replace the date and backend
with the values for the actual run.

## Preview submissions

Add `--dry-run` to print the `sbatch` commands without submitting them:

```sh
python3 examples/benchmarks/submit_linear_algebra.py --dry-run
python3 examples/benchmarks/submit_tensors.py --dry-run
python3 examples/benchmarks/submit_sparse_dense_contractions.py --dry-run
```

All three scripts also accept `--time`, `--partition`, `--account`,
`--executable`, and `--output-dir`. On Matrix, the scripts automatically add
`--gpus 1` to every Slurm allocation and append `_gpu` to every job and result
name. For example, `linear_algebra_2000x200.out` becomes
`linear_algebra_2000x200_gpu.out`. This allows CPU and GPU results to coexist
in one output directory.

## Linear-algebra study

Submit the 2000 x 200, 200 x 2000, and 600 x 600 matrix studies:

```sh
python3 examples/benchmarks/submit_linear_algebra.py \
  --repetitions 10 \
  --output-dir benchmark_jobs/2026-10-21_cpu/linear_algebra \
  --time 02:00:00
```

This produces:

```text
linear_algebra_2000x200.out
linear_algebra_200x2000.out
linear_algebra_600x600.out
```

Each file contains measurements for the seven matrix families (`ones`,
`identity`, `hilbert`, `vandermonde`, `laplacian`, `derivative`, and
`low_rank_vandermonde`) and every supported factorization kernel. A timing is
reported as an operation header followed later by an elapsed-time line:

```text
CPU double ones 2000x200 qr
...
 elapsed <time> microseconds
```

Records ending in `skipped` are unsupported or inapplicable combinations and
must not be treated as zero-time measurements.

For plotting, retain one row per repetition with these columns:

```text
backend,rows,columns,matrix,kernel,repetition,time_us
```

Plot time by shape and kernel, and use matrix family as a color or facet. Use a
median with an interquartile range when reducing the 10 samples; retain the
individual samples in the processed data.

## Tensor study

Submit the representative permutation with its largest modes near 50 and the
contractions with their largest modes near 100:

```sh
python3 examples/benchmarks/submit_tensors.py \
  --mode-size 50 \
  --precision 2 \
  --repetitions 10 \
  --output-dir benchmark_jobs/2026-10-21_cpu/tensors \
  --time 02:00:00
```

`--precision 1` selects single precision and `--precision 2` selects double
precision. The command above produces:

```text
tensor_permutation_4d.out
tensor_contraction_single.out
tensor_contraction_double.out
tensor_contraction_triple.out
```

Timing records are semicolon-delimited and end with the duration and unit:

```text
Permutation; <backend>; <input-shape>-><output-shape>; <time>; (us)
Contraction; <backend>; <input-shapes>-><output-shape>; <time>; (us)
```

Each job first reports 10 samples from the BoBa fallback implementation and
then 10 samples from the third-party implementation. Preserve that section as
an `implementation` field when parsing the log. Suggested plotting columns
are:

```text
backend,kernel,implementation,input_shape,output_shape,repetition,time_us
```

Plot fallback and third-party results separately or distinguish them by color.
Do not average different tensor shapes or kernel types together.

## Sparse/dense contraction study

Submit mode-size-30 contractions at 0.01%, 0.1%, 1%, and 10% nonzeros:

```sh
python3 examples/benchmarks/submit_sparse_dense_contractions.py \
  --mode-size 30 \
  --repetitions 10 \
  --output-dir benchmark_jobs/2026-10-21_cpu/sparse_dense_contractions \
  --time 02:00:00
```

This produces:

```text
sparse_dense_0p01_percent.out
sparse_dense_0p1_percent.out
sparse_dense_1_percent.out
sparse_dense_10_percent.out
```

Each file contains one plot-ready `MIXED_TIMING` record. Times are the mean per
contraction over the final five repetitions; the preceding repetitions are
warmups:

```text
MIXED_TIMING sparsity=<fraction> sparsity_percent=<percent> mode_size=30 repetitions=10 warmup_repetitions=5 measured_repetitions=5 dense_dense_ms=<time> dense_sparse_ms=<time> sparse_dense_ms=<time> sparse_sparse_ms=<time> sparse_A_nnz=<count> sparse_B_nnz=<count>
```

Extract only these records when creating a table:

```sh
rg '^MIXED_TIMING ' benchmark_jobs/2026-10-21_cpu/sparse_dense_contractions
```

Use the key/value names from the record as table columns. Plot
`sparsity_percent` on a logarithmic x-axis and the four `*_ms` fields as
separate timing series. Keep the nonzero counts in the processed data so the
requested sparsity can be checked against the realized tensors.

The mixed sparse/dense contraction benchmark constructs its sparse inputs on the
host and copies them to the selected execution space before timing. GPU runs
therefore measure device contractions, including the selected naive or
third-party sparse backend.

## After submission

Check the jobs with the site-provided Slurm commands, for example `squeue` and
`sacct`. Before plotting, confirm that every expected output file exists, that
the jobs exited successfully, and that no validation check failed. Keep the
raw `.out` files unchanged; write parsed CSV files and plots beside them or in
a separate `processed` directory. This makes plots reproducible and allows a
new parser to be applied to the original measurements later.

Generate the study figures with:

```sh
python3 examples/benchmarks/plot_benchmarks.py benchmark_jobs/2026-10-21_cpu
```

This writes one `linear_algebra_<shape>_<kernel>.png` for each measured
shape/kernel combination, plus `tensors.png` and
`sparse_dense_contractions.png`, in the campaign directory. Each linear-algebra
figure shows every matrix type once. Panel A uses a logarithmic time axis.
Panel B uses a linear axis for CPU time divided by GPU time. Only repetitions 6
through 10 are averaged; incomplete jobs and failed validation cases are
omitted. The parsed mean runtimes are also written to
`runtimes_<hostname>.yaml` in the campaign directory. The script can be rerun
safely as jobs finish.

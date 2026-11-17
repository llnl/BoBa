# Matrix norm benchmark

Run the CUDA sweep from the repository root with
`./examples/benchmarks/matrix_norms/benchmark_matrix_norms.bash`.
Results are written to `results/matrix_norms` by default.

Build with `make test_matrix_norms BOBA_CI=1`. The executable suffix follows the selected backend, such as `test_matrix_norms_cpu.out`.

Run correctness checks and collect average wall times per call:

```sh
./test_matrix_norms_cpu.out --rows 1024 --cols 1024 --timing --repetitions 50
```

Matrix construction and copying happen before timing. Checks compare host and device results with independently computed host values. The timed calls cover `matrix_one_norm`, `matrix_inf_norm`, and `norm_frobenius` with default and `force_naive` paths. Matrix one-norm is the maximum column sum; matrix infinity-norm is the maximum row sum.

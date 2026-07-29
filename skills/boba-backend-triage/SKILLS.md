# boba-backend-triage

Use this skill when a bug or build failure appears only on CPU, CUDA, HIP, MPI, debug, CI, or sanitizer variants.

## Triage Order

1. Confirm the failing variant and the exact command used.
2. Check whether the failure is already encoded as unsupported in `ci.yaml` or the `Makefile`.
3. Inspect `Makefile_boba` and `CMakeLists.txt` for variant-specific logic.
4. Reduce to the smallest failing target.

## Important Places

- `Makefile_boba` for flags, compilers, and install roots
- `ci.yaml` for skip behavior and environment overrides
- `examples/tests/CMakeLists.txt` for backend-specific CMake filtering
- `examples/exercises/CMakeLists.txt` for backend-specific example filtering
- `recipes/` and vendored TPL host configs for machine-specific build expectations

## Useful Commands

```bash
make test_boba_tensor_train
make test_boba_tensor_train BOBA_DEBUG=1
make test_boba_tensor_train BOBA_CI=1
make test_boba_tensor_train BOBA_ASAN=1
make test_boba_tensor_train BOBA_UBSAN=1
```

Change the target and backend flags as needed.

## Heuristics

- CPU-only failures often point to stricter warnings, Eigen paths, or code paths skipped elsewhere.
- CUDA/HIP-only failures often come from unsupported code paths, backend filtering, or compiler limitations.
- `BOBA_CI=1` can expose warnings that are ignored locally.
- Sanitizer failures should be treated as real until proven otherwise.

## Avoid

- Do not “fix” a backend issue by broadening skip lists unless that is the explicit outcome.
- Do not assume a CMake filter and a `Makefile` conditional stay in sync automatically.

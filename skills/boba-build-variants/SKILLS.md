# boba-build-variants

Use this skill when you need to build or verify BoBa for a specific backend or toolchain variant.

## What To Check First

- Read `README.md` for the public developer workflow.
- Read `Makefile_boba` for actual compiler, warning, sanitizer, and backend behavior.
- Read top-level `CMakeLists.txt` when the task is on the CMake path rather than the `Makefile` path.

## Variant Flags

Common `make` flags:

- `BOBA_CPU=1`
- `BOBA_CUDA=1` for nvidia devices
- `BOBA_HIP=1` for AMD devices
- `BOBA_ENABLE_MPI=1`
- `BOBA_DEBUG=1` for a debug build to find problems and enable extra asserts
- `BOBA_CI=1` for strict warning checks
- `BOBA_ASAN=1` for address sanitizer
- `BOBA_UBSAN=1` for undefined behavior sanitizer

Important notes:

- `Makefile_boba` defaults to a CPU-style build if neither `BOBA_CUDA` nor `BOBA_HIP` is set.
- `BOBA_CI=1` promotes many warnings to errors.
- Hostname-dependent logic in `Makefile_boba` changes compiler choice and install paths.
- CUDA and HIP builds may require newer CMake or machine-specific toolchains.

## Preferred Verification

Use targeted builds first:

```bash
make test_boba_tensor_train
make test_boba_tensor_train BOBA_DEBUG=1
make test_boba_tensor_train BOBA_CI=1
```

Only run a broad compile sweep when the change is wide enough to justify it:

```bash
make clean && make all -j 10
```

## CMake Path

When the task touches CMake wiring or export logic:

- inspect `CMakeLists.txt`
- inspect `cmake/SetupMacros.cmake`
- inspect `examples/*/CMakeLists.txt`
- validate with a focused CMake configure/build if practical

## Avoid

- Do not assume a CPU pass implies CUDA or HIP correctness.
- Do not change host-specific compiler logic casually.
- Do not update only CMake or only `Makefile` without checking whether both paths matter.

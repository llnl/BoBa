# BoBa Skills

This directory holds local repository-specific skills for working in BoBa.

Check this file first when looking for repo-local guidance beyond `AGENTS.md`.

Available skills:

- `boba-build-variants`
  - Use when choosing or running CPU, CUDA, HIP, MPI, debug, CI, ASAN, or UBSAN builds.
- `boba-debugging`
  - Use when debugging with `checkpoint()`, `always_checkpoint()`, `boba_print(...)`, `BOBA_CHECKPOINTS=1`, or `BOBA_DEBUG=1`.
- `boba-test-authoring`
  - Use when adding or editing tests in `examples/tests/` or updating `ci.yaml`.
- `boba-cmake-and-makefile`
  - Use when wiring new examples, tutorials, tests, or library sources into the build.
- `boba-backend-triage`
  - Use when a failure only happens on one backend or build variant.
- `boba-cali-macros`
  - Use when placing, moving, or reviewing `BOBA_CALI_MARK`, `BOBA_CALI_BEGIN`, `BOBA_CALI_SWITCH`, and `BOBA_CALI_END` in BoBa code.
- `boba-caliper-build-run`
  - Use when compiling and running BoBa with Caliper, saving `runtime-report` output, or interpreting reported hotspots.

Repository reminders:

- When checking correctness of changes, a good complete check for compilation issues is to try `make clean && make all -j 20 BOBA_CI=1 BOBA_HDF5=1`
- Primary code lives in `include/BOBA/`.
- The repo uses both CMake and a `Makefile` flow.
- `tpl/` is vendored third-party code and should only be touched intentionally.
- Before adding a new test executable, check whether the behavior is already covered by an existing test or miniapp and only needs new `ci.yaml` entries.

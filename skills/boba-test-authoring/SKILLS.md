# boba-test-authoring

Use this skill when adding, modifying, or debugging tests under `examples/tests/`.

## Source Conventions

- Test files are named `examples/tests/test_*.cpp`.
- Most tests include `examples/tests/common.hpp`.
- New tests should follow the repo’s existing check style:
  - accumulate a `bool check`
  - use `pass_or_fail(...)` or `pass_or_fail_bool(...)`
  - return `final_check(check)` from `main`

Inspect existing patterns with:

```bash
rg -n "pass_or_fail|final_check\\(" examples/tests
```

## Build Registration

There are two paths to keep aligned:

- CMake auto-discovers `examples/tests/test_*.cpp` in `examples/tests/CMakeLists.txt`.
- `Makefile` still needs explicit target wiring for test executables used by local workflows.

When adding a new test that should be part of normal developer workflows:

- add a `Makefile` target
- consider adding it to `all`
- add CI coverage in `ci.yaml`

## CI Metadata

`ci.yaml` defines:

- logical test names
- command-line inputs
- environment overrides
- backend skip lists

If your test only works on some backends, encode that explicitly in `ci.yaml`.

## Validation Strategy

Prefer:

```bash
make <new_target>
./<new_target>_cpu.out
```

Adjust the output name for the active backend suffix.

## Avoid

- Do not invent a new result-reporting style when `common.hpp` already fits.
- Do not add a test without checking backend skip behavior.
- Do not rely on CMake auto-discovery alone if the `Makefile` flow is expected to support the test.

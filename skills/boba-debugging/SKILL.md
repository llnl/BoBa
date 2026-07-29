---
name: boba-debugging
description: Debug BoBa code with `checkpoint()`, `always_checkpoint()`, `boba_print(...)`, and debug builds such as `make <target> BOBA_DEBUG=1`. Use when narrowing where a failure occurs, adding temporary runtime breadcrumbs, or printing variable state while a test or example runs.
---

# BoBa Debugging

Use this skill when debugging runtime behavior in `include/BOBA/`, `source/`, or `examples/`.

## Quick Choice

- Use `checkpoint();` when you want a file/function/line breadcrumb and are willing to rebuild with `BOBA_CHECKPOINTS=1`.
- Use `always_checkpoint();` when you want the same breadcrumb without depending on `BOBA_CHECKPOINTS`.
- Use `boba_print(value);` when you want to print a variable or expression at runtime.
- Rebuild with `BOBA_DEBUG=1` when you need a debug-oriented build, including extra checking such as bounds checks.

## Macro Behavior

- `checkpoint()` expands to a host-side print of `function`, `file`, and `line` only when `BOBA_CHECKPOINTS` is defined. Otherwise it compiles away.
- `always_checkpoint()` prints the same breadcrumb on host code without needing `BOBA_CHECKPOINTS`.
- Both checkpoint macros compile away in device-code regions guarded by `BOBA_DEVICE_CODE`.
- `boba_print(x)` prints the expression name and value, then returns the value. It is safe to use either as a standalone line or inline in a larger expression, but standalone calls are usually easier to remove later.

## Build Commands

`make` flow:

```bash
make test_boba_tensor_train BOBA_DEBUG=1
make test_boba_tensor_train BOBA_CHECKPOINTS=1
make test_boba_tensor_train BOBA_DEBUG=1 BOBA_CHECKPOINTS=1
```

CMake flow uses environment variables instead of `make`-style arguments:

```bash
BOBA_DEBUG=1 BOBA_CHECKPOINTS=1 cmake -S . -B build
cmake --build build --target test_boba_tensor_train -j
```

## Placement Guidance

- Add `checkpoint();` or `always_checkpoint();` before and after a suspicious call, branch, or data-motion step to bracket where execution stops or diverges.
- Prefer a small number of well-placed checkpoints over instrumenting every line.
- Use `always_checkpoint();` for temporary triage when you do not want to depend on a special checkpoint build.
- Remove temporary debug prints once the failure is understood.

## `boba_print(...)` Guidance

- Good targets: scalars, booleans, names, dimensions, ranks, residuals, and small containers.
- Typical usage:

```cpp
checkpoint();
boba_print(tensor.name());
boba_print(residual_final);
always_checkpoint();
```

- Because `boba_print(x)` returns `x`, it can also be used in assignments or conditions, but avoid that if it makes the debug path harder to read.

## Common Workflow

1. Rebuild the failing target with `BOBA_DEBUG=1`.
2. If you need execution breadcrumbs, add `checkpoint();` and rebuild with `BOBA_CHECKPOINTS=1`.
3. If you want breadcrumbs without changing build flags, use `always_checkpoint();`.
4. Add `boba_print(...)` next to the checkpoint that first shows suspicious state.

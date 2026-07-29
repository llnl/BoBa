# AGENTS.md

## Scope

These instructions apply to the whole BoBa repository.

## Quick Navigation

- Repository-wide agent guidance lives in this root `AGENTS.md`.
- Repository-local skill index lives in `skills/SKILLS.md`.
- CI invocation metadata lives in `ci.yaml`.
- Makefile build system lives in `Makefile` and `Makefile_boba`.
- CMake discovery and target helpers live in `examples/*/CMakeLists.txt` and `cmake/SetupMacros.cmake`.

## Repository Overview

- BoBa is a mostly header-only C++20 tensor library with a small compiled source layer in `source/`.
- Primary library code lives in `include/BOBA/`.
- The compiled library source is currently just `source/boba.cpp`, wired by `source/CMakeLists.txt`.
- User-facing examples and verification code live under `examples/`:
  - `examples/tests/` for CI-style test executables
  - `examples/tutorials/` for tutorials
  - `examples/exercises/` for larger examples
  - `examples/cmake_examples/` for downstream integration examples
- Build-system logic lives in `Makefile` and `cmake/`.
- Vendored third-party libraries live in `tpl/`. Do not modify `tpl/` unless the task is explicitly about third-party code or vendored patches.

## Build Systems

This repo has two active build paths. Do not assume updating one is enough.

- CMake is the modern path:
  - top-level `CMakeLists.txt`
  - `examples/tests/CMakeLists.txt`
  - `examples/tutorials/CMakeLists.txt`
  - `examples/exercises/CMakeLists.txt`
  - `cmake/SetupMacros.cmake`
- `Makefile` and `Makefile_boba` are still important for local workflows, README instructions, and CI-style developer usage.
- When testing changes, prefer using `make all BOBA_CI=1 -j 20` to quickly check for compilation correctness

## Discovery Rules

- CMake auto-discovers:
  - `examples/tests/test_*.cpp`
  - `examples/tutorials/tutorial_*.cpp`
  - top-level `examples/exercises/example_*.cpp`
- The `Makefile` manually wires many targets, including nested exercises and the MATLAB tutorial flow.
- If you add a new test/exercises/tutorial, check whether both CMake and `Makefile` need updates.

## Test Conventions

- Test sources follow the `examples/tests/test_*.cpp` naming pattern.
- Tests typically include `examples/tests/common.hpp`.
- Use the existing `pass_or_fail(...)`, `pass_or_fail_bool(...)`, and `final_check(check)` conventions for new tests.
- New tests should return `final_check(check)` from `main`.
- Before adding a brand-new test file, first check whether the changed behavior is already exercised by an existing test or miniapp and only needs additional `ci.yaml` coverage.
- If you add a new test:
  - add a target to `Makefile`
  - add it to the `all` target if appropriate
  - add CI coverage in `ci.yaml`, including `skip` rules for unsupported backends

## Backend And Variant Awareness

- BoBa supports CPU, CUDA, HIP, and optional MPI-oriented builds.
- The repo uses variant flags such as `BOBA_CPU`, `BOBA_CUDA`, `BOBA_HIP`, `BOBA_ENABLE_MPI`, `BOBA_DEBUG`, `BOBA_CI`, `BOBA_ASAN`, and `BOBA_UBSAN`.
- `Makefile_boba` also changes compiler choice and TPL paths by host and backend.
- HIP/Clang is sensitive to attribute ordering on declarations.
  Keep standard C++ attributes such as `[[nodiscard]]` before `__boba_host_device__`, `__boba_host__`, or `__boba_device__`.
  Do not place `[[...]]` after those macros on the following line.
- Some tests/examples are intentionally skipped for some backends. Before changing that behavior, inspect:
  - `ci.yaml`
  - conditional logic in `Makefile`
  - filtering in `examples/tests/CMakeLists.txt` and `examples/exercises/CMakeLists.txt`

## Verification Expectations

- Prefer targeted verification over blanket rebuilds.
- Good first checks:
  - `make <target>`
  - `make <target> BOBA_DEBUG=1`
  - `make <target> BOBA_CI=1`
- If a change affects CMake wiring, also validate with a CMake configure/build when practical.
- Follow the README guidance that `make clean && make all -j <N>` is the broad compile sanity check, but do not run a full rebuild unless the change justifies it.

## File-Specific Guidance

- `README.md`, `README_CONTRIBUTING.md`, and `examples/cmake_examples/README.md` describe real workflows. Keep them aligned with code changes.
- `ci.yaml` is the source of test invocation metadata for CI-style runs.
- When a behavior change can be covered by extra invocations of an existing executable, prefer adding new named cases in `ci.yaml` over creating a new test executable.
- `cmake/SetupMacros.cmake` defines `boba_add_executable`, `boba_add_test`, and related conventions.
- `examples/tests/common.hpp` defines the check/reporting macros used across tests.
- Repo-local Codex skills live under `skills/`.
- `skills/SKILLS.md` is the local index of repository-specific skills.
- When adding or updating a repo skill, prefer a dedicated folder such as `skills/<skill-name>/` and keep the skill instructions there.

## Documentation Style

- Less is more. Prefer code self-documentation for simple cases:
  - clear names
  - small functions
  - obvious control flow
- Do not add comments that merely restate the code line by line.
- Document functions when the signature and implementation do not already make the behavior clear.
- When documentation is added to functions, classes, or other APIs, use Doxygen-style comments.
- Prioritize documentation for:
  - public or widely reused APIs
  - non-obvious invariants
  - ownership or lifetime expectations
  - backend-specific behavior
  - numerical assumptions, tolerances, or algorithmic tradeoffs
  - surprising preconditions or failure modes
- Keep function documentation concise. A short summary plus a note on inputs, outputs, and important constraints is usually enough.
- Prefer concise Doxygen blocks over long prose. Document what the API means, not what each obvious line of code does.
- For internal helper functions, prefer improving naming and structure before adding prose.
- For complicated algorithms that could be simply demonstrated with a text example, add this to the documentation
- If a block of code is hard to understand, first see whether extracting a helper or renaming variables makes the comment unnecessary.

## Change Boundaries

- Avoid broad style churn in headers. This codebase is large and warning-sensitive.
- Prefer small, targeted changes in `include/BOBA/` and matching tests in `examples/tests/`.
- Do not silently “fix” unrelated files in `tpl/`, machine recipes, or host configs while working on a local feature.
- Directories such as `tpl/`, `build/`, `install/`, `documentation/` should be treated as read-only.

## Useful Commands

- Find code quickly: `rg <pattern> include source examples cmake`
- Find top-level docs: `rg --files -g 'README*'`
- Inspect test conventions: `rg -n "pass_or_fail|final_check\\(" examples/tests`
- Check tracked changes before editing: `git status --short`

## Token Savers

- Keep responses short and task-focused.
- Do not explain obvious code unless asked.
- Prefer minimal diffs over full-file rewrites.
- Reuse existing patterns and utilities before adding new abstractions.
- Only include code relevant to the requested change.
- Avoid marketing, filler, and motivational language.
- Summarize tradeoffs in 1 to 3 bullets when needed.
- Ask questions before making large changes to ensure you are going in the right direction
- Default to concise comments, add them only where logic is not obvious.
- For code generation, prefer simple, maintainable solutions over elaborate designs.
- Do not restate the prompt or repeat requirements unnecessarily.
- When listing options, give the best option first and keep alternatives brief.

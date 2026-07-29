# boba-cmake-and-makefile

Use this skill when wiring new code into BoBa’s build systems.

## Key Fact

BoBa has both:

- a CMake build
- a hand-written `Makefile` build

For many tasks, both must be updated.

## Auto-Discovery Rules

CMake auto-discovers:

- `examples/tests/test_*.cpp`
- `examples/tutorials/tutorial_*.cpp`
- top-level `examples/exercises/example_*.cpp`

CMake does not auto-discover nested special cases like:

- `examples/exercises/example_implicit_block/example_implicit_block.cpp`
- `examples/exercises/example_eigel/main.cpp`

Those kinds of cases usually require manual treatment.

## Files To Inspect

- `CMakeLists.txt`
- `source/CMakeLists.txt`
- `examples/tests/CMakeLists.txt`
- `examples/tutorials/CMakeLists.txt`
- `examples/exercises/CMakeLists.txt`
- `cmake/SetupMacros.cmake`
- `Makefile`
- `Makefile_boba`

## Typical Changes

If adding library implementation:

- update `source/CMakeLists.txt` if new compiled sources are required
- confirm public headers are under `include/BOBA/`

If adding a tutorial/test/example:

- verify whether CMake auto-discovers it
- add or update the `Makefile` target when needed
- update `ci.yaml` if it should run in CI

## Validation

- run a targeted `make <target>`
- if CMake was touched, run a focused CMake configure/build
- do not stop after one build path passes if the other path is still user-facing

## Avoid

- Do not assume nested example layouts are automatically handled by CMake.
- Do not remove `Makefile` support unless that change is intentional and coordinated.

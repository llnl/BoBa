# BoBa Tensor Decomposition Library

[![GitHub release](https://img.shields.io/github/release/llnl/BoBa.svg)](https://github.com/llnl/BoBa/releases/latest)

Huge datasets and tables, as well as high-dimensional structured problems, are challenging to work with due to their large memory requirements, even by today's standards. This motivates the development of data compression techniques that allow us to work directly with data in its compressed format. The **BoBa library** aims to provide performance-portable abstractions that enable tensor methods that are low-memory, controllably accurate, and efficient across heterogeneous hardware.

BoBa is a (mostly) header-only C++ library containing algorithms for matrices, tensors, and tensor decompositions designed for next-generation heterogeneous architectures. It enables the creation of matrix, tensor, and tensor algorithms that achieve high performance on both laptops and HPC clusters.

BoBa is very much a work in progress. Contributions are welcome!

## Core Features

At its core, BoBa offers a variety of features for matrices and tensors:

- **Performance portability** across CPUs and GPUs for dense matrix and tensor algebra, including support for CUDA, HIP, and [Eigen](https://gitlab.com/libeigen/eigen).
- **Dense, memory-owning object abstractions** for arrays, vectors, matrices, and tensors that follow the "resource allocation is initialization" philosophy.
- Capabilities to create **views** of such objects, including 'atomic' and 'const' views.
- Routines for calling open-source and vendor linear algebra operations, including **LU**, **Cholesky**, **QR**, **SVD**, and more.
- Routines for calling open-source and vendor tensor algebra operations, including **permutations**, **contractions**, **reductions**, and more.
- **Portable multi-dimensional loops and reductions** enabled by [RAJA](https://github.com/LLNL/RAJA).
- **Dense object memory pooling** enabled by [Umpire](https://github.com/LLNL/Umpire).
- **Static (compile-time sized) versions** of matrices and tensors, with relevant algorithms.
- A **C++20 feature set**.

## Tensor Decomposition Library

Built atop BoBa's core functionality is a performance-portable tensor library that includes implementations for arbitrary-dimensional tensor decompositions:

- **tensor trains**
- **Tucker decompositions**
- **Hierarchical Tucker decompositions**
- **Canonical Polyadic decompositions**
- Matrix/operator variations of some of the above, e.g. **tensor train matrices**
- **Static (compile-time sized) views** of some of these decompositions, useful for reading data from decompositions that were computed offline

Further, BoBa offers implementations for arbitrary-dimensional **multilinear and nonlinear solvers**.

Lastly, tutorials, informative unit tests, and exercises demonstrate all of the above.

## Citation
Please use this citation when citing BoBa
```
  @misc{bobalibrary,
    title  = {BoBa: Performance-portable Tensor Decomposition Library},
    author = {Guthrey, Pierson and
              Blomquist, Matthew Thomas and
              Burmark, Jason and
              Yao, Jin and
              Vuchkov, Radoslav and
              De, Saibal and
              Nelson, Austin and
              Sands, Bill and
              Ricketson, Lee and
              Jones, Holger and
              Walton, Steven and
              Demir, Sinan and
              Joseph, Ilon and
              Minner, Paul and
              Irving, Samuel},
    url    = {https://github.com/llnl/BoBa},
    year   = {2026}
  }
```

# License

BoBa is distributed under the Apache-2.0 with LLVM exception License. See [LICENSE](LICENSE) for
the full terms and [NOTICE](NOTICE) for the LLNL/DOE government notice.

LLNL-CODE-2022014

## Getting Started

- Our tutorials [examples/tutorials](examples/tutorials) explain some BoBa basics and are a great place to start learning the library.
- You can self-generate the Doxygen documentation in the documentation directory to easily find information in HTML format.
- When you feel ready to run some examples, follow the installation instructions below.
- Next, start in the [examples/tests](examples/tests) directory which has very simple examples that demonstrate the interesting features of tensor methods, but also serve as our CI testing suite.
- Then, move on to our [examples/exercises](examples/exercises), which demonstrate some more elaborate use cases of tensor decompositions.
- When you are ready to integrate BoBa into your project, take a look at our [examples/cmake_examples](examples/cmake_examples), which provide exemplary integrations on various platforms.
- Additionally, browse our list of peer-reviewed publications in the bibtex files in [documentation/bibtex](documentation/bibtex)

## Developer Navigation

- CI test matrix and example invocations: [`ci.yaml`](ci.yaml)
- Makefile build wiring: [`Makefile`](Makefile) and [`Makefile_boba`](Makefile_boba)
- CMake target wiring: [`cmake/SetupMacros.cmake`](cmake/SetupMacros.cmake)
- Repository-wide contributor and agent guidance: [`AGENTS.md`](AGENTS.md)
- Repository-local skill index: [`skills/SKILLS.md`](skills/SKILLS.md)

## Installing TPLs

BoBa requires a few third-party libraries (TPLs). Install them by calling the builder directly; it auto-selects the appropriate machine recipe on supported hosts:

```bash
./boba_builder.py
```

You may need to add a recipe and builder logic for your system.

If you are reproducing a CI-style build, add `--ci`.

Third-party dependencies currently include:

- BLT (BSD-3-Clause)
- Caliper (BSD-3-Clause)
- camp (BSD-3-Clause)
- Eigen (MPL-2.0 with additional bundled notices)
- fmt (MIT)
- HDF5 (HDF5 license)
- RAJA (BSD-3-Clause)
- Umpire (MIT)

See [tpl/license/README.md](tpl/license/README.md) for more information.

BoBa's link-time dependencies currently include:

- Apple Accelerate
- Apple Metal
- NVIDIA libraries: cuBLAS, cuSOLVER, cuSPARSE, cuTENSOR
- AMD hipBLAS, hipSOLVER, rocBLAS, rocSOLVER, hipTENSOR

## Quick example

To ensure that everything is ready to go, let's try to build everything.
```
make clean && make all -j 10
```
This is an excellent test to make sure the code compiles for the given system configuration. Now we can start running the built examples, e.g.:
```
./test_boba_tensor_train_cpu.out
```

## Tips and Tricks

- BoBa has [Caliper](https://github.com/LLNL/Caliper) integration, which allows you to see runtime hotspots in the code. This is done by building with the `BOBA_ENABLE_CALIPER=1` flag and exporting the `CALI_CONFIG=runtime-report` environment variable. Use `BOBA_ENABLE_CALIPER_EXTERNAL=1` when you want Caliper scopes in code outside BoBa internals, such as exercises and tests:
```
source caliper_lib_path_info_cpu
make clean && make test_boba_tensor_train BOBA_ENABLE_CALIPER=1
CALI_CONFIG=runtime-report ./test_boba_tensor_train_cpu_cali.out
```
The `caliper_lib_path_info_cpu` file is generated by `./boba_builder.py`.

- When building, add the `BOBA_CHECKPOINTS=1` flag to activate checkpointing throughout the code so that you can find the spot where your code runs into a bug.
```
make clean && make test_boba_tensor_train BOBA_CHECKPOINTS=1 && ./test_boba_tensor_train_cpu.out
```

- When building, add the flag `BOBA_DEBUG=1` to (for example) perform bounds checking on all array accesses (except inside GPU kernels) and generate other debugging information. This may help find out-of-bounds array access and other pesky bugs.
```
make clean && make test_boba_tensor_train BOBA_DEBUG=1 && ./test_boba_tensor_train_cpu_debug.out
```

## Quick example on AMD GPUs (on LC)

- Log on to a machine such as `tuolumne`, then try to build and run a test on the GPU
```
make clean && make test_boba_tensor_train BOBA_HIP=1
```
- Now run the produced file
```
./test_boba_tensor_train_hip.out
```

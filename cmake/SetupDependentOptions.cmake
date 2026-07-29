# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


##
## Here are the CMake dependent options in BOBA.
##

cmake_dependent_option(BOBA_ENABLE_OPENMP "Build with OpenMP support" On "ENABLE_OPENMP" Off)
cmake_dependent_option(BOBA_ENABLE_CUDA "Build with CUDA support" On "ENABLE_CUDA" Off)
cmake_dependent_option(BOBA_ENABLE_HIP "Build with HIP support" On "ENABLE_HIP" Off)

cmake_dependent_option(BOBA_ENABLE_COVERAGE "Enable coverage (only supported with GCC)" On "ENABLE_COVERAGE" Off)
cmake_dependent_option(BOBA_ENABLE_TESTS "Build tests" On "ENABLE_TESTS" Off)
cmake_dependent_option(BOBA_ENABLE_EXAMPLES "Build simple examples" On "ENABLE_EXAMPLES" off)
cmake_dependent_option(BOBA_ENABLE_TUTORIALS "Build simple examples" On "ENABLE_TUTORIALS" off)
cmake_dependent_option(BOBA_ENABLE_BENCHMARKS "Build benchmarks" On "ENABLE_BENCHMARKS" Off)

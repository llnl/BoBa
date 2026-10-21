# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FILE(GLOB boba_benchmarks test_*.cpp)

foreach(benchmark ${boba_benchmarks})
  get_filename_component(basename "${benchmark}" NAME_WLE)
  message(STATUS "Adding benchmark: ${basename}")
  boba_add_benchmark(
    NAME ${basename}${boba_nameflag}
    SOURCES ${basename}.cpp
  )
endforeach()

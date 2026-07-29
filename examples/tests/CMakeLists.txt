# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


FILE(GLOB boba_tests test_*.cpp)
if(BOBA_ENABLE_HIP OR BOBA_ENABLE_CUDA)
  filter_regex(EXCLUDE "test_add_reduce" boba_tests ${boba_tests})
  filter_regex(EXCLUDE "test_cp" boba_tests ${boba_tests})
  filter_regex(EXCLUDE "test_cross" boba_tests ${boba_tests})
  filter_regex(EXCLUDE "test_explicit_burgers" boba_tests ${boba_tests})
  filter_regex(EXCLUDE "test_cur" boba_tests ${boba_tests})
endif()

foreach(test ${boba_tests})
  get_filename_component(basename "${test}" NAME_WLE)
  message(STATUS "Adding test: ${basename}")
  boba_add_test(
    NAME ${basename}${boba_nameflag}
    SOURCES ${basename}.cpp
  )
endforeach()

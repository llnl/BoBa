# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


FILE(GLOB boba_examples example_*.cpp)
if(BOBA_ENABLE_HIP OR BOBA_ENABLE_CUDA)

endif()
filter_regex(EXCLUDE "parallel" boba_examples ${boba_examples})

foreach(example ${boba_examples})
  get_filename_component(basename "${example}" NAME_WLE)
  message(STATUS "Adding example: ${basename}")
  boba_add_executable(
    NAME ${basename}${boba_nameflag}
    SOURCES ${basename}.cpp
  )
endforeach()

add_subdirectory(example_implicit_block)
add_subdirectory(example_eigel)

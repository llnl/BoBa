# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception



FILE(GLOB boba_tutorials tutorial_*.cpp)
foreach(tutorial ${boba_tutorials})
  get_filename_component(basename "${tutorial}" NAME_WLE)
  message(STATUS "Adding tutorial: ${tutorial}")
  boba_add_executable(
    NAME ${basename}${boba_nameflag}
    SOURCES ${basename}.cpp
  )
endforeach()

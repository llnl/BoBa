# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(BOBA_ENABLE_TUTORIALS)
  add_subdirectory(tutorials)
endif()
if(BOBA_ENABLE_EXAMPLES)
  add_subdirectory(exercises)
endif()
if(BOBA_ENABLE_TESTS)
  add_subdirectory(tests)
endif()

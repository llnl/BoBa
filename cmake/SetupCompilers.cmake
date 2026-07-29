# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0" CACHE STRING "")

if (BOBA_ENABLE_MODULES)
  message(WARNING "BOBA_ENABLE_MODULES is deprecated, please add the -fmodules flag manually if desired.")
  set(BOBA_ENABLE_MODULES Off CACHE BOOL "" FORCE)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3)
    message(FATAL_ERROR "BOBA requires GCC 8.3 or greater!")
  endif ()
  if (BOBA_ENABLE_COVERAGE)
    if(NOT BOBA_ENABLE_CUDA)
      message(INFO "Coverage analysis enabled")
      set(CMAKE_CXX_FLAGS "-coverage ${CMAKE_CXX_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "-coverage ${CMAKE_EXE_LINKER_FLAGS}")
    endif()
  endif()
endif()

set(BOBA_COMPILER "BOBA_COMPILER_${CMAKE_CXX_COMPILER_ID}")

if (BOBA_ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD "11" CACHE STRING "Version of C++ standard for CUDA Builds")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict -arch ${CUDA_ARCH} --extended-lambda --expt-relaxed-constexpr -Xcudafe \"--display_error_number\"")

  if (NOT BOBA_HOST_CONFIG_LOADED)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo -O2")

    if(BOBA_ENABLE_COVERAGE)
      if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
        message(INFO "Coverage analysis enabled")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -coverage -Xlinker -coverage")
        set(CMAKE_EXE_LINKER_FLAGS "-coverage ${CMAKE_EXE_LINKER_FLAGS}")
      else()
        message(WARNING "Code coverage specified but not enabled -- GCC was not detected")
      endif()
    endif()
  endif()
endif()
# end BOBA_ENABLE_CUDA section

if (BOBA_ENABLE_HIP)
  if (NOT BOBA_HOST_CONFIG_LOADED)
    #list(APPEND BOBA_EXTRA_HIPCC_FLAGS)

    set(BOBA_HIPCC_FLAGS_RELEASE -O2 CACHE STRING "")
    set(BOBA_HIPCC_FLAGS_DEBUG -g; -O0 CACHE STRING "")
    set(BOBA_HIPCC_FLAGS_MINSIZEREL -Os CACHE STRING "")
    set(BOBA_HIPCC_FLAGS_RELWITHDEBINFO -g; -O2 CACHE STRING "")

    if(BOBA_ENABLE_COVERAGE)
      set(BOBA_EXTRA_HIPCC_FLAGS ${BOBA_EXTRA_HIPCC_FLAGS}; -fcoverage-mapping)
      set(CMAKE_EXE_LINKER_FLAGS "-fcoverage-mapping ${CMAKE_EXE_LINKER_FLAGS}")
    endif()
  endif()
  set(BOBA_HIPCC_FLAGS ${BOBA_EXTRA_HIPCC_FLAGS} CACHE STRING "")
  set(HIP_HIPCC_FLAGS ${BOBA_HIPCC_FLAGS})
  set(HIP_HIPCC_FLAGS_RELEASE ${BOBA_HIPCC_FLAGS_RELEASE})
  set(HIP_HIPCC_FLAGS_DEBUG ${BOBA_HIPCC_FLAGS_DEBUG})
  set(HIP_HIPCC_FLAGS_MINSIZEREL ${BOBA_HIPCC_FLAGS_MINSIZEREL})
  set(HIP_HIPCC_FLAGS_RELWITHDEBINFO ${BOBA_HIPCC_FLAGS_RELWITHDEBINFO})
endif()
# end BOBA_ENABLE_HIP section

# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


set(BOBA_ENABLE_EIGEN On CACHE BOOL "BOBA Use Eigen3")
set(BOBA_EIGEN_TENSOR On CACHE BOOL "BOBA Use Eigen Tensor (unsupported)")
set(BOBA_ENABLE_APPLE "${APPLE}" CACHE BOOL "BOBA Use Apple Accelerate")
set(BOBA_ENABLE_CALIPER Off CACHE BOOL "BOBA Enable Caliper")
set(BOBA_ENABLE_HDF5 Off CACHE BOOL "BOBA Use HDF5")
set(BOBA_ENABLE_METAL Off CACHE BOOL "BOBA Use Metal GPU backend")
set(BOBA_HIP_LIBS Off CACHE BOOL "BOBA Use ROCM/HIP software stack")
set(BOBA_CUTENSOR "${ENABLE_CUTENSOR}" CACHE BOOL "Legacy compatibility option for cuTENSOR support")
set(BOBA_HIPTENSOR "${ENABLE_HIPTENSOR}" CACHE BOOL "Legacy compatibility option for hiptensor support")
set(BOBA_ENABLE_UMPIRE On CACHE BOOL "BOBA Use Umpire")
set(BOBA_ENABLE_CAMP On CACHE BOOL "BOBA Use Camp")
if(ENABLE_CUDA OR ENABLE_HIP)
set(BOBA_ENABLE_RAJA On CACHE BOOL "BOBA Use RAJA")
set(BOBA_ENABLE_FMT On CACHE BOOL "BOBA Use fmt")
else()
set(BOBA_ENABLE_RAJA Off CACHE BOOL "BOBA Use RAJA")
set(BOBA_ENABLE_FMT Off CACHE BOOL "BOBA Use fmt")
endif()
set(BOBA_ENABLE_WARNINGS_AS_ERRORS Off CACHE BOOL "")
set(ENABLE_GTEST_DEATH_TESTS On CACHE BOOL "Enable tests asserting failure.")

if(BOBA_CUTENSOR)
  set(ENABLE_CUTENSOR On CACHE BOOL "Enable cuTENSOR" FORCE)
else()
  set(ENABLE_CUTENSOR Off CACHE BOOL "Enable cuTENSOR" FORCE)
endif()

if(BOBA_HIPTENSOR)
  set(ENABLE_HIPTENSOR On CACHE BOOL "Enable hiptensor" FORCE)
else()
  set(ENABLE_HIPTENSOR Off CACHE BOOL "Enable hiptensor" FORCE)
endif()

if(BOBA_ENABLE_EIGEN AND BOBA_EIGEN_TENSOR)
  set(BOBA_EIGEN_TENSOR On CACHE BOOL "BOBA Use Eigen Tensor (unsupported)" FORCE)
else()
  set(BOBA_EIGEN_TENSOR Off CACHE BOOL "BOBA Use Eigen Tensor (unsupported)" FORCE)
endif()

set(BLT_EXPORT_THIRDPARTY ON CACHE BOOL "")

#------------------------------------------------------------------------------
# Pickup some options from the environment
#------------------------------------------------------------------------------
if(DEFINED ENV{BOBA_VERBOSE_MEMORY})
  list(APPEND boba_defines "BOBA_VERBOSE_MEMORY")
  set(BOBA_VERBOSE_MEMORY On)
endif()

if(DEFINED ENV{BOBA_DEBUG})
  list(APPEND boba_defines "BOBA_DEBUG")
  set(CMAKE_BUILD_TYPE "Debug")
  set(BOBA_DEBUG On)
endif()

if(DEFINED ENV{BOBA_ENABLE_MPI})
  list(APPEND boba_defines "BOBA_ENABLE_MPI")
  set(BOBA_ENABLE_MPI On)
endif()

if(DEFINED ENV{BOBA_ENABLE_CALIPER})
  set(BOBA_ENABLE_CALIPER On)
endif()

if(DEFINED ENV{BOBA_HDF5})
  set(BOBA_ENABLE_HDF5 On)
endif()

if(DEFINED ENV{BOBA_ENABLE_CALIPER_OBJECTS})
  list(APPEND boba_defines "BOBA_ENABLE_CALIPER_OBJECTS")
  set(BOBA_ENABLE_CALIPER On)
  set(BOBA_ENABLE_CALIPER_OBJECTS On)
endif()

if(DEFINED ENV{BOBA_ENABLE_CALIPER_EXTERNAL})
  list(APPEND boba_defines "BOBA_ENABLE_CALIPER_EXTERNAL")
  set(BOBA_ENABLE_CALIPER On)
  set(BOBA_ENABLE_CALIPER_EXTERNAL On)
endif()

if(DEFINED ENV{BOBA_CHECKPOINTS})
  list(APPEND boba_defines "BOBA_CHECKPOINTS")
  set(BOBA_CHECKPOINTS On)
endif()

if(DEFINED ENV{BOBA_CHECKPOINTS_OBJECTS})
  list(APPEND boba_defines "BOBA_CHECKPOINTS_OBJECTS")
  set(BOBA_CHECKPOINTS_OBJECTS On)
endif()

if(DEFINED ENV{BOBA_PROFILING})
  list(APPEND boba_defines "BOBA_PROFILING")
  set(BOBA_PROFILING On)
endif()

if(DEFINED ENV{BOBA_CI})
  list(APPEND boba_defines "BOBA_CI")
  set(BOBA_CI On)
endif()

if(DEFINED ENV{BOBA_ASAN})
  list(APPEND boba_defines "BOBA_ASAN")
  set(BOBA_ASAN On)
endif()

if(DEFINED ENV{BOBA_UBSAN})
  list(APPEND boba_defines "BOBA_UBSAN")
  set(BOBA_UBSAN On)
endif()

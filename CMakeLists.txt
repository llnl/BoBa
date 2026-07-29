# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


if (ENABLE_HIP OR BOBA_ENABLE_HIP)
    cmake_minimum_required(VERSION 3.21)
else()
    # Do not bump this over 3.14 due to it changing CMake policies and breaking
    # the CUDA build. We check the required version specifically for this below
    cmake_minimum_required(VERSION 3.14)
endif()


# Check for minimum CMake version required w/o changing policies like cmake_minimum_required
if(ENABLE_CUDA AND ${CMAKE_VERSION} VERSION_LESS 3.18.0)
    message(FATAL_ERROR "BOBA requires CMake version 3.18.0+ when CUDA is enabled.")
endif()

cmake_policy(SET CMP0042 NEW)  # MACOSX_RPATH related
cmake_policy(SET CMP0048 NEW)  # project() manages version variables
cmake_policy(SET CMP0076 NEW)
cmake_policy(SET CMP0104 OLD)  # NOT Initialize CMAKE_CUDA_ARCHITECTURES
if (APPLE)
 cmake_policy(SET CMP0025 NEW)  # use compiler ID AppleClang
endif()

# Set version number
set(BOBA_VERSION_MAJOR 0)
set(BOBA_VERSION_MINOR 01)

if (BOBA_LOADED AND (NOT BOBA_LOADED STREQUAL "${BOBA_VERSION_MAJOR}.${BOBA_VERSION_MINOR}"))
  message(FATAL_ERROR "You are mixing BOBA versions. Loaded is ${BOBA_LOADED}, expected ${BOBA_VERSION_MAJOR}.${BOBA_VERSION_MINOR}")
endif()

if (BOBA_LOADED)
  return() # Stop processing file, avoids nesting the whole file
endif()
set (BOBA_LOADED "${BOBA_VERSION_MAJOR}.${BOBA_VERSION_MINOR}")

# Promote BOBA_LOADED to PARENT_SCOPE if it exists, which is only if we are bringing
# in BOBA as a subproject to a larger CMake project
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  set (BOBA_LOADED ${BOBA_LOADED} PARENT_SCOPE)
endif()

mark_as_advanced(BOBA_LOADED)

# C is required for googletest to find Threads
project(BOBA LANGUAGES CXX C
  VERSION ${BOBA_LOADED})

# BLT Options need to be set in the CACHE
set(ENABLE_OPENMP On CACHE STRING "Enable OpenMP")
set(ENABLE_CUDA Off CACHE STRING "Enable CUDA")
set(ENABLE_HIP Off CACHE STRING "Enable HIP")
set(ENABLE_MPI Off CACHE STRING "Enable MPI.")
set(ENABLE_TESTS On CACHE STRING "Enable Tests.")
set(ENABLE_EXAMPLES On CACHE STRING "Enable Examples.")
set(ENABLE_TUTORIALS On CACHE STRING "Enable Tutorials.")
set(ENABLE_DOCS Off CACHE STRING "Enable Docs.")
set(CMAKE_CXX_STANDARD 20 CACHE STRING "Default CXX Standard Target Property")
set(CMAKE_CUDA_STANDARD 20 CACHE STRING "Default CUDA Standard Target Property")
set(CMAKE_CXX_EXTENSIONS Off)
set(BLT_CXX_STD "c++20" CACHE STRING "Version of C++ standard")
#set(gtest_disable_pthreads ON CACHE BOOL "")

set(CUDA_NAMEFLAG "")
set(HIP_NAMEFLAG "")
set(UMP_NAMEFLAG "")
set(PROF_NAMEFLAG "")
set(CALI_NAMEFLAG "")
set(DEBUG_NAMEFLAG "")

set(boba_sources "")
set(boba_defines "") # see SetupOptions.cmake

#------------------------------------------------------------------------------
# Include build system logic and options
#------------------------------------------------------------------------------
include(CMakeDependentOption)
include(cmake/SetupBoBaBLT.cmake)
include(cmake/CMakeBasics.cmake)

#------------------------------------------------------------------------------
# Add source directories
#------------------------------------------------------------------------------
add_subdirectory(source)

#------------------------------------------------------------------------------
# Generate header file with configuration options
#------------------------------------------------------------------------------
include(cmake/SetupBoBaConfig.cmake)

set( boba_depends )
blt_list_append( TO boba_depends ELEMENTS camp IF CAMP_FOUND )
blt_list_append( TO boba_depends ELEMENTS fmt::fmt IF FMT_FOUND )
blt_list_append( TO boba_depends ELEMENTS caliper IF CALIPER_FOUND )
blt_list_append( TO boba_depends ELEMENTS hdf5-static IF HDF5_FOUND )
blt_list_append( TO boba_depends ELEMENTS Eigen3::Eigen IF EIGEN_FOUND )
blt_list_append( TO boba_depends ELEMENTS umpire IF UMPIRE_FOUND )
blt_list_append( TO boba_depends ELEMENTS RAJA IF RAJA_FOUND )
blt_list_append( TO boba_depends ELEMENTS nvToolsExt IF BOBA_ENABLE_CUDA )
blt_list_append( TO boba_depends ELEMENTS roc::hipblas IF HIPBLAS_FOUND )
blt_list_append( TO boba_depends ELEMENTS roc::hipsolver IF HIPSOLVER_FOUND )
blt_list_append( TO boba_depends ELEMENTS roc::rocsolver IF ROCSOLVER_FOUND )
blt_list_append( TO boba_depends ELEMENTS roc::rocblas IF ROCBLAS_FOUND )

if(TARGET RAJA::roctx)
    list(APPEND boba_depends RAJA::roctx)
endif()

message(STATUS "boba_depends: ${boba_depends} ${TPL_DEPS}")

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if("${uppercase_CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  set(${DEBUG_NAMEFLAG} "_debug")
endif()

set(boba_nameflag "${CUDA_NAMEFLAG}${HIP_NAMEFLAG}${UMP_NAMEFLAG}${PROF_NAMEFLAG}${CALI_NAMEFLAG}${DEBUG_NAMEFLAG}")

message(STATUS "boba_defines: ${boba_defines}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
blt_add_library(
  NAME BOBA
  SOURCES ${boba_sources}
  DEFINES ${boba_defines}
  DEPENDS_ON ${boba_depends} ${TPL_DEPS})

if(ENABLE_CUDA)
  set_property(TARGET BOBA PROPERTY CUDA_STANDARD ${CMAKE_CUDA_STANDARD})
endif()

# Basic includes that should be inherited to all BOBA targets
target_include_directories(BOBA PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )

install(TARGETS BOBA ${boba_name}
  EXPORT BOBATargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ GROUP_WRITE
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDE_DIR}
  )

install(
  EXPORT BOBATargets
  NAMESPACE BOBA::
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ GROUP_WRITE
  DESTINATION share/boba/cmake/)

include(GNUInstallDirs)
install(
  DIRECTORY include/BOBA
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  FILES LICENSE NOTICE
  DESTINATION share/doc/BOBA)

set(boba_license_files
)

# For externally provided TPLs, BoBa does not install upstream license files.
# The external source of those TPLs is responsible for their notices.
if(NOT BOBA_EXTERNAL_BLT)
  list(APPEND boba_license_files
    tpl/blt/LICENSE
    tpl/blt/NOTICE)
endif()

if(CALIPER_FOUND AND NOT BOBA_EXTERNAL_CALIPER)
  list(APPEND boba_license_files
    tpl/Caliper/LICENSE
    tpl/Caliper/NOTICE)
endif()

if(RAJA_FOUND AND NOT BOBA_EXTERNAL_RAJA)
  list(APPEND boba_license_files
    tpl/RAJA/LICENSE
    tpl/RAJA/COPYRIGHT)
endif()

if(CAMP_FOUND AND NOT BOBA_EXTERNAL_CAMP)
  list(APPEND boba_license_files
    tpl/camp/LICENSE
    tpl/camp/NOTICE
    tpl/camp/LLVM_LICENSE)
endif()

if(EIGEN_FOUND AND NOT BOBA_EXTERNAL_EIGEN)
  list(APPEND boba_license_files
    tpl/eigen/COPYING.MPL2
    tpl/eigen/COPYING.BSD
    tpl/eigen/COPYING.APACHE
    tpl/eigen/COPYING.MINPACK
    tpl/eigen/COPYING.README
    tpl/eigen/LICENSE)
endif()

if(FMT_FOUND AND NOT BOBA_EXTERNAL_FMT)
  list(APPEND boba_license_files
    tpl/fmt/LICENSE)
endif()

if(HDF5_FOUND AND NOT BOBA_EXTERNAL_HDF5)
  list(APPEND boba_license_files
    tpl/hdf5/COPYING
    tpl/hdf5/COPYING_LBNL_HDF5)
endif()

if(UMPIRE_FOUND AND NOT BOBA_EXTERNAL_UMPIRE)
  list(APPEND boba_license_files
    tpl/umpire/LICENSE
    tpl/umpire/NOTICE
    tpl/umpire/COPYRIGHT)
endif()

if(boba_license_files)
  install(
    FILES ${boba_license_files}
    DESTINATION share/doc/BOBA/licenses)
endif()

set(boba_license_index_files
)

if(NOT BOBA_EXTERNAL_BLT)
  list(APPEND boba_license_index_files
    tpl/license/blt)
endif()

if(CALIPER_FOUND AND NOT BOBA_EXTERNAL_CALIPER)
  list(APPEND boba_license_index_files
    tpl/license/caliper)
endif()

if(RAJA_FOUND AND NOT BOBA_EXTERNAL_RAJA)
  list(APPEND boba_license_index_files
    tpl/license/raja)
endif()

if(CAMP_FOUND AND NOT BOBA_EXTERNAL_CAMP)
  list(APPEND boba_license_index_files
    tpl/license/camp)
endif()

if(EIGEN_FOUND AND NOT BOBA_EXTERNAL_EIGEN)
  list(APPEND boba_license_index_files
    tpl/license/eigen)
endif()

if(FMT_FOUND AND NOT BOBA_EXTERNAL_FMT)
  list(APPEND boba_license_index_files
    tpl/license/fmt)
endif()

if(HDF5_FOUND AND NOT BOBA_EXTERNAL_HDF5)
  list(APPEND boba_license_index_files
    tpl/license/hdf5)
endif()

if(UMPIRE_FOUND AND NOT BOBA_EXTERNAL_UMPIRE)
  list(APPEND boba_license_index_files
    tpl/license/umpire)
endif()

if(boba_license_index_files)
  list(INSERT boba_license_index_files 0 tpl/license/README.md)
  install(
    FILES ${boba_license_index_files}
    DESTINATION share/doc/BOBA/licenses)
endif()

add_subdirectory(examples)

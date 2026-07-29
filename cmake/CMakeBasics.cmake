# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other BOBA Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Setup dependent options 
#-------------------------------------------------------------------------------
include(cmake/SetupDependentOptions.cmake)

#-------------------------------------------------------------------------------
# Setup build options and their default values
#-------------------------------------------------------------------------------
include(cmake/SetupOptions.cmake)

#------------------------------------------------------------------------------
# Macros for BOBA's build system
#------------------------------------------------------------------------------
include(cmake/SetupMacros.cmake)

#-------------------------------------------------------------------------------
# Setup compiler options and their default values
#-------------------------------------------------------------------------------
include(cmake/SetupCompilers.cmake)

#------------------------------------------------------------------------------
# BOBA's Third party library setup
#------------------------------------------------------------------------------
#include(cmake/SetupPackages.cmake)
include(cmake/SetupBoBaThirdParty.cmake)

#------------------------------------------------------------------------------
# Shared vs Static Libs
#------------------------------------------------------------------------------
if(BUILD_SHARED_LIBS)
    message(STATUS "Building shared libraries (BUILD_SHARED_LIBS == ON)")
else()
    message(STATUS "Building static libraries (BUILD_SHARED_LIBS == OFF)")
endif()

#------------------------------------------------------------------------------
# Setup some additional compiler options that can be useful in various targets
# These are stored in their own variables.
# Usage: To add one of these sets of flags to some source files:
#   get_source_file_property(_origflags <src_file> COMPILE_FLAGS)
#   set_source_files_properties(<list_of_src_files>
#        PROPERTIES COMPILE_FLAGS "${_origFlags} ${<flags_variable}" )
#------------------------------------------------------------------------------

set(custom_compiler_flags_list) # Tracks custom compiler flags for logging

# Flag for disabling warnings about omp pragmas in the code
blt_append_custom_compiler_flag(FLAGS_VAR BOBA_DISABLE_OMP_PRAGMA_WARNINGS
                  DEFAULT      "-Wno-unknown-pragmas"
                  XL           " "
                  INTEL        "-diag-disable 3180"
                  MSVC         "/wd4068"
                  MSVC_INTEL   "/Qdiag-disable:3180"
                  )
list(APPEND custom_compiler_flags_list BOBA_DISABLE_OMP_PRAGMA_WARNINGS)

# Flag for disabling warnings about unused parameters.
# Useful when we include external code.
blt_append_custom_compiler_flag(FLAGS_VAR BOBA_DISABLE_UNUSED_PARAMETER_WARNINGS
                  DEFAULT     "-Wno-unused-parameter"
                  XL          " "
                  MSVC        "/wd4100"
                  MSVC_INTEL  "/Qdiag-disable:869"
                  )
list(APPEND custom_compiler_flags_list BOBA_DISABLE_UNUSED_PARAMETER_WARNINGS)

# Flag for disabling warnings about unused variables
# Useful when we include external code.
blt_append_custom_compiler_flag(FLAGS_VAR BOBA_DISABLE_UNUSED_VARIABLE_WARNINGS
                  DEFAULT     "-Wno-unused-variable"
                  XL          " "
                  MSVC        "/wd4101"
                  MSVC_INTEL  "/Qdiag-disable:177"
                  )
list(APPEND custom_compiler_flags_list BOBA_DISABLE_UNUSED_VARIABLE_WARNINGS)

# Flag for disabling warnings about variables that may be uninitialized.
# Useful when we are using compiler generated interface code (e.g. in shroud)
blt_append_custom_compiler_flag(FLAGS_VAR BOBA_DISABLE_UNINITIALIZED_WARNINGS
                  DEFAULT     "-Wno-uninitialized"
                  XL          "-qsuppress=1540-1102"
                  MSVC        "/wd4700"
                  MSVC_INTEL  "/Qdiag-disable:592"
                  )
list(APPEND custom_compiler_flags_list BOBA_DISABLE_UNINITIALIZED_WARNINGS)

# Flag for disabling warnings about strict aliasing.
# Useful when we are using compiler generated interface code (e.g. in shroud)
blt_append_custom_compiler_flag(FLAGS_VAR BOBA_DISABLE_ALIASING_WARNINGS
                  DEFAULT "-Wno-strict-aliasing"
                  XL      " "
                  MSVC    " "
                  )
list(APPEND custom_compiler_flags_list BOBA_DISABLE_ALIASING_WARNINGS)

# Flag for disabling warnings about unused local typedefs.
# Note: Clang 3.5 and below are not aware of this warning, but later versions are
if(C_COMPILER_FAMILY_IS_CLANG AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.5))
  set(clang_unused_local_typedef "-Wno-unused-local-typedefs")
endif()

blt_append_custom_compiler_flag(FLAGS_VAR BOBA_DISABLE_UNUSED_LOCAL_TYPEDEF
                  DEFAULT " "
                  CLANG   "${clang_unused_local_typedef}"
                  GNU     "-Wno-unused-local-typedefs"
                  MSVC    " "
                  XL      "-Wno-unused-local-typedefs"
                  )
list(APPEND custom_compiler_flags_list BOBA_DISABLE_UNUSED_LOCAL_TYPEDEF)

# Linker flag for allowing multiple definitions of a symbol
blt_append_custom_compiler_flag(FLAGS_VAR BOBA_ALLOW_MULTIPLE_DEFINITIONS
                  DEFAULT " "
                  CLANG   "-Wl,--allow-multiple-definition"
                  GNU     "-Wl,--allow-multiple-definition"
                  MSVC    " "
                  )
list(APPEND custom_compiler_flags_list BOBA_ALLOW_MULTIPLE_DEFINITIONS)

# Flag for allowing constant conditionals e.g. if(sizeof(T) > sizeof(int)) {...}
# There appears to be a bug in how some versions of Visual Studio treat this
blt_append_custom_compiler_flag(FLAGS_VAR BOBA_ALLOW_CONSTANT_CONDITIONALS
                  DEFAULT     " "
                  MSVC        "/wd4127"
                  MSVC_INTEL  "/Qdiag-disable:4127"
                  )
list(APPEND custom_compiler_flags_list BOBA_ALLOW_CONSTANT_CONDITIONALS)

# Flag for allowing truncation of constant values.
blt_append_custom_compiler_flag(FLAGS_VAR BOBA_ALLOW_TRUNCATING_CONSTANTS
                  DEFAULT     " "
                  MSVC        "/wd4309"
                  MSVC_INTEL  "/Qdiag-disable:4309"
                  )
list(APPEND custom_compiler_flags_list BOBA_ALLOW_TRUNCATING_CONSTANTS)

# Fix for https://github.com/LLNL/BOBA/issues/559
blt_append_custom_compiler_flag(FLAGS_VAR CMAKE_CXX_FLAGS
                  DEFAULT     " "
                  PGI         "-Wc,--pending_instantiations=900"
                  )

blt_append_custom_compiler_flag(FLAGS_VAR CMAKE_CXX_FLAGS
                  DEFAULT     " "
                  GNU         "-fmax-errors=4"
                  CLANG       "-ferror-limit=4"
                  )

blt_append_custom_compiler_flag(FLAGS_VAR CMAKE_CXX_FLAGS_DEBUG
                  DEFAULT     " "
                  CLANG       "-fstandalone-debug"
                  )

blt_append_custom_compiler_flag(FLAGS_VAR BOBA_NINJA_FLAGS
                  DEFAULT     " "
                  GNU         "-fdiagnostics-color=always"
                  CLANG       "-fcolor-diagnostics"
                  )

if(${BOBA_ENABLE_EXPORTS})
  set(CMAKE_ENABLE_EXPORTS ON)
endif()

if( ${CMAKE_MAKE_PROGRAM} STREQUAL "ninja" OR ${CMAKE_MAKE_PROGRAM} MATCHES ".*/ninja$" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BOBA_NINJA_FLAGS}")
endif()

# message(STATUS "Custom compiler flags:")
# foreach(flag ${custom_compiler_flags_list})
#    message(STATUS "\tvalue of ${flag} is '${${flag}}'")
# endforeach()

# Disable warnings about conditionals over constants
if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BOBA_ALLOW_CONSTANT_CONDITIONALS}")
endif()

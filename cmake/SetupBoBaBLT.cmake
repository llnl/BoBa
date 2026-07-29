# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#------------------------------------------------------------------------------
# Initialize BLT build system
#------------------------------------------------------------------------------
if (DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    # Use internal 'blt' submodule path if BLT_SOURCE_DIR not provided
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "Cannot locate BLT. "
            "Either run the following command in your git repository: \n"
            "    git submodule update --init --recursive\n"
            "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif()
endif()

if (“${PROJECT_SOURCE_DIR}” STREQUAL “${CMAKE_SOURCE_DIR}”)
    # Set some default BLT options before loading BLT only if not included in
    # another project
    if (NOT BLT_CXX_STD)
        set(BLT_CXX_STD "c++20" CACHE STRING "")
    endif()

    # These are not used in BOBA, turn them off
    set(_unused_blt_tools
        CLANGQUERY
        CLANGTIDY
        VALGRIND
        ASTYLE
        CMAKEFORMAT
        UNCRUSTIFY
        YAPF)
    foreach(_tool ${_unused_blt_tools})
        set(ENABLE_${_tool} OFF CACHE BOOL "")
    endforeach()

    # These are only used by BOBA developers, so turn them off
    # unless an explicit executable path is given
    set(_used_blt_tools
        CLANGFORMAT
        CPPCHECK
        DOXYGEN
        SPHINX)
    foreach(_tool ${_used_blt_tools})
        if(NOT ${_tool}_EXECUTABLE)
            set(ENABLE_${_tool} OFF CACHE BOOL "")
        else()
            set(ENABLE_${_tool} ON CACHE BOOL "")
        endif()
    endforeach()

    set(BLT_REQUIRED_CLANGFORMAT_VERSION  "10" CACHE STRING "")

    # If BOBA is the top project and BOBA_ENABLE_TESTS is off, force ENABLE_TESTS to off so
    # gtest doesn't build when it's not needed
    if(DEFINED _ENABLE_TESTS AND NOT BOBA_ENABLE_TESTS)
        set(ENABLE_TESTS OFF CACHE BOOL "")
    endif()

    # Mark BLT's built-in third-party targets as EXPORTABLE so they can be added
    # to the BOBA-targets export set
    set(BLT_EXPORT_THIRDPARTY ON CACHE BOOL "")
endif()

if("${BLT_CXX_STD}" STREQUAL "c++98" OR "${BLT_CXX_STD}" STREQUAL "c++11" OR "${BLT_CXX_STD}" STREQUAL "c++14" OR "${BLT_CXX_STD}" STREQUAL "c++17")
    message(FATAL_ERROR "BOBA requires BLT_CXX_STD to be 'c++20' or above.")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

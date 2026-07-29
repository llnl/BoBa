# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


macro(boba_add_executable)
  set(options )
  set(singleValueArgs NAME TUTORIAL TEST EXERCISE)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON BOBA)
  list (APPEND arg_DEPENDS_ON ${boba_depends})
  list (APPEND arg_DEPENDS_ON ${TPL_DEPS})

  if (${arg_TEST})
    set (_output_dir ${CMAKE_BINARY_DIR}/test)
  elseif (${arg_TUTORIAL})
    set (_output_dir ${CMAKE_BINARY_DIR}/tutorial)
  elseif (${arg_EXERCISE})
    set (_output_dir ${CMAKE_BINARY_DIR}/exercise)
  else ()
    set (_output_dir ${CMAKE_BINARY_DIR}/bin)
  endif()

  blt_add_executable(
    NAME ${arg_NAME}
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    OUTPUT_DIR ${_output_dir}
    )

  if (ENABLE_CUDA)
    set_property(TARGET ${arg_NAME} PROPERTY CUDA_STANDARD "20")
  endif()
endmacro(boba_add_executable)

# Allows strings embedded in test files to used to process ctest results.
# Only works for new testing framework/structure (no effect on old tests).
# Borrowed heavily from CAMP.
function(boba_set_failtest TESTNAME)
  set(test_name ${TESTNAME})

  # Chopping off backend from test name
  string(REGEX REPLACE "\-Sequential|\-OpenMP|\-OpenMPTarget|\-TBB|\-CUDA|\-HIP" "" test_nobackend ${test_name})

  # Finding test source code
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/${test_nobackend}.hpp")
    list(APPEND TEST_LIST "${CMAKE_CURRENT_SOURCE_DIR}/tests/${test_nobackend}.hpp")
    list(REMOVE_DUPLICATES TEST_LIST)
  endif()

  list(GET TEST_LIST 0 source_file)

  if(EXISTS ${source_file})
    set(test_regex  ".*(WILL_FAIL|PASS_REGEX|FAIL_REGEX):?[ ]*(.*)[ ]*")

    file(STRINGS ${source_file} test_lines REGEX "${test_regex}")

    # Search test source code for fail string
    foreach(line ${test_lines})
      if(NOT line MATCHES "${test_regex}")
        continue()
      endif()

      if(CMAKE_MATCH_1 STREQUAL "WILL_FAIL")
        set_property( TARGET ${test_name}.exe   # TARGET more conformant to BLT
                      APPEND PROPERTY WILL_FAIL )
      elseif(CMAKE_MATCH_1 STREQUAL "PASS_REGEX")
        set_property( TARGET ${test_name}.exe
                      APPEND PROPERTY PASS_REGULAR_EXPRESSION "${CMAKE_MATCH_2}")
      elseif(CMAKE_MATCH_1 STREQUAL "FAIL_REGEX")
        set_property( TARGET ${test_name}.exe
                      APPEND PROPERTY FAIL_REGULAR_EXPRESSION "${CMAKE_MATCH_2}")
      endif()
    endforeach()
  endif()
endfunction()

macro(boba_add_test)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON gtest ${CMAKE_THREAD_LIBS_INIT})

  set(original_test_name ${arg_NAME})

  boba_add_executable(
    NAME ${arg_NAME}.exe
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    TEST On)

  blt_add_test(
    NAME ${arg_NAME}
    #COMMAND ${TEST_DRIVER} $<TARGET_FILE:${arg_NAME}>)
    COMMAND ${TEST_DRIVER} ${arg_NAME})

  boba_set_failtest(${original_test_name})
endmacro(boba_add_test)

macro(boba_add_reproducer)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  boba_add_executable(
    NAME ${arg_NAME}.exe
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    REPRODUCER On)
endmacro(boba_add_reproducer)

macro(boba_add_benchmark)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON gbenchmark)

  boba_add_executable(
    NAME ${arg_NAME}.exe
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    BENCHMARK On)

  blt_add_benchmark(
    NAME ${arg_NAME}
    COMMAND ${TEST_DRIVER} ${arg_NAME})
endmacro(boba_add_benchmark)

##------------------------------------------------------------------------------
## boba_configure_file
##
## This macro is a thin wrapper over the builtin configure_file command.
## It has the same arguments/options as configure_file but introduces an
## intermediate file that is only copied to the target file if the target differs
## from the intermediate.
##------------------------------------------------------------------------------
macro(boba_configure_file _source _target)
    set(_tmp_target ${_target}.tmp)
    configure_file(${_source} ${_tmp_target} ${ARGN})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_tmp_target} ${_target})
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${_tmp_target})
endmacro(boba_configure_file)


macro(remove_list_item LIST OLD_VALUE)
    list(FIND ${LIST} ${OLD_VALUE} OLD_VALUE_INDEX)
    if(OLD_VALUE_INDEX GREATER_EQUAL 0)
        list(REMOVE_AT ${LIST} ${OLD_VALUE_INDEX})
    endif()
endmacro()

# Filter values through regex
#   filter_regex({INCLUDE | EXCLUDE} <regex> <listname> [items...])
#   Element will included into result list if
#     INCLUDE is specified and it matches with regex or
#     EXCLUDE is specified and it doesn't match with regex.
# Example:
#   filter_regex(INCLUDE "(a|c)" LISTOUT a b c d) => a c
#   filter_regex(EXCLUDE "(a|c)" LISTOUT a b c d) => b d
function(filter_regex _action _regex _listname)
    # check an action
    if("${_action}" STREQUAL "INCLUDE")
        set(has_include TRUE)
    elseif("${_action}" STREQUAL "EXCLUDE")
        set(has_include FALSE)
    else()
        message(FATAL_ERROR "Incorrect value for ACTION: ${_action}")
    endif()

    set(${_listname})
    foreach(element ${ARGN})
        string(REGEX MATCH ${_regex} result ${element})
        if(result)
            if(has_include)
                list(APPEND ${_listname} ${element})
            endif()
        else()
            if(NOT has_include)
                list(APPEND ${_listname} ${element})
            endif()
        endif()
    endforeach()

    # put result in parent scope variable
    set(${_listname} ${${_listname}} PARENT_SCOPE)
endfunction()

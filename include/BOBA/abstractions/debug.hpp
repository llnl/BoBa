// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace boba
{

// -----------------------------------------------------
// Printing
// -----------------------------------------------------
#define boba_print(A) boba::boba_print_(A, #A)

/**
 * @brief Prints a value and returns it unchanged.
 * @tparam T Printed value type.
 * @param in Value to print.
 * @param in_name Label to print alongside `in`.
 * @return `in`.
 */
template <typename T>
inline T boba_print_(T in, const char* in_name)
{
#ifndef BOBA_DEVICE_CODE
  std::cout << in_name << " = " << in << std::endl;
#else
  ::boba::detail::ignore(in);
  ::boba::detail::ignore(in_name);
#endif
  return in;
}

/**
 * @brief Prints a `double` and returns it unchanged.
 * @param in Value to print.
 * @param in_name Label to print alongside `in`.
 * @return `in`.
 */
__boba_host_device__ inline double boba_print_(double in, const char* in_name)
{
  printf("%s = %.7e\n", in_name, in);
  return in;
}

/**
 * @brief Prints an `int` and returns it unchanged.
 * @param in Value to print.
 * @param in_name Label to print alongside `in`.
 * @return `in`.
 */
__boba_host_device__ inline int boba_print_(int in, const char* in_name)
{
  printf("%s = %d\n", in_name, in);
  return in;
}

/**
 * @brief Prints a `size_t` and returns it unchanged.
 * @param in Value to print.
 * @param in_name Label to print alongside `in`.
 * @return `in`.
 */
__boba_host_device__ inline size_t boba_print_(size_t in, const char* in_name)
{
  printf("%s = %lu\n", in_name, in);
  return in;
}

/**
 * @brief Prints a C string and returns it unchanged.
 * @param in String to print.
 * @param in_name Ignored label.
 * @return `in`.
 */
__boba_host_device__ inline const char* boba_print_(const char* in, const char* in_name)
{
  ::boba::detail::ignore(in_name);
  printf("%s\n", in);
  return in;
}

/**
 * @brief Prints a vector and returns it unchanged.
 * @tparam T Vector element type.
 * @param in Vector to print.
 * @param in_name Label to print alongside `in`.
 * @return `in`.
 */
template <typename T>
inline std::vector<T> boba_print_(std::vector<T> in, const char* in_name)
{
  std::cout << in_name << " =";
  bool first = true;
  for (auto elem : in)
  {
    std::cout << (first ? ("= ") : (", ")) << elem;
    first = false;
  }
  std::cout << std::endl;
  return in;
}

/**
 * @brief Prints an unordered map and returns it unchanged.
 * @tparam T Map key type.
 * @tparam S Map value type.
 * @param in Map to print.
 * @param in_name Label to print alongside `in`.
 * @return `in`.
 */
template <typename T, typename S>
inline std::unordered_map<T, S> boba_print_(std::unordered_map<T, S> in, const char* in_name)
{
  std::cout << in_name << " ";
  bool first = true;
  for (auto map : in)
  {
    std::cout << (first ? ("= ") : (", ")) << std::get<0>(map) << " -> " << std::get<1>(map);
    first = false;
  }
  std::cout << std::endl;
  return in;
}

// -----------------------------------------------------
// Rename
// -----------------------------------------------------
#define boba_rename(A) \
  {                    \
    A.rename(#A);      \
  }
#define boba_rename_print(A) \
  {                          \
    auto A_print = A;        \
    A_print.rename(#A);      \
    A_print.print();         \
  }

// -----------------------------------------------------
// Checkpoints
// -----------------------------------------------------
#if !(defined(BOBA_DEVICE_CODE))
#define always_checkpoint() boba::checkpoint_(__LINE__, __FUNCTION__, __FILE__);
#else
#define always_checkpoint() /* do nothing */
#endif

#if defined(BOBA_CHECKPOINTS) && !(defined(BOBA_DEVICE_CODE))
#define checkpoint() boba::checkpoint_(__LINE__, __FUNCTION__, __FILE__);
#else
#define checkpoint() /* do nothing */
#endif

#if defined(BOBA_CHECKPOINTS_OBJECTS) && !(defined(BOBA_DEVICE_CODE))
#define checkpoint_objects() boba::checkpoint_(__LINE__, __FUNCTION__, __FILE__);
#else
#define checkpoint_objects() /* do nothing */
#endif

/**
 * @brief Prints a source-location checkpoint after synchronizing device work.
 * @param line Source line number.
 * @param function Function name.
 * @param file Source file path.
 */
inline void checkpoint_(size_t line, const char* function, const char* file)
{
  ::boba::detail::device_sync();
  std::cout << "In " << function << " in " << file << ":" << line
            << std::endl;
}

} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <iostream>

/**
 * \brief
 * Useful for managing size_t literals before C++23's 'z'
 * See include/BOBA/abstractions/types.hpp
 */

using boba::operator""_z;

/*
  Some useful common functions
*/

#define pass_or_fail(check, error, tolerance)                                                                      \
  {                                                                                                                \
    /* take the declared type of error with decyltype, remove any &, and ensure tolerance is the same type */      \
    auto typed_error = static_cast<typename std::remove_reference<decltype(tolerance)>::type>(error);              \
    bool this_check = boba::abs(typed_error) < tolerance;                                                          \
    std::cout << " checking " << #error << " = " << std::scientific << typed_error << " < " << tolerance << " ? "; \
    std::cout << (this_check ? ("pass") : ("fail")) << std::endl;                                                  \
    check = check && this_check;                                                                                   \
    std::cout << " cumulative check = " << (check ? ("pass") : ("fail")) << std::endl;                             \
    bool fail_immediately = boba::is_env_nonempty("FAIL_IMMEDIATELY");                                             \
    bool fail_never = boba::is_env_nonempty("FAIL_NEVER");                                                         \
    if (not(check) and fail_immediately and not(fail_never))                                                       \
    {                                                                                                              \
      boba_error("Check failed!");                                                                                 \
    }                                                                                                              \
  }

#define pass_or_fail_bool(check, condition)                                            \
  {                                                                                    \
    bool this_check = static_cast<bool>(condition);                                    \
    std::cout << " checking " << #condition << " ? "                                   \
              << (this_check ? "pass" : "fail") << std::endl;                          \
    check = check && this_check;                                                       \
    std::cout << " cumulative check = " << (check ? ("pass") : ("fail")) << std::endl; \
    bool fail_immediately = boba::is_env_nonempty("FAIL_IMMEDIATELY");                 \
    bool fail_never = boba::is_env_nonempty("FAIL_NEVER");                             \
    if (not(check) and fail_immediately and not(fail_never))                           \
    {                                                                                  \
      boba_error("Check failed!");                                                     \
    }                                                                                  \
  }

#define final_check(check) boba::is_env_nonempty("FAIL_NEVER") ? 0 : not(check);

/**
 * \brief
 * Useful for speedup calculations
 */

template <typename output_type = size_t>
inline std::string divide_string(double a, double b)
{
  if (b > 0)
  {
    return std::to_string(output_type(a / b));
  }
  else if (a > 0)
  {
    return std::string("infinity");
  }
  return std::string("indeterminant");
}

inline std::string fill_up_string_end(std::string in, size_t target_length)
{
  const size_t current_length = in.length();
  const size_t repeat = current_length < target_length ? target_length - current_length : 0;
  std::string out = std::string(in) + std::string(repeat, ' ');
  return out;
}

namespace boba
{
template <typename T>
std::string to_string(const T in, const size_t precision = 16)
{
  // From https://stackoverflow.com/questions/16605967/set-precision-of-stdto-string-when-converting-floating-point-values#:~:text=In%20C%2B%2B11%2C%20std,of%20type%20float%20or%20double%20.
  std::ostringstream out;
  out.precision(static_cast<std::streamsize>(precision));
  out << std::fixed << in;
  return std::move(out).str();
}
} // namespace boba

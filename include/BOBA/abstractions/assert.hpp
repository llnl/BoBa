// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <string>

namespace boba
{

// ---------------------
// Asserts
// ---------------------

#if !(defined(BOBA_DEVICE_CODE))

#define boba_always_assert(a, err) ::boba::assert_(a, false, "false", (#a), err, __LINE__, __FUNCTION__, __FILE__);
#define boba_always_assert_not(a, err) ::boba::assert_(a, true, "true", (#a), err, __LINE__, __FUNCTION__, __FILE__);
#define boba_always_assert_equal(a, b, err) ::boba::assert_equal_(a, b, true, (#a), (#b), err, __LINE__, __FUNCTION__, __FILE__);
#define boba_always_assert_ne(a, b, err) ::boba::assert_equal_(a, b, false, (#a), (#b), err, __LINE__, __FUNCTION__, __FILE__);
#define boba_always_assert_modulo(a, b, err) ::boba::assert_modulo_(a, b, (#a), (#b), err, __LINE__, __FUNCTION__, __FILE__);
#define boba_always_assert_lt(a, b, err) ::boba::assert_compare_(a, b, a < b, " < ", (#a), (#b), err, __LINE__, __FUNCTION__, __FILE__);
#define boba_always_assert_le(a, b, err) ::boba::assert_compare_(a, b, a <= b, " <= ", (#a), (#b), err, __LINE__, __FUNCTION__, __FILE__);
#define boba_always_assert_ge(a, b, err) ::boba::assert_compare_(a, b, a >= b, " >= ", (#a), (#b), err, __LINE__, __FUNCTION__, __FILE__);
#define boba_always_assert_gt(a, b, err) ::boba::assert_compare_(a, b, a > b, " > ", (#a), (#b), err, __LINE__, __FUNCTION__, __FILE__);
#define boba_always_assert_positive(a, err) ::boba::assert_compare_(a, 0, a > 0, " > ", (#a), "0", err, __LINE__, __FUNCTION__, __FILE__);
#define boba_always_assert_nonnegative(a, err) ::boba::assert_compare_(a, 0, ::boba::check_nonnegative_(a), " >= ", (#a), "0", err, __LINE__, __FUNCTION__, __FILE__);
#define boba_warn(err) ::boba::error_(err, false, __LINE__, __FUNCTION__, __FILE__);
#define boba_error(err) ::boba::error_(err, true, __LINE__, __FUNCTION__, __FILE__);

#else

#define boba_always_assert(a, err) \
  assert(a);                       \
  ::boba::detail::ignore(a, err);
#define boba_always_assert_not(a, err) \
  assert(!a);                          \
  ::boba::detail::ignore(a, err);
#define boba_always_assert_equal(a, b, err) \
  assert(not((a < b) or (a > b)));          \
  ::boba::detail::ignore(a, b, err);
#define boba_always_assert_ne(a, b, err) \
  assert(a != b);                        \
  ::boba::detail::ignore(a, b, err);
#define boba_always_assert_modulo(a, b, err) \
  assert(a % b == 0);                        \
  ::boba::detail::ignore(a, b, err);
#define boba_always_assert_lt(a, b, err) \
  assert(a < b);                         \
  ::boba::detail::ignore(a, b, err);
#define boba_always_assert_le(a, b, err) \
  assert(a <= b);                        \
  ::boba::detail::ignore(a, b, err);
#define boba_always_assert_ge(a, b, err) \
  assert(a >= b);                        \
  ::boba::detail::ignore(a, b, err);
#define boba_always_assert_gt(a, b, err) \
  assert(a > b);                         \
  ::boba::detail::ignore(a, b, err);
#define boba_always_assert_positive(a, err) \
  assert(a > 0);                            \
  ::boba::detail::ignore(a, a, err);
#define boba_always_assert_nonnegative(a, err) \
  assert(::boba::check_nonnegative_(a));       \
  ::boba::detail::ignore(a, a, err);
#define boba_warn(err) ::boba::detail::ignore(err);
#define boba_error(err) \
  assert(false);        \
  ::boba::detail::ignore(err);

#endif

#define boba_never_assert(a, err) ::boba::detail::ignore(a, err);
#define boba_never_assert_not(a, err) ::boba::detail::ignore(a, err);
#define boba_never_assert_equal(a, b, err) ::boba::detail::ignore(a, b, err);
#define boba_never_assert_ne(a, b, err) ::boba::detail::ignore(a, b, err);
#define boba_never_assert_modulo(a, b, err) ::boba::detail::ignore(a, b, err);
#define boba_never_assert_lt(a, b, err) ::boba::detail::ignore(a, b, err);
#define boba_never_assert_le(a, b, err) ::boba::detail::ignore(a, b, err);
#define boba_never_assert_ge(a, b, err) ::boba::detail::ignore(a, b, err);
#define boba_never_assert_gt(a, b, err) ::boba::detail::ignore(a, b, err);
#define boba_never_assert_positive(a, err) ::boba::detail::ignore(a, 0, err);
#define boba_never_assert_nonnegative(a, err) ::boba::detail::ignore(a, 0, err);

// Debug asserts
#if defined(BOBA_DEBUG)
#define boba_assert(a, err) boba_always_assert(a, err)
#define boba_assert_not(a, err) boba_always_assert_not(a, err)
#define boba_assert_equal(a, b, err) boba_always_assert_equal(a, b, err)
#define boba_assert_ne(a, b, err) boba_always_assert_ne(a, b, err)
#define boba_assert_modulo(a, b, err) boba_always_assert_modulo(a, b, err)
#define boba_assert_lt(a, b, err) boba_always_assert_lt(a, b, err)
#define boba_assert_le(a, b, err) boba_always_assert_le(a, b, err)
#define boba_assert_ge(a, b, err) boba_always_assert_ge(a, b, err)
#define boba_assert_gt(a, b, err) boba_always_assert_gt(a, b, err)
#define boba_assert_positive(a, err) boba_always_assert_positive(a, err)
#define boba_assert_nonnegative(a, err) boba_always_assert_nonnegative(a, err)
#else
#define boba_assert(a, err) boba_never_assert(a, err)
#define boba_assert_not(a, err) boba_never_assert_not(a, err)
#define boba_assert_equal(a, b, err) boba_never_assert_equal(a, b, err)
#define boba_assert_ne(a, b, err) boba_never_assert_ne(a, b, err)
#define boba_assert_modulo(a, b, err) boba_never_assert_modulo(a, b, err)
#define boba_assert_lt(a, b, err) boba_never_assert_lt(a, b, err)
#define boba_assert_le(a, b, err) boba_never_assert_le(a, b, err)
#define boba_assert_ge(a, b, err) boba_never_assert_ge(a, b, err)
#define boba_assert_gt(a, b, err) boba_never_assert_gt(a, b, err)
#define boba_assert_positive(a, err) boba_never_assert_positive(a, err)
#define boba_assert_nonnegative(a, err) boba_never_assert_nonnegative(a, err)
#endif
// end debug asserts

/**
 * \brief
 * Warn or error
 * @param error Message to report.
 * @param terminate Whether to throw after reporting the message.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file path.
 */

inline void error_(
  std::string_view error,
  bool terminate,
  size_t line,
  std::string_view function,
  std::string_view file)
{

  std::cout << (terminate ? "Error in " : "Warning in ");
  std::cout << function << ", " << file << ":" << line << std::endl;
  std::cout << error << std::endl;

  if (terminate)
  {
    throw std::runtime_error("BoBa raised an error");
  }
}

/**
 * \brief
 * Assert check is true
 * @param check Evaluated condition.
 * @param trigger_if Condition value that triggers failure.
 * @param problem Text describing the failing truth value.
 * @param expression String form of the checked expression.
 * @param error Additional error message.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file path.
 */

inline void assert_(
  bool check,
  bool trigger_if,
  std::string_view problem,
  std::string_view expression,
  std::string_view error,
  size_t line,
  std::string_view function,
  std::string_view file)
{

  if (check == trigger_if)
  {
    std::cout << "Error in " << function << ", " << file << ":" << line
              << std::endl;
    std::cout << "Assert evaluates to " << problem << ": " << expression << std::endl;
    std::cout << error << std::endl;
    throw std::runtime_error("assertion failed");
  }
}

/**
 * \brief
 * Assert a == b or a != b
 * @tparam type Compared value type.
 * @param expression_a Left-hand value.
 * @param expression_b Right-hand value.
 * @param target `true` to require equality, `false` to require inequality.
 * @param string_a String form of `expression_a`.
 * @param string_b String form of `expression_b`.
 * @param error Additional error message.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file path.
 */

template <typename type>
inline void assert_equal_(
  type expression_a,
  type expression_b,
  bool target, // if true we seek equality, if false we seek inequality
  std::string_view string_a,
  std::string_view string_b,
  std::string_view error,
  size_t line,
  std::string_view function,
  std::string_view file)
{

  bool check = target ? (expression_a == expression_b) : (expression_a != expression_b);

  std::string comparator = target ? " == " : " != ";

  if (!check)
  {
    std::cout << "Error in " << function << ", " << file << ":" << line
              << std::endl;
    std::cout << "Failed: " << string_a << comparator << string_b << std::endl;
    std::cout << error << ": " << expression_a << comparator << expression_b << std::endl;
    throw std::runtime_error("assertion failed");
  }
}

/**
 * @brief Compares floating-point values with epsilon tolerance.
 * @tparam float_t Floating-point type.
 * @param expression_a Left-hand value.
 * @param expression_b Right-hand value.
 * @param target `true` to require near-equality, `false` to require separation.
 * @param string_a String form of `expression_a`.
 * @param string_b String form of `expression_b`.
 * @param error Additional error message.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file path.
 */
template <typename float_t>
inline void assert_equal_float_(
  float_t expression_a,
  float_t expression_b,
  bool target, // if true we seek equality, if false we seek inequality
  std::string_view string_a,
  std::string_view string_b,
  std::string_view error,
  size_t line,
  std::string_view function,
  std::string_view file)
{

  auto diff = ::fabs(expression_a - expression_b);
  bool is_small = diff <= static_cast<float_t>(1.1) * std::numeric_limits<float_t>::epsilon();

  bool check = target ? is_small : not(is_small);

  std::string comparator = target ? " == " : " != ";

  if (!check)
  {
    std::cout << "Error in " << function << ", " << file << ":" << line
              << std::endl;
    std::cout << "Failed: " << string_a << comparator << string_b << std::endl;
    std::cout << error << ": " << expression_a << comparator << expression_b << std::endl;
    throw std::runtime_error("assertion failed");
  }
}

/**
 * @brief Specializes `assert_equal_` for `double`.
 * @param expression_a Left-hand value.
 * @param expression_b Right-hand value.
 * @param target `true` to require near-equality, `false` to require separation.
 * @param string_a String form of `expression_a`.
 * @param string_b String form of `expression_b`.
 * @param error Additional error message.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file path.
 */
template <>
inline void assert_equal_<double>(
  double expression_a,
  double expression_b,
  bool target, // if true we seek equality, if false we seek inequality
  std::string_view string_a,
  std::string_view string_b,
  std::string_view error,
  size_t line,
  std::string_view function,
  std::string_view file)
{
  assert_equal_float_(expression_a, expression_b, target, string_a, string_b, error, line, function, file);
}

/**
 * @brief Specializes `assert_equal_` for `float`.
 * @param expression_a Left-hand value.
 * @param expression_b Right-hand value.
 * @param target `true` to require near-equality, `false` to require separation.
 * @param string_a String form of `expression_a`.
 * @param string_b String form of `expression_b`.
 * @param error Additional error message.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file path.
 */
template <>
inline void assert_equal_<float>(
  float expression_a,
  float expression_b,
  bool target, // if true we seek equality, if false we seek inequality
  std::string_view string_a,
  std::string_view string_b,
  std::string_view error,
  size_t line,
  std::string_view function,
  std::string_view file)
{
  assert_equal_float_(expression_a, expression_b, target, string_a, string_b, error, line, function, file);
}

/**
 * \brief
 * Assert a % b == 0
 * @tparam type_1 Left-hand value type.
 * @tparam comparable_type Right-hand value type.
 * @param expression_a Dividend.
 * @param expression_b Divisor.
 * @param string_a String form of `expression_a`.
 * @param string_b String form of `expression_b`.
 * @param error Additional error message.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file path.
 */

template <typename type_1, typename comparable_type>
inline void assert_modulo_(
  type_1 expression_a,
  comparable_type expression_b,
  std::string_view string_a,
  std::string_view string_b,
  std::string_view error,
  int line,
  std::string_view function,
  std::string_view file)
{

  auto result = expression_a % expression_b;
  bool check = (result == type_1(0));
  if (!check)
  {
    std::cout << "Error in " << function << ", " << file << ":" << line
              << std::endl;
    std::cout << "Failed: " << string_a << " modulo " << string_b << " == 0 " << std::endl;
    std::cout << error << ": " << expression_a << " % " << expression_b << " = " << result << std::endl;
    throw std::runtime_error("assertion failed");
  }
}

/**
 * @brief Checks an already-evaluated comparison and reports failures.
 * @tparam type_1 Left-hand value type.
 * @tparam comparable_type Right-hand value type.
 * @param expression_a Left-hand value.
 * @param expression_b Right-hand value.
 * @param check Comparison result.
 * @param comparator Text form of the comparison operator.
 * @param string_a String form of `expression_a`.
 * @param string_b String form of `expression_b`.
 * @param error Additional error message.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file path.
 */
template <typename type_1, typename comparable_type>
inline void assert_compare_(
  type_1 expression_a,
  comparable_type expression_b,
  bool check,
  std::string_view comparator,
  std::string_view string_a,
  std::string_view string_b,
  std::string_view error,
  size_t line,
  std::string_view function,
  std::string_view file)
{

  if (!check)
  {
    std::cout << "Error in " << function << ", " << file << ":" << line
              << std::endl;
    std::cout << "Failed: " << string_a << comparator << string_b << std::endl;
    std::cout << error << ": " << expression_a << comparator << expression_b << std::endl;
    throw std::runtime_error("assertion failed");
  }
}

/**
 * @brief Returns `true` for unsigned values.
 * @param expression_a Value to check.
 * @return Always `true`.
 */
__boba_host_device__ inline bool check_nonnegative_(
  size_t expression_a)
{
  ::boba::detail::ignore(expression_a);
  return true;
}

/**
 * @brief Checks whether a value is nonnegative.
 * @tparam type Value type.
 * @param expression_a Value to check.
 * @return `true` when `expression_a >= 0`.
 */
template <typename type>
__boba_host_device__ inline bool check_nonnegative_(
  type expression_a)
{
  return (expression_a >= static_cast<std::remove_reference_t<decltype(expression_a)>>(0));
}

} // namespace boba

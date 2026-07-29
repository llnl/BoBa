// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/abstractions/common.hpp"

#include <limits>
#include <math.h>
#include <type_traits>

namespace boba
{

#ifdef __INTEL_COMPILE
// intel compiler warns "external function definition with no prior declaration"
// disabling this warning here
#pragma warning(push)
#pragma warning(disable : 1418)
#endif

/**
 * \brief
 * Archimedes' constant.
 */

constexpr double pi = 3.141592653589793238462643383279;

/**
 * \brief
 * exp(1)
 */

constexpr double e = 2.718281828459045235360287471352;

/**
 * \brief
 * Absolute value.
 * @param x Input value.
 * @return `|x|`.
 */

__boba_host_device__ constexpr float abs(float x)
{
  return ::fabs(x);
}

/**
 * \brief
 * Absolute value.
 * @param x Input value.
 * @return `|x|`.
 */

__boba_host_device__ constexpr double abs(double x)
{
  return ::fabs(x);
}

/**
 * \brief
 * Absolute value.
 * @param x Input value.
 * @return `|x|`.
 */

__boba_host_device__ constexpr int abs(int x)
{
  return ::abs(x);
}

/**
 * \brief
 * Absolute value.
 * @param x Input value.
 * @return `x`.
 */

__boba_host_device__ constexpr size_t abs(size_t x)
{
  return x;
}

/**
 * \brief
 * Use python-consistent modulo
 * @param a Dividend.
 * @param b Divisor.
 * @return `a mod b` mapped into `[0, b)`.
 */

__boba_host_device__ constexpr size_t mod(const size_t a, const size_t b)
{
  return (b + (a % b)) % b;
}

/**
 * \brief
 * Sine.
 * @param x Input angle in radians.
 * @return `sin(x)`.
 */

__boba_host_device__ inline double sin(double x)
{
  return ::sin(x);
}

/**
 * \brief
 * Sine.
 * @param x Input angle in radians.
 * @return `sin(x)`.
 */

__boba_host_device__ inline float sin(float x)
{
  return ::sin(x);
}

/**
 * \brief
 * Cosine.
 * @param x Input angle in radians.
 * @return `cos(x)`.
 */

__boba_host_device__ inline double cos(double x)
{
  return ::cos(x);
}

/**
 * \brief
 * Cosine.
 * @param x Input angle in radians.
 * @return `cos(x)`.
 */

__boba_host_device__ inline float cos(float x)
{
  return ::cos(x);
}

/**
 * \brief
 * Arc cosine.
 * @param x Input value.
 * @return `acos(x)`.
 */

__boba_host_device__ inline double acos(double x)
{
  return ::acos(x);
}

/**
 * \brief
 * Two-argument arctangent.
 * @param numerator Y coordinate.
 * @param denominator X coordinate.
 * @return `atan2(numerator, denominator)`.
 */

__boba_host_device__ inline double atan(double numerator, double denominator)
{
  return ::atan2(numerator, denominator);
}

/**
 * \brief
 * Two-argument arctangent.
 * @param numerator Y coordinate.
 * @param denominator X coordinate.
 * @return `atan2(numerator, denominator)`.
 */

__boba_host_device__ inline float atan(float numerator, float denominator)
{
  return ::atan2(numerator, denominator);
}

/**
 * \brief
 * Raises x to the power p.
 * @param x Base value.
 * @param p Exponent.
 * @return `x` raised to `p`.
 */

__boba_host_device__ constexpr double pow(double x, double p)
{
  return ::pow(x, p);
}

/**
 * \brief
 * Raises x to the power p.
 * @param x Base value.
 * @param p Exponent.
 * @return `x` raised to `p`.
 */

__boba_host_device__ constexpr float pow(float x, float p)
{
  return ::pow(x, p);
}

/**
 * \brief
 * Raises x to the power p.
 * @param x Base value.
 * @param p Exponent.
 * @return `x` raised to `p`.
 */

__boba_host_device__ constexpr size_t pow(size_t x, size_t p)
{
  return ::pow(x, p);
}

/**
 * \brief
 * Square root.
 * @param x Input value.
 * @return `sqrt(x)`.
 */

__boba_host_device__ constexpr double sqrt(double x)
{
  return ::sqrt(x);
}

/**
 * \brief
 * Square root.
 * @param x Input value.
 * @return `sqrt(x)`.
 */

__boba_host_device__ constexpr float sqrt(float x)
{
  return ::sqrt(x);
}

/**
 * \brief
 * Square root.
 * @param x Input value.
 * @return `sqrt(x)`.
 */

__boba_host_device__ constexpr size_t sqrt(size_t x)
{
  return ::sqrt(x);
}

/**
 * \brief
 * Square root.
 * @param x Input value.
 * @return `sqrt(x)`.
 */

__boba_host_device__ constexpr int sqrt(int x)
{
  return ::sqrt(x);
}

/**
 * \brief
 * Cube root.
 * @param x Input value.
 * @return `cbrt(x)`.
 */

__boba_host_device__ constexpr double cbrt(double x)
{
  return ::cbrt(x);
}

/**
 * \brief
 * Cube root.
 * @param x Input value.
 * @return `cbrt(x)`.
 */

__boba_host_device__ constexpr size_t cbrt(size_t x)
{
  return ::cbrt(x);
}

/**
 * \brief
 * Cube root.
 * @param x Input value.
 * @return `cbrt(x)`.
 */

__boba_host_device__ constexpr int cbrt(int x)
{
  return ::cbrt(x);
}

/**
 * \brief
 * Exponential.
 * @param x Input value.
 * @return `exp(x)`.
 */

__boba_host_device__ constexpr double exp(double x)
{
  return ::exp(x);
}

/**
 * \brief
 * Exponential.
 * @param x Input value.
 * @return `exp(x)`.
 */

__boba_host_device__ constexpr float exp(float x)
{
  return ::exp(x);
}

/**
 * \brief
 * Floor.
 * @param x Input value.
 * @return Largest integer not greater than `x`.
 */

__boba_host_device__ constexpr double floor(double x)
{
  return ::floor(x);
}

/**
 * \brief
 * Ceiling.
 * @param x Input value.
 * @return Smallest integer not less than `x`.
 */

__boba_host_device__ constexpr double ceil(double x)
{
  return ::ceil(x);
}

/**
 * \brief
 * Integer logarithm, base 2.
 * @param x Positive input value.
 * @return Integer base-2 logarithm of `x`.
 */

__boba_host_device__ constexpr size_t log2(size_t x)
{
#ifndef BOBA_DEVICE_CODE
  if (x <= 0)
  {
    throw std::domain_error("log(x) undefined for x <= 0.");
  }
#endif
  return (x == 1) ? 0 : 1 + log2(x >> 1);
}

/**
 * \brief
 * Integer logarithm, base b.
 * @param x Positive input value.
 * @param b Logarithm base.
 * @return Integer base-`b` logarithm of `x`.
 */

__boba_host_device__ constexpr size_t logb(size_t x, size_t b)
{
#ifndef BOBA_DEVICE_CODE
  if (x <= 0)
  {
    throw std::domain_error("log_b(x) undefined for x <= 0.");
  }
  if (b <= 0)
  {
    throw std::domain_error("log_b(x) undefined for b <= 0.");
  }
  if (b <= 1)
  {
    throw std::domain_error("log_b(x) undefined for b <= 1.");
  }
#endif
  return (x == 1) ? 0 : 1 + logb(x / b, b);
}

/**
 * \brief
 * double logarithm, base e.
 * @param x Positive input value.
 * @return Natural logarithm of `x`.
 */

__boba_host_device__ constexpr double log(double x)
{
#ifndef BOBA_DEVICE_CODE
  if (x <= 0.)
  {
    throw std::domain_error("log_b(x) undefined for x <= 0.");
  }
#endif
  return ::log(x);
}

/**
 * \brief
 * Natural logarithm.
 * @param x Positive input value.
 * @return Natural logarithm of `x`.
 */

__boba_host_device__ constexpr float log(float x)
{
#ifndef BOBA_DEVICE_CODE
  if (x <= 0.0f)
  {
    throw std::domain_error("log_b(x) undefined for x <= 0.");
  }
#endif
  return ::log(x);
}

/**
 * \brief
 * double logarithm, base 10.
 * @param x Positive input value.
 * @return Base-10 logarithm of `x`.
 */

__boba_host_device__ constexpr double log10(double x)
{
#ifndef BOBA_DEVICE_CODE
  if (x <= 0.)
  {
    throw std::domain_error("log_b(x) undefined for x <= 0.");
  }
#endif
  return ::log10(x);
}

//
// Fininiteness
//

/**
 * \brief
 * NaN check.
 * @param x Input value.
 * @return `true` when `x` is NaN.
 */

__boba_host_device__ constexpr bool isnan(double x)
{
#ifdef BOBA_DEVICE_CODE
  /*
  With intel 2023 compiler:
  ./include/BOBA/abstractions/math.hpp:310:12: error: explicit comparison with NaN in fast floating point mode [-Werror,-Wtautological-constant-compare]
  return ::isnan(x);
  */
  return ::isnan(x);
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-constant-compare"
  const bool result = __builtin_isnan(x);
#pragma clang diagnostic pop
  return result;
#elif defined(__GNUC__)
  return __builtin_isnan(x);
#else
  return std::isnan(x);
#endif
}

/**
 * \brief
 * Finite-value check.
 * @param x Input value.
 * @return `true` when `x` is finite.
 */

__boba_host_device__ constexpr bool isfinite(float x)
{
  return ::isfinite(x);
}

/**
 * \brief
 * Finite-value check.
 * @param x Input value.
 * @return `true` when `x` is finite.
 */

__boba_host_device__ constexpr bool isfinite(double x)
{
  return ::isfinite(x);
}

/**
 * \brief
 * Real type determination. By default real_type_t<T> == T. It will be overriden for complex types in BOBA/objects/Complex.hpp
 */
template <typename T>
struct RealType
{
  using type = T;
};

template <typename T>
using real_type_t = typename RealType<T>::type;

/**
 * \brief Extracts the real part of an arithmetic scalar.
 *
 * Fallback for non-complex arithmetic types: real(x) = x.
 * Complex overloads are provided in BOBA/objects/Complex.hpp.
 * @tparam T Arithmetic type.
 * @param x Input value.
 * @return `x`.
 */
template <typename T>
  requires std::is_arithmetic_v<T>
__boba_host_device__ constexpr T real(T x) noexcept
{
  return x;
}

/**
 * \brief Extracts the imaginary part of an arithmetic scalar.
 *
 * Fallback for non-complex arithmetic types: imag(x) = 0.
 * Complex overloads are provided in BOBA/objects/Complex.hpp.
 * @tparam T Arithmetic type.
 * @param x Input value.
 * @return Zero of type `T`.
 */
template <typename T>
  requires std::is_arithmetic_v<T>
__boba_host_device__ constexpr T imag(T) noexcept
{
  return T{0};
}

/**
 * \brief Complex conjugation for arithmetic scalars.
 *
 * For real types, conj(x) = x. Complex overloads are provided
 * in BOBA/objects/Complex.hpp.
 * @tparam T Arithmetic type.
 * @param x Input value.
 * @return `x`.
 */
template <typename T>
  requires std::is_arithmetic_v<T>
__boba_host_device__ constexpr T conj(T x) noexcept
{
  return x;
}

/**
 * \brief
 * Minimum of two values.
 * @tparam comparable Comparable type.
 * @param x Left-hand value.
 * @param y Right-hand value.
 * @return The smaller of `x` and `y`.
 */

template <typename comparable>
__boba_host_device__ constexpr comparable const& min(comparable const& x, comparable const& y)
{
  return (x < y) ? x : y;
}

/**
 * \brief
 * Maximum of two values.
 * @tparam comparable Comparable type.
 * @param x Left-hand value.
 * @param y Right-hand value.
 * @return The larger of `x` and `y`.
 */

template <typename comparable>
__boba_host_device__ constexpr comparable const& max(comparable const& x, comparable const& y)
{
  return (x < y) ? y : x;
}

/**
 * \brief
 * Is x odd?
 * @tparam type Integral type.
 * @param x Input value.
 * @return `true` when `x` is odd.
 */

template <typename type>
  requires std::is_integral_v<type>
__boba_host_device__ constexpr bool is_odd(type x)
{
  return (x & 1); // bitwise and
}

static_assert(is_odd(1));
static_assert(is_odd(3));

/**
 * \brief
 * Is x even?
 * @tparam type Integral type.
 * @param x Input value.
 * @return `true` when `x` is even.
 */

template <typename type>
  requires std::is_integral_v<type>
__boba_host_device__ constexpr bool is_even(type x)
{
  return not(is_odd(x));
}

static_assert(is_even(0));
static_assert(is_even(2));

/**
 * \brief
 * Computes n! for nonnegative n.
 * @tparam data_t Integral type.
 * @param n Input value.
 * @return `n!`.
 */

template <typename data_t>
  requires std::is_integral_v<data_t>
__boba_host_device__
data_t
factorial(data_t n)
{
  boba_assert_nonnegative(n, "Factorial not defined for negative numbers");
  data_t value = 1;
  for (data_t i = 2; i <= n; ++i)
  {
    value *= i;
  }
  return value;
}

/**
 * \brief
 * Computes n!!
 * @tparam data_t Integral type.
 * @param n Input value.
 * @return `n!!`.
 */

template <typename data_t>
  requires std::is_integral_v<data_t>
__boba_host_device__
data_t
double_factorial(data_t n)
{
  boba_assert_nonnegative(n, "Factorial not defined for negative numbers");
  data_t value = 1;
  for (data_t i = n; i > 1; i = i - 2)
  {
    value *= i;
  }
  return value;
}

/**
 * \brief
 * Computes $(1/d!) \prod_{k=0}^{d-1} (n+k)$ (with d=dimension), which gives you the number of non-increasing multi-indices
 * in d-dimensions of size n
 * @tparam dimension Number of dimensions.
 * @tparam index_t Index type.
 * @param n Multi-index size.
 * @return Number of non-increasing multi-indices.
 */
template <size_t dimension>
__boba_host_device__
index_t
number_nonincreasing_multiindices(index_t n)
{
  boba_assert_nonnegative(n, "number_nonincreasing_multiindices not defined for negative inputs");
  index_t value = 1;
  for (index_t k = 0; k < dimension; k++)
  {
    value *= n + k;
  }
  value /= factorial(dimension);
  return value;
}

/**
 * \brief
 * Maps to zero if negative, otherwise returns x: (x + |x|)/2
 * @tparam scalar Real scalar type.
 * @param x Input value.
 * @return Positive part of `x`.
 */

template <typename scalar>
__boba_host_device__ constexpr scalar positive_part(scalar x)
{
  static_assert(std::is_same_v<real_type_t<scalar>, scalar>, "positive_part is not defined for complex type");
  return 0.5 * (x + abs(x));
}

/**
 * \brief
 * Maps to zero if positive, otherwise returns x: (x - |x|)/2
 * @tparam scalar Real scalar type.
 * @param x Input value.
 * @return Negative part of `x`.
 */

template <typename scalar>
__boba_host_device__ constexpr scalar negative_part(scalar x)
{
  static_assert(std::is_same_v<real_type_t<scalar>, scalar>, "negative_part is not defined for complex type");
  return 0.5 * (x - abs(x));
}

/**
 * \brief
 * Highest possible value of type, such that nothing can be greater than it.
 * @tparam type Real scalar type.
 * @return Largest representable value of `type`.
 */

template <typename type>
__boba_host_device__ constexpr type highest_value() noexcept
{
  static_assert(std::is_same_v<real_type_t<type>, type>, "highest_value is not defined for complex types");
  return std::numeric_limits<type>::max();
}

/**
 * \brief
 * Lowest possible value of type, such that nothing can be less than it.
 * @tparam type Real scalar type.
 * @return Lowest representable value of `type`.
 */

template <typename type>
__boba_host_device__ constexpr type lowest_value() noexcept
{
  static_assert(std::is_same_v<real_type_t<type>, type>, "lowest_value is not defined for complex types");
  return std::numeric_limits<type>::lowest();
}

/**
 * \brief
 * Machine precision epsilon.
 * @tparam data_t Real scalar type.
 * @return Machine epsilon for `data_t`.
 */

template <typename data_t>
__boba_host_device__ constexpr data_t epsilon()
{
  static_assert(std::is_same_v<real_type_t<data_t>, data_t>, "epsilon is not defined for complex type");
  return std::numeric_limits<data_t>::epsilon();
}

/**
 * \brief
 * Is x at or below the level of machine precision?
 * @tparam type Scalar type.
 * @param x Input value.
 * @return `true` when `|x|` is at most machine precision.
 */

template <typename type>
__boba_host_device__ constexpr bool is_tiny(type x)
{
  return (::boba::abs(x) <= static_cast<real_type_t<type>>(1.1) * epsilon<type>());
}

#ifdef __INTEL_COMPILE
#pragma warning(pop)
#endif
} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <array>
#include <iomanip>

#if defined(BOBA_CUDA)
#include <cuComplex.h>
#elif defined(BOBA_HIP)
#include <hip/hip_complex.h>
#else
#include <complex>
#endif

#include "BOBA/abstractions/common.hpp"
#include "BOBA/abstractions/math.hpp"

#include <concepts>
#include <type_traits>

namespace boba
{

/**
 * \brief
 * boba::complex<T> type for T = float, double
 */

#if defined(BOBA_CUDA)
template <typename T>
using complex = std::conditional_t<std::is_same_v<T, float>, cuFloatComplex, std::conditional_t<std::is_same_v<T, double>, cuDoubleComplex, void>>;
#elif defined(BOBA_HIP)
template <typename T>
using complex = std::conditional_t<std::is_same_v<T, float>, hipFloatComplex, std::conditional_t<std::is_same_v<T, double>, hipDoubleComplex, void>>;
#else
template <typename T>
using complex = std::conditional_t<std::is_same_v<T, float>, std::complex<float>, std::conditional_t<std::is_same_v<T, double>, std::complex<double>, void>>;
#endif

/**
 * \brief
 * Real type determination. Overrides default implementation from BOBA/abstractions/math.hpp.
 */

template <>
struct RealType<complex<float>>
{
  using type = float;
};

template <>
struct RealType<complex<double>>
{
  using type = double;
};

template <typename type, typename data_t>
concept IsRealDataType = std::same_as<type, real_type_t<data_t>>;

/**
 * @brief Complex real part.
 * @param x Complex value to inspect.
 * @return The real component of `x`.
 */
__boba_host_device__ inline float real(complex<float> x)
{
#if defined(BOBA_CUDA)
  return cuCrealf(x);
#elif defined(BOBA_HIP)
  return hipCrealf(x);
#else
  return std::real(x);
#endif
}

/**
 * @brief Complex real part.
 * @param x Complex value to inspect.
 * @return The real component of `x`.
 */
__boba_host_device__ inline double real(complex<double> x)
{
#if defined(BOBA_CUDA)
  return cuCreal(x);
#elif defined(BOBA_HIP)
  return hipCreal(x);
#else
  return std::real(x);
#endif
}

/**
 * @brief Complex imaginary part.
 * @param x Complex value to inspect.
 * @return The imaginary component of `x`.
 */
__boba_host_device__ inline float imag(complex<float> x)
{
#if defined(BOBA_CUDA)
  return cuCimagf(x);
#elif defined(BOBA_HIP)
  return hipCimagf(x);
#else
  return std::imag(x);
#endif
}

/**
 * @brief Complex imaginary part.
 * @param x Complex value to inspect.
 * @return The imaginary component of `x`.
 */
__boba_host_device__ inline double imag(complex<double> x)
{
#if defined(BOBA_CUDA)
  return cuCimag(x);
#elif defined(BOBA_HIP)
  return hipCimag(x);
#else
  return std::imag(x);
#endif
}

/**
 * @brief Complex conjugate. Overrides default implementation from `BOBA/abstractions/math.hpp`.
 * @param x Complex value to conjugate.
 * @return The conjugate of `x`.
 */
__boba_host_device__ inline boba::complex<double> conj(boba::complex<double> x)
{
  return boba::complex<double>{real(x), -imag(x)};
}

/**
 * @brief Complex conjugate. Overrides default implementation from `BOBA/abstractions/math.hpp`.
 * @param x Complex value to conjugate.
 * @return The conjugate of `x`.
 */
__boba_host_device__ inline boba::complex<float> conj(boba::complex<float> x)
{
  return boba::complex<float>{real(x), -imag(x)};
}

} // namespace boba

// Define operators outside of the boba namespace

/**
 * \brief
 * Complex-complex arithmetic
 */

#if defined(BOBA_CUDA)

/**
 * @brief Complex-complex arithmetic addition.
 * @param x Left operand.
 * @param y Right operand.
 * @return The sum `x + y`.
 */
__boba_host_device__ inline boba::complex<float> operator+(boba::complex<float> x, boba::complex<float> y)
{
  return cuCaddf(x, y);
}

/**
 * @brief Adds two double-precision complex values.
 * @param x Left operand.
 * @param y Right operand.
 * @return The sum `x + y`.
 */
__boba_host_device__ inline boba::complex<double> operator+(boba::complex<double> x, boba::complex<double> y)
{
  return cuCadd(x, y);
}

/**
 * @brief Subtracts two single-precision complex values.
 * @param x Left operand.
 * @param y Right operand.
 * @return The difference `x - y`.
 */
__boba_host_device__ inline boba::complex<float> operator-(boba::complex<float> x, boba::complex<float> y)
{
  return cuCsubf(x, y);
}

/**
 * @brief Subtracts two double-precision complex values.
 * @param x Left operand.
 * @param y Right operand.
 * @return The difference `x - y`.
 */
__boba_host_device__ inline boba::complex<double> operator-(boba::complex<double> x, boba::complex<double> y)
{
  return cuCsub(x, y);
}

/**
 * @brief Multiplies two single-precision complex values.
 * @param x Left operand.
 * @param y Right operand.
 * @return The product `x * y`.
 */
__boba_host_device__ inline boba::complex<float> operator*(boba::complex<float> x, boba::complex<float> y)
{
  return cuCmulf(x, y);
}

/**
 * @brief Multiplies two double-precision complex values.
 * @param x Left operand.
 * @param y Right operand.
 * @return The product `x * y`.
 */
__boba_host_device__ inline boba::complex<double> operator*(boba::complex<double> x, boba::complex<double> y)
{
  return cuCmul(x, y);
}

/**
 * @brief Divides two single-precision complex values.
 * @param x Numerator.
 * @param y Denominator.
 * @return The quotient `x / y`.
 */
__boba_host_device__ inline boba::complex<float> operator/(boba::complex<float> x, boba::complex<float> y)
{
  return cuCdivf(x, y);
}

/**
 * @brief Divides two double-precision complex values.
 * @param x Numerator.
 * @param y Denominator.
 * @return The quotient `x / y`.
 */
__boba_host_device__ inline boba::complex<double> operator/(boba::complex<double> x, boba::complex<double> y)
{
  return cuCdiv(x, y);
}

/**
 * @brief Adds a single-precision complex value in place.
 * @param x Value to update.
 * @param y Addend.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<float>& operator+=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x + y;
  return x;
}

/**
 * @brief Adds a double-precision complex value in place.
 * @param x Value to update.
 * @param y Addend.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<double>& operator+=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x + y;
  return x;
}

/**
 * @brief Subtracts a single-precision complex value in place.
 * @param x Value to update.
 * @param y Subtrahend.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<float>& operator-=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x - y;
  return x;
}

/**
 * @brief Subtracts a double-precision complex value in place.
 * @param x Value to update.
 * @param y Subtrahend.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<double>& operator-=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x - y;
  return x;
}

/**
 * @brief Multiplies a single-precision complex value in place.
 * @param x Value to update.
 * @param y Multiplier.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<float>& operator*=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x * y;
  return x;
}

/**
 * @brief Multiplies a double-precision complex value in place.
 * @param x Value to update.
 * @param y Multiplier.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<double>& operator*=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x * y;
  return x;
}

/**
 * @brief Divides a single-precision complex value in place.
 * @param x Value to update.
 * @param y Divisor.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<float>& operator/=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x / y;
  return x;
}

/**
 * @brief Divides a double-precision complex value in place.
 * @param x Value to update.
 * @param y Divisor.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<double>& operator/=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x / y;
  return x;
}

#elif defined(BOBA_HIP)

/**
 * @brief Adds two single-precision complex values.
 * @param x Left operand.
 * @param y Right operand.
 * @return The sum `x + y`.
 */
__boba_host_device__ inline boba::complex<float> operator+(boba::complex<float> x, boba::complex<float> y)
{
  return hipCaddf(x, y);
}

/**
 * @brief Adds two double-precision complex values.
 * @param x Left operand.
 * @param y Right operand.
 * @return The sum `x + y`.
 */
__boba_host_device__ inline boba::complex<double> operator+(boba::complex<double> x, boba::complex<double> y)
{
  return hipCadd(x, y);
}

/**
 * @brief Subtracts two single-precision complex values.
 * @param x Left operand.
 * @param y Right operand.
 * @return The difference `x - y`.
 */
__boba_host_device__ inline boba::complex<float> operator-(boba::complex<float> x, boba::complex<float> y)
{
  return hipCsubf(x, y);
}

/**
 * @brief Subtracts two double-precision complex values.
 * @param x Left operand.
 * @param y Right operand.
 * @return The difference `x - y`.
 */
__boba_host_device__ inline boba::complex<double> operator-(boba::complex<double> x, boba::complex<double> y)
{
  return hipCsub(x, y);
}

/**
 * @brief Multiplies two single-precision complex values on the host.
 * @param x Left operand.
 * @param y Right operand.
 * @return The product `x * y`.
 */
__boba_host__
inline boba::complex<float> operator*(boba::complex<float> x, boba::complex<float> y)
{
  return hipCmulf(x, y);
}

/**
 * @brief Multiplies two single-precision complex values on the device.
 * @param x Left operand.
 * @param y Right operand.
 * @return The product `x * y`.
 */
__boba_device__
inline boba::complex<float> operator*(boba::complex<float> x, boba::complex<float> y)
{
  return hipCmulf(x, y);
}

/**
 * @brief Multiplies two double-precision complex values on the host.
 * @param x Left operand.
 * @param y Right operand.
 * @return The product `x * y`.
 */
__boba_host__
inline boba::complex<double> operator*(boba::complex<double> x, boba::complex<double> y)
{
  return hipCmul(x, y);
}

/**
 * @brief Multiplies two double-precision complex values on the device.
 * @param x Left operand.
 * @param y Right operand.
 * @return The product `x * y`.
 */
__boba_device__
inline boba::complex<double> operator*(boba::complex<double> x, boba::complex<double> y)
{
  return hipCmul(x, y);
}

/**
 * @brief Divides two single-precision complex values on the host.
 * @param x Numerator.
 * @param y Denominator.
 * @return The quotient `x / y`.
 */
__boba_host__
inline boba::complex<float> operator/(boba::complex<float> x, boba::complex<float> y)
{
  return hipCdivf(x, y);
}

/**
 * @brief Divides two single-precision complex values on the device.
 * @param x Numerator.
 * @param y Denominator.
 * @return The quotient `x / y`.
 */
__boba_device__
inline boba::complex<float> operator/(boba::complex<float> x, boba::complex<float> y)
{
  return hipCdivf(x, y);
}

/**
 * @brief Divides two double-precision complex values on the host.
 * @param x Numerator.
 * @param y Denominator.
 * @return The quotient `x / y`.
 */
__boba_host__
inline boba::complex<double> operator/(boba::complex<double> x, boba::complex<double> y)
{
  return hipCdiv(x, y);
}

/**
 * @brief Divides two double-precision complex values on the device.
 * @param x Numerator.
 * @param y Denominator.
 * @return The quotient `x / y`.
 */
__boba_device__
inline boba::complex<double> operator/(boba::complex<double> x, boba::complex<double> y)
{
  return hipCdiv(x, y);
}

/**
 * @brief Adds a single-precision complex value in place on the host.
 * @param x Value to update.
 * @param y Addend.
 * @return `x` after the update.
 */
__boba_host__
inline boba::complex<float>& operator+=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x + y;
  return x;
}

/**
 * @brief Adds a single-precision complex value in place on the device.
 * @param x Value to update.
 * @param y Addend.
 * @return `x` after the update.
 */
__boba_device__
inline boba::complex<float>& operator+=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x + y;
  return x;
}

/**
 * @brief Adds a double-precision complex value in place on the host.
 * @param x Value to update.
 * @param y Addend.
 * @return `x` after the update.
 */
__boba_host__
inline boba::complex<double>& operator+=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x + y;
  return x;
}

/**
 * @brief Adds a double-precision complex value in place on the device.
 * @param x Value to update.
 * @param y Addend.
 * @return `x` after the update.
 */
__boba_device__
inline boba::complex<double>& operator+=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x + y;
  return x;
}

/**
 * @brief Subtracts a single-precision complex value in place on the host.
 * @param x Value to update.
 * @param y Subtrahend.
 * @return `x` after the update.
 */
__boba_host__
inline boba::complex<float>& operator-=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x - y;
  return x;
}

/**
 * @brief Subtracts a single-precision complex value in place on the device.
 * @param x Value to update.
 * @param y Subtrahend.
 * @return `x` after the update.
 */
__boba_device__
inline boba::complex<float>& operator-=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x - y;
  return x;
}

/**
 * @brief Subtracts a double-precision complex value in place on the host.
 * @param x Value to update.
 * @param y Subtrahend.
 * @return `x` after the update.
 */
__boba_host__
inline boba::complex<double>& operator-=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x - y;
  return x;
}

/**
 * @brief Subtracts a double-precision complex value in place on the device.
 * @param x Value to update.
 * @param y Subtrahend.
 * @return `x` after the update.
 */
__boba_device__
inline boba::complex<double>& operator-=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x - y;
  return x;
}

/**
 * @brief Multiplies a single-precision complex value in place on the host.
 * @param x Value to update.
 * @param y Multiplier.
 * @return `x` after the update.
 */
__boba_host__
inline boba::complex<float>& operator*=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x * y;
  return x;
}

/**
 * @brief Multiplies a single-precision complex value in place on the device.
 * @param x Value to update.
 * @param y Multiplier.
 * @return `x` after the update.
 */
__boba_device__
inline boba::complex<float>& operator*=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x * y;
  return x;
}

/**
 * @brief Multiplies a double-precision complex value in place on the host.
 * @param x Value to update.
 * @param y Multiplier.
 * @return `x` after the update.
 */
__boba_host__
inline boba::complex<double>& operator*=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x * y;
  return x;
}

/**
 * @brief Multiplies a double-precision complex value in place on the device.
 * @param x Value to update.
 * @param y Multiplier.
 * @return `x` after the update.
 */
__boba_device__
inline boba::complex<double>& operator*=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x * y;
  return x;
}

/**
 * @brief Divides a single-precision complex value in place on the host.
 * @param x Value to update.
 * @param y Divisor.
 * @return `x` after the update.
 */
__boba_host__
inline boba::complex<float>& operator/=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x / y;
  return x;
}

/**
 * @brief Divides a single-precision complex value in place on the device.
 * @param x Value to update.
 * @param y Divisor.
 * @return `x` after the update.
 */
__boba_device__
inline boba::complex<float>& operator/=(boba::complex<float>& x, boba::complex<float> y)
{
  x = x / y;
  return x;
}

/**
 * @brief Divides a double-precision complex value in place on the host.
 * @param x Value to update.
 * @param y Divisor.
 * @return `x` after the update.
 */
__boba_host__
inline boba::complex<double>& operator/=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x / y;
  return x;
}

/**
 * @brief Divides a double-precision complex value in place on the device.
 * @param x Value to update.
 * @param y Divisor.
 * @return `x` after the update.
 */
__boba_device__
inline boba::complex<double>& operator/=(boba::complex<double>& x, boba::complex<double> y)
{
  x = x / y;
  return x;
}

#endif

/**
 * \brief
 * Unary minus, mixed complex-real arithmetic
 */

#if defined(BOBA_CUDA) || defined(BOBA_HIP)

/**
 * @brief Unary minus for complex values.
 * @param x Value to negate.
 * @return `-x`.
 */
__boba_host_device__ inline boba::complex<double> operator-(boba::complex<double> x)
{
  return boba::complex<double>{-boba::real(x), -boba::imag(x)};
}

/**
 * @brief Unary minus for complex values.
 * @param x Value to negate.
 * @return `-x`.
 */
__boba_host_device__ inline boba::complex<float> operator-(boba::complex<float> x)
{
  return boba::complex<float>{-boba::real(x), -boba::imag(x)};
}

/**
 * @brief Adds a real value to a single-precision complex value.
 * @param x Real addend.
 * @param y Complex addend.
 * @return The sum `x + y`.
 */
__boba_host_device__ inline boba::complex<float> operator+(float x, boba::complex<float> y)
{
  boba::complex<float> xx{x, 0.0f};
  return xx + y;
}

/**
 * @brief Adds a real value to a double-precision complex value.
 * @param x Real addend.
 * @param y Complex addend.
 * @return The sum `x + y`.
 */
__boba_host_device__ inline boba::complex<double> operator+(double x, boba::complex<double> y)
{
  boba::complex<double> xx{x, 0.0};
  return xx + y;
}

/**
 * @brief Adds a real value to a single-precision complex value.
 * @param x Complex addend.
 * @param y Real addend.
 * @return The sum `x + y`.
 */
__boba_host_device__ inline boba::complex<float> operator+(boba::complex<float> x, float y)
{
  boba::complex<float> yy{y, 0.0f};
  return x + yy;
}

/**
 * @brief Adds a real value to a double-precision complex value.
 * @param x Complex addend.
 * @param y Real addend.
 * @return The sum `x + y`.
 */
__boba_host_device__ inline boba::complex<double> operator+(boba::complex<double> x, double y)
{
  boba::complex<double> yy{y, 0.0};
  return x + yy;
}

/**
 * @brief Subtracts a complex value from a real single-precision value.
 * @param x Real minuend.
 * @param y Complex subtrahend.
 * @return The difference `x - y`.
 */
__boba_host_device__ inline boba::complex<float> operator-(float x, boba::complex<float> y)
{
  boba::complex<float> xx{x, 0.0f};
  return xx - y;
}

/**
 * @brief Subtracts a complex value from a real double-precision value.
 * @param x Real minuend.
 * @param y Complex subtrahend.
 * @return The difference `x - y`.
 */
__boba_host_device__ inline boba::complex<double> operator-(double x, boba::complex<double> y)
{
  boba::complex<double> xx{x, 0.0};
  return xx - y;
}

/**
 * @brief Subtracts a real value from a single-precision complex value.
 * @param x Complex minuend.
 * @param y Real subtrahend.
 * @return The difference `x - y`.
 */
__boba_host_device__ inline boba::complex<float> operator-(boba::complex<float> x, float y)
{
  boba::complex<float> yy{y, 0.0f};
  return x - yy;
}

/**
 * @brief Subtracts a real value from a double-precision complex value.
 * @param x Complex minuend.
 * @param y Real subtrahend.
 * @return The difference `x - y`.
 */
__boba_host_device__ inline boba::complex<double> operator-(boba::complex<double> x, double y)
{
  boba::complex<double> yy{y, 0.0};
  return x - yy;
}

/**
 * @brief Multiplies a real value with a single-precision complex value.
 * @param x Real multiplier.
 * @param y Complex multiplier.
 * @return The product `x * y`.
 */
__boba_host_device__ inline boba::complex<float> operator*(float x, boba::complex<float> y)
{
  boba::complex<float> xx{x, 0.0f};
  return xx * y;
}

/**
 * @brief Multiplies a real value with a double-precision complex value.
 * @param x Real multiplier.
 * @param y Complex multiplier.
 * @return The product `x * y`.
 */
__boba_host_device__ inline boba::complex<double> operator*(double x, boba::complex<double> y)
{
  boba::complex<double> xx{x, 0.0};
  return xx * y;
}

/**
 * @brief Multiplies a single-precision complex value by a real value.
 * @param x Complex multiplier.
 * @param y Real multiplier.
 * @return The product `x * y`.
 */
__boba_host_device__ inline boba::complex<float> operator*(boba::complex<float> x, float y)
{
  boba::complex<float> yy{y, 0.0f};
  return x * yy;
}

/**
 * @brief Multiplies a double-precision complex value by a real value.
 * @param x Complex multiplier.
 * @param y Real multiplier.
 * @return The product `x * y`.
 */
__boba_host_device__ inline boba::complex<double> operator*(boba::complex<double> x, double y)
{
  boba::complex<double> yy{y, 0.0};
  return x * yy;
}

/**
 * @brief Divides a real value by a single-precision complex value.
 * @param x Real numerator.
 * @param y Complex denominator.
 * @return The quotient `x / y`.
 */
__boba_host_device__ inline boba::complex<float> operator/(float x, boba::complex<float> y)
{
  boba::complex<float> xx{x, 0.0f};
  return xx / y;
}

/**
 * @brief Divides a real value by a double-precision complex value.
 * @param x Real numerator.
 * @param y Complex denominator.
 * @return The quotient `x / y`.
 */
__boba_host_device__ inline boba::complex<double> operator/(double x, boba::complex<double> y)
{
  boba::complex<double> xx{x, 0.0};
  return xx / y;
}

/**
 * @brief Divides a single-precision complex value by a real value.
 * @param x Complex numerator.
 * @param y Real denominator.
 * @return The quotient `x / y`.
 */
__boba_host_device__ inline boba::complex<float> operator/(boba::complex<float> x, float y)
{
  boba::complex<float> yy{y, 0.0f};
  return x / yy;
}

/**
 * @brief Divides a double-precision complex value by a real value.
 * @param x Complex numerator.
 * @param y Real denominator.
 * @return The quotient `x / y`.
 */
__boba_host_device__ inline boba::complex<double> operator/(boba::complex<double> x, double y)
{
  boba::complex<double> yy{y, 0.0};
  return x / yy;
}

/**
 * @brief Adds a real value to a double-precision complex value in place.
 * @param x Value to update.
 * @param y Real addend.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<double>& operator+=(boba::complex<double>& x, double y)
{
  x = x + y;
  return x;
}

/**
 * @brief Adds a real value to a single-precision complex value in place.
 * @param x Value to update.
 * @param y Real addend.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<float>& operator+=(boba::complex<float>& x, float y)
{
  x = x + y;
  return x;
}

/**
 * @brief Subtracts a real value from a double-precision complex value in place.
 * @param x Value to update.
 * @param y Real subtrahend.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<double>& operator-=(boba::complex<double>& x, double y)
{
  x = x - y;
  return x;
}

/**
 * @brief Subtracts a real value from a single-precision complex value in place.
 * @param x Value to update.
 * @param y Real subtrahend.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<float>& operator-=(boba::complex<float>& x, float y)
{
  x = x - y;
  return x;
}

/**
 * @brief Multiplies a double-precision complex value by a real value in place.
 * @param x Value to update.
 * @param y Real multiplier.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<double>& operator*=(boba::complex<double>& x, double y)
{
  x = x * y;
  return x;
}

/**
 * @brief Multiplies a single-precision complex value by a real value in place.
 * @param x Value to update.
 * @param y Real multiplier.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<float>& operator*=(boba::complex<float>& x, float y)
{
  x = x * y;
  return x;
}

/**
 * @brief Divides a double-precision complex value by a real value in place.
 * @param x Value to update.
 * @param y Real divisor.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<double>& operator/=(boba::complex<double>& x, double y)
{
  x = x / y;
  return x;
}

/**
 * @brief Divides a single-precision complex value by a real value in place.
 * @param x Value to update.
 * @param y Real divisor.
 * @return `x` after the update.
 */
__boba_host_device__ inline boba::complex<float>& operator/=(boba::complex<float>& x, float y)
{
  x = x / y;
  return x;
}

#endif

/**
 * \brief
 * IO
 */

#if defined(BOBA_CUDA) || defined(BOBA_HIP)
/**
 * @brief IO for single-precision complex values.
 * @param os Output stream.
 * @param z Complex value to print.
 * @return The output stream.
 */
__boba_host__
inline std::ostream& operator<<(std::ostream& os, const boba::complex<float>& z)
{
  os << "("
     << std::setprecision(std::numeric_limits<float>::digits10) << boba::real(z)
     << ","
     << std::setprecision(std::numeric_limits<float>::digits10) << boba::imag(z)
     << ")";
  return os;
}

/**
 * @brief IO for double-precision complex values.
 * @param os Output stream.
 * @param z Complex value to print.
 * @return The output stream.
 */
__boba_host__
inline std::ostream& operator<<(std::ostream& os, const boba::complex<double>& z)
{
  os << "("
     << std::setprecision(std::numeric_limits<double>::digits10) << boba::real(z)
     << ","
     << std::setprecision(std::numeric_limits<double>::digits10) << boba::imag(z)
     << ")";
  return os;
}

/**
 * @brief IO for single-precision complex values.
 * @param is Input stream.
 * @param z Complex value to overwrite.
 * @return The input stream.
 */
__boba_host__
inline std::istream& operator>>(std::istream& is, boba::complex<float>& z)
{
  std::string z_str;
  is >> z_str;
  const auto [loc1, loc2, loc3] = std::array{
    z_str.find('('),
    z_str.find(','),
    z_str.find(')')};
  boba_always_assert((loc1 != std::string::npos) && (loc2 != std::string::npos) && (loc3 != std::string::npos) && (loc1 + 1 < loc2) && (loc2 + 1 < loc3),
                     "complex number string representation must be enclosed by () and contain a comma");
  float x = std::stof(z_str.substr(loc1 + 1, loc2 - loc1 - 1));
  float y = std::stof(z_str.substr(loc2 + 1, loc3 - loc2 - 1));
  z = boba::complex<float>{x, y};
  return is;
}

/**
 * @brief IO for double-precision complex values.
 * @param is Input stream.
 * @param z Complex value to overwrite.
 * @return The input stream.
 */
__boba_host__
inline std::istream& operator>>(std::istream& is, boba::complex<double>& z)
{
  std::string z_str;
  is >> z_str;
  const auto [loc1, loc2, loc3] = std::array{
    z_str.find('('),
    z_str.find(','),
    z_str.find(')')};
  boba_always_assert((loc1 != std::string::npos) && (loc2 != std::string::npos) && (loc3 != std::string::npos) && (loc1 + 1 < loc2) && (loc2 + 1 < loc3),
                     "complex number string representation must be enclosed by () and contain a comma");
  double x = std::stod(z_str.substr(loc1 + 1, loc2 - loc1 - 1));
  double y = std::stod(z_str.substr(loc2 + 1, loc3 - loc2 - 1));
  z = boba::complex<double>{x, y};
  return is;
}
#endif

// math functions

namespace boba
{

/**
 * @brief Complex absolute value.
 * @param x Complex value to inspect.
 * @return The magnitude of `x`.
 */
__boba_host_device__ inline float abs(complex<float> x)
{
#if defined(BOBA_CUDA)
  return cuCabsf(x);
#elif defined(BOBA_HIP)
  return hipCabsf(x);
#else
  return std::abs(x);
#endif
}

/**
 * @brief Complex absolute value.
 * @param x Complex value to inspect.
 * @return The magnitude of `x`.
 */
__boba_host_device__ inline double abs(complex<double> x)
{
#if defined(BOBA_CUDA)
  return cuCabs(x);
#elif defined(BOBA_HIP)
  return hipCabs(x);
#else
  return std::abs(x);
#endif
}

/**
 * @brief Complex exponential using the principal branch cut for the base logarithm.
 * @param x Complex base.
 * @param p Complex exponent.
 * @return `x` raised to `p`.
 */
__boba_host_device__ inline complex<float> pow(complex<float> x, complex<float> p)
{
#if defined(BOBA_CUDA) || defined(BOBA_HIP)
  // x = r * exp(i * theta) = exp(log(r) + i * theta) => log(x) = log(r) + i * theta
  float log_r = log(abs(x));
  float theta = atan(boba::imag(x), boba::real(x));

  // log(x^p) = log(x) * p
  complex<float> log_xp = complex<float>{log_r, theta} * p;

  // log(x^p) = a + i * b => x^p = exp(a + i * b) = exp(a) * cos(b) + i * exp(a) * sin(b)
  float exp_a = exp(boba::real(log_xp));
  float b = boba::imag(log_xp);

  return complex<float>{exp_a * cos(b), exp_a * sin(b)};
#else
  return std::pow(x, p);
#endif
}

/**
 * @brief Complex exponential using the principal branch cut for the base logarithm.
 * @param x Complex base.
 * @param p Complex exponent.
 * @return `x` raised to `p`.
 */
__boba_host_device__ inline complex<double> pow(complex<double> x, complex<double> p)
{
#if defined(BOBA_CUDA) || defined(BOBA_HIP)
  // x = r * exp(i * theta) = exp(log(r) + i * theta) => log(x) = log(r) + i * theta
  double log_r = log(abs(x));
  double theta = atan(boba::imag(x), boba::real(x));

  // log(x^p) = log(x) * p
  complex<double> log_xp = complex<double>{log_r, theta} * p;

  // log(x^p) = a + i * b => x^p = exp(a + i * b) = exp(a) * cos(b) + i * exp(a) * sin(b)
  double exp_a = exp(boba::real(log_xp));
  double b = boba::imag(log_xp);

  return complex<double>{exp_a * cos(b), exp_a * sin(b)};
#else
  return std::pow(x, p);
#endif
}

//
//  Cast from real to complex or preserve scalar type
//  Example: PotentiallyComplex<data_t>::value(1.0)
//      here data_t could be double or complex double.
//

template <typename T>
struct PotentiallyComplex
{
  /**
   * @brief Cast a scalar to `T`.
   * @param x Scalar value to forward.
   * @return `x` converted to `T`.
   */
  template <typename scalar_type>
  __boba_host_device__ static constexpr T value(scalar_type x)
  {
    return static_cast<T>(x);
  }
};

template <>
struct PotentiallyComplex<complex<double>>
{
  /**
   * @brief Returns an existing `complex<double>` unchanged.
   * @param x Value to forward.
   * @return `x`.
   */
  __boba_host_device__ static constexpr complex<double> value(complex<double> x)
  {
    return x;
  }

  /**
   * @brief Converts a scalar value to `complex<double>`.
   * @param x Scalar value to convert.
   * @return `x` with zero imaginary part.
   */
  template <typename scalar_type>
    requires(!std::same_as<std::remove_cvref_t<scalar_type>, complex<double>>)
  __boba_host_device__ static constexpr complex<double> value(scalar_type x)
  {
    return {static_cast<double>(x), 0.0};
  }
};

template <>
struct PotentiallyComplex<complex<float>>
{
  /**
   * @brief Returns an existing `complex<float>` unchanged.
   * @param x Value to forward.
   * @return `x`.
   */
  __boba_host_device__ static constexpr complex<float> value(complex<float> x)
  {
    return x;
  }

  /**
   * @brief Converts a scalar value to `complex<float>`.
   * @param x Scalar value to convert.
   * @return `x` with zero imaginary part.
   */
  template <typename scalar_type>
    requires(!std::same_as<std::remove_cvref_t<scalar_type>, complex<float>>)
  __boba_host_device__ static constexpr complex<float> value(scalar_type x)
  {
    return {static_cast<float>(x), 0.0f};
  }
};
} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <math.h>

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
 * Heaviside function (0 or 1).
 * @tparam scalar Scalar type.
 * @param x Input value.
 * @return `0` when `x < 0`, otherwise `1`.
 */

template <typename scalar>
__boba_host_device__ constexpr scalar heaviside(scalar x)
{
  return (x < static_cast<scalar>(0)) ? static_cast<scalar>(0) : static_cast<scalar>(1);
}

/**
 * \brief
 * Two-point Lagrange interpolation.
 * @tparam scalar Scalar type.
 * @param xi Interpolation abscissae.
 * @param x Evaluation point.
 * @return Lagrange weights for `x`.
 */

template <typename scalar>
__boba_host_device__ constexpr Array<scalar, 2> lagrange_weights(Array<scalar, 2> xi, scalar x)
{
  scalar l0 = (x - xi[1]) / (xi[0] - xi[1]);
  scalar l1 = (x - xi[0]) / (xi[1] - xi[0]);
  return Array<scalar, 2>{l0, l1};
}

/**
 * \brief
 * Two-point Lagrange interpolation.
 * @tparam scalar Scalar type.
 * @param xi Interpolation abscissae.
 * @param yi Interpolation ordinates.
 * @param x Evaluation point.
 * @return Interpolated value at `x`.
 */

template <typename scalar>
__boba_host_device__ constexpr scalar lagrange_interpolation(Array<scalar, 2> xi, Array<scalar, 2> yi, scalar x)
{
  auto weights = lagrange_weights(xi, x);
  return weights[0] * yi[0] + weights[1] * yi[1];
}

/**
 * \brief
 * Inverse of two-point Lagrange interpolation.
 * @tparam scalar Scalar type.
 * @param xi Interpolation abscissae.
 * @param yi Interpolation ordinates.
 * @param y Interpolated value.
 * @return `x` such that the two-point interpolant equals `y`.
 */

template <typename scalar>
__boba_host_device__ constexpr scalar inverse_lagrange_interpolation(Array<scalar, 2> xi, Array<scalar, 2> yi, scalar y)
{
  double dx = xi[0] - xi[1];
  double dy = yi[0] - yi[1];
  double x = (y * dx + yi[0] * xi[1] - yi[1] * xi[0]) / dy;
  return x;
}

/**
 * \brief
 * Assumes x is periodic over [begin, end) and maps x back to that interval
 * @tparam scalar Scalar type.
 * @param x Input value.
 * @param begin Interval lower bound.
 * @param end Interval upper bound.
 * @return `x` mapped into `[begin, end)`.
 */

template <typename scalar>
__boba_host_device__ constexpr scalar periodic(const scalar x, const scalar begin, const scalar end)
{
  scalar period = end - begin;
  if (x >= end)
  {
    scalar periods_over = 1.0 + floor((x - end) / period);
    return x - periods_over * period;
  }
  if (x < begin)
  {
    scalar periods_under = ceil((begin - x) / period);
    return x + periods_under * period;
  }
  return x;
}

//
// Summations
//

/**
 * \brief
 * Evaluates \sum_{i=0}^N i
 * @param N Upper summation limit.
 * @return Sum of integers from `0` through `N`.
 */

__boba_host_device__ constexpr size_t sum_of_i(size_t N)
{
  return (N * (N + 1)) / 2;
}

/**
 * \brief
 * Evaluates \sum_{i=0}^N i^2
 * @param N Upper summation limit.
 * @return Sum of squares from `0` through `N`.
 */

__boba_host_device__ constexpr size_t sum_of_i2(size_t N)
{
  return (N * (N + 1) * (2 * N + 1)) / 6;
}

//
// Binary search
//

/**
 * \brief
 * Finds id_bracket_left, id_bracket_right such that view(id_bracket_left) <= search_value < view(id_bracket_right)
 * The brackets should be initialized to good guesses
 * @tparam value_t Search value type.
 * @tparam viewlike_t View-like callable type.
 * @param search_value Value to locate.
 * @param view Monotone view used for the search.
 * @param id_bracket_left Lower bracket index, updated in place.
 * @param id_bracket_right Upper bracket index, updated in place.
 */

template <typename value_t, typename viewlike_t>
__boba_host_device__ void binary_search(
  value_t search_value,
  viewlike_t const& view,
  size_t& id_bracket_left,
  size_t& id_bracket_right)
{
  while (id_bracket_right > id_bracket_left + 1)
  {
    size_t id_guess = (id_bracket_left + id_bracket_right) / 2;
    auto value = view(id_guess);
    if (value < search_value)
    {
      id_bracket_left = id_guess;
    }
    else
    {
      id_bracket_right = id_guess;
    }
  }
}

#ifdef __INTEL_COMPILE
#pragma warning(pop)
#endif
} // namespace boba

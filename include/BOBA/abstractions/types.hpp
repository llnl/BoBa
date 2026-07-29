// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * @brief Default index type used by BoBa containers and helpers.
 */
using index_t = std::size_t;

#ifdef __INTEL_COMPILE
// intel compiler complain about "external function definition with no prior declaration"
// disabling this warning here
#pragma warning(push)
#pragma warning(disable : 1418)
#endif

/**
 * \brief
 * For size_t literals
 * @param n Unsigned literal value.
 * @return `n` converted to `std::size_t`.
 */

constexpr std::size_t operator""_z(unsigned long long n) noexcept
{
  return std::size_t{n};
}

#ifdef __INTEL_COMPILE
#pragma warning(pop)
#endif

} // namespace boba

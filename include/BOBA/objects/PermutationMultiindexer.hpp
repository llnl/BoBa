// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * @brief Utility for working with permutations of the sequence {0, 1, ..., N-1}.
 *
 * This class provides:
 *  - Conversion from a permutation index (lexicographic order) to a permutation.
 *  - Conversion from a permutation back to its lexicographic index.
 *
 * The permutations are taken over the set {0, 1, ..., N-1} and are ordered
 * lexicographically. For example, for N = 3, the permutations and their indices are:
 *
 *  index 0: {0, 1, 2}
 *  index 1: {0, 2, 1}
 *  index 2: {1, 0, 2}
 *  index 3: {1, 2, 0}
 *  index 4: {2, 0, 1}
 *  index 5: {2, 1, 0}
 *
 * Template parameter @p N must be greater than 0.
 *
 * @tparam dimension Size of the permutation, i.e. the number of elements.
 */
template <size_t dimension>
struct PermutationMultiindexer
{
  static_assert(dimension > 0, "Permutation<N> requires N > 0");
  static_assert(dimension < 20, "factorial(20) is too big!, we need a different index_t");

  using array_type = Array<index_t, dimension>;

  /**
   * @brief Returns the number of permutations.
   * @return `dimension!`.
   */
  static constexpr index_t size() noexcept
  {
    return factorial(dimension);
  }

  /**
   * @brief Constructs the k-th permutation in lexicographic order.
   * @param index Zero-based permutation index to generate.
   * @return The permutation corresponding to `index`.
   */
  static array_type multiindex(index_t index)
  {
    boba_assert_lt(index, size(), "Index out of range.");

    // Factoradic representation of k
    Array<index_t, dimension> factoradic{};
    {
      index_t m = index;
      for (size_t i = 1; i <= dimension; ++i)
      {
        factoradic[dimension - i] = mod(m, i);
        m /= i;
      }
    }

    // Convert factoradic to permutation
    // Start with a sorted list of available elements
    Array<index_t, dimension> elements{};
    for (size_t i = 0; i < dimension; ++i)
    {
      elements[i] = static_cast<index_t>(i);
    }

    array_type result{};
    for (size_t i = 0; i < dimension; ++i)
    {
      const index_t idx = factoradic[i];

      // choose idx-th remaining element
      result[i] = elements[idx];

      // remove that element from the pool by shifting
      for (size_t j = static_cast<size_t>(idx); j + 1 < dimension - i; ++j)
      {
        elements[j] = elements[j + 1];
      }
    }

    return result;
  }

  /**
   * @brief Computes the lexicographic index of a given permutation.
   * @param perm Permutation whose index is to be computed.
   * @return The zero-based lexicographic index of `perm`.
   */
  static index_t index(const array_type& perm)
  {
    // Validate permutation and build a presence array.
    boba_assert(is_valid_permutation(perm), "Invalid permutation");

    // Frequency array: how many times each value is still "available"
    auto freq = filled_array<dimension, size_t>(1);

    index_t index = 0;

    // For each position i, count how many unused smaller values we could put here
    // and add the corresponding block of permutations.
    for (size_t i = 0; i < dimension; ++i)
    {
      const index_t v = perm[i];
      size_t smaller_unused = 0;

      // Count smaller values that are still available
      for (index_t x = 0; x < v; ++x)
      {
        if (freq[static_cast<size_t>(x)] > 0)
        {
          ++smaller_unused;
        }
      }

      const index_t block_size = factorial(dimension - 1 - i);
      index += static_cast<index_t>(smaller_unused) * block_size;

      // Mark v as used
      freq[static_cast<size_t>(v)] = 0;
    }

    return index;
  }
};

} // namespace boba

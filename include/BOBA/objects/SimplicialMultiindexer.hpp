// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace boba
{

/**
 * \brief
 * Simplicial multiindexer which can be used in device code. Helps convert between index and a multiindex.
 * Similar to the regular multiindexer, but only allows multiindices where the components are non-increasing
 * -- e.g. the upper triangular part of a 2-tensor (matrix).
 */

template <std::size_t _dimension>
struct SimplicialMultiindexer
{
  static constexpr std::size_t dimension = _dimension;
  static constexpr std::size_t dimension_m1 = (_dimension > 2) ? _dimension - 1 : 2_z;

  using index_array = ::boba::Array<index_t, dimension>;
  using index_subarray = ::boba::Array<index_t, dimension_m1>;

  /**
   * @brief Converts a simplicial linear index into a multi-index.
   * @param _size Size of each dimension.
   * @param index Flattened simplicial index.
   * @return The corresponding non-increasing multi-index.
   */
  __boba_host_device__ static constexpr index_array multiindex(index_t _size, index_t index) noexcept
  {
    index_array indices = ::boba::filled_array<dimension>(static_cast<index_t>(0));
    if constexpr (dimension > 2)
    {
      index_t pow_arg = factorial(dimension) * (number_nonincreasing_multiindices<dimension>(_size) - index);
      double power = 1.0 / dimension;
      index_t last_entry = floor(std::max(_size + dimension_m1 / 4. - pow(static_cast<double>(pow_arg), power), 0.));
      index_t sub_index = compute_subindex(_size, index, last_entry);
      while (sub_index >= number_nonincreasing_multiindices<dimension_m1>(_size))
      {
        last_entry += 1;
        sub_index = compute_subindex(_size, index, last_entry);
      }
      indices[dimension - 1] = last_entry;
      SimplicialMultiindexer<dimension_m1> sub_multiindexer(_size);
      index_subarray sub_multiindex = sub_multiindexer.multiindex(_size, sub_index);
      for (size_t d = 0; d < dimension_m1; d++)
      {
        indices[d] = sub_multiindex[d];
      }
    }
    else if constexpr (dimension == 2)
    {
      double sqrtb = sqrt(static_cast<double>((2 * _size + 1) * (2 * _size + 1) - 8 * index));
      indices[1] = (2 * _size + 1 - sqrtb) / 2;
      indices[0] = index + indices[1] * (indices[1] + 1) / 2 - indices[1] * _size;
    }
    else if constexpr (dimension == 1)
    {
      indices[0] = index;
    }
    return indices;
  }

  /**
   * @brief Computes the recursive sub-index for the last entry.
   * @param _size Size of each dimension.
   * @param index Flattened simplicial index.
   * @param last_entry Candidate final entry in the multi-index.
   * @return The corresponding sub-index in the lower-dimensional problem.
   */
  __boba_host_device__ static index_t compute_subindex(index_t _size, index_t index, index_t last_entry)
  {
    SimplicialMultiindexer<dimension_m1> sub_multiindexer(_size);
    index_subarray cloned_last_index = ::boba::filled_array<dimension_m1>(last_entry);
    index_t sub_index = index + sub_multiindexer.index(cloned_last_index) + number_nonincreasing_multiindices<dimension>(_size - last_entry) - number_nonincreasing_multiindices<dimension>(_size);
    return sub_index;
  }

  /**
   * \brief default constructor
   */
  SimplicialMultiindexer() = default;

  /**
   * \brief copy constructor
   */
  SimplicialMultiindexer(SimplicialMultiindexer const&) = default;

  /**
   * \brief move constructor
   */
  SimplicialMultiindexer(SimplicialMultiindexer&&) = default;

  /**
   * \brief copy assignment operator
   */
  SimplicialMultiindexer& operator=(SimplicialMultiindexer const&) = default;

  /**
   * \brief move assignment operator
   */
  SimplicialMultiindexer& operator=(SimplicialMultiindexer&&) = default;

  /**
   * \brief destructor
   */
  ~SimplicialMultiindexer() = default;

  /**
   * @brief Constructs a simplicial multiindexer for a fixed per-dimension size.
   * @param _size Size of each dimension.
   */
  constexpr SimplicialMultiindexer(
    index_t _size)
      : m_size(_size)
  {
  }

  /**
   * @brief Returns the total number of simplicial multi-indices.
   * @return The number of valid non-increasing indices.
   */
  __boba_host_device__ constexpr index_t total_size() const
  {
    return number_nonincreasing_multiindices<dimension>(m_size);
  }

  /**
   * @brief Returns the size used for each dimension.
   * @return The shared dimension size.
   */
  __boba_host_device__ constexpr index_t size_per_dimension() const
  {
    return m_size;
  }

  /**
   * @brief Converts a simplicial multi-index into a linear index.
   * @param indices Non-increasing multi-index to flatten.
   * @return The corresponding simplicial index.
   */
  __boba_host_device__
  index_t
  index(index_array indices) const
  {
    index_t g = 0;
    if constexpr (dimension > 2_z)
    {
      index_subarray sub_indices = get_sub_multiindex(indices);
      index_t last_index = indices[dimension - 1];
      index_subarray cloned_last_index = ::boba::filled_array<dimension_m1>(last_index);
      SimplicialMultiindexer<dimension_m1> sub_multiindexer(m_size);

      g = sub_multiindexer.index(sub_indices) + number_nonincreasing_multiindices<dimension>(m_size) - number_nonincreasing_multiindices<dimension>(m_size - last_index) - sub_multiindexer.index(cloned_last_index);
    }
    else if constexpr (dimension == 2_z)
    {
      g = indices[0] + m_size * indices[1] - indices[1] * (indices[1] + 1) / 2;
    }
    else if constexpr (dimension == 1_z)
    {
      g = indices[0];
    }

    return g;
  }

  /**
   * @brief Drops the last entry from a multi-index.
   * @param mid Full simplicial multi-index.
   * @return The leading `dimension - 1` entries.
   */
  __boba_host_device__
  index_subarray
  get_sub_multiindex(index_array mid) const
  {
    index_subarray sub_mid;
    for (size_t d = 0; d < dimension - 1; d++)
    {
      sub_mid[d] = mid[d];
    }
    return sub_mid;
  }

  /**
   * @brief Converts a simplicial linear index into a checked multi-index.
   * @param index Flattened simplicial index.
   * @return The corresponding non-increasing multi-index.
   */
  __boba_host_device__
  index_array
  multiindex(index_t index) const
  {
    if constexpr (dimension == 0)
    {
      return index_array{};
    }
    return assert_indices(multiindex(m_size, index));
  }

  /**
   * @brief Generates the next permutation in lexicographic order for the simplicial convention.
   *
   * This algorithm is adapted from the Wikipedia page for "Permutation". Here the entries are
   * arranged in non-increasing order rather than non-decreasing order.
   *
   * @param indices Permutation to update in place.
   */
  __boba_host_device__ void get_next_permutation(index_array& indices) const
  {
    size_t ind_k = dimension - 2;
    while (ind_k > 0 && (indices[ind_k + 1] >= indices[ind_k]))
    {
      ind_k -= 1;
    }
    size_t ind_l = dimension - 1;
    while (ind_l > 0 && (indices[ind_l] >= indices[ind_k]))
    {
      ind_l -= 1;
    }

    // Swap the ind_k and ind_l entries
    index_t temp_k = indices[ind_k];
    indices[ind_k] = indices[ind_l];
    indices[ind_l] = temp_k;

    // Reverse order of entries from ind_k+1 to the end of the array
    size_t i = ind_k + 1;
    size_t j = dimension - 1;
    while (i < j)
    {
      index_t t2 = indices[i];
      indices[i] = indices[j];
      indices[j] = t2;
      ++i;
      --j;
    }
  }

  /**
   * @brief Returns the number of unique permutations of a simplicial multi-index.
   *
   * If there are repeated elements, permutations that only swap equal entries are counted once.
   *
   * @param indices Non-increasing multi-index to analyze.
   * @return The number of unique permutations of `indices`.
   */
  __boba_host_device__
  size_t
  number_permutations(index_array indices) const
  {
    assert_indices(indices);

    size_t number_permutations = factorial(dimension);

    size_t entry_multiplicity = 1;
    for (size_t i = 1; i < dimension; i++)
    {
      if (indices[i] == indices[i - 1])
      {
        entry_multiplicity += 1;
      }
      else
      {
        number_permutations /= factorial(entry_multiplicity);
        entry_multiplicity = 1;
      }
    }
    if (entry_multiplicity > 1)
    {
      number_permutations /= factorial(entry_multiplicity);
    }

    return number_permutations;
  }

  /**
   * @brief Returns the number of dimensions.
   * @return The compile-time dimension.
   */
  __boba_host_device__ constexpr index_t get_dimension() const noexcept
  {
    return dimension;
  }

  /**
   * @brief Returns whether the multiindexer has zero dimensions.
   * @return `true` when `dimension` is zero.
   */
  __boba_host_device__ constexpr bool empty() const noexcept
  {
    return (dimension == 0);
  }

protected:
  index_t m_size = static_cast<index_t>(0);

  /**
   * @brief Validates a candidate simplicial multi-index.
   * @param indices Multi-index to validate.
   * @return The validated multi-index.
   */
  __boba_host_device__
  index_array
  assert_indices(index_array indices) const
  {
    boba_assert_nonnegative(indices, "Negative index");
    for (index_t ind = 0; ind < dimension - 1; ind++)
    {
      boba_assert_lt(indices[ind + 1], indices[ind] + 1, "Indices must be in non-increasing order");
    }
    boba_assert_lt(indices, size_per_dimension(), "Out of bounds");
    return indices;
  }
};

} // namespace boba

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
 * Multiindexer which can be used in device code. Helps convert between index and a multiindex.
 */

template <std::size_t _dimension>
struct Multiindexer
{
  static constexpr std::size_t dimension = _dimension;

  using index_array = ::boba::Array<index_t, dimension>;

  /**
   * @brief Computes row-major strides for a size array.
   * @param _sizes Extent of each dimension.
   * @return The stride for each dimension.
   */
  __boba_host_device__ static constexpr index_array precompute_strides(index_array _sizes) noexcept
  {
    index_array _strides = ::boba::filled_array<dimension>(static_cast<index_t>(0));
    if constexpr (dimension > 0_z)
    {
      _strides[0] = 1;
      for (std::size_t d = 1; d < dimension; ++d)
      {
        _strides[d] = _sizes[d - 1] * _strides[d - 1];
      }
    }
    return _strides;
  }

  /**
   * @brief Converts a multi-index into a linear index.
   * @param _strides Strides associated with each dimension.
   * @param indices Index values for each dimension.
   * @return The flattened index.
   */
  __boba_host_device__ static constexpr index_t index(index_array _strides, index_array indices) noexcept
  {
    index_t index = 0;
    if constexpr (dimension > 0_z)
    {
      for (std::size_t d = 0; d < dimension; ++d)
      {
        index += _strides[d] * indices[d];
      }
    }
    return index;
  }

  /**
   * @brief Converts a linear index into a multi-index.
   * @param _sizes Extent of each dimension.
   * @param index Flattened index to expand.
   * @return The corresponding multi-index.
   */
  __boba_host_device__ static constexpr index_array multiindex(index_array _sizes, index_t index) noexcept
  {
    index_array indices;
    if constexpr (dimension > 0_z)
    {
      for (std::size_t d = 0; d < dimension - 1; d++)
      {
        indices[d] = index % _sizes[d];
        index /= _sizes[d];
      }
      indices[dimension - 1] = index;
    }
    return indices;
  }

  /**
   * \brief default constructor
   */
  Multiindexer() = default;

  /**
   * \brief copy constructor
   */
  Multiindexer(Multiindexer const&) = default;

  /**
   * \brief move constructor
   */
  Multiindexer(Multiindexer&&) = default;

  /**
   * \brief copy assignment operator
   */
  Multiindexer& operator=(Multiindexer const&) = default;

  /**
   * \brief move assignment operator
   */
  Multiindexer& operator=(Multiindexer&&) = default;

  /**
   * \brief destructor
   */
  ~Multiindexer() = default;

  /**
   * @brief Constructs a multiindexer from explicit sizes and strides.
   * @param _sizes Extent of each dimension.
   * @param _strides Stride of each dimension.
   */
  constexpr Multiindexer(
    index_array _sizes,
    index_array _strides) noexcept
      : m_sizes(_sizes),
        m_strides(_strides)
  {
  }

  /**
   * @brief Constructs a multiindexer and computes its strides.
   * @param _sizes Extent of each dimension.
   */
  constexpr Multiindexer(
    index_array _sizes) noexcept
      : m_sizes(_sizes),
        m_strides(precompute_strides(_sizes))
  {
  }

  /**
   * @brief Returns the total number of addressable entries.
   * @return The product of all dimension sizes.
   */
  __boba_host_device__ constexpr index_t size() const
  {
    return ::boba::product(m_sizes);
  }

  /**
   * @brief Returns the array of sizes.
   * @return The extent of each dimension.
   */
  __boba_host_device__ constexpr index_array sizes() const
  {
    return m_sizes;
  }

  /**
   * @brief Returns the array of strides.
   * @return The stride for each dimension.
   */
  __boba_host_device__ constexpr index_array strides() const
  {
    return m_strides;
  }

  /**
   * @brief Returns the ith stride.
   * @param i Dimension index.
   * @return The stride for dimension `i`.
   */
  __boba_host_device__ constexpr index_t strides(index_t i) const
  {
    return m_strides[i];
  }

  /**
   * @brief Returns one size entry.
   * @param i Dimension index.
   * @return The size for dimension `i`.
   */
  __boba_host_device__ constexpr index_t sizes(index_t i) const
  {
    boba_assert_nonnegative(i, "Negative index");
    boba_assert_lt(i, static_cast<index_t>(dimension), "Out of range size");
    return m_sizes[i];
  }

  /**
   * @brief Converts a checked multi-index into a linear index.
   * @param indices Index values for each dimension.
   * @return The flattened index.
   */
  __boba_host_device__
  index_t
  index(index_array indices) const
  {
    return index(m_strides, assert_indices(indices));
  }

  /**
   * @brief Converts a linear index into a checked multi-index.
   * @param index Flattened index to expand.
   * @return The corresponding multi-index.
   */
  __boba_host_device__
  index_array
  multiindex(index_t index) const
  {
    if constexpr (dimension == 0)
    {
      return index_array{};
    }
    return assert_indices(multiindex(m_sizes, index));
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
  index_array m_sizes = ::boba::filled_array<dimension>(static_cast<index_t>(0));
  index_array m_strides = ::boba::filled_array<dimension>(static_cast<index_t>(0));

  /**
   * @brief Validates a candidate multi-index.
   * @param indices Index values to validate.
   * @return The validated indices.
   */
  __boba_host_device__
  index_array
  assert_indices(index_array indices) const
  {
    boba_assert_nonnegative(indices, "Negative index");
    boba_assert_lt(indices, sizes(), "Out of bounds");
    return indices;
  }
};

} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief Statically sized matrix backed by StaticTensor storage.
 */
template <typename _sizes, typename _data_t>
struct StaticMatrix : StaticTensor<_sizes, _data_t>
{
  using base = StaticTensor<_sizes, _data_t>;

  template <std::size_t dim, std::size_t new_size>
  using resize_dimension_type = StaticMatrix<
    typename _sizes::template replace_value_at_index_t<new_size, dim>,
    _data_t>;

  using typename base::t_sizes;

  using typename base::data_array;
  using typename base::data_t;
  using typename base::index_array;

  using base::size;

  /**
   * \brief Constructs a matrix from a flat row-major coefficient array.
   * \param data Flat storage used to initialize the matrix entries.
   */
  __boba_host_device__ constexpr StaticMatrix(Array<_data_t, size()> const data)
      : base{}
  {
    auto coefficient_view = base::view();
    for (size_t i = 0; i < base::view_type::size(); i++)
    {
      coefficient_view(i) = data[i];
    }
  }

  /**
   * \brief default constructor
   */
  constexpr StaticMatrix() = default;

  /**
   * \brief Returns the number of rows in the matrix.
   * \return The compile-time row count.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_t rows() const noexcept
  {
    return base::sizes(0);
  }

  /**
   * \brief Returns the number of columns in the matrix.
   * \return The compile-time column count.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_t cols() const noexcept
  {
    return base::sizes(1);
  }
};

} // namespace boba

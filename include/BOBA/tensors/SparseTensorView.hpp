// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <type_traits>

namespace boba
{

/**
 * \brief Read-only sparse tensor view backed by per-dimension index lists and values.
 */
template <typename _accessor, std::size_t _dimension>
struct SparseTensorView : Multiindexer<_dimension>
{
  static constexpr std::size_t dimension = _dimension;

  using accessor = _accessor;
  using data_t = typename accessor::data_t;
  using value_t = std::remove_const_t<data_t>;
  using data_pointer = typename accessor::data_pointer;
  using data_const_pointer = data_t const*;
  using index_array = ::boba::Array<index_t, dimension>;
  using index_pointer_array = ::boba::Array<index_t const*, dimension>;

  using base = Multiindexer<dimension>;
  using base::assert_indices;
  using base::get_dimension;
  using base::index;
  using base::multiindex;
  using base::precompute_strides;
  using base::size;
  using base::sizes;
  using base::strides;

  SparseTensorView() = default;
  SparseTensorView(SparseTensorView const&) = default;
  SparseTensorView(SparseTensorView&&) = default;
  SparseTensorView& operator=(SparseTensorView const&) = default;
  SparseTensorView& operator=(SparseTensorView&&) = default;
  ~SparseTensorView() = default;

  /**
   * \brief Constructs a sparse tensor view from sparse buffers and extents.
   */
  constexpr SparseTensorView(index_pointer_array index_lists,
                             data_pointer values,
                             index_t number_nonzeros,
                             index_array view_sizes,
                             index_array view_strides) noexcept
      : base(view_sizes, view_strides),
        m_index_lists(index_lists),
        m_values(values),
        m_number_nonzeros(number_nonzeros)
  {
  }

  /**
   * \brief Constructs a sparse tensor view from sparse buffers and extents.
   */
  constexpr SparseTensorView(index_pointer_array index_lists,
                             data_pointer values,
                             index_t number_nonzeros,
                             index_array view_sizes) noexcept
      : base(view_sizes),
        m_index_lists(index_lists),
        m_values(values),
        m_number_nonzeros(number_nonzeros)
  {
  }

  /**
   * \brief Returns the stored value at a tensor multi-index or zero if absent.
   */
  __boba_host_device__ value_t operator()(index_array indices) const
  {
    assert_indices(indices);
    for (index_t entry = 0; entry < m_number_nonzeros; ++entry)
    {
      bool entry_matches = true;
      for (std::size_t d = 0; d < dimension; ++d)
      {
        entry_matches = entry_matches && (m_index_lists[d][entry] == indices[d]);
      }
      if (entry_matches)
      {
        return accessor{}.access(m_values, entry);
      }
    }
    return value_t{};
  }

  /**
   * \brief Returns the stored value at a linear index or zero if absent.
   */
  __boba_host_device__ value_t operator()(index_t linear_index) const
  {
    boba_assert_nonnegative(linear_index, "Negative index");
    boba_assert_lt(linear_index, size(), "Out of bounds");

    return this->operator()(multiindex(linear_index));
  }

  /**
   * \brief Returns the sparse index-list buffer for one tensor dimension.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_t const* index_list(std::size_t d) const noexcept
  {
    return m_index_lists[d];
  }

  /**
   * \brief Returns all sparse index-list buffers.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_pointer_array index_lists() const noexcept
  {
    return m_index_lists;
  }

  /**
   * \brief Returns the sparse value buffer.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_pointer values() const noexcept
  {
    return m_values;
  }

  /**
   * \brief Returns the number of explicitly stored entries.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_t number_nonzeros() const noexcept
  {
    return m_number_nonzeros;
  }

private:
  index_pointer_array m_index_lists = ::boba::filled_array<dimension>(static_cast<index_t const*>(nullptr));
  data_pointer m_values = nullptr;
  index_t m_number_nonzeros = 0;
};

} // namespace boba

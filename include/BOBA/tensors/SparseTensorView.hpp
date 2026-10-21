// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <type_traits>

namespace boba
{

/**
 * \brief Read-only sparse tensor view backed by sorted linear indices and values.
 *
 * Entry `e` is stored at the column-major linear index `linear_indices_ptr()[e]`.
 * Linear indices are strictly increasing, so lookups use binary search.
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
  constexpr SparseTensorView(index_t const* linear_indices,
                             data_pointer values,
                             index_t nnz,
                             index_array view_sizes,
                             index_array view_strides) noexcept
      : base(view_sizes, view_strides),
        m_linear_indices(linear_indices),
        m_values(values),
        m_nnz(nnz)
  {
  }

  /**
   * \brief Constructs a sparse tensor view from sparse buffers and extents.
   */
  constexpr SparseTensorView(index_t const* linear_indices,
                             data_pointer values,
                             index_t nnz,
                             index_array view_sizes) noexcept
      : base(view_sizes),
        m_linear_indices(linear_indices),
        m_values(values),
        m_nnz(nnz)
  {
  }

  /**
   * \brief Returns the stored value at a tensor multi-index or zero if absent.
   */
  __boba_host_device__ value_t operator()(index_array indices) const
  {
    return this->operator()(index(indices));
  }

  /**
   * \brief Returns the stored value at a linear index or zero if absent.
   */
  __boba_host_device__ value_t operator()(index_t linear_index) const
  {
    boba_assert_nonnegative(linear_index, "Negative index");
    boba_assert_lt(linear_index, size(), "Out of bounds");

    const index_t entry = entry_lower_bound(linear_index);
    if (entry < m_nnz && m_linear_indices[entry] == linear_index)
    {
      return accessor{}.access(m_values, entry);
    }
    return value_t{};
  }

  /**
   * \brief Returns the first entry whose linear index is not less than \p linear_index.
   *
   * Returns nnz() when every stored linear index is smaller. Binary search,
   * O(log nnz); runs wherever it is called (host code or a device kernel) and
   * requires the view's buffers to be accessible there.
   */
  [[nodiscard]]
  __boba_host_device__ index_t entry_lower_bound(index_t linear_index) const noexcept
  {
    index_t low = 0;
    index_t high = m_nnz;
    while (low < high)
    {
      const index_t middle = low + (high - low) / 2;
      if (m_linear_indices[middle] < linear_index)
      {
        low = middle + 1;
      }
      else
      {
        high = middle;
      }
    }
    return low;
  }

  /**
   * \brief Returns the sorted linear-index buffer.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_t const* linear_indices_ptr() const noexcept
  {
    return m_linear_indices;
  }

  /**
   * \brief Returns the linear index of a stored sparse entry.
   */
  [[nodiscard]]
  __boba_host_device__ index_t entry_linear_index(index_t entry) const
  {
    boba_assert_lt(entry, m_nnz, "Out of bounds");
    return m_linear_indices[entry];
  }

  /**
   * \brief Returns the tensor multi-index for a stored sparse entry.
   */
  [[nodiscard]]
  __boba_host_device__ index_array entry_multiindex(index_t entry) const
  {
    return base::multiindex(sizes(), entry_linear_index(entry));
  }

  /**
   * \brief Returns the sparse value buffer.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_pointer values_ptr() const noexcept
  {
    return m_values;
  }

  /**
   * \brief Returns the number of explicitly stored entries.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_t nnz() const noexcept
  {
    return m_nnz;
  }

private:
  index_t const* m_linear_indices = nullptr;
  data_pointer m_values = nullptr;
  index_t m_nnz = 0;
};

} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <cstdint>
#include <type_traits>

namespace boba
{

/**
 * \brief Non-owning CSR view usable in device code and for sparse-library handoff.
 *
 * The pointers alias the owning `CSRMatrix` containers; the execution space is
 * tracked by the owner, not the view. The view is invalidated when the owning
 * matrix is resized or destroyed. Index arrays are zero-based signed 64-bit
 * integers, matching the `CUSPARSE_INDEX_64I` / `HIPSPARSE_INDEX_64I` / Eigen
 * `StorageIndex` layouts.
 *
 * Values are accessed through the accessor. Indices are always read-only so
 * that a mutable view cannot break the sorted-row invariant.
 *
\verbatim
 auto A_view = A.const_view();
 boba::loop<space, 1>(A_view.rows(), [=] __boba_host_device__(index_t i)
 {
   for (auto k = A_view.row_begin(i); k < A_view.row_end(i); ++k)
   {
     y_view(i) += A_view.value(k) * x_view(A_view.column_index(k));
   }
 });
\endverbatim
 *
 * \tparam _accessor Accessor for the values; `DefaultAccessor<T const>` for read-only views.
 */
template <typename _accessor>
struct CSRView
{
  using accessor = _accessor;
  using data_t = typename accessor::data_t;
  using value_t = std::remove_const_t<data_t>;
  using data_reference = typename accessor::data_reference;
  using data_pointer = typename accessor::data_pointer;
  using index_pointer = std::int64_t const*;
  using values_view_type = TensorView<accessor, 1>;

  CSRView() = default;

  /**
   * \brief Constructs a view over existing CSR storage.
   * \param rows Number of matrix rows.
   * \param cols Number of matrix columns.
   * \param nnz Number of stored entries.
   * \param values Length-`nnz` value array.
   * \param column_indices Length-`nnz` column index array.
   * \param row_offsets Length-`rows + 1` row offset array.
   */
  constexpr CSRView(index_t rows,
                    index_t cols,
                    index_t nnz,
                    data_pointer values,
                    index_pointer column_indices,
                    index_pointer row_offsets) noexcept
      : m_rows(rows),
        m_cols(cols),
        m_nnz(nnz),
        m_values(values),
        m_column_indices(column_indices),
        m_row_offsets(row_offsets)
  {
  }

  [[nodiscard]]
  __boba_host_device__ constexpr index_t rows() const noexcept
  {
    return m_rows;
  }

  [[nodiscard]]
  __boba_host_device__ constexpr index_t cols() const noexcept
  {
    return m_cols;
  }

  [[nodiscard]]
  __boba_host_device__ constexpr index_t nnz() const noexcept
  {
    return m_nnz;
  }

  /**
   * \brief Returns the first stored-entry offset of row `i`.
   */
  [[nodiscard]]
  __boba_host_device__ index_t row_begin(index_t i) const
  {
    boba_assert_lt(i, m_rows, "CSR row out of bounds");
    return static_cast<index_t>(m_row_offsets[i]);
  }

  /**
   * \brief Returns one past the last stored-entry offset of row `i`.
   */
  [[nodiscard]]
  __boba_host_device__ index_t row_end(index_t i) const
  {
    boba_assert_lt(i, m_rows, "CSR row out of bounds");
    return static_cast<index_t>(m_row_offsets[i + 1]);
  }

  /**
   * \brief Returns the number of stored entries in row `i`.
   *
   * Row offsets are nondecreasing, so the difference cannot wrap.
   */
  [[nodiscard]]
  __boba_host_device__ index_t row_nnz(index_t i) const
  {
    const index_t begin = row_begin(i);
    const index_t end = row_end(i);
    boba_assert_ge(end, begin, "CSR row offsets must be nondecreasing");
    return end - begin;
  }

  /**
   * \brief Returns the column of stored entry `k`.
   */
  [[nodiscard]]
  __boba_host_device__ index_t column_index(index_t k) const
  {
    boba_assert_lt(k, m_nnz, "CSR entry out of bounds");
    return static_cast<index_t>(m_column_indices[k]);
  }

  /**
   * \brief Returns stored entry `k` through the accessor.
   */
  [[nodiscard]]
  __boba_host_device__ data_reference value(index_t k) const
  {
    boba_assert_lt(k, m_nnz, "CSR entry out of bounds");
    return accessor{}.access(m_values, k);
  }

  /**
   * \brief Returns the stored values as a length-`nnz` dense view.
   */
  [[nodiscard]]
  values_view_type values_view() const noexcept
  {
    return {m_values, {m_nnz}};
  }

  /**
   * \brief Raw value pointer for sparse-library handoff.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_pointer values_data() const noexcept
  {
    return m_values;
  }

  /**
   * \brief Raw column index pointer for sparse-library handoff.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_pointer column_indices_data() const noexcept
  {
    return m_column_indices;
  }

  /**
   * \brief Raw row offset pointer for sparse-library handoff.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_pointer row_offsets_data() const noexcept
  {
    return m_row_offsets;
  }

private:
  index_t m_rows = 0;
  index_t m_cols = 0;
  index_t m_nnz = 0;
  data_pointer m_values = nullptr;
  index_pointer m_column_indices = nullptr;
  index_pointer m_row_offsets = nullptr;
};

} // namespace boba

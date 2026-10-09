// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_common.hpp"

namespace boba
{

/**
 * \brief Compressed sparse row matrix.
 *
 * The logical matrix has shape `(m_rows, m_cols)`. `m_values` and
 * `m_column_indices` have length `nnz()`, while `m_row_offsets` has length
 * `m_rows + 1`. Entries in row `i` occupy the half-open range
 * `[m_row_offsets(i), m_row_offsets(i + 1))` in the two length-`nnz()` arrays.
 * Row offsets are nondecreasing, begin at zero, and end at the number of
 * stored entries. Column indices are unique within a row but need not be
 * sorted.
 */
template <typename data_t = double>
struct CSRMatrix
{
  using data_type = data_t;
  using value_container = Vector<host_space, data_t>;
  using index_container = Vector<host_space, index_t>;
  index_t m_rows = 0, m_cols = 0;
  value_container m_values;
  index_container m_column_indices, m_row_offsets;

  /**
   * \brief Constructs an empty CSR matrix.
   */
  CSRMatrix() = default;

  /**
   * \brief Constructs a matrix with the requested dimensions.
   */
  CSRMatrix(index_t m, index_t n)
      : m_rows(m),
        m_cols(n),
        m_row_offsets({m + 1})
  {
    m_row_offsets.fill_with_zeros();
  }

  /**
   * \brief Returns the number of matrix rows.
   */
  index_t rows() const noexcept
  {
    return m_rows;
  }

  /**
   * \brief Returns the number of matrix columns.
   */
  index_t cols() const noexcept
  {
    return m_cols;
  }

  /**
   * \brief Returns the logical matrix shape.
   */
  Array<index_t, 2> shape() const noexcept
  {
    return {m_rows, m_cols};
  }

  /**
   * \brief Returns the number of stored entries or blocks.
   */
  index_t nnz() const noexcept
  {
    return m_values.size();
  }

  /**
   * \brief Returns the dense scalar size divided by the CSR storage size.
   *
   * The storage size counts values, column indices, and row offsets. The
   * result is truncated to two decimal places, matching tensor
   * compression-rate semantics.
   */
  [[nodiscard]]
  float compression_rate() const noexcept
  {
    double compressed_size = static_cast<double>(m_values.size()) + static_cast<double>(m_column_indices.size()) + static_cast<double>(m_row_offsets.size());
    double full_size = static_cast<double>(m_rows) * static_cast<double>(m_cols);
    return static_cast<float>(std::floor(full_size / compressed_size * 100.0) / 100.0);
  }

  /**
   * \brief Returns a stored value or zero when the coordinate is absent.
   */
  data_t get_element(index_t i, index_t j) const
  {
    if (i >= m_rows || j >= m_cols)
    {
      return data_t{};
    }
    auto ov = m_row_offsets.const_view();
    auto cv = m_column_indices.const_view();
    auto vv = m_values.const_view();
    for (index_t k = ov(i); k < ov(i + 1); ++k)
    {
      if (cv(k) == j)
      {
        return vv(k);
      }
    }
    return data_t{};
  }

  /**
   * \brief Computes a dense-vector product using the native sparse storage.
   */
  void matvec(Vector<host_space, data_t> const& x, Vector<host_space, data_t>& y, data_t alpha = 1, data_t beta = 0) const
  {
    boba_always_assert_equal(x.size(), m_cols, "CSR matvec input has the wrong size");
    boba_always_assert_equal(y.size(), m_rows, "CSR matvec output has the wrong size");
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto ov = m_row_offsets.const_view();
    auto cv = m_column_indices.const_view();
    auto vv = m_values.const_view();
    for (index_t i = 0; i < m_rows; ++i)
    {
      ::boba::loop<host_space, 1>(ov(i), ov(i + 1), [&](index_t k)
      {
        yv(i) += alpha * vv(k) * xv(cv(k));
      });
    }
  }

  /**
   * \brief Replaces one scalar value, inserting storage when necessary.
   */
  void set_element(index_t i, index_t j, data_t value)
  {
    boba_always_assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols, "Sparse coordinate out of bounds");
    auto ov = m_row_offsets.view();
    auto cv = m_column_indices.view();
    auto vv = m_values.view();

    // Search only this row's half-open storage segment for the column.
    for (index_t k = ov(i); k < ov(i + 1); ++k)
    {
      if (cv(k) == j)
      {
        vv(k) = value;
        return;
      }
    }
    if (!(abs(value) > 0))
    {
      // A zero absent from the row does not need a structural entry.
      return;
    }
    index_t at = ov(i + 1);
    // Insert at the end of the row, then shift later rows right by one slot.
    m_values.resize(nnz() + 1);
    m_column_indices.resize(nnz());
    // The resized arrays may have moved, so refresh their views before shifting.
    auto values_after_resize = m_values.view();
    auto columns_after_resize = m_column_indices.view();
    // Shift entries backward to leave the insertion position available.
    for (index_t k = nnz() - 1; k > at; --k)
    {
      values_after_resize(k) = values_after_resize(k - 1);
      columns_after_resize(k) = columns_after_resize(k - 1);
    }
    values_after_resize(at) = value;
    columns_after_resize(at) = j;
    // Every row beginning at i + 1 now starts one position later.
    ::boba::loop<host_space, 1>(i + 1, m_rows + 1, [&](index_t r)
    {
      ov(r)++;
    });
  }

  /**
   * \brief Adds a scalar value at one coordinate.
   */
  void add_element(index_t i, index_t j, data_t value)
  {
    set_element(i, j, get_element(i, j) + value);
  }

  /**
   * \brief Sets all stored values to zero without changing structure.
   */
  void zero_values()
  {
    m_values.fill_with_zeros();
  }

  /**
   * \brief Scales all stored values.
   */
  void scale(data_t value)
  {
    m_values *= value;
  }
};

/**
 * \brief Imports a dense matrix into CSR storage using an absolute tolerance.
 */
template <typename data_t>
CSRMatrix<data_t> from_dense_csr(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  CSRMatrix<data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    for (index_t j = 0; j < dense.cols(); ++j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        out.set_element(i, j, dv({i, j}));
      }
    }
  }
  return out;
}

/**
 * \brief Exports CSR storage to a dense host matrix.
 */
template <typename data_t>
Matrix<host_space, data_t> to_dense(CSRMatrix<data_t> const& sparse)
{
  Matrix<host_space, data_t> out({sparse.rows(), sparse.cols()});
  out.fill_with_zeros();
  auto ov = out.view();
  auto offsets = sparse.m_row_offsets.const_view();
  auto columns = sparse.m_column_indices.const_view();
  auto values = sparse.m_values.const_view();
  for (index_t i = 0; i < sparse.rows(); ++i)
  {
    ::boba::loop<host_space, 1>(offsets(i), offsets(i + 1), [&](index_t k)
    {
      ov({i, columns(k)}) = values(k);
    });
  }
  return out;
}

} // namespace boba

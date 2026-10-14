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
  index_t m_rows = 0;
  index_t m_cols = 0;
  value_container m_values;
  index_container m_column_indices, m_row_offsets;

  /**
   * \brief Constructs an empty CSR matrix.
   */
  CSRMatrix() = default;

  /**
   * \brief Constructs a matrix with the requested dimensions.
   *
   * \param m Number of rows.
   * \param n Number of columns.
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
   *
   * \return Logical row count.
   */
  index_t rows() const noexcept
  {
    return m_rows;
  }

  /**
   * \brief Returns the number of matrix columns.
   *
   * \return Logical column count.
   */
  index_t cols() const noexcept
  {
    return m_cols;
  }

  /**
   * \brief Returns the logical matrix shape.
   *
   * \return `{rows(), cols()}`.
   */
  Array<index_t, 2> shape() const noexcept
  {
    return {m_rows, m_cols};
  }

  /**
   * \brief Returns the number of explicitly stored scalar entries.
   *
   * \return Number of entries represented by the value/column arrays.
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
   *
   * \return Dense scalar count divided by sparse storage count.
   */
  [[nodiscard]]
  float compression_rate() const noexcept
  {
    double compressed_size = static_cast<double>(m_values.size()) + static_cast<double>(m_column_indices.size()) + static_cast<double>(m_row_offsets.size());
    double full_size = static_cast<double>(m_rows) * static_cast<double>(m_cols);
    return static_cast<float>(std::floor(full_size / compressed_size * 100.0) / 100.0);
  }

  /**
   * \brief Scans one CSR row for a scalar entry.
   *
   * \param i Row index.
   * \param j Column index.
   * \return The stored value at `(i, j)`, or zero when it is absent.
   * \warning An out-of-bounds `i` or `j` value produces an error in debug mode.
   */
  data_t get_element_naive(index_t i, index_t j) const
  {
    boba_assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols, "Sparse coordinate out of bounds");
    auto offsets = m_row_offsets.const_view();
    auto columns = m_column_indices.const_view();
    auto values = m_values.const_view();
    for (index_t k = offsets(i); k < offsets(i + 1); ++k)
    {
      if (columns(k) == j)
      {
        return values(k);
      }
    }
    return data_t{};
  }

  /**
   * \brief Computes a dense-vector product using the native sparse storage.
   *
   * Computes `y = alpha * A * x + beta * y`.
   *
   * \param x Dense input vector with length `cols()`.
   * \param y Dense output vector with length `rows()`.
   * \param alpha Matrix-vector product coefficient.
   * \param beta Existing-output coefficient.
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

};

/**
 * \brief Imports a dense matrix into CSR storage using an absolute tolerance.
 *
 * \param dense Dense matrix to import.
 * \param tolerance Entries with magnitude no greater than this value are omitted.
 * \return CSR storage containing the retained entries.
 */
template <typename data_t>
CSRMatrix<data_t> from_dense_csr(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  CSRMatrix<data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  auto offsets = out.m_row_offsets.view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    for (index_t j = 0; j < dense.cols(); ++j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        ++offsets(i + 1);
      }
    }
  }

  for (index_t i = 0; i < dense.rows(); ++i)
  {
    offsets(i + 1) += offsets(i);
  }
  out.m_column_indices.resize({offsets(dense.rows())});
  out.m_values.resize({offsets(dense.rows())});

  Vector<host_space, index_t> next({dense.rows()});
  auto next_view = next.view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    next_view(i) = offsets(i);
  }
  auto columns = out.m_column_indices.view();
  auto values = out.m_values.view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    for (index_t j = 0; j < dense.cols(); ++j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        const index_t sparse_index = next_view(i)++;
        columns(sparse_index) = j;
        values(sparse_index) = dv({i, j});
      }
    }
  }
  return out;
}

/**
 * \brief Exports CSR storage to a dense host matrix.
 *
 * \param sparse CSR matrix to export.
 * \return Dense matrix with missing coordinates initialized to zero.
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

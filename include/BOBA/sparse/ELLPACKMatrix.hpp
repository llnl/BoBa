// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_common.hpp"

namespace boba
{

/**
 * \brief ELLPACK sparse matrix with a common row-slot width.
 *
 * The logical matrix has shape `(m_rows, m_cols)`. `m_values` and
 * `m_column_indices` each have shape `(m_rows, m_width)`, so every row owns
 * `m_width` slots. A nonnegative column index represents an entry and its
 * scalar value is `m_values(i, k)`; `-1` marks an unused slot whose value is
 * ignored. Column indices are unique within a row but need not be sorted.
 * Inserting into a full row increases the common width for every row.
 */
template <typename data_t = double>
struct ELLPACKMatrix
{
  using data_type = data_t;
  index_t m_rows = 0, m_cols = 0, m_width = 0;
  Matrix<host_space, data_t> m_values;
  Matrix<host_space, std::int64_t> m_column_indices;

  /**
   * \brief Constructs an empty ELLPACK matrix.
   */
  ELLPACKMatrix() = default;

  /**
   * \brief Constructs a matrix with the requested dimensions.
   */
  ELLPACKMatrix(index_t m, index_t n, index_t w = 0)
      : m_rows(m),
        m_cols(n),
        m_width(w),
        m_values({m, w}),
        m_column_indices({m, w})
  {
    for (index_t i = 0; i < m * w; ++i)
    {
      m_column_indices.data()[i] = -1;
    }
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
    return 0;
  }

  /**
   * \brief Returns the dense scalar size divided by the ELLPACK storage size.
   *
   * The storage size counts the value and column-index slot for every row.
   * The result is truncated to two decimal places, matching tensor
   * compression-rate semantics.
   */
  [[nodiscard]]
  float compression_rate() const noexcept
  {
    if (m_values.size() == 0 && m_column_indices.size() == 0)
    {
      return 0.0F;
    }
    double compressed_size = static_cast<double>(m_values.size()) + static_cast<double>(m_column_indices.size());
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
      return {};
    }
    auto c = m_column_indices.const_view();
    auto v = m_values.const_view();
    for (index_t k = 0; k < m_width; ++k)
    {
      if (c({i, k}) == static_cast<std::int64_t>(j))
      {
        return v({i, k});
      }
    }
    return {};
  }

  /**
   * \brief Computes a dense-vector product using the native sparse storage.
   */
  void matvec(Vector<host_space, data_t> const& x, Vector<host_space, data_t>& y, data_t alpha = 1, data_t beta = 0) const
  {
    boba_always_assert_equal(x.size(), m_cols, "ELLPACK matvec input has the wrong size");
    boba_always_assert_equal(y.size(), m_rows, "ELLPACK matvec output has the wrong size");
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto c = m_column_indices.const_view();
    auto v = m_values.const_view();
    for (index_t i = 0; i < m_rows; ++i)
    {
      for (index_t k = 0; k < m_width; ++k)
      {
        if (c({i, k}) >= 0)
        {
          yv(i) += alpha * v({i, k}) * xv(static_cast<index_t>(c({i, k})));
        }
      }
    }
  }

  /**
   * \brief Replaces one scalar value, inserting storage when necessary.
   */
  void set_element(index_t i, index_t j, data_t value)
  {
    boba_always_assert(i < m_rows && j < m_cols, "Sparse coordinate out of bounds");
    auto columns = m_column_indices.view();
    auto values = m_values.view();

    // First update an existing slot so the row width remains unchanged.
    for (index_t k = 0; k < m_width; ++k)
    {
      if (columns({i, k}) == j)
      {
        values({i, k}) = value;
        return;
      }
    }
    if (!(abs(value) > 0))
    {
      // An absent zero does not need a sentinel slot.
      return;
    }
    // Use the first sentinel slot already allocated to this row when possible.
    for (index_t k = 0; k < m_width; ++k)
    {
      if (columns({i, k}) < 0)
      {
        columns({i, k}) = j;
        values({i, k}) = value;
        return;
      }
    }
    index_t old = m_width;
    // ELLPACK has one common width, so a full row grows every row equally.
    m_values.resize({m_rows, m_width + 1});
    m_column_indices.resize({m_rows, m_width + 1});
    // Initialize the new column as unused before assigning this row's entry.
    auto columns_after_resize = m_column_indices.view();
    auto values_after_resize = m_values.view();
    for (index_t r = 0; r < m_rows; ++r)
    {
      columns_after_resize({r, old}) = -1;
    }
    m_width = old + 1;
    columns_after_resize({i, old}) = j;
    values_after_resize({i, old}) = value;
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
 * \brief Imports a dense matrix into ELLPACK storage using an absolute tolerance.
 */
template <typename data_t>
ELLPACKMatrix<data_t> from_dense_ellpack(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  index_t w = 0;
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    index_t c = 0;
    for (index_t j = 0; j < dense.cols(); ++j)
    {
      c += sparse_detail::keep(dv({i, j}), tolerance);
    }
    w = std::max(w, c);
  }
  ELLPACKMatrix<data_t> out(dense.rows(), dense.cols(), w);
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    for (index_t j = 0, k = 0; j < dense.cols(); ++j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        out.m_column_indices.view()({i, k}) = static_cast<std::int64_t>(j);
        out.m_values.view()({i, k++}) = dv({i, j});
      }
    }
  }
  return out;
}

/**
 * \brief Exports ELLPACK storage to a dense host matrix.
 */
template <typename data_t>
Matrix<host_space, data_t> to_dense(ELLPACKMatrix<data_t> const& sparse)
{
  Matrix<host_space, data_t> out({sparse.rows(), sparse.cols()});
  out.fill_with_zeros();
  auto ov = out.view();
  auto columns = sparse.m_column_indices.const_view();
  auto values = sparse.m_values.const_view();
  for (index_t i = 0; i < sparse.rows(); ++i)
  {
    for (index_t k = 0; k < sparse.m_width; ++k)
    {
      if (columns({i, k}) >= 0)
      {
        ov({i, static_cast<index_t>(columns({i, k}))}) = values({i, k});
      }
    }
  }
  return out;
}

} // namespace boba

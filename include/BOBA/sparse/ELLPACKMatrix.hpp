// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/SparseMatrixCommon.hpp"

namespace boba
{

/**
 * \brief ELLPACK sparse matrix with a common row-slot width.
 * Unused column slots contain the signed sentinel `-1`.
 */
template <typename data_t = double>
struct ELLPACKMatrix
{
  using data_type = data_t;
  using const_view_type = SparseMatrixConstView<ELLPACKMatrix>;
  index_t m_rows = 0, m_cols = 0, m_width = 0;
  sparse_detail::host_matrix<data_t> m_values;
  sparse_detail::host_matrix<std::int64_t> m_column_indices;

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
   * \brief Returns a non-owning read-only view.
   */
  const_view_type as_const_view() const noexcept
  {
    return {this};
  }

  /**
   * \brief Returns a stored value or zero when the coordinate is absent.
   */
  data_t get_element(index_t i, index_t j) const
  {
    if (i >= m_rows || j >= m_cols)
      return {};
    auto c = m_column_indices.const_view();
    auto v = m_values.const_view();
    for (index_t k = 0; k < m_width; ++k)
    {
      if (c({i, k}) == static_cast<std::int64_t>(j))
        return v({i, k});
    }
    return {};
  }

  /**
   * \brief Computes a dense-vector product using the native sparse storage.
   */
  void matvec(sparse_detail::host_vector<data_t> const& x, sparse_detail::host_vector<data_t>& y, data_t alpha = 1, data_t beta = 0) const
  {
    sparse_detail::check_vector_sizes(m_rows, m_cols, x, y);
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
          yv(i) += alpha * v({i, k}) * xv(static_cast<index_t>(c({i, k})));
      }
    }
  }

  /**
   * \brief Replaces one scalar value, inserting storage when necessary.
   */
  void set_element(index_t i, index_t j, data_t value)
  {
    boba_always_assert(i < m_rows && j < m_cols, "Sparse coordinate out of bounds");
    for (index_t k = 0; k < m_width; ++k)
    {
      if (m_column_indices.view()({i, k}) == j)
      {
        m_values.view()({i, k}) = value;
        return;
      }
    }
    if (!(abs(value) > 0))
      return;
    for (index_t k = 0; k < m_width; ++k)
    {
      if (m_column_indices.view()({i, k}) < 0)
      {
        m_column_indices.view()({i, k}) = j;
        m_values.view()({i, k}) = value;
        return;
      }
    }
    index_t old = m_width;
    m_values.resize({m_rows, m_width + 1});
    m_column_indices.resize({m_rows, m_width + 1});
    for (index_t r = 0; r < m_rows; ++r)
    {
      m_column_indices.view()({r, old}) = -1;
    }
    m_width = old + 1;
    m_column_indices.view()({i, old}) = j;
    m_values.view()({i, old}) = value;
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

} // namespace boba

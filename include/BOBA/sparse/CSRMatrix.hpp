// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/SparseMatrixCommon.hpp"

namespace boba
{

/**
 * \brief Compressed sparse row matrix.
 */
template <typename data_t = double>
struct CSRMatrix
{
  using data_type = data_t;
  using value_container = sparse_detail::host_vector<data_t>;
  using index_container = sparse_detail::host_vector<index_t>;
  using const_view_type = SparseMatrixConstView<CSRMatrix>;
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
      return data_t{};
    auto ov = m_row_offsets.const_view();
    auto cv = m_column_indices.const_view();
    auto vv = m_values.const_view();
    for (index_t k = ov(i); k < ov(i + 1); ++k)
    {
      if (cv(k) == j)
        return vv(k);
    }
    return data_t{};
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
    auto ov = m_row_offsets.const_view();
    auto cv = m_column_indices.const_view();
    auto vv = m_values.const_view();
    for (index_t i = 0; i < m_rows; ++i)
    {
      for (index_t k = ov(i); k < ov(i + 1); ++k)
      {
        yv(i) += alpha * vv(k) * xv(cv(k));
      }
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
    for (index_t k = ov(i); k < ov(i + 1); ++k)
    {
      if (cv(k) == j)
      {
        vv(k) = value;
        return;
      }
    }
    if (!(abs(value) > 0))
      return;
    index_t at = ov(i + 1);
    m_values.resize(nnz() + 1);
    m_column_indices.resize(nnz());
    for (index_t k = nnz() - 1; k > at; --k)
    {
      m_values.view()(k) = m_values.view()(k - 1);
      m_column_indices.view()(k) = m_column_indices.view()(k - 1);
    }
    m_values.view()(at) = value;
    m_column_indices.view()(at) = j;
    for (index_t r = i + 1; r <= m_rows; ++r)
    {
      m_row_offsets.view()(r)++;
    }
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

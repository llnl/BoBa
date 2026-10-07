// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/SparseMatrixCommon.hpp"

namespace boba
{

/**
 * \brief Block coordinate sparse matrix with dense rectangular blocks.
 *
 * The scalar matrix shape is
 * `(m_block_rows * m_block_grid_rows, m_block_cols * m_block_grid_cols)`.
 * `m_values` has shape `(m_block_rows, m_block_cols, nnz())` and
 * `m_indices` has shape `(nnz(), 2)`. Block `k` has grid coordinates
 * `(m_indices(k, 0), m_indices(k, 1))`; for local coordinates `(u, v)`, its
 * definition is `A(p * m_block_rows + u, q * m_block_cols + v) =
 * m_values(u, v, k)`. Block coordinates are unique but need not be sorted; a
 * missing block represents an all-zero block. Only full, rectangular blocks
 * are supported.
 */
template <typename data_t = double>
struct BCOOMatrix
{
  using data_type = data_t;
  using const_view_type = SparseMatrixConstView<BCOOMatrix>;
  index_t m_block_rows = 0, m_block_cols = 0, m_block_grid_rows = 0, m_block_grid_cols = 0;
  sparse_detail::host_tensor3<data_t> m_values;
  sparse_detail::host_matrix<index_t> m_indices;

  /**
   * \brief Constructs an empty BCOO matrix.
   */
  BCOOMatrix() = default;

  /**
   * \brief Constructs a matrix with the requested dimensions.
   */
  BCOOMatrix(index_t br, index_t bc, index_t gr, index_t gc)
      : m_block_rows(br),
        m_block_cols(bc),
        m_block_grid_rows(gr),
        m_block_grid_cols(gc),
        m_values({br, bc, 0}),
        m_indices({0, 2})
  {
  }

  /**
   * \brief Returns the number of matrix rows.
   */
  index_t rows() const noexcept
  {
    return m_block_rows * m_block_grid_rows;
  }

  /**
   * \brief Returns the number of matrix columns.
   */
  index_t cols() const noexcept
  {
    return m_block_cols * m_block_grid_cols;
  }

  /**
   * \brief Returns the logical matrix shape.
   */
  Array<index_t, 2> shape() const noexcept
  {
    return {rows(), cols()};
  }

  /**
   * \brief Returns the number of stored entries or blocks.
   */
  index_t nnz() const noexcept
  {
    return m_indices.sizes(0);
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
    if (i >= rows() || j >= cols())
    {
      return {};
    }
    auto row_index = Multiindexer<2>::multiindex({m_block_rows, m_block_grid_rows}, i);
    auto col_index = Multiindexer<2>::multiindex({m_block_cols, m_block_grid_cols}, j);
    index_t u = row_index[0], p = row_index[1];
    index_t v = col_index[0], q = col_index[1];
    auto iv = m_indices.const_view();
    auto vv = m_values.const_view();
    for (index_t k = 0; k < nnz(); ++k)
    {
      if (iv({k, 0}) == p && iv({k, 1}) == q)
      {
        return vv({u, v, k});
      }
    }
    return {};
  }

  /**
   * \brief Computes a dense-vector product using the native sparse storage.
   */
  void matvec(sparse_detail::host_vector<data_t> const& x, sparse_detail::host_vector<data_t>& y, data_t alpha = 1, data_t beta = 0) const
  {
    sparse_detail::check_vector_sizes(rows(), cols(), x, y);
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto iv = m_indices.const_view();
    auto vv = m_values.const_view();
    for (index_t k = 0; k < nnz(); ++k)
    {
      index_t p = iv({k, 0}), q = iv({k, 1});
      for (index_t u = 0; u < m_block_rows; ++u)
      {
        for (index_t v = 0; v < m_block_cols; ++v)
        {
          yv(p * m_block_rows + u) += alpha * vv({u, v, k}) * xv(q * m_block_cols + v);
        }
      }
    }
  }

  /**
   * \brief Replaces one scalar value, inserting storage when necessary.
   */
  void set_element(index_t i, index_t j, data_t value)
  {
    boba_always_assert(i < rows() && j < cols(), "Sparse coordinate out of bounds");
    auto row_index = Multiindexer<2>::multiindex({m_block_rows, m_block_grid_rows}, i);
    auto col_index = Multiindexer<2>::multiindex({m_block_cols, m_block_grid_cols}, j);
    index_t u = row_index[0], p = row_index[1];
    index_t v = col_index[0], q = col_index[1];
    for (index_t k = 0; k < nnz(); ++k)
    {
      if (m_indices.view()({k, 0}) == p && m_indices.view()({k, 1}) == q)
      {
        m_values.view()({u, v, k}) = value;
        return;
      }
    }
    if (!(abs(value) > 0))
    {
      return;
    }
    index_t old = nnz();
    m_indices.resize({old + 1, 2});
    m_values.resize({m_block_rows, m_block_cols, old + 1});
    for (index_t a = 0; a < m_block_rows; ++a)
    {
      for (index_t b = 0; b < m_block_cols; ++b)
      {
        m_values.view()({a, b, old}) = data_t{};
      }
    }
    m_indices.view()({old, 0}) = p;
    m_indices.view()({old, 1}) = q;
    m_values.view()({u, v, old}) = value;
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

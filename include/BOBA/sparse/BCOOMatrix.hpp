// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_common.hpp"

namespace boba
{

/**
 * \brief Block coordinate sparse matrix with dense rectangular blocks.
 * Blocks are assumed all the same size and stored as dense objects in a containing COO format.
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
  index_t m_block_rows = 0, m_block_cols = 0, m_block_grid_rows = 0, m_block_grid_cols = 0;
  Tensor<3, host_space, data_t> m_values;
  Matrix<host_space, index_t> m_indices;

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
   * \brief Returns the dense scalar size divided by the BCOO storage size.
   *
   * The storage size counts all dense block values and two block-coordinate
   * indices per block. The result is truncated to two decimal places,
   * matching tensor compression-rate semantics.
   */
  [[nodiscard]]
  float compression_rate() const noexcept
  {
    if (nnz() == 0)
    {
      return 0.0F;
    }
    double compressed_size = static_cast<double>(m_block_rows) * static_cast<double>(m_block_cols) * static_cast<double>(nnz()) + 2.0 * static_cast<double>(nnz());
    double full_size = static_cast<double>(rows()) * static_cast<double>(cols());
    return static_cast<float>(std::floor(full_size / compressed_size * 100.0) / 100.0);
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
  void matvec(Vector<host_space, data_t> const& x, Vector<host_space, data_t>& y, data_t alpha = 1, data_t beta = 0) const
  {
    boba_always_assert_equal(x.size(), cols(), "BCOO matvec input has the wrong size");
    boba_always_assert_equal(y.size(), rows(), "BCOO matvec output has the wrong size");
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
          auto out_id = p * m_block_rows + u;
          auto in_id = q * m_block_cols + v;
          yv(out_id) += alpha * vv({u, v, k}) * xv(in_id);
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
    // Split the scalar coordinate into a local block coordinate and a block-grid
    // coordinate; the latter identifies the stored block and the former its slot.
    auto row_index = Multiindexer<2>::multiindex({m_block_rows, m_block_grid_rows}, i);
    auto col_index = Multiindexer<2>::multiindex({m_block_cols, m_block_grid_cols}, j);
    index_t u = row_index[0], p = row_index[1];
    index_t v = col_index[0], q = col_index[1];
    auto indices = m_indices.view();
    auto values = m_values.view();

    // Update an existing block in place when its grid coordinate is present.
    for (index_t k = 0; k < nnz(); ++k)
    {
      if (indices({k, 0}) == p && indices({k, 1}) == q)
      {
        values({u, v, k}) = value;
        return;
      }
    }
    if (!(abs(value) > 0))
    {
      // Do not allocate a full block for an absent zero scalar.
      return;
    }
    index_t old = nnz();
    // A new BCOO entry always allocates a complete dense block and one pair of
    // block-grid indices, even when most block values are zero.
    m_indices.resize({old + 1, 2});
    m_values.resize({m_block_rows, m_block_cols, old + 1});
    // Resizing may invalidate earlier views; initialize the new block through
    // fresh views before writing its requested local value.
    auto indices_after_resize = m_indices.view();
    auto values_after_resize = m_values.view();
    for (index_t a = 0; a < m_block_rows; ++a)
    {
      for (index_t b = 0; b < m_block_cols; ++b)
      {
        values_after_resize({a, b, old}) = data_t{};
      }
    }
    indices_after_resize({old, 0}) = p;
    indices_after_resize({old, 1}) = q;
    values_after_resize({u, v, old}) = value;
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
 * \brief Imports a dense matrix into BCOO storage using explicit block dimensions.
 */
template <typename data_t>
BCOOMatrix<data_t> from_dense_bcoo(Matrix<host_space, data_t> const& dense, index_t br, index_t bc, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  boba_always_assert(br > 0 && bc > 0, "BCOO block dimensions must be positive");
  boba_always_assert(dense.rows() % br == 0 && dense.cols() % bc == 0, "BCOO dimensions must be divisible by block dimensions");
  BCOOMatrix<data_t> out(br, bc, dense.rows() / br, dense.cols() / bc);
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
 * \brief Exports BCOO storage to a dense host matrix.
 */
template <typename data_t>
Matrix<host_space, data_t> to_dense(BCOOMatrix<data_t> const& sparse)
{
  Matrix<host_space, data_t> out({sparse.rows(), sparse.cols()});
  out.fill_with_zeros();
  auto ov = out.view();
  auto indices = sparse.m_indices.const_view();
  auto values = sparse.m_values.const_view();

  auto block_rows = sparse.m_block_rows;
  auto block_cols = sparse.m_block_cols;

  for (index_t k = 0; k < sparse.nnz(); ++k)
  {
    index_t p = indices({k, 0});
    index_t q = indices({k, 1});
    for (index_t u = 0; u < block_rows; ++u)
    {
      for (index_t v = 0; v < block_cols; ++v)
      {
        auto row_id = p * block_rows + u;
        auto col_id = q * block_cols + v;
        ov({row_id, col_id}) = values({u, v, k});
      }
    }
  }
  return out;
}

} // namespace boba

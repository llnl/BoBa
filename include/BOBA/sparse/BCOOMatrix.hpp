// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_common.hpp"

namespace boba
{

/**
 * \brief Block coordinate sparse matrix with dense rectangular blocks.
 * Blocks are all the same size for a given matrix and are stored as dense
 * objects in a containing COO format.
 *
 * The scalar matrix shape is
 * `(m_rows_per_block * m_block_grid_rows, m_cols_per_block * m_block_grid_cols)`.
 * `m_values` has shape `(m_rows_per_block, m_cols_per_block, nonzero_blocks())`
 * and `m_indices` has shape `(nonzero_blocks(), 2)`. Block `k` has grid coordinates
 * `(m_indices(k, 0), m_indices(k, 1))`; for local coordinates `(u, v)`, its
 * definition is `A(p * m_rows_per_block + u, q * m_cols_per_block + v) =
 * m_values(u, v, k)`. Block coordinates are unique but need not be sorted; a
 * missing block represents an all-zero block. Only full, rectangular blocks
 * are supported.
 */
template <typename data_t = double>
struct BCOOMatrix
{
  using data_type = data_t;
  index_t m_rows_per_block = 0;
  index_t m_cols_per_block = 0;
  index_t m_block_grid_rows = 0;
  index_t m_block_grid_cols = 0;
  Tensor<3, host_space, data_t> m_values;
  Matrix<host_space, index_t> m_indices;

  /**
   * \brief Constructs an empty BCOO matrix.
   */
  BCOOMatrix() = default;

  /**
   * \brief Constructs an empty BCOO matrix with fixed block dimensions.
   *
   * \param br Number of rows in every block.
   * \param bc Number of columns in every block.
   * \param gr Number of block rows in the grid.
   * \param gc Number of block columns in the grid.
   */
  BCOOMatrix(index_t br, index_t bc, index_t gr, index_t gc)
      : m_rows_per_block(br),
        m_cols_per_block(bc),
        m_block_grid_rows(gr),
        m_block_grid_cols(gc),
        m_values({br, bc, 0}),
        m_indices({0, 2})
  {
  }

  /**
   * \brief Returns the number of matrix rows.
   *
   * \return Logical row count.
   */
  index_t rows() const noexcept
  {
    return m_rows_per_block * m_block_grid_rows;
  }

  /**
   * \brief Returns the number of matrix columns.
   *
   * \return Logical column count.
   */
  index_t cols() const noexcept
  {
    return m_cols_per_block * m_block_grid_cols;
  }

  /**
   * \brief Returns the logical matrix shape.
   *
   * \return `{rows(), cols()}`.
   */
  Array<index_t, 2> shape() const noexcept
  {
    return {rows(), cols()};
  }

  /**
   * \brief Returns the number of explicitly stored blocks.
   *
   * \return Number of block-coordinate entries in `m_indices`.
   */
  index_t nonzero_blocks() const noexcept
  {
    return m_indices.sizes(0);
  }

  /**
   * \brief Returns the size of the dense block-value tensor.
   *
   * \return Total number of scalar slots in `m_values`, including zero-valued
   *         slots inside stored blocks.
   */
  index_t nnz() const noexcept
  {
    return m_values.size();
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
    if (m_values.size() == 0 && m_indices.size() == 0)
    {
      return 0.0F;
    }
    double compressed_size = static_cast<double>(m_values.size()) + static_cast<double>(m_indices.size());
    double full_size = static_cast<double>(rows()) * static_cast<double>(cols());
    return static_cast<float>(std::floor(full_size / compressed_size * 100.0) / 100.0);
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
  void matvec(Vector<host_space, data_t> const& x,
              Vector<host_space, data_t>& y,
              data_t alpha = 1,
              data_t beta = 0) const
  {
    boba_always_assert_equal(x.size(), cols(), "BCOO matvec input has the wrong size");
    boba_always_assert_equal(y.size(), rows(), "BCOO matvec output has the wrong size");
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto indices = m_indices.const_view();
    auto values = m_values.const_view();
    ::boba::loop<host_space, 3>({nonzero_blocks(), m_rows_per_block, m_cols_per_block}, [&](Array<index_t, 3> loop_indices)
    {
      auto [k, u, v] = loop_indices;
      const index_t p = indices({k, 0});
      const index_t q = indices({k, 1});
      const index_t out_id = p * m_rows_per_block + u;
      const index_t in_id = q * m_cols_per_block + v;
      yv(out_id) += alpha * values({u, v, k}) * xv(in_id);
    });
  }
};

/**
 * \brief Imports a dense matrix into BCOO storage using fixed block dimensions.
 *
 * A block is stored when at least one value in it passes the tolerance test;
 * values within that block that do not pass the test are initialized to zero.
 *
 * \param dense Dense matrix to import.
 * \param br Number of rows in every block.
 * \param bc Number of columns in every block.
 * \param tolerance Entries with magnitude no greater than this value are omitted.
 * \return BCOO storage containing the retained blocks.
 */
template <typename data_t>
BCOOMatrix<data_t> from_dense_bcoo(Matrix<host_space, data_t> const& dense,
                                   index_t br,
                                   index_t bc,
                                   data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  boba_always_assert(br > 0 && bc > 0, "BCOO block dimensions must be positive");
  boba_always_assert(dense.rows() % br == 0 && dense.cols() % bc == 0,
                     "BCOO dimensions must be divisible by block dimensions");

  BCOOMatrix<data_t> out(br, bc, dense.rows() / br, dense.cols() / bc);
  auto dense_view = dense.const_view();
  index_t number_blocks = 0;
  for (index_t p = 0; p < out.m_block_grid_rows; ++p)
  {
    for (index_t q = 0; q < out.m_block_grid_cols; ++q)
    {
      bool keep_block = false;
      for (index_t u = 0; u < br && !keep_block; ++u)
      {
        for (index_t v = 0; v < bc; ++v)
        {
          if (sparse_detail::keep(dense_view({p * br + u, q * bc + v}), tolerance))
          {
            keep_block = true;
            break;
          }
        }
      }
      if (keep_block)
      {
        ++number_blocks;
      }
    }
  }

  out.m_indices.resize({number_blocks, 2});
  out.m_values.resize({br, bc, number_blocks});
  out.m_values.fill_with_zeros();
  auto indices = out.m_indices.view();
  auto values = out.m_values.view();
  index_t block_index = 0;
  for (index_t p = 0; p < out.m_block_grid_rows; ++p)
  {
    for (index_t q = 0; q < out.m_block_grid_cols; ++q)
    {
      bool keep_block = false;
      for (index_t u = 0; u < br && !keep_block; ++u)
      {
        for (index_t v = 0; v < bc; ++v)
        {
          if (sparse_detail::keep(dense_view({p * br + u, q * bc + v}), tolerance))
          {
            keep_block = true;
            break;
          }
        }
      }
      if (keep_block)
      {
        indices({block_index, 0}) = p;
        indices({block_index, 1}) = q;
        for (index_t u = 0; u < br; ++u)
        {
          for (index_t v = 0; v < bc; ++v)
          {
            if (sparse_detail::keep(dense_view({p * br + u, q * bc + v}), tolerance))
            {
              values({u, v, block_index}) = dense_view({p * br + u, q * bc + v});
            }
          }
        }
        ++block_index;
      }
    }
  }
  return out;
}

/**
 * \brief Exports BCOO storage to a dense host matrix.
 *
 * \param sparse BCOO matrix to export.
 * \return Dense matrix with missing blocks initialized to zero.
 */
template <typename data_t>
Matrix<host_space, data_t> to_dense(BCOOMatrix<data_t> const& sparse)
{
  Matrix<host_space, data_t> out({sparse.rows(), sparse.cols()});
  out.fill_with_zeros();
  auto output = out.view();
  auto indices = sparse.m_indices.const_view();
  auto values = sparse.m_values.const_view();
  ::boba::loop<host_space, 3>({sparse.nonzero_blocks(), sparse.m_rows_per_block, sparse.m_cols_per_block},
                               [&](Array<index_t, 3> loop_indices)
  {
    auto [k, u, v] = loop_indices;
    const index_t p = indices({k, 0});
    const index_t q = indices({k, 1});
    output({p * sparse.m_rows_per_block + u, q * sparse.m_cols_per_block + v}) = values({u, v, k});
  });
  return out;
}

} // namespace boba

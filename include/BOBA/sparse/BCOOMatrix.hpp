// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_common.hpp"

namespace boba
{

/**
 * \brief Block coordinate sparse matrix with dense rectangular blocks.
 * Blocks use row and column partitions stored as BoBa vectors. Block `(p, q)`
 * has shape `(m_block_rows(p), m_block_cols(q))` and is stored densely in a
 * flattened value vector.
 *
 * The scalar matrix shape is
 * `m_indices` has shape `(nnz(), 2)`. Block `k` has grid coordinates
 * `(m_indices(k, 0), m_indices(k, 1))`; its values begin at
 * `m_value_offsets(k)` and have size `m_block_rows(p) * m_block_cols(q)`.
 * Block coordinates are unique but need not be sorted; a missing block
 * represents an all-zero block.
 */
template <typename data_t = double>
struct BCOOMatrix
{
  using data_type = data_t;
  Vector<host_space, index_t> m_block_rows;
  Vector<host_space, index_t> m_block_cols;
  Vector<host_space, data_t> m_values;
  Vector<host_space, index_t> m_value_offsets;
  Matrix<host_space, index_t> m_indices;

  /**
   * \brief Constructs an empty BCOO matrix.
   */
  BCOOMatrix() = default;

  /**
   * \brief Constructs a matrix with the requested dimensions.
   */
  BCOOMatrix(index_t br, index_t bc, index_t gr, index_t gc)
      : m_block_rows(std::vector<index_t>(gr, br)),
        m_block_cols(std::vector<index_t>(gc, bc)),
        m_value_offsets({1}),
        m_indices({0, 2})
  {
    m_value_offsets.fill_with_zeros();
  }

  /**
   * rief Constructs a matrix from variable row and column block sizes.
   */
  BCOOMatrix(Vector<host_space, index_t> block_rows, Vector<host_space, index_t> block_cols)
      : m_block_rows(std::move(block_rows)),
        m_block_cols(std::move(block_cols)),
        m_value_offsets({1}),
        m_indices({0, 2})
  {
    boba_always_assert(m_block_rows.size() > 0 && m_block_cols.size() > 0, "BCOO block partitions must be nonempty");
    auto rows = m_block_rows.const_view();
    auto cols = m_block_cols.const_view();
    for (index_t p = 0; p < m_block_rows.size(); ++p)
    {
      boba_always_assert(rows(p) > 0, "BCOO row block sizes must be positive");
    }
    for (index_t q = 0; q < m_block_cols.size(); ++q)
    {
      boba_always_assert(cols(q) > 0, "BCOO column block sizes must be positive");
    }
    m_value_offsets.fill_with_zeros();
  }

  /**
   * \brief Returns the number of matrix rows.
   */
  index_t rows() const noexcept
  {
    index_t result = 0;
    auto block_rows = m_block_rows.const_view();
    for (index_t p = 0; p < m_block_rows.size(); ++p)
    {
      result += block_rows(p);
    }
    return result;
  }

  /**
   * \brief Returns the number of matrix columns.
   */
  index_t cols() const noexcept
  {
    index_t result = 0;
    auto block_cols = m_block_cols.const_view();
    for (index_t q = 0; q < m_block_cols.size(); ++q)
    {
      result += block_cols(q);
    }
    return result;
  }

  /**
   * \brief Returns the logical matrix shape.
   */
  Array<index_t, 2> shape() const noexcept
  {
    return {rows(), cols()};
  }

  index_t row_block(index_t i, index_t& local_i) const
  {
    auto block_rows = m_block_rows.const_view();
    for (index_t p = 0; p < m_block_rows.size(); ++p)
    {
      if (i < block_rows(p))
      {
        local_i = i;
        return p;
      }
      i -= block_rows(p);
    }
    local_i = 0;
    return m_block_rows.size();
  }

  index_t col_block(index_t j, index_t& local_j) const
  {
    auto block_cols = m_block_cols.const_view();
    for (index_t q = 0; q < m_block_cols.size(); ++q)
    {
      if (j < block_cols(q))
      {
        local_j = j;
        return q;
      }
      j -= block_cols(q);
    }
    local_j = 0;
    return m_block_cols.size();
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
    if (m_values.size() == 0 && m_indices.size() == 0)
    {
      return 0.0F;
    }
    double compressed_size = static_cast<double>(m_values.size()) + static_cast<double>(m_indices.size());
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

    index_t u = 0, v = 0;
    index_t p = row_block(i, u);
    index_t q = col_block(j, v);
    auto iv = m_indices.const_view();
    auto vv = m_values.const_view();
    for (index_t k = 0; k < nnz(); ++k)
    {
      if (iv({k, 0}) == p && iv({k, 1}) == q)
      {
        auto row_size = m_block_rows.const_view()(p);
        auto col_size = m_block_cols.const_view()(q);
        return vv(m_value_offsets.const_view()(k) + u * col_size + v);
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
    auto block_rows = m_block_rows.const_view();
    auto block_cols = m_block_cols.const_view();
    auto value_offsets = m_value_offsets.const_view();
    for (index_t k = 0; k < nnz(); ++k)
    {
      index_t p = iv({k, 0}), q = iv({k, 1});
      index_t row_offset = 0;
      for (index_t prior = 0; prior < p; ++prior)
      {
        row_offset += block_rows(prior);
      }
      index_t col_offset = 0;
      for (index_t prior = 0; prior < q; ++prior)
      {
        col_offset += block_cols(prior);
      }
      for (index_t u = 0; u < block_rows(p); ++u)
      {
        for (index_t v = 0; v < block_cols(q); ++v)
        {
          auto out_id = row_offset + u;
          auto in_id = col_offset + v;
          yv(out_id) += alpha * vv(value_offsets(k) + u * block_cols(q) + v) * xv(in_id);
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
    index_t u = 0, v = 0;
    index_t p = row_block(i, u);
    index_t q = col_block(j, v);
    auto indices = m_indices.view();
    auto values = m_values.view();
    auto value_offsets = m_value_offsets.view();

    // Update an existing block in place when its grid coordinate is present.
    for (index_t k = 0; k < nnz(); ++k)
    {
      if (indices({k, 0}) == p && indices({k, 1}) == q)
      {
        values(value_offsets(k) + u * m_block_cols.const_view()(q) + v) = value;
        return;
      }
    }
    if (!(abs(value) > 0))
    {
      // Do not allocate a full block for an absent zero scalar.
      return;
    }
    index_t old = nnz();
    // A new BCOO entry allocates the dense block selected by its row and
    // column block partitions.
    const index_t block_size = m_block_rows.const_view()(p) * m_block_cols.const_view()(q);
    const index_t value_offset = m_values.size();
    m_indices.resize({old + 1, 2});
    m_value_offsets.resize({old + 2});
    m_values.resize({value_offset + block_size});
    // Resizing may invalidate earlier views; initialize the new block through
    // fresh views before writing its requested local value.
    auto indices_after_resize = m_indices.view();
    auto values_after_resize = m_values.view();
    auto offsets_after_resize = m_value_offsets.view();
    for (index_t offset = 0; offset < block_size; ++offset)
    {
      values_after_resize(value_offset + offset) = data_t{};
    }
    indices_after_resize({old, 0}) = p;
    indices_after_resize({old, 1}) = q;
    offsets_after_resize(old) = value_offset;
    offsets_after_resize(old + 1) = value_offset + block_size;
    values_after_resize(value_offset + u * m_block_cols.const_view()(q) + v) = value;
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
 * \brief Imports a dense matrix into BCOO storage using variable block sizes.
 */
template <typename data_t>
BCOOMatrix<data_t> from_dense_bcoo(
  Matrix<host_space, data_t> const& dense,
  Vector<host_space, index_t> block_rows,
  Vector<host_space, index_t> block_cols,
  data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  BCOOMatrix<data_t> out(std::move(block_rows), std::move(block_cols));
  boba_always_assert_equal(dense.rows(), out.rows(), "BCOO row partitions do not match the dense matrix");
  boba_always_assert_equal(dense.cols(), out.cols(), "BCOO column partitions do not match the dense matrix");
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
  auto block_rows = sparse.m_block_rows.const_view();
  auto block_cols = sparse.m_block_cols.const_view();
  auto value_offsets = sparse.m_value_offsets.const_view();

  for (index_t k = 0; k < sparse.nnz(); ++k)
  {
    index_t p = indices({k, 0});
    index_t q = indices({k, 1});
    index_t row_offset = 0;
    for (index_t prior = 0; prior < p; ++prior)
    {
      row_offset += block_rows(prior);
    }
    index_t col_offset = 0;
    for (index_t prior = 0; prior < q; ++prior)
    {
      col_offset += block_cols(prior);
    }
    for (index_t u = 0; u < block_rows(p); ++u)
    {
      for (index_t v = 0; v < block_cols(q); ++v)
      {
        ov({row_offset + u, col_offset + v}) = values(value_offsets(k) + u * block_cols(q) + v);
      }
    }
  }
  return out;
}

} // namespace boba

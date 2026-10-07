// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_common.hpp"

namespace boba
{

/**
 * \brief Sparse matrix stored by diagonal offsets `column - row`.
 *
 * The logical matrix has shape `(m_rows, m_cols)`. `m_offsets` has length
 * `ndiag` and `m_values` has shape `(m_rows, ndiag)`. Diagonal `k` has offset
 * `m_offsets(k)` and represents
 * `A(i, i + m_offsets(k)) = m_values(i, k)` whenever the column is in bounds.
 * Rows outside the logical shape are padding and do not represent matrix
 * entries. Offsets are unique, and adding a new diagonal allocates one value
 * slot for every logical row.
 */
template <typename data_t = double>
struct DIAMatrix
{
  using data_type = data_t;
  index_t m_rows = 0, m_cols = 0;
  Matrix<host_space, data_t> m_values;
  Vector<host_space, std::int64_t> m_offsets;

  /**
   * \brief Constructs an empty DIA matrix.
   */
  DIAMatrix() = default;

  /**
   * \brief Constructs a matrix with the requested dimensions.
   */
  DIAMatrix(index_t m, index_t n)
      : m_rows(m),
        m_cols(n),
        m_values({m, 0})
  {
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
   * \brief Returns the dense scalar size divided by the DIA storage size.
   *
   * The storage size counts every value slot and every diagonal offset. The
   * result is truncated to two decimal places, matching tensor
   * compression-rate semantics.
   */
  [[nodiscard]]
  float compression_rate() const noexcept
  {
    if (m_offsets.size() == 0)
    {
      return 0.0F;
    }
    double diagonal_count = static_cast<double>(m_offsets.size());
    double compressed_size = diagonal_count * static_cast<double>(m_rows) + diagonal_count;
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
    auto o = m_offsets.const_view();
    auto v = m_values.const_view();
    for (index_t k = 0; k < m_offsets.size(); ++k)
    {
      if (static_cast<std::int64_t>(j) - static_cast<std::int64_t>(i) == o(k))
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
    boba_always_assert_equal(x.size(), m_cols, "DIA matvec input has the wrong size");
    boba_always_assert_equal(y.size(), m_rows, "DIA matvec output has the wrong size");
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto offs = m_offsets.const_view();
    auto v = m_values.const_view();
    for (index_t i = 0; i < m_rows; ++i)
    {
      for (index_t k = 0; k < m_offsets.size(); ++k)
      {
        auto j = static_cast<std::int64_t>(i) + offs(k);
        if (j >= 0 && j < static_cast<std::int64_t>(m_cols))
        {
          yv(i) += alpha * v({i, k}) * xv(static_cast<index_t>(j));
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
    // DIA identifies a complete diagonal by its signed column-minus-row offset.
    std::int64_t d = static_cast<std::int64_t>(j) - static_cast<std::int64_t>(i);
    auto offsets = m_offsets.view();
    auto values = m_values.view();
    // If the diagonal already exists, only its value slot needs updating.
    for (index_t k = 0; k < m_offsets.size(); ++k)
    {
      if (offsets(k) == d)
      {
        values({i, k}) = value;
        return;
      }
    }
    if (!(abs(value) > 0))
    {
      // Do not allocate a whole diagonal for an absent zero.
      return;
    }
    index_t old = m_offsets.size();
    // A new diagonal allocates one value slot for every matrix row, including
    // padding positions whose columns lie outside the rectangular matrix.
    m_offsets.resize(old + 1);
    m_values.resize({m_rows, old + 1});
    // Resizing may invalidate earlier views, so acquire views of the new column.
    auto offsets_after_resize = m_offsets.view();
    auto values_after_resize = m_values.view();
    offsets_after_resize(old) = d;
    for (index_t r = 0; r < m_rows; ++r)
    {
      values_after_resize({r, old}) = data_t{};
    }
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
 * \brief Imports a dense matrix into DIA storage using an absolute tolerance.
 */
template <typename data_t>
DIAMatrix<data_t> from_dense_dia(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  DIAMatrix<data_t> out(dense.rows(), dense.cols());
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
 * \brief Exports DIA storage to a dense host matrix.
 */
template <typename data_t>
Matrix<host_space, data_t> to_dense(DIAMatrix<data_t> const& sparse)
{
  Matrix<host_space, data_t> out({sparse.rows(), sparse.cols()});
  out.fill_with_zeros();
  auto ov = out.view();
  auto offsets = sparse.m_offsets.const_view();
  auto values = sparse.m_values.const_view();
  for (index_t i = 0; i < sparse.rows(); ++i)
  {
    for (index_t k = 0; k < sparse.m_offsets.size(); ++k)
    {
      auto j = static_cast<std::int64_t>(i) + offsets(k);
      if (j >= 0 && j < static_cast<std::int64_t>(sparse.cols()))
      {
        ov({i, static_cast<index_t>(j)}) = values({i, k});
      }
    }
  }
  return out;
}

} // namespace boba

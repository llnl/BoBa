// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_common.hpp"

namespace boba
{

/**
 * \brief Coordinate-list sparse matrix with unsorted zero-based coordinates.
 *
 * The logical matrix has shape `(m_rows, m_cols)`. `m_values` has length
 * `nnz()` and `m_indices` has shape `(nnz(), 2)`. Entry `k` is represented by
 * `m_indices(k, 0)` for its row, `m_indices(k, 1)` for its column, and
 * `m_values(k)` for its scalar value. Coordinates are unique by contract but
 * need not be sorted. Missing coordinates represent zero.
 */
template <typename data_t = double>
struct COOMatrix
{
  using data_type = data_t;
  using index_container = Matrix<host_space, index_t>;
  using value_container = Vector<host_space, data_t>;

  index_t m_rows = 0;
  index_t m_cols = 0;
  value_container m_values;
  index_container m_indices;

  /**
   * \brief Constructs an empty COO matrix.
   */
  COOMatrix() = default;

  /**
   * \brief Constructs a matrix with the requested dimensions.
   *
   * \param m Number of rows.
   * \param n Number of columns.
   */
  COOMatrix(index_t m, index_t n)
      : m_rows(m),
        m_cols(n),
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
   * \return Number of entries represented by the coordinate/value arrays.
   */
  index_t nnz() const noexcept
  {
    return m_values.size();
  }

  /**
   * \brief Returns the dense scalar size divided by the COO storage size.
   *
   * The storage size counts one value and two coordinate indices per entry.
   * The result is truncated to two decimal places, matching tensor
   * compression-rate semantics.
   *
   * \return Dense scalar count divided by sparse storage count.
   */
  [[nodiscard]]
  float compression_rate() const noexcept
  {
    if (m_values.size() == 0 && m_indices.size() == 0)
    {
      return 0.0F;
    }
    double compressed_size = static_cast<double>(m_values.size()) + static_cast<double>(m_indices.size());
    double full_size = static_cast<double>(m_rows) * static_cast<double>(m_cols);
    return static_cast<float>(std::floor(full_size / compressed_size * 100.0) / 100.0);
  }

  /**
   * \brief Returns a stored value or zero when the coordinate is absent.
   *
   * \param i Row index.
   * \param j Column index.
   * \return The stored value at `(i, j)`, or zero when it is absent or out of bounds.
   */
  data_t get_element(index_t i, index_t j) const
  {
    if (i >= m_rows || j >= m_cols)
    {
      return data_t{};
    }
    auto iv = m_indices.const_view();
    auto vv = m_values.const_view();
    for (index_t k = 0; k < nnz(); ++k)
    {
      if (iv({k, 0}) == i && iv({k, 1}) == j)
      {
        return vv(k);
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
  void matvec(Vector<host_space, data_t> const& x,
              Vector<host_space, data_t>& y,
              data_t alpha = 1,
              data_t beta = 0) const
  {
    boba_always_assert_equal(x.size(), m_cols, "COO matvec input has the wrong size");
    boba_always_assert_equal(y.size(), m_rows, "COO matvec output has the wrong size");
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto iv = m_indices.const_view();
    auto vv = m_values.const_view();
    ::boba::loop<host_space, 1>(0, nnz(), [&](index_t k)
    {
      yv(iv({k, 0})) += alpha * vv(k) * xv(iv({k, 1}));
    });
  }

};

/**
 * \brief Imports a dense matrix into COO storage using an absolute tolerance.
 *
 * \param dense Dense matrix to import.
 * \param tolerance Entries with magnitude no greater than this value are omitted.
 * \return COO storage containing the retained entries.
 */
template <typename data_t>
COOMatrix<data_t> from_dense_coo(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  COOMatrix<data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  index_t number_nonzeros = 0;
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    for (index_t j = 0; j < dense.cols(); ++j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        ++number_nonzeros;
      }
    }
  }

  out.m_values.resize({number_nonzeros});
  out.m_indices.resize({number_nonzeros, 2});
  auto values = out.m_values.view();
  auto indices = out.m_indices.view();
  index_t sparse_index = 0;
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    for (index_t j = 0; j < dense.cols(); ++j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        indices({sparse_index, 0}) = i;
        indices({sparse_index, 1}) = j;
        values(sparse_index) = dv({i, j});
        ++sparse_index;
      }
    }
  }
  return out;
}

/**
 * \brief Exports COO storage to a dense host matrix.
 *
 * \param sparse COO matrix to export.
 * \return Dense matrix with missing coordinates initialized to zero.
 */
template <typename data_t>
Matrix<host_space, data_t> to_dense(COOMatrix<data_t> const& sparse)
{
  Matrix<host_space, data_t> out({sparse.rows(), sparse.cols()});
  out.fill_with_zeros();
  auto ov = out.view();
  auto iv = sparse.m_indices.const_view();
  auto vv = sparse.m_values.const_view();
  ::boba::loop<host_space, 1>(0, sparse.nnz(), [&](index_t k)
  {
    ov({iv({k, 0}), iv({k, 1})}) = vv(k);
  });
  return out;
}

} // namespace boba

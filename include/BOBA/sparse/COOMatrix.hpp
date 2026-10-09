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
   */
  COOMatrix(index_t m, index_t n)
      : m_rows(m),
        m_cols(n),
        m_indices({0, 2})
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
    return m_values.size();
  }

  /**
   * \brief Returns the dense scalar size divided by the COO storage size.
   *
   * The storage size counts one value and two coordinate indices per entry.
   * The result is truncated to two decimal places, matching tensor
   * compression-rate semantics.
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

  /**
   * \brief Replaces one scalar value, inserting storage when necessary.
   */
  void set_element(index_t i, index_t j, data_t value)
  {
    boba_always_assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols, "Sparse coordinate out of bounds");
    auto indices = m_indices.view();
    auto values = m_values.view();
    // COO has no row directory, so locate an existing coordinate by scanning
    // the coordinate list.
    for (index_t k = 0; k < nnz(); ++k)
    {
      if (indices({k, 0}) == i && indices({k, 1}) == j)
      {
        values(k) = value;
        return;
      }
    }
    if (!(abs(value) > 0))
    {
      // Do not create an explicit zero entry in the coordinate list.
      return;
    }
    index_t old = nnz();
    // Appending preserves the coordinate/value pairing; no sorting is needed.
    m_values.resize(old + 1);
    m_indices.resize({old + 1, 2});
    // Resizing may invalidate earlier views, so acquire views of the new storage.
    auto values_after_resize = m_values.view();
    auto indices_after_resize = m_indices.view();
    values_after_resize(old) = value;
    indices_after_resize({old, 0}) = i;
    indices_after_resize({old, 1}) = j;
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
 * \brief Imports a dense matrix into COO storage using an absolute tolerance.
 */
template <typename data_t>
COOMatrix<data_t> from_dense_coo(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  COOMatrix<data_t> out(dense.rows(), dense.cols());
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
 * \brief Exports COO storage to a dense host matrix.
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

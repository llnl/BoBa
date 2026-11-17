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
 *
 * \tparam space Execution space used by the slot and column-index storage.
 * \tparam _data_t Scalar value type.
 */
template <execution_space space, typename _data_t>
struct ELLPACKMatrix
{
  using data_t = _data_t;
  index_t m_rows = 0;
  index_t m_cols = 0;
  index_t m_width = 0;
  Matrix<space, data_t> m_values;
  Matrix<space, std::int64_t> m_column_indices;

  /**
   * \brief Constructs an empty ELLPACK matrix.
   */
  ELLPACKMatrix() = default;

  /**
   * \brief Returns the execution space used by this matrix's storage.
   *
   * \return Matrix storage execution space.
   */
  static constexpr execution_space get_space() noexcept
  {
    return space;
  }

  /**
   * \brief Copies an ELLPACK matrix from another execution space.
   *
   * \param rhs ELLPACK matrix to copy.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  ELLPACKMatrix(ELLPACKMatrix<rhs_space, data_t> const& rhs)
      : m_rows(rhs.m_rows),
        m_cols(rhs.m_cols),
        m_width(rhs.m_width),
        m_values(rhs.m_values),
        m_column_indices(rhs.m_column_indices)
  {
  }

  /**
   * \brief Moves an ELLPACK matrix from another execution space.
   *
   * The source is left empty after its data has been copied to the target
   * execution space, following Tensor's cross-space move semantics.
   *
   * \param rhs ELLPACK matrix to move.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  ELLPACKMatrix(ELLPACKMatrix<rhs_space, data_t>&& rhs)
      : m_rows(rhs.m_rows),
        m_cols(rhs.m_cols),
        m_width(rhs.m_width),
        m_values(rhs.m_values),
        m_column_indices(rhs.m_column_indices)
  {
    rhs.m_rows = 0;
    rhs.m_cols = 0;
    rhs.m_width = 0;
    rhs.m_values.resize({0, 0});
    rhs.m_column_indices.resize({0, 0});
  }

  /**
   * \brief Copies an ELLPACK matrix from another execution space.
   *
   * \param rhs ELLPACK matrix to copy.
   * \return This matrix after assignment.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  ELLPACKMatrix& operator=(ELLPACKMatrix<rhs_space, data_t> const& rhs)
  {
    m_rows = rhs.m_rows;
    m_cols = rhs.m_cols;
    m_width = rhs.m_width;
    m_values = rhs.m_values;
    m_column_indices = rhs.m_column_indices;
    return *this;
  }

  /**
   * \brief Moves an ELLPACK matrix from another execution space.
   *
   * \param rhs ELLPACK matrix to move.
   * \return This matrix after assignment.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  ELLPACKMatrix& operator=(ELLPACKMatrix<rhs_space, data_t>&& rhs)
  {
    *this = static_cast<ELLPACKMatrix<rhs_space, data_t> const&>(rhs);
    rhs.m_rows = 0;
    rhs.m_cols = 0;
    rhs.m_width = 0;
    rhs.m_values.resize({0, 0});
    rhs.m_column_indices.resize({0, 0});
    return *this;
  }

  /**
   * \brief Constructs a matrix with the requested dimensions.
   *
   * \param m Number of rows.
   * \param n Number of columns.
   * \param w Number of slots allocated per row.
   */
  ELLPACKMatrix(index_t m, index_t n, index_t w = 0)
      : m_rows(m),
        m_cols(n),
        m_width(w),
        m_values({m, w}),
        m_column_indices({m, w})
  {
    m_values.fill_with_zeros();
    m_column_indices.fill_with(-1);
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
   * \brief Returns the number of allocated value slots.
   *
   * ELLPACK stores a fixed-width row slot array, so this includes slots whose
   * column index is the unused sentinel `-1`.
   *
   * \return Number of entries in `m_values`.
   */
  index_t nnz() const noexcept
  {
    return m_values.size();
  }

  /**
   * \brief Returns the dense scalar size divided by the ELLPACK storage size.
   *
   * The storage size counts the value and column-index slot for every row.
   * The result is truncated to two decimal places, matching tensor
   * compression-rate semantics.
   *
   * \return Dense scalar count divided by sparse storage count.
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
   * \brief Scans the ELLPACK slots for a scalar entry.
   *
   * \param i Row index.
   * \param j Column index.
   * \return The stored value at `(i, j)`, or zero when it is absent.
   * \warning An out-of-bounds `i` or `j` value produces an error in debug mode.
   */
  data_t get_element_naive(index_t i, index_t j) const
    requires(space == host_space)
  {
    boba_assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols, "Sparse coordinate out of bounds");
    auto columns = m_column_indices.const_view();
    auto values = m_values.const_view();
    for (index_t k = 0; k < m_width; ++k)
    {
      if (columns({i, k}) == static_cast<std::int64_t>(j))
      {
        return values({i, k});
      }
    }
    return {};
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
   * \warning This operation is currently available only for host-space matrices.
   */
  void matvec(Vector<space, data_t> const& x, Vector<space, data_t>& y, data_t alpha = 1, data_t beta = 0) const
    requires(space == host_space)
  {
    boba_always_assert_equal(x.size(), m_cols, "ELLPACK matvec input has the wrong size");
    boba_always_assert_equal(y.size(), m_rows, "ELLPACK matvec output has the wrong size");
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto c = m_column_indices.const_view();
    auto v = m_values.const_view();
    ::boba::loop<host_space, 2>({m_rows, m_width}, [&](Array<index_t, 2> indices)
    {
      auto [i, k] = indices;
      if (c({i, k}) >= 0)
      {
        yv(i) += alpha * v({i, k}) * xv(static_cast<index_t>(c({i, k})));
      }
    });
  }

};

/**
 * \brief Imports a dense matrix into ELLPACK storage using an absolute tolerance.
 *
 * \param dense Dense matrix to import.
 * \param tolerance Entries with magnitude no greater than this value are omitted.
 * \return ELLPACK storage containing the retained entries.
 * \tparam space Must be `host_space`; dense conversion is host-only.
 */
template <execution_space space, typename data_t>
  requires(space == host_space)
ELLPACKMatrix<space, data_t> from_dense_ellpack(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  index_t w = 0;
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    index_t c = 0;
    ::boba::loop<host_space, 1>(0, dense.cols(), [&](index_t j)
    {
      c += sparse_detail::keep(dv({i, j}), tolerance);
    });
    w = std::max(w, c);
  }
  ELLPACKMatrix<space, data_t> out(dense.rows(), dense.cols(), w);
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    index_t k = 0;
    auto columns = out.m_column_indices.view();
    auto values = out.m_values.view();
    ::boba::loop<host_space, 1>(0, dense.cols(), [&](index_t j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        columns({i, k}) = static_cast<std::int64_t>(j);
        values({i, k++}) = dv({i, j});
      }
    });
  }
  return out;
}

/**
 * \brief Exports ELLPACK storage to a dense host matrix.
 *
 * \param sparse ELLPACK matrix to export.
 * \return Dense matrix with unused slots and missing coordinates represented as zero.
 * \tparam space Must be `host_space`; dense conversion is host-only.
 */
template <execution_space space, typename data_t>
  requires(space == host_space)
Matrix<host_space, data_t> to_dense(ELLPACKMatrix<space, data_t> const& sparse)
{
  Matrix<host_space, data_t> out({sparse.rows(), sparse.cols()});
  out.fill_with_zeros();
  auto ov = out.view();
  auto columns = sparse.m_column_indices.const_view();
  auto values = sparse.m_values.const_view();
  ::boba::loop<host_space, 2>({sparse.rows(), sparse.m_width}, [&](Array<index_t, 2> indices)
  {
    auto [i, k] = indices;
    if (columns({i, k}) >= 0)
    {
      ov({i, static_cast<index_t>(columns({i, k}))}) = values({i, k});
    }
  });
  return out;
}

} // namespace boba

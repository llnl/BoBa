// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/CSRView.hpp"
#include "BOBA/sparse/sparse_op.hpp"
#include "BOBA/sparse/sparse_common.hpp"

namespace boba
{

/**
 * \brief Compressed sparse row matrix.
 *
 * The logical matrix has shape `(m_rows, m_cols)`. `m_values` and
 * `m_column_indices` have length `nnz()`, while `m_row_offsets` has length
 * `m_rows + 1`. Entries in row `i` occupy the half-open range
 * `[m_row_offsets(i), m_row_offsets(i + 1))` in the two length-`nnz()` arrays.
 * Row offsets are nondecreasing, begin at zero, and end at the number of
 * stored entries. Column indices are zero-based, unique, and sorted in
 * increasing order within each row. Callers that fill the storage directly
 * must keep rows sorted; backends rely on it and `is_sorted()` checks it in
 * debug builds. Indices are stored as `std::int64_t` so views can be handed
 * to Eigen and vendor sparse libraries without conversion.
 *
 * \tparam space Execution space used by the row, column, and value storage.
 * \tparam _data_t Scalar value type.
 */
template <execution_space space, typename _data_t>
struct CSRMatrix
{
  using data_t = _data_t;
  using value_container = Vector<space, data_t>;
  using index_container = Vector<space, std::int64_t>;
  using view_type = CSRView<::boba::DefaultAccessor<data_t>>;
  using const_view_type = CSRView<::boba::DefaultAccessor<data_t const>>;
  index_t m_rows = 0;
  index_t m_cols = 0;
  value_container m_values;
  index_container m_column_indices, m_row_offsets;

  /**
   * \brief Constructs an empty CSR matrix.
   */
  CSRMatrix() = default;

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
   * \brief Copies a CSR matrix from another execution space.
   *
   * \param rhs CSR matrix to copy.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  CSRMatrix(CSRMatrix<rhs_space, data_t> const& rhs)
      : m_rows(rhs.m_rows),
        m_cols(rhs.m_cols),
        m_values(rhs.m_values),
        m_column_indices(rhs.m_column_indices),
        m_row_offsets(rhs.m_row_offsets)
  {
  }

  /**
   * \brief Moves a CSR matrix from another execution space.
   *
   * The source is left empty after its data has been copied to the target
   * execution space, following Tensor's cross-space move semantics.
   *
   * \param rhs CSR matrix to move.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  CSRMatrix(CSRMatrix<rhs_space, data_t>&& rhs)
      : m_rows(rhs.m_rows),
        m_cols(rhs.m_cols),
        m_values(rhs.m_values),
        m_column_indices(rhs.m_column_indices),
        m_row_offsets(rhs.m_row_offsets)
  {
    rhs.m_rows = 0;
    rhs.m_cols = 0;
    rhs.m_values.resize({0});
    rhs.m_column_indices.resize({0});
    rhs.m_row_offsets.resize({0});
  }

  /**
   * \brief Copies a CSR matrix from another execution space.
   *
   * \param rhs CSR matrix to copy.
   * \return This matrix after assignment.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  CSRMatrix& operator=(CSRMatrix<rhs_space, data_t> const& rhs)
  {
    m_rows = rhs.m_rows;
    m_cols = rhs.m_cols;
    m_values = rhs.m_values;
    m_column_indices = rhs.m_column_indices;
    m_row_offsets = rhs.m_row_offsets;
    return *this;
  }

  /**
   * \brief Moves a CSR matrix from another execution space.
   *
   * \param rhs CSR matrix to move.
   * \return This matrix after assignment.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  CSRMatrix& operator=(CSRMatrix<rhs_space, data_t>&& rhs)
  {
    *this = static_cast<CSRMatrix<rhs_space, data_t> const&>(rhs);
    rhs.m_rows = 0;
    rhs.m_cols = 0;
    rhs.m_values.resize({0});
    rhs.m_column_indices.resize({0});
    rhs.m_row_offsets.resize({0});
    return *this;
  }

  /**
   * \brief Constructs a matrix with the requested dimensions.
   *
   * \param m Number of rows.
   * \param n Number of columns.
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
   * \return Number of entries represented by the value/column arrays.
   */
  index_t nnz() const noexcept
  {
    return m_values.size();
  }

  /**
   * \brief Returns the dense scalar size divided by the CSR storage size.
   *
   * The storage size counts values, column indices, and row offsets. The
   * result is truncated to two decimal places, matching tensor
   * compression-rate semantics.
   *
   * \return Dense scalar count divided by sparse storage count.
   */
  [[nodiscard]]
  float compression_rate() const noexcept
  {
    if (m_values.size() == 0 && m_column_indices.size() == 0 && m_row_offsets.size() == 0)
    {
      return 0.0F;
    }
    double compressed_size = static_cast<double>(m_values.size()) + static_cast<double>(m_column_indices.size()) + static_cast<double>(m_row_offsets.size());
    double full_size = static_cast<double>(m_rows) * static_cast<double>(m_cols);
    return static_cast<float>(std::floor(full_size / compressed_size * 100.0) / 100.0);
  }

  /**
   * \brief Scans one CSR row for a scalar entry.
   *
   * \param i Row index.
   * \param j Column index.
   * \return The stored value at `(i, j)`, or zero when it is absent.
   * \warning An out-of-bounds `i` or `j` value produces an error in debug mode.
   */
  data_t get_element_naive(index_t i, index_t j) const
    requires(space == host_space)
  {
    boba_assert(i < m_rows && j < m_cols, "Sparse coordinate out of bounds");
    auto offsets = m_row_offsets.const_view();
    auto columns = m_column_indices.const_view();
    auto values = m_values.const_view();
    for (auto k = static_cast<index_t>(offsets(i)); k < static_cast<index_t>(offsets(i + 1)); ++k)
    {
      if (columns(k) == static_cast<std::int64_t>(j))
      {
        return values(k);
      }
    }
    return data_t{};
  }

  /**
   * \brief Computes `y = alpha * op(A) * x + beta * y`.
   *
   * Forwards to `sparse_matrix_vector_product`. When `beta == 0`, `y` is
   * overwritten without being read.
   *
   * \param x Dense input vector with length `cols()` (`rows()` when transposed).
   * \param y Dense output vector with length `rows()` (`cols()` when transposed).
   * \param alpha Matrix-vector product coefficient.
   * \param beta Existing-output coefficient.
   * \param op Selects `A` or its transpose.
   */
  void matvec(Vector<space, data_t> const& x, Vector<space, data_t>& y, data_t alpha = 1, data_t beta = 0, sparse_op op = sparse_op::none) const
  {
    sparse_matrix_vector_product<space, data_t>(*this, x, y, alpha, beta, op);
  }

  /**
   * \brief Computes `C = alpha * op(A) * B + beta * C` for dense column-major `B` and `C`.
   *
   * Forwards to `sparse_matrix_matrix_product`. When `beta == 0`, `C` is
   * overwritten without being read.
   *
   * \param B Dense input matrix with `cols()` rows (`rows()` when transposed).
   * \param C Dense output matrix with `rows()` rows (`cols()` when transposed)
   *        and as many columns as `B`.
   * \param alpha Matrix-matrix product coefficient.
   * \param beta Existing-output coefficient.
   * \param op Selects `A` or its transpose.
   */
  void matmat(Matrix<space, data_t> const& B, Matrix<space, data_t>& C, data_t alpha = 1, data_t beta = 0, sparse_op op = sparse_op::none) const
  {
    sparse_matrix_matrix_product<space, data_t>(*this, B, C, alpha, beta, op);
  }

  /**
   * \brief Returns a mutable non-owning view for backend handoff.
   */
  view_type view()
  {
    return {m_rows, m_cols, nnz(), m_values.data(), m_column_indices.data(), m_row_offsets.data()};
  }

  /**
   * \brief Returns a read-only non-owning view for backend handoff.
   */
  const_view_type const_view() const
  {
    return {m_rows, m_cols, nnz(), m_values.const_data(), m_column_indices.const_data(), m_row_offsets.const_data()};
  }

  /**
   * \brief Checks that column indices are strictly increasing within each row.
   *
   * Backends assume sorted rows, and CSR storage is expected to be built
   * sorted. This check reads every stored index in `space`, so callers
   * should only invoke it inside `boba_assert`, i.e. in debug builds.
   *
   * \return `true` when every row is sorted with no duplicate columns.
   */
  [[nodiscard]]
  bool is_sorted() const
  {
    auto const this_view = const_view();
    index_t unsorted_rows = 0;
    ::boba::sum_reduce<space>(unsorted_rows, index_t(0), m_rows, [=] __boba_host_device__(index_t i, sum_reducer_operator<index_t> & local_unsorted)
    {
      bool sorted = true;
      for (auto k = this_view.row_begin(i) + 1; sorted && k < this_view.row_end(i); ++k)
      {
        sorted = this_view.column_index(k - 1) < this_view.column_index(k);
      }
      if (!sorted)
      {
        local_unsorted += index_t(1);
      }
    });
    return unsorted_rows == 0;
  }

};

/**
 * \brief Imports a dense matrix into CSR storage using an absolute tolerance.
 *
 * \param dense Dense matrix to import.
 * \param tolerance Entries with magnitude no greater than this value are omitted.
 * \return CSR storage containing the retained entries.
 * \tparam space Must be `host_space`; dense conversion is host-only.
 */
template <execution_space space, typename data_t>
  requires(space == host_space)
CSRMatrix<space, data_t> from_dense_csr(Matrix<host_space, data_t> const& dense, real_type_t<data_t> tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  CSRMatrix<space, data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  auto offsets = out.m_row_offsets.view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    for (index_t j = 0; j < dense.cols(); ++j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        ++offsets(i + 1);
      }
    }
  }

  for (index_t i = 0; i < dense.rows(); ++i)
  {
    offsets(i + 1) += offsets(i);
  }
  auto const stored = static_cast<index_t>(offsets(dense.rows()));
  out.m_column_indices.resize({stored});
  out.m_values.resize({stored});

  Vector<host_space, index_t> next({dense.rows()});
  auto next_view = next.view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    next_view(i) = static_cast<index_t>(offsets(i));
  }
  auto columns = out.m_column_indices.view();
  auto values = out.m_values.view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    for (index_t j = 0; j < dense.cols(); ++j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        const index_t sparse_index = next_view(i)++;
        columns(sparse_index) = static_cast<std::int64_t>(j);
        values(sparse_index) = dv({i, j});
      }
    }
  }
  boba_assert(out.is_sorted(), "from_dense_csr produced unsorted CSR rows");
  return out;
}

/**
 * \brief Exports CSR storage to a dense host matrix.
 *
 * \param sparse CSR matrix to export.
 * \return Dense matrix with missing coordinates initialized to zero.
 * \tparam space Must be `host_space`; dense conversion is host-only.
 */
template <execution_space space, typename data_t>
  requires(space == host_space)
Matrix<host_space, data_t> to_dense(CSRMatrix<space, data_t> const& sparse)
{
  Matrix<host_space, data_t> out({sparse.rows(), sparse.cols()});
  out.fill_with_zeros();
  auto ov = out.view();
  auto offsets = sparse.m_row_offsets.const_view();
  auto columns = sparse.m_column_indices.const_view();
  auto values = sparse.m_values.const_view();
  for (index_t i = 0; i < sparse.rows(); ++i)
  {
    ::boba::loop<host_space, 1>(static_cast<index_t>(offsets(i)), static_cast<index_t>(offsets(i + 1)), [&](index_t k)
    {
      ov({i, static_cast<index_t>(columns(k))}) = values(k);
    });
  }
  return out;
}

} // namespace boba

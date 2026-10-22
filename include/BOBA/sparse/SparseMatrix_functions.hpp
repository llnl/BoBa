// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/BCOOMatrix.hpp"
#include "BOBA/sparse/COOMatrix.hpp"
#include "BOBA/sparse/CSRMatrix.hpp"
#include "BOBA/sparse/ELLPACKMatrix.hpp"
#include "BOBA/tensors/SparseTensor.hpp"
#include "BOBA/backends/boba_cusparse.hpp"
#include "BOBA/backends/boba_eigen_sparse.hpp"
#include "BOBA/backends/boba_hipsparse.hpp"

#include <limits>

namespace boba
{

namespace detail
{

/**
 * \brief Concept that evaluates to true for scalar types accepted by cuSPARSE and hipSPARSE.
 */
template <typename data_t>
inline constexpr bool is_sparse_library_type_v =
  std::is_same_v<data_t, float> or
  std::is_same_v<data_t, double> or
  std::is_same_v<data_t, complex<float>> or
  std::is_same_v<data_t, complex<double>>;

/**
 * \brief Sets `output = beta * output` with existing Tensor operations.
 *
 * `beta == 0` zero-fills instead of scaling so that NaN or uninitialized
 * entries are overwritten, matching BLAS semantics.
 */
template <typename dense_t, typename data_t>
void scale_dense_output(dense_t& output, data_t beta)
{
  if (abs(beta) <= 0)
  {
    output.fill_with_zeros();
  }
  else
  {
    output *= beta;
  }
}

/**
 * \brief Naive `Y = alpha * A * X + beta * Y` for dense column-major rank-2 views `X` and `Y`.
 *
 * Each output entry is a row dot product. A vector is passed as a single column.
 */
template <execution_space space, typename A_view_t, typename X_view_t, typename Y_view_t, typename data_t>
void naive_csr_product(
  A_view_t const& A,
  X_view_t const& X,
  Y_view_t const& Y,
  data_t alpha,
  data_t beta)
{
  bool const overwrite = abs(beta) <= 0;
  ::boba::loop<space, 2>({A.rows(), X.sizes(1)}, [=] __boba_host_device__(Array<index_t, 2> ir)
  {
    auto const [i, r] = ir;
    data_t sum{0};
    for (auto k = A.row_begin(i); k < A.row_end(i); ++k)
    {
      sum += A.value(k) * X({A.column_index(k), r});
    }
    auto& output = Y({i, r});
    output = overwrite ? alpha * sum : alpha * sum + beta * output;
  });
}

/**
 * \brief Naive `Y += alpha * A^T * X` for dense column-major rank-2 views `X` and `Y`.
 *
 * Every stored entry of row `i` scatters into `Y` with an atomic add, so the
 * caller applies `beta` to `Y` beforehand.
 */
template <execution_space space, typename A_view_t, typename X_view_t, typename Y_view_t, typename data_t>
void naive_csr_transpose_accumulate(
  A_view_t const& A,
  X_view_t const& X,
  Y_view_t const& Y,
  data_t alpha)
{
  TensorView<::boba::atomics::Accessor<space, data_t>, 2> Y_atomic(Y.data(), Y.sizes());
  ::boba::loop<space, 2>({A.rows(), X.sizes(1)}, [=] __boba_host_device__(Array<index_t, 2> ir)
  {
    auto const [i, r] = ir;
    auto const x = alpha * X({i, r});
    for (auto k = A.row_begin(i); k < A.row_end(i); ++k)
    {
      Y_atomic({A.column_index(k), r}) += A.value(k) * x;
    }
  });
}

/**
 * \brief Builds the CSR form of the transpose of a column-major-sorted sparse matrix.
 *
 * \p unfolded stores strictly increasing keys `key = r + rows * c`, so sorting by
 * key sorts by column `c` first and row `r` second. Read as the transpose, this
 * is CSR order: CSR row `c`, CSR column `r`. No sort is needed.
 *
 * Example with a 2 x 3 matrix with keys {1, 2, 5} = {(1,0), (0,1), (1,2)}:
 *
 *   M = [0 a 0]    keys sorted by (c, r): (1,0) (0,1) (1,2)
 *       [b 0 c]    Mᵀ CSR: row_offsets {0, 1, 2, 3}, columns {1, 0, 1}
 *
 * Column indices are `key % rows`; row offset `c` is the first entry with
 * `key >= c * rows`, found by binary search.
 *
 * \param unfolded Rank-2 sparse tensor with sorted unique linear keys.
 * \return CSR matrix of shape `(unfolded.sizes(1), unfolded.sizes(0))`.
 */
template <execution_space space, typename data_t>
CSRMatrix<space, data_t> transposed_csr_from_sorted_keys(const SparseTensor<2, space, data_t>& unfolded)
{
  BOBA_CALI_MARK
  const auto sizes = unfolded.sizes();
  const index_t csr_rows = sizes[1];
  const index_t csr_cols = sizes[0];
  const index_t nnz = unfolded.nnz();
  boba_always_assert(unfolded.size() <= static_cast<index_t>(std::numeric_limits<std::int64_t>::max()),
                     "Unfolded sparse tensor is too large for int64 CSR indices");

  CSRMatrix<space, data_t> csr(csr_rows, csr_cols);
  csr.m_values = unfolded.values();
  csr.m_column_indices = Vector<space, std::int64_t>({nnz});

  const auto& keys = unfolded.linear_indices();
  const auto keys_view = keys.const_view();
  auto column_view = csr.m_column_indices.view();
  auto offsets_view = csr.m_row_offsets.view();

  ::boba::loop<space, 1>(0_z, nnz, [=] __boba_host_device__(index_t entry)
  {
    column_view(entry) = static_cast<std::int64_t>(keys_view(entry) % csr_cols);
  });

  // offsets(i) is the first entry whose key is at least i * csr_cols.
  ::boba::loop<space, 1>(0_z, csr_rows + 1, [=] __boba_host_device__(index_t row)
  {
    const index_t target = row * csr_cols;
    index_t low = 0;
    index_t high = nnz;
    while (low < high)
    {
      const index_t middle = low + (high - low) / 2;
      if (keys_view(middle) < target)
      {
        low = middle + 1;
      }
      else
      {
        high = middle;
      }
    }
    offsets_view(row) = static_cast<std::int64_t>(low);
  });

  boba_assert(csr.is_sorted(), "CSR built from sparse tensor keys must have sorted rows");
  return csr;
}

/**
 * \brief Exclusive prefix sum of \p counts in \p space. Returns the total.
 *
 * Uses CUB/hipCUB when available and a single-thread loop otherwise.
 */
template <execution_space space>
index_t exclusive_scan_with_total(Vector<space, index_t> const& counts, Vector<space, index_t>& offsets)
{
  const index_t n = counts.size();
  if (n == 0)
  {
    return 0;
  }
  const auto counts_view = counts.const_view();
  auto offsets_view = offsets.view();
  if constexpr (cub_available<space>())
  {
    exclusive_scan<space>(counts_view.data(), offsets_view.data(), n);
  }
  else
  {
    ::boba::loop<space, 1>(0_z, 1_z, [=] __boba_host_device__(index_t)
    {
      index_t running = 0;
      for (index_t i = 0; i < n; ++i)
      {
        offsets_view(i) = running;
        running += counts_view(i);
      }
    });
  }

  index_t last_offset = 0;
  index_t last_count = 0;
  ::boba::detail::memcpy<host_space, space>(&last_offset, offsets_view.data() + (n - 1), 1);
  ::boba::detail::memcpy<host_space, space>(&last_count, counts_view.data() + (n - 1), 1);
  ::boba::detail::synchronize<space>();
  return last_offset + last_count;
}

/**
 * \brief Expand-sort-compress SpGEMM `C = A * B` in any execution space.
 *
 * 1. Count the products `a_ik * b_kj` of each row of A and scan to offsets.
 * 2. Write every product with key `j + B.cols() * i`, the column-major key of Cᵀ.
 * 3. Sort and sum duplicate keys with the SparseTensor constructor
 *    (CUB/hipCUB on device, otherwise a host round-trip).
 * 4. Read the sorted keys of Cᵀ as CSR of C.
 *
 * \note Memory is O(number of products), which can greatly exceed nnz(C).
 */
template <execution_space space, typename data_t>
CSRMatrix<space, data_t> esc_csr_spgemm(CSRMatrix<space, data_t> const& A, CSRMatrix<space, data_t> const& B)
{
  BOBA_CALI_MARK
  const auto A_view = A.const_view();
  const auto B_view = B.const_view();
  const index_t rows = A.rows();
  const index_t cols = B.cols();

  Vector<space, index_t> counts({rows});
  auto counts_view = counts.view();
  ::boba::loop<space, 1>(0_z, rows, [=] __boba_host_device__(index_t i)
  {
    index_t count = 0;
    for (auto k = A_view.row_begin(i); k < A_view.row_end(i); ++k)
    {
      const auto b_row = A_view.column_index(k);
      count += B_view.row_nnz(b_row);
    }
    counts_view(i) = count;
  });

  Vector<space, index_t> offsets({rows});
  const index_t products = exclusive_scan_with_total(counts, offsets);

  Vector<space, index_t> keys({products});
  Vector<space, data_t> values({products});
  const auto offsets_view = offsets.const_view();
  auto keys_view = keys.view();
  auto values_view = values.view();
  ::boba::loop<space, 1>(0_z, rows, [=] __boba_host_device__(index_t i)
  {
    index_t out = offsets_view(i);
    for (auto k = A_view.row_begin(i); k < A_view.row_end(i); ++k)
    {
      const auto b_row = A_view.column_index(k);
      const auto a_value = A_view.value(k);
      for (auto l = B_view.row_begin(b_row); l < B_view.row_end(b_row); ++l)
      {
        keys_view(out) = B_view.column_index(l) + cols * i;
        values_view(out) = a_value * B_view.value(l);
        ++out;
      }
    }
  });

  const SparseTensor<2, space, data_t> C_transposed({cols, rows}, std::move(keys), std::move(values));
  return transposed_csr_from_sorted_keys(C_transposed);
}

} // namespace detail

/**
 * \brief Computes `y = alpha * op(A) * x + beta * y` for a CSR matrix.
 *
 * Dispatches to cuSPARSE, hipSPARSE, or Eigen when available for `space` and
 * `data_t`, and otherwise to a naive `boba::loop` kernel. When `beta == 0`,
 * `y` is overwritten without being read.
 *
 * \tparam force_naive Skip vendor libraries and use the naive kernel.
 * \param A CSR matrix with sorted rows.
 * \param x Dense input vector with length `A.cols()` (`A.rows()` when transposed).
 * \param y Dense output vector with length `A.rows()` (`A.cols()` when transposed).
 */
template <execution_space space, typename data_t, bool force_naive = false>
void sparse_matrix_vector_product(
  CSRMatrix<space, data_t> const& A,
  Vector<space, data_t> const& x,
  Vector<space, data_t>& y,
  data_t alpha = 1,
  data_t beta = 0,
  sparse_op op = sparse_op::none)
{
  BOBA_CALI_OBJECT_MARK

  auto const transposed = op == sparse_op::transpose;
  boba_always_assert_equal(x.size(), transposed ? A.rows() : A.cols(), "CSR matvec input has the wrong size");
  boba_always_assert_equal(y.size(), transposed ? A.cols() : A.rows(), "CSR matvec output has the wrong size");
  boba_assert(A.is_sorted(), "CSR matvec requires sorted column indices in each row");

  if (y.size() == 0)
  {
    return;
  }
  if (A.nnz() == 0 || x.size() == 0)
  {
    detail::scale_dense_output(y, beta);
    return;
  }

  auto const A_view = A.const_view();
  auto const x_view = x.const_view();
  auto const y_view = y.view();

  if constexpr (
    (space == ::boba::execution_space::CUDA) and
    ::boba::boba_cuda_enabled() and
    ::boba::boba_cuda_libs_enabled() and
    detail::is_sparse_library_type_v<data_t> and
    not(force_naive))
  {
    ::boba::detail::cusparse_spmv(A_view, x_view, y_view, alpha, beta, op);
    return;
  }
  else if constexpr (
    (space == ::boba::execution_space::HIP) and
    ::boba::boba_hip_enabled() and
    ::boba::boba_hip_libs_enabled() and
    detail::is_sparse_library_type_v<data_t> and
    not(force_naive))
  {
    ::boba::detail::hipsparse_spmv(A_view, x_view, y_view, alpha, beta, op);
    return;
  }
  else if constexpr (
    (space == ::boba::execution_space::CPU) and
    ::boba::is_cpu_enabled() and
    ::boba::boba_eigen_enabled() and
    not(force_naive))
  {
    ::boba::detail::eigen_csr_spmv(A_view, x_view, y_view, alpha, beta, op);
    return;
  }

  TensorView<DefaultAccessor<data_t const>, 2> x_column(x_view.data(), {x.size(), index_t(1)});
  TensorView<DefaultAccessor<data_t>, 2> y_column(y_view.data(), {y.size(), index_t(1)});
  if (transposed)
  {
    detail::scale_dense_output(y, beta);
    detail::naive_csr_transpose_accumulate<space>(A_view, x_column, y_column, alpha);
    return;
  }
  detail::naive_csr_product<space>(A_view, x_column, y_column, alpha, beta);
}

/**
 * \brief Computes `C = alpha * op(A) * B + beta * C` for a CSR matrix and dense column-major `B` and `C`.
 *
 * Dispatches to cuSPARSE, hipSPARSE, or Eigen when available for `space` and
 * `data_t`, and otherwise to a naive `boba::loop` kernel. When `beta == 0`,
 * `C` is overwritten without being read.
 *
 * \tparam force_naive Skip vendor libraries and use the naive kernel.
 * \param A CSR matrix with sorted rows.
 * \param B Dense input matrix with `A.cols()` rows (`A.rows()` when transposed).
 * \param C Dense output matrix with `A.rows()` rows (`A.cols()` when transposed)
 *        and as many columns as `B`.
 */
template <execution_space space, typename data_t, bool force_naive = false>
void sparse_matrix_matrix_product(
  CSRMatrix<space, data_t> const& A,
  Matrix<space, data_t> const& B,
  Matrix<space, data_t>& C,
  data_t alpha = 1,
  data_t beta = 0,
  sparse_op op = sparse_op::none)
{
  BOBA_CALI_OBJECT_MARK

  auto const transposed = op == sparse_op::transpose;
  boba_always_assert_equal(B.rows(), transposed ? A.rows() : A.cols(), "CSR matmat input has the wrong number of rows");
  boba_always_assert_equal(C.rows(), transposed ? A.cols() : A.rows(), "CSR matmat output has the wrong number of rows");
  boba_always_assert_equal(C.cols(), B.cols(), "CSR matmat output has the wrong number of columns");
  boba_assert(A.is_sorted(), "CSR matmat requires sorted column indices in each row");

  if (C.size() == 0)
  {
    return;
  }
  if (A.nnz() == 0 || B.size() == 0)
  {
    detail::scale_dense_output(C, beta);
    return;
  }

  auto const A_view = A.const_view();
  auto const B_view = B.const_view();
  auto const C_view = C.view();

  if constexpr (
    (space == ::boba::execution_space::CUDA) and
    ::boba::boba_cuda_enabled() and
    ::boba::boba_cuda_libs_enabled() and
    detail::is_sparse_library_type_v<data_t> and
    not(force_naive))
  {
    ::boba::detail::cusparse_spmm(A_view, B_view, C_view, alpha, beta, op);
    return;
  }
  else if constexpr (
    (space == ::boba::execution_space::HIP) and
    ::boba::boba_hip_enabled() and
    ::boba::boba_hip_libs_enabled() and
    detail::is_sparse_library_type_v<data_t> and
    not(force_naive))
  {
    ::boba::detail::hipsparse_spmm(A_view, B_view, C_view, alpha, beta, op);
    return;
  }
  else if constexpr (
    (space == ::boba::execution_space::CPU) and
    ::boba::is_cpu_enabled() and
    ::boba::boba_eigen_enabled() and
    not(force_naive))
  {
    ::boba::detail::eigen_csr_spmm(A_view, B_view, C_view, alpha, beta, op);
    return;
  }

  if (transposed)
  {
    detail::scale_dense_output(C, beta);
    detail::naive_csr_transpose_accumulate<space>(A_view, B_view, C_view, alpha);
    return;
  }
  detail::naive_csr_product<space>(A_view, B_view, C_view, alpha, beta);
}

/**
 * \brief Computes the sparse product `C = A * B` of two CSR matrices.
 *
 * Dispatches to cuSPARSE, hipSPARSE, or Eigen when available for `space` and
 * `data_t`, and otherwise to an expand-sort-compress kernel. Exact
 * cancellations may be kept as stored zeros. The result has sorted rows.
 *
 * \tparam force_naive Skip vendor libraries and use the expand-sort-compress kernel.
 * \param A Left CSR matrix with sorted rows.
 * \param B Right CSR matrix with sorted rows and `A.cols()` rows.
 * \return CSR matrix of shape `(A.rows(), B.cols())`.
 */
template <execution_space space, typename data_t, bool force_naive = false>
CSRMatrix<space, data_t> sparse_matrix_sparse_matrix_product(
  CSRMatrix<space, data_t> const& A,
  CSRMatrix<space, data_t> const& B)
{
  BOBA_CALI_OBJECT_MARK

  boba_always_assert_equal(A.cols(), B.rows(), "CSR SpGEMM inner dimensions must match");
  boba_assert(A.is_sorted(), "CSR SpGEMM requires sorted column indices in each row of A");
  boba_assert(B.is_sorted(), "CSR SpGEMM requires sorted column indices in each row of B");
  constexpr auto int64_limit = static_cast<index_t>(std::numeric_limits<std::int64_t>::max());
  boba_always_assert(B.cols() == 0 || A.rows() <= int64_limit / B.cols(),
                     "CSR SpGEMM output is too large for int64 indices");

  CSRMatrix<space, data_t> C(A.rows(), B.cols());
  if (A.nnz() == 0 || B.nnz() == 0)
  {
    return C;
  }

  auto const A_view = A.const_view();
  auto const B_view = B.const_view();

  if constexpr (
    (space == ::boba::execution_space::CUDA) and
    ::boba::boba_cuda_enabled() and
    ::boba::boba_cuda_libs_enabled() and
    detail::is_sparse_library_type_v<data_t> and
    not(force_naive))
  {
    ::boba::detail::cusparse_spgemm(A_view, B_view, C);
    return C;
  }
  else if constexpr (
    (space == ::boba::execution_space::HIP) and
    ::boba::boba_hip_enabled() and
    ::boba::boba_hip_libs_enabled() and
    detail::is_sparse_library_type_v<data_t> and
    not(force_naive))
  {
    ::boba::detail::hipsparse_spgemm(A_view, B_view, C);
    return C;
  }
  else if constexpr (
    (space == ::boba::execution_space::CPU) and
    ::boba::is_cpu_enabled() and
    ::boba::boba_eigen_enabled() and
    not(force_naive))
  {
    ::boba::detail::eigen_csr_spgemm(A_view, B_view, C);
    return C;
  }
  else
  {
    return detail::esc_csr_spgemm(A, B);
  }
}

/**
 * \brief Converts a rank-2 sparse tensor to a CSR matrix.
 *
 * Sparse tensor keys are column-major, so one transpose (a sort) puts them in
 * CSR order. Contractions avoid that sort by unfolding into the transposed
 * layout and calling detail::transposed_csr_from_sorted_keys() directly.
 *
 * \param matrix Rank-2 sparse tensor.
 * \return CSR matrix with the same shape and entries.
 */
template <execution_space space, typename data_t>
CSRMatrix<space, data_t> convert_to_CSRMatrix(SparseTensor<2, space, data_t> const& matrix)
{
  auto transposed = matrix;
  permute(transposed, Array<index_t, 2>{1, 0});
  return detail::transposed_csr_from_sorted_keys(transposed);
}

/**
 * \brief Computes `y = A*x` and returns a new dense vector.
 *
 * \tparam MatrixType Sparse matrix type providing `data_t`, `rows()`, and
 *         `matvec()`, and `get_space()`.
 * \param A Sparse matrix operand.
 * \param x Dense input vector with length `A.cols()`.
 * \return Dense result vector with length `A.rows()`.
 * \warning This operation is currently available only for host-space matrices. It uses the naive matvec
 * when the matrix is not a CSRMatrix.
 */
template <typename MatrixType>
  requires(MatrixType::get_space() == host_space && !std::is_same_v<MatrixType, CSRMatrix<host_space, typename MatrixType::data_t>>)
Vector<host_space, typename MatrixType::data_t> matvec(
    MatrixType const& A,
    Vector<host_space, typename MatrixType::data_t> const& x)
{
  Vector<host_space, typename MatrixType::data_t> y({A.rows()});
  A.matvec(x, y);
  return y;
}

/**
 * \brief Computes `y = alpha*A*x + beta*y` using caller-provided storage.
 *
 * \tparam MatrixType Sparse matrix type providing `data_t`, `matvec()`, and
 *         `get_space()`.
 * \param A Sparse matrix operand.
 * \param x Dense input vector with length `A.cols()`.
 * \param y Dense output vector with length `A.rows()`.
 * \param alpha Matrix-vector product coefficient.
 * \param beta Existing-output coefficient.
 * \warning This operation is currently available only for host-space matrices. It uses the naive matvec
 * when the matrix is not a CSRMatrix.
 */
template <typename MatrixType>
  requires(MatrixType::get_space() == host_space && !std::is_same_v<MatrixType, CSRMatrix<host_space, typename MatrixType::data_t>>)
void matvec(
    MatrixType const& A,
    Vector<host_space, typename MatrixType::data_t> const& x,
    Vector<host_space, typename MatrixType::data_t>& y,
    typename MatrixType::data_t alpha = 1,
    typename MatrixType::data_t beta = 0)
{
  A.matvec(x, y, alpha, beta);
}

/**
 * \brief Computes `y = op(A)*x` for a CSR matrix and returns a new dense vector.
 *
 * \param A CSR matrix operand.
 * \param x Dense input vector.
 * \param op Selects `A` or its transpose.
 * \return Dense result vector.
 */
template <execution_space space, typename data_t>
Vector<space, data_t> matvec(
    CSRMatrix<space, data_t> const& A,
    Vector<space, data_t> const& x,
    sparse_op op = sparse_op::none)
{
  Vector<space, data_t> y({op == sparse_op::transpose ? A.cols() : A.rows()});
  A.matvec(x, y, data_t{1}, data_t{0}, op);
  return y;
}

/**
 * \brief Computes `y = alpha*op(A)*x + beta*y` for a CSR matrix using caller-provided storage.
 */
template <execution_space space, typename data_t>
void matvec(
    CSRMatrix<space, data_t> const& A,
    Vector<space, data_t> const& x,
    Vector<space, data_t>& y,
    data_t alpha = 1,
    data_t beta = 0,
    sparse_op op = sparse_op::none)
{
  A.matvec(x, y, alpha, beta, op);
}

/**
 * \brief Computes `C = op(A)*B` for a CSR matrix and returns a new dense matrix.
 *
 * \param A CSR matrix operand.
 * \param B Dense column-major input matrix.
 * \param op Selects `A` or its transpose.
 * \return Dense result matrix.
 */
template <execution_space space, typename data_t>
Matrix<space, data_t> matmat(
    CSRMatrix<space, data_t> const& A,
    Matrix<space, data_t> const& B,
    sparse_op op = sparse_op::none)
{
  Matrix<space, data_t> C({op == sparse_op::transpose ? A.cols() : A.rows(), B.cols()});
  A.matmat(B, C, data_t{1}, data_t{0}, op);
  return C;
}

/**
 * \brief Computes `C = alpha*op(A)*B + beta*C` for a CSR matrix using caller-provided storage.
 */
template <execution_space space, typename data_t>
void matmat(
    CSRMatrix<space, data_t> const& A,
    Matrix<space, data_t> const& B,
    Matrix<space, data_t>& C,
    data_t alpha = 1,
    data_t beta = 0,
    sparse_op op = sparse_op::none)
{
  A.matmat(B, C, alpha, beta, op);
}

/**
 * \brief Imports a dense matrix into the selected sparse format using an absolute tolerance.
 *
 * \tparam sparse_t Host-space sparse matrix type selected for the import.
 * \tparam data_t Dense matrix scalar type.
 * \param dense Dense matrix to import.
 * \param tolerance Entries with magnitude no greater than this value are omitted.
 * \return The imported sparse matrix.
 */
template <typename sparse_t, typename data_t>
  requires(
      std::is_same_v<sparse_t, COOMatrix<host_space, data_t>> ||
      std::is_same_v<sparse_t, CSRMatrix<host_space, data_t>> ||
      std::is_same_v<sparse_t, ELLPACKMatrix<host_space, data_t>>)
sparse_t from_dense(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  if constexpr (std::is_same_v<sparse_t, COOMatrix<host_space, data_t>>)
    return from_dense_coo<host_space>(dense, tolerance);
  else if constexpr (std::is_same_v<sparse_t, CSRMatrix<host_space, data_t>>)
    return from_dense_csr<host_space>(dense, tolerance);
  else
    return from_dense_ellpack<host_space>(dense, tolerance);
}

/**
 * \brief Imports a dense matrix into the selected sparse format using an absolute tolerance.
 *
 * This overload is defined for BCOO and supplies its fixed block dimensions.
 *
 * \tparam sparse_t Host-space sparse matrix type; must be
 *         `BCOOMatrix<host_space, data_t>`.
 * \tparam data_t Dense matrix scalar type.
 * \param dense Dense matrix to import.
 * \param br Number of rows in every BCOO block.
 * \param bc Number of columns in every BCOO block.
 * \param tolerance Entries with magnitude no greater than this value are omitted.
 * \return The imported BCOO matrix.
 */
template <typename sparse_t, typename data_t>
  requires(std::is_same_v<sparse_t, BCOOMatrix<host_space, data_t>>)
sparse_t from_dense(Matrix<host_space, data_t> const& dense, index_t br, index_t bc, data_t tolerance = 0)
{
  return from_dense_bcoo<host_space>(dense, br, bc, tolerance);
}

} // namespace boba

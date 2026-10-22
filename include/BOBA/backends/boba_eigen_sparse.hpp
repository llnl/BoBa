// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_op.hpp"

#ifdef BOBA_ENABLE_EIGEN
#include <Eigen/Sparse>
#endif

namespace boba
{

namespace detail
{

#ifdef BOBA_ENABLE_EIGEN

/**
 * @brief Wraps a host CSR view as a zero-copy row-major Eigen sparse map.
 * @param A Host CSR view with sorted column indices.
 * @return Read-only Eigen map aliasing the view's storage.
 */
template <typename A_view_t>
[[nodiscard]]
auto get_eigen_csr_map(A_view_t const& A)
{
  using data_t = typename A_view_t::value_t;
  return Eigen::Map<const Eigen::SparseMatrix<data_t, Eigen::RowMajor, std::int64_t>>(
    static_cast<Eigen::Index>(A.rows()),
    static_cast<Eigen::Index>(A.cols()),
    static_cast<Eigen::Index>(A.nnz()),
    A.row_offsets_data(),
    A.column_indices_data(),
    A.values_data());
}

/**
 * @brief Computes `Y = alpha * op(A) * X + beta * Y` with Eigen.
 *
 * Shared by SpMV and SpMM: X and Y are dense column-major operands mapped
 * as Eigen matrices.
 *
 * @param A Host CSR view.
 * @param X_map Eigen map of the dense input.
 * @param Y_map Eigen map of the dense output.
 */
template <typename A_view_t, typename X_map_t, typename Y_map_t, typename data_t>
void eigen_csr_product(
  A_view_t const& A,
  X_map_t const& X_map,
  Y_map_t& Y_map,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  auto A_map = get_eigen_csr_map(A);
  // beta == 0 must not read Y, which may hold uninitialized values.
  bool const overwrite = abs(beta) <= 0;
  if (op == sparse_op::transpose)
  {
    if (overwrite)
    {
      Y_map = alpha * (A_map.transpose() * X_map);
    }
    else
    {
      Y_map = beta * Y_map + alpha * (A_map.transpose() * X_map);
    }
  }
  else
  {
    if (overwrite)
    {
      Y_map = alpha * (A_map * X_map);
    }
    else
    {
      Y_map = beta * Y_map + alpha * (A_map * X_map);
    }
  }
}

/**
 * @brief Computes `y = alpha * op(A) * x + beta * y` with Eigen.
 * @param A Host CSR view.
 * @param x Dense input vector view.
 * @param y Dense output vector view.
 */
template <typename A_view_t, typename x_view_t, typename y_view_t, typename data_t>
void eigen_csr_spmv(
  A_view_t const& A,
  x_view_t const& x,
  y_view_t const& y,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  Eigen::Map<const Eigen::Vector<data_t, Eigen::Dynamic>> x_map(x.const_data(), static_cast<Eigen::Index>(x.size()));
  Eigen::Map<Eigen::Vector<data_t, Eigen::Dynamic>> y_map(y.data(), static_cast<Eigen::Index>(y.size()));
  eigen_csr_product(A, x_map, y_map, alpha, beta, op);
}

/**
 * @brief Computes `C = alpha * op(A) * B + beta * C` with Eigen.
 * @param A Host CSR view.
 * @param B Dense column-major input matrix view.
 * @param C Dense column-major output matrix view.
 */
template <typename A_view_t, typename B_view_t, typename C_view_t, typename data_t>
void eigen_csr_spmm(
  A_view_t const& A,
  B_view_t const& B,
  C_view_t const& C,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  Eigen::Map<const Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> B_map(
    B.const_data(), static_cast<Eigen::Index>(B.sizes(0)), static_cast<Eigen::Index>(B.sizes(1)));
  Eigen::Map<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> C_map(
    C.data(), static_cast<Eigen::Index>(C.sizes(0)), static_cast<Eigen::Index>(C.sizes(1)));
  eigen_csr_product(A, B_map, C_map, alpha, beta, op);
}

/**
 * @brief Computes the sparse product `C = A * B` with Eigen.
 * @param A Host CSR view.
 * @param B Host CSR view.
 * @param C CSR matrix of shape `(A.rows(), B.cols())`; its storage is replaced.
 */
template <typename A_view_t, typename B_view_t, typename C_matrix_t>
void eigen_csr_spgemm(
  A_view_t const& A,
  B_view_t const& B,
  C_matrix_t& C)
{
  using data_t = typename C_matrix_t::data_t;
  const Eigen::SparseMatrix<data_t, Eigen::RowMajor, std::int64_t> product = get_eigen_csr_map(A) * get_eigen_csr_map(B);
  const auto nnz = static_cast<index_t>(product.nonZeros());
  C.m_values.resize(nnz);
  C.m_column_indices.resize(nnz);
  auto values_view = C.m_values.view();
  auto columns_view = C.m_column_indices.view();
  auto offsets_view = C.m_row_offsets.view();
  std::copy(product.valuePtr(), product.valuePtr() + nnz, values_view.data());
  std::copy(product.innerIndexPtr(), product.innerIndexPtr() + nnz, columns_view.data());
  std::copy(product.outerIndexPtr(), product.outerIndexPtr() + C.rows() + 1, offsets_view.data());
}

#else

/**
 * @brief Reports that Eigen CSR SpMV support is unavailable.
 */
template <typename A_view_t, typename x_view_t, typename y_view_t, typename data_t>
void eigen_csr_spmv(
  A_view_t const& A,
  x_view_t const& x,
  y_view_t const& y,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  detail::ignore(A, x, y, alpha, beta, op);
  boba_error("Build error! You are trying to run eigen_csr_spmv without BOBA_ENABLE_EIGEN");
}

/**
 * @brief Reports that Eigen CSR SpMM support is unavailable.
 */
template <typename A_view_t, typename B_view_t, typename C_view_t, typename data_t>
void eigen_csr_spmm(
  A_view_t const& A,
  B_view_t const& B,
  C_view_t const& C,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  detail::ignore(A, B, C, alpha, beta, op);
  boba_error("Build error! You are trying to run eigen_csr_spmm without BOBA_ENABLE_EIGEN");
}


/**
 * @brief Reports that Eigen CSR SpGEMM support is unavailable.
 */
template <typename A_view_t, typename B_view_t, typename C_matrix_t>
void eigen_csr_spgemm(
  A_view_t const& A,
  B_view_t const& B,
  C_matrix_t& C)
{
  detail::ignore(A, B, C);
  boba_error("Build error! You are trying to run eigen_csr_spgemm without BOBA_ENABLE_EIGEN");
}

#endif

} // namespace detail
} // namespace boba

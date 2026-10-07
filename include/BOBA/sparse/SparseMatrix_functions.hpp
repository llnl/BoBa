// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/BCOOMatrix.hpp"
#include "BOBA/sparse/COOMatrix.hpp"
#include "BOBA/sparse/CSRMatrix.hpp"
#include "BOBA/sparse/DIAMatrix.hpp"
#include "BOBA/sparse/ELLPACKMatrix.hpp"

namespace boba
{

/**
 * \brief Computes `y = A*x` and returns a new dense vector.
 */
template <typename MatrixType>
Vector<host_space, typename MatrixType::data_type> matvec(MatrixType const& A, Vector<host_space, typename MatrixType::data_type> const& x)
{
  Vector<host_space, typename MatrixType::data_type> y({A.rows()});
  A.matvec(x, y);
  return y;
}

/**
 * \brief Computes `y = alpha*A*x + beta*y` using caller-provided storage.
 */
template <typename MatrixType>
void matvec(MatrixType const& A, Vector<host_space, typename MatrixType::data_type> const& x, Vector<host_space, typename MatrixType::data_type>& y, typename MatrixType::data_type alpha = 1, typename MatrixType::data_type beta = 0)
{
  A.matvec(x, y, alpha, beta);
}

/**
 * \brief Imports a dense matrix into the selected sparse format using an absolute tolerance.
 */
template <typename sparse_t, typename data_t>
sparse_t from_dense(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  if constexpr (std::is_same_v<sparse_t, COOMatrix<data_t>>)
    return from_dense_coo(dense, tolerance);
  else if constexpr (std::is_same_v<sparse_t, CSRMatrix<data_t>>)
    return from_dense_csr(dense, tolerance);
  else if constexpr (std::is_same_v<sparse_t, ELLPACKMatrix<data_t>>)
    return from_dense_ellpack(dense, tolerance);
  else if constexpr (std::is_same_v<sparse_t, DIAMatrix<data_t>>)
    return from_dense_dia(dense, tolerance);
  else
    static_assert(std::is_same_v<sparse_t, void>, "BCOO from_dense requires block dimensions");
}

/**
 * \brief Imports a dense matrix into the selected sparse format using an absolute tolerance.
 */
template <typename sparse_t, typename data_t>
sparse_t from_dense(Matrix<host_space, data_t> const& dense, index_t br, index_t bc, data_t tolerance = 0)
{
  static_assert(std::is_same_v<sparse_t, BCOOMatrix<data_t>>, "Block-parameterized from_dense is only defined for BCOO");
  return from_dense_bcoo(dense, br, bc, tolerance);
}

} // namespace boba

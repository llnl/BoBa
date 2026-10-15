// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/BCOOMatrix.hpp"
#include "BOBA/sparse/COOMatrix.hpp"
#include "BOBA/sparse/CSRMatrix.hpp"
#include "BOBA/sparse/ELLPACKMatrix.hpp"

namespace boba
{

/**
 * \brief Computes `y = A*x` and returns a new dense vector.
 *
 * \tparam MatrixType Sparse matrix type providing `data_t`, `rows()`, and
 *         `matvec()`, and `get_space()`.
 * \param A Sparse matrix operand.
 * \param x Dense input vector with length `A.cols()`.
 * \return Dense result vector with length `A.rows()`.
 * \warning This operation is currently available only for host-space matrices.
 */
template <typename MatrixType>
  requires(MatrixType::get_space() == host_space)
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
 * \warning This operation is currently available only for host-space matrices.
 */
template <typename MatrixType>
  requires(MatrixType::get_space() == host_space)
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

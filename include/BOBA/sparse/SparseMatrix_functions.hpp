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
 * \brief Imports a dense matrix into this sparse format using an absolute tolerance.
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
        out.set_element(i, j, dv({i, j}));
    }
  }
  return out;
}

/**
 * \brief Imports a dense matrix into this sparse format using an absolute tolerance.
 */
template <typename data_t>
CSRMatrix<data_t> from_dense_csr(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  CSRMatrix<data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    for (index_t j = 0; j < dense.cols(); ++j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
        out.set_element(i, j, dv({i, j}));
    }
  }
  return out;
}

/**
 * \brief Imports a dense matrix into this sparse format using an absolute tolerance.
 */
template <typename data_t>
ELLPACKMatrix<data_t> from_dense_ellpack(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  index_t w = 0;
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    index_t c = 0;
    for (index_t j = 0; j < dense.cols(); ++j)
    {
      c += sparse_detail::keep(dv({i, j}), tolerance);
    }
    w = std::max(w, c);
  }
  ELLPACKMatrix<data_t> out(dense.rows(), dense.cols(), w);
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    for (index_t j = 0, k = 0; j < dense.cols(); ++j)
    {
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        out.m_column_indices.view()({i, k}) = static_cast<std::int64_t>(j);
        out.m_values.view()({i, k++}) = dv({i, j});
      }
    }
  }
  return out;
}

/**
 * \brief Imports a dense matrix into this sparse format using an absolute tolerance.
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
        out.set_element(i, j, dv({i, j}));
    }
  }
  return out;
}

/**
 * \brief Imports a dense matrix into BCOO storage with explicit block dimensions.
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
        out.set_element(i, j, dv({i, j}));
    }
  }
  return out;
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

/**
 * \brief Exports a sparse matrix to a dense host matrix.
 */
template <typename MatrixType>
Matrix<host_space, typename MatrixType::data_type> to_dense(MatrixType const& sparse)
{
  using data_t = typename MatrixType::data_type;
  Matrix<host_space, data_t> out({sparse.rows(), sparse.cols()});
  out.fill_with_zeros();
  auto ov = out.view();
  for (index_t i = 0; i < sparse.rows(); ++i)
  {
    for (index_t j = 0; j < sparse.cols(); ++j)
    {
      ov({i, j}) = sparse.get_element(i, j);
    }
  }
  return out;
}

} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/BCOOMatrix.hpp"
#include "BOBA/sparse/COOMatrix.hpp"
#include "BOBA/sparse/CSRMatrix.hpp"
#include "BOBA/sparse/DIAMatrix.hpp"
#include "BOBA/sparse/ELLPACKMatrix.hpp"

namespace boba
{

template <typename MatrixType>
Vector<host_space, typename MatrixType::data_type> matvec(MatrixType const& A, Vector<host_space, typename MatrixType::data_type> const& x)
{
  Vector<host_space, typename MatrixType::data_type> y({A.nrows()});
  A.matvec(x, y);
  return y;
}

template <typename MatrixType>
void matvec(MatrixType const& A, Vector<host_space, typename MatrixType::data_type> const& x, Vector<host_space, typename MatrixType::data_type>& y, typename MatrixType::data_type alpha = 1, typename MatrixType::data_type beta = 0)
{
  A.matvec(x, y, alpha, beta);
}

template <typename data_t>
COOMatrix<data_t> from_dense_coo(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  COOMatrix<data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
    for (index_t j = 0; j < dense.cols(); ++j)
      if (sparse_detail::keep(dv({i, j}), tolerance))
        out.set_element(i, j, dv({i, j}));
  return out;
}

template <typename data_t>
CSRMatrix<data_t> from_dense_csr(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  CSRMatrix<data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
    for (index_t j = 0; j < dense.cols(); ++j)
      if (sparse_detail::keep(dv({i, j}), tolerance))
        out.set_element(i, j, dv({i, j}));
  return out;
}
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
      c += sparse_detail::keep(dv({i, j}), tolerance);
    w = std::max(w, c);
  }
  ELLPACKMatrix<data_t> out(dense.rows(), dense.cols(), w);
  for (index_t i = 0; i < dense.rows(); ++i)
    for (index_t j = 0, k = 0; j < dense.cols(); ++j)
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        out.column_indices.view()({i, k}) = static_cast<std::int64_t>(j);
        out.values.view()({i, k++}) = dv({i, j});
      }
  return out;
}
template <typename data_t>
DIAMatrix<data_t> from_dense_dia(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  DIAMatrix<data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
    for (index_t j = 0; j < dense.cols(); ++j)
      if (sparse_detail::keep(dv({i, j}), tolerance))
        out.set_element(i, j, dv({i, j}));
  return out;
}
template <typename data_t>
BCOOMatrix<data_t> from_dense_bcoo(Matrix<host_space, data_t> const& dense, index_t br, index_t bc, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  boba_always_assert(br > 0 && bc > 0, "BCOO block dimensions must be positive");
  boba_always_assert(dense.rows() % br == 0 && dense.cols() % bc == 0, "BCOO dimensions must be divisible by block dimensions");
  BCOOMatrix<data_t> out(br, bc, dense.rows() / br, dense.cols() / bc);
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
    for (index_t j = 0; j < dense.cols(); ++j)
      if (sparse_detail::keep(dv({i, j}), tolerance))
        out.set_element(i, j, dv({i, j}));
  return out;
}

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

template <typename sparse_t, typename data_t>
sparse_t from_dense(Matrix<host_space, data_t> const& dense, index_t br, index_t bc, data_t tolerance = 0)
{
  static_assert(std::is_same_v<sparse_t, BCOOMatrix<data_t>>, "Block-parameterized from_dense is only defined for BCOO");
  return from_dense_bcoo(dense, br, bc, tolerance);
}

template <typename MatrixType>
Matrix<host_space, typename MatrixType::data_type> to_dense(MatrixType const& sparse)
{
  using data_t = typename MatrixType::data_type;
  Matrix<host_space, data_t> out({sparse.nrows(), sparse.ncols()});
  out.fill_with_zeros();
  auto ov = out.view();
  for (index_t i = 0; i < sparse.nrows(); ++i)
    for (index_t j = 0; j < sparse.ncols(); ++j)
      ov({i, j}) = sparse.get_element(i, j);
  return out;
}

/**
 * \brief Replaces every value in one sparse matrix row.
 */
template <typename MatrixType>
void set_row(MatrixType& sparse, index_t row, Vector<host_space, typename MatrixType::data_type> const& values)
{
  boba_always_assert(row < sparse.nrows(), "Sparse row out of bounds");
  boba_always_assert_equal(values.size(), sparse.ncols(), "Sparse row has the wrong size");
  auto input = values.const_view();
  for (index_t j = 0; j < sparse.ncols(); ++j)
    sparse.set_element(row, j, typename MatrixType::data_type{});
  for (index_t j = 0; j < sparse.ncols(); ++j)
    sparse.set_element(row, j, input(j));
}

/**
 * \brief Replaces every value in one sparse matrix column.
 */
template <typename MatrixType>
void set_col(MatrixType& sparse, index_t col, Vector<host_space, typename MatrixType::data_type> const& values)
{
  boba_always_assert(col < sparse.ncols(), "Sparse column out of bounds");
  boba_always_assert_equal(values.size(), sparse.nrows(), "Sparse column has the wrong size");
  auto input = values.const_view();
  for (index_t i = 0; i < sparse.nrows(); ++i)
    sparse.set_element(i, col, typename MatrixType::data_type{});
  for (index_t i = 0; i < sparse.nrows(); ++i)
    sparse.set_element(i, col, input(i));
}

/**
 * \brief Replaces a diagonal identified by `column - row`.
 */
template <typename MatrixType>
void set_diagonal(MatrixType& sparse, std::int64_t diagonal, Vector<host_space, typename MatrixType::data_type> const& values)
{
  auto first_row = std::max<std::int64_t>(0, -diagonal);
  auto first_col = std::max<std::int64_t>(0, diagonal);
  auto length = std::min<std::int64_t>(static_cast<std::int64_t>(sparse.nrows()) - first_row,
                                       static_cast<std::int64_t>(sparse.ncols()) - first_col);
  boba_always_assert_equal(values.size(), static_cast<index_t>(length), "Sparse diagonal has the wrong size");
  auto input = values.const_view();
  for (std::int64_t k = 0; k < length; ++k)
    sparse.set_element(static_cast<index_t>(first_row + k), static_cast<index_t>(first_col + k), input(static_cast<index_t>(k)));
}

} // namespace boba

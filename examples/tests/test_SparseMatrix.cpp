// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>

using vector_t = boba::Vector<boba::host_space, double>;
using matrix_t = boba::Matrix<boba::host_space, double>;

/**
 * \brief Takes a dense matrix and compares sparse construction from this matrix
 * Tests sparse lookup, to_dense export, and matvec
 */
template <typename Sparse>
void check_sparse(Sparse const& sparse,
                  matrix_t const& expected,
                  vector_t const& x,
                  vector_t const& y_initial,
                  vector_t const& y_expected,
                  bool& check)
{
  auto dense = boba::to_dense(sparse);
  auto expected_view = expected.const_view();
  auto dense_view = dense.const_view();
  double error = 0.0;
  for (boba::index_t i = 0; i < expected.rows(); ++i)
  {
    for (boba::index_t j = 0; j < expected.cols(); ++j)
    {
      error = std::max(error, boba::abs(dense_view({i, j}) - expected_view({i, j})));
      error = std::max(error, boba::abs(sparse.get_element(i, j) - expected_view({i, j})));
    }
  }
  pass_or_fail(check, error, 1.0e-14);

  vector_t y = y_initial;
  sparse.matvec(x, y, 3.0, 2.0);
  double matvec_error = 0.0;
  auto y_view = y.const_view();
  auto y_expected_view = y_expected.const_view();
  for (boba::index_t i = 0; i < y.size(); ++i)
  {
    matvec_error = std::max(matvec_error, boba::abs(y_view(i) - y_expected_view(i)));
  }
  pass_or_fail(check, matvec_error, 1.0e-13);
}

/**
 * \brief Exercises all initial sparse matrix formats against dense storage.
 */
int main()
{
  bool check = true;
  matrix_t dense({4, 5});
  auto dv = dense.view();
  for (boba::index_t i = 0; i < dense.rows(); ++i)
  {
    for (boba::index_t j = 0; j < dense.cols(); ++j)
    {
      dv({i, j}) = ((i + 2 * j) % 3 == 0) ? static_cast<double>(i) - static_cast<double>(j) + 1.0 : 0.0;
    }
  }
  dv({0, 1}) = 1.0e-8;
  dv({1, 1}) = -1.0e-3;

  constexpr double tolerance = 1.0e-7;
  auto dense_thresholded = apply_function(dense, [tolerance](auto x){ return (boba::abs(x) > tolerance) ? x : 0.0; });

  vector_t x({dense.cols()}), y_initial({dense.rows()});
  auto xv = x.view();
  auto y_initial_view = y_initial.view();
  for (boba::index_t i = 0; i < x.size(); ++i)
  {
    xv(i) = static_cast<double>(i + 1);
  }
  for (boba::index_t i = 0; i < y_initial.size(); ++i)
  {
    y_initial_view(i) = 0.5 * static_cast<double>(i + 1);
  }

  auto y_expected = 3.0 * (dense_thresholded * x) + 2.0 * y_initial;

  check_sparse(boba::from_dense<boba::COOMatrix<double>>(dense, tolerance), dense_thresholded, x, y_initial, y_expected, check);
  check_sparse(boba::from_dense<boba::CSRMatrix<double>>(dense, tolerance), dense_thresholded, x, y_initial, y_expected, check);
  check_sparse(boba::from_dense<boba::ELLPACKMatrix<double>>(dense, tolerance), dense_thresholded, x, y_initial, y_expected, check);
  check_sparse(boba::from_dense<boba::DIAMatrix<double>>(dense, tolerance), dense_thresholded, x, y_initial, y_expected, check);
  check_sparse(boba::from_dense<boba::BCOOMatrix<double>>(dense, 2, 1, tolerance), dense_thresholded, x, y_initial, y_expected, check);

  return final_check(check);
}

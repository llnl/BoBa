// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>
#include <iostream>

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
                  char const* format,
                  bool& check)
{
  auto dense = boba::to_dense(sparse);
  auto expected_view = expected.const_view();
  auto dense_view = dense.const_view();
  double error = 0.0;
  //
  // Checks that sparse matrix is identically equal to dense matrix
  //
  for (boba::index_t i = 0; i < expected.rows(); ++i)
  {
    for (boba::index_t j = 0; j < expected.cols(); ++j)
    {
      error = std::max(error, boba::abs(dense_view({i, j}) - expected_view({i, j})));
      error = std::max(error, boba::abs(sparse.get_element(i, j) - expected_view({i, j})));
    }
  }

  pass_or_fail(check, error, 1.0e-14);
  std::cout << format << " compression rate: " << sparse.compression_rate() << "x" << std::endl;

  //
  // Verifies matvec
  //
  vector_t y = y_initial;
  sparse.matvec(x, y, 3.0, 2.0);
  double matvec_error = norm_difference_inf(y, y_expected);

  pass_or_fail(check, matvec_error, 1.0e-13);
}

/**
 * \brief Exercises all initial sparse matrix formats against dense storage.
 */
int main()
{
  bool check = true;

  //
  //
  //
  matrix_t dense({4, 4});
  auto dv = dense.view();
  dense.fill_with_zeros();
  dv({0, 0}) = 1.0;
  dv({0, 1}) = 2.0;
  dv({0, 2}) = 1.0e-20;
  dv({1, 0}) = 3.0;
  dv({1, 1}) = 4.0;
  dv({1, 2}) = 5.0;
  dv({2, 1}) = 2.0e-20;
  dv({2, 2}) = 6.0;
  dv({2, 3}) = 7.0;
  dv({3, 2}) = 8.0;
  dv({3, 3}) = 9.0;

  constexpr double tolerance = 1.0e-12;
  auto dense_thresholded = boba::apply_function(dense, [tolerance](auto x)
  {
    return (boba::abs(x) > tolerance) ? x : 0.0;
  });

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

  check_sparse(boba::from_dense<boba::COOMatrix<double>>(dense, tolerance), dense_thresholded, x, y_initial, y_expected, "COO", check);
  check_sparse(boba::from_dense<boba::CSRMatrix<double>>(dense, tolerance), dense_thresholded, x, y_initial, y_expected, "CSR", check);
  check_sparse(boba::from_dense<boba::ELLPACKMatrix<double>>(dense, tolerance), dense_thresholded, x, y_initial, y_expected, "ELLPACK", check);
  check_sparse(boba::from_dense<boba::DIAMatrix<double>>(dense, tolerance), dense_thresholded, x, y_initial, y_expected, "DIA", check);
  check_sparse(boba::from_dense<boba::BCOOMatrix<double>>(dense, 2, 2, tolerance), dense_thresholded, x, y_initial, y_expected, "BCOO", check);

  return final_check(check);
}

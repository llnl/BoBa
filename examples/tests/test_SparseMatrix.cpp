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
                  vector_t const& y,
                  vector_t const& z,
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
  boba::loop<boba::host_space, 2>({expected.rows(), expected.cols()}, [&](boba::Array<boba::index_t, 2> indices)
  {
    auto [i, j] = indices;
    error = std::max(error, boba::abs(dense_view({i, j}) - expected_view({i, j})));
    error = std::max(error, boba::abs(sparse.get_element(i, j) - expected_view({i, j})));
  });

  pass_or_fail(check, error, 1.0e-14);
  std::cout << format << " compression rate: " << sparse.compression_rate() << "x" << std::endl;

  //
  // Verifies matvec
  //
  vector_t z_matvec = y;
  sparse.matvec(x, z_matvec, 3.0, 2.0);
  double matvec_error = norm_difference_inf(z_matvec, z);

  pass_or_fail(check, matvec_error, 1.0e-13);
}

/**
 * \brief Exercises all initial sparse matrix formats against dense storage.
 */
int main()
{
  bool check = true;

  //
  // Generate matrix
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

  //
  // Delete small entries in a way consistent with the sparse tensor constructors
  //
  constexpr double tolerance = 1.0e-12;
  auto dense_thresholded = boba::apply_function(dense, [=]__boba_host_device__(auto x)
  {
    return (boba::abs(x) > tolerance) ? x : 0.0;
  });

  //
  // Initialize some vectors
  //
  vector_t x({dense.cols()}), y({dense.rows()});
  auto xv = x.view();
  auto y_view = y.view();
  boba::loop<boba::host_space, 1>(0, x.size(), [&](boba::index_t i)
  {
    xv(i) = static_cast<double>(i + 1);
  });
  boba::loop<boba::host_space, 1>(0, y.size(), [&](boba::index_t i)
  {
    y_view(i) = 0.5 * static_cast<double>(i + 1);
  });

  //
  // Dense operation that we will test with each format
  //
  auto z = 3.0 * (dense_thresholded * x) + 2.0 * y;

  //
  // Test sparse formats
  //
  check_sparse(boba::from_dense<boba::COOMatrix<double>>(dense, tolerance), dense_thresholded, x, y, z, "COO", check);
  check_sparse(boba::from_dense<boba::CSRMatrix<double>>(dense, tolerance), dense_thresholded, x, y, z, "CSR", check);
  check_sparse(boba::from_dense<boba::ELLPACKMatrix<double>>(dense, tolerance), dense_thresholded, x, y, z, "ELLPACK", check);
  check_sparse(boba::from_dense<boba::DIAMatrix<double>>(dense, tolerance), dense_thresholded, x, y, z, "DIA", check);
  check_sparse(boba::from_dense<boba::BCOOMatrix<double>>(dense, 2, 2, tolerance), dense_thresholded, x, y, z, "BCOO", check);

  return final_check(check);
}

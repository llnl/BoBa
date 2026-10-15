// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

using vector_t = boba::Vector<boba::host_space, double>;
using matrix_t = boba::Matrix<boba::host_space, double>;

/**
 * \brief Validates the common sparse-matrix API against a dense reference.
 *
 * This exercises shape metadata, naive lookup, dense export, compression-rate
 * reporting, both matvec overloads, and the sparse matrix-vector operation.
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
  pass_or_fail_bool(check, sparse.rows() == expected.rows());
  pass_or_fail_bool(check, sparse.cols() == expected.cols());
  const auto shape = sparse.shape();
  pass_or_fail_bool(check, shape[0] == expected.rows() && shape[1] == expected.cols());

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
    error = std::max(error, boba::abs(sparse.get_element_naive(i, j) - expected_view({i, j})));
  });

  pass_or_fail(check, error, 1.0e-14);
  const auto compression_rate = sparse.compression_rate();
  pass_or_fail_bool(check, std::isfinite(static_cast<double>(compression_rate)));
  std::cout << format << " compression rate: " << compression_rate << "x" << std::endl;

  //
  // Verifies matvec
  //
  vector_t z_matvec = y;
  sparse.matvec(x, z_matvec, 3.0, 2.0);
  double matvec_error = norm_difference_inf(z_matvec, z);

  pass_or_fail(check, matvec_error, 1.0e-13);

  auto reference_matvec = expected * x;
  auto wrapper_matvec = boba::matvec(sparse, x);
  pass_or_fail(check, norm_difference_inf(wrapper_matvec, reference_matvec), 1.0e-13);
  vector_t wrapper_output = y;
  boba::matvec(sparse, x, wrapper_output, 3.0, 2.0);
  pass_or_fail(check, norm_difference_inf(wrapper_output, z), 1.0e-13);
}

/**
 * \brief Validates the zero-state metadata of a default-constructed matrix.
 */
template <typename Sparse>
void check_empty_sparse(Sparse const& sparse, bool& check)
{
  pass_or_fail_bool(check, sparse.rows() == 0);
  pass_or_fail_bool(check, sparse.cols() == 0);
  const auto shape = sparse.shape();
  pass_or_fail_bool(check, shape[0] == 0 && shape[1] == 0);
  pass_or_fail_bool(check, sparse.nnz() == 0);
  pass_or_fail(check, static_cast<double>(sparse.compression_rate()), 1.0e-14);
}

/**
 * \brief Exercises all initial sparse matrix formats against dense storage.
 */
int main()
{
  boba::init();

  bool check = true;

  boba::COOMatrix<double> empty_coo;
  boba::CSRMatrix<double> empty_csr;
  boba::ELLPACKMatrix<double> empty_ellpack;
  boba::BCOOMatrix<double> empty_bcoo;
  check_empty_sparse(empty_coo, check);
  check_empty_sparse(empty_csr, check);
  check_empty_sparse(empty_ellpack, check);
  check_empty_sparse(empty_bcoo, check);
  pass_or_fail_bool(check, empty_bcoo.nonzero_blocks() == 0);

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
  auto dense_thresholded = boba::apply_function(dense, [=]__boba_host_device__(double x)
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
  auto coo = boba::from_dense<boba::COOMatrix<double>>(dense, tolerance);
  auto csr = boba::from_dense<boba::CSRMatrix<double>>(dense, tolerance);
  auto ellpack = boba::from_dense<boba::ELLPACKMatrix<double>>(dense, tolerance);
  auto bcoo = boba::from_dense<boba::BCOOMatrix<double>>(dense, 2, 2, tolerance);
  pass_or_fail_bool(check, coo.nnz() == 9);
  pass_or_fail_bool(check, csr.nnz() == 9);
  pass_or_fail_bool(check, ellpack.nnz() == 9);
  pass_or_fail_bool(check, bcoo.nnz() == 12);
  pass_or_fail_bool(check, bcoo.nonzero_blocks() == 3);
  check_sparse(coo, dense_thresholded, x, y, z, "COO", check);
  check_sparse(csr, dense_thresholded, x, y, z, "CSR", check);
  check_sparse(ellpack, dense_thresholded, x, y, z, "ELLPACK", check);
  check_sparse(bcoo, dense_thresholded, x, y, z, "BCOO", check);

  const int result = final_check(check);
  boba::finalize();
  return result;
}

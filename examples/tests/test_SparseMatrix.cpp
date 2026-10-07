// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>

using vector_t = boba::Vector<boba::host_space, double>;
using matrix_t = boba::Matrix<boba::host_space, double>;

/**
 * \brief Computes the dense reference result after applying the sparse tolerance.
 */
vector_t make_y_expected(matrix_t dense, vector_t const& x, vector_t const& y, double tolerance)
{
  auto dense_view = dense.view();
  for (boba::index_t i = 0; i < dense.rows(); ++i)
  {
    for (boba::index_t j = 0; j < dense.cols(); ++j)
    {
      if (boba::abs(dense_view({i, j})) <= tolerance)
      {
        dense_view({i, j}) = 0.0;
      }
    }
  }

  vector_t y_expected = y;
  vector_t dense_product = dense * x;
  auto expected_view = y_expected.view();
  auto product_view = dense_product.const_view();
  for (boba::index_t i = 0; i < y_expected.size(); ++i)
  {
    expected_view(i) += 2.0 * product_view(i);
  }
  return y_expected;
}

/**
 * \brief Takes a dense matrix and compares sparse construction from this matrix
 * Tests sparse lookup and to_dense export.
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
  sparse.matvec(x, y, 2.0, 1.0);
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
  auto thresholded = dense;
  auto thresholded_view = thresholded.view();
  for (boba::index_t i = 0; i < thresholded.rows(); ++i)
  {
    for (boba::index_t j = 0; j < thresholded.cols(); ++j)
    {
      if (boba::abs(thresholded_view({i, j})) <= tolerance)
      {
        thresholded_view({i, j}) = 0.0;
      }
    }
  }

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
  auto y_expected = make_y_expected(dense, x, y_initial, tolerance);

  check_sparse(boba::from_dense<boba::COOMatrix<double>>(dense, tolerance), thresholded, x, y_initial, y_expected, check);
  check_sparse(boba::from_dense<boba::CSRMatrix<double>>(dense, tolerance), thresholded, x, y_initial, y_expected, check);
  check_sparse(boba::from_dense<boba::ELLPACKMatrix<double>>(dense, tolerance), thresholded, x, y_initial, y_expected, check);
  check_sparse(boba::from_dense<boba::DIAMatrix<double>>(dense, tolerance), thresholded, x, y_initial, y_expected, check);
  check_sparse(boba::from_dense<boba::BCOOMatrix<double>>(dense, 2, 1, tolerance), thresholded, x, y_initial, y_expected, check);

  boba::COOMatrix<double> manual(3, 4);
  manual.set_element(2, 3, 4.0);
  manual.set_element(0, 1, -2.0);
  pass_or_fail(check, manual.get_element(1, 2), 1.0e-14);
  manual.add_element(0, 1, 3.0);
  pass_or_fail(check, manual.get_element(0, 1) - 1.0, 1.0e-14);

  return final_check(check);
}

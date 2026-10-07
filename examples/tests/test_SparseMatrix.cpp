// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>

using boba::host_space;
using boba::Matrix;
using boba::Vector;

template <typename Sparse>
void check_sparse(Sparse const& sparse, Matrix<host_space, double> const& expected, bool& check)
{
  auto dense = boba::to_dense(sparse);
  auto expected_view = expected.const_view();
  auto dense_view = dense.const_view();
  double error = 0.0;
  for (boba::index_t i = 0; i < expected.rows(); ++i)
    for (boba::index_t j = 0; j < expected.cols(); ++j)
    {
      error = std::max(error, boba::abs(dense_view({i, j}) - expected_view({i, j})));
      error = std::max(error, boba::abs(sparse.get_element(i, j) - expected_view({i, j})));
      error = std::max(error, boba::abs(sparse.as_const_view().get_element(i, j) - expected_view({i, j})));
    }
  pass_or_fail(check, error, 1.0e-14);

  Vector<host_space, double> x({expected.cols()}), y({expected.rows()}), y_expected({expected.rows()});
  auto xv = x.view();
  auto yv = y.view();
  auto eyv = y_expected.view();
  for (boba::index_t i = 0; i < x.size(); ++i)
    xv(i) = static_cast<double>(i + 1);
  for (boba::index_t i = 0; i < y.size(); ++i)
    yv(i) = 0.5 * static_cast<double>(i + 1);
  y_expected = y;
  auto exv = y_expected.view();
  for (boba::index_t i = 0; i < expected.rows(); ++i)
    for (boba::index_t j = 0; j < expected.cols(); ++j)
      exv(i) += 2.0 * expected_view({i, j}) * xv(j);
  sparse.matvec(x, y, 2.0, 1.0);
  double matvec_error = 0.0;
  for (boba::index_t i = 0; i < y.size(); ++i)
    matvec_error = std::max(matvec_error, boba::abs(y.const_view()(i) - exv(i)));
  pass_or_fail(check, matvec_error, 1.0e-13);
}

int main()
{
  bool check = true;
  Matrix<host_space, double> dense({4, 5});
  auto dv = dense.view();
  for (boba::index_t i = 0; i < dense.rows(); ++i)
    for (boba::index_t j = 0; j < dense.cols(); ++j)
      dv({i, j}) = ((i + 2 * j) % 3 == 0) ? static_cast<double>(i - j + 1) : 0.0;
  dv({0, 1}) = 1.0e-8;
  dv({1, 1}) = -1.0e-3;

  auto thresholded = dense;
  thresholded.view()({0, 1}) = 0.0;
  check_sparse(boba::from_dense<boba::COOMatrix<double>>(dense, 1.0e-7), thresholded, check);
  check_sparse(boba::from_dense<boba::CSRMatrix<double>>(dense, 1.0e-7), thresholded, check);
  check_sparse(boba::from_dense<boba::ELLPACKMatrix<double>>(dense, 1.0e-7), thresholded, check);
  check_sparse(boba::from_dense<boba::DIAMatrix<double>>(dense, 1.0e-7), thresholded, check);
  check_sparse(boba::from_dense<boba::BCOOMatrix<double>>(dense, 2, 1, 1.0e-7), thresholded, check);

  boba::COOMatrix<double> manual(3, 4);
  manual.set_element(2, 3, 4.0);
  manual.set_element(0, 1, -2.0);
  pass_or_fail(check, manual.get_element(1, 2), 1.0e-14);
  manual.add_element(0, 1, 3.0);
  pass_or_fail(check, manual.get_element(0, 1) - 1.0, 1.0e-14);

  return final_check(check);
}

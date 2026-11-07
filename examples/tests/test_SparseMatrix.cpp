// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>

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
 * \brief Verifies a COO host-device-host copy round trip.
 * \param original Host COO matrix used as the reference.
 * \param check Cumulative test status.
 */
void check_coo_round_trip(boba::COOMatrix<boba::host_space, double> const& original, bool& check)
{
  boba::COOMatrix<boba::default_execution_space, double> device_copy(original);
  boba::COOMatrix<boba::host_space, double> round_trip(device_copy);
  pass_or_fail_bool(check, original.m_values.sizes() == round_trip.m_values.sizes());
  pass_or_fail(check, boba::norm_difference_inf(original.m_values, round_trip.m_values), 1.0e-14);
  pass_or_fail_bool(check, original.m_indices.sizes() == round_trip.m_indices.sizes());
  pass_or_fail(check, boba::norm_difference_inf(original.m_indices, round_trip.m_indices), 1.0e-14);
  std::cout << "COO host-device-host copy: pass" << std::endl;
}

/**
 * \brief Verifies a CSR host-device-host copy round trip.
 * \param original Host CSR matrix used as the reference.
 * \param check Cumulative test status.
 */
void check_csr_round_trip(boba::CSRMatrix<boba::host_space, double> const& original, bool& check)
{
  boba::CSRMatrix<boba::default_execution_space, double> device_copy(original);
  boba::CSRMatrix<boba::host_space, double> round_trip(device_copy);
  pass_or_fail_bool(check, original.m_values.sizes() == round_trip.m_values.sizes());
  pass_or_fail(check, boba::norm_difference_inf(original.m_values, round_trip.m_values), 1.0e-14);
  pass_or_fail_bool(check, original.m_column_indices.sizes() == round_trip.m_column_indices.sizes());
  pass_or_fail(check, boba::norm_difference_inf(original.m_column_indices, round_trip.m_column_indices), 1.0e-14);
  pass_or_fail_bool(check, original.m_row_offsets.sizes() == round_trip.m_row_offsets.sizes());
  pass_or_fail(check, boba::norm_difference_inf(original.m_row_offsets, round_trip.m_row_offsets), 1.0e-14);
  std::cout << "CSR host-device-host copy: pass" << std::endl;
}

/**
 * \brief Verifies an ELLPACK host-device-host copy round trip.
 * \param original Host ELLPACK matrix used as the reference.
 * \param check Cumulative test status.
 */
void check_ellpack_round_trip(boba::ELLPACKMatrix<boba::host_space, double> const& original, bool& check)
{
  boba::ELLPACKMatrix<boba::default_execution_space, double> device_copy(original);
  boba::ELLPACKMatrix<boba::host_space, double> round_trip(device_copy);
  pass_or_fail_bool(check, original.m_values.sizes() == round_trip.m_values.sizes());
  pass_or_fail(check, boba::norm_difference_inf(original.m_values, round_trip.m_values), 1.0e-14);
  pass_or_fail_bool(check, original.m_column_indices.sizes() == round_trip.m_column_indices.sizes());
  pass_or_fail(check, boba::norm_difference_inf(original.m_column_indices, round_trip.m_column_indices), 1.0e-14);
  std::cout << "ELLPACK host-device-host copy: pass" << std::endl;
}

/**
 * \brief Verifies a BCOO host-device-host copy round trip.
 * \param original Host BCOO matrix used as the reference.
 * \param check Cumulative test status.
 */
void check_bcoo_round_trip(boba::BCOOMatrix<boba::host_space, double> const& original, bool& check)
{
  boba::BCOOMatrix<boba::default_execution_space, double> device_copy(original);
  boba::BCOOMatrix<boba::host_space, double> round_trip(device_copy);
  pass_or_fail_bool(check, original.m_values.sizes() == round_trip.m_values.sizes());
  pass_or_fail(check, boba::norm_difference_inf(original.m_values, round_trip.m_values), 1.0e-14);
  pass_or_fail_bool(check, original.m_indices.sizes() == round_trip.m_indices.sizes());
  pass_or_fail(check, boba::norm_difference_inf(original.m_indices, round_trip.m_indices), 1.0e-14);
  std::cout << "BCOO host-device-host copy: pass" << std::endl;
}

/**
 * \brief Returns a deterministic test value; complex types get a nonzero imaginary part.
 */
template <typename data_t>
data_t csr_test_value(boba::index_t seed)
{
  const double real = 0.25 * static_cast<double>(seed % 7) - 0.5;
  if constexpr (std::is_same_v<data_t, boba::complex<double>> || std::is_same_v<data_t, boba::complex<float>>)
  {
    using real_t = boba::real_type_t<data_t>;
    return data_t{static_cast<real_t>(real), static_cast<real_t>(0.125 * static_cast<double>(seed % 5))};
  }
  else
  {
    return static_cast<data_t>(real);
  }
}

/**
 * \brief Fills a host vector with one value.
 */
template <typename data_t>
void fill_with_value(boba::Vector<boba::host_space, data_t>& vector, data_t value)
{
  auto vector_view = vector.view();
  for (boba::index_t i = 0; i < vector.size(); ++i)
  {
    vector_view(i) = value;
  }
}

/**
 * \brief Fills a dense host matrix with deterministic values.
 */
template <typename data_t>
void fill_csr_test_values(boba::Matrix<boba::host_space, data_t>& matrix, boba::index_t offset)
{
  auto matrix_view = matrix.view();
  boba::loop<boba::host_space, 2>({matrix.rows(), matrix.cols()}, [&](boba::Array<boba::index_t, 2> indices)
  {
    auto [i, j] = indices;
    matrix_view({i, j}) = csr_test_value<data_t>(offset + 3 * i + 5 * j);
  });
}

/**
 * \brief Validates CSR SpMV, SpMM, and SpGEMM in `default_execution_space` against dense products.
 *
 * Covers both `sparse_op` values, accumulation with nonzero beta, overwrite
 * with zero beta, and an empty matrix.
 */
template <typename data_t>
void check_csr_backend(double tolerance, bool& check)
{
  constexpr auto space = boba::default_execution_space;
  using host_matrix_t = boba::Matrix<boba::host_space, data_t>;
  using host_vector_t = boba::Vector<boba::host_space, data_t>;
  using real_t = boba::real_type_t<data_t>;
  const auto alpha = boba::PotentiallyComplex<data_t>::value(static_cast<real_t>(3));
  const auto beta = boba::PotentiallyComplex<data_t>::value(static_cast<real_t>(2));
  const auto zero = boba::PotentiallyComplex<data_t>::value(static_cast<real_t>(0));

  constexpr boba::index_t rows = 7;
  constexpr boba::index_t cols = 5;
  constexpr boba::index_t num_rhs = 3;
  host_matrix_t dense({rows, cols});
  dense.fill_with_zeros();
  auto dense_view = dense.view();
  for (boba::index_t i = 0; i < rows; ++i)
  {
    for (boba::index_t j = 0; j < cols; ++j)
    {
      if ((i + 2 * j) % 3 == 0)
      {
        dense_view({i, j}) = csr_test_value<data_t>(i * cols + j + 1);
      }
    }
  }
  const auto host_csr = boba::from_dense_csr<boba::host_space>(dense);
  const boba::CSRMatrix<space, data_t> csr(host_csr);
  const auto dense_transpose = dense.transpose();

  for (const auto op : {boba::sparse_op::none, boba::sparse_op::transpose})
  {
    const bool transposed = op == boba::sparse_op::transpose;
    const auto& op_dense = transposed ? dense_transpose : dense;
    const auto in_size = op_dense.cols();
    const auto out_size = op_dense.rows();

    host_matrix_t B({in_size, num_rhs}), C0({out_size, num_rhs});
    fill_csr_test_values(B, 11);
    fill_csr_test_values(C0, 17);
    const auto C_expected = alpha * (op_dense * B) + beta * C0;
    const auto C_overwrite_expected = alpha * (op_dense * B);

    boba::Matrix<space, data_t> B_device(B), C_device(C0);
    csr.matmat(B_device, C_device, alpha, beta, op);
    pass_or_fail(check, boba::norm_difference_inf(host_matrix_t(C_device), C_expected), tolerance);

    // beta == 0 must overwrite, even when C holds NaN.
    host_matrix_t C_nan({out_size, num_rhs});
    auto C_nan_view = C_nan.view();
    const auto nan = boba::PotentiallyComplex<data_t>::value(std::numeric_limits<real_t>::quiet_NaN());
    for (boba::index_t k = 0; k < C_nan.size(); ++k)
    {
      C_nan_view.data()[k] = nan;
    }
    boba::Matrix<space, data_t> C_nan_device(C_nan);
    boba::matmat(csr, B_device, C_nan_device, alpha, zero, op);
    pass_or_fail(check, boba::norm_difference_inf(host_matrix_t(C_nan_device), C_overwrite_expected), tolerance);

    host_vector_t x({in_size}), y0({out_size});
    auto x_view = x.view();
    auto y0_view = y0.view();
    auto B_view = B.const_view();
    auto C0_view = C0.const_view();
    for (boba::index_t i = 0; i < in_size; ++i)
    {
      x_view(i) = B_view({i, 0});
    }
    for (boba::index_t i = 0; i < out_size; ++i)
    {
      y0_view(i) = C0_view({i, 0});
    }
    const auto y_expected = alpha * (op_dense * x) + beta * y0;

    boba::Vector<space, data_t> x_device(x), y_device(y0);
    csr.matvec(x_device, y_device, alpha, beta, op);
    pass_or_fail(check, boba::norm_difference_inf(host_vector_t(y_device), y_expected), tolerance);

    auto y_returned = boba::matvec(csr, x_device, op);
    pass_or_fail(check, boba::norm_difference_inf(host_vector_t(y_returned), op_dense * x), tolerance);

    auto C_returned = boba::matmat(csr, B_device, op);
    pass_or_fail(check, boba::norm_difference_inf(host_matrix_t(C_returned), op_dense * B), tolerance);

    // The naive kernel must agree with the vendor-library path.
    boba::Matrix<space, data_t> C_naive(C0);
    boba::sparse_matrix_matrix_product<space, data_t, true>(csr, B_device, C_naive, alpha, beta, op);
    pass_or_fail(check, boba::norm_difference_inf(host_matrix_t(C_naive), C_expected), tolerance);

    boba::Vector<space, data_t> y_naive(y0);
    boba::sparse_matrix_vector_product<space, data_t, true>(csr, x_device, y_naive, alpha, beta, op);
    pass_or_fail(check, boba::norm_difference_inf(host_vector_t(y_naive), y_expected), tolerance);
  }

  // Empty pattern: only the beta scaling applies.
  const boba::CSRMatrix<space, data_t> empty_csr(boba::CSRMatrix<boba::host_space, data_t>(rows, cols));
  host_vector_t x_empty({cols}), y_empty({rows});
  fill_with_value(x_empty, alpha);
  fill_with_value(y_empty, alpha);
  boba::Vector<space, data_t> x_empty_device(x_empty), y_empty_device(y_empty);
  empty_csr.matvec(x_empty_device, y_empty_device, alpha, beta);
  pass_or_fail(check, boba::norm_difference_inf(host_vector_t(y_empty_device), beta * y_empty), tolerance);

  // SpGEMM against the dense product, with an empty row and a rectangular right factor.
  constexpr boba::index_t inner_cols = 6;
  host_matrix_t dense_right({cols, inner_cols});
  dense_right.fill_with_zeros();
  auto dense_right_view = dense_right.view();
  for (boba::index_t i = 1; i < cols; ++i)
  {
    for (boba::index_t j = 0; j < inner_cols; ++j)
    {
      if ((2 * i + j) % 3 != 1)
      {
        dense_right_view({i, j}) = csr_test_value<data_t>(i * inner_cols + j + 2);
      }
    }
  }
  const boba::CSRMatrix<space, data_t> csr_right(boba::from_dense_csr<boba::host_space>(dense_right));
  const auto product_expected = dense * dense_right;
  const boba::CSRMatrix<boba::host_space, data_t> product(boba::sparse_matrix_sparse_matrix_product(csr, csr_right));
  pass_or_fail_bool(check, product.is_sorted());
  pass_or_fail(check, boba::norm_difference_inf(boba::to_dense(product), product_expected), tolerance);
  const boba::CSRMatrix<boba::host_space, data_t> product_naive(boba::sparse_matrix_sparse_matrix_product<space, data_t, true>(csr, csr_right));
  pass_or_fail_bool(check, product_naive.is_sorted());
  pass_or_fail(check, boba::norm_difference_inf(boba::to_dense(product_naive), product_expected), tolerance);

  const boba::CSRMatrix<space, data_t> empty_right(boba::CSRMatrix<boba::host_space, data_t>(cols, inner_cols));
  const auto empty_product = boba::sparse_matrix_sparse_matrix_product(csr, empty_right);
  pass_or_fail_bool(check, empty_product.nnz() == 0 && empty_product.rows() == rows && empty_product.cols() == inner_cols);

  std::cout << "CSR backend matvec/matmat/spgemm: done" << std::endl;
}

/**
 * \brief Exercises all initial sparse matrix formats against dense storage.
 */
int main()
{
  boba::init();

  bool check = true;

  boba::COOMatrix<boba::host_space, double> empty_coo;
  boba::CSRMatrix<boba::host_space, double> empty_csr;
  boba::ELLPACKMatrix<boba::host_space, double> empty_ellpack;
  boba::BCOOMatrix<boba::host_space, double> empty_bcoo;
  check_empty_sparse(empty_coo, check);
  check_empty_sparse(empty_csr, check);
  check_empty_sparse(empty_ellpack, check);
  check_empty_sparse(empty_bcoo, check);
  pass_or_fail_bool(check, empty_bcoo.nonzero_blocks() == 0);
  pass_or_fail_bool(check, empty_csr.is_sorted());

  // Cross-space copies and moves use the same host/device transfer semantics
  // as the dense tensor containers. Empty objects keep this test backend-neutral.
  constexpr auto device_space = boba::default_execution_space;
  boba::COOMatrix<device_space, double> coo_alternate(empty_coo);
  boba::COOMatrix<boba::host_space, double> coo_host(coo_alternate);
  coo_alternate = empty_coo;
  coo_host = std::move(coo_alternate);
  boba::CSRMatrix<device_space, double> csr_alternate(empty_csr);
  boba::CSRMatrix<boba::host_space, double> csr_host(csr_alternate);
  csr_alternate = empty_csr;
  csr_host = std::move(csr_alternate);
  boba::ELLPACKMatrix<device_space, double> ellpack_alternate(empty_ellpack);
  boba::ELLPACKMatrix<boba::host_space, double> ellpack_host(ellpack_alternate);
  ellpack_alternate = empty_ellpack;
  ellpack_host = std::move(ellpack_alternate);
  boba::BCOOMatrix<device_space, double> bcoo_alternate(empty_bcoo);
  boba::BCOOMatrix<boba::host_space, double> bcoo_host(bcoo_alternate);
  bcoo_alternate = empty_bcoo;
  bcoo_host = std::move(bcoo_alternate);

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
  auto coo = boba::from_dense<boba::COOMatrix<boba::host_space, double>>(dense, tolerance);
  auto csr = boba::from_dense<boba::CSRMatrix<boba::host_space, double>>(dense, tolerance);
  auto ellpack = boba::from_dense<boba::ELLPACKMatrix<boba::host_space, double>>(dense, tolerance);
  auto bcoo = boba::from_dense<boba::BCOOMatrix<boba::host_space, double>>(dense, 2, 2, tolerance);
  pass_or_fail_bool(check, coo.nnz() == 9);
  pass_or_fail_bool(check, csr.nnz() == 9);
  pass_or_fail_bool(check, csr.is_sorted());
  auto unsorted_csr = csr;
  auto unsorted_columns = unsorted_csr.m_column_indices.view();
  std::swap(unsorted_columns(0), unsorted_columns(1));
  pass_or_fail_bool(check, !unsorted_csr.is_sorted());

  // Mutable views update values in place; indices stay read-only.
  auto scaled_csr = csr;
  auto scaled_view = scaled_csr.view();
  static_assert(std::is_same_v<decltype(scaled_view.column_indices_data()), std::int64_t const*>);
  auto scaled_values = scaled_view.values_view();
  pass_or_fail_bool(check, scaled_values.size() == csr.nnz());
  for (boba::index_t k = 0; k < scaled_view.nnz(); ++k)
  {
    scaled_view.value(k) *= 2.0;
  }
  auto const csr_view = csr.const_view();
  bool view_matches = scaled_view.rows() == csr_view.rows() && scaled_view.cols() == csr_view.cols();
  for (boba::index_t i = 0; i < csr_view.rows(); ++i)
  {
    for (auto k = csr_view.row_begin(i); k < csr_view.row_end(i); ++k)
    {
      view_matches = view_matches && scaled_view.column_index(k) == csr_view.column_index(k);
      view_matches = view_matches && std::abs(scaled_values(k) - 2.0 * csr_view.value(k)) < 1.0e-14;
    }
  }
  pass_or_fail_bool(check, view_matches);
  pass_or_fail_bool(check, ellpack.nnz() == 12);
  pass_or_fail_bool(check, bcoo.nnz() == 12);
  pass_or_fail_bool(check, bcoo.nonzero_blocks() == 3);
  check_coo_round_trip(coo, check);
  check_csr_round_trip(csr, check);
  check_ellpack_round_trip(ellpack, check);
  check_bcoo_round_trip(bcoo, check);
  check_sparse(coo, dense_thresholded, x, y, z, "COO", check);
  check_sparse(csr, dense_thresholded, x, y, z, "CSR", check);
  check_sparse(ellpack, dense_thresholded, x, y, z, "ELLPACK", check);
  check_sparse(bcoo, dense_thresholded, x, y, z, "BCOO", check);
  check_csr_backend<double>(1.0e-13, check);
  check_csr_backend<float>(1.0e-5, check);
  // TODO<bugfix> does not build on cuda systems, see https://lc.llnl.gov/gitlab/boba/boba-oss/-/jobs/4086482
  // check_csr_backend<boba::complex<double>>(1.0e-13, check);

  const int result = final_check(check);
  boba::finalize();
  return result;
}

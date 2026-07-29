// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "common.hpp"
#include "common_ttm.hpp"

#include <BOBA/boba.hpp>
#include <iostream>
#include <stdio.h>

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

/*
   Tests for tensor train and tensor train matrix norms.
*/

int main(int argc, char* argv[])
{

  boba::splash();
  std::cout << "Tests for boba tensor train matrix Frobenius norm." << std::endl;
  boba::init();

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;
  checkpoint();

  size_t n = 6;
  size_t m = 6;
  size_t num = 2;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(n,
                             "-n",
                             "--rows",
                             "Number of rows to test.");

  args.add_optional_argument(m,
                             "-m",
                             "--columns",
                             "Number of columns to test.");

  args.parse_check();

  //
  // Test TTM compression
  //
  checkpoint();
  {
    boba::Matrix<::space, double> test_matrix({n * n, n * n});
    test_matrix.set_to_identity_matrix();

    boba::TensorTrainMatrix<2, space, double> test_ttm_A({n, n}, {n, n});
    test_ttm_A.rename("test_ttm_A");
    boba::TensorTrainMatrix<2, space, double> test_ttm_B({n, n}, {n, n});
    test_ttm_B.rename("test_ttm_B");

    checkpoint();
    boba::TicToc<tictoc_units> time_a;
    checkpoint();
    test_ttm_A.compress(test_matrix);
    time_a.end_and_print("compress ttm");

    checkpoint();
    boba::TicToc<tictoc_units> time_b;
    test_ttm_B.compress_via_tt(test_matrix);
    time_b.end_and_print("compress ttm via tt");

    pass_or_fail(check, boba::norm_difference_inf(test_ttm_A.decompress(), test_matrix), 1.0e-11);
    pass_or_fail(check, boba::norm_difference_inf(test_ttm_B.decompress(), test_matrix), 1.0e-11);
    pass_or_fail(check, boba::norm_difference_inf(test_ttm_A.decompress(), test_ttm_B.decompress()), 1.0e-11);
  }

  //
  // Test TTM add and round
  //
  checkpoint();
  {
    boba::Matrix<space, double> A({m, n});
    A.fill_with_random();

    boba::Matrix<space, double> B({m, n});
    B.fill_with_random();

    auto C = boba::make_ttm_from_matrices<3, space, double>({A, B, A}) +
             boba::make_ttm_from_matrices<3, space, double>({B, A, A}) +
             boba::make_ttm_from_matrices<3, space, double>({B, B, A});

    std::cout << "before rounding" << std::endl
              << "  rows  " << C.core_rows() << std::endl
              << "  cols  " << C.core_cols() << std::endl
              << "  ranks " << C.ranks_string() << std::endl;

    C.round();

    std::cout << "after rounding" << std::endl
              << "  rows  " << C.core_rows() << std::endl
              << "  cols  " << C.core_cols() << std::endl
              << "  ranks " << C.ranks_string() << std::endl;

    auto D = C.decompress();

    std::cout << "after unrolling" << std::endl
              << "  rows  " << D.rows() << std::endl
              << "  cols  " << D.cols() << std::endl;
  }

  //
  // Test TTM compression via operators
  //
  checkpoint();
  {
    // call helper function to generate
    common_ttm<2, space, double> operators(n, 1.0, false);

    boba::TensorTrainMatrix<2, space, double> test_ttm_C;
    test_ttm_C = operators.laplacian_interior;
    auto test_matrix = test_ttm_C.decompress();

    boba::TensorTrainMatrix<2, space, double> test_ttm_A({n, n}, {n, n});
    test_ttm_A.rename("test_ttm_A");
    boba::TensorTrainMatrix<2, space, double> test_ttm_B({n, n}, {n, n});
    test_ttm_B.rename("test_ttm_B");

    checkpoint();
    boba::TicToc<tictoc_units> time_a;
    checkpoint();
    test_ttm_A.compress(test_matrix);
    time_a.end_and_print("compress ttm");

    checkpoint();
    boba::TicToc<tictoc_units> time_b;
    test_ttm_B.compress_via_tt(test_matrix);
    time_b.end_and_print("compress ttm via tt");

    pass_or_fail(check, boba::norm_difference_inf(test_ttm_A.decompress(), test_matrix), 1.0e-11);
    pass_or_fail(check, boba::norm_difference_inf(test_ttm_B.decompress(), test_matrix), 1.0e-11);
    pass_or_fail(check, boba::norm_difference_inf(test_ttm_C.decompress(), test_matrix), 1.0e-11);
    pass_or_fail(check, boba::norm_difference_frobenius(test_ttm_A, test_ttm_B), 1.0e-11);
    pass_or_fail(check, boba::norm_difference_frobenius(test_ttm_A, test_ttm_C), 1.0e-11);
    pass_or_fail(check, boba::norm_difference_frobenius(test_ttm_B, test_ttm_C), 1.0e-11);
  }

  size_t constexpr dimension = 2;
  auto row_sizes = ::boba::filled_array<dimension>(n);
  auto col_sizes = ::boba::filled_array<dimension>(m);

  size_t N = ::boba::product(row_sizes);
  size_t M = ::boba::product(col_sizes);

  ::boba::Matrix<space, double> test_matrix1({N, M});
  auto test_matrix1_view = test_matrix1.view();

  ::boba::loop<space, 2>({N, M},
                         [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
  {
    size_t i = ij[0];
    size_t j = ij[1];
    test_matrix1_view({i, j}) = num;
  });
  double fnorm1_exact = ::boba::sqrt(num * num * N * M);

  ::boba::Matrix<space, double> test_matrix2({N, M});
  auto test_matrix2_view = test_matrix2.view();

  ::boba::loop<space, 2>({N, M},
                         [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
  {
    size_t i = ij[0];
    size_t j = ij[1];
    test_matrix2_view({i, j}) = i + 1;
  });
  double fnorm2_exact = ::boba::sqrt(double(M * N * (N + 1) * (2 * N + 1)) / double(6));

  ::boba::Matrix<space, double> test_matrix3({M, N});
  auto test_matrix3_view = test_matrix3.view();

  ::boba::loop<space, 2>({M, N},
                         [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
  {
    size_t i = ij[0];
    size_t j = ij[1];
    test_matrix3_view({i, j}) = ::boba::mod(i, 2);
  });
  double fnorm3_exact = ::boba::sqrt(::boba::floor(M / 2) * N);

  // Computing the norm and comparing it with the exact value for accuracy.
  double fnorm1 = ::boba::norm_frobenius(test_matrix1);
  double fnorm2 = ::boba::norm_frobenius(test_matrix2);
  double fnorm3 = ::boba::norm_frobenius(test_matrix3);

  pass_or_fail(check, fnorm1 - fnorm1_exact, 1.0e-13);
  pass_or_fail(check, fnorm2 - fnorm2_exact, 1.0e-13);
  pass_or_fail(check, fnorm3 - fnorm3_exact, 1.0e-13);

  boba::TensorTrainMatrix<dimension, space, double> test_ttm_1(row_sizes, col_sizes);
  boba::TensorTrainMatrix<dimension, space, double> test_ttm_2(row_sizes, col_sizes);
  boba::TensorTrainMatrix<dimension, space, double> test_ttm_3(col_sizes, row_sizes);

  test_ttm_1.compress(test_matrix1);
  test_ttm_2.compress(test_matrix2);
  test_ttm_3.compress(test_matrix3);

  double ttm_norm_1 = ::boba::norm_frobenius(test_ttm_1);
  double ttm_norm_2 = ::boba::norm_frobenius(test_ttm_2);
  double ttm_norm_3 = ::boba::norm_frobenius(test_ttm_3);

  pass_or_fail(check, ::boba::abs(ttm_norm_1 - fnorm1_exact), 1.0e-11);
  pass_or_fail(check, ::boba::abs(ttm_norm_2 - fnorm2_exact), 1.0e-11);
  pass_or_fail(check, ::boba::abs(ttm_norm_3 - fnorm3_exact), 1.0e-11);

  std::cout << " ----------------------------" << std::endl;
  std::cout << " Test ttm compress/unroll" << std::endl;
  std::cout << " ----------------------------" << std::endl;
  size_t Ipowmax = 3;

  for (size_t Ipow = 0; Ipow < Ipowmax; Ipow++)
  {
    size_t I = 6 * ::boba::pow(2, Ipow);
    size_t J = 8 * ::boba::pow(3, Ipow);
    size_t K = 2 * ::boba::pow(2, Ipow);

    size_t Ir = I + 1;
    size_t Jr = J + 1;
    size_t Kr = K + 1;

    {
      std::cout << "Testing sizes J = " << J << "  I = " << I << std::endl;
      ::boba::Matrix<space, double> full_matrix({Ir * Jr, I * J});
      full_matrix.set_to_identity_matrix();

      ::boba::TensorTrainMatrix<2, space, double> ttm({Jr, Ir}, {J, I});
      ttm.compress(full_matrix);

      auto ttm_decompress = ttm.decompress();
      boba_print(ttm.ranks_string());
      boba_print(ttm.compression_rate());
      pass_or_fail(check, ::boba::norm_difference_inf(ttm_decompress, full_matrix), 1.0e-09);

      ttm.compress(ttm_decompress);
      pass_or_fail(check, ::boba::norm_difference_inf(ttm.decompress(), full_matrix), 1.0e-09);

      auto ttm_decompress_2 = ttm.decompress();
      pass_or_fail(check, ::boba::norm_difference_inf(ttm_decompress_2, full_matrix), 1.0e-09);

      ::boba::TensorTrainSplitMatrix<2, space, double> sttm(ttm);
      boba_print(sttm.ranks_string());
      boba_print(sttm.compression_rate());
      pass_or_fail(check, ::boba::norm_difference_inf(sttm.decompress(), full_matrix), 1.0e-08);
    }
    if (Ipow == 0)
    {
      std::cout << "Testing sizes K = " << K << "  J = " << J << "  I = " << I << std::endl;
      ::boba::Matrix<space, double> full_matrix({Kr * Jr * Ir, K * J * I});
      full_matrix.set_to_identity_matrix();

      ::boba::TensorTrainMatrix<3, space, double> ttm({Kr, Jr, Ir}, {K, J, I});

      ttm.compress(full_matrix);
      boba_print(ttm.ranks_string());
      boba_print(ttm.compression_rate());
      auto ttm_decompress = ttm.decompress();
      pass_or_fail(check, ::boba::norm_difference_inf(ttm_decompress, full_matrix), 1.0e-09);

      ::boba::TensorTrainSplitMatrix<3, space, double> sttm(ttm);
      boba_print(sttm.ranks_string());
      boba_print(sttm.compression_rate());
      pass_or_fail(check, ::boba::norm_difference_inf(sttm.decompress(), full_matrix), 1.0e-07);
    }
    {
      std::cout << "Testing tensor products " << std::endl;
      ::boba::Matrix<space, double> matrix_A({Ir, I});
      ::boba::Matrix<space, double> matrix_B({Jr, J});
      matrix_B.set_to_identity_matrix();
      matrix_B.fill_row(2, 2.0);
      matrix_B.fill_col(3, -1.0);

      matrix_A.set_to_identity_matrix();
      matrix_A.fill_diagonal(1, -1.0);
      matrix_A.fill_diagonal(0, 2.0);
      matrix_A.fill_diagonal(-1, -1.0);

      auto full_matrix = ::boba::tensor_product(matrix_B, matrix_A);

      ::boba::TensorTrainMatrix<2, space, double> ttm({Jr, Ir}, {J, I});
      ttm.compress(full_matrix);

      auto ttm_from_matrices = ::boba::make_ttm_from_matrices<2, space, double>({matrix_B, matrix_A});
      pass_or_fail(check, ::boba::norm_difference_inf(ttm.decompress(), ttm_from_matrices.decompress()), 1.0e-09);

      auto ttm_decompress = ttm.decompress();
      boba_print(ttm.ranks_string());
      boba_print(ttm.compression_rate());
      pass_or_fail(check, ::boba::norm_difference_inf(ttm_decompress, full_matrix), 1.0e-09);

      ttm.compress(ttm_decompress);
      pass_or_fail(check, ::boba::norm_difference_inf(ttm.decompress(), full_matrix), 1.0e-09);

      auto ttm_decompress_2 = ttm.decompress();
      pass_or_fail(check, ::boba::norm_difference_inf(ttm_decompress_2, full_matrix), 1.0e-09);

      ::boba::TensorTrainSplitMatrix<2, space, double> sttm(ttm);
      boba_print(sttm.ranks_string());
      boba_print(sttm.compression_rate());
      pass_or_fail(check, ::boba::norm_difference_inf(sttm.decompress(), full_matrix), 1.0e-09);
    }
  }

  boba::finalize();
  return final_check(check);
}

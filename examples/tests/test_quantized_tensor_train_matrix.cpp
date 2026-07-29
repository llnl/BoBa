// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "BOBA/boba.hpp"
#include "common.hpp"

#include <cmath>
#include <iostream>

/*
  Test BoBa's QTTM functions
*/

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::execution_space host_space = boba::execution_space::CPU;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

/*
  This test driver therefore verifies:
    (a) compress/decompress roundtrip: decompress(compress(A)) ≈ A
    (b) identity ordering: decompress(I) is identity and matvec agrees with dense
    (c) round(): rounding preserves the represented matrix
    (d) GMRES sanity: solve I*x=b and check x≈b in dense space
*/

int main(int argc, char* argv[])
{
  if (boba::is_env_nonempty("VERBOSE"))
  {
    boba::splash();
    std::cout << "Test: QuantizedTensorTrainMatrix ordering + round + GMRES" << std::endl;
  }

  boba::init();
  BOBA_CALI_EXTERNAL_MARK

  bool check = true;

  // ----------------------------
  // Configurable parameters
  // ----------------------------
  size_t rows_base_per_core = 2;
  size_t cols_base_per_core = 2;
  size_t qtt_exponent_numcores = 3;
  double tolerance = 1.0e-14;

  double svd_tolerance_relative = 1.0e-14;
  double svd_tolerance_absolute = 1.0e-14;

  double acceptable_error_inf_norm = 1.0e-10;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(rows_base_per_core, "-r", "--rows_base", "Per-core row base (rows_base). Full rows = rows_base^exponent.");
  args.add_optional_argument(cols_base_per_core, "-c", "--cols_base", "Per-core col base (cols_base). Full cols = cols_base^exponent.");
  args.add_optional_argument(qtt_exponent_numcores, "-e", "--exponent", "Number of cores (exponent).");
  args.add_optional_argument(acceptable_error_inf_norm, "-t", "--tol", "Acceptable infinity-norm error for A - decompress(compress(A)).");

  args.parse_check();

  const bool verbose = boba::is_env_nonempty("VERBOSE");

  // ----------------------------
  // Construct object under test
  // ----------------------------
  ::boba::QuantizedTensorTrainMatrix<space, double>
    qtt_matrix_under_test(rows_base_per_core, cols_base_per_core, qtt_exponent_numcores);

  qtt_matrix_under_test.rename("qtt_matrix_under_test");
  qtt_matrix_under_test.svd_tolerance_relative = svd_tolerance_relative;
  qtt_matrix_under_test.svd_tolerance_absolute = svd_tolerance_absolute;
  qtt_matrix_under_test.svd_max_kept_values = ::boba::highest_value<size_t>();

  const size_t full_num_rows = qtt_matrix_under_test.rows();
  const size_t full_num_cols = qtt_matrix_under_test.cols();

  if (verbose)
  {
    std::cout << "% rows_base = " << rows_base_per_core << "\n";
    std::cout << "% cols_base = " << cols_base_per_core << "\n";
    std::cout << "% exponent  = " << qtt_exponent_numcores << "\n";
    std::cout << "% full size = " << full_num_rows << " x " << full_num_cols << "\n";
  }

  // =============================================================================
  // (1) ORDERING CHECK: compress -> unroll roundtrip
  // =============================================================================
  ::boba::Matrix<space, double> A({full_num_rows, full_num_cols});
  A.rename("A_dense");
  auto A_view = A.view();
  boba::loop<space, 2>(A.sizes(),
                       [=] __boba_host_device__(::boba::Array<size_t, 2> mid)
  {
    auto [i, j] = mid;
    A_view(mid) = 0.1 * double(i + 2 * j) + 0.01 * double((i * j) % 7);
  });

  checkpoint();
  boba::TicToc<tictoc_units> timer;

  timer.tic();
  qtt_matrix_under_test.compress(A);
  const double compress_time = timer.toc();
  checkpoint();

  checkpoint();
  timer.tic();
  auto A2 = qtt_matrix_under_test.decompress();
  const double unroll_time = timer.toc();
  checkpoint();

  const double roundtrip_error_inf = ::boba::norm_difference_inf(A2, A);

  std::cout << "compress/unroll roundtrip:\n";
  std::cout << "  ||A2 - A||_inf = " << std::scientific << roundtrip_error_inf << "\n";
  pass_or_fail(check, roundtrip_error_inf, acceptable_error_inf_norm);

  if (verbose)
  {
    std::cout << "%   compress: " << compress_time << "\n";
    std::cout << "%   unroll  : " << unroll_time << "\n";
    std::cout << "% Storage\n";
    std::cout << "%   stored elements = " << qtt_matrix_under_test.get_number_elements() << "\n";
    std::cout << "%   full elements   = " << qtt_matrix_under_test.get_full_size() << "\n";
    std::cout << "%   compression rate= " << qtt_matrix_under_test.compression_rate() << "x\n";
  }

  using qtt_vec_t = boba::QuantizedTensorTrain<space, double>;
  using qtt_host_vec_t = boba::QuantizedTensorTrain<host_space, double>;
  using qtt_mat_t = boba::QuantizedTensorTrainMatrix<space, double>;

  // =============================================================================
  // (2) ORDERING CHECK: identity unroll is identity + matvec matches unroll
  // =============================================================================
  {
    const size_t base = 2;
    const size_t exponent = 6; // n = 2^6 = 64
    const size_t n = boba::pow(base, exponent);

    qtt_mat_t I_qtt(base, base, exponent);
    I_qtt.set_to_identity_train();

    qtt_vec_t x_test(base, exponent);

    {
      qtt_host_vec_t x_test_host(base, exponent);
      x_test_host.fill_with_zeros();
      x_test_host.cores[0].resize({1_z, base, 1_z});
      x_test_host.cores[0]({0, 0, 0}) = 0.3;
      x_test_host.cores[0]({0, 1, 0}) = -0.7;
      for (size_t d = 1; d < exponent; ++d)
      {
        x_test_host.cores[d].resize({1_z, base, 1_z});
        x_test_host.cores[d]({0, 0, 0}) = 1.0;
        x_test_host.cores[d]({0, 1, 0}) = 0.5;
      }
      x_test = x_test_host;
    }

    auto y_qtt = I_qtt * x_test;

    auto I_dense = I_qtt.decompress();
    auto x_dense = x_test.decompress();
    auto y_dense_from_qtt = y_qtt.decompress();

    std::cout << "\nIDENTITY ordering test:\n";

    auto Ix = I_dense * x_dense;
    const double Ix_norm = ::boba::norm_frobenius(Ix);
    const double diff_norm = boba::norm_difference_frobenius(y_dense_from_qtt, Ix);
    const double rel_mv = diff_norm / Ix_norm;
    std::cout << "  matvec check rel = " << std::scientific << rel_mv << "\n";

    pass_or_fail(check, rel_mv, tolerance);
  }

  // =============================================================================
  // (3) ROUND TEST: does round preserve matrix (within tolerance)?
  // =============================================================================
  {
    const size_t rows_base = 2, cols_base = 2, exponent = 4; // 16x16
    boba::QuantizedTensorTrainMatrix<space, double> M_qtt(rows_base, cols_base, exponent);

    const size_t r = 3;

    for (size_t d = 0; d < exponent; ++d)
    {
      const size_t rl = (d == 0) ? 1 : r;
      const size_t rr = (d + 1 == exponent) ? 1 : r;

      M_qtt.cores[d].resize({rl, rows_base, cols_base, rr});
      auto core_view = M_qtt.cores[d].view();

      const size_t dd = d; // avoid capturing loop index directly in device lambda
      ::boba::Multiindexer<4> mider(M_qtt.cores[d].sizes());

      ::boba::loop<space, 1>(mider.size(),
                             [=] __boba_host_device__(size_t index)
      {
        auto [a, i, j, b] = mider.multiindex(index);
        core_view({a, i, j, b}) = 0.1 * double(1 + a + 2 * i + 3 * j + 4 * b + 5 * dd);
      });
    }

    auto M_before = M_qtt.decompress();
    const auto elems_before = M_qtt.get_number_elements();

    M_qtt.svd_tolerance_relative = 1e-14;
    M_qtt.svd_tolerance_absolute = 1e-14;
    M_qtt.svd_max_kept_values = 100000;
    M_qtt.round();

    auto M_after = M_qtt.decompress();
    const auto elems_after = M_qtt.get_number_elements();

    auto a1 = ::boba::norm_frobenius(M_before - M_after);
    auto rel = a1 / ::boba::norm_frobenius(M_before);

    std::cout << "\nQTT-matrix round() test:\n";
    std::cout << "  elements before: " << elems_before << "\n";
    std::cout << "  elements after : " << elems_after << "\n";
    std::cout << "  rel Frobenius(unroll(before)-unroll(after)) = " << std::scientific << rel << "\n";

    pass_or_fail(check, rel, 10 * tolerance);
  }

  // =============================================================================
  // (4) GMRES TEST (small, well-defined): A = I, solve Ax=b
  // =============================================================================
  {
    const size_t base = 2;
    const size_t exponent = 6;
    const size_t n = boba::pow(base, exponent);

    qtt_mat_t A_qtt(base, base, exponent);
    A_qtt.set_to_identity_train();

    // matvec consistency check for this A_qtt
    {
      qtt_vec_t x_test(base, exponent);

      {
        qtt_host_vec_t x_test_host(base, exponent);
        x_test_host.fill_with_zeros();
        x_test_host.cores[0].resize({1_z, base, 1_z});
        x_test_host.cores[0]({0, 0, 0}) = 0.3;
        x_test_host.cores[0]({0, 1, 0}) = -0.7;
        for (size_t d = 1; d < exponent; ++d)
        {
          x_test_host.cores[d].resize({1_z, base, 1_z});
          x_test_host.cores[d]({0, 0, 0}) = 1.0;
          x_test_host.cores[d]({0, 1, 0}) = 0.5;
        }
        x_test = x_test_host;
      }

      auto y_qtt = A_qtt * x_test;

      auto A_dense = A_qtt.decompress();
      auto x_dense = x_test.decompress();
      auto y_dense_from_qtt = y_qtt.decompress();

      auto Ax = A_dense * x_dense;
      const double Ax_norm = ::boba::norm_frobenius(Ax);
      const double diff_norm = boba::norm_difference_frobenius(y_dense_from_qtt, Ax);
      const double rel_mv = diff_norm / Ax_norm;
      std::cout << "\nQTT GMRES test (identity):\n";
      std::cout << "  n = " << n << "\n";
      std::cout << "  matvec check rel = " << std::scientific << rel_mv << "\n";
      pass_or_fail(check, rel_mv, 10 * tolerance);
    }

    // RHS b_qtt
    qtt_vec_t b_qtt(base, exponent);
    {
      qtt_host_vec_t b_qtt_host(base, exponent);
      b_qtt_host.fill_with_zeros();
      b_qtt_host.cores[0].resize({1_z, base, 1_z});
      b_qtt_host.cores[0]({0, 0, 0}) = 1.0;
      b_qtt_host.cores[0]({0, 1, 0}) = -0.5;
      for (size_t d = 1; d < exponent; ++d)
      {
        b_qtt_host.cores[d].resize({1_z, base, 1_z});
        b_qtt_host.cores[d]({0, 0, 0}) = 1.0;
        b_qtt_host.cores[d]({0, 1, 0}) = 1.0;
      }
      b_qtt = b_qtt_host;
    }

    qtt_vec_t x0_qtt(base, exponent);
    x0_qtt.fill_with_zeros();

    boba::Krylov<qtt_mat_t, qtt_vec_t> kryl(/*outer=*/1, /*inner=*/20, /*rel=*/1e-12);
    kryl.absolute_threshold = 1e-14;
    kryl.method = boba::KrylovMethods::gmres;
    kryl.verbose = true;
    kryl.set_matrix(A_qtt);

    qtt_vec_t x_qtt(base, exponent);
    kryl.solve(b_qtt, x0_qtt, x_qtt);

    auto b_dense = b_qtt.decompress();
    auto x_dense = x_qtt.decompress();

    // For vectors, Frobenius norm == Euclidean 2-norm.
    const double b_norm = ::boba::norm_frobenius(b_dense);
    const double norm_diff = boba::norm_difference_frobenius(x_dense, b_dense);

    const double rel_xb = norm_diff / b_norm;
    std::cout << "  dense check rel ||x-b||/||b|| = " << std::scientific << rel_xb << "\n";
    pass_or_fail(check, rel_xb, 10 * tolerance);
  }

  // =============================================================================
  // (5) Frobenius Norm TEST
  // =============================================================================
  {
    checkpoint();

    size_t base = 2;
    size_t exponent = 4;
    ::boba::QuantizedTensorTrainMatrix<space, double> FrobNormEx(base, base, exponent);

    auto R1 = 2_z;
    auto R2 = 3_z;
    auto R3 = 2_z;

    FrobNormEx.cores[0].resize({1, base, base, R1});
    FrobNormEx.cores[1].resize({R1, base, base, R2});
    FrobNormEx.cores[2].resize({R2, base, base, R3});
    FrobNormEx.cores[3].resize({R3, base, base, 1});

    checkpoint();
    for (size_t d = 0; d < exponent; d++)
    {
      FrobNormEx.cores[d].fill_with_random();
    }

    auto FrobNormEx_norm = ::boba::norm_frobenius(FrobNormEx);
    auto FrobNormEx_dense = FrobNormEx.decompress();
    auto FrobNormEx_dense_norm = ::boba::norm_frobenius(FrobNormEx_dense);

    auto rel = boba::abs(FrobNormEx_norm - FrobNormEx_dense_norm) / FrobNormEx_dense_norm;

    std::cout << "\nQTT-matrix norm_frobenius() test:\n";
    std::cout << "  rel (Frob(QTTM) - Frob(decompress(QTTM))/Frob(decompress(QTTM)) = " << std::scientific << rel << "\n";

    pass_or_fail(check, rel, tolerance);

    checkpoint();
  }

  checkpoint();
  boba::finalize();
  return final_check(check);
}

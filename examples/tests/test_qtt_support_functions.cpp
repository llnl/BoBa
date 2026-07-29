// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Test BoBa's QTT support functions
*/

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::microseconds;

int main(int argc, char* argv[])
{

  // Initialize the Boba library and display a splash screen
  boba::splash();
  boba::init();
  std::cout << "Tests for quantized tensor train support functions: QTT_FFT x I, QTT_iFFT x I, QTT_COLUMN_Inset, QTT_COLUMN_EXTRACT" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  bool check = true; // Flag to track test results

  checkpoint(); // Create a checkpoint for performance measurement

  // Define dimensions and parameters for the tensor train
  size_t d1 = 5;                   // Number of dimensions for the spatial tensor
  size_t d2 = 4;                   // Number of dimensions for the temporal tensor
  size_t base = 2;                 // Base for tensor size calculation
  size_t nx = boba::pow(base, d1); // Size of the spatial tensor
  size_t nt = boba::pow(base, d2); // Size of the temporal tensor
  size_t resolution = ::pow(base, d1 + d2);
  double tolerance = 1.0e-14;

  checkpoint();

  // Initialize a vector to hold the input data
  ::boba::Vector<space, double> x_vector({resolution});
  {
    auto x_view = x_vector.view(); // Create a view for the vector
    // Fill the vector with sine values
    ::boba::loop<space, 1>(resolution, [=] __boba_host_device__(size_t i)
    {
      x_view(i) = boba::sin((i * boba::pi) / resolution);
    });
  }

  // Compress the vector into a quantized tensor train format
  auto x_qtt = ::boba::compress_to_QuantizedTensorTrain(x_vector, base, tolerance);

  // Apply the Kronecker product of DFT and identity on the quantized tensor train
  auto w = boba::qtt_dft_kron_id(x_qtt, d1, d2, tolerance);

  // Prepare for the test (V x I)
  boba::Matrix<space, boba::complex<double>> Xmat({nx, nt});                   // Matrix to hold complex values
  auto x_rec = x_qtt.decompress();                                             // Unroll the quantized tensor train
  auto x_view = x_rec.view();                                                  // Create a view for the decompressed tensor
  ::boba::Vector<space, boba::complex<double>> x_vector_complex({resolution}); // Vector for complex values
  {
    auto x_c_view = x_vector_complex.view(); // Create a view for the complex vector
    // Fill the complex vector with the decompressed tensor values
    ::boba::loop<space, 1>(resolution, [=] __boba_host_device__(size_t i)
    {
      x_c_view(i) = boba::complex<double>{x_view(i), 0.0}; // Convert to complex
    });
  }
  Xmat.reshape(x_vector_complex); // Reshape the matrix to hold the complex vector

  // Compute the FFT of the matrix along the temporal dimension
  auto fft_of_Xmat = boba::fft_along_dimension<2, space, double>(Xmat, 1, boba::fft_operation::forward);

  // Create a new vector to hold the FFT results
  boba::Vector<space, boba::complex<double>> x_fft_vec({resolution});
  x_fft_vec.reshape(fft_of_Xmat); // Reshape to hold the FFT results

  // Compute the normalization scalar
  auto scalar = boba::complex<double>(1.0 / boba::sqrt(static_cast<double>(nt)), 0.0);

  // Calculate the difference between the computed and expected results
  auto diff = w.decompress() - scalar * x_fft_vec;
  auto checking = ::boba::norm_frobenius(diff) / double(resolution); // Compute the Frobenius norm

  const double error_tolerance = boba::is_gpu(space) ? 2.0e-2 : 10.0 * tolerance;

  // Check if the results pass the tolerance criteria
  pass_or_fail(check, checking, error_tolerance);

  // Test (V^{-1} x I)
  auto z = boba::qtt_idft_kron_id(x_qtt, d1, d2, tolerance);                                               // Apply the inverse DFT
  auto ifft_of_Xmat = boba::fft_along_dimension<2, space, double>(Xmat, 1, boba::fft_operation::backward); // Compute the IFFT

  // Create a vector to hold the IFFT results
  boba::Vector<space, boba::complex<double>> x_ifft_vec({resolution});
  x_ifft_vec.reshape(ifft_of_Xmat); // Reshape to hold the IFFT results

  // Compute the normalization scalar for the IFFT
  auto scalar2 = boba::complex<double>(boba::sqrt(static_cast<double>(nt)), 0.0);
  auto diff2 = z.decompress() - scalar2 * x_ifft_vec;            // Calculate the difference for IFFT
  auto checking2 = ::boba::norm_frobenius(diff2) / (resolution); // Compute the Frobenius norm for IFFT

  // Check if the IFFT results pass the tolerance criteria
  pass_or_fail(check, checking2, error_tolerance);

  // Test column extraction
  for (size_t j = 0; j < nt; ++j)
  {
    auto col_qtt = boba::qtt_column_extract(x_qtt, d1, d2, j, tolerance);
    auto col_vec = col_qtt.decompress();

    boba::Vector<space, boba::complex<double>> col_true({nx});
    {
      auto flat = x_vector_complex.view();
      auto cv = col_true.view();
      ::boba::loop<space, 1>(nx, [=] __boba_host_device__(size_t i)
      {
        cv(i) = flat(i + nx * j);
      });
    }

    auto diff_c = col_vec - col_true;
    auto denom = ::boba::norm_frobenius(col_true);
    double err = (denom > 0.0) ? (::boba::norm_frobenius(diff_c) / denom) : ::boba::norm_frobenius(diff_c);
    pass_or_fail(check, err, 10.0 * tolerance);
  }

  // Test column add-into
  // Reconstruct x_qtt by adding all columns into a zero tensor
  // ------------------------
  // start from zero tensor with same shape as x_qtt
  auto x_qtt_c = boba::to_complex(x_qtt);
  auto Arec = x_qtt_c;
  for (size_t k = 0; k < Arec.exponent; ++k)
  {
    Arec.cores[k].fill_with_zeros();
  }

  for (size_t j = 0; j < nt; ++j)
  {
    auto col_qtt = boba::qtt_column_extract(x_qtt, d1, d2, j, tolerance);
    Arec = boba::qtt_column_addinto(Arec, d1, d2, j, col_qtt, tolerance);
  }

  // Compare reconstructed tensor to original
  auto diffA = (Arec.decompress() - x_qtt_c.decompress());
  double relerr = ::boba::norm_frobenius(diffA) / ::boba::norm_frobenius(x_qtt.decompress());

  pass_or_fail(check, relerr, 10.0 * tolerance);

  // =============================================================================
  // Verify identity unroll is identity + matvec matches unroll
  // =============================================================================
  {
    using qtt_vec_t = boba::QuantizedTensorTrain<space, double>;
    using qtt_mat_t = boba::QuantizedTensorTrainMatrix<space, double>;

    const size_t _base = 2;
    const size_t _exponent = 6;

    qtt_mat_t I_qtt(_base, _base, _exponent);
    I_qtt.set_to_identity_train();

    qtt_vec_t x_test(_base, _exponent);
    for (size_t d = 0; d < _exponent; ++d)
    {
      x_test.cores[d].resize({1_z, _base, 1_z});
      x_test.cores[d].fill_with_random();
    }

    auto y_qtt = I_qtt * x_test;

    auto I_dense = I_qtt.decompress();
    auto x_dense = x_test.decompress();
    auto y_dense_from_qtt = y_qtt.decompress();
    auto Ax = I_dense * x_dense;

    std::cout << "\nIDENTITY ordering test:\n";

    const auto error = boba::norm_difference_frobenius(y_dense_from_qtt, Ax);
    pass_or_fail(check, error, tolerance);
  }
  // =============================================================================
  // Verify qtt Matrix product A kron B
  // =============================================================================
  {
    using qttm = boba::QuantizedTensorTrainMatrix<space, double>;
    using ten2 = boba::Tensor<2, space, double>;
    // Note hard limit on d = d1 + d2 <= 12 for the unroll currently set in BOBA
    const size_t d1_temp = 2;
    const size_t d2_temp = 3;
    const size_t r = 2; // internal TT rank for randomness

    // Build QTT matrix A
    // -------------------------
    qttm Aqtt(base, base, d1_temp);
    for (size_t k = 0; k < d1_temp; ++k)
    {
      const size_t rL = (k == 0) ? 1 : r;
      const size_t rR = (k == d1_temp - 1) ? 1 : r;
      Aqtt.cores[k].resize({rL, base, base, rR});
      Aqtt.cores[k].fill_with_random();
    }
    qttm Bqtt(base, base, d2_temp);
    for (size_t k = 0; k < d2_temp; ++k)
    {
      const size_t rL = (k == 0) ? 1 : r;
      const size_t rR = (k == d2_temp - 1) ? 1 : r;
      Bqtt.cores[k].resize({rL, base, base, rR});
      Bqtt.cores[k].fill_with_random();
    }
    // QTT "kron" by core concatenation
    auto Kqtt = tensor_product(Aqtt, Bqtt);

    // Unroll to dense
    auto A = Aqtt.decompress();
    auto B = Bqtt.decompress();
    auto K = Kqtt.decompress();

    auto Kref = boba::tensor_product<2, space, double>(A, B);

    // Compare
    auto diffKron = K - Kref;
    std::cout << "\n Test kron product test:\n";
    double relerrKron = ::boba::norm_frobenius(diffKron) / ::boba::norm_frobenius(Kref);

    boba_print(relerrKron);
    pass_or_fail(check, relerrKron, tolerance);
  }
  // =============================================================================
  // Verify qtt Matrix product A kron B via matvec identity
  // (A kron B)(x kron y) = (Ax) kron (By)
  // =============================================================================
  {
    using qttm = boba::QuantizedTensorTrainMatrix<space, double>;
    using qttv = boba::QuantizedTensorTrain<space, double>;

    const size_t rM = 2;
    const size_t rV = 2;

    // random QTT matrix A of size 2^d1 x 2^d1
    qttm Aqtt(base, base, d1);
    for (size_t k = 0; k < d1; ++k)
    {
      const size_t rL = (k == 0) ? 1 : rM;
      const size_t rR = (k == d1 - 1) ? 1 : rM;
      Aqtt.cores[k].resize({rL, base, base, rR});
      Aqtt.cores[k].fill_with_random();
    }

    // Similarly, random QTT matrix B size 2^d2 x 2^d2
    qttm Bqtt(base, base, d2);
    for (size_t k = 0; k < d2; ++k)
    {
      const size_t rL = (k == 0) ? 1 : rM;
      const size_t rR = (k == d2 - 1) ? 1 : rM;
      Bqtt.cores[k].resize({rL, base, base, rR});
      Bqtt.cores[k].fill_with_random();
    }

    // Kron matrix K exponent d1+d2
    auto Kqtt = tensor_product(Aqtt, Bqtt);

    // Some random vector (length 2^d1)
    qttv x(base, d1);
    for (size_t k = 0; k < d1; ++k)
    {
      const size_t rL = (k == 0) ? 1 : rV;
      const size_t rR = (k == d1 - 1) ? 1 : rV;
      x.cores[k].resize({rL, base, rR}); // vector core: (rL, n, rR)
      x.cores[k].fill_with_random();
    }

    // Another random vector y (length 2^d2)
    qttv y(base, d2);
    for (size_t k = 0; k < d2; ++k)
    {
      const size_t rL = (k == 0) ? 1 : rV;
      const size_t rR = (k == d2 - 1) ? 1 : rV;
      y.cores[k].resize({rL, base, rR});
      y.cores[k].fill_with_random();
    }

    // Form v = x kron y as QTT vector by core concatenation
    qttv v(base, d1 + d2);
    for (size_t k = 0; k < d1; ++k)
    {
      v.cores[k] = x.cores[k];
    }
    for (size_t k = 0; k < d2; ++k)
    {
      v.cores[d1 + k] = y.cores[k];
    }

    // Left side: (A kron B) (x kron y)
    auto lhs = Kqtt.apply(v);

    // Right side: (Ax) kron (By)
    auto Ax = Aqtt.apply(x);
    auto By = Bqtt.apply(y);

    qttv rhs(base, d1 + d2);
    for (size_t k = 0; k < d1; ++k)
    {
      rhs.cores[k] = Ax.cores[k];
    }
    for (size_t k = 0; k < d2; ++k)
    {
      rhs.cores[d1 + k] = By.cores[k];
    }

    // Compare in TT norm
    auto diffKronVec = lhs - rhs;
    diffKronVec.round(); // if available
    lhs.round();
    rhs.round();

    double denom = ::boba::norm_frobenius(rhs);
    std::cout << "\n Test kron product using vector properties test:\n";
    double relerrKronMatvec = (denom > 0.0) ? (::boba::norm_frobenius(diffKronVec) / denom) : ::boba::norm_frobenius(diff);

    boba_print(relerrKronMatvec);
    pass_or_fail(check, relerrKronMatvec, tolerance);
  }

  boba::finalize();

  return final_check(check);
}

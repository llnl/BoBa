// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::microseconds;

/**
 * @brief Computes the Discrete Fourier Transform (DFT) of a given vector.
 *
 * This function takes a vector of real numbers and computes its DFT, returning a vector of complex numbers.
 * The DFT is computed using the formula:
 *
 * \[
 * Y[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-2\pi i k n / N}
 * \]
 *
 * where \( Y[k] \) is the DFT output, \( x[n] \) is the input vector, \( N \) is the number of input samples,
 * and \( i \) is the imaginary unit.
 *
 * @param x A constant reference to a boba::Vector containing the input real values.
 *          The size of the vector must be greater than zero.
 *
 * @return A boba::Vector containing the DFT of the input vector, where each element is a complex number.
 * @note The output vector is scaled by \( \sqrt{1/N} \) to normalize the DFT result.
 */
boba::Vector<space, boba::complex<double>> dft(const boba::Vector<space, double>& x)
{
  const boba::complex<double> imag{0.0, 1.0};

  const size_t n = x.sizes(0);

  boba::Vector<space, boba::complex<double>> y({n});
  y.fill_with_zeros();

  const double theta = 2.0 * boba::pi / n;

  auto x_view = x.const_view();
  auto y_view = y.atomic_view();
  boba::loop<space, 2>({n, n},
                       [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
  {
    auto [i, j] = ij;
    const boba::complex<double> omega = boba::cos(i * j * theta) - imag * boba::sin(i * j * theta);
    y_view(i) += omega * boba::complex<double>{x_view(j), 0.0};
  });

  const boba::complex<double> scale{boba::sqrt(1.0 / static_cast<double>(n)), 0.0};
  y *= scale;

  return y;
}

/**
 * @brief Computes the Inverse Discrete Fourier Transform (IDFT) of a given vector.
 *
 * This function takes a vector of complex numbers representing the frequency domain
 * and computes its inverse DFT, returning a vector of complex numbers in the time domain.
 * The IDFT is computed using the formula:
 *
 * \[
 * x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{2\pi i k n / N}
 * \]
 *
 * where \( x[n] \) is the IDFT output, \( X[k] \) is the input vector of complex numbers,
 * \( N \) is the number of input samples, and \( i \) is the imaginary unit.
 *
 * @param x A constant reference to a boba::Vector containing the input complex values.
 *          The size of the vector must be greater than zero.
 *
 * @return A boba::Vector containing the IDFT of the input vector, where each element is a complex number.
 * @note The output vector is scaled by \( \frac{1}{N} \) to normalize the IDFT result.
 */
boba::Vector<space, boba::complex<double>> idft(const boba::Vector<space, boba::complex<double>>& x)
{
  const boba::complex<double> imag{0.0, 1.0};
  const size_t n = x.sizes(0);

  boba::Vector<space, boba::complex<double>> y({n});
  y.fill_with_zeros();

  const double theta = 2.0 * boba::pi / n;

  auto x_view = x.const_view();
  auto y_view = y.atomic_view();
  boba::loop<space, 2>({n, n},
                       [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
  {
    auto [i, j] = ij;
    const boba::complex<double> omega = boba::cos(i * j * theta) + imag * boba::sin(i * j * theta);
    y_view(i) += omega * x_view(j);
  });

  const boba::complex<double> scale{boba::sqrt(1.0 / n), 0.0};
  y *= scale;

  return y;
}

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  std::cout << "Tests for quantized tensor train Fast Fourier Transform (FFT) and Inverse Fast Fourier Transform  implementation" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  bool check = 1;

  checkpoint();

  size_t exponent = boba::is_boba_debug_mode() ? 3 : 12;

  size_t base = 2;
  size_t resolution = ::pow(base, exponent);
  double tolerance = 1.0e-15;

  checkpoint();

  ::boba::Vector<space, double> x_vector({resolution});
  {
    auto x_view = x_vector.view();
    ::boba::loop<space, 1>(resolution, [=] __boba_host_device__(size_t i)
    {
      x_view(i) = boba::sin((i * boba::pi) / resolution);
    });
  }
  ::boba::Vector<space, boba::complex<double>> x_vector_complex({resolution});
  {
    auto x_c_view = x_vector_complex.view();
    ::boba::loop<space, 1>(resolution, [=] __boba_host_device__(size_t i)
    {
      x_c_view(i) = boba::complex<double>{boba::sin((i * boba::pi) / resolution), 0.0};
    });
  }

  auto y_vector = dft(x_vector);
  auto z_vector = idft(x_vector_complex);
  auto x_qtt = ::boba::compress_to_QuantizedTensorTrain(x_vector, base, tolerance);
  auto fx = boba::fft(x_qtt, tolerance);
  auto ifx = boba::ifft(x_qtt, tolerance);
  auto iffx = boba::ifft(fx, tolerance); // The output is complex
  auto fifx = boba::fft(ifx, tolerance); // The output is complex

  // Checking accuracy of fft
  auto fft_minus_dft = ::boba::norm_frobenius(y_vector - fx.decompress()) / resolution;
  // Checking if iff is the inverse of fft x
  auto x_minus_ifft_fft_x = ::boba::norm_frobenius(x_vector_complex - iffx.decompress()) / resolution;
  // Checking if fft is the inverse of ifft
  auto x_minus_fft_ifft_x = ::boba::norm_frobenius(x_vector_complex - fifx.decompress()) / resolution;
  // Checking accuracy of ifft
  auto ifft_minus_idft = ::boba::norm_frobenius(z_vector - ifx.decompress()) / resolution;

  const double error_tolerance = boba::is_gpu(space) ? 2.0e-2 : 10.0 * tolerance;

  pass_or_fail(check, fft_minus_dft, error_tolerance);
  pass_or_fail(check, x_minus_ifft_fft_x, error_tolerance);
  pass_or_fail(check, x_minus_fft_ifft_x, error_tolerance);
  pass_or_fail(check, ifft_minus_idft, error_tolerance);

  boba::finalize();

  return final_check(check);
}

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cassert>
#include <vector>

#ifdef BOBA_HIP_LIBS
#include <hipfft/hipfft.h>
#endif

namespace boba
{

#ifdef BOBA_HIP_LIBS

#define hipfft_assert(a) hip_fft_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a hipFFT error and terminates.
 * @param error hipFFT error code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void hip_fft_assert_(
  hipfftResult error,
  const std::string& call,
  int line,
  const std::string& function,
  const std::string& file)
{
  if (error == HIPFFT_SUCCESS)
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;
  exit(1);
}

/**
 * @brief Executes a batched complex double FFT on HIP tensors.
 * @tparam dimension Tensor rank.
 * @tparam index_t Index type.
 * @param in Input tensor batch.
 * @param out Output tensor batch.
 * @param batch_size Number of transforms in the batch.
 * @param is_inverse Selects inverse instead of forward transform.
 */
template <size_t dimension>
void hip_fft(
  Tensor<dimension, execution_space::HIP, complex<double>>& in,
  Tensor<dimension, execution_space::HIP, complex<double>>& out,
  index_t batch_size,
  bool is_inverse)
{
  auto mode_size = static_cast<int>(in.sizes(0));

  hipfftHandle plan;

  // https://rocm.docs.amd.com/projects/hipFFT/en/latest/conceptual/overview.html#advanced-hipfft-usage

  hipfft_assert(
    hipfftPlanMany(
      &plan,
      1,          // The number of dimensions for the FFT (1D, 2D, or 3D)
      &mode_size, // Array specifying the size of the FFT in each dimension.
      &mode_size, // inembed: The dimensions of the input data layout in memory.
      1,          // istride: Stride between elements in the input array.
      mode_size,  // idist: Distance between consecutive FFTs in the input array.
      &mode_size, // onembed: The dimensions of the output data layout in memory.
      1,          // ostride: Stride between elements in the output array
      mode_size,  // odist: Distance between consecutive FFTs in the output array.
      HIPFFT_Z2Z,
      batch_size));

  auto direction = is_inverse ? HIPFFT_BACKWARD : HIPFFT_FORWARD;

  hipfftExecZ2Z(plan, in.data(), out.data(), direction);

  if (is_inverse)
  {
    out /= complex<double>{static_cast<double>(mode_size), 0.0};
  }

  hipfftDestroy(plan);
}

/**
 * @brief Executes a batched complex float FFT on HIP tensors.
 * @tparam dimension Tensor rank.
 * @tparam index_t Index type.
 * @param in Input tensor batch.
 * @param out Output tensor batch.
 * @param batch_size Number of transforms in the batch.
 * @param is_inverse Selects inverse instead of forward transform.
 */
template <size_t dimension>
void hip_fft(
  Tensor<dimension, execution_space::HIP, complex<float>>& in,
  Tensor<dimension, execution_space::HIP, complex<float>>& out,
  index_t batch_size,
  bool is_inverse)
{
  auto mode_size = static_cast<int>(in.sizes(0));

  hipfftHandle plan;

  // https://rocm.docs.amd.com/projects/hipFFT/en/latest/conceptual/overview.html#advanced-hipfft-usage

  hipfft_assert(
    hipfftPlanMany(
      &plan,
      1,          // The number of dimensions for the FFT (1D, 2D, or 3D)
      &mode_size, // Array specifying the size of the FFT in each dimension.
      &mode_size, // inembed: The dimensions of the input data layout in memory.
      1,          // istride: Stride between elements in the input array.
      mode_size,  // idist: Distance between consecutive FFTs in the input array.
      &mode_size, // onembed: The dimensions of the output data layout in memory.
      1,          // ostride: Stride between elements in the output array
      mode_size,  // odist: Distance between consecutive FFTs in the output array.
      HIPFFT_C2C,
      batch_size));

  auto direction = is_inverse ? HIPFFT_BACKWARD : HIPFFT_FORWARD;

  hipfftExecC2C(plan, in.data(), out.data(), direction);

  if (is_inverse)
  {
    out /= complex<float>{static_cast<float>(mode_size), 0.0f};
  }

  hipfftDestroy(plan);
}

#else

/**
 * @brief Reports that HIP FFT support is unavailable for complex double tensors.
 * @tparam dimension Tensor rank.
 * @tparam index_t Index type.
 * @param in Input tensor batch.
 * @param out Output tensor batch.
 * @param batch_size Number of transforms in the batch.
 * @param is_inverse Selects inverse instead of forward transform.
 */
template <size_t dimension>
void hip_fft(
  Tensor<dimension, execution_space::HIP, complex<double>>& in,
  Tensor<dimension, execution_space::HIP, complex<double>>& out,
  index_t batch_size,
  bool is_inverse)
{
  detail::ignore(in);
  detail::ignore(out);
  detail::ignore(batch_size);
  detail::ignore(is_inverse);
  boba_error("Hip is not enabled");
}

/**
 * @brief Reports that HIP FFT support is unavailable for complex float tensors.
 * @tparam dimension Tensor rank.
 * @tparam index_t Index type.
 * @param in Input tensor batch.
 * @param out Output tensor batch.
 * @param batch_size Number of transforms in the batch.
 * @param is_inverse Selects inverse instead of forward transform.
 */
template <size_t dimension>
void hip_fft(
  Tensor<dimension, execution_space::HIP, complex<float>>& in,
  Tensor<dimension, execution_space::HIP, complex<float>>& out,
  index_t batch_size,
  bool is_inverse)
{
  detail::ignore(in);
  detail::ignore(out);
  detail::ignore(batch_size);
  detail::ignore(is_inverse);
  boba_error("Hip is not enabled");
}
#endif

} // namespace boba

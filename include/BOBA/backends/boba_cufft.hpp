// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cassert>
#include <vector>

#ifdef BOBA_CUDA_LIBS
#include <cufft.h>
#endif

namespace boba
{

#ifdef BOBA_CUDA_LIBS

/**
 * @brief Executes a batched complex double FFT on CUDA tensors.
 * @tparam dimension Tensor rank.
 * @tparam index_t Index type.
 * @param in Input tensor batch.
 * @param out Output tensor batch.
 * @param batch_size Number of transforms in the batch.
 * @param is_inverse Selects inverse instead of forward transform.
 */
template <size_t dimension>
void cuda_fft(
  Tensor<dimension, execution_space::CUDA, complex<double>>& in,
  Tensor<dimension, execution_space::CUDA, complex<double>>& out,
  index_t batch_size,
  bool is_inverse)
{
  auto mode_size = static_cast<int>(in.sizes(0));
  cufftHandle plan;
  cufftPlanMany(
    &plan,
    1,          // The number of dimensions for the FFT (1D, 2D, or 3D)
    &mode_size, // Array specifying the size of the FFT in each dimension.
    &mode_size, // inembed: The dimensions of the input data layout in memory.
    1,          // istride: Stride between elements in the input array.
    mode_size,  // idist: Distance between consecutive FFTs in the input array.
    &mode_size, // onembed: The dimensions of the output data layout in memory.
    1,          // ostride: Stride between elements in the output array
    mode_size,  // odist: Distance between consecutive FFTs in the output array.
    CUFFT_Z2Z,  // type
    batch_size);

  auto direction = is_inverse ? CUFFT_INVERSE : CUFFT_FORWARD;

  cufftExecZ2Z(plan, in.data(), out.data(), direction);

  if (is_inverse)
  {
    out /= complex<double>{static_cast<double>(mode_size), 0.0};
  }

  cufftDestroy(plan);
}

/**
 * @brief Executes a batched complex float FFT on CUDA tensors.
 * @tparam dimension Tensor rank.
 * @tparam index_t Index type.
 * @param in Input tensor batch.
 * @param out Output tensor batch.
 * @param batch_size Number of transforms in the batch.
 * @param is_inverse Selects inverse instead of forward transform.
 */
template <size_t dimension>
void cuda_fft(
  Tensor<dimension, execution_space::CUDA, complex<float>>& in,
  Tensor<dimension, execution_space::CUDA, complex<float>>& out,
  index_t batch_size,
  bool is_inverse)
{
  auto mode_size = static_cast<int>(in.sizes(0));
  cufftHandle plan;
  cufftPlanMany(
    &plan,
    1,          // The number of dimensions for the FFT (1D, 2D, or 3D)
    &mode_size, // Array specifying the size of the FFT in each dimension.
    &mode_size, // inembed: The dimensions of the input data layout in memory.
    1,          // istride: Stride between elements in the input array.
    mode_size,  // idist: Distance between consecutive FFTs in the input array.
    &mode_size, // onembed: The dimensions of the output data layout in memory.
    1,          // ostride: Stride between elements in the output array
    mode_size,  // odist: Distance between consecutive FFTs in the output array.
    CUFFT_C2C,  // type
    batch_size);

  auto direction = is_inverse ? CUFFT_INVERSE : CUFFT_FORWARD;

  cufftExecC2C(plan, in.data(), out.data(), direction);

  if (is_inverse)
  {
    out /= complex<float>{static_cast<float>(mode_size), 0.0f};
  }

  cufftDestroy(plan);
}

#else

/**
 * @brief Reports that CUDA FFT support is unavailable for complex double tensors.
 * @tparam dimension Tensor rank.
 * @tparam index_t Index type.
 * @param in Input tensor batch.
 * @param out Output tensor batch.
 * @param batch_size Number of transforms in the batch.
 * @param is_inverse Selects inverse instead of forward transform.
 */
template <size_t dimension>
void cuda_fft(
  Tensor<dimension, execution_space::CUDA, complex<double>>& in,
  Tensor<dimension, execution_space::CUDA, complex<double>>& out,
  index_t batch_size,
  bool is_inverse)
{
  detail::ignore(in);
  detail::ignore(out);
  detail::ignore(batch_size);
  detail::ignore(is_inverse);
  boba_error("Cuda is not enabled");
}

/**
 * @brief Reports that CUDA FFT support is unavailable for complex float tensors.
 * @tparam dimension Tensor rank.
 * @tparam index_t Index type.
 * @param in Input tensor batch.
 * @param out Output tensor batch.
 * @param batch_size Number of transforms in the batch.
 * @param is_inverse Selects inverse instead of forward transform.
 */
template <size_t dimension>
void cuda_fft(
  Tensor<dimension, execution_space::CUDA, complex<float>>& in,
  Tensor<dimension, execution_space::CUDA, complex<float>>& out,
  index_t batch_size,
  bool is_inverse)
{
  detail::ignore(in);
  detail::ignore(out);
  detail::ignore(batch_size);
  detail::ignore(is_inverse);
  boba_error("Cuda is not enabled");
}

#endif

} // namespace boba

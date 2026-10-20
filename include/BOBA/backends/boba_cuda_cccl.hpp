// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/backends/boba_cuda.hpp"

#ifdef BOBA_CUDA
#include <cub/device/device_reduce.cuh>
#endif

namespace boba
{

namespace detail
{

/**
 * @brief Computes a device-wide sum with NVIDIA CCCL/CUB.
 * @param data Device-accessible input data.
 * @param length Number of input elements.
 * @return Sum of the input elements.
 */
template <typename data_t, typename index_t>
inline data_t cuda_cccl_sum_reduce(const data_t* data, index_t length)
{
#ifdef BOBA_CUDA
  if (length == 0)
  {
    return data_t{};
  }

  data_t* d_output = nullptr;
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cuda_assert(cudaMalloc(&d_output, sizeof(data_t)));

  cuda_assert(cub::DeviceReduce::Sum(
    d_temp_storage,
    temp_storage_bytes,
    data,
    d_output,
    length));
  cuda_assert(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cuda_assert(cub::DeviceReduce::Sum(
    d_temp_storage,
    temp_storage_bytes,
    data,
    d_output,
    length));

  data_t result{};
  cuda_assert(cudaMemcpy(&result, d_output, sizeof(data_t), cudaMemcpyDefault));
  cuda_assert(cudaFree(d_temp_storage));
  cuda_assert(cudaFree(d_output));
  return result;
#else
  ::boba::detail::ignore(data, length);
  return data_t{};
#endif
}

} // namespace detail
} // namespace boba

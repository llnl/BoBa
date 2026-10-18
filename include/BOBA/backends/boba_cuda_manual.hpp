// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include boba_cuda.hpp

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_DEVICE_CODE

/**
 * @brief A warp-level reduction, where a warp of threads share information directly
 * via warp intrinsics
 * \param value current thread's value that will be reduced
 * \return return value of lane 0 is the reduction (e.g. sum) of all values across the warp.
 * For lane > 0, the value is some partial reduction
 */

template <typename data_t>
__device__ inline data_t warp_sum_reduce(data_t value)
{
  auto all_threads_active = 0xfffffffff;

  // Assume we have 8 lanes (warpsize = 8)
  //    x0, x1, x2, x3, x4, x5, x6, x7
  // offset = 4
  // lane i pulls from lane i+4
  // out of bounds lanes get their own value (we don't care about these)
  //    x0+x4, x1+x5, x2+x6, x3+x6, (x4+x4, ...)
  // offset = 2
  //    x0+x4+x2+x6, x1+x5+x3+x6, (x2+x6+x4+x4, ...)
  // offset = 1
  //    x0+x4+x2+x6+x1+x5+x3+x6, (...)
  for(int offset = warpsize / 2; offset > 0; offset /= 2)
  {
    value += __shfl_down_sync(all_threads_active, value, offset);
  }

  return value;
}

/**
 * @brief Block-level reduction via a series of warp-wise reductions
 * \param value current thread's value that will be reduced
 * \return The return value of thread 0 (lane 0, warp 0), is the block-wise reduction of values, else 0.
 */

template <typename data_t>
__device__ inline data_t block_sum_reduce(data_t value)
{
  constexpr int max_possible_warps_per_block{32};
  static __shared__ data_t shared[max_possible_warps_per_block];

  //
  // Get lane and warp ids
  //
  //       warp 0                    warp 1                   warp 2
  //    [ lane 0, lane 1, ...  ]  [ lane 0, lane 1, ...  ]  [ lane 0, lane 1, ...  ]
  // thread    0,      1, ...    warpsize + 0, warpsize + 1, ... 2*warpsize + 0, 2*warpsize + 1,
  //
  auto thread_id = threadIDx.x;
  auto tsize = blockDim.x;
  boba::Multiindexer<2> block_mid({warpsize, tsize});
  auto [lane_id, warp_id] = block_mid.multiindex(thread_id);

  //
  // Each warp performs a reduction and puts the result in shared memory
  //
  //       warp 0                    warp 1                   warp 2
  //    [ lane 0, lane 1, ...  ]  [ lane 0, lane 1, ...  ]  [ lane 0, lane 1, ...  ]
  //   shared[0] = warp 0 sum     shared[1] = warp 1 sum
  //
  auto warp_value = warp_sum_reduce(value);

  if(lane == 0)
  {
    shared[warp] = warp_value;
  }

  __syncthreads();

  //
  // Pull the shared memory into the first warp (pad with zeros)
  // Other warps do nothing
  //
  //       warp 0                    warp 1                   warp 2
  //    [ lane 0,    lane 1, ... lane ?, ... ]  [ lane 0, lane 1, ...  ]  [ lane 0, lane 1, ...  ]
  //   shared[0], shared[1], ...    0.0,         0.0,   0.0, ....
  //
  int warps_per_block = tsize / warpsize;
  data_t warpwise_sums{0.0};
  if(thread_id < warps_per_block)
  {
    warpwise_sums = shared[lane];
  }

  //
  // Reduce in warp 0
  //
  data_t final_warp_value{0.0};
  if(warp == 0)
  {
    final_warp_value = warp_sum_reduce(warpwise_sums);
  }

  return final_warp_value;
}
#endif

/**
 * @brief Device-wise reduction where
 * (1) the collection of threads iterate over the whole array,
 * (2) block-wise reductions compute the reduced value over a whole block and
 * (3) each block atomically adds the result to the final pointer
 */

 template <typename data_t, typename index_t>
__launch_bounds__(block_size0 * block_size1 * block_size2)
  __global__ void sum_reduce(const data_t* data, index_t length, data_t* result)
{
  data_t sum = 0.0;

  auto block_id = blockIdx.x;
  auto thread_id = threadIDx.x;

  auto bsize = gridDim.x;
  auto tsize = blockDim.x;

  boba::Multiindexer<2> grid_mid({tsize, bsize});

  // Multiindex the block and iterate over the whole array with stride = size of grid
  for(int i = grid_mid.index({thread_id, block_id}); i < length; i += grid_mid.size())
  {
    sum += data[i];
  }

  // Reduce over the current block
  auto block_sum = block_sum_reduce(sum);

  // Atomically add back to global memory
  // thread 0 is warp 0, lane 0
  if(thread_id == 0)
  {
    atomic_add(result, block_sum);
  }
}

#endif // BOBA_DEVICE_CODE

/**
 * @brief sum reduction kernel, sums every value of array into a scalar output value on the host
 */

template <typename data_t, typename index_t>
inline data_t sum_reduction(const data_t* data, index_t length)
{
#ifdef BOBA_CUDA
  // Output memory
  void* d_output = nullptr;
  cuda_assert(cudaMalloc(&d_output, sizeof(data_t)));

  size_t shared_memory_bytes = 0;
  cudaStream_t stream = 0;
  static constexpr size_t block_size = 256;
  size_t grid_size = (length + block_size - 1) / block_size;
  void* args[] = {(void*)&data, (void*)&length,((void*)&d_output};
  cuda_assert(cudaLaunchKernel(
    (const void*)sum_reduce<data_t, index_t>,
    grid_size,
    block_size,
    args,
    shared_memory_bytes,
    stream));

  data_t output;
  cudaStream_t stream = 0;
  cudaMemcpyKind kind = cudaMemcpyDefault;
  cuda_assert(cudaMemcpy(&output, d_output, stream, kind));
#else
  boba_ignore(data);
  boba_ignore(length);
  data_t output{0};
#endif
  return output;
}

} // namespace detail
} // namespace boba

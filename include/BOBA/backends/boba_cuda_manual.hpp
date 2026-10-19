// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/backends/boba_cuda.hpp"

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

constexpr size_t reduction_block_size = 512;
constexpr size_t atomic_cutoff = 2e9;//reduction_block_size * 2;

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

  // Assume we have 8 lanes (warpSize = 8)
  //    x0, x1, x2, x3, x4, x5, x6, x7
  // offset = 4
  // lane i pulls from lane i+4
  // out of bounds lanes get their own value (we don't care about these)
  //    x0+x4, x1+x5, x2+x6, x3+x6, (x4+x4, ...)
  // offset = 2
  //    x0+x4+x2+x6, x1+x5+x3+x6, (x2+x6+x4+x4, ...)
  // offset = 1
  //    x0+x4+x2+x6+x1+x5+x3+x6, (...)
  for(int offset = warpSize / 2; offset > 0; offset /= 2)
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
  // thread    0,      1, ...    warpSize + 0, warpSize + 1, ... 2*warpSize + 0, 2*warpSize + 1,
  //
  auto thread_id = threadIdx.x;
  auto tsize = blockDim.x;
  boba::Multiindexer<2> block_mid({warpSize, tsize});
  auto [lane_id, warp_id] = block_mid.multiindex(thread_id);

  //
  // Each warp performs a reduction and puts the result in shared memory
  //
  //       warp 0                    warp 1                   warp 2
  //    [ lane 0, lane 1, ...  ]  [ lane 0, lane 1, ...  ]  [ lane 0, lane 1, ...  ]
  //   shared[0] = warp 0 sum     shared[1] = warp 1 sum
  //
  auto warp_value = warp_sum_reduce(value);

  if(lane_id == 0)
  {
    shared[warp_id] = warp_value;
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
  int warps_per_block = tsize / warpSize;
  data_t warpwise_sums{0.0};
  if(thread_id < warps_per_block)
  {
    warpwise_sums = shared[lane_id];
  }

  //
  // Reduce in warp 0
  //
  data_t final_warp_value{0.0};
  if(warp_id == 0)
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

 template <size_t block_size, typename data_t, typename index_t, bool force_atomics>
__launch_bounds__(block_size)
  __global__ void cuda_sum_reduce(const data_t* data, index_t length, data_t* result)
{
#ifdef BOBA_DEVICE_CODE
  data_t sum = 0.0;

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;

  int bsize = gridDim.x;
  int tsize = blockDim.x;

  boba::Multiindexer<2> grid_mid({tsize, bsize});

  auto thread_offset = grid_mid.index({thread_id, block_id});
  auto grid_size = grid_mid.size();

  // Multiindex the block and iterate over the whole array with stride = size of grid
#if 1
  data_t sum0{0}, sum1{0}, sum2{0}, sum3{0};

  auto safe_access = [=](auto id) { return (id < length) ? data[id] : data_t{0.0}; };

  auto safe_iterations = length / (grid_size * 4);
  auto upper_bound = safe_iterations * (grid_size * 4);

  for(int i = 0; i < upper_bound; i += grid_size * 4)
  {
    sum0 += data[i + thread_offset + 0*grid_size];
    sum1 += data[i + thread_offset + 1*grid_size];
    sum2 += data[i + thread_offset + 2*grid_size];
    sum3 += data[i + thread_offset + 3*grid_size];
  }
  for(int i = upper_bound; i < length; i += grid_size * 4)
  {
    sum0 += safe_access(i + thread_offset + 0*grid_size);
    sum1 += safe_access(i + thread_offset + 1*grid_size);
    sum2 += safe_access(i + thread_offset + 2*grid_size);
    sum3 += safe_access(i + thread_offset + 3*grid_size);
  }
  sum = sum0 + sum1 + sum2 + sum3;
#else
  for(int i = thread_offset; i < length; i += grid_size)
  {
    sum += data[i];
  }
#endif

  // Reduce over the current block
  auto block_sum = block_sum_reduce(sum);

  // Atomically add back to global memory
  // thread 0 is warp 0, lane 0
  if(thread_id == 0)
  {
    if((length < atomic_cutoff) or (force_atomics))
    {
      atomicAdd(result, block_sum);
    }
    else
    {
      result[block_id] = block_sum;
    }
  }
#endif
}

/**
 * @brief sum reduction kernel, sums every value of array into a scalar output value on the host
 */

template <typename data_t, typename index_t>
inline data_t sum_reduction_final(const data_t* data, index_t length)
{
#ifdef BOBA_CUDA
  // Output memory
  void* d_output = nullptr;
  cuda_assert(cudaMalloc(&d_output, sizeof(data_t)));
  cudaStream_t stream = 0;
  cuda_assert(cudaMemsetAsync(d_output, 0, sizeof(data_t), stream));

  size_t shared_memory_bytes = 0;
  static constexpr size_t block_size = 512;
  size_t grid_size = block_size * 64;
  void* args[] = {(void*)&data, (void*)&length, (void*)&d_output};
  cuda_assert(cudaLaunchKernel(
    (const void*)cuda_sum_reduce<block_size, data_t, index_t, true>,
    grid_size,
    block_size,
    args,
    shared_memory_bytes,
    stream));

  data_t output;
  cudaMemcpyKind kind = cudaMemcpyDefault;
  cuda_assert(cudaMemcpy(&output, d_output, sizeof(data_t), kind));

  cuda_assert(cudaFree(d_output));
#else
  boba_ignore(data);
  boba_ignore(length);
  data_t output{0};
#endif
  return output;
}

template <size_t block_size, typename data_t, typename index_t>
inline void sum_reduction_workspace(index_t grid_size, const data_t* data, index_t length, data_t* d_output)
{
#ifdef BOBA_CUDA
  size_t shared_memory_bytes = 0;
  cudaStream_t stream = 0;
  void* args[] = {(void*)&data, (void*)&length, (void*)&d_output};
  cuda_assert(cudaLaunchKernel(
    (const void*)cuda_sum_reduce<block_size, data_t, index_t, false>,
    grid_size,
    block_size,
    args,
    shared_memory_bytes,
    stream));
#else
  boba_ignore(data);
  boba_ignore(length);
  boba_ignore(output);
#endif
}


template <typename data_t, typename index_t>
inline data_t sum_reduction_driver(const data_t* data, index_t length)
{
  static int reductions = 0;

  printf("length %lu  atomic_cutoff %lu\n", length, atomic_cutoff);

  if(length < atomic_cutoff)
  {
    if(reductions > 1)
    {
      std::cout << "unexpected number of reductions = " << reductions << std::endl;
    }
    return sum_reduction_final(data, length);
  }
  reductions += 1;

  //
  // Prepare data to be reduced into grid_size block summations, stored in the workspace
  //
  auto grid_size = (length + reduction_block_size - 1) / reduction_block_size;
  data_t* d_workspace = nullptr;
  cuda_assert(cudaMalloc(&d_workspace, grid_size * sizeof(data_t)));

  //
  // Perform reduction
  //
  printf("grid_size %lu \n ", grid_size);

  sum_reduction_workspace<reduction_block_size>(grid_size, data, length, d_workspace);

  //
  // Recursively call this function on the reduced workspace
  //
  auto recursed_value = sum_reduction_driver(d_workspace, grid_size);

  cuda_assert(cudaFree(d_workspace));
  return recursed_value;
}

} // namespace detail
} // namespace boba

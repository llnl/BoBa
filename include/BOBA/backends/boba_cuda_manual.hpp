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

constexpr size_t reduction_block_size = 1024;

/**
 * @brief Supported reduction types
 */

enum struct reduction_t
{
  sum,
  max,
  min,
};

#ifdef BOBA_CUDA

/**
 * @brief A warp-level reduction, where a warp of threads share information directly
 * via warp intrinsics
 * \param value current thread's value that will be reduced
 * \return return value of lane 0 is the reduction (e.g. sum) of all values across the warp.
 * For lane > 0, the value is some partial reduction
 */

template <reduction_t reduction, typename data_t>
__device__ inline data_t warp_reduce(data_t value)
{
  auto all_threads_active = 0xfffffffff;

  // Assume we have 8 lanes (warpSize = 8)
  //    x0, x1, x2, x3, x4, x5, x6, x7
  // offset = 4
  // lane i pulls from lane i+4
  // out of bounds lanes get their own value (we don't care about these)
  // e.g. for sum reduction:
  //    x0+x4, x1+x5, x2+x6, x3+x6, (x4+x4, ...)
  // offset = 2
  //    x0+x4+x2+x6, x1+x5+x3+x6, (x2+x6+x4+x4, ...)
  // offset = 1
  //    x0+x4+x2+x6+x1+x5+x3+x6, (...)
  for(int offset = warpSize / 2; offset > 0; offset /= 2)
  {
    auto other_value = __shfl_down_sync(all_threads_active, value, offset);
    if constexpr(reduction == reduction_t::sum)
    {
      value += other_value;
    }
    if constexpr(reduction == reduction_t::min)
    {
      value = ::min(other_value, value);
    }
    if constexpr(reduction == reduction_t::max)
    {
      value = ::max(other_value, value);
    }
  }

  return value;
}

/**
 * @brief Block-level reduction via a series of warp-wise reductions
 * \param value current thread's value that will be reduced
 * \return The return value of thread 0 (lane 0, warp 0), is the block-wise reduction of values, else 0.
 */

template <reduction_t reduction, typename data_t>
__device__ inline auto block_reduce(data_t value)
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
  boba::Multiindexer<2> block_mid(
    {static_cast<boba::index_t>(warpSize), static_cast<boba::index_t>(tsize)});
  auto [lane_id, warp_id] = block_mid.multiindex(thread_id);

  //
  // Each warp performs a reduction and puts the result in shared memory
  //
  //       warp 0                    warp 1                   warp 2
  //    [ lane 0, lane 1, ...  ]  [ lane 0, lane 1, ...  ]  [ lane 0, lane 1, ...  ]
  //   shared[0] = warp 0 sum     shared[1] = warp 1 sum
  //
  auto warp_value = warp_reduce<reduction>(value);

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
    final_warp_value = warp_reduce<reduction>(warpwise_sums);
  }

  return final_warp_value;
}

/**
 * @brief Device-wise reduction where
 * (1) the collection of threads iterate over the whole array pulling in data elements
 * (2) "operation" is applied to each data element
 * (3) block-wise reductions compute the reduced value over a whole block
 * (4) each block writes the result to either an output array or atomically to the final result (see force_atomics)
 */

 template <reduction_t reduction, size_t block_size, bool force_atomics, typename data_t, typename index_t, typename out_data_t, typename operation_t>
__launch_bounds__(block_size)
__global__ 
void cuda_reduce(const data_t* data, index_t length, const operation_t operation, out_data_t* result)
{
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;

  int bsize = gridDim.x;
  int tsize = blockDim.x;

  boba::Multiindexer<2> grid_mid({static_cast<boba::index_t>(tsize), static_cast<boba::index_t>(bsize)});

  auto thread_offset = grid_mid.index({static_cast<boba::index_t>(thread_id), static_cast<boba::index_t>(block_id)});
  auto grid_size = grid_mid.size();

  // Multiindex the block and iterate over the whole array with stride = size of grid
  // Use multiple accumulators to prevent serialization of the reads
  out_data_t value{0.0};

  for(int i = thread_offset; i < length; i += grid_size)
  {
    auto raw_data_value = data[i];
    auto data_value = operation(raw_data_value);
    if constexpr(reduction == reduction_t::sum)
    {
      value += data_value;
    }
    if constexpr(reduction == reduction_t::min)
    {
      value = ::min(data_value, value);
    }
    if constexpr(reduction == reduction_t::max)
    {
      value = ::max(data_value, value);
    }
  }

  // Reduce over the current block
  auto block_reduction_value = block_reduce<reduction>(value);

  // Atomically add back to global memory
  // thread 0 is warp 0, lane 0
  if(thread_id == 0)
  {
    if(force_atomics)
    {
      atomicAdd(result, block_reduction_value);
    }
    else
    {
      result[block_id] = block_reduction_value;
    }
  }
}

#endif

/**
 * @brief See reduction_driver
 */

template <reduction_t reduction, int block_size, typename data_t, typename index_t, typename operation_t>
inline std::invoke_result_t<operation_t, data_t> reduction_final(const data_t* data, index_t length, operation_t final_operation)
{
  using output_t = std::invoke_result_t<operation_t, data_t>;
#ifdef BOBA_CUDA
  // Output memory
  void* d_output = nullptr;
  cuda_assert(cudaMalloc(&d_output, sizeof(output_t)));
  cudaStream_t stream = 0;
  cuda_assert(cudaMemsetAsync(d_output, 0, sizeof(output_t), stream));

  size_t shared_memory_bytes = 0;
  constexpr size_t threads_per_block = static_cast<size_t>(block_size);
  size_t grid_size = (static_cast<size_t>(length) + threads_per_block - 1) / threads_per_block;
  if (grid_size > reduction_block_size)
  {
    grid_size = reduction_block_size;
  }

  void* args[] = {(void*)&data, (void*)&length, (void*)&final_operation, (void*)&d_output};
  cuda_assert(cudaLaunchKernel(
    (const void*)cuda_reduce<reduction, block_size, true, data_t, index_t, output_t, decltype(final_operation)>,
    grid_size,
    block_size,
    args,
    shared_memory_bytes,
    stream));

  output_t output;
  cudaMemcpyKind kind = cudaMemcpyDefault;
  cuda_assert(cudaMemcpy(&output, d_output, sizeof(output_t), kind));

  cuda_assert(cudaFree(d_output));
#else
  boba_ignore(data);
  boba_ignore(length);
  output_t output{0};
#endif
  return output;
}

/**
  * @brief See reduction_driver
  */

template <reduction_t reduction, size_t block_size, typename data_t, typename index_t, typename out_data_t, typename operation_t>
inline void reduction_workspace(index_t grid_size, const data_t* data, index_t length, out_data_t* d_output, operation_t&& operation)
{
#ifdef BOBA_CUDA
  size_t shared_memory_bytes = 0;
  cudaStream_t stream = 0;
  void* args[] = {(void*)&data, (void*)&length, (void*)&operation, (void*)&d_output};
  cuda_assert(cudaLaunchKernel(
    (const void*)cuda_reduce<reduction, block_size, false, data_t, index_t, out_data_t, std::decay_t<operation_t>>,
    grid_size,
    block_size,
    args,
    shared_memory_bytes,
    stream));
#else
  boba_ignore(grid_size);
  boba_ignore(data);
  boba_ignore(length);
  boba_ignore(d_output);
#endif
}

/**
  * @brief Performs a reduction of operation(data_i) on gpus, using a workspace to store intermediate results
  */

template <reduction_t reduction, typename data_t, typename index_t, typename operation_t>
inline std::invoke_result_t<operation_t, data_t> reduction_driver(const data_t* data, index_t length, operation_t&& operation)
{
  using operation_result_t = std::invoke_result_t<operation_t, data_t>;
#ifdef BOBA_CUDA

  if (length == 0)
  {
    return operation_result_t{};
  }

  if (length < reduction_block_size*reduction_block_size)
  {
    return reduction_final<reduction, 512>(data, length, std::forward<operation_t>(operation));
  }

  //
  // Prepare data to be reduced into grid_size block summations, stored in the workspace
  //
  auto grid_size = reduction_block_size;
  operation_result_t* d_workspace = nullptr;
  cuda_assert(cudaMalloc(&d_workspace, grid_size * sizeof(operation_result_t)));

  //
  // Perform reduction into workspace
  //
  reduction_workspace<reduction, reduction_block_size>(grid_size, data, length, d_workspace, std::forward<operation_t>(operation));

  //
  // Perform reduction on the workspace to a final scalar value.
  //
  auto identity = []__boba_host_device__(operation_result_t x) { return x; };
  auto final_value = reduction_final<reduction, reduction_block_size>(d_workspace, grid_size, identity);

  cuda_assert(cudaFree(d_workspace));
  return final_value;
#else
  boba_ignore(data);
  boba_ignore(length);
  boba_error("This function requires BOBA_CUDA=1");
  return operation_result_t{0};
#endif
}

/**
  * @brief Performs a reduction of data_i on gpus (the operation is the identity)
  */

template <reduction_t reduction, typename data_t, typename index_t>
inline data_t reduction_driver(const data_t* data, index_t length)
{
  // No operation
  return reduction_driver<reduction>(data, length, []__boba_host_device__(data_t x){ return x; });
}

} // namespace detail
} // namespace boba

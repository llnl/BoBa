// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/abstractions/math.hpp"
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

/** @brief Returns the neutral value for a manual reduction. */
template <reduction_t reduction, typename data_t>
__boba_host_device__ constexpr data_t reduction_identity()
{
  if constexpr (reduction == reduction_t::max)
  {
    return ::boba::lowest_value<data_t>();
  }
  else if constexpr (reduction == reduction_t::min)
  {
    return ::boba::highest_value<data_t>();
  }
  else
  {
    return data_t{};
  }
}

template <typename data_t>
struct floating_atomic_bits;

template <>
struct floating_atomic_bits<float>
{
  using bits_t = unsigned int;

  static __device__ float value(bits_t bits)
  {
    return __uint_as_float(bits);
  }

  static __device__ bits_t bits(float value)
  {
    return __float_as_uint(value);
  }
};

template <>
struct floating_atomic_bits<double>
{
  using bits_t = unsigned long long;

  static __device__ double value(bits_t bits)
  {
    return __longlong_as_double(static_cast<long long>(bits));
  }

  static __device__ bits_t bits(double value)
  {
    return static_cast<bits_t>(__double_as_longlong(value));
  }
};

/** @brief Atomically combines a block extremum, using CAS for floating point values. */
template <reduction_t reduction, typename data_t>
__device__ inline void atomic_extrema(data_t* destination, data_t candidate)
{
  static_assert(reduction == reduction_t::min or reduction == reduction_t::max);
  if constexpr (std::is_floating_point_v<data_t>)
  {
    using conversion = floating_atomic_bits<data_t>;
    using bits_t = typename conversion::bits_t;
    auto* address = reinterpret_cast<bits_t*>(destination);
    bits_t observed = atomicCAS(address, bits_t{}, bits_t{});
    while (true)
    {
      const bits_t assumed = observed;
      const data_t current = conversion::value(assumed);
      const bool replace = reduction == reduction_t::min ? candidate < current : current < candidate;
      if (not replace)
      {
        break;
      }
      observed = atomicCAS(address, assumed, conversion::bits(candidate));
      if (observed == assumed)
      {
        break;
      }
    }
  }
  else if constexpr (reduction == reduction_t::min)
  {
    atomicMin(destination, candidate);
  }
  else
  {
    atomicMax(destination, candidate);
  }
}

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
 * \return Thread 0 holds the block-wise reduction of values.
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
  // Pull the shared memory into the first warp (pad with the reduction identity (eg 0 for sum))
  // Other warps do nothing
  // Example for sum reduce:
  //
  //       warp 0                    warp 1                   warp 2
  //    [ lane 0,    lane 1, ... lane ?, ... ]  [ lane 0, lane 1, ...  ]  [ lane 0, lane 1, ...  ]
  //   shared[0], shared[1], ...    0.0,          0.0, ....
  //
  int warps_per_block = tsize / warpSize;
  data_t warpwise_sums = reduction_identity<reduction, data_t>();
  if(thread_id < warps_per_block)
  {
    warpwise_sums = shared[lane_id];
  }

  //
  // Reduce in warp 0
  //
  data_t final_warp_value = reduction_identity<reduction, data_t>();
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
  out_data_t value = reduction_identity<reduction, out_data_t>();

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

  // Atomically combine block results in global memory
  // thread 0 is warp 0, lane 0
  if(thread_id == 0)
  {
    if constexpr(force_atomics)
    {
      if constexpr(reduction == reduction_t::sum)
      {
        atomicAdd(result, block_reduction_value);
      }
      else
      {
        atomic_extrema<reduction>(result, block_reduction_value);
      }
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
  if constexpr (reduction == reduction_t::sum)
  {
    cuda_assert(cudaMemsetAsync(d_output, 0, sizeof(output_t), stream));
  }
  else
  {
    const output_t identity = reduction_identity<reduction, output_t>();
    cuda_assert(cudaMemcpy(d_output, &identity, sizeof(output_t), cudaMemcpyHostToDevice));
  }

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
    return reduction_identity<reduction, operation_result_t>();
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

#ifdef BOBA_CUDA

/** @brief Reduces each short row or column, then keeps the largest sum in its section. */
template <bool sum_columns, typename data_t, typename index_t>
__global__
void reduce_short_matrix_norm_sections(
  const data_t* input, index_t rows, index_t cols, data_t* output, index_t section_size)
{
  const index_t sum_size = sum_columns ? cols : rows;
  const index_t max_size = sum_columns ? rows : cols;
  const index_t first = static_cast<index_t>(blockIdx.x) * section_size;
  const index_t last = boba::min(first + section_size, max_size);

  data_t section_max = reduction_identity<reduction_t::max, data_t>();
  for (index_t max_index = first; max_index < last; ++max_index)
  {
    data_t value{};
    if (threadIdx.x < sum_size)
    {
      if constexpr (sum_columns)
      {
        value = boba::abs(input[max_index + threadIdx.x * rows]);
      }
      else
      {
        value = boba::abs(input[threadIdx.x + max_index * rows]);
      }
    }

    const data_t sum = block_reduce<reduction_t::sum>(value);
    if (threadIdx.x == 0)
    {
      section_max = ::max(section_max, sum);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    output[blockIdx.x] = section_max;
  }
}

/** @brief Lets each thread sum a very short row or column before reducing block maxima. */
template <bool sum_columns, typename data_t, typename index_t>
__global__
void reduce_short_matrix_norm_thread_sections(
  const data_t* input, index_t rows, index_t cols, data_t* output)
{
  const index_t sum_size = sum_columns ? cols : rows;
  const index_t max_size = sum_columns ? rows : cols;
  const index_t first = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const index_t stride = static_cast<index_t>(gridDim.x) * blockDim.x;

  data_t thread_max = reduction_identity<reduction_t::max, data_t>();
  for (index_t max_index = first; max_index < max_size; max_index += stride)
  {
    data_t sum{};
    for (index_t sum_index = 0; sum_index < sum_size; ++sum_index)
    {
      if constexpr (sum_columns)
      {
        sum += boba::abs(input[max_index + sum_index * rows]);
      }
      else
      {
        sum += boba::abs(input[sum_index + max_index * rows]);
      }
    }
    thread_max = ::max(thread_max, sum);
  }

  const data_t block_max = block_reduce<reduction_t::max>(thread_max);
  if (threadIdx.x == 0)
  {
    output[blockIdx.x] = block_max;
  }
}

/** @brief Computes a matrix norm with a short summation direction in two reductions. */
template <bool sum_columns, typename data_t, typename index_t>
inline data_t short_matrix_norm(const data_t* matrix, index_t rows, index_t cols)
{
  const index_t sum_size = sum_columns ? cols : rows;
  const index_t max_size = sum_columns ? rows : cols;

  data_t* d_workspace = nullptr;
  cuda_assert(cudaMalloc(&d_workspace, reduction_block_size * sizeof(data_t)));
  if (sum_size <= 32)
  {
    size_t block_threads = 32;
    while (block_threads < 256 && block_threads * reduction_block_size < max_size)
    {
      block_threads *= 2;
    }

    void* args[] = {(void*)&matrix, (void*)&rows, (void*)&cols, (void*)&d_workspace};
    cuda_assert(cudaLaunchKernel(
      (const void*)reduce_short_matrix_norm_thread_sections<sum_columns, data_t, index_t>,
      dim3(reduction_block_size),
      dim3(block_threads),
      args,
      0,
      0));
  }
  else
  {
    index_t section_size = 1 + (max_size - 1) / reduction_block_size;
    size_t block_threads = 32;
    while (block_threads < sum_size)
    {
      block_threads *= 2;
    }

    void* args[] = {(void*)&matrix, (void*)&rows, (void*)&cols, (void*)&d_workspace, (void*)&section_size};
    cuda_assert(cudaLaunchKernel(
      (const void*)reduce_short_matrix_norm_sections<sum_columns, data_t, index_t>,
      dim3(reduction_block_size),
      dim3(block_threads),
      args,
      0,
      0));
  }

  auto identity = []__boba_host_device__(data_t x) { return x; };
  const data_t result = reduction_final<reduction_t::max, reduction_block_size>(d_workspace, reduction_block_size, identity);
  cuda_assert(cudaFree(d_workspace));
  return result;
}

/** @brief Reduces column portions for neighboring rows with coalesced loads. */

constexpr size_t matrix_linf_row_tile_size = 32;
constexpr size_t matrix_linf_column_groups = 8;

template<typename data_t, typename index_t>
__global__
void reduce_row_tile_portion(const data_t* input, index_t rows, index_t cols, data_t* output, index_t size_cols_portion)
{
  const index_t num_row_tiles = 1 + (rows - 1) / matrix_linf_row_tile_size;
  const index_t row_tile = blockIdx.x % num_row_tiles;
  const index_t column_block = blockIdx.x / num_row_tiles;
  const index_t row = row_tile * matrix_linf_row_tile_size + threadIdx.x;
  const index_t first_col = column_block * size_cols_portion;
  const index_t last_col = first_col + boba::min(size_cols_portion, cols - first_col);

  data_t partial_sum{};
  if (row < rows)
  {
    for (index_t col = first_col + threadIdx.y; col < last_col; col += blockDim.y)
    {
      partial_sum += boba::abs(input[row + col * rows]);
    }
  }

  __shared__ data_t group_sums[matrix_linf_column_groups][matrix_linf_row_tile_size];
  group_sums[threadIdx.y][threadIdx.x] = partial_sum;
  __syncthreads();

  if (threadIdx.y == 0 && row < rows)
  {
    data_t sum{};
    for (size_t group = 0; group < matrix_linf_column_groups; ++group)
    {
      sum += group_sums[group][threadIdx.x];
    }
    output[row + column_block * rows] = sum;
  }
}

template <typename data_t, typename index_t>
inline void row_wise_reduction(const data_t* input, index_t rows, index_t cols, data_t* d_output, index_t cols_per_portion)
{
  size_t shared_memory_bytes = 0;
  cudaStream_t stream = 0;
  const index_t num_row_tiles = 1 + (rows - 1) / matrix_linf_row_tile_size;
  const index_t num_cols_portions = 1 + (cols - 1) / cols_per_portion;
  const index_t grid_size = num_row_tiles * num_cols_portions;
  void* args[] = {(void*)&input, (void*)&rows, (void*)&cols, (void*)&d_output, (void*)&cols_per_portion};
  cuda_assert(cudaLaunchKernel(
    (const void*)reduce_row_tile_portion<data_t, index_t>,
    dim3(grid_size),
    dim3(matrix_linf_row_tile_size, matrix_linf_column_groups),
    args,
    shared_memory_bytes,
    stream));
}
#endif

template <typename data_t, typename index_t>
inline data_t matrix_linf_norm(const data_t* matrix, index_t rows, index_t cols)
{
  data_t result;
#ifdef BOBA_CUDA
  if (rows == 1)
  {
    result = reduction_driver<reduction_t::sum>(matrix, cols, []__boba_host_device__(data_t x){ return ::abs(x); });
  }
  else if (cols > 1 && cols <= 32)
  {
    result = short_matrix_norm<true>(matrix, rows, cols);
  }
  else if(cols > 1)
  {
    const index_t num_row_tiles = 1 + (rows - 1) / matrix_linf_row_tile_size;
    const index_t target_portions = 1 + (reduction_block_size - 1) / num_row_tiles;
    const index_t cols_per_portion = boba::max(index_t{32}, 1 + (cols - 1) / target_portions);
    const index_t num_cols_portions = 1 + (cols - 1) / cols_per_portion;
    const index_t workspace_size = rows * num_cols_portions;

    data_t* d_workspace = nullptr;
    cuda_assert(cudaMalloc(&d_workspace, workspace_size * sizeof(data_t)));

    //
    // Phase 1 - reduce to smaller grid
    //
    row_wise_reduction(matrix, rows, cols, d_workspace, cols_per_portion);

    //
    // Phase 2 - Recurse until each row has one sum, then take the maximum.
    //
    result = matrix_linf_norm(d_workspace, rows, num_cols_portions);

    cuda_assert(cudaFree(d_workspace));

  }
  else
  {
    result = reduction_driver<reduction_t::max>(matrix, rows, []__boba_host_device__(data_t x){ return ::abs(x); });
  }
#endif
  return result;
}


#ifdef BOBA_CUDA

/**
  * @brief Performs reduction of a portion of the matrix rows
  */

constexpr size_t reduction_block_cols = 1024;

template<typename data_t, typename index_t>
__global__
void reduce_col_portion(const data_t* input, index_t rows, index_t cols, data_t* output, index_t size_rows_portion)
{
  const index_t num_rows_portions = 1 + (rows - 1) / size_rows_portion;
  const index_t row_block = blockIdx.x % num_rows_portions;
  const index_t col = blockIdx.x / num_rows_portions;
  const index_t first_row = row_block * size_rows_portion;
  const index_t last_row = boba::min(first_row + size_rows_portion, rows);

  data_t sum{0.0};
  for(index_t row = first_row + threadIdx.x; row < last_row; row += blockDim.x)
  {
    sum += boba::abs(input[row + col * rows]);
  }

  const data_t portion_sum = block_reduce<reduction_t::sum>(sum);

  if(threadIdx.x == 0)
  {
    output[row_block + col * num_rows_portions] = portion_sum;
  }
}

template <typename data_t, typename index_t>
inline void col_wise_reduction(const data_t* input, index_t rows, index_t cols, data_t* d_output, index_t rows_per_portion)
{
  size_t shared_memory_bytes = 0;
  cudaStream_t stream = 0;
  static constexpr size_t block_size0 = reduction_block_cols;
  const index_t num_rows_portions = 1 + (rows - 1) / rows_per_portion;
  const index_t grid_size = cols * num_rows_portions;
  void* args[] = {(void*)&input, (void*)&rows, (void*)&cols, (void*)&d_output, (void*)&rows_per_portion};
  cuda_assert(cudaLaunchKernel(
    (const void*)reduce_col_portion<data_t, index_t>,
    dim3(grid_size),
    dim3(block_size0),
    args,
    shared_memory_bytes,
    stream));
}
#endif

template <typename data_t, typename index_t>
inline data_t matrix_one_norm(const data_t* matrix, index_t rows, index_t cols)
{
  data_t result;
#ifdef BOBA_CUDA
  if (rows > 1 && rows < reduction_block_size)
  {
    result = short_matrix_norm<false>(matrix, rows, cols);
  }
  else if(rows > 1)
  {
    const index_t block_parallelism = 132;
    const index_t target_portions = 1 + (block_parallelism - 1) / cols;
    const index_t rows_per_portion = boba::max(index_t{2}, 1 + (rows - 1) / target_portions);
    const index_t num_rows_portions = 1 + (rows - 1) / rows_per_portion;
    const index_t workspace_size = cols * num_rows_portions;

    data_t* d_workspace = nullptr;
    cuda_assert(cudaMalloc(&d_workspace, workspace_size * sizeof(data_t)));

    //
    // Phase 1 - reduce to smaller grid
    //
    col_wise_reduction(matrix, rows, cols, d_workspace, rows_per_portion);

    //
    // Phase 2 - Recurse until each column has one sum, then take the maximum.
    //
    result = matrix_one_norm(d_workspace, num_rows_portions, cols);

    cuda_assert(cudaFree(d_workspace));

  }
  else
  {
    result = reduction_driver<reduction_t::max>(matrix, cols, []__boba_host_device__(data_t x){ return ::abs(x); });
  }
#endif
  return result;
}


} // namespace detail
} // namespace boba

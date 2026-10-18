// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_CUDA
static constexpr cudaDataType cuda_double = CUDA_R_64F;

template <typename data_t>
struct cuda_data_t;

template <>
struct cuda_data_t<double>
{
  static constexpr cudaDataType value = CUDA_R_64F;
};

template <>
struct cuda_data_t<float>
{
  static constexpr cudaDataType value = CUDA_R_32F;
};

template <>
struct cuda_data_t<complex<double>>
{
  static constexpr cudaDataType value = CUDA_C_64F;
};

template <>
struct cuda_data_t<complex<float>>
{
  static constexpr cudaDataType value = CUDA_C_32F;
};

template <typename data_t>
inline constexpr cudaDataType cuda_data_t_v = cuda_data_t<data_t>::value;

// -----------------------------------------------------
// Debugging
// -----------------------------------------------------

#define cuda_assert(a) cuda_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a CUDA runtime error and terminates.
 * @param error CUDA error code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void cuda_assert_(
  cudaError_t error,
  const char* call,
  size_t line,
  const char* function,
  const char* file)
{
  if (error == cudaSuccess)
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;

  std::cout << cudaGetErrorName(error) << " " << cudaGetErrorString(error) << std::endl;
  exit(1);
}

// -----------------------------------------------------
// Memory utilities
// -----------------------------------------------------
/**
 * @brief Fills CUDA-accessible memory with a byte value.
 * @tparam T Element type.
 * @param ptr Destination pointer.
 * @param val Byte value passed to `cudaMemsetAsync`.
 * @param nelems Number of elements to touch.
 */
template <typename T>
inline void cuda_memset(T* ptr, size_t val, size_t nelems)
{
  if (nelems > 0u)
  {
    cudaStream_t stream = 0;
    cuda_assert(cudaMemsetAsync(ptr, val, sizeof(T) * nelems, stream));
  }
}

/**
 * @brief Copies a contiguous buffer using `cudaMemcpyAsync`.
 * @tparam T Element type.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param nelems Number of elements to copy.
 */
template <typename T>
inline void cuda_memcpy(T* dst, const T* src, size_t nelems)
{
  boba_assert(nelems > 0u, "can't copy nonpositve nelems");
  if (nelems > 0u)
  {
    cudaStream_t stream = 0;
    cudaMemcpyKind kind = cudaMemcpyDefault;
    cuda_assert(cudaMemcpyAsync(dst, src, sizeof(T) * nelems, kind, stream));
  }
}

/**
 * @brief Allocates CUDA device storage for a contiguous element range.
 * @tparam T Element type.
 * @tparam allocator_t Allocator handle type used when Umpire is enabled.
 * @param nelems Number of elements to allocate.
 * @param allocator Allocator used by the Umpire path.
 * @return Pointer to the allocated storage, or `nullptr` when `nelems == 0`.
 */
template <typename T, typename allocator_t>
[[nodiscard]]
inline T* cuda_malloc(size_t nelems, allocator_t& allocator)
{
  void* ptr = nullptr;
  if (nelems > 0u)
  {
#ifdef BOBA_UMPIRE
    ptr = ::boba::detail::umpire_allocate(allocator, sizeof(T) * nelems);
#else
    cuda_assert(cudaMalloc(&ptr, sizeof(T) * nelems));
#endif
  }
  return static_cast<T*>(ptr);
}

/**
 * @brief Frees CUDA storage and nulls the caller's pointer.
 * @tparam T Element type.
 * @tparam allocator_t Allocator handle type used when Umpire is enabled.
 * @param ptr Pointer to release.
 * @param allocator Allocator used by the Umpire path.
 */
template <typename T, typename allocator_t>
inline void cuda_free(T*& ptr, allocator_t& allocator)
{
  if (ptr != nullptr)
  {
#ifdef BOBA_UMPIRE
    ::boba::detail::umpire_deallocate(allocator, ptr);
#else
    cuda_assert(cudaFree(ptr));
#endif
    ptr = nullptr;
  }
}

// -----------------------------------------------------
// Loop details
// -----------------------------------------------------
/**
 * @brief Launches a one-dimensional CUDA kernel wrapper around a lambda.
 * @tparam block_size0 Thread-block size in the x dimension.
 * @tparam Lambda Callable type.
 * @param begin Inclusive starting index.
 * @param end Exclusive ending index.
 * @param lambda Device-callable body.
 */
template <size_t block_size0, typename Lambda>
__launch_bounds__(block_size0)
  __global__ void boba_lambda_kernel(size_t begin, size_t end, Lambda lambda)
{
  size_t i = begin + threadIdx.x + block_size0 * blockIdx.x;
  if (i < end)
  {
    lambda(i);
  }
}

/**
 * @brief Launches a two-dimensional CUDA kernel wrapper around a lambda.
 * @tparam block_size0 Thread-block size in the x dimension.
 * @tparam block_size1 Thread-block size in the y dimension.
 * @tparam Lambda Callable type.
 * @param begin0 Inclusive starting index for the first dimension.
 * @param end0 Exclusive ending index for the first dimension.
 * @param begin1 Inclusive starting index for the second dimension.
 * @param end1 Exclusive ending index for the second dimension.
 * @param lambda Device-callable body.
 */
template <size_t block_size0, size_t block_size1, typename Lambda>
__launch_bounds__(block_size0* block_size1)
  __global__ void boba_lambda_kernel_2d(
    size_t begin0, size_t end0, size_t begin1, size_t end1, Lambda lambda)
{
  size_t i0 = begin0 + threadIdx.x + block_size0 * blockIdx.x;
  size_t i1 = begin1 + threadIdx.y + block_size1 * blockIdx.y;
  if (i0 < end0 && i1 < end1)
  {
    lambda(i0, i1);
  }
}

/**
 * @brief Launches a three-dimensional CUDA kernel wrapper around a lambda.
 * @tparam block_size0 Thread-block size in the x dimension.
 * @tparam block_size1 Thread-block size in the y dimension.
 * @tparam block_size2 Thread-block size in the z dimension.
 * @tparam Lambda Callable type.
 * @param begin0 Inclusive starting index for the first dimension.
 * @param end0 Exclusive ending index for the first dimension.
 * @param begin1 Inclusive starting index for the second dimension.
 * @param end1 Exclusive ending index for the second dimension.
 * @param begin2 Inclusive starting index for the third dimension.
 * @param end2 Exclusive ending index for the third dimension.
 * @param lambda Device-callable body.
 */
template <size_t block_size0, size_t block_size1, size_t block_size2, typename Lambda>
__launch_bounds__(block_size0 * block_size1 * block_size2)
  __global__ void boba_lambda_kernel_3d(
    size_t begin0, size_t end0, size_t begin1, size_t end1, size_t begin2, size_t end2, Lambda lambda)
{
  size_t i0 = begin0 + threadIdx.x + block_size0 * blockIdx.x;
  size_t i1 = begin1 + threadIdx.y + block_size1 * blockIdx.y;
  size_t i2 = begin2 + threadIdx.z + block_size2 * blockIdx.z;
  if (i0 < end0 && i1 < end1 && i2 < end2)
  {
    lambda(i0, i1, i2);
  }
}

/**
 * @brief Launches a one-dimensional CUDA lambda kernel over `[begin, end)`.
 * @tparam Lambda Callable type.
 * @param begin Inclusive starting index.
 * @param end Exclusive ending index.
 * @param lambda Device-callable body.
 */
template <typename Lambda>
inline void cuda_launch(size_t begin, size_t end, Lambda&& lambda)
{
  size_t shared_memory_bytes = 0;
  cudaStream_t stream = 0;
  static constexpr size_t block_size = 256;
  size_t grid_size = (end - begin + block_size - 1) / block_size;
  void* args[] = {(void*)&begin, (void*)&end, (void*)&lambda};
  cuda_assert(cudaLaunchKernel(
    (const void*)boba_lambda_kernel<block_size, std::decay_t<Lambda>>,
    grid_size,
    block_size,
    args,
    shared_memory_bytes,
    stream));
}

/**
 * @brief Launches a two-dimensional CUDA lambda kernel.
 * @tparam Lambda Callable type.
 * @param begin0 Inclusive starting index for the first dimension.
 * @param end0 Exclusive ending index for the first dimension.
 * @param begin1 Inclusive starting index for the second dimension.
 * @param end1 Exclusive ending index for the second dimension.
 * @param lambda Device-callable body.
 */
template <typename Lambda>
inline void cuda_launch_2d(
  size_t begin0, size_t end0, size_t begin1, size_t end1, Lambda&& lambda)
{
  size_t shared_memory_bytes = 0;
  cudaStream_t stream = 0;
  static constexpr size_t block_size0 = 32;
  static constexpr size_t block_size1 = 8;
  size_t grid_size0 = (end0 - begin0 + block_size0 - 1) / block_size0;
  size_t grid_size1 = (end1 - begin1 + block_size1 - 1) / block_size1;
  void* args[] = {(void*)&begin0, (void*)&end0, (void*)&begin1, (void*)&end1, (void*)&lambda};
  cuda_assert(cudaLaunchKernel(
    (const void*)boba_lambda_kernel_2d<block_size0, block_size1, std::decay_t<Lambda>>,
    dim3(grid_size0, grid_size1),
    dim3(block_size0, block_size1),
    args,
    shared_memory_bytes,
    stream));
}

/**
 * @brief Launches a three-dimensional CUDA lambda kernel.
 * @tparam Lambda Callable type.
 * @param begin0 Inclusive starting index for the first dimension.
 * @param end0 Exclusive ending index for the first dimension.
 * @param begin1 Inclusive starting index for the second dimension.
 * @param end1 Exclusive ending index for the second dimension.
 * @param begin2 Inclusive starting index for the third dimension.
 * @param end2 Exclusive ending index for the third dimension.
 * @param lambda Device-callable body.
 */
template <typename Lambda>
inline void cuda_launch_3d(
  size_t begin0, size_t end0, size_t begin1, size_t end1, size_t begin2, size_t end2, Lambda&& lambda)
{
  size_t shared_memory_bytes = 0;
  cudaStream_t stream = 0;
  static constexpr size_t block_size0 = 32;
  static constexpr size_t block_size1 = 4;
  static constexpr size_t block_size2 = 2;
  size_t grid_size0 = (end0 - begin0 + block_size0 - 1) / block_size0;
  size_t grid_size1 = (end1 - begin1 + block_size1 - 1) / block_size1;
  size_t grid_size2 = (end2 - begin2 + block_size2 - 1) / block_size2;
  void* args[] = {(void*)&begin0, (void*)&end0, (void*)&begin1, (void*)&end1, (void*)&begin2, (void*)&end2, (void*)&lambda};
  cuda_assert(cudaLaunchKernel(
    (const void*)boba_lambda_kernel_3d<block_size0, block_size1, block_size2, std::decay_t<Lambda>>,
    dim3(grid_size0, grid_size1, grid_size2),
    dim3(block_size0, block_size1, block_size2),
    args,
    shared_memory_bytes,
    stream));
}

/**
 * @brief Synchronizes the default CUDA stream.
 */
inline void cuda_syncronize()
{
  cudaStream_t stream = 0;
  cuda_assert(cudaStreamSynchronize(stream));
}

/**
 * @brief Synchronizes threads within the current CUDA thread block.
 */
__boba_host_device__ inline void cuda_thread_synchronize()
{
#ifdef BOBA_DEVICE_CODE
  __syncthreads();
#endif
}

#else

/**
 * @brief Reports that CUDA memset support is unavailable.
 * @tparam T Element type.
 * @param ptr Destination pointer.
 * @param val Byte value.
 * @param nelems Number of elements to touch.
 */
template <typename T>
inline void cuda_memset(T* ptr, size_t val, size_t nelems)
{
  ::boba::detail::ignore(ptr);
  ::boba::detail::ignore(val);
  ::boba::detail::ignore(nelems);
  boba_error("cuda_memset requires a CUDA build.");
}

template <typename T, typename allocator_t>
inline T* cuda_malloc(size_t nelems, allocator_t& allocator)
{
  ::boba::detail::ignore(nelems);
  ::boba::detail::ignore(allocator);
  boba_error("cuda_malloc requires a CUDA build.");
  return nullptr;
}

template <typename T>
inline T* cuda_malloc_memcpy(const T* src, size_t nelems)
{
  ::boba::detail::ignore(src);
  ::boba::detail::ignore(nelems);
  boba_error("cuda_malloc_memcpy requires a CUDA build.");
  return nullptr;
}

template <typename T, typename allocator_t>
inline void cuda_free(T*& ptr, allocator_t& allocator)
{
  ::boba::detail::ignore(ptr);
  ::boba::detail::ignore(allocator);
  boba_error("cuda_free requires a CUDA build.");
}

template <typename T>
inline void cuda_memcpy(T* dst, const T* src, size_t nelems)
{
  ::boba::detail::ignore(dst);
  ::boba::detail::ignore(src);
  ::boba::detail::ignore(nelems);
  boba_error("cuda_memcpy requires a CUDA build.");
}

template <typename T>
inline void cuda_memcpy_free(T* dst, T*& ptr, size_t nelems)
{
  ::boba::detail::ignore(dst);
  ::boba::detail::ignore(ptr);
  ::boba::detail::ignore(nelems);
  boba_error("cuda_memcpy_free requires a CUDA build.");
}

#endif

} // namespace detail
} // namespace boba

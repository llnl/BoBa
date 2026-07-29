// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

// AMD Documentation
// https://docs.amd.com/bundle/Welcome-to-hipBLAS-s-documentation----hipBLAS-documentation/page/usermanual.html

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_HIP

// -----------------------------------------------------
// Debugging
// -----------------------------------------------------

#define hip_assert(a) hip_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a HIP runtime error and terminates.
 * @param error HIP error code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void hip_assert_(
  hipError_t error,
  const char* call,
  size_t line,
  const char* function,
  const char* file)
{
  if (error == hipSuccess)
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;

  std::cout << hipGetErrorName(error) << " " << hipGetErrorString(error) << std::endl;
  exit(1);
}

// -----------------------------------------------------
// Memory utilities
// -----------------------------------------------------
/**
 * @brief Fills HIP-accessible memory with a byte value.
 * @tparam T Element type.
 * @param ptr Destination pointer.
 * @param val Byte value passed to `hipMemsetAsync`.
 * @param nelems Number of elements to touch.
 */
template <typename T>
inline void hip_memset(T* ptr, size_t val, size_t nelems)
{
  if (nelems > 0u)
  {
    hipStream_t stream = nullptr;
    hip_assert(hipMemsetAsync(ptr, val, sizeof(T) * nelems, stream));
  }
}

/**
 * @brief Copies a contiguous buffer using `hipMemcpyAsync`.
 * @tparam T Element type.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param nelems Number of elements to copy.
 */
template <typename T>
inline void hip_memcpy(T* dst, const T* src, size_t nelems)
{
  boba_assert(nelems > 0u, "can't copy nonpositve nelems");
  if (nelems > 0u)
  {
    hipStream_t stream = nullptr;
    hipMemcpyKind kind = hipMemcpyDefault;
    hip_assert(hipMemcpyAsync(dst, src, sizeof(T) * nelems, kind, stream));
  }
}

/**
 * @brief Allocates HIP device storage for a contiguous element range.
 * @tparam T Element type.
 * @tparam allocator_t Allocator handle type used when Umpire is enabled.
 * @param nelems Number of elements to allocate.
 * @param allocator Allocator used by the caller.
 * @return Pointer to the allocated storage, or `nullptr` when `nelems == 0`.
 */
template <typename T, typename allocator_t>
[[nodiscard]]
inline T* hip_malloc(size_t nelems, allocator_t& allocator)
{
  void* ptr = nullptr;
  if (nelems > 0u)
  {
#ifdef BOBA_UMPIRE
    ptr = ::boba::detail::umpire_allocate(::boba::detail::device_allocator, sizeof(T) * nelems);
#else
    hip_assert(hipMalloc(&ptr, sizeof(T) * nelems));
#endif
  }
  return static_cast<T*>(ptr);
}

/**
 * @brief Frees HIP storage and nulls the caller's pointer.
 * @tparam T Element type.
 * @tparam allocator_t Allocator handle type used when Umpire is enabled.
 * @param ptr Pointer to release.
 * @param allocator Allocator used by the Umpire path.
 */
template <typename T, typename allocator_t>
inline void hip_free(T*& ptr, allocator_t& allocator)
{
  if (ptr != nullptr)
  {
#ifdef BOBA_UMPIRE
    ::boba::detail::umpire_deallocate(allocator, ptr);
#else
    hip_assert(hipFree(ptr));
#endif
    ptr = nullptr;
  }
}

// -----------------------------------------------------
// Loop details
// -----------------------------------------------------
/**
 * @brief Launches a one-dimensional HIP kernel wrapper around a lambda.
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
 * @brief Launches a two-dimensional HIP kernel wrapper around a lambda.
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
 * @brief Launches a three-dimensional HIP kernel wrapper around a lambda.
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
 * @brief Launches a one-dimensional HIP lambda kernel over `[begin, end)`.
 * @tparam Lambda Callable type.
 * @param begin Inclusive starting index.
 * @param end Exclusive ending index.
 * @param lambda Device-callable body.
 */
template <typename Lambda>
inline void hip_launch(size_t begin, size_t end, Lambda&& lambda)
{
  size_t shared_memory_bytes = 0;
  hipStream_t stream = nullptr;
  static constexpr size_t block_size = 256;
  size_t grid_size = (end - begin + block_size - 1) / block_size;
  void* args[] = {(void*)&begin, (void*)&end, (void*)&lambda};
  hip_assert(hipLaunchKernel(
    (const void*)boba_lambda_kernel<block_size, std::decay_t<Lambda>>,
    grid_size,
    block_size,
    args,
    shared_memory_bytes,
    stream));
}

/**
 * @brief Launches a two-dimensional HIP lambda kernel.
 * @tparam Lambda Callable type.
 * @param begin0 Inclusive starting index for the first dimension.
 * @param end0 Exclusive ending index for the first dimension.
 * @param begin1 Inclusive starting index for the second dimension.
 * @param end1 Exclusive ending index for the second dimension.
 * @param lambda Device-callable body.
 */
template <typename Lambda>
inline void hip_launch_2d(
  size_t begin0, size_t end0, size_t begin1, size_t end1, Lambda&& lambda)
{
  size_t shared_memory_bytes = 0;
  hipStream_t stream = nullptr;
  static constexpr size_t block_size0 = 32;
  static constexpr size_t block_size1 = 8;
  size_t grid_size0 = (end0 - begin0 + block_size0 - 1) / block_size0;
  size_t grid_size1 = (end1 - begin1 + block_size1 - 1) / block_size1;
  void* args[] = {(void*)&begin0, (void*)&end0, (void*)&begin1, (void*)&end1, (void*)&lambda};
  hip_assert(hipLaunchKernel(
    (const void*)boba_lambda_kernel_2d<block_size0, block_size1, std::decay_t<Lambda>>,
    dim3(grid_size0, grid_size1),
    dim3(block_size0, block_size1),
    args,
    shared_memory_bytes,
    stream));
}

/**
 * @brief Launches a three-dimensional HIP lambda kernel.
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
inline void hip_launch_3d(
  size_t begin0, size_t end0, size_t begin1, size_t end1, size_t begin2, size_t end2, Lambda&& lambda)
{
  size_t shared_memory_bytes = 0;
  hipStream_t stream = nullptr;
  static constexpr size_t block_size0 = 32;
  static constexpr size_t block_size1 = 4;
  static constexpr size_t block_size2 = 2;
  size_t grid_size0 = (end0 - begin0 + block_size0 - 1) / block_size0;
  size_t grid_size1 = (end1 - begin1 + block_size1 - 1) / block_size1;
  size_t grid_size2 = (end2 - begin2 + block_size2 - 1) / block_size2;
  void* args[] = {(void*)&begin0, (void*)&end0, (void*)&begin1, (void*)&end1, (void*)&begin2, (void*)&end2, (void*)&lambda};
  hip_assert(hipLaunchKernel(
    (const void*)boba_lambda_kernel_3d<block_size0, block_size1, block_size2, std::decay_t<Lambda>>,
    dim3(grid_size0, grid_size1, grid_size2),
    dim3(block_size0, block_size1, block_size2),
    args,
    shared_memory_bytes,
    stream));
}

/**
 * @brief Synchronizes the default HIP stream.
 */
inline void hip_syncronize()
{
  hipStream_t stream = nullptr;
  hip_assert(hipStreamSynchronize(stream));
}

/**
 * @brief Synchronizes threads within the current HIP thread block.
 */
__boba_host_device__ inline void hip_thread_synchronize()
{
  __syncthreads();
}

#else

/**
 * @brief Reports that HIP memset support is unavailable.
 * @tparam T Element type.
 * @param ptr Destination pointer.
 * @param val Byte value.
 * @param nelems Number of elements to touch.
 */
template <typename T>
inline void hip_memset(T* ptr, size_t val, size_t nelems)
{
  ::boba::detail::ignore(ptr);
  ::boba::detail::ignore(val);
  ::boba::detail::ignore(nelems);
  boba_error("hip_memset requires a HIP build.");
}

/**
 * @brief Reports that HIP memcpy support is unavailable.
 * @tparam T Element type.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param nelems Number of elements to copy.
 */
template <typename T>
inline void hip_memcpy(T* dst, const T* src, size_t nelems)
{
  ::boba::detail::ignore(dst);
  ::boba::detail::ignore(src);
  ::boba::detail::ignore(nelems);
  boba_error("hip_memcpy requires a HIP build.");
}

/**
 * @brief Reports that HIP allocation support is unavailable.
 * @tparam T Element type.
 * @tparam allocator_t Allocator type.
 * @param nelems Number of elements to allocate.
 * @param allocator Allocator used for the allocation.
 * @return Always `nullptr` in non-HIP builds.
 */
template <typename T, typename allocator_t>
inline T* hip_malloc(size_t nelems, allocator_t& allocator)
{
  ::boba::detail::ignore(nelems);
  ::boba::detail::ignore(allocator);
  boba_error("hip_malloc requires a HIP build.");
  return nullptr;
}

/**
 * @brief Reports that HIP deallocation support is unavailable.
 * @tparam T Element type.
 * @tparam allocator_t Allocator type.
 * @param ptr Pointer to release.
 * @param allocator Allocator used for the deallocation.
 */
template <typename T, typename allocator_t>
inline void hip_free(T*& ptr, allocator_t& allocator)
{
  ::boba::detail::ignore(ptr);
  ::boba::detail::ignore(allocator);
  boba_error("hip_free requires a HIP build.");
}

#endif

} // namespace detail
} // namespace boba

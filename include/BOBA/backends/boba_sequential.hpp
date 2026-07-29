// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstring>

namespace boba
{
namespace detail
{

// -------------------------------------------------------------------------------------
// Memory details
// -------------------------------------------------------------------------------------
/**
 * @brief Fills host memory with a byte value.
 * @tparam T Element type.
 * @param ptr Destination pointer.
 * @param val Byte value passed to `std::memset`.
 * @param nelems Number of elements to touch.
 */
template <typename T>
inline void host_memset(T* ptr, size_t val, size_t nelems)
{
  if (nelems > 0u)
  {
    std::memset(ptr, val, sizeof(T) * nelems);
  }
}

/**
 * @brief Copies host memory between contiguous buffers.
 * @tparam T Element type.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param nelems Number of elements to copy.
 */
template <typename T>
inline void host_memcpy(T* dst, const T* src, size_t nelems)
{
  if (nelems > 0u)
  {
    std::memcpy(dst, src, sizeof(T) * nelems);
  }
}

/**
 * @brief Allocates host storage for a contiguous element range.
 * @tparam T Element type.
 * @tparam allocator_t Allocator handle type used when Umpire is enabled.
 * @param nelems Number of elements to allocate.
 * @param allocator Allocator used by the Umpire path.
 * @return Pointer to the allocated storage, or `nullptr` when `nelems == 0`.
 */
template <typename T, typename allocator_t>
[[nodiscard]]
inline T* host_malloc(size_t nelems, allocator_t& allocator)
{
  void* ptr = nullptr;
  if (nelems > 0u)
  {
#ifdef BOBA_UMPIRE
    ptr = ::boba::detail::umpire_allocate(allocator, sizeof(T) * nelems);
#else
    ::boba::detail::ignore(allocator);
    ptr = std::malloc(sizeof(T) * nelems);
#endif
  }
  return static_cast<T*>(ptr);
}

/**
 * @brief Frees host storage and nulls the caller's pointer.
 * @tparam T Element type.
 * @tparam allocator_t Allocator handle type used when Umpire is enabled.
 * @param ptr Pointer to release.
 * @param allocator Allocator used by the Umpire path.
 */
template <typename T, typename allocator_t>
inline void host_free(T*& ptr, allocator_t& allocator)
{
  if (ptr != nullptr)
  {
#ifdef BOBA_UMPIRE
    ::boba::detail::umpire_deallocate(allocator, ptr);
#else
    ::boba::detail::ignore(allocator);
    std::free(ptr);
#endif
    ptr = nullptr;
  }
}

// -----------------------------------------------------
// Loop details
// -----------------------------------------------------
/**
 * @brief Executes a one-dimensional loop sequentially on the host.
 * @tparam index_t Loop index type.
 * @tparam Lambda Callable type.
 * @param begin Inclusive starting index.
 * @param end Exclusive ending index.
 * @param lambda Callable invoked once per index.
 */
template <typename Lambda>
inline void seq_loop(index_t begin, index_t end, Lambda&& lambda)
{
  for (index_t i = begin; i < end; ++i)
  {
    lambda(i);
  }
}

/**
 * @brief Executes a two-dimensional loop sequentially on the host.
 * @tparam index_t Loop index type.
 * @tparam Lambda Callable type.
 * @param begin0 Inclusive starting index for the first dimension.
 * @param end0 Exclusive ending index for the first dimension.
 * @param begin1 Inclusive starting index for the second dimension.
 * @param end1 Exclusive ending index for the second dimension.
 * @param lambda Callable invoked with `(i0, i1)`.
 */
template <typename Lambda>
inline void seq_loop_2d(index_t begin0, index_t end0, index_t begin1, index_t end1, Lambda&& lambda)
{
  for (index_t i1 = begin1; i1 < end1; ++i1)
  {
    for (index_t i0 = begin0; i0 < end0; ++i0)
    {
      lambda(i0, i1);
    }
  }
}

/**
 * @brief Executes a three-dimensional loop sequentially on the host.
 * @tparam index_t Loop index type.
 * @tparam Lambda Callable type.
 * @param begin0 Inclusive starting index for the first dimension.
 * @param end0 Exclusive ending index for the first dimension.
 * @param begin1 Inclusive starting index for the second dimension.
 * @param end1 Exclusive ending index for the second dimension.
 * @param begin2 Inclusive starting index for the third dimension.
 * @param end2 Exclusive ending index for the third dimension.
 * @param lambda Callable invoked with `(i0, i1, i2)`.
 */
template <typename Lambda>
inline void seq_loop_3d(index_t begin0, index_t end0, index_t begin1, index_t end1, index_t begin2, index_t end2, Lambda&& lambda)
{
  for (index_t i2 = begin2; i2 < end2; ++i2)
  {
    for (index_t i1 = begin1; i1 < end1; ++i1)
    {
      for (index_t i0 = begin0; i0 < end0; ++i0)
      {
        lambda(i0, i1, i2);
      }
    }
  }
}

} // namespace detail
} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include "BOBA/boba.hpp"

namespace boba
{

namespace detail
{

// ---------------------------------------------------------------------------
// Memory Abstractions
// ---------------------------------------------------------------------------
/**
 * @brief Fills memory in an execution space.
 * @tparam space Target execution space.
 * @tparam T Element type.
 * @param ptr Destination pointer.
 * @param val Byte value forwarded to the backend memset.
 * @param nelems Number of elements to initialize.
 */
template <execution_space space, typename T>
inline void memset(T* ptr, size_t val, size_t nelems)
{
  if (nelems > 0u)
  {
    if constexpr (space == execution_space::CPU)
    {
      boba::detail::host_memset<T>(ptr, val, nelems);
    }
    else if constexpr (space == execution_space::CUDA)
    {
      boba::detail::cuda_memset<T>(ptr, val, nelems);
    }
    else if constexpr (space == execution_space::HIP)
    {
      boba::detail::hip_memset<T>(ptr, val, nelems);
    }
    else
    {
      boba_error("Unknown space");
    }
  }
}

/**
 * @brief Copies memory between execution spaces.
 * @tparam dst_space Destination execution space.
 * @tparam src_space Source execution space.
 * @tparam T Element type.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param nelems Number of elements to copy.
 */
template <execution_space dst_space, execution_space src_space, typename T>
inline void memcpy(T* dst, const T* src, size_t nelems)
{
  if (nelems > 0u)
  {
    if constexpr (dst_space == execution_space::CPU &&
                  src_space == execution_space::CPU)
    {
      boba::detail::host_memcpy<T>(dst, src, nelems);
    }
    else if constexpr (dst_space == execution_space::CUDA ||
                       src_space == execution_space::CUDA)
    {
      boba::detail::cuda_memcpy<T>(dst, src, nelems);
    }
    else if constexpr (dst_space == execution_space::HIP ||
                       src_space == execution_space::HIP)
    {
      boba::detail::hip_memcpy<T>(dst, src, nelems);
    }
    else
    {
      boba_error("Unknown space");
    }
  }
}

/**
 * @brief Allocates elements in an execution space.
 * @tparam space Allocation execution space.
 * @tparam T Element type.
 * @tparam allocator_t Allocator type.
 * @param nelems Number of elements to allocate.
 * @param allocator Allocator used for the allocation.
 * @return Pointer to the allocated storage, or `nullptr` when `nelems == 0`.
 */
template <execution_space space, typename T, typename allocator_t>
inline T* malloc(size_t nelems, allocator_t& allocator)
{
#ifdef BOBA_VERBOSE_MEMORY
  auto space_name = ::boba::detail::execution_space_name(space);
  printf("::boba::malloc %s %zu\n", space_name.c_str(), nelems);
#endif
  void* ptr = nullptr;
  if (nelems > 0u)
  {
    if constexpr (space == execution_space::CPU)
    {
      ptr = boba::detail::host_malloc<T>(nelems, allocator);
    }
    else if constexpr (space == execution_space::CUDA)
    {
      ptr = boba::detail::cuda_malloc<T>(nelems, allocator);
    }
    else if constexpr (space == execution_space::HIP)
    {
      ptr = boba::detail::hip_malloc<T>(nelems, allocator);
    }
    else
    {
      boba_error("Unknown space");
    }
  }
#ifdef BOBA_VERBOSE_MEMORY
  printf("::boba::malloc %s %zu => %p\n", space_name.c_str(), nelems, ptr);
#endif
  return static_cast<T*>(ptr);
}

/**
 * @brief Allocates storage and copies host data into it.
 * @tparam space Destination execution space.
 * @tparam T Element type.
 * @tparam allocator_t Allocator type.
 * @param src Source pointer.
 * @param nelems Number of elements to copy.
 * @param allocator Allocator used for the allocation.
 * @return Newly allocated copy of `src`.
 */
template <execution_space space, typename T, typename allocator_t>
inline T* malloc_memcpy(const T* src, size_t nelems, allocator_t& allocator)
{
  T* ptr = boba::detail::malloc<space, T>(nelems, allocator);
  boba::detail::memcpy<space>(ptr, src, nelems);
  return ptr;
}

/**
 * @brief Allocates storage and copies host data into it.
 * @tparam space Destination execution space.
 * @tparam T Element type.
 * @param src Source pointer.
 * @param nelems Number of elements to copy.
 * @return Newly allocated copy of `src`.
 */
template <execution_space space, typename T>
inline T* malloc_memcpy(const T* src, size_t nelems)
{
  auto allocator = ::boba::detail::get_default_allocator<space>();
  return boba::detail::malloc_memcpy<space, T>(src, nelems, allocator);
}

/**
 * @brief Frees storage in an execution space.
 * @tparam space Allocation execution space.
 * @tparam T Element type.
 * @tparam allocator_t Allocator type.
 * @param ptr Pointer to free. Reset to `nullptr` on return.
 * @param allocator Allocator used for the deallocation.
 */
template <execution_space space, typename T, typename allocator_t>
inline void free(T*& ptr, allocator_t& allocator)
{
#ifdef BOBA_VERBOSE_MEMORY
  auto space_name = ::boba::detail::execution_space_name(space);
  printf("::boba::free %s %p\n", space_name.c_str(), ptr);
#endif
  if (ptr != nullptr)
  {
    if constexpr (space == execution_space::CPU)
    {
      boba::detail::host_free<T>(ptr, allocator);
    }
    else if constexpr (space == execution_space::CUDA)
    {
      boba::detail::cuda_free<T>(ptr, allocator);
    }
    else if constexpr (space == execution_space::HIP)
    {
      boba::detail::hip_free<T>(ptr, allocator);
    }
    else
    {
      boba_error("Unknown space");
    }
    ptr = nullptr;
  }
}

/**
 * @brief Copies data out of an allocation and then frees it.
 * @tparam space Source execution space.
 * @tparam T Element type.
 * @tparam allocator_t Allocator type.
 * @param dst Destination pointer.
 * @param ptr Source allocation. Reset to `nullptr` on return.
 * @param nelems Number of elements to copy.
 * @param allocator Allocator used for the deallocation.
 */
template <execution_space space, typename T, typename allocator_t>
inline void memcpy_free(T* dst, T*& ptr, size_t nelems, allocator_t& allocator)
{
  boba::detail::memcpy<space>(dst, ptr, nelems);
  boba::detail::free<space>(ptr, allocator);
}

/**
 * @brief Copies data out of an allocation and then frees it.
 * @tparam space Source execution space.
 * @tparam T Element type.
 * @param dst Destination pointer.
 * @param ptr Source allocation. Reset to `nullptr` on return.
 * @param nelems Number of elements to copy.
 */
template <execution_space space, typename T>
inline void memcpy_free(T* dst, T*& ptr, size_t nelems)
{
  auto allocator = ::boba::detail::get_default_allocator<space>();
  boba::detail::memcpy_free<space, T>(dst, ptr, nelems, allocator);
}

} // namespace detail
} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string>

#ifdef BOBA_UMPIRE
#include "umpire/Allocator.hpp"
#endif

namespace boba
{

namespace detail
{

// ---------------------
// Umpire
// ---------------------

#ifdef BOBA_UMPIRE
using Allocator_t = umpire::Allocator;
extern Allocator_t host_allocator;
#if defined(BOBA_CUDA) || defined(BOBA_HIP)
extern Allocator_t device_allocator;
#endif

/**
 * @brief Returns the default allocator for an execution space.
 * @tparam space Execution space.
 * @return Host or device allocator associated with `space`.
 */
template <execution_space space>
Allocator_t get_default_allocator()
{
#if defined(BOBA_CUDA) || defined(BOBA_HIP)
  return (space == execution_space::CPU) ? host_allocator : device_allocator;
#else
  return host_allocator;
#endif
}

extern void print_umpire_records(FILE* f, bool print_backtrace);
extern void print_umpire_stats(FILE* f);

/**
 * @brief Allocates bytes with an Umpire allocator and reports failures.
 * @param allocator Umpire allocator.
 * @param nbytes Number of bytes to allocate.
 * @return Allocated storage pointer.
 */
inline void* umpire_allocate(Allocator_t& allocator, size_t nbytes)
{
  try
  {
    return allocator.allocate(nbytes);
  }
  catch (const std::exception& e)
  {
    fprintf(stdout, "Exception in umpire allocate %s\n", e.what());
    print_umpire_stats(stdout);
#ifdef BOBA_VERBOSE_MEMORY
    print_umpire_records(stdout, true);
#endif
    throw;
  }
}

/**
 * @brief Frees storage with an Umpire allocator and reports failures.
 * @param allocator Umpire allocator.
 * @param ptr Pointer to free.
 */
inline void umpire_deallocate(Allocator_t& allocator, void* ptr)
{
  try
  {
    return allocator.deallocate(ptr);
  }
  catch (const std::exception& e)
  {
    fprintf(stdout, "Exception in umpire deallocate %s\n", e.what());
    print_umpire_stats(stdout);
#ifdef BOBA_VERBOSE_MEMORY
    print_umpire_records(stdout, true);
#endif
    throw;
  }
}

#else

/*
 * This dummy allocator is used as a placeholder for when you want to pass around an
 * allocator, e.g. an Umpire allocator, but need something for when not building with Umpire
 */

struct DummyAllocatorStruct
{
  bool is_null = true;
};

using Allocator_t = DummyAllocatorStruct;

extern Allocator_t dummy_allocator;

/**
 * @brief Returns a dummy allocator placeholder.
 * @tparam space Execution space.
 * @return Non-null dummy allocator instance.
 */
template <execution_space space>
auto get_default_allocator()
{
  return DummyAllocatorStruct{false};
}

#endif

#ifdef BOBA_UMPIRE

#else

/**
 * @brief Compares dummy allocators for equality.
 * @param alloc_a Left-hand allocator.
 * @param alloc_b Right-hand allocator.
 * @return `true` when both allocators share the same null state.
 */
inline bool operator==(const DummyAllocatorStruct& alloc_a, const DummyAllocatorStruct& alloc_b)
{
  return (alloc_a.is_null == alloc_b.is_null);
}

/**
 * @brief Compares dummy allocators for inequality.
 * @param alloc_a Left-hand allocator.
 * @param alloc_b Right-hand allocator.
 * @return `true` when the allocators differ in null state.
 */
inline bool operator!=(const DummyAllocatorStruct& alloc_a, const DummyAllocatorStruct& alloc_b)
{
  return (alloc_a.is_null != alloc_b.is_null);
}

#endif

} // namespace detail

} // namespace boba

#ifdef BOBA_UMPIRE
namespace umpire
{

/**
 * @brief Compares Umpire allocators for inequality.
 * @param alloc_a Left-hand allocator.
 * @param alloc_b Right-hand allocator.
 * @return `true` when allocator ids differ.
 */
inline bool operator!=(const umpire::Allocator& alloc_a, const umpire::Allocator& alloc_b)
{
  return (alloc_a.getId() != alloc_b.getId());
}

/**
 * @brief Compares Umpire allocators for equality.
 * @param alloc_a Left-hand allocator.
 * @param alloc_b Right-hand allocator.
 * @return `true` when allocator ids match.
 */
inline bool operator==(const umpire::Allocator& alloc_a, const umpire::Allocator& alloc_b)
{
  return (alloc_a.getId() == alloc_b.getId());
}

} // namespace umpire
#endif

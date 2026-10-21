// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <climits>
#include <cstddef>

#if defined(BOBA_CUDA_LIBS)
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#elif defined(BOBA_HIP_LIBS)
#include <hipcub/device/device_radix_sort.hpp>
#include <hipcub/device/device_reduce.hpp>
#include <hipcub/device/device_scan.hpp>
#endif

namespace boba
{

namespace detail
{

/**
 * \brief Returns true when CUB (CUDA) or hipCUB (HIP) device primitives are compiled in.
 */
constexpr bool boba_cub_enabled() noexcept
{
#if defined(BOBA_CUDA_LIBS) || defined(BOBA_HIP_LIBS)
  return true;
#else
  return false;
#endif
}

/**
 * \brief Returns true when \p space can use the CUB/hipCUB wrappers in this file.
 */
template <execution_space space>
constexpr bool cub_available() noexcept
{
#if defined(BOBA_CUDA_LIBS)
  return space == execution_space::CUDA;
#elif defined(BOBA_HIP_LIBS)
  return space == execution_space::HIP;
#else
  return false;
#endif
}

#if defined(BOBA_CUDA_LIBS) || defined(BOBA_HIP_LIBS)

#if defined(BOBA_CUDA_LIBS)
namespace gpucub = ::cub;
#define boba_cub_assert(a) cuda_assert(a)
#else
namespace gpucub = ::hipcub;
#define boba_cub_assert(a) hip_assert(a)
#endif

/**
 * \brief Device scratch buffer for CUB/hipCUB, allocated through the BoBa allocator.
 *
 * Uses the default device allocator, so it draws from the Umpire pool when
 * BOBA_UMPIRE is enabled. Freed on destruction. Without Umpire every wrapper
 * call performs a raw cudaMalloc/hipMalloc and free, which is slow and
 * synchronizing.
 */
template <execution_space space>
struct CubTempStorage
{
  explicit CubTempStorage(std::size_t bytes)
      : m_allocator(get_default_allocator<space>()),
        m_bytes(bytes),
        m_data(::boba::detail::malloc<space, std::byte>(bytes, m_allocator))
  {
  }

  CubTempStorage(CubTempStorage const&) = delete;
  CubTempStorage& operator=(CubTempStorage const&) = delete;

  ~CubTempStorage()
  {
    ::boba::detail::free<space>(m_data, m_allocator);
  }

  void* data() noexcept
  {
    return m_data;
  }

  std::size_t& bytes() noexcept
  {
    return m_bytes;
  }

private:
  Allocator_t m_allocator;
  std::size_t m_bytes = 0;
  std::byte* m_data = nullptr;
};

/**
 * \brief Converts an item count to the int expected by CUB/hipCUB device algorithms.
 *
 * Limits each wrapper call to INT_MAX (2^31 - 1) items, e.g. nnz of a sparse tensor.
 */
inline int cub_item_count(index_t n)
{
  boba_always_assert_le(n, static_cast<index_t>(INT_MAX), "CUB item count exceeds INT_MAX");
  return static_cast<int>(n);
}

/**
 * \brief Binary sum used by reduce_by_key; works for real and boba::complex values.
 */
struct CubSumOp
{
  template <typename T>
  __boba_host_device__ T operator()(T const& a, T const& b) const
  {
    return a + b;
  }
};

/**
 * \brief Stable radix sort of (key, value) pairs on device.
 *
 * Only bits [0, end_bit) of each key are compared, so callers that know the
 * key range (e.g. linear indices below a tensor size) can skip high passes.
 */
template <execution_space space, typename key_t, typename value_t>
void sort_pairs(key_t const* keys_in,
                key_t* keys_out,
                value_t const* values_in,
                value_t* values_out,
                index_t n,
                int end_bit = static_cast<int>(sizeof(key_t) * CHAR_BIT))
{
  static_assert(cub_available<space>(), "sort_pairs requires a CUB/hipCUB device space");
  if (n == 0)
  {
    return;
  }
  const int count = cub_item_count(n);
  std::size_t bytes = 0;
  boba_cub_assert(gpucub::DeviceRadixSort::SortPairs(nullptr, bytes, keys_in, keys_out, values_in, values_out, count, 0, end_bit));
  CubTempStorage<space> storage(bytes);
  boba_cub_assert(gpucub::DeviceRadixSort::SortPairs(storage.data(), storage.bytes(), keys_in, keys_out, values_in, values_out, count, 0, end_bit));
}

/**
 * \brief Radix sort of keys on device.
 */
template <execution_space space, typename key_t>
void sort_keys(key_t const* keys_in,
               key_t* keys_out,
               index_t n,
               int end_bit = static_cast<int>(sizeof(key_t) * CHAR_BIT))
{
  static_assert(cub_available<space>(), "sort_keys requires a CUB/hipCUB device space");
  if (n == 0)
  {
    return;
  }
  const int count = cub_item_count(n);
  std::size_t bytes = 0;
  boba_cub_assert(gpucub::DeviceRadixSort::SortKeys(nullptr, bytes, keys_in, keys_out, count, 0, end_bit));
  CubTempStorage<space> storage(bytes);
  boba_cub_assert(gpucub::DeviceRadixSort::SortKeys(storage.data(), storage.bytes(), keys_in, keys_out, count, 0, end_bit));
}

/**
 * \brief Exclusive prefix sum on device: out[i] = in[0] + ... + in[i-1], out[0] = 0.
 */
template <execution_space space, typename T>
void exclusive_scan(T const* in, T* out, index_t n)
{
  static_assert(cub_available<space>(), "exclusive_scan requires a CUB/hipCUB device space");
  if (n == 0)
  {
    return;
  }
  const int count = cub_item_count(n);
  std::size_t bytes = 0;
  boba_cub_assert(gpucub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, count));
  CubTempStorage<space> storage(bytes);
  boba_cub_assert(gpucub::DeviceScan::ExclusiveSum(storage.data(), storage.bytes(), in, out, count));
}

/**
 * \brief Sums runs of equal consecutive keys on device.
 *
 * Example: keys {2, 2, 5, 7, 7} with values {1, 3, 4, 2, 6} give unique keys
 * {2, 5, 7} and aggregates {4, 4, 8}. Input must be grouped by key (e.g. sorted).
 * Synchronizes to read back the number of runs.
 *
 * \return Number of unique keys written to \p unique_keys_out.
 */
template <execution_space space, typename key_t, typename value_t>
index_t reduce_by_key(key_t const* keys_in,
                      key_t* unique_keys_out,
                      value_t const* values_in,
                      value_t* aggregates_out,
                      index_t n)
{
  static_assert(cub_available<space>(), "reduce_by_key requires a CUB/hipCUB device space");
  if (n == 0)
  {
    return 0;
  }
  const int count = cub_item_count(n);
  Allocator_t allocator = get_default_allocator<space>();
  int* number_runs_device = ::boba::detail::malloc<space, int>(1, allocator);

  std::size_t bytes = 0;
  boba_cub_assert(gpucub::DeviceReduce::ReduceByKey(nullptr, bytes, keys_in, unique_keys_out, values_in, aggregates_out, number_runs_device, CubSumOp{}, count));
  {
    CubTempStorage<space> storage(bytes);
    boba_cub_assert(gpucub::DeviceReduce::ReduceByKey(storage.data(), storage.bytes(), keys_in, unique_keys_out, values_in, aggregates_out, number_runs_device, CubSumOp{}, count));
  }

  int number_runs = 0;
  ::boba::detail::memcpy<host_space, space>(&number_runs, number_runs_device, 1);
  ::boba::detail::synchronize<space>();
  ::boba::detail::free<space>(number_runs_device, allocator);
  return static_cast<index_t>(number_runs);
}

#undef boba_cub_assert

#else // !(BOBA_CUDA_LIBS || BOBA_HIP_LIBS)

// Stubs so callers can name these inside `if constexpr (cub_available<space>())`
// branches that are discarded in non-GPU builds. Instantiating one is a
// compile-time error.

template <execution_space space, typename key_t, typename value_t>
void sort_pairs(key_t const*, key_t*, value_t const*, value_t*, index_t, int = 0)
{
  static_assert(cub_available<space>(), "sort_pairs requires BOBA_CUDA_LIBS or BOBA_HIP_LIBS");
}

template <execution_space space, typename key_t>
void sort_keys(key_t const*, key_t*, index_t, int = 0)
{
  static_assert(cub_available<space>(), "sort_keys requires BOBA_CUDA_LIBS or BOBA_HIP_LIBS");
}

template <execution_space space, typename T>
void exclusive_scan(T const*, T*, index_t)
{
  static_assert(cub_available<space>(), "exclusive_scan requires BOBA_CUDA_LIBS or BOBA_HIP_LIBS");
}

template <execution_space space, typename key_t, typename value_t>
index_t reduce_by_key(key_t const*, key_t*, value_t const*, value_t*, index_t)
{
  static_assert(cub_available<space>(), "reduce_by_key requires BOBA_CUDA_LIBS or BOBA_HIP_LIBS");
  return 0;
}

#endif // BOBA_CUDA_LIBS || BOBA_HIP_LIBS

} // namespace detail

} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include "BOBA/boba.hpp"

#ifdef BOBA_RAJA
#include "RAJA/RAJA.hpp"
#endif

namespace boba
{

namespace atomics
{

// TODO<documentation> use concepts to constrain types used in atomics to supported types

/**
 * @brief Atomically adds a value to a destination.
 * @tparam T Value type.
 * @param destination_ptr Destination pointer.
 * @param source_value Value to add.
 */
template <typename T>
__boba_device__
void atomic_add(T* destination_ptr, const T& source_value)
{
#ifdef BOBA_DEVICE_CODE
  atomicAdd(destination_ptr, source_value);
#else
  *destination_ptr += source_value;
#endif
}

#if defined(BOBA_CUDA) || defined(BOBA_HIP)
/**
 * @brief Atomically adds a single-precision complex value on GPU backends.
 * \warning This implementaton atomically adds the real and imaginary components separately, which may be unsafe.
 * @param destination_ptr Destination pointer.
 * @param source_value Value to add.
 */
__boba_host_device__ inline void atomic_add(boba::complex<float>* destination_ptr, const boba::complex<float>& source_value)
{
#ifdef BOBA_DEVICE_CODE
  atomicAdd(&destination_ptr->x, source_value.x);
  atomicAdd(&destination_ptr->y, source_value.y);
#else
  *destination_ptr += source_value;
#endif
}

/**
 * @brief Atomically adds a double-precision complex value on GPU backends. Warning! This
 * \warning This implementaton atomically adds the real and imaginary components separately, which may be unsafe.
 * @param destination_ptr Destination pointer.
 * @param source_value Value to add.
 */
__boba_host_device__ inline void atomic_add(boba::complex<double>* destination_ptr, const boba::complex<double>& source_value)
{
#ifdef BOBA_DEVICE_CODE
  atomicAdd(&destination_ptr->x, source_value.x);
  atomicAdd(&destination_ptr->y, source_value.y);
#else
  *destination_ptr += source_value;
#endif
}
#endif

/**
 * @brief Atomically updates a destination with the maximum value.
 * @tparam T Value type.
 * @param destination_ptr Destination pointer.
 * @param source_value Candidate maximum.
 */
template <typename T>
__boba_device__
void atomic_max(T* destination_ptr, const T& source_value)
{
#ifdef BOBA_DEVICE_CODE
  atomicMax(destination_ptr, source_value);
#else
  *destination_ptr = boba::max(*destination_ptr, source_value);
#endif
}

/**
 * @brief Atomically updates a destination with the minimum value.
 * @tparam T Value type.
 * @param destination_ptr Destination pointer.
 * @param source_value Candidate minimum.
 */
template <typename T>
__boba_device__
void atomic_min(T* destination_ptr, const T& source_value)
{
#ifdef BOBA_DEVICE_CODE
  atomicMin(destination_ptr, source_value);
#else
  *destination_ptr = boba::min(*destination_ptr, source_value);
#endif
}

template <::boba::execution_space space>
struct SpaceAtomicTraits;

#ifdef BOBA_RAJA

template <>
struct SpaceAtomicTraits<::boba::execution_space::CPU>
{
  using raja_atomic_policy = RAJA::seq_atomic;
};

#ifdef BOBA_CUDA
template <>
struct SpaceAtomicTraits<::boba::execution_space::CUDA>
{
  using raja_atomic_policy = RAJA::cuda_atomic;
};
#endif

#ifdef BOBA_HIP
template <>
struct SpaceAtomicTraits<::boba::execution_space::HIP>
{
  using raja_atomic_policy = RAJA::hip_atomic;
};
#endif

template <::boba::execution_space space, typename _data_t>
struct ReferenceSelector
{
  using type = RAJA::AtomicRef<_data_t, typename SpaceAtomicTraits<space>::raja_atomic_policy>;
};

#if defined(BOBA_CUDA) || defined(BOBA_HIP)
template <typename T>
struct DeviceComplexReference
{
  using value_type = T;

  __boba_host_device__ constexpr explicit DeviceComplexReference(value_type* value_ptr)
      : m_value_ptr(value_ptr)
  {
  }

  __boba_host_device__
  value_type
  load() const
  {
    return *m_value_ptr;
  }

  __boba_host_device__
  operator value_type() const
  {
    return load();
  }

  __boba_host_device__
  value_type
  operator=(value_type rhs) const
  {
    *m_value_ptr = rhs;
    return rhs;
  }

  __boba_host_device__
  value_type
  operator+=(value_type rhs) const
  {
    atomic_add(m_value_ptr, rhs);
    return load();
  }

  __boba_host_device__
  value_type
  operator-=(value_type rhs) const
  {
    value_type negative_rhs{-rhs.x, -rhs.y};
    atomic_add(m_value_ptr, negative_rhs);
    return load();
  }

private:
  value_type* m_value_ptr;
};

#ifdef BOBA_CUDA
template <>
struct ReferenceSelector<::boba::execution_space::CUDA, boba::complex<float>>
{
  using type = DeviceComplexReference<boba::complex<float>>;
};

template <>
struct ReferenceSelector<::boba::execution_space::CUDA, boba::complex<double>>
{
  using type = DeviceComplexReference<boba::complex<double>>;
};
#endif

#ifdef BOBA_HIP
template <>
struct ReferenceSelector<::boba::execution_space::HIP, boba::complex<float>>
{
  using type = DeviceComplexReference<boba::complex<float>>;
};

template <>
struct ReferenceSelector<::boba::execution_space::HIP, boba::complex<double>>
{
  using type = DeviceComplexReference<boba::complex<double>>;
};
#endif
#endif

template <::boba::execution_space space, typename _data_t>
using space_atomic_reference = typename ReferenceSelector<space, _data_t>::type;

#else

template <>
struct SpaceAtomicTraits<::boba::execution_space::CPU>
{
};

/*!
 * \brief Atomic wrapper object, but actually non-atomic.
 *
 * Based on RAJA::AtomicRef but changed to only do RAJA::seq_atomic non-atomic
 * operations. Therefore this is only compatible with CPU execution.
 *
 * Provides an interface akin to that provided by std::atomic, but for an
 * arbitrary memory location.
 *
 * This is used on GPUs when you haven't set BOBA_RAJA
 */
template <typename T, ::boba::execution_space space>
struct Reference
{
  /* TODO, put this error back
  static_assert(space == ::boba::execution_space::CPU,
      "atomics::Reference only works with CPU");
  */

  using value_type = T;

  /**
   * @brief Wraps an existing value as an atomic-like reference.
   * @param value_ptr Referenced value pointer.
   */
  __boba_host_device__ constexpr explicit Reference(value_type* value_ptr)
      : m_value_ptr(value_ptr)
  {
    if constexpr (space != ::boba::execution_space::CPU)
    {
      boba_always_assert_equal(space, ::boba::execution_space::CPU, "atomics::Reference fallback implementation only works with CPU. Build with RAJA to enable correct implementations guarded by BOBA_RAJA for GPUs.");
    }
  }

  /**
   * \brief copy constructor
   */
  Reference(Reference const&) = default;

  /**
   * \brief move constructor
   */
  Reference(Reference&&) = default;

  /**
   * \brief copy assignment operator
   */
  __boba_host_device__ Reference& operator=(Reference const&) = delete;

  /**
   * \brief move assignment operator
   */
  __boba_host_device__ Reference& operator=(Reference&&) = delete;

  /**
   * @brief Returns the wrapped pointer.
   * @return Wrapped pointer.
   */
  __boba_host_device__ value_type* getPointer() const
  {
    return m_value_ptr;
  }

  /**
   * @brief Stores a new value.
   * @param rhs Replacement value.
   */
  __boba_host_device__ void store(value_type rhs) const
  {
    *m_value_ptr = rhs;
  }

  /**
   * @brief Assigns a new value.
   * @param rhs Replacement value.
   * @return Stored value.
   */
  __boba_host_device__
  value_type
  operator=(value_type rhs) const
  {
    return *m_value_ptr = rhs;
  }

  /**
   * @brief Loads the current value.
   * @return Current value.
   */
  __boba_host_device__
  value_type
  load() const
  {
    return *m_value_ptr;
  }

  /**
   * @brief Converts to the wrapped value type.
   * @return Current value.
   */
  __boba_host_device__
  operator value_type() const
  {
    return *m_value_ptr;
  }

  /**
   * @brief Replaces the value and returns the previous one.
   * @param rhs Replacement value.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  exchange(value_type rhs) const
  {
    value_type old = *m_value_ptr;
    *m_value_ptr = rhs;
    return old;
  }

  //! See: https://en.wikipedia.org/wiki/Compare-and-swap
  /**
   * @brief Performs a compare-and-swap operation.
   * @param compare Expected old value.
   * @param rhs Replacement value.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  CAS(value_type compare, value_type rhs) const
  {
    value_type old = *m_value_ptr;
    if (compare == old)
    {
      *m_value_ptr = rhs;
    }
    return old;
  }

  /**
   * @brief Performs a strong compare-and-exchange.
   * @param expect Expected value, updated on failure.
   * @param rhs Replacement value.
   * @return `true` when the exchange succeeds.
   */
  __boba_host_device__ bool compare_exchange_strong(value_type& expect, value_type rhs) const
  {
    value_type old = *m_value_ptr;
    value_type compare = expect;
    if (compare == old)
    {
      *m_value_ptr = rhs;
      return true;
    }
    else
    {
      expect = old;
      return false;
    }
  }

  /**
   * @brief Performs a weak compare-and-exchange.
   * @param expect Expected value, updated on failure.
   * @param rhs Replacement value.
   * @return `true` when the exchange succeeds.
   */
  __boba_host_device__ bool compare_exchange_weak(value_type& expect, value_type rhs) const
  {
    return this->compare_exchange_strong(expect, rhs);
  }

  /**
   * @brief Prefix-increments the value.
   * @return Incremented value.
   */
  __boba_host_device__
  value_type
  operator++() const
  {
    return ++(*m_value_ptr);
  }

  /**
   * @brief Postfix-increments the value.
   * @param Unused postfix marker.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  operator++(int) const
  {
    return (*m_value_ptr)++;
  }

  /**
   * @brief Prefix-decrements the value.
   * @return Decremented value.
   */
  __boba_host_device__
  value_type
  operator--() const
  {
    return --(*m_value_ptr);
  }

  /**
   * @brief Postfix-decrements the value.
   * @param Unused postfix marker.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  operator--(int) const
  {
    return (*m_value_ptr)--;
  }

  /**
   * @brief Adds a value and returns the previous one.
   * @param rhs Value to add.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  fetch_add(value_type rhs) const
  {
    value_type old = *m_value_ptr;
    *m_value_ptr = old + rhs;
    return old;
  }

  /**
   * @brief Adds a value in place.
   * @param rhs Value to add.
   * @return Updated value.
   */
  __boba_host_device__
  value_type
  operator+=(value_type rhs) const
  {
    return *m_value_ptr += rhs;
  }

  /**
   * @brief Subtracts a value and returns the previous one.
   * @param rhs Value to subtract.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  fetch_sub(value_type rhs) const
  {
    value_type old = *m_value_ptr;
    *m_value_ptr = old - rhs;
    return old;
  }

  /**
   * @brief Subtracts a value in place.
   * @param rhs Value to subtract.
   * @return Updated value.
   */
  __boba_host_device__
  value_type
  operator-=(value_type rhs) const
  {
    return *m_value_ptr -= rhs;
  }

  /**
   * @brief Replaces the value with the minimum and returns the previous one.
   * @param rhs Candidate minimum.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  fetch_min(value_type rhs) const
  {
    value_type old = *m_value_ptr;
    if (rhs < old)
    {
      *m_value_ptr = rhs;
    }
    return old;
  }

  /**
   * @brief Replaces the value with the minimum and returns the result.
   * @param rhs Candidate minimum.
   * @return Stored minimum value.
   */
  __boba_host_device__
  value_type
  min(value_type rhs) const
  {
    value_type old = *m_value_ptr;
    if (rhs < old)
    {
      *m_value_ptr = rhs;
      return rhs;
    }
    return old;
  }

  /**
   * @brief Replaces the value with the maximum and returns the previous one.
   * @param rhs Candidate maximum.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  fetch_max(value_type rhs) const
  {
    value_type old = *m_value_ptr;
    if (old < rhs)
    {
      *m_value_ptr = rhs;
    }
    return old;
  }

  /**
   * @brief Replaces the value with the maximum and returns the result.
   * @param rhs Candidate maximum.
   * @return Stored maximum value.
   */
  __boba_host_device__
  value_type
  max(value_type rhs) const
  {
    value_type old = *m_value_ptr;
    if (old < rhs)
    {
      *m_value_ptr = rhs;
      return rhs;
    }
    return old;
  }

  /**
   * @brief Applies bitwise AND and returns the previous value.
   * @param rhs Bitmask operand.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  fetch_and(value_type rhs) const
  {
    value_type old = *m_value_ptr;
    *m_value_ptr = old & rhs;
    return old;
  }

  /**
   * @brief Applies bitwise AND in place.
   * @param rhs Bitmask operand.
   * @return Updated value.
   */
  __boba_host_device__
  value_type
  operator&=(value_type rhs) const
  {
    return *m_value_ptr &= rhs;
  }

  /**
   * @brief Applies bitwise OR and returns the previous value.
   * @param rhs Bitmask operand.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  fetch_or(value_type rhs) const
  {
    value_type old = *m_value_ptr;
    *m_value_ptr = old | rhs;
    return old;
  }

  /**
   * @brief Applies bitwise OR in place.
   * @param rhs Bitmask operand.
   * @return Updated value.
   */
  __boba_host_device__
  value_type
  operator|=(value_type rhs) const
  {
    return *m_value_ptr |= rhs;
  }

  /**
   * @brief Applies bitwise XOR and returns the previous value.
   * @param rhs Bitmask operand.
   * @return Previous value.
   */
  __boba_host_device__
  value_type
  fetch_xor(value_type rhs) const
  {
    value_type old = *m_value_ptr;
    *m_value_ptr = old ^ rhs;
    return old;
  }

  /**
   * @brief Applies bitwise XOR in place.
   * @param rhs Bitmask operand.
   * @return Updated value.
   */
  __boba_host_device__
  value_type
  operator^=(value_type rhs) const
  {
    return *m_value_ptr ^= rhs;
  }

private:
  value_type* m_value_ptr;
};

template <::boba::execution_space space, typename _data_t>
using space_atomic_reference = Reference<_data_t, space>;

#endif

/**
 * \brief
 * get an atomic reference
\verbatim
 atomic::reference<double, execution_space::CPU>(data[i]) += 1.0;
\endverbatim
 * @tparam space Execution space.
 * @tparam data_t Referenced value type.
 * @param value Referenced value.
 * @return Atomic reference wrapper for `value`.
 */
template <::boba::execution_space space,
          typename _data_t>
__boba_host_device__ auto reference(_data_t& value)
{
  return space_atomic_reference<space, _data_t>{&value};
}

/**
 * \brief
 * Atomic Accessor usable with views
\verbatim
 auto view = TensorView<atomic::Accessor<execution_space::CPU, double>, 1>();
\endverbatim
 * @tparam space Execution space.
 * @tparam data_t Value type.
 */
template <::boba::execution_space space, typename _data_t>
struct Accessor
{
  using data_t = _data_t;
  using data_reference = space_atomic_reference<space, data_t>;
  using data_pointer = data_t*;

  /**
   * \brief default constructor
   */
  constexpr Accessor() noexcept = default;

  /**
   * @brief Returns an atomic reference to an indexed element.
   * @param p Base pointer.
   * @param i Element index.
   * @return Atomic reference to `p[i]`.
   */
  __boba_host_device__ constexpr data_reference access(data_pointer p, index_t i) const noexcept
  {
    return data_reference{p + i};
  }
  /**
   * @brief Returns an offset pointer.
   * @param p Base pointer.
   * @param i Element index.
   * @return `p + i`.
   */
  __boba_host_device__ constexpr data_pointer offset(data_pointer p, index_t i) const noexcept
  {
    return p + i;
  }
};

} // namespace atomics

} // namespace boba

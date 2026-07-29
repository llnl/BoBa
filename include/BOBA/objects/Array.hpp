// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <sstream>
#include <utility>
#include <vector>

namespace boba
{

/**
 * \brief
 * BoBa-specific re-implementation of std::array.
 */

template <typename T, std::size_t N>
struct Array
{
  //
  // type aliases
  //
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  T array_data[N];

  //
  // use implicit constructor, destructor, and assignment operators
  //

  //
  // capacity
  //
  /**
   * @brief Returns whether the array is empty.
   * @return `true` when `N` is zero.
   */
  __boba_host_device__ constexpr bool empty() const noexcept
  {
    return N == 0;
  }

  /**
   * @brief Returns the number of elements.
   * @return The compile-time extent `N`.
   */
  __boba_host_device__ constexpr size_type size() const noexcept
  {
    return N;
  }

  /**
   * @brief Returns the element at a checked index.
   * @param i Zero-based element index.
   * @return A reference to the element at `i`.
   */
  __boba_host_device__
  reference
  at(size_type i)
  {
    boba_always_assert_nonnegative(i, "Out of bounds");
    boba_always_assert_lt(i, N, "Out of bounds");
    return array_data[i];
  }

  /**
   * @brief Returns the element at a checked index.
   * @param i Zero-based element index.
   * @return A const reference to the element at `i`.
   */
  __boba_host_device__
  const_reference
  at(size_type i) const
  {
    boba_always_assert_nonnegative(i, "Out of bounds");
    boba_always_assert_lt(i, N, "Out of bounds");
    return array_data[i];
  }

  /**
   * @brief Returns the element at an unchecked index.
   * @param i Zero-based element index.
   * @return A reference to the element at `i`.
   */
  __boba_host_device__ constexpr reference operator[](size_type i)
  {
    // TODO<implementation details> these are not constexpr
    // boba_assert_ge(i, 0, "Out of bounds");
    // boba_assert_lt(i, N, "Out of bounds");
    return array_data[i];
  }

  /**
   * @brief Returns the element at an unchecked index.
   * @param i Zero-based element index.
   * @return A const reference to the element at `i`.
   */
  __boba_host_device__ constexpr const_reference operator[](size_type i) const
  {
    // TODO<implementation details> these are not constexpr
    // boba_assert_ge(i, 0, "Out of bounds");
    // boba_assert_lt(i, N, "Out of bounds");
    return array_data[i];
  }

  /**
   * @brief Adds another array elementwise.
   * @param a Array to add.
   * @return This array after the addition.
   */
  __boba_host_device__ constexpr Array& operator+=(Array a)
  {
    for (size_t i = 0; i < N; i++)
    {
      this->operator[](i) += a[i];
    }
    return *this;
  }

  /**
   * @brief Returns the elementwise sum with another array.
   * @param a Array to add.
   * @return The elementwise sum.
   */
  __boba_host_device__ constexpr Array operator+(Array a) const
  {
    Array c{*this};
    c += a;
    return c;
  }

  /**
   * @brief Subtracts another array elementwise.
   * @param a Array to subtract.
   * @return This array after the subtraction.
   */
  __boba_host_device__ constexpr Array& operator-=(Array a)
  {
    for (size_t i = 0; i < N; i++)
    {
      this->operator[](i) -= a[i];
    }
    return *this;
  }

  /**
   * @brief Returns the elementwise difference with another array.
   * @param a Array to subtract.
   * @return The elementwise difference.
   */
  __boba_host_device__ constexpr Array operator-(Array a) const
  {
    Array c{*this};
    c -= a;
    return c;
  }

  /**
   * @brief Multiplies by another array elementwise.
   * @param a Array of multipliers.
   * @return This array after the multiplication.
   */
  __boba_host_device__ constexpr Array& operator*=(Array a)
  {
    for (size_t i = 0; i < N; i++)
    {
      this->operator[](i) *= a[i];
    }
    return *this;
  }

  /**
   * @brief Returns the elementwise product with another array.
   * @param a Array of multipliers.
   * @return The elementwise product.
   */
  __boba_host_device__ constexpr Array operator*(Array a) const
  {
    Array c{*this};
    c *= a;
    return c;
  }

  /**
   * @brief Multiplies every element by a scalar.
   * @param a Scalar multiplier.
   * @return This array after scaling.
   */
  __boba_host_device__ constexpr Array& operator*=(value_type a)
  {
    for (size_t i = 0; i < N; i++)
    {
      this->operator[](i) *= a;
    }
    return *this;
  }

  /**
   * @brief Returns a scaled copy of the array.
   * @param a Scalar multiplier.
   * @return The scaled array.
   */
  __boba_host_device__ constexpr Array operator*(value_type a) const
  {
    Array c{*this};
    c *= a;
    return c;
  }

  /**
   * @brief Divides by another array elementwise.
   * @param a Array of divisors.
   * @return This array after the division.
   */
  __boba_host_device__ constexpr Array& operator/=(const Array a)
  {
    for (size_t i = 0; i < N; i++)
    {
      this->operator[](i) /= a[i];
    }
    return *this;
  }

  /**
   * @brief Divides every element by a scalar.
   * @param a Scalar divisor.
   * @return This array after scaling.
   */
  __boba_host_device__ constexpr Array& operator/=(value_type a)
  {
    for (size_t i = 0; i < N; i++)
    {
      this->operator[](i) /= a;
    }
    return *this;
  }

  /**
   * @brief Returns a copy divided by a scalar.
   * @param a Scalar divisor.
   * @return The scaled array.
   */
  __boba_host_device__ constexpr Array operator/(value_type a) const
  {
    Array c{*this};
    c /= a;
    return c;
  }

  /**
   * @brief Assigns the same scalar value to every element.
   * @param a Value to store in each element.
   * @return This array after the assignment.
   */
  __boba_host_device__ constexpr Array& operator=(value_type a)
  {
    for (size_t i = 0; i < N; i++)
    {
      this->operator[](i) = a;
    }
    return *this;
  }

  /**
   * @brief Returns the first element.
   * @return A reference to the first element.
   */
  __boba_host_device__ constexpr reference front()
  {
    return array_data[0];
  }

  /**
   * @brief Returns the first element.
   * @return A const reference to the first element.
   */
  __boba_host_device__ constexpr const_reference front() const
  {
    return array_data[0];
  }

  /**
   * @brief Returns the last element.
   * @return A reference to the last element.
   */
  __boba_host_device__ constexpr reference back()
  {
    return array_data[N - 1];
  }

  /**
   * @brief Returns the last element.
   * @return A const reference to the last element.
   */
  __boba_host_device__ constexpr const_reference back() const
  {
    return array_data[N - 1];
  }

  /**
   * @brief Returns a pointer to the underlying storage.
   * @return Pointer to the first element.
   */
  __boba_host_device__ constexpr pointer data() noexcept
  {
    return array_data;
  }

  /**
   * @brief Returns a pointer to the underlying storage.
   * @return Const pointer to the first element.
   */
  __boba_host_device__ constexpr const_pointer data() const noexcept
  {
    return array_data;
  }

  /**
   * @brief Returns a const pointer to the underlying storage.
   * @return Const pointer to the first element.
   */
  __boba_host_device__ constexpr const_pointer const_data() const noexcept
  {
    return array_data;
  }

  //
  // iterators
  //
  /**
   * @brief Returns an iterator to the first element.
   * @return Iterator to the beginning of the array.
   */
  __boba_host_device__ constexpr iterator begin() noexcept
  {
    return data();
  }

  /**
   * @brief Returns an iterator to the first element.
   * @return Const iterator to the beginning of the array.
   */
  __boba_host_device__ constexpr const_iterator begin() const noexcept
  {
    return data();
  }

  /**
   * @brief Returns a const iterator to the first element.
   * @return Const iterator to the beginning of the array.
   */
  __boba_host_device__ constexpr const_iterator const_begin() const noexcept
  {
    return data();
  }

  /**
   * @brief Returns an iterator one past the last element.
   * @return Iterator to the end of the array.
   */
  __boba_host_device__ constexpr iterator end() noexcept
  {
    return begin() + N;
  }

  /**
   * @brief Returns an iterator one past the last element.
   * @return Const iterator to the end of the array.
   */
  __boba_host_device__ constexpr const_iterator end() const noexcept
  {
    return begin() + N;
  }

  /**
   * @brief Returns a const iterator one past the last element.
   * @return Const iterator to the end of the array.
   */
  __boba_host_device__ constexpr const_iterator const_end() const noexcept
  {
    return const_begin() + N;
  }

  //
  // modifiers
  //
  /**
   * @brief Fills the array with one value.
   * @param value Value to assign to each element.
   */
  __boba_host_device__ constexpr void fill(const T& value)
  {
    for (size_type i = 0; i < N; ++i)
    {
      array_data[i] = value;
    }
  }
};

/**
 * \brief
 * Specialization of Array for when N = 0 (the empty case).
 */

template <typename T>
struct Array<T, 0>
{
  //
  // type aliases
  //
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  T* array_data = nullptr;

  //
  // use implicit constructor, destructor, and assignment operators
  //

  //
  // capacity
  //
  /**
   * @brief Returns whether the array is empty.
   * @return Always `true`.
   */
  __boba_host_device__ constexpr bool empty() const noexcept
  {
    return true;
  }

  /**
   * @brief Returns the number of elements.
   * @return Always `0`.
   */
  __boba_host_device__ constexpr size_type size() const noexcept
  {
    return 0;
  }

  //
  // data access
  //
  /**
   * @brief Reports out-of-bounds access on an empty array.
   * @param i Ignored element index.
   * @return A reference to invalid storage.
   */
  __boba_host_device__
  reference
  at(size_type)
  {
    boba_error("Out of bounds");
    return array_data[0];
  }

  /**
   * @brief Reports out-of-bounds access on an empty array.
   * @param i Ignored element index.
   * @return A const reference to invalid storage.
   */
  __boba_host_device__
  const_reference
  at(size_type) const
  {
    boba_error("Out of bounds");
    return array_data[0];
  }

  /**
   * @brief Returns the requested element from empty storage.
   * @param i Ignored element index.
   * @return A reference to invalid storage.
   */
  __boba_host_device__ constexpr reference operator[](size_type)
  {
    return array_data[0];
  }

  /**
   * @brief Returns the requested element from empty storage.
   * @param i Ignored element index.
   * @return A const reference to invalid storage.
   */
  __boba_host_device__ constexpr const_reference operator[](size_type) const
  {
    return array_data[0];
  }

  /**
   * @brief Returns the first element from empty storage.
   * @return A reference to invalid storage.
   */
  __boba_host_device__ constexpr reference front()
  {
    return array_data[0];
  }

  /**
   * @brief Returns the first element from empty storage.
   * @return A const reference to invalid storage.
   */
  __boba_host_device__ constexpr const_reference front() const
  {
    return array_data[0];
  }

  /**
   * @brief Returns the last element from empty storage.
   * @return A reference to invalid storage.
   */
  __boba_host_device__ constexpr reference back()
  {
    return array_data[0];
  }

  /**
   * @brief Returns the last element from empty storage.
   * @return A const reference to invalid storage.
   */
  __boba_host_device__ constexpr const_reference back() const
  {
    return array_data[0];
  }

  /**
   * @brief Returns the storage pointer.
   * @return The null storage pointer.
   */
  __boba_host_device__ constexpr pointer data() noexcept
  {
    return array_data;
  }

  /**
   * @brief Returns the storage pointer.
   * @return The null const storage pointer.
   */
  __boba_host_device__ constexpr const_pointer data() const noexcept
  {
    return array_data;
  }

  /**
   * @brief Returns the storage pointer.
   * @return The null const storage pointer.
   */
  __boba_host_device__ constexpr const_pointer const_data() const noexcept
  {
    return array_data;
  }

  //
  // iterators
  //
  /**
   * @brief Returns an iterator to the beginning.
   * @return Iterator to empty storage.
   */
  __boba_host_device__ constexpr iterator begin() noexcept
  {
    return data();
  }

  /**
   * @brief Returns an iterator to the beginning.
   * @return Const iterator to empty storage.
   */
  __boba_host_device__ constexpr const_iterator begin() const noexcept
  {
    return data();
  }

  /**
   * @brief Returns a const iterator to the beginning.
   * @return Const iterator to empty storage.
   */
  __boba_host_device__ constexpr const_iterator const_begin() const noexcept
  {
    return data();
  }

  /**
   * @brief Returns an iterator to the end.
   * @return Iterator to empty storage.
   */
  __boba_host_device__ constexpr iterator end() noexcept
  {
    return begin();
  }

  /**
   * @brief Returns an iterator to the end.
   * @return Const iterator to empty storage.
   */
  __boba_host_device__ constexpr const_iterator end() const noexcept
  {
    return begin();
  }

  /**
   * @brief Returns a const iterator to the end.
   * @return Const iterator to empty storage.
   */
  __boba_host_device__ constexpr const_iterator const_end() const noexcept
  {
    return const_begin();
  }

  //
  // modifiers
  //
  /**
   * @brief Ignores fill requests on an empty array.
   * @param value Ignored fill value.
   */
  __boba_host_device__ constexpr void fill(const T&)
  {
  }
};

//
// Operators
//
/**
 * @brief Adds a scalar to each array entry.
 * @param x Scalar addend.
 * @param a Array operand.
 * @return The elementwise sum.
 */
template <typename scalar_t, class T, std::size_t N>
__boba_host_device__ constexpr Array<scalar_t, N> operator+(scalar_t x, const Array<T, N> a)
{
  Array<scalar_t, N> b{0};
  for (size_t d = 0; d < N; d++)
  {
    b[d] = static_cast<scalar_t>(a[d]) + x;
  }
  return b;
}

/**
 * @brief Adds a scalar to each array entry.
 * @param x Scalar addend.
 * @param a Array operand.
 * @return The elementwise sum.
 */
template <typename scalar_t, class T, std::size_t N>
__boba_host_device__ constexpr Array<scalar_t, N> operator+(const Array<T, N> a, scalar_t x)
{
  return x + a;
}

/**
 * @brief Subtracts each array entry from a scalar.
 * @param x Scalar minuend.
 * @param a Array operand.
 * @return The elementwise difference.
 */
template <typename scalar_t, class T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> operator-(scalar_t x, Array<T, N> a)
{
  Array<T, N> c{0};
  for (size_t i = 0; i < N; i++)
  {
    c[i] = x - a[i];
  }
  return c;
}

/**
 * @brief Subtracts a scalar from each array entry.
 * @param a Array operand.
 * @param x Scalar subtrahend.
 * @return The elementwise difference.
 */
template <typename scalar_t, class T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> operator-(Array<T, N> a, scalar_t x)
{
  Array<T, N> c{0};
  for (size_t i = 0; i < N; i++)
  {
    c[i] = a[i] - x;
  }
  return c;
}

/**
 * @brief Multiplies each array entry by a scalar.
 * @param x Scalar multiplier.
 * @param a Array operand.
 * @return The scaled array.
 */
template <typename scalar_t, class T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> operator*(scalar_t x, Array<T, N> a)
{
  auto b = a;
  b *= static_cast<T>(x);
  return b;
}

/**
 * @brief Multiplies each array entry by a scalar.
 * @param a Array operand.
 * @param x Scalar multiplier.
 * @return The scaled array.
 */
template <typename scalar_t, class T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> operator*(Array<T, N> a, scalar_t x)
{
  return x * a;
}

/**
 * @brief Divides each array entry by a scalar.
 * @param a Array operand.
 * @param x Scalar divisor.
 * @return The scaled array.
 */
template <class T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> operator/(Array<T, N> a, T x)
{
  return (static_cast<T>(1.0) / x) * a;
}

/**
 * @brief Applies `abs` to every array entry.
 * @param a Array operand.
 * @return The elementwise absolute value.
 */
template <class T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> abs(Array<T, N> a)
{
  auto b = a;
  for (size_t d = 0_z; d < N; d++)
  {
    b[d] = abs(b[d]);
  }
  return b;
}

//
// Array get functions for structured binding
//
/**
 * @brief Returns an array element for structured binding.
 * @tparam I Zero-based element index.
 * @param a Array to access.
 * @return A reference to element `I`.
 */
template <std::size_t I, class T, std::size_t N>
__boba_host_device__ constexpr T& get(Array<T, N>& a) noexcept
{
  return a[I];
}

/**
 * @brief Returns an array element for structured binding.
 * @tparam I Zero-based element index.
 * @param a Array to access.
 * @return A const reference to element `I`.
 */
template <std::size_t I, class T, std::size_t N>
__boba_host_device__ constexpr T const& get(Array<T, N> const& a) noexcept
{
  return a[I];
}

/**
 * @brief Returns an rvalue array element for structured binding.
 * @tparam I Zero-based element index.
 * @param a Array to access.
 * @return An rvalue reference to element `I`.
 */
template <std::size_t I, class T, std::size_t N>
__boba_host_device__ constexpr T&& get(Array<T, N>&& a) noexcept
{
  return static_cast<T&&>(a[I]);
}

/**
 * @brief Returns a const rvalue array element for structured binding.
 * @tparam I Zero-based element index.
 * @param a Array to access.
 * @return A const rvalue reference to element `I`.
 */
template <std::size_t I, class T, std::size_t N>
__boba_host_device__ constexpr T const&& get(Array<T, N> const&& a) noexcept
{
  return static_cast<T const&&>(a[I]);
}

namespace detail
{
//
// Array folding functions
//
/**
 * @brief Compares two arrays elementwise for exact equality.
 * @param lhs Left array.
 * @param rhs Right array.
 * @param indices Expansion indices.
 * @return `true` when all selected entries are equal.
 */
template <typename T, std::size_t N, std::size_t... Is>
__boba_host_device__ constexpr bool equal(Array<T, N> const& lhs,
                                          Array<T, N> const& rhs,
                                          std::index_sequence<Is...>)
{
  return (true && ... && (lhs[Is] == rhs[Is]));
}

//
// Array folding functions
//
/**
 * @brief Compares two `double` arrays using tiny-difference checks.
 * @param lhs Left array.
 * @param rhs Right array.
 * @param indices Expansion indices.
 * @return `true` when all selected entries differ only by a tiny amount.
 */
template <std::size_t N, std::size_t... Is>
__boba_host_device__ constexpr bool equal(Array<double, N> const& lhs,
                                          Array<double, N> const& rhs,
                                          std::index_sequence<Is...>)
{
  return (true && ... && ::boba::is_tiny(lhs[Is] - rhs[Is]));
}

/**
 * @brief Performs a lexicographical comparison.
 * @param lhs Left array.
 * @param rhs Right array.
 * @return `true` when `lhs` is lexicographically smaller than `rhs`.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr bool lexicographical_less_than(Array<T, N> const& lhs,
                                                              Array<T, N> const& rhs)
{
  for (std::size_t i = 0; i < N; ++i)
  {
    if (lhs[i] < rhs[i])
    {
      return true;
    }
    if (rhs[i] < lhs[i])
    {
      return false;
    }
  }
  return false;
}

} // namespace detail

//
// Array comparison functions
//
/**
 * @brief Tests two arrays for equality.
 * @param lhs Left array.
 * @param rhs Right array.
 * @return `true` when all entries are equal.
 */
template <class T, std::size_t N>
__boba_host_device__ constexpr bool operator==(Array<T, N> const& lhs,
                                               Array<T, N> const& rhs)
{
  if constexpr (N > 0)
  {
    for (size_t d = 0_z; d < N; d++)
    {
      if (lhs[d] != rhs[d])
      {
        return false;
      }
    }
    return true;
  }
  return true;
}

/**
 * @brief Tests two `double` arrays for near-equality.
 * @param lhs Left array.
 * @param rhs Right array.
 * @return `true` when all entries differ only by a tiny amount.
 */
template <std::size_t N>
__boba_host_device__ constexpr bool operator==(Array<double, N> const& lhs,
                                               Array<double, N> const& rhs)
{
  if constexpr (N > 0)
  {
    for (size_t d = 0; d < N; d++)
    {
      auto diff = ::fabs(lhs[d] - rhs[d]);
      if (not(::boba::is_tiny(diff)))
      {
        return false;
      }
    }
    return true;
  }
  return true;
}

/**
 * @brief Tests two arrays for inequality.
 * @param lhs Left array.
 * @param rhs Right array.
 * @return `true` when any entry differs.
 */
template <class T, std::size_t N>
__boba_host_device__ constexpr bool operator!=(Array<T, N> const& lhs,
                                               Array<T, N> const& rhs)
{
  if constexpr (N > 0)
  {
    return not(lhs == rhs);
  }
  return false;
}

/**
 * @brief Tests whether every entry in one array is strictly less than the other.
 * @param lhs Left array.
 * @param rhs Right array.
 * @return `true` when `lhs[d] < rhs[d]` for every `d`.
 */
template <class T, std::size_t N>
__boba_host_device__ constexpr bool operator<(Array<T, N> const& lhs,
                                              Array<T, N> const& rhs)
{
  if constexpr (N > 0)
  {
    for (size_t d = 0; d < N; d++)
    {
      if (lhs[d] >= rhs[d])
      {
        return false;
      }
    }
    return true;
  }
  return false;
}

/**
 * @brief Tests whether every entry in one array is less than or equal to the other.
 * @param lhs Left array.
 * @param rhs Right array.
 * @return `true` when `lhs[d] <= rhs[d]` for every `d`.
 */
template <class T, std::size_t N>
__boba_host_device__ constexpr bool operator<=(Array<T, N> const& lhs,
                                               Array<T, N> const& rhs)
{
  if constexpr (N > 0)
  {
    for (size_t d = 0; d < N; d++)
    {
      if (lhs[d] > rhs[d])
      {
        return false;
      }
    }
    return true;
  }
  return true;
}

/**
 * @brief Tests whether every entry in one array is strictly greater than the other.
 * @param lhs Left array.
 * @param rhs Right array.
 * @return `true` when `lhs[d] > rhs[d]` for every `d`.
 */
template <class T, std::size_t N>
__boba_host_device__ constexpr bool operator>(Array<T, N> const& lhs,
                                              Array<T, N> const& rhs)
{
  if constexpr (N > 0)
  {
    for (size_t d = 0; d < N; d++)
    {
      if (lhs[d] <= rhs[d])
      {
        return false;
      }
    }
    return true;
  }
  return false;
}

/**
 * @brief Tests whether every entry in one array is greater than or equal to the other.
 * @param lhs Left array.
 * @param rhs Right array.
 * @return `true` when `lhs[d] >= rhs[d]` for every `d`.
 */
template <class T, std::size_t N>
__boba_host_device__ constexpr bool operator>=(Array<T, N> const& lhs,
                                               Array<T, N> const& rhs)
{
  if constexpr (N > 0)
  {
    for (size_t d = 0; d < N; d++)
    {
      if (lhs[d] < rhs[d])
      {
        return false;
      }
    }
    return true;
  }
  return true;
}

/**
 * @brief Tests whether every array entry is less than a scalar.
 * @param lhs Array operand.
 * @param rhs Scalar bound.
 * @return `true` when every entry is less than `rhs`.
 */
template <class T, typename S, std::size_t N>
__boba_host_device__ constexpr bool operator<(Array<T, N> const& lhs,
                                              S const rhs)
{
  for (size_t d = 0; d < N; d++)
  {
    if (lhs[d] >= static_cast<T>(rhs))
    {
      return false;
    }
  }
  return true;
}

/**
 * @brief Tests whether every array entry is less than or equal to a scalar.
 * @param lhs Array operand.
 * @param rhs Scalar bound.
 * @return `true` when every entry is at most `rhs`.
 */
template <class T, typename S, std::size_t N>
__boba_host_device__ constexpr bool operator<=(Array<T, N> const& lhs,
                                               S const rhs)
{
  for (size_t d = 0; d < N; d++)
  {
    if (lhs[d] > static_cast<T>(rhs))
    {
      return false;
    }
  }
  return true;
}

/**
 * @brief Tests whether every array entry is greater than a scalar.
 * @param lhs Array operand.
 * @param rhs Scalar bound.
 * @return `true` when every entry is greater than `rhs`.
 */
template <class T, typename S, std::size_t N>
__boba_host_device__ constexpr bool operator>(Array<T, N> const& lhs,
                                              S const rhs)
{
  for (size_t d = 0; d < N; d++)
  {
    if (lhs[d] <= static_cast<T>(rhs))
    {
      return false;
    }
  }
  return true;
}

/**
 * @brief Tests whether every array entry is greater than or equal to a scalar.
 * @param lhs Array operand.
 * @param rhs Scalar bound.
 * @return `true` when every entry is at least `rhs`.
 */
template <class T, typename S, std::size_t N>
__boba_host_device__ constexpr bool operator>=(Array<T, N> const& lhs,
                                               S const rhs)
{
  for (size_t d = 0; d < N; d++)
  {
    if (lhs[d] < static_cast<T>(rhs))
    {
      return false;
    }
  }
  return true;
}

/**
 * @brief Tests whether a scalar is less than every array entry.
 * @param lhs Scalar bound.
 * @param rhs Array operand.
 * @return `true` when `lhs` is less than every entry in `rhs`.
 */
template <class T, typename S, std::size_t N>
__boba_host_device__ constexpr bool operator<(S const lhs,
                                              Array<T, N> const& rhs)
{
  return rhs > lhs;
}

/**
 * @brief Tests whether a scalar is less than or equal to every array entry.
 * @param lhs Scalar bound.
 * @param rhs Array operand.
 * @return `true` when `lhs` is at most every entry in `rhs`.
 */
template <class T, typename S, std::size_t N>
__boba_host_device__ constexpr bool operator<=(S const lhs,
                                               Array<T, N> const& rhs)
{
  return rhs >= lhs;
}

/**
 * @brief Tests whether a scalar is greater than every array entry.
 * @param lhs Scalar bound.
 * @param rhs Array operand.
 * @return `true` when `lhs` is greater than every entry in `rhs`.
 */
template <class T, typename S, std::size_t N>
__boba_host_device__ constexpr bool operator>(S const lhs,
                                              Array<T, N> const& rhs)
{
  return rhs < lhs;
}

/**
 * @brief Tests whether a scalar is greater than or equal to every array entry.
 * @param lhs Scalar bound.
 * @param rhs Array operand.
 * @return `true` when `lhs` is at least every entry in `rhs`.
 */
template <class T, typename S, std::size_t N>
__boba_host_device__ constexpr bool operator>=(S const lhs,
                                               Array<T, N> const& rhs)
{
  return rhs <= lhs;
}

//
// Array print functions
//
/**
 * @brief Writes an array to a stream.
 * @param stream Output stream.
 * @param rhs Array to print.
 * @return The output stream.
 */
template <class T, std::size_t N>
inline std::ostream& operator<<(std::ostream& stream, Array<T, N> const& rhs)
{
  if constexpr (N > 0)
  {
    stream << "Array{" << rhs[0];
    for (std::size_t i = 1; i < N; ++i)
    {
      stream << ", " << rhs[i];
    }
    stream << "}";
  }
  else
  {
    stream << "Array{}";
  }
  return stream;
}

} // namespace boba

namespace std
{

//
// specializations of std classes for structured binding
//
template <class T, std::size_t N>
struct tuple_size<::boba::Array<T, N>>
    : std::integral_constant<std::size_t, N>
{
};

template <std::size_t I, class T, std::size_t N>
struct tuple_element<I, ::boba::Array<T, N>>
{
  using type = T;
};

} // namespace std

namespace boba
{

namespace detail
{

/**
 * @brief Returns the first argument unchanged.
 * @param val Value to forward.
 * @param args Ignored trailing arguments.
 * @return `val` forwarded with its value category.
 */
template <typename T, typename... U>
__boba_host_device__ constexpr T&& first(T&& val, U&&...) noexcept
{
  return std::forward<T>(val);
}

/**
 * @brief Builds an array by invoking a generator for each index.
 * @param func Callable that accepts an index.
 * @param indices Expansion indices.
 * @return The generated array.
 */
template <std::size_t N, typename T, typename Func, std::size_t... Is>
__boba_host_device__ constexpr Array<T, N> make_array(Func&& func, std::index_sequence<Is...>)
{
  return {{func(Is)...}};
}

/**
 * @brief Copies a C array into a `boba::Array`.
 * @param a Source C array.
 * @param indices Expansion indices.
 * @return The copied array.
 */
template <typename T, std::size_t N, std::size_t... Is>
__boba_host_device__ constexpr Array<std::remove_cv_t<T>, N> to_array(T (&a)[N], std::index_sequence<Is...>)
{
  return {{a[Is]...}};
}

/**
 * @brief Casts a `boba::Array` to a new element type.
 * @param a Source array.
 * @param indices Expansion indices.
 * @return The cast array.
 */
template <typename T, typename U, std::size_t N, std::size_t... Is>
__boba_host_device__ constexpr Array<T, N> typed_array(Array<U, N> const& a, std::index_sequence<Is...>)
{
  return {{static_cast<T>(a[Is])...}};
}

/**
 * @brief Casts an rvalue `boba::Array` to a new element type.
 * @param a Source array.
 * @param indices Expansion indices.
 * @return The cast array.
 */
template <typename T, typename U, std::size_t N, std::size_t... Is>
__boba_host_device__ constexpr Array<T, N> typed_array(Array<U, N>&& a, std::index_sequence<Is...>)
{
  return {{static_cast<T>(std::move(a[Is]))...}};
}

/**
 * @brief Casts a C array to a `boba::Array`.
 * @param a Source C array.
 * @param indices Expansion indices.
 * @return The cast array.
 */
template <typename T, typename U, std::size_t N, std::size_t... Is>
__boba_host_device__ constexpr Array<T, N> typed_array(U (&a)[N], std::index_sequence<Is...>)
{
  return {{static_cast<T>(a[Is])...}};
}

/**
 * @brief Casts an rvalue C array to a `boba::Array`.
 * @param a Source C array.
 * @param indices Expansion indices.
 * @return The cast array.
 */
template <typename T, typename U, std::size_t N, std::size_t... Is>
__boba_host_device__ constexpr Array<T, N> typed_array(U (&&a)[N], std::index_sequence<Is...>)
{
  return {{static_cast<T>(std::move(a[Is]))...}};
}

/**
 * @brief Builds an array filled with one value.
 * @param val Fill value.
 * @param indices Expansion indices.
 * @return The filled array.
 */
template <std::size_t N, typename T, std::size_t... Is>
__boba_host_device__ constexpr Array<std::decay_t<T>, N> filled_array(T&& val, std::index_sequence<Is...>)
{
  return {{first(val, Is)...}};
}

//
// Array folding functions
//
/**
 * @brief Computes the sum of all array entries plus an initial value.
 * @param a Input array.
 * @param init Initial accumulator value.
 * @param indices Expansion indices.
 * @return The accumulated sum.
 */
template <typename T, std::size_t N, std::size_t... Is>
__boba_host_device__ constexpr T sum(Array<T, N> a, T init, std::index_sequence<Is...>)
{
  return (init + ... + a[Is]);
}

/**
 * @brief Computes the product of all array entries times an initial value.
 * @param a Input array.
 * @param init Initial accumulator value.
 * @param indices Expansion indices.
 * @return The accumulated product.
 */
template <typename T, std::size_t N, std::size_t... Is>
__boba_host_device__ constexpr T product(Array<T, N> a, T init, std::index_sequence<Is...>)
{
  return (init * ... * a[Is]);
}

} // namespace detail

/**
 * @brief boba Array dot product.
 * @param a Left array.
 * @param b Right array.
 * @return The dot product.
 */
template <typename T, size_t N>
__boba_host_device__ constexpr T dot(Array<T, N> a, Array<T, N> b)
{
  T sum = 0;
  for (size_t i = 0; i < N; i++)
  {
    sum += a[i] * b[i];
  }
  return sum;
}

//
// Array creation functions
//
/**
 * @brief Builds an array by invoking a generator for each index.
 * @param func Callable that accepts an index.
 * @return The generated array.
 */
template <std::size_t N, typename Func, typename T = std::remove_cv_t<decltype(std::declval<Func>()(N))>>
__boba_host_device__ constexpr Array<T, N> make_array(Func&& func)
{
  return ::boba::detail::make_array<N, T>(std::forward<Func>(func), std::make_index_sequence<N>{});
}

/**
 * @brief Converts an integer sequence to a runtime array.
 * @param sequence Integer sequence to convert.
 * @return An array containing the sequence values.
 */
template <typename T, T... Ints>
__boba_host_device__ constexpr Array<std::remove_cv_t<T>, sizeof...(Ints)> to_array(std::integer_sequence<T, Ints...>)
{
  return {Ints...};
}

/**
 * @brief Copies a C array into a runtime array.
 * @param a Source C array.
 * @return The copied array.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr Array<std::remove_cv_t<T>, N> to_array(T (&a)[N])
{
  return ::boba::detail::to_array(a, std::make_index_sequence<N>{});
}

/**
 * @brief Casts a runtime array to a new element type.
 * @param a Source array.
 * @return The cast array.
 */
template <typename T, typename U, std::size_t N>
__boba_host_device__ constexpr Array<T, N> typed_array(Array<U, N> const& a)
{
  return ::boba::detail::typed_array<T>(a, std::make_index_sequence<N>{});
}

/**
 * @brief Casts an rvalue runtime array to a new element type.
 * @param a Source array.
 * @return The cast array.
 */
template <typename T, typename U, std::size_t N>
__boba_host_device__ constexpr Array<T, N> typed_array(Array<U, N>&& a)
{
  return ::boba::detail::typed_array<T>(std::move(a), std::make_index_sequence<N>{});
}

/**
 * @brief Casts a C array to a runtime array.
 * @param a Source C array.
 * @return The cast array.
 */
template <typename T, typename U, std::size_t N>
__boba_host_device__ constexpr Array<T, N> typed_array(U (&a)[N])
{
  return ::boba::detail::typed_array<T>(a, std::make_index_sequence<N>{});
}

/**
 * @brief Casts an rvalue C array to a runtime array.
 * @param a Source C array.
 * @return The cast array.
 */
template <typename T, typename U, std::size_t N>
__boba_host_device__ constexpr Array<T, N> typed_array(U (&&a)[N])
{
  return ::boba::detail::typed_array<T>(std::move(a), std::make_index_sequence<N>{});
}

/**
 * @brief Builds an array filled with one value.
 * @param val Fill value.
 * @return An array containing `N` copies of `val`.
 */
template <std::size_t N, typename T>
__boba_host_device__ constexpr Array<std::decay_t<T>, N> filled_array(T&& val)
{
  return ::boba::detail::filled_array<N>(std::forward<T>(val), std::make_index_sequence<N>{});
}

/**
 * @brief Builds an array filled with one value.
 * @param val Fill value.
 * @return An array containing `N` copies of `val`.
 */
template <std::size_t N, typename T>
__boba_host_device__ constexpr Array<std::decay_t<T>, N> filled_array(const T& val)
{
  return ::boba::detail::filled_array<N>(val, std::make_index_sequence<N>{});
}

/**
 * @brief Returns an array of size `N - 1` containing all values except the deleted element.
 * @param array Source array.
 * @param element_to_delete Index of the element to remove.
 * @return A copy of `array` without the selected element.
 */
template <std::size_t N, typename T>
__boba_host_device__ constexpr Array<std::decay_t<T>, N - 1> delete_element(const Array<T, N>& array, size_t element_to_delete)
{
  static_assert(N > 1, "Negative-sized arrays are not defined, and N = 0 case is specified.");
  Array<std::decay_t<T>, N - 1> new_array{0};
  for (size_t i = 0; i < element_to_delete; i++)
  {
    new_array[i] = array[i];
  }
  for (size_t i = element_to_delete + 1; i < N; i++)
  {
    new_array[i - 1] = array[i];
  }
  return new_array;
}

/**
 * @brief Specialization of `delete_element` for an input array of size `1`.
 * @param array Source array.
 * @param element_to_delete Index of the element to remove.
 * @return An empty array.
 */
template <typename T>
__boba_host_device__ constexpr Array<std::decay_t<T>, 0> delete_element(const Array<T, 1>& array, size_t element_to_delete)
{
  detail::ignore(array);
  detail::ignore(element_to_delete);
  Array<std::decay_t<T>, 0> new_array;
  return new_array;
}

/**
 * @brief Returns an array of size `N + 1` containing `value_to_insert` before `element_to_insert`.
 * @param array Source array.
 * @param element_to_insert Index before which to insert.
 * @param value_to_insert Value to insert.
 * @return A copy of `array` with the new element inserted.
 */
template <std::size_t N, typename T>
__boba_host_device__ constexpr Array<std::decay_t<T>, N + 1> insert_element(Array<T, N>& array, size_t element_to_insert, T value_to_insert)
{
  static_assert(N > 0, "N = 0 case requires specialization.");
  Array<std::decay_t<T>, N + 1> new_array{0};
  for (size_t i = 0; i < element_to_insert; i++)
  {
    new_array[i] = array[i];
  }
  new_array[element_to_insert] = value_to_insert;
  for (size_t i = element_to_insert; i < N; i++)
  {
    new_array[i + 1] = array[i];
  }
  return new_array;
}

/**
 * @brief Specialization of `insert_element` for an input array of size `0`.
 * @param array Source array.
 * @param element_to_insert Insertion index.
 * @param value_to_insert Value to insert.
 * @return A size-one array containing the inserted value.
 */
template <typename T>
__boba_host_device__ constexpr Array<std::decay_t<T>, 1> insert_element(Array<T, 0>& array, size_t element_to_insert, T value_to_insert)
{
  ::boba::detail::ignore(array);
  Array<std::decay_t<T>, 1> new_array{value_to_insert};
  boba_assert_equal(element_to_insert, 0_z, "Invalid element_to_insert");
  return new_array;
}

/**
 * @brief Returns an array consistent with `{array_a, array_b}`.
 * @param array_a Left array.
 * @param array_b Right array.
 * @return The concatenated array.
 */
template <std::size_t N_a, std::size_t N_b, typename T>
__boba_host_device__ constexpr Array<std::decay_t<T>, N_a + N_b> concatenate(Array<T, N_a> array_a, Array<T, N_b> array_b)
{
  static_assert(N_a + N_b > 0, "Concatenating two empty arrays is not supported.");
  Array<std::decay_t<T>, N_a + N_b> new_array{0};
  for (size_t d = 0; d < N_a; d++)
  {
    new_array[d] = array_a[d];
  }
  for (size_t d = 0; d < N_b; d++)
  {
    new_array[d + N_a] = array_b[d];
  }
  return new_array;
}

//
// Array folding functions
//
/**
 * @brief Sums all entries in an array.
 * @param a Input array.
 * @param init Initial accumulator value.
 * @return The accumulated sum.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr T sum(Array<T, N> a, T init = 0)
{
  return ::boba::detail::sum(a, init, std::make_index_sequence<N>{});
}

/**
 * @brief Multiplies all entries in an array.
 * @param a Input array.
 * @param init Initial accumulator value.
 * @return The accumulated product.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr T product(Array<T, N> a, T init = (N > 0) ? 1 : 0)
{
  return ::boba::detail::product(a, init, std::make_index_sequence<N>{});
}

/**
 * @brief Cast Array from one type to another.
 * @param a Source array.
 * @return The cast array.
 */
template <typename T, typename S, std::size_t N>
__boba_host_device__ constexpr Array<T, N> cast(Array<S, N> a)
{
  Array<T, N> b{0};
  for (size_t i = 0_z; i < N; i++)
  {
    b[i] = static_cast<T>(a[i]);
  }
  return b;
}

/**
 * @brief boba Array sin.
 * @param a Input array.
 * @return The elementwise sine.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> sin(Array<T, N> a)
{
  Array<T, N> b{0};
  for (size_t i = 0_z; i < N; i++)
  {
    b[i] = ::boba::sin(a[i]);
  }
  return b;
}

/**
 * @brief boba Array cos.
 * @param a Input array.
 * @return The elementwise cosine.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> cos(Array<T, N> a)
{
  Array<T, N> b{0};
  for (size_t i = 0_z; i < N; i++)
  {
    b[i] = ::boba::cos(a[i]);
  }
  return b;
}

/**
 * @brief boba Array pow.
 * @param a Input array.
 * @param p Exponent applied to each entry.
 * @return The elementwise power.
 */
template <class T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> pow(Array<T, N> a, T p)
{
  Array<T, N> b{0};
  for (size_t d = 0_z; d < N; d++)
  {
    b[d] = pow(a[d], p);
  }
  return b;
}

/**
 * @brief boba Array mod.
 * @param a Input array.
 * @param p Modulus.
 * @return The elementwise remainder.
 */
template <class T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> mod(Array<T, N> a, T p)
{
  Array<T, N> b{0};
  for (size_t d = 0_z; d < N; d++)
  {
    b[d] = mod(a[d], p);
  }
  return b;
}

/**
 * @brief boba Array floor.
 * @param a Input array.
 * @return The elementwise floor.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> floor(Array<T, N> a)
{
  Array<T, N> b{0};
  for (size_t i = 0_z; i < N; i++)
  {
    b[i] = ::boba::floor(a[i]);
  }
  return b;
}

/**
 * @brief boba Array ceil.
 * @param a Input array.
 * @return The elementwise ceiling.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> ceil(Array<T, N> a)
{
  Array<T, N> b{0};
  for (size_t i = 0_z; i < N; i++)
  {
    b[i] = ::boba::ceil(a[i]);
  }
  return b;
}

/**
 * @brief boba Array log.
 * @param a Input array.
 * @return The elementwise logarithm.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> log(Array<T, N> a)
{
  Array<T, N> b{0};
  for (size_t i = 0_z; i < N; i++)
  {
    b[i] = ::boba::log(a[i]);
  }
  return b;
}

/**
 * @brief boba Array exp.
 * @param a Input array.
 * @return The elementwise exponential.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> exp(Array<T, N> a)
{
  Array<T, N> b{0};
  for (size_t i = 0_z; i < N; i++)
  {
    b[i] = ::boba::exp(a[i]);
  }
  return b;
}

/**
 * @brief Check if Array is a valid permutation.
 * @param permutation Candidate permutation.
 * @return `true` when the entries are distinct and in range.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr bool is_valid_permutation(Array<T, N> permutation)
{
  bool is_valid = (permutation >= 0) and (permutation < N);
  for (size_t i = 0_z; i < N; i++)
  {
    for (size_t j = (i + 1); j < N; j++)
    {
      if ((i != j) and (permutation[i] == permutation[j]))
      {
        is_valid = false;
        j = N; // terminate both loops
        i = N;
      }
    }
  }
  return is_valid;
}

/**
 * @brief Array permute.
 * @param a Input array.
 * @param permutation Permutation of element indices.
 * @return The permuted array.
 */
template <typename T, typename S, std::size_t N>
__boba_host_device__ constexpr Array<T, N> permute(Array<T, N> a, Array<S, N> permutation)
{
  boba_assert(is_valid_permutation(permutation), "Permutation violates one or more conditions.");
  Array<T, N> b;
  for (size_t i = 0_z; i < N; i++)
  {
    b[i] = a[permutation[i]];
  }
  return b;
}

/**
 * @brief Permutes a given Array using einsum labels.
 * @param labels Array of labels naming the input dimensions.
 * @param input Array to permute.
 * @param permuted_labels Desired label order.
 * @return The permuted array.
 */
template <typename T, std::size_t N, typename DimLabel = std::string>
constexpr Array<T, N> permute(Array<DimLabel, N> labels, Array<T, N> input, Array<DimLabel, N> permuted_labels)
{
  Array<T, N> index_permutation{0};
  for (size_t i = 0_z; i < N; i++)
  {
    for (size_t j = 0_z; j < N; j++)
    {
      if (permuted_labels[i] == labels[j])
      {
        index_permutation[i] = j;
      }
    }
  }
  return permute(input, index_permutation);
}

/**
 * @brief boba Array range function that creates the sequence `0, ..., N - 1`.
 * @return An array containing the index range.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> range()
{
  Array<T, N> b{0};
  for (size_t i = 0_z; i < N; i++)
  {
    b[i] = static_cast<T>(i);
  }
  return b;
}

/**
 * @brief Elementwise Array min.
 * @param a Left array.
 * @param b Right array.
 * @return The elementwise minimum.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> min(const Array<T, N>& a, const Array<T, N>& b)
{
  Array<T, N> c{0};
  for (size_t i = 0_z; i < N; i++)
  {
    c[i] = min(a[i], b[i]);
  }
  return c;
}

/**
 * @brief Elementwise Array max.
 * @param a Left array.
 * @param b Right array.
 * @return The elementwise maximum.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr Array<T, N> max(const Array<T, N>& a, const Array<T, N>& b)
{
  Array<T, N> c{0};
  for (size_t i = 0_z; i < N; i++)
  {
    c[i] = max(a[i], b[i]);
  }
  return c;
}

/**
 * @brief Returns the lowest value of the array according to `min`.
 * @param a Input array.
 * @return The minimum entry.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr T min(const Array<T, N>& a)
{
  auto value = a[0];
  for (size_t i = 1_z; i < N; i++)
  {
    value = min(value, a[i]);
  }
  return value;
}

static_assert(min(Array<size_t, 3>{5, 2, 3}) == 2, "Second value, 2 is the lowest");

/**
 * @brief Returns the first index with the lowest value.
 * @param a Input array.
 * @return Index of the first minimum entry.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr size_t argmin(const Array<T, N>& a)
{
  size_t arg_id = 0;
  auto arg_value = a[arg_id];
  for (size_t i = 1_z; i < N; i++)
  {
    auto this_value = a[i];
    if (this_value < arg_value)
    {
      arg_id = i;
      arg_value = this_value;
    }
  }
  return arg_id;
}

static_assert(argmin(Array<size_t, 3>{5, 2, 3}) == 1, "Second value, 2 is the lowest");

/**
 * @brief Returns the largest value of the array according to `max`.
 * @param a Input array.
 * @return The maximum entry.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr T max(const Array<T, N>& a)
{
  auto value = a[0];
  for (size_t i = 1_z; i < N; i++)
  {
    value = max(value, a[i]);
  }
  return value;
}

static_assert(max(Array<size_t, 3>{5, 2, 3}) == 5, "First value value, 5 is the maximum");

/**
 * @brief Returns the first index with the greatest value.
 * @param a Input array.
 * @return Index of the first maximum entry.
 */
template <typename T, std::size_t N>
__boba_host_device__ constexpr size_t argmax(const Array<T, N>& a)
{
  size_t arg_id = 0;
  T arg_value = a[arg_id];
  for (size_t i = 1_z; i < N; i++)
  {
    T this_value = a[i];
    if (this_value > arg_value)
    {
      arg_id = i;
      arg_value = this_value;
    }
  }
  return arg_id;
}

static_assert(argmax(Array<size_t, 3>{5, 2, 3}) == 0, "First value, 5 is the greatest");

/**
 * @brief Make a `std::vector` from a `boba::Array`.
 * @param boba_array Source array.
 * @return A vector containing the same values.
 */
template <typename T, size_t N>
constexpr std::vector<T> make_std_vector(const Array<T, N>& boba_array)
{
  std::vector<T> output;
  for (size_t i = 0; i < N; i++)
  {
    output.push_back(boba_array[i]);
  }
  return output;
}

/**
 * @brief Make a `std::array` from a `boba::Array`.
 * @param boba_array Source array.
 * @return A standard array containing the same values.
 */
template <typename T, typename S, size_t N>
constexpr std::array<T, N> make_std_array(const Array<S, N>& boba_array)
{
  std::array<T, N> output;
  for (size_t i = 0; i < N; i++)
  {
    output[i] = static_cast<T>(boba_array[i]);
  }
  return output;
}

/**
 * @brief Checks nonnegativity for `Array<size_t, N>`.
 * @param expression_a Array to inspect.
 * @return Always `true`.
 */
template <size_t N>
__boba_host_device__ inline bool check_nonnegative_(
  const ::boba::Array<size_t, N>& expression_a)
{
  ::boba::detail::ignore(expression_a);
  return true;
}

/**
 * @brief Checks whether every array entry is nonnegative.
 * @param expression_a Array to inspect.
 * @return `true` when all entries are nonnegative.
 */
template <typename type, size_t N>
__boba_host_device__ inline bool check_nonnegative_(
  ::boba::Array<type, N> expression_a)
{
  return (expression_a >= static_cast<type>(0));
}

/**
 * @brief Checks whether an array is non-increasing.
 * @param expression_a Array to inspect.
 * @return `true` when each entry is at most the previous entry.
 */
template <typename type, size_t N>
__boba_host_device__ inline bool is_nonincreasing(
  ::boba::Array<type, N> expression_a)
{
  bool nonincreasing = true;
  for (std::size_t ind = 1; ind < N; ind++)
  {
    nonincreasing = nonincreasing && (expression_a[ind] <= expression_a[ind - 1]);
  }
  return nonincreasing;
}

// TODO<organization> move this elsewhere since it is not inherently an array thing
/**
 * @brief Joins vector entries into a delimited string.
 * @param vec_a Values to join.
 * @param delimiter String inserted between values.
 * @return The formatted string.
 */
template <typename type>
std::string make_delimited_string(
  const std::vector<type>& vec_a,
  const std::string delimiter = ", ")
{

  std::ostringstream out;
  if (!vec_a.empty())
  {
    std::copy(std::begin(vec_a), std::end(vec_a) - 1, std::ostream_iterator<type>(out, delimiter.c_str()));
    out << vec_a.back();
  }

  return out.str();
}

/**
 * @brief Joins array entries into a delimited string.
 * @param expression_a Values to join.
 * @param delimiter String inserted between values.
 * @return The formatted string.
 */
template <typename type, size_t N>
std::string make_delimited_string(
  const ::boba::Array<type, N>& expression_a,
  const std::string delimiter = ", ")
{
  return make_delimited_string(make_std_vector(expression_a), delimiter);
}
} // end namespace boba

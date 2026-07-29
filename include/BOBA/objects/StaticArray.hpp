// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <cstddef>
#include <iostream>
#include <utility>

namespace boba
{

/**
 * \brief
 * BoBa-specific implementation of std::array like class with compile-time values.
 */
template <typename T, T... values>
struct StaticArray
{
  //
  // type aliases
  //
  using value_type = T;
  using size_type = std::size_t;

  using array_data = std::integer_sequence<T, values...>;

  /**
   * @brief Replaces one compile-time entry with a new value.
   *
   * For example, given `[1, 2, 3]`, replacing index `2` with `5` yields `[1, 2, 5]`.
   *
   * @tparam value Replacement value.
   * @tparam index Zero-based position to replace.
   * @return A `StaticArray` with the updated entry.
   */
  template <T value, std::size_t index>
  static constexpr auto replace_value_at_index() noexcept
  {
    return replace_value_at_index<value, index>(
      std::make_integer_sequence<std::size_t, sizeof...(values)>{});
  }

  /**
   * @brief Replaces one compile-time entry using an explicit index sequence.
   * @tparam value Replacement value.
   * @tparam index Zero-based position to replace.
   * @param indices Sequence used to expand the array entries.
   * @return A `StaticArray` with the updated entry.
   */
  template <T value, std::size_t index, std::size_t... indices>
  static constexpr auto replace_value_at_index(std::integer_sequence<std::size_t, indices...>) noexcept
  {
    static_assert(index < sizeof...(values), "Out-of-bounds.");
    return StaticArray<T, ((indices == index) ? value : values)...>{};
  }

  template <T value, std::size_t index>
  using replace_value_at_index_t = decltype(replace_value_at_index<value, index>());

  //
  // use implicit constructor, destructor, and assignment operators
  //

  //
  // capacity
  //
  /**
   * @brief Returns whether the array is empty.
   * @return `true` when the array has no values.
   */
  __boba_host_device__ static constexpr bool empty() noexcept
  {
    return sizeof...(values) == 0_z;
  }

  /**
   * @brief Returns the number of compile-time values.
   * @return The array size.
   */
  __boba_host_device__ static constexpr size_type size() noexcept
  {
    return sizeof...(values);
  }

  //
  // data access
  //
  /**
   * @brief Returns the value at a given index.
   * @param i Zero-based element index.
   * @return The value stored at `i`.
   */
  __boba_host_device__ static constexpr value_type at(size_type i) noexcept
  {
    T tmp[] = {values...};
    return tmp[i];
  }

  /**
   * @brief Returns the value at a given index.
   * @param i Zero-based element index.
   * @return The value stored at `i`.
   */
  __boba_host_device__ constexpr value_type operator[](size_type i) const noexcept
  {
    T tmp[] = {values...};
    return tmp[i];
  }

  template <std::size_t N = sizeof...(values)>
    requires(N > 0_z)
  /**
   * @brief Returns the first value.
   * @return The first compile-time value.
   */
  __boba_host_device__ static constexpr value_type front() noexcept
  {
    T tmp[] = {values...};
    return tmp[0];
  }

  template <std::size_t N = sizeof...(values)>
    requires(N > 0_z)
  /**
   * @brief Returns the last value.
   * @return The last compile-time value.
   */
  __boba_host_device__ static constexpr value_type back() noexcept
  {
    T tmp[] = {values...};
    return tmp[sizeof...(values) - 1];
  }

  /**
   * @brief Converts this static array to a runtime `Array`.
   * @return A runtime array containing the same values.
   */
  __boba_host_device__ constexpr operator Array<T, sizeof...(values)>() const noexcept
  {
    return Array<T, sizeof...(values)>{{values...}};
  }
};

/**
 * @brief Returns the `I`th value for structured binding support.
 * @tparam I Zero-based element index.
 * @param array Array to access.
 * @return The value at index `I`.
 */
template <std::size_t I, typename T, T... values>
__boba_host_device__ constexpr T get(StaticArray<T, values...> const& array) noexcept
{
  return array[I];
}

} // end namespace boba

namespace std
{

//
// specializations of std classes for structured binding
//
template <typename T, T... values>
struct tuple_size<::boba::StaticArray<T, values...>>
    : std::integral_constant<std::size_t, sizeof...(values)>
{
};

template <std::size_t I, typename T, T... values>
struct tuple_element<I, ::boba::StaticArray<T, values...>>
{
  using type = T;
};

} // namespace std

namespace boba
{

/**
 * @brief Converts an integer sequence to a `StaticArray`.
 * @param sequence Integer sequence to convert.
 * @return A `StaticArray` with the same compile-time values.
 */
template <typename T, T... values>
__boba_host_device__ constexpr StaticArray<std::remove_cv_t<T>, values...> to_static_array(std::integer_sequence<T, values...>)
{
  return {};
}

/**
 * @brief Converts a `StaticArray` to a runtime `Array`.
 * @param array Static array to convert.
 * @return A runtime array with the same values.
 */
template <typename T, T... values>
__boba_host_device__ constexpr Array<std::remove_cv_t<T>, sizeof...(values)> to_array(StaticArray<T, values...>)
{
  return {values...};
}

/**
 * @brief Converts a `StaticArray` to a runtime `Array` with a new value type.
 * @tparam T Output element type.
 * @param array Static array to convert.
 * @return A runtime array with each value cast to `T`.
 */
template <typename T, typename U, U... values>
__boba_host_device__ constexpr Array<T, sizeof...(values)> typed_array(StaticArray<U, values...>)
{
  return {T(values)...};
}

} // namespace boba

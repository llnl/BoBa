// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace boba
{

/**
 * \brief Subtensor view that can be used in device code.
\verbatim
 auto TensorBase_view = TensorBase.view();
 device_code{
    x = TensorBase_view(i, j, k, ...)
    TensorBase_view(i, j, k, ...) = 2*x
 }
\endverbatim
 */

template <typename _accessor, std::size_t _dimension>
struct SubtensorView : TensorView<_accessor, _dimension>
{
  static constexpr std::size_t dimension = _dimension;

  using accessor = _accessor;
  using data_t = typename _accessor::data_t;
  using data_const_reference = data_t const&;
  using data_reference = typename _accessor::data_reference;
  using data_const_pointer = data_t const*;
  using data_pointer = typename _accessor::data_pointer;
  using index_array = ::boba::Array<index_t, dimension>;

  using base = TensorView<accessor, dimension>;

  using base::assert_indices;
  using base::get_dimension;
  using base::index;
  using base::multiindex;
  using base::size;
  using base::sizes;
  using base::strides;

  /**
   * \brief default constructor
   */
  SubtensorView() = default;

  /**
   * \brief copy constructor
   */
  SubtensorView(SubtensorView const&) = default;

  /**
   * \brief move constructor
   */
  SubtensorView(SubtensorView&&) = default;

  /**
   * \brief copy assignment operator
   */
  SubtensorView& operator=(SubtensorView const&) = default;

  /**
   * \brief move assignment operator
   */
  SubtensorView& operator=(SubtensorView&&) = default;

  /**
   * \brief destructor
   */
  ~SubtensorView() = default;

  /**
   * \brief Constructs a subtensor view with explicit bounds metadata.
   * \param data Pointer to the first element in the full tensor storage.
   * \param offset Linear offset of the subtensor origin within the full tensor.
   * \param lower_bound Inclusive lower bounds of the subtensor.
   * \param full_sizes Extent of each dimension in the full tensor.
   * \param sizes Extent of each dimension in the subtensor.
   * \param strides Stride of each dimension in the full tensor storage.
   */
  constexpr SubtensorView(data_pointer data,
                          index_t view_offset,
                          index_array view_lower_bound,
                          index_array tensor_full_sizes,
                          index_array view_sizes,
                          index_array view_strides) noexcept
      : base(data, view_sizes, view_strides),
        m_offset(view_offset),
        m_lower_bound(view_lower_bound),
        m_full_sizes(tensor_full_sizes)
  {
  }

  /**
   * \brief Returns a reference to the element at a subtensor multi-index.
   * \param indices Tensor indices relative to the subtensor bounds.
   * \return A reference to the selected element.
   */
  __boba_host_device__
  data_reference
  operator()(index_array indices) const
  {
    return accessor{}.access(this->data(), m_offset + index(this->m_strides, assert_indices(indices)));
  }

  /**
   * \brief Returns a reference to the element at a subtensor linear offset.
   * \param index Linear element offset within the subtensor.
   * \return A reference to the selected element.
   */
  __boba_host_device__
  data_reference
  operator()(index_t linear_index) const
  {
    boba_assert_nonnegative(linear_index, "Negative index");
    boba_assert_lt(linear_index, product(this->m_sizes), "Out of bounds");
    return operator()(this->multiindex(linear_index));
  }

  /**
   * \brief Returns the underlying data pointer.
   * \return Pointer to the full tensor storage referenced by the view.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_pointer data() const noexcept
  {
    return base::data();
  }

  /**
   * \brief Returns the underlying const data pointer.
   * \return Pointer to the full tensor storage referenced by the view.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_const_pointer const_data() const noexcept
  {
    return base::const_data();
  }

  /**
   * \brief Returns the linear offset of the subtensor origin.
   * \return Linear offset into the full tensor storage.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_t offset() const noexcept
  {
    return m_offset;
  }

  /**
   * \brief Returns the inclusive lower bounds of the subtensor.
   * \return Lower bounds for each subtensor dimension.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_array lower_bound() const noexcept
  {
    return m_lower_bound;
  }

  /**
   * \brief Returns the full tensor extents.
   * \return Extent of each dimension in the full tensor.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr index_array full_sizes() const noexcept
  {
    return m_full_sizes;
  }

private:
  index_t m_offset = 0;
  index_array m_lower_bound = ::boba::filled_array<dimension>(static_cast<index_t>(0));
  index_array m_full_sizes = ::boba::filled_array<dimension>(static_cast<index_t>(0));
};

/**
 * \brief Creates a mutable view of a bounded subtensor.
 * \param full_tensor Tensor providing the underlying storage.
 * \param lower_bound Inclusive lower bounds of the subtensor.
 * \param upper_bound Exclusive upper bounds of the subtensor.
 * \return Subtensor view corresponding to the requested bounds.
 */
template <size_t dimension, execution_space space, typename data_t>
[[nodiscard]]
SubtensorView<DefaultAccessor<data_t>, dimension>
make_subtensor_view(
  Tensor<dimension, space, data_t>& full_tensor,
  const Array<index_t, dimension>& lower_bound,
  const Array<index_t, dimension>& upper_bound)
{
  boba_assert_nonnegative(lower_bound, "Negative index");
  boba_assert_le(upper_bound, full_tensor.m_sizes, "Out of bounds");
  boba_assert_lt(lower_bound, upper_bound, "Ill-defined bounds");

  auto* data = full_tensor.data();
  auto offset = full_tensor.index(lower_bound);
  auto full_sizes = full_tensor.sizes();
  auto sizes = upper_bound - lower_bound;
  auto strides = full_tensor.strides();

  using accessor = DefaultAccessor<data_t>;
  SubtensorView<accessor, dimension> subtensor_view(data, offset, lower_bound, full_sizes, sizes, strides);
  return subtensor_view;
}

/**
 * \brief Creates a const view of a bounded subtensor.
 * \param full_tensor Tensor providing the underlying storage.
 * \param lower_bound Inclusive lower bounds of the subtensor.
 * \param upper_bound Exclusive upper bounds of the subtensor.
 * \return Const subtensor view corresponding to the requested bounds.
 */
template <size_t dimension, execution_space space, typename data_t>
[[nodiscard]]
SubtensorView<DefaultAccessor<data_t const>, dimension>
make_subtensor_const_view(
  const Tensor<dimension, space, data_t>& full_tensor,
  const Array<index_t, dimension>& lower_bound,
  const Array<index_t, dimension>& upper_bound)
{
  boba_assert_nonnegative(lower_bound, "Negative index");
  boba_assert_le(upper_bound, full_tensor.m_sizes, "Out of bounds");
  boba_assert_lt(lower_bound, upper_bound, "Ill-defined bounds");

  auto* data = full_tensor.data();
  auto offset = full_tensor.index(lower_bound);
  auto full_sizes = full_tensor.sizes();
  auto sizes = upper_bound - lower_bound;
  auto strides = full_tensor.strides();

  using accessor = DefaultAccessor<data_t const>;
  SubtensorView<accessor, dimension> subtensor_view(data, offset, lower_bound, full_sizes, sizes, strides);
  return subtensor_view;
}

} // namespace boba

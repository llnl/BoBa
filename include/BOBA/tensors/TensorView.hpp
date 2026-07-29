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
 * \brief Default accessor used by TensorView.
 *
\verbatim
 auto view = TensorView<DefaultAccessor<double>, 1>();
\endverbatim
 */
template <typename _data_t>
struct DefaultAccessor
{
  using data_t = _data_t;
  using data_reference = data_t&;
  using data_pointer = data_t*;

  /**
   * \brief default constructor
   */
  constexpr DefaultAccessor() noexcept = default;

  /**
   * \brief Returns a reference to the element at a linear offset.
   * \param p Pointer to the first element in the view.
   * \param i Linear element offset.
   * \return A reference to the selected element.
   */
  __boba_host_device__ constexpr data_reference access(data_pointer p, index_t i) const noexcept
  {
    return p[i];
  }

  /**
   * \brief Returns a pointer offset from the base pointer.
   * \param p Pointer to the first element in the view.
   * \param i Offset in elements.
   * \return Pointer advanced by \p i elements.
   */
  __boba_host_device__ constexpr data_pointer offset(data_pointer p, index_t i) const noexcept
  {
    return p + i;
  }
};

/**
 * \brief Tensor view that can be used in device code.
\verbatim
 auto TensorBase_view = TensorBase.view();
 device_code{
    x = TensorBase_view(i, j, k, ...)
    TensorBase_view(i, j, k, ...) = 2*x
 }
\endverbatim
 */

template <typename _accessor, std::size_t _dimension>
struct TensorView : Multiindexer<_dimension>
{
  static constexpr std::size_t dimension = _dimension;

  using accessor = _accessor;
  using data_t = typename _accessor::data_t;
  using data_const_reference = data_t const&;
  using data_reference = typename _accessor::data_reference;
  using data_const_pointer = data_t const*;
  using data_pointer = typename _accessor::data_pointer;
  using index_array = ::boba::Array<index_t, dimension>;

  using base = Multiindexer<dimension>;
  using base::assert_indices;
  using base::get_dimension;
  using base::index;
  using base::multiindex;
  using base::precompute_strides;
  using base::size;
  using base::sizes;
  using base::strides;

  /**
   * \brief default constructor
   */
  TensorView() = default;

  /**
   * \brief copy constructor
   */
  TensorView(TensorView const&) = default;

  /**
   * \brief move constructor
   */
  TensorView(TensorView&&) = default;

  /**
   * \brief copy assignment operator
   */
  TensorView& operator=(TensorView const&) = default;

  /**
   * \brief move assignment operator
   */
  TensorView& operator=(TensorView&&) = default;

  /**
   * \brief destructor
   */
  ~TensorView() = default;

  /**
   * \brief Constructs a tensor view from data, extents, and explicit strides.
   * \param data Pointer to the first element in the view.
   * \param sizes Extent of each dimension.
   * \param strides Stride of each dimension.
   */
  constexpr TensorView(data_pointer data,
                       index_array view_sizes,
                       index_array view_strides) noexcept
      : base(view_sizes, view_strides),
        m_data(data)
  {
  }

  /**
   * \brief Constructs a tensor view from data and extents.
   * \param data Pointer to the first element in the view.
   * \param sizes Extent of each dimension.
   */
  constexpr TensorView(data_pointer data,
                       index_array view_sizes) noexcept
      : base(view_sizes),
        m_data(data)
  {
  }

  /**
   * \brief Returns a reference to the element at a tensor multi-index.
   * \param indices Tensor indices identifying the element.
   * \return A reference to the selected element.
   */
  __boba_host_device__
  data_reference
  operator()(index_array indices) const
  {
    return accessor{}.access(m_data, index(this->m_strides, assert_indices(indices)));
  }

  /**
   * \brief Returns a reference to the element at a linear offset.
   * \param index Linear element offset.
   * \return A reference to the selected element.
   */
  __boba_host_device__
  data_reference
  operator()(index_t linear_index) const
  {
    boba_assert_nonnegative(linear_index, "Negative index");
    boba_assert_lt(linear_index, product(this->m_sizes), "Out of bounds");
    return accessor{}.access(m_data, linear_index);
  }

  /**
   * \brief Returns the underlying data pointer.
   * \return Pointer to the first element in the view.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_pointer data() const noexcept
  {
    return m_data;
  }

  /**
   * \brief Returns the underlying const data pointer.
   * \return Pointer to the first element in the view.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_const_pointer const_data() const noexcept
  {
    return m_data;
  }

  /**
   * \brief Reshapes the view while preserving the total element count.
   * \tparam rhs_dimension New tensor dimension.
   * \param new_sizes Extent of each dimension after reshaping.
   */
  template <size_t rhs_dimension>
  __boba_host_device__ void reshape(const Array<index_t, rhs_dimension>& new_sizes)
  {
    boba_always_assert_equal(size(), product(new_sizes), "reshape sizes must match");
    this->m_sizes = new_sizes;
    this->m_strides = precompute_strides(new_sizes);
  }

private:
  data_pointer m_data = nullptr;
};

} // namespace boba

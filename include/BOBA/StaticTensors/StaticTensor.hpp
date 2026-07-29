// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief Statically sized tensor with inline storage.
 *
 * Example:
 * using my_sizes = boba::StaticArray<std::size_t, 3,4,5>;
 * StaticTensor<my_sizes> my_tensor_of_doubles;
 * StaticTensor<my_sizes> my_other_tensor_of_doubles;
 * StaticTensor<boba::StaticArray<std::size_t, 7,3,1>> my_third_tensor_of_doubles;
 */
template <typename _sizes, typename _data_t>
struct StaticTensor
{
  using t_sizes = _sizes;

  template <std::size_t dim, std::size_t new_size>
  using resize_dimension_type = StaticTensor<
    typename t_sizes::template replace_value_at_index_t<new_size, dim>,
    _data_t>;

  using view_type = StaticTensorView<_data_t, t_sizes>;

  static constexpr std::size_t dimension = view_type::dimension;

  using index_array = ::boba::Array<index_t, dimension>;

  // ---------------------------------------------------------------------------
  // Static member variables
  // ---------------------------------------------------------------------------

  static_assert(dimension > 0, "StaticTensor must have positive dimension");

  // ---------------------------------------------------------------------------
  // Member types
  // ---------------------------------------------------------------------------

  using data_t = _data_t;

  using const_view_type = StaticTensorView<data_t const, t_sizes>;

  using data_array = ::boba::Array<data_t, view_type::size()>;
  using data_reference = data_t&;
  using data_const_reference = data_t const&;
  using data_pointer = data_t*;
  using data_const_pointer = data_t const*;

  // ---------------------------------------------------------------------------
  // Constructors/destructors
  // ---------------------------------------------------------------------------

  /**
   * \brief default constructor
   */
  constexpr StaticTensor() = default;

  /**
   * \brief Constructs a tensor from flat storage.
   * \param data Flat array containing all tensor coefficients.
   */
  __boba_host_device__ constexpr StaticTensor(data_array data)
      : m_data(data)
  {
  }

  /**
   * \brief copy constructor
   */
  StaticTensor(StaticTensor const& rhs) = default;

  /**
   * \brief move constructor
   */
  StaticTensor(StaticTensor&& rhs) = default;

  /**
   * \brief copy assignment operator
   */
  StaticTensor& operator=(StaticTensor const& rhs) = default;

  /**
   * \brief move assignment operator
   */
  StaticTensor& operator=(StaticTensor&& rhs) = default;

  // ---------------------------------------------------------------------------
  // Section: Getters
  // ---------------------------------------------------------------------------

  /**
   * \brief Returns whether the tensor has no elements.
   * \return False for any valid StaticTensor instantiation.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr bool empty() const noexcept
  {
    return size() == 0;
  }

  /**
   * \brief Returns the total number of tensor elements.
   * \return The product of the compile-time extents.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_t size() noexcept
  {
    return view_type::size();
  }

  /**
   * \brief Returns the extent of one dimension.
   * \param d Dimension index.
   * \return The extent of dimension \p d.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_t sizes(index_t d) noexcept
  {
    return view_type::sizes()[d];
  }

  /**
   * \brief Returns the extent of each dimension.
   * \return The compile-time tensor extents.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_array sizes() noexcept
  {
    return view_type::sizes();
  }

  /**
   * \brief Returns the stride of one dimension.
   * \param d Dimension index.
   * \return The row-major stride of dimension \p d.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_t strides(index_t d) noexcept
  {
    return view_type::strides()[d];
  }

  /**
   * \brief Returns the row-major stride of each dimension.
   * \return The compile-time tensor strides.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_array strides() noexcept
  {
    return view_type::strides();
  }

  // ---------------------------------------------------------------------------
  // Section: Read/write
  // ---------------------------------------------------------------------------

  /**
   * \brief Returns a mutable pointer to the tensor storage.
   * \return Pointer to the first tensor element.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_pointer data() noexcept
  {
    return m_data.data();
  }

  /**
   * \brief Returns a const pointer to the tensor storage.
   * \return Pointer to the first tensor element.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_const_pointer data() const noexcept
  {
    return m_data.data();
  }

  /**
   * \brief Returns a const pointer to the tensor storage.
   * \return Pointer to the first tensor element.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_const_pointer const_data() const noexcept
  {
    return m_data.const_data();
  }

  /**
   * \brief Returns a mutable tensor view over the storage.
   * \return A mutable static tensor view.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr view_type view() noexcept
  {
    return {m_data.data()};
  }

  /**
   * \brief Returns a const tensor view over the storage.
   * \return A const static tensor view.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr const_view_type view() const noexcept
  {
    return {m_data.data()};
  }

  /**
   * \brief Returns a const tensor view over the storage.
   * \return A const static tensor view.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr const_view_type const_view() const noexcept
  {
    return {m_data.const_data()};
  }

  /**
   * \brief Returns a mutable element reference for a tensor multi-index.
   * \param multiindex Tensor indices identifying the element.
   * \return A reference to the selected element.
   */
  __boba_host_device__
  data_reference
  operator()(index_array multiindex)
  {
    return view()(multiindex);
  }

  /**
   * \brief Returns a const element reference for a tensor multi-index.
   * \param multiindex Tensor indices identifying the element.
   * \return A reference to the selected element.
   */
  __boba_host_device__
  data_const_reference
  operator()(index_array multiindex) const
  {
    return const_view()(multiindex);
  }

  /**
   * \brief Returns a mutable element reference for a linear index.
   * \param index Linear element offset.
   * \return A reference to the selected element.
   */
  __boba_host_device__
  data_reference
  operator()(index_t index)
  {
    return view()(index);
  }

  /**
   * \brief Returns a const element reference for a linear index.
   * \param index Linear element offset.
   * \return A reference to the selected element.
   */
  __boba_host_device__
  data_const_reference
  operator()(index_t index) const
  {
    return const_view()(index);
  }

  // ---------------------------------------------------------------------------
  // Section: Operators
  // ---------------------------------------------------------------------------

  /**
   * \brief Multiplies every tensor entry by a scalar.
   * \param rhs Scalar multiplier.
   * \return This tensor after scaling.
   */
  __boba_host_device__ constexpr StaticTensor& operator*=(data_t rhs) noexcept
  {
    for (index_t i = 0; i < size(); i++)
    {
      m_data[i] *= rhs;
    }
    return *this;
  }

  /**
   * \brief Adds another tensor elementwise.
   * \param rhs Tensor to add.
   * \return This tensor after the addition.
   */
  __boba_host_device__ constexpr StaticTensor& operator+=(StaticTensor const& rhs) noexcept
  {
    for (index_t i = 0; i < size(); i++)
    {
      m_data[i] += rhs.m_data[i];
    }
    return *this;
  }

  /**
   * \brief Subtracts another tensor elementwise.
   * \param rhs Tensor to subtract.
   * \return This tensor after the subtraction.
   */
  __boba_host_device__ constexpr StaticTensor& operator-=(StaticTensor const& rhs) noexcept
  {
    for (index_t i = 0; i < size(); i++)
    {
      m_data[i] -= rhs.m_data[i];
    }
    return *this;
  }

  // ---------------------------------------------------------------------------
  // Section: Friend Operators
  // ---------------------------------------------------------------------------

  /**
   * \brief Returns a tensor scaled by a scalar on the right.
   * \param lhs Tensor to scale.
   * \param rhs Scalar multiplier.
   * \return The scaled tensor.
   */
  [[nodiscard]]
  __boba_host_device__ friend constexpr StaticTensor operator*(StaticTensor const& lhs, data_t rhs) noexcept
  {
    StaticTensor result;
    for (index_t i = 0; i < size(); i++)
    {
      result.m_data[i] = lhs.m_data[i] * rhs;
    }
    return result;
  }

  /**
   * \brief Returns a tensor scaled by a scalar on the left.
   * \param lhs Scalar multiplier.
   * \param rhs Tensor to scale.
   * \return The scaled tensor.
   */
  [[nodiscard]]
  __boba_host_device__ friend constexpr StaticTensor operator*(data_t lhs, StaticTensor const& rhs) noexcept
  {
    StaticTensor result;
    for (index_t i = 0; i < size(); i++)
    {
      result.m_data[i] = lhs * rhs.m_data[i];
    }
    return result;
  }

  /**
   * \brief Returns the elementwise sum of two tensors.
   * \param lhs Left-hand operand.
   * \param rhs Right-hand operand.
   * \return The elementwise sum.
   */
  [[nodiscard]]
  __boba_host_device__ friend constexpr StaticTensor operator+(StaticTensor const& lhs, StaticTensor const& rhs) noexcept
  {
    StaticTensor result;
    for (index_t i = 0; i < size(); i++)
    {
      result.m_data[i] = lhs.m_data[i] + rhs.m_data[i];
    }
    return result;
  }

  /**
   * \brief Returns the elementwise difference of two tensors.
   * \param lhs Left-hand operand.
   * \param rhs Right-hand operand.
   * \return The elementwise difference.
   */
  [[nodiscard]]
  __boba_host_device__ friend constexpr StaticTensor operator-(StaticTensor const& lhs, StaticTensor const& rhs) noexcept
  {
    StaticTensor result;
    for (index_t i = 0; i < size(); i++)
    {
      result.m_data[i] = lhs.m_data[i] - rhs.m_data[i];
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // Section: Apply operator left
  // ---------------------------------------------------------------------------

  template <
    typename operation_type_d>
  /**
   * \brief Applies the final operator in a repeated left contraction chain.
   * \param operation_d Operator applied to dimension zero of this tensor.
   * \return The contracted tensor.
   */
  [[nodiscard]]
  __boba_host_device__ inline auto apply_repeated_operator_left_helper(
    operation_type_d&& operation_d)
  {
    StaticTensor recurse(*this);
    return operation_d.template contraction<0>(std::move(recurse));
  }

  template <
    typename operation_type_d,
    typename operation_type_dp1,
    typename... operation_types>
  /**
   * \brief Recursively applies one operator per remaining tensor dimension.
   * \param operation_d Operator applied after the recursive contraction step.
   * \param operation_dp1 Next operator in the contraction chain.
   * \param operations Remaining operators in the contraction chain.
   * \return The contracted tensor.
   */
  [[nodiscard]]
  __boba_host_device__ inline auto apply_repeated_operator_left_helper(
    operation_type_d&& operation_d,
    operation_type_dp1&& operation_dp1,
    operation_types&&... operations)
  {
    constexpr size_t d = 1 + sizeof...(operation_types);
    auto recurse = apply_repeated_operator_left_helper(
      std::forward<operation_type_dp1>(operation_dp1),
      std::forward<operation_types>(operations)...);
    return operation_d.template contraction<d>(std::move(recurse));
  }

  /**
   * \brief Applies one operator per tensor dimension from the left.
   * \param operations Operators applied as a Kronecker product in dimension order.
   * \return The transformed tensor.
   *
   * TODO<feature> update this to handle rectangular matrices (output size not equal to input size)
   * TODO<feature> update this to handle anisitropic operators
   */
  template <
    typename... operation_types>
  [[nodiscard]]
  __boba_host_device__ inline auto apply_repeated_operator_left(
    operation_types&&... operations)
  {
    constexpr size_t d = sizeof...(operation_types);
    static_assert(d == dimension, "Invalid number of operators for this tensor.");
    return apply_repeated_operator_left_helper(std::forward<operation_types>(operations)...);
  }

  // ---------------------------------------------------------------------------
  // Section: Read/write
  // ---------------------------------------------------------------------------

  /**
   * \brief Fills every tensor entry with the same value.
   * \param value Value assigned to every tensor entry.
   */
  __boba_host_device__ constexpr void fill_with(data_t value) noexcept
  {
    for (index_t i = 0; i < size(); i++)
    {
      m_data[i] = value;
    }
  }

  /**
   * \brief Fills the tensor with zeros.
   */
  __boba_host_device__ constexpr void fill_with_zeros() noexcept
  {
    if constexpr (std::is_arithmetic<data_t>::value)
    {
      for (size_t i = 0; i < size(); i++)
      {
        m_data[i] = 0.0;
      }
    }
    else
    {
      for (size_t i = 0; i < size(); i++)
      {
        m_data[i].fill_with_zeros();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Section: Read/write
  // ---------------------------------------------------------------------------

  /**
   * \brief Prints the tensor contents to std::cout.
   */
  void print() const
  {
    for (index_t i = 0; i < size(); i++)
    {
      std::cout << "m_data[" << i << "] " << m_data[i] << "\n";
    }
  }

  /**
   * \brief Prints the tensor contents to a stream.
   * \param os Stream receiving the formatted tensor entries.
   * \return \p os after writing the tensor contents.
   */
  std::ostream& print(std::ostream& os) const
  {
    for (index_t i = 0; i < size(); i++)
    {
      os << "m_data[" << i << "] " << m_data[i] << "\n";
    }
    return os;
  }

  /**
   * \brief Streams the tensor contents.
   * \param os Destination stream.
   * \param rhs Tensor to print.
   * \return \p os after writing the tensor contents.
   */
  inline friend std::ostream& operator<<(std::ostream& os, StaticTensor const& rhs)
  {
    // Use a helper function as this function is only a friend of this
    // TensorBase instantiation and can't make a copy in another execution space
    return rhs.print(os);
  }

  // -------------------------------------------------------------------------------------
  // Section: Shaping
  // -------------------------------------------------------------------------------------

  // TODO<feature> permute

  // TODO<feature> reshape

private:
  data_array m_data{::boba::filled_array<size()>(data_t{})};
};

/**
 * \brief Contracts a static vector with a static matrix.
 * \param tensor_a Left-hand vector operand of length \p size_a.
 * \param tensor_b Right-hand matrix operand with shape (\p size_a, \p size_b_cols).
 * \return The contracted vector of length \p size_b_cols.
 */
template <
  size_t size_a,
  size_t size_b_cols,
  typename _data_t>
[[nodiscard]]
__boba_host_device__ inline StaticTensor<StaticArray<std::size_t, size_b_cols>, _data_t> static_tensor_contraction(
  const StaticTensor<StaticArray<std::size_t, size_a>, _data_t>& tensor_a,
  const StaticTensor<StaticArray<std::size_t, size_a, size_b_cols>, _data_t>& tensor_b)
{
  auto tensor_a_view = tensor_a.view();
  auto tensor_b_view = tensor_b.view();
  constexpr size_t contraction_length = size_a;

  StaticTensor<StaticArray<std::size_t, size_b_cols>, _data_t> tensor_c;
  auto tensor_c_view = tensor_c.view();

  /*
  // Using views
  for(size_t i = 0; i < size_b_cols; i++)
  {
    _data_t sum = 0.0;
    for(size_t k = 0; k < contraction_length; k++)
    {
      sum += tensor_a_view(k)*tensor_b_view({k, i});
    }
    tensor_c_view(i) = sum;
  }
  */
  // Using offset logic
  for (size_t i = 0; i < size_b_cols; i++)
  {
    _data_t sum = 0.0;
    auto offset = i * contraction_length;
    for (size_t k = 0; k < contraction_length; k++)
    {
      sum += tensor_a_view(k) * tensor_b_view(offset + k);
    }
    tensor_c_view(i) = sum;
  }

  return tensor_c;
}

} // namespace boba

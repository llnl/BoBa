// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief Non-owning view over statically sized tensor storage.
 */
template <typename _data_t, typename _sizes>
struct StaticTensorView
{
  using t_sizes = _sizes;

  using data_t = _data_t;
  using data_reference = data_t&;
  using data_pointer = data_t*;

  static constexpr std::size_t dimension = t_sizes::size();
  using index_array = ::boba::Array<index_t, dimension>;

  /**
   * \brief Returns the extent of each dimension.
   * \return The compile-time tensor extents.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_array sizes() noexcept
  {
    return ::boba::typed_array<index_t>(::boba::to_array(t_sizes{}));
  }

  /**
   * \brief Returns the extent of one dimension.
   * \param i Dimension index.
   * \return The extent of dimension \p i.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_t sizes(index_t i) noexcept
  {
    return sizes()[i];
  }

  /**
   * \brief Returns the total number of elements in the view.
   * \return The product of all extents.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_t size() noexcept
  {
    return ::boba::product(sizes());
  }

  /**
   * \brief Returns the row-major stride of each dimension.
   * \return The compile-time tensor strides.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_array strides() noexcept
  {
    return ::boba::Multiindexer<dimension>::precompute_strides(sizes());
  }

  /**
   * \brief Returns the stride of one dimension.
   * \param i Dimension index.
   * \return The stride of dimension \p i.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_t strides(index_t i) noexcept
  {
    return strides()[i];
  }

  /**
   * \brief Converts a multi-index to a linear offset.
   * \param strides Row-major strides for each dimension.
   * \param indices Tensor indices to convert.
   * \return The linear offset corresponding to \p indices.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_t index(index_array const strides, index_array const indices) noexcept
  {
    index_t linear_index = 0;
    if constexpr (dimension > 0)
    {
      linear_index += indices[0];
      for (std::size_t d = 1; d < dimension; ++d)
      {
        linear_index += strides[d] * indices[d];
      }
    }
    return linear_index;
  }

  /**
   * \brief Converts a linear offset to a multi-index.
   * \param sizes Extent of each dimension.
   * \param index Linear offset to convert.
   * \return The tensor indices corresponding to \p index.
   */
  [[nodiscard]]
  __boba_host_device__ static constexpr index_array multiindex(index_array sizes, index_t index) noexcept
  {
    index_array indices;
    if constexpr (dimension > 0)
    {
      for (std::size_t d = 0; d < dimension - 1; d++)
      {
        indices[d] = index % sizes[d];
        index /= sizes[d];
      }
      indices[dimension - 1] = index;
    }
    return indices;
  }

  /**
   * \brief default constructor
   */
  StaticTensorView() = default;

  /**
   * \brief copy constructor
   */
  StaticTensorView(StaticTensorView const&) = default;

  /**
   * \brief move constructor
   */
  StaticTensorView(StaticTensorView&&) = default;

  /**
   * \brief copy assignment operator
   */
  StaticTensorView& operator=(StaticTensorView const&) = default;

  /**
   * \brief move assignment operator
   */
  StaticTensorView& operator=(StaticTensorView&&) = default;

  /**
   * \brief destructor
   */
  ~StaticTensorView() = default;

  /**
   * \brief Constructs a view from a raw data pointer.
   * \param data Pointer to the first tensor element.
   */
  __boba_host_device__ constexpr StaticTensorView(data_pointer data) noexcept
      : m_data(data)
  {
  }

  /**
   * \brief Resets the view to reference a dynamically sized tensor view with matching extents.
   * \param tensor_view Tensor view whose data pointer will be adopted.
   */
  template <typename accessor>
  __boba_host_device__ void reset(TensorView<accessor, dimension>& tensor_view)
  {
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      boba_always_assert_equal(sizes(d), tensor_view.sizes(d), "Incorrect sizing");
    }
    m_data = tensor_view.data();
  }

  /**
   * \brief Resets the view to reference a dynamically sized tensor with matching extents.
   * \param tensor Tensor whose data pointer will be adopted.
   */
  template <execution_space space>
  __boba_host_device__ void reset(Tensor<dimension, space, data_t>& tensor)
  {
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      boba_always_assert_equal(sizes(d), tensor.sizes(d), "Incorrect sizing");
    }
    m_data = tensor.data();
  }

  /**
   * \brief Returns the linear offset of a tensor index.
   * \param indices Tensor indices to convert.
   * \return The validated linear offset for \p indices.
   */
  [[nodiscard]]
  __boba_host_device__
  index_t index(index_array indices) const
  {
    return index(strides(), assert_indices(indices));
  }

  /**
   * \brief Returns the tensor index corresponding to a linear offset.
   * \param index Linear offset to convert.
   * \return The validated tensor multi-index.
   */
  [[nodiscard]]
  __boba_host_device__
  index_array multiindex(index_t index) const
  {
    return assert_indices(multiindex(sizes(), index));
  }

  /**
   * \brief Returns a reference to the element at a tensor multi-index.
   * \param multiindex Tensor indices identifying the element.
   * \return A reference to the selected element.
   */
  __boba_host_device__
  data_reference
  operator()(index_array multiindex) const
  {
    return m_data[index(strides(), assert_indices(multiindex))];
  }

  /**
   * \brief Returns a reference to the element at a linear offset.
   * \param index Linear element offset.
   * \return A reference to the selected element.
   */
  __boba_host_device__
  data_reference
  operator()(index_t index) const
  {
    boba_assert_nonnegative(index, "Negative index");
    boba_assert_lt(index, size(), "Out of bounds");
    return m_data[index];
  }

  /**
   * \brief Returns the underlying data pointer.
   * \return The raw pointer stored by this view.
   */
  [[nodiscard]]
  __boba_host_device__ constexpr data_pointer data() const noexcept
  {
    return m_data;
  }

private:
  data_pointer m_data = nullptr;

  /**
   * \brief Validates tensor indices in debug builds.
   * \param indices Tensor indices to validate.
   * \return The validated indices.
   */
  __boba_host_device__ static index_array assert_indices(index_array indices)
  {
    if constexpr (boba::is_boba_debug_mode())
    {
      for (std::size_t d = 0; d < dimension; d++)
      {
        index_t current_index = indices[d];
        boba_assert_nonnegative(current_index, "Negative index");
        boba_assert_lt(current_index, sizes(d), "Out of bounds");
      }
    }
    return indices;
  }

public:
  /**
   * \brief Interpolates the tensor from a separable stencil of indices and weights.
   * \tparam interpolation_points Number of interpolation points per dimension.
   * \param weights Interpolation weights for each dimension.
   * \param indices Sample indices for each dimension.
   * \return The interpolated tensor value.
   */
  template <size_t interpolation_points>
  [[nodiscard]]
  __boba_host_device__
  data_t interpolation(
    const ::boba::Array<::boba::Array<data_t, interpolation_points>, dimension>& weights,
    const ::boba::Array<::boba::Array<index_t, interpolation_points>, dimension>& indices) const
  {
    checkpoint();
    ::boba::Multiindexer<dimension> multilinear_index(::boba::filled_array<dimension>(interpolation_points));
    auto value = data_t(0.0);
    for (size_t i = 0_z; i < multilinear_index.size(); i++)
    {
      auto local_mid = multilinear_index.multiindex(i);
      ::boba::Array<index_t, dimension> full_tensor_mid;
      auto weight = data_t(1.0);
      for (size_t d = 0; d < dimension; d++)
      {
        auto interpolation_point = local_mid[d];
        full_tensor_mid[d] = indices[d][interpolation_point];
        weight *= weights[d][interpolation_point];
      }
      auto point_value = this->operator()(full_tensor_mid);
      value += weight * point_value;
    }
    return value;
  }
};

} // namespace boba

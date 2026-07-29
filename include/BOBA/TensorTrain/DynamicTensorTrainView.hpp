// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{
/**
 * \brief
 * Dynamic view of a tensor train. Useful when tt sizes are NOT known at compile time.
 */

template <size_t _dimension, typename _data_t>
struct DynamicTensorTrainView
{
  using data_t = _data_t;
  using data_reference = data_t&;
  using data_pointer = data_t*;

  static constexpr size_t dimension = _dimension;

  using index_array = boba::Array<index_t, dimension>;
  using data_array = boba::Array<data_t, dimension>;
  using multiindexer = Multiindexer<dimension>;

  index_array m_sizes;
  Array<index_t, dimension + 1> m_ranks;

  DynamicTensorTrainView() = default;

  /**
   * \brief
   * Create a DynamicTensorTrainView of a tensor train.
   * Requires that the tensor train is consistent with the view.
   */

  DynamicTensorTrainView(const TensorTrain<dimension, host_space, data_t>& tensor_train)
  {
    reset(tensor_train);
  }

  /**
   * \brief
   *  boba::Array of sizes
   */

  index_array sizes() const noexcept
  {
    return m_sizes;
  }

  /**
   * \brief
   *  i'th tensor size
   */
  __boba_host_device__
  index_t
  sizes(size_t i) const
  {
    checkpoint();
    return m_sizes[i];
  }

  /**
   * \brief
   *  Full size of compressed tensor
   */

  __boba_host_device__
  index_t
  size() const
  {
    return product(sizes());
  }

  /**
   * \brief
   *  Array of ranks
   */

  index_array ranks() const noexcept
  {
    return m_ranks;
  }

  /**
   * \brief
   *  i'th rank
   */

  __boba_host_device__
  index_t
  ranks(size_t i) const
  {
    return m_ranks[i];
  }

  /**
   * \brief
   *  number of cores
   */

  __boba_host_device__
  index_t
  get_number_cores() const
  {
    return dimension;
  }

  /**
   * \brief
   *  get_ranks_left
   */

  __boba_host_device__
  index_t
  get_ranks_left(index_t d) const
  {
    return ranks(d);
  }

  /**
   * \brief
   *  get_ranks_right
   */

  index_t get_ranks_right(index_t d) const
  {
    return ranks(d + 1);
  }

  /**
   * \brief
   *  total number of elements
   */

  index_t get_number_elements() const
  {
    size_t result = 0;
    for (size_t d = 0; d < dimension; ++d)
    {
      result += ranks(d) * sizes(d) * ranks(d + 1);
    }
    return result;
  }

  using const_core_view_t = TensorView<DefaultAccessor<data_t const>, 3>;
  Array<const_core_view_t, dimension> core_data;

  /**
   * \brief
   * Convert multiindex to index
   */

  __boba_host_device__
  index_t
  index(index_array indices) const
  {
    return multiindexer::index(multiindexer::precompute_strides(sizes()), indices);
  }

  /**
   * \brief
   * Convert index to multiindex
   */

  __boba_host_device__
  index_array
  multiindex(index_t index) const
  {
    return multiindexer::multiindex(sizes(), index);
  }

  /**
   * \brief
   * Create a DynamicQuantizedTensorTrainView of any type of tensor train.
   * Requires that the tensor train is consistent with the view.
   */

  template <typename tt_type>
  void reset(tt_type& tensor_train)
  {
    for (size_t d = 0; d < dimension; d++)
    {
      m_sizes[d] = tensor_train.sizes(d);
      m_ranks[d] = tensor_train.get_ranks_left(d);
    }
    m_ranks[dimension] = tensor_train.get_ranks_right(dimension - 1);

    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      core_data[d] = tensor_train.cores[d].const_view();
    }
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train.
   */

  __boba_host_device__
  data_t
  unroll_value(const index_array& multiindex) const
  {
    checkpoint();
    Array<Array<data_t, 1>, dimension> weights;
    Array<Array<index_t, 1>, dimension> indices;
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      weights[d][0] = static_cast<index_t>(1.0);
      indices[d][0] = multiindex[d];
    }
    checkpoint();
    return interpolation<1>(weights, indices);
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train.
   */

  __boba_host_device__
  data_t
  unroll_value(const index_t index) const
  {
    checkpoint();
    return unroll_value(multiindex(index));
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train.
   */

  __boba_host_device__
  data_t
  operator()(const index_array& multiindex) const
  {
    return unroll_value(multiindex);
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train.
   */

  __boba_host_device__
  data_t
  operator()(const index_t index) const
  {
    return unroll_value(multiindex(index));
  }

  /**
   * \brief
   * Given an interpolation stencil indices and weights, interpolates this tensor train.
   */

  template <size_t interpolation_points>
  __boba_host_device__
  data_t
  interpolation(
    const Array<Array<data_t, interpolation_points>, dimension>& weights,
    const Array<Array<index_t, interpolation_points>, dimension>& indices) const
  {
    checkpoint();
    Tensor<1, host_space, data_t> output = interpolation_helper<interpolation_points>(dimension, weights, indices);
    auto output_view = output.view();
    return output_view(0);
  }

  template <size_t interpolation_points>
  Tensor<1, host_space, data_t> interpolation_helper(
    size_t recursion_index,
    const Array<Array<data_t, interpolation_points>, dimension>& weights,
    const Array<Array<index_t, interpolation_points>, dimension>& indices) const
  {
    if (recursion_index == 0)
    {
      Tensor<1, host_space, data_t> seed({1});
      auto seed_view = seed.view();
      seed_view(0) = 1.0;
      return seed;
    }

    checkpoint();
    size_t core_index = recursion_index - 1;
    size_t ranks_right = ranks(recursion_index);

    auto traversal = interpolation_helper<interpolation_points>(recursion_index - 1, weights, indices);

    Tensor<1, host_space, data_t> output({ranks(recursion_index)});
    auto output_view = output.view();
    auto core_weights = weights[core_index];
    auto core_indices = indices[core_index];
    for (size_t r = 0; r < ranks_right; r++)
    {
      output_view(r) = 0.0;
    }
    for (size_t p = 0; p < interpolation_points; p++)
    {
      auto traversed_core = contract_core(core_index, traversal, core_indices[p]);
      auto traversed_core_view = traversed_core.view();
      auto weight = core_weights[p];
      for (size_t r = 0; r < ranks_right; r++)
      {
        output_view(r) += traversed_core_view(r) * weight;
      }
    }
    return output;
  }

  /**
   * \brief
   * Given a vector and an extent index, contracts a slice of a core against a vector.
   * x_r = \sum\limits_l v_l C_{lir}
   */

  Tensor<1, host_space, data_t> contract_core(
    size_t core_index,
    Tensor<1, host_space, data_t> traversal,
    const index_t& index) const
  {
    checkpoint();
    size_t ranks_left = ranks(core_index);
    size_t ranks_right = ranks(core_index + 1);
    Tensor<1, host_space, data_t> output({ranks_right});

    //
    // Views
    //
    auto traversal_view = traversal.const_view();
    auto output_view = output.view();
    auto core_view = core_data[core_index];

    {
      for (size_t rr = 0; rr < ranks_right; rr++)
      {
        auto result = data_t(0.0);
        auto offset = core_view.index({0, index, rr});
        for (size_t rl = 0; rl < ranks_left; rl++)
        {
          auto traverse_value = traversal_view(rl);
          auto core_value = core_view(offset + rl);
          result += traverse_value * core_value;
        }
        output_view(rr) = result;
      }
    }
    return output;
  }
};

/**
 * \brief
 * Dynamic view of a tensor train. Useful when tt sizes are NOT known at compile time.
 */

template <typename _data_t, bool permute_cores = false>
struct DynamicQuantizedTensorTrainView
{
  using data_t = _data_t;
  using data_reference = data_t&;
  using data_pointer = data_t*;

  std::size_t dimension = 0;

  using index_array = std::vector<index_t>;
  using data_array = std::vector<data_t>;

  index_array m_sizes;
  index_array m_ranks;

  DynamicQuantizedTensorTrainView() = default;

  /**
   * \brief
   * Create a DynamicQuantizedTensorTrainView of a quantized tensor train.
   * Requires that the tensor train is consistent with the view.
   */

  DynamicQuantizedTensorTrainView(QuantizedTensorTrain<host_space, data_t>& tensor_train)
  {
    reset(tensor_train);
  }

  /**
   * \brief
   * Create a DynamicQuantizedTensorTrainView of a tensor train.
   * Requires that the tensor train is consistent with the view.
   */

  template <size_t _dimension>
  DynamicQuantizedTensorTrainView(TensorTrain<_dimension, host_space, data_t>& tensor_train)
  {
    reset(tensor_train);
  }

  /**
   * \brief
   *  std::vector of sizes
   */

  index_array sizes() const noexcept
  {
    return m_sizes;
  }

  /**
   * \brief
   *  i'th tensor size
   */
  __boba_host_device__
  index_t
  sizes(size_t i) const
  {
    checkpoint();
    return sizes()[i];
  }

  /**
   * \brief
   *  Full size of compressed tensor
   */

  __boba_host_device__
  index_t
  size() const
  {
    auto result = data_t(1.0);
    for (const auto& extent : sizes())
    {
      result *= extent;
    }
    return result;
  }

  /**
   * \brief
   *  std::vector of ranks
   */

  index_array ranks() const noexcept
  {
    return m_ranks;
  }

  /**
   * \brief
   *  i'th rank
   */

  __boba_host_device__
  index_t
  ranks(size_t i) const
  {
    return m_ranks[i];
  }

  /**
   * \brief
   *  number of cores
   */

  __boba_host_device__
  index_t
  get_number_cores() const
  {
    return dimension;
  }

  /**
   * \brief
   *  get_ranks_left
   */

  __boba_host_device__
  index_t
  get_ranks_left(index_t d) const
  {
    return ranks(d);
  }

  /**
   * \brief
   *  get_ranks_right
   */

  index_t get_ranks_right(index_t d) const
  {
    return ranks(d + 1);
  }

  /**
   * \brief
   *  total number of elements
   */

  index_t get_number_elements() const
  {
    size_t result = 0;
    for (size_t d = 0; d < dimension; ++d)
    {
      result += ranks(d) * sizes(d) * ranks(d + 1);
    }
    return result;
  }

  /**
   * \brief
   * A string describing the tensor train ranks
   */

  std::string ranks_string() const
  {
    std::string string_of_ranks = "";
    string_of_ranks += " ( ";
    for (size_t d = 0; d < get_number_cores(); d++)
    {
      bool is_last = d == get_number_cores() - 1;
      size_t rank = get_ranks_left(d);
      if (is_last)
      {
        string_of_ranks += std::to_string(rank) + ", ";
        string_of_ranks += std::to_string(get_ranks_right(d)) + " )";
      }
      else
      {
        string_of_ranks += std::to_string(rank) + ", ";
      }
    }
    return string_of_ranks;
  }

  /**
   * \brief
   * A string describing the size/length of each core
   */

  std::string sizes_string() const
  {
    std::string string_of_sizes = "";
    string_of_sizes += " ( " + std::to_string(sizes(0));
    for (size_t d = 1; d < m_sizes.size(); d++)
    {
      string_of_sizes += ", " + std::to_string(sizes(d));
    }
    string_of_sizes += " )";
    return string_of_sizes;
  }

  using const_core_view_t = TensorView<DefaultAccessor<data_t const>, 3>;
  std::vector<const_core_view_t> core_data;

  /**
   * \brief
   * Convert multiindex to index
   */

  __boba_host_device__
  index_t
  index(index_array indices) const
  {
    index_t stride = 1;
    index_t index = 0;
    index += indices[0];
    for (std::size_t d = 1; d < dimension; ++d)
    {
      index += stride * indices[d];
      stride *= sizes(d - 1);
    }
    return index;
  }

  /**
   * \brief
   * Convert index to multiindex
   */

  __boba_host_device__
  index_array
  multiindex(index_t index) const
  {
    index_array indices;
    indices.resize(dimension);
    checkpoint();
    for (std::size_t d = 0; d < dimension - 1; d++)
    {
      indices[d] = index % sizes(d);
      index /= sizes(d);
    }
    checkpoint();
    indices[dimension - 1] = index;
    checkpoint();
    return indices;
  }

  /**
   * \brief
   * Create a DynamicQuantizedTensorTrainView of any type of tensor train.
   * Requires that the tensor train is consistent with the view.
   */

  template <typename tt_type>
  void reset(tt_type& tensor_train)
  {
    dimension = tensor_train.get_number_cores();
    m_sizes.resize(dimension);
    m_ranks.resize(dimension + 1);

    for (size_t d = 0; d < dimension; d++)
    {
      m_sizes[d] = tensor_train.get_core_size(d);
      m_ranks[d] = tensor_train.get_ranks_left(d);
    }

    m_ranks[dimension] = tensor_train.get_ranks_right(dimension - 1);

    checkpoint();
    core_data.resize(dimension);

    for (size_t d = 0; d < dimension; d++)
    {
      if (permute_cores)
      {
        permute({"rank left", "index", "rank right"}, tensor_train.cores[d], {"rank left", "rank right", "index"});
      }
      core_data[d] = tensor_train.cores[d].const_view();
    }
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train.
   */

  __boba_host_device__
  data_t
  unroll_value(const index_array& multiindex) const
  {
    checkpoint();
    std::vector<::boba::Array<data_t, 1>> weights;
    std::vector<::boba::Array<index_t, 1>> indices;
    weights.resize(dimension);
    indices.resize(dimension);
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      weights[d][0] = static_cast<index_t>(1.0);
      indices[d][0] = multiindex[d];
    }
    checkpoint();
    return interpolation<1>(weights, indices);
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train.
   */

  __boba_host_device__
  data_t
  unroll_value(const index_t index) const
  {
    checkpoint();
    return unroll_value(multiindex(index));
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train.
   */

  __boba_host_device__
  data_t
  operator()(const index_array& multiindex) const
  {
    return unroll_value(multiindex);
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train.
   */

  __boba_host_device__
  data_t
  operator()(const index_t index) const
  {
    return unroll_value(multiindex(index));
  }

  /**
   * \brief
   * Given an interpolation stencil indices and weights, interpolates this tensor train.
   */

  template <size_t interpolation_points>
  __boba_host_device__
  data_t
  interpolation(
    const std::vector<::boba::Array<data_t, interpolation_points>>& weights,
    const std::vector<::boba::Array<index_t, interpolation_points>>& indices) const
  {
    checkpoint();
    Tensor<1, host_space, data_t> output = interpolation_helper<interpolation_points>(dimension, weights, indices);
    auto output_view = output.view();
    return output_view(0);
  }

  template <size_t interpolation_points>
  Tensor<1, host_space, data_t> interpolation_helper(
    size_t recursion_index,
    const std::vector<::boba::Array<data_t, interpolation_points>>& weights,
    const std::vector<::boba::Array<index_t, interpolation_points>>& indices) const
  {
    if (recursion_index == 0)
    {
      Tensor<1, host_space, data_t> seed({1});
      auto seed_view = seed.view();
      seed_view(0) = 1.0;
      return seed;
    }

    checkpoint();
    size_t core_index = recursion_index - 1;
    size_t ranks_right = ranks(recursion_index);

    auto traversal = interpolation_helper<interpolation_points>(recursion_index - 1, weights, indices);

    Tensor<1, host_space, data_t> output({ranks(recursion_index)});
    auto output_view = output.view();
    auto core_weights = weights[core_index];
    auto core_indices = indices[core_index];
    for (size_t r = 0; r < ranks_right; r++)
    {
      output_view(r) = 0.0;
    }
    for (size_t p = 0; p < interpolation_points; p++)
    {
      auto traversed_core = contract_core(core_index, traversal, core_indices[p]);
      auto traversed_core_view = traversed_core.view();
      auto weight = core_weights[p];
      for (size_t r = 0; r < ranks_right; r++)
      {
        output_view(r) += traversed_core_view(r) * weight;
      }
    }
    return output;
  }

  /**
   * \brief
   * Given a vector and an extent index, contracts a slice of a core against a vector.
   * x_r = \sum\limits_l v_l C_{lir}
   */

  Tensor<1, host_space, data_t> contract_core(
    size_t core_index,
    Tensor<1, host_space, data_t> traversal,
    const index_t& index) const
  {
    checkpoint();
    size_t ranks_left = ranks(core_index);
    size_t ranks_right = ranks(core_index + 1);
    Tensor<1, host_space, data_t> output({ranks_right});

    //
    // Views
    //
    auto traversal_view = traversal.const_view();
    auto output_view = output.view();
    auto core_view = core_data[core_index];

    if constexpr (permute_cores)
    {
      for (size_t rr = 0; rr < ranks_right; rr++)
      {
        auto result = data_t(0.0);
        auto offset = core_view.index({0, rr, index});
        for (size_t rl = 0; rl < ranks_left; rl++)
        {
          auto traverse_value = traversal_view(rl);
          auto core_value = core_view(offset + rl);
          result += traverse_value * core_value;
        }
        output_view(rr) = result;
      }
    }
    else
    {
      for (size_t rr = 0; rr < ranks_right; rr++)
      {
        auto result = data_t(0.0);
        auto offset = core_view.index({0, index, rr});
        for (size_t rl = 0; rl < ranks_left; rl++)
        {
          auto traverse_value = traversal_view(rl);
          auto core_value = core_view(offset + rl);
          result += traverse_value * core_value;
        }
        output_view(rr) = result;
      }
    }
    return output;
  }

  /**
   * \brief
   * Binary search of viewed tt.
   * Given f, Find i and j such that f(x_i) < f < f(x_j)
   */

  Array<index_t, 2> binary_search(const data_t search_value) const
  {
    size_t id_left = 0;
    size_t id_right = size() - 1;
    while (id_right > id_left + 1)
    {
      auto id_guess = (id_left + id_right) / 2;
      auto value = unroll_value(id_guess);

      if (value < search_value)
      {
        id_left = id_guess;
      }
      else
      {
        id_right = id_guess;
      }
    }

    Array<index_t, 2> bracket;
    bracket[0] = id_left;
    bracket[1] = id_right;

    boba_assert_lt(bracket[0], bracket[1], "Unexpected values");

    return bracket;
  }
};

} // namespace boba

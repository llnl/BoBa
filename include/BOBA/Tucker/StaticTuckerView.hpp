// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Mode ordering of StaticTuckerView. When StaticTuckerView is created from a Tucker, the cores will be transposed
 * from the canonical mode ordering to the mode ordering chosed by this variable
 * this has strong performance implications
 */

enum tucker_mode_order : size_t
{
  ir, // extent, rank
  ri, // rank, extent
};

/**
 * \brief
 * Static view of a Tucker. Useful when sizes and ranks are known at compile time.
 * See StaticTensor - static_sizes
 */

template <typename tucker_sizes, typename tucker_ranks, typename _data_t, tucker_mode_order mode_ordering = tucker_mode_order::ir>
struct StaticTuckerView
{
  using data_t = _data_t;
  using data_reference = data_t&;
  using data_pointer = data_t*;

  static constexpr std::size_t dimension = tucker_sizes::size();
  using index_array = ::boba::Array<index_t, dimension>;

  __boba_host_device__ static constexpr index_array sizes() noexcept
  {
    return ::boba::typed_array<index_t>(::boba::to_array(tucker_sizes{}));
  }

  __boba_host_device__ static constexpr index_t sizes(size_t i) noexcept
  {
    return sizes()[i];
  }

  __boba_host_device__ static constexpr index_t size() noexcept
  {
    return ::boba::product(sizes());
  }

  __boba_host_device__ static constexpr ::boba::Array<index_t, dimension> ranks() noexcept
  {
    return ::boba::typed_array<index_t>(::boba::to_array(tucker_ranks{}));
  }

  __boba_host_device__ static constexpr index_t ranks(size_t i) noexcept
  {
    return ranks()[i];
  }

  ::boba::Array<data_pointer, dimension> core_data;
  data_pointer R_core_data;

  __boba_host_device__ static constexpr index_t index(index_array indices) noexcept
  {
    // TODO<organization> reimpl in terms of Multiindexer
    constexpr index_array sizes = ::boba::typed_array<index_t>(::boba::to_array(tucker_sizes{}));
    constexpr index_array strides = ::boba::Multiindexer<dimension>::precompute_strides(sizes);
    index_t index = 0;
    if constexpr (dimension > 0)
    {
      index += indices[0];
      for (std::size_t d = 1; d < dimension; ++d)
      {
        index += strides[d] * indices[d];
      }
    }
    return index;
  }

  __boba_host_device__ static constexpr index_array multiindex(index_t index) noexcept
  {
    // TODO<organization> reimpl in terms of Multiindexer
    index_array indices;
    if constexpr (dimension > 0)
    {
      for (std::size_t d = 0; d < dimension - 1; d++)
      {
        indices[d] = index % sizes(d);
        index /= sizes(d);
      }
      indices[dimension - 1] = index;
    }
    return indices;
  }

  StaticTuckerView()
  {
  }

  /**
   * \brief
   * Create a StaticTuckerView of a tucker.
   * Requires that the tucker is consistent with the view.
   */

  template <execution_space space>
  StaticTuckerView(Tucker<dimension, space, data_t>& Tucker)
  {
    reset(Tucker);
  }

  /**
   * \brief
   * Create a StaticTuckerView of any type of tucker.
   * Requires that the tucker is consistent with the view.
   */

  template <typename tensorlike_t>
  void reset(tensorlike_t& Tucker)
  {
    checkpoint();
    boba_always_assert_equal(dimension, static_cast<size_t>(Tucker.get_number_cores()), "Mismatched number of cores.");
    for (size_t d = 0; d < dimension; d++)
    {
      boba_always_assert_equal(sizes(d), Tucker.cores[d].sizes(0), "Incorrect sizing");
      boba_always_assert_equal(ranks(d), Tucker.get_ranks(d), "Incorrect sizing");
      if constexpr (mode_ordering == tucker_mode_order::ri)
      {
        permute({"index", "ranks"}, Tucker.cores[d], {"ranks", "index"});
      }
      core_data[d] = Tucker.cores[d].data();
    }
    R_core_data = Tucker.R_core.data();
  }

  /**
   * \brief
   * Unrolls a single value of the tucker.
   */

  __boba_host_device__
  data_t
  unroll_value(const ::boba::Array<index_t, dimension>& multiindex) const
  {
    checkpoint();
    ::boba::Array<::boba::Array<data_t, 1>, dimension> weights;
    ::boba::Array<::boba::Array<index_t, 1>, dimension> indices;
    for (size_t d = 0; d < dimension; d++)
    {
      weights[d][0] = static_cast<index_t>(1.0);
      indices[d][0] = multiindex[d];
    }
    return interpolation<1>(weights, indices);
  }

  /**
   * \brief
   * Unrolls a single value of the tucker.
   */

  __boba_host_device__
  data_t
  unroll_value(const index_t index) const
  {
    return unroll_value(multiindex(index));
  }

  /**
   * \brief
   * Given an interpolation stencil indices and weights, interpolates this tucker.
   */

  template <size_t interpolation_points>
  __boba_host_device__
  data_t
  interpolation(
    const ::boba::Array<::boba::Array<data_t, interpolation_points>, dimension>& weights,
    const ::boba::Array<::boba::Array<index_t, interpolation_points>, dimension>& indices) const
  {
    Multiindexer<dimension> interp_mider(filled_array<dimension>(interpolation_points));

    data_t summation = 0;

    for (size_t id = 0; id < interp_mider.size(); id++)
    {
      // Multilinear interpolation multiindex
      auto mid = interp_mider.multiindex(id);

      // Compute weight
      data_t weight = 1.0;
      for (size_t dim = 0; dim < dimension; dim++)
      {
        auto interp_id_along_dim = mid[dim];
        weight *= weights[dim][interp_id_along_dim];
      }

      // Compute value
      Multiindexer<dimension> ranks_mider(ranks());
      for (size_t ir = 0; ir < ranks_mider.size(); ir++)
      {
        auto ranks_ids = ranks_mider.multiindex(ir);

        auto value = R_core_data[ir];
        for (size_t dim = 0; dim < dimension; dim++)
        {
          auto interp_id_along_dim = mid[dim];
          auto core_id = indices[dim][interp_id_along_dim];
          auto core_rank_id = ranks_ids[dim];
          if constexpr (mode_ordering == tucker_mode_order::ri)
          {
            value *= core_data[dim][core_rank_id + ranks(dim) * core_id];
          }
          else
          {
            value *= core_data[dim][core_id + sizes(dim) * core_rank_id];
          }
        }

        // Compute interpolation
        summation += value * weight;
      }
    }

    return summation;
  }

  /**
   * \brief
   * Given an interpolation stencil indices and weights, interpolates this tucker.
   */

  template <size_t interpolation_points>
  __boba_device__
  data_t
  interpolation_teams(
    const ::boba::Array<::boba::Array<data_t, interpolation_points>, dimension>& weights,
    const ::boba::Array<::boba::Array<index_t, interpolation_points>, dimension>& indices) const
  {
    detail::ignore(weights);
    detail::ignore(indices);
    /*
    TODO
    */
  }
};

} // namespace boba

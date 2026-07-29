// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Mode ordering of StaticTensorTrainView. When StaticTensorTrainView is created from a tt, the cores will be permuted
 * from the canonical mode ordering to the mode ordering chosed by this variable
 * this has strong performance implications
 */

enum tt_mode_order : size_t
{
  lir, // left rank, extent, right rank (default)
  ilr, // extent, left rank, right rank
  lri, // left rank, right rank, extent
};

/**
 * \brief
 * Static view of a tensor train. Useful when tt sizes are known at compile time.
 * See StaticTensor - static_sizes
 */

template <typename tt_sizes, typename tt_ranks, typename _data_t, tt_mode_order mode_ordering = tt_mode_order::lir>
struct StaticTensorTrainView
{
  using data_t = _data_t;
  using data_reference = data_t&;
  using data_pointer = data_t*;

  static constexpr std::size_t dimension = tt_sizes::size();
  using index_array = ::boba::Array<index_t, dimension>;
  using mider_type = Multiindexer<dimension>;

  __boba_host_device__ static constexpr index_array sizes() noexcept
  {
    return ::boba::typed_array<index_t>(::boba::to_array(tt_sizes{}));
  }

  __boba_host_device__ static constexpr index_t sizes(size_t i) noexcept
  {
    return sizes()[i];
  }

  __boba_host_device__ static constexpr index_t size() noexcept
  {
    return ::boba::product(sizes());
  }

  __boba_host_device__ static constexpr ::boba::Array<index_t, dimension + 1> ranks() noexcept
  {
    return ::boba::typed_array<index_t>(::boba::to_array(tt_ranks{}));
  }

  __boba_host_device__ static constexpr index_t ranks(size_t i) noexcept
  {
    return ranks()[i];
  }

  ::boba::Array<data_pointer, dimension> core_data;

  __boba_host_device__ static constexpr index_t index(index_array indices) noexcept
  {
    constexpr index_array sizes = ::boba::typed_array<index_t>(::boba::to_array(tt_sizes{}));
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

  StaticTensorTrainView()
  {
  }

  /**
   * \brief
   * Create a StaticTensorTrainView of a quantized tensor train.
   * Requires that the tensor train is consistent with the view.
   */

  template <execution_space space>
  StaticTensorTrainView(QuantizedTensorTrain<space, data_t>& tensor_train)
  {
    reset(tensor_train);
  }

  /**
   * \brief
   * Create a StaticTensorTrainView of a tensor train.
   * Requires that the tensor train is consistent with the view.
   */

  template <execution_space space>
  StaticTensorTrainView(TensorTrain<dimension, space, data_t>& tensor_train)
  {
    reset(tensor_train);
  }

  /**
   * \brief
   * Create a StaticTensorTrainView of any type of tensor train.
   * Requires that the tensor train is consistent with the view.
   */

  template <typename tt_type>
  void reset(tt_type& tensor_train)
  {
    checkpoint();
    boba_always_assert_equal(dimension, static_cast<size_t>(tensor_train.get_number_cores()), "Mismatched number of cores.");
    for (size_t d = 0; d < dimension; d++)
    {
      boba_always_assert_equal(sizes(d), static_cast<index_t>(tensor_train.sizes(d)), "Incorrect sizing");
      boba_always_assert_equal(ranks(d), static_cast<index_t>(tensor_train.get_ranks_left(d)), "Incorrect sizing");
      boba_always_assert_equal(ranks(d + 1), static_cast<index_t>(tensor_train.get_ranks_right(d)), "Incorrect sizing");
      if constexpr (mode_ordering == tt_mode_order::ilr)
      {
        permute({"l", "i", "r"}, tensor_train.cores[d], {"i", "l", "r"});
      }
      else if constexpr (mode_ordering == tt_mode_order::lri)
      {
        permute({"l", "i", "r"}, tensor_train.cores[d], {"l", "r", "i"});
      }
      else
      {
        static_assert(mode_ordering == tt_mode_order::lir);
      }
      core_data[d] = tensor_train.cores[d].data();
    }
  }

  /**
   * \brief
   * Return a staticArray of the sizes of the dim'th core.
   */

  template <size_t dim>
  __boba_host_device__ static constexpr auto get_core_sizes()
  {
    constexpr size_t ranks_left = ranks(dim);
    constexpr size_t size = sizes(dim);
    constexpr size_t ranks_right = ranks(dim + 1);
    if constexpr (mode_ordering == tt_mode_order::lir)
    {
      return StaticArray<index_t, ranks_left, size, ranks_right>{};
    }
    else if constexpr (mode_ordering == tt_mode_order::ilr)
    {
      return StaticArray<index_t, size, ranks_left, ranks_right>{};
    }
    else if constexpr (mode_ordering == tt_mode_order::lri)
    {
      return StaticArray<index_t, ranks_left, ranks_right, size>{};
    }
    else // unsupported case
    {
      boba_error("Unsupported mode ordering");
      return StaticArray<index_t, 0, 0, 0>{};
    }
  }

  __boba_host_device__ auto get_core_sizes(size_t dim) const
  {
    size_t ranks_left = ranks(dim);
    size_t size = sizes(dim);
    size_t ranks_right = ranks(dim + 1);
    if constexpr (mode_ordering == tt_mode_order::lir)
    {
      return Array<index_t, 3>{ranks_left, size, ranks_right};
    }
    else if constexpr (mode_ordering == tt_mode_order::ilr)
    {
      return Array<index_t, 3>{size, ranks_left, ranks_right};
    }
    else if constexpr (mode_ordering == tt_mode_order::lri)
    {
      return Array<index_t, 3>{ranks_left, ranks_right, size};
    }
    else
    {
      boba_error("Unsupported mode ordering");
      return Array<index_t, 3>{0, 0, 0};
    }
  }

  /**
   * \brief
   * Fetch the value of the dim'th core at index i and left/right ranks l/r
   */

  template <size_t dim>
  __boba_host_device__
  data_t
  get_core_value(index_t l, index_t i, index_t r)
  {
    StaticTensorView<data_t, decltype(get_core_sizes<dim>())> core_view(core_data[dim]);
    if constexpr (mode_ordering == tt_mode_order::lir)
    {
      return core_view({l, i, r});
    }
    else if constexpr (mode_ordering == tt_mode_order::ilr)
    {
      return core_view({i, l, r});
    }
    else if constexpr (mode_ordering == tt_mode_order::lri)
    {
      return core_view({l, r, i});
    }
    else
    {
      boba_error("Unsupported mode ordering");
      return core_view({l, r, i});
    }
  }

  /**
   * \brief
   * Fetch the value of the dim'th core at index i and left/right ranks l/r
   */

  __boba_host_device__
  data_t
  get_core_value(index_t dim, index_t l, index_t i, index_t r) const
  {
    if constexpr (mode_ordering == tt_mode_order::lir)
    {
      TensorView<DefaultAccessor<data_t>, 3> core_view(core_data[dim], {ranks(dim), sizes(dim), ranks(dim + 1)});
      return core_view({l, i, r});
    }
    else if constexpr (mode_ordering == tt_mode_order::ilr)
    {
      TensorView<DefaultAccessor<data_t>, 3> core_view(core_data[dim], {sizes(dim), ranks(dim), ranks(dim + 1)});
      return core_view({i, l, r});
    }
    else if constexpr (mode_ordering == tt_mode_order::lri)
    {
      TensorView<DefaultAccessor<data_t>, 3> core_view(core_data[dim], {ranks(dim), ranks(dim + 1), sizes(dim)});
      return core_view({l, r, i});
    }
    else
    {
      boba_error("Unsupported mode ordering");
      TensorView<DefaultAccessor<data_t>, dimension> core_view(core_data[dim], {0, 0, 0});
      return core_view({1, 1, 1});
    }
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train.
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
   * Unrolls a single value of the tensor train, specialized for GPUs
   */

  __boba_host_device__
  data_t
  unroll_value_teams(const ::boba::Array<::boba::Array<index_t, 1>, dimension>& indices) const
  {
    ::boba::Array<index_t, dimension> indices_array{0};
    for (size_t dim = 0; dim < dimension; dim++)
    {
      indices_array[dim] = indices[dim][0];
    }
    return unroll_value_teams(indices_array);
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train, specialized for GPUs
   */

  template <size_t block_size = 256>
  __boba_host_device__
  data_t
  unroll_value_teams(const ::boba::Array<index_t, dimension>& multiindex) const
  {
    checkpoint();
    auto id = mider_type::index(mider_type::precompute_strides(tt_sizes()), multiindex);
    auto tid = detail::thread_id();
    auto num_threads = detail::threads_actual();

    // Put the list of ids and whether or not their outputs have been computed into shared memory
    __boba_shared__ size_t id_set[block_size]; // TODO<optimization> short to save on memory?
    __boba_shared__ bool id_computed[block_size];
    id_set[tid] = id;
    id_computed[tid] = false;

    // Scratch memory for computing the contractions
    constexpr size_t max_ranks = max(ranks());
    __boba_shared__ data_t scratch_in[max_ranks];
    __boba_shared__ data_t scratch_out[max_ranks];

    data_t output;

    detail::thread_synchronize(); // TODO<optimization> check if this can be removed

    // Loop over tids and compute output for each tid's id, then share result with tids with same id
    for (size_t active_tid = 0; active_tid < num_threads; active_tid++)
    {
      bool needs_computed = not(id_computed[active_tid]);
      if (needs_computed)
      {
        if (tid == 0)
        {
          scratch_in[0] = 1.0;
          scratch_out[0] = 1.0;
        }
        detail::thread_synchronize();
        auto active_id = id_set[active_tid];
        auto active_mid = mider_type::multiindex(sizes(), active_id);
        for (size_t dim_p1 = dimension; dim_p1 > 0; dim_p1--)
        {
          auto dim = dim_p1 - 1;
          bool is_dim_even = is_even(dim);
          // when dim == 0, view_out is guaranteed to be scratch_out
          auto* view_out = is_dim_even ? scratch_out : scratch_in;
          auto* view_in = is_dim_even ? scratch_in : scratch_out;
          // set view_out[r] = 0 for all r \in [0, max_ranks)
          {
            size_t offset = 0;
            while (offset < max_ranks)
            {
              size_t offset_tid = offset + tid;
              if (offset_tid < max_ranks)
              {
                view_out[offset_tid] = 0.0;
              }
              offset += num_threads;
            }
          }
          detail::thread_synchronize();
          // Perform contraction view_out[l] = \sum_r core({l, i, r})*scratch[r] for given i
          Multiindexer<2> slice_mider({ranks(dim), ranks(dim + 1)});
          size_t offset = 0;
          while (offset < slice_mider.size())
          {
            size_t offset_tid = offset + tid;
            if (offset_tid < slice_mider.size())
            {
              auto i = active_mid[dim];
              auto [l, r] = slice_mider.multiindex(offset_tid);
              auto core_value = get_core_value(dim, l, i, r);
              auto vec_value = view_in[r];
              atomics::atomic_add(&view_out[l], core_value * vec_value);
            }
            offset += num_threads;
          }
          detail::thread_synchronize(); // sync after atomic add to ensure all threads have the same value
        }

        // if this thread's id is equal to the id for which we just computed the output,
        // then take the result as output and mark this thread's id as computed
        if (id == active_id)
        {
          output = scratch_out[0];
          id_computed[tid] = true;
        }
        detail::thread_synchronize();
      }
    }

    return output;
  }

  /**
   * \brief
   * Unrolls a single value of the tensor train.
   */

  __boba_host_device__
  data_t
  unroll_value(const index_t index) const
  {
    return unroll_value(multiindex(index));
  }

  /**
   * \brief
   * Given an interpolation stencil indices and weights, interpolates this tensor train.
   */

  template <index_t interpolation_points>
  __boba_host_device__
  data_t
  interpolation(
    const ::boba::Array<::boba::Array<data_t, interpolation_points>, dimension>& weights,
    const ::boba::Array<::boba::Array<index_t, interpolation_points>, dimension>& indices) const
  {
    checkpoint();
    StaticTensor<StaticArray<std::size_t, 1>, data_t> output = interpolation_helper<dimension, interpolation_points>(weights, indices);
    static_assert(output.dimension == 1, "Sanity check output type fail.");
    static_assert(output.size() == 1, "Sanity check output size fail.");
    auto output_view = output.view();
    return output_view(0);
  }

  /**
   * \brief
   * Given the description of of a tensor size 'subextents', computes the total size of a temporary array needed to unroll the associated sub-tensor-train
   * Unrolling the parts of the tensor train requires dimension+1 temporaries, which would go into a scratch memory array.
   * This function makes use of the fact that you can reuse parts of the scratch memory.
   *
   * @param[in] subextents describing the size of the tensor to read from the tensor train
   * @param[inout] temporary_offsets Array containing the offsets into temporary memory that each temporary would need
   * @return Total size of the needed temporary scratch memory
   */

  __boba_host_device__
  size_t
  compute_space_needed_to_fetch_subtensor(
    const Array<index_t, dimension>& subextents,
    Array<index_t, dimension + 1>& temporary_offsets) const
  {
    // Compute size of all temporaries for tensor contractions
    ::boba::Array<index_t, 1> one{1};
    Multiindexer<dimension + 1> subtensor_sizes_helper(::boba::concatenate(subextents, one));

    ::boba::Array<index_t, dimension + 1> temporary_sizes;
    for (size_t d = 0; d < dimension + 1; d++)
    {
      index_t rank = ranks(d);
      index_t size = subtensor_sizes_helper.strides(d);
      temporary_sizes[d] = size * rank;
    }

    // Compute offsets of all temporaries for tensor contractions
    // Optimization: restart the offset counter to zero if the next object would fit in the space between 0 and offset

    index_t total_size = 0;
    temporary_offsets[0] = 0;
    for (size_t d = 1; d < dimension + 1; d++)
    {
      index_t last_offset = temporary_offsets[d - 1];
      index_t last_size = temporary_sizes[d - 1];
      index_t this_size = temporary_sizes[d];
      if (this_size < last_offset)
      {
        temporary_offsets[d] = 0;
      }
      else
      {
        temporary_offsets[d] = last_offset + last_size;
      }
      total_size = ::boba::max(total_size, last_offset + last_size + this_size);
    }

    return total_size;
  }

  /**
   * \brief
   * Given a description of a bounding box, computes the corresponding subtensor
   */

  template <index_t scratch_memory_size>
  __boba_host_device__ ::boba::Multiindexer<dimension> fetch_subtensor(
    ::boba::Array<index_t, dimension> indices_minima,
    ::boba::Array<index_t, dimension> indices_maxima,
    data_t (&scratch_memory)[scratch_memory_size],
    index_t threads_actual,
    index_t& tensor_offset) const
  {
    // Find size of bounding box (call this subextents)
    ::boba::Array<index_t, dimension> subextents;
    for (size_t d = 0; d < dimension; d++)
    {
      subextents[d] = indices_maxima[d] - indices_minima[d] + 1;
    }

    // Compute size of all temporaries for tensor contractions
    ::boba::Array<index_t, 1> one{1};
    Multiindexer<dimension + 1> subtensor_sizes_helper(::boba::concatenate(subextents, one));

    ::boba::Array<index_t, dimension + 1> temporary_sizes;
    for (size_t d = 0; d < dimension + 1; d++)
    {
      index_t rank = ranks(d);
      index_t size = subtensor_sizes_helper.strides(d);
      temporary_sizes[d] = size * rank;
    }

    // Compute offsets of all temporaries for tensor contractions
    // Optimization: restart the offset counter to zero if you can fit something there
    ::boba::Array<index_t, dimension + 1> temporary_offsets;
    index_t total_size = compute_space_needed_to_fetch_subtensor(subextents, temporary_offsets);
    temporary_offsets[0] = 0;
    for (size_t d = 1; d < dimension + 1; d++)
    {
      index_t last_offset = temporary_offsets[d - 1];
      index_t last_size = temporary_sizes[d - 1];
      index_t this_size = temporary_sizes[d];
      if (this_size < last_offset)
      {
        temporary_offsets[d] = 0;
      }
      else
      {
        temporary_offsets[d] = last_offset + last_size;
      }

      total_size = ::boba::max(total_size, last_offset + last_size + this_size);
    }

    ::boba::Multiindexer<dimension> subtensor_mider(subextents);

    tensor_offset = temporary_offsets[dimension];
    scratch_memory[temporary_offsets[0]] = 1;

    for (size_t d = 0; d < dimension; d++)
    {
      contract_core_teams<dimension>(
        temporary_offsets[d],
        temporary_offsets[d + 1],
        scratch_memory,
        subextents,
        subtensor_sizes_helper,
        indices_minima[d],
        threads_actual,
        d);
    }

    tensor_offset = temporary_offsets[dimension];
    return subtensor_mider;
  }

  /**
   * \brief
   * Given a pointer to a subtensor and indexing information, interpolate the subtensor
   */

  template <size_t interpolation_points, size_t scratch_memory_size>
  __boba_host_device__
  data_t
  interpolate_from_subtensor(
    Array<Array<index_t, interpolation_points>, dimension> indices,
    Array<Array<data_t, interpolation_points>, dimension> weights,
    boba::Array<index_t, dimension> indices_minima,
    Multiindexer<dimension> subtensor_mider,
    data_t (&scratch_memory)[scratch_memory_size],
    index_t threads_actual,
    index_t& tensor_offset) const
  {
    detail::ignore(threads_actual);

    ::boba::Array<::boba::Array<index_t, interpolation_points>, dimension> index_bounds;
    for (size_t d = 0; d < dimension; d++)
    {
      for (index_t p = 0; p < interpolation_points; p++)
      {
        index_bounds[d][p] = indices[d][p] - indices_minima[d];
      }
    }

    ::boba::Multiindexer<dimension> multilinear_index(::boba::filled_array<dimension>(interpolation_points));
    auto interpolated_tt = data_t(0.0);
    for (index_t i = 0; i < multilinear_index.size(); i++)
    {
      auto local_mid = multilinear_index.multiindex(i);
      ::boba::Array<index_t, dimension> full_tensor_mid;
      auto weight = data_t(1.0);
      for (size_t d = 0; d < dimension; d++)
      {
        auto highlow = local_mid[d];
        full_tensor_mid[d] = index_bounds[d][highlow];
        weight *= weights[d][highlow];
      }
      // Todo, get rid of multiindexing when possible
      auto id = subtensor_mider.index(full_tensor_mid);
      data_t value = scratch_memory[tensor_offset + id];
      interpolated_tt += weight * value;
    }

    return interpolated_tt;
  }

  /**
   * \brief
   * Interpolate this using the given weights and indices
   */

  template <index_t interpolation_points, index_t batch_size>
  __boba_host_device__
  Array<data_t, batch_size>
  interpolation_batched(
    const Array<Array<Array<data_t, interpolation_points>, dimension>, batch_size>& weights,
    const Array<Array<Array<index_t, interpolation_points>, dimension>, batch_size>& indices) const
  {
    checkpoint();
    // Create bounding box for multiindices
    ::boba::Array<index_t, dimension> indices_minima;
    ::boba::Array<index_t, dimension> indices_maxima;
    for (index_t d = 0; d < dimension; d++)
    {
      indices_minima[d] = indices[0][d][0];
      indices_maxima[d] = indices[0][d][interpolation_points - 1];
      for (index_t b = 1; b < batch_size; b++)
      {
        indices_minima[d] = ::boba::min(indices_minima[d], indices[b][d][0]);
        indices_maxima[d] = ::boba::max(indices_maxima[d], indices[b][d][interpolation_points - 1]);
      }
    }

    // Prepare scratch memory

    static constexpr index_t shared_memory_limit = 7 * 1024;
    static constexpr index_t memory_scratchdata_t = shared_memory_limit / sizeof(data_t) - dimension * 3;

    data_t scratch_memory[memory_scratchdata_t];

    index_t tensor_offset = 0;

    auto subtensor_mider = fetch_subtensor(
      indices_minima,
      indices_maxima,
      scratch_memory,
      1, // serial
      tensor_offset);

    Array<data_t, batch_size> results;
    for (index_t b = 0; b < batch_size; b++)
    {
      results[b] = interpolate_from_subtensor(
        indices[b],
        weights[b],
        indices_minima,
        subtensor_mider,
        scratch_memory,
        1, // serial
        tensor_offset);
    }

    return results;
  }

  /**
   * \brief
   * Interpolate this using the given weights and indices
   */

  template <index_t interpolation_points>
  __boba_host_device__
  data_t
  interpolation_teams(
    const ::boba::Array<::boba::Array<data_t, interpolation_points>, dimension>& weights,
    const ::boba::Array<::boba::Array<index_t, interpolation_points>, dimension>& indices) const
  {
    checkpoint();
    // Create bounding box for multiindices
    ::boba::Array<index_t, dimension> indices_minima;
    ::boba::Array<index_t, dimension> indices_maxima;
    for (size_t d = 0; d < dimension; d++)
    {
      indices_minima[d] = indices[d][0];
      indices_maxima[d] = indices[d][interpolation_points - 1];
    }
    indices_minima = detail::threads_min_reduce(indices_minima);
    indices_maxima = detail::threads_max_reduce(indices_maxima);

    ::boba::detail::thread_synchronize();

    // Prepare scratch memory
    static constexpr index_t shared_memory_limit = 7 * 1024;
    static constexpr index_t memory_scratchdata_t = shared_memory_limit / sizeof(data_t) - dimension * 3;

    __boba_shared__ data_t scratch_memory[memory_scratchdata_t];

    index_t threads_actual = detail::threads_actual();

    ::boba::detail::thread_synchronize();

    index_t tensor_offset = 0;

    auto subtensor_mider = fetch_subtensor(
      indices_minima,
      indices_maxima,
      scratch_memory,
      threads_actual,
      tensor_offset);

    auto interpolated_tt = interpolate_from_subtensor(
      indices,
      weights,
      indices_minima,
      subtensor_mider,
      scratch_memory,
      threads_actual,
      tensor_offset);

    return interpolated_tt;
  }

  template <typename TensorView_like>
  __boba_host_device__ static constexpr index_t get_contract_core_teams_offset(TensorView_like& data_view, index_t rr, index_t index)
  {
    if constexpr (mode_ordering == tt_mode_order::lir)
    {
      return data_view.index({0, index, rr});
    }
    else if constexpr (mode_ordering == tt_mode_order::ilr)
    {
      return data_view.index({index, 0, rr});
    }
    else if constexpr (mode_ordering == tt_mode_order::lri)
    {
      return data_view.index({0, rr, index});
    }
    else
    {
      boba_error("Unsupported mode ordering");
      return index_t(0);
    }
  }

  __boba_host_device__
  index_t
  get_contract_core_teams_stride(size_t core_dim) const
  {
    if constexpr (mode_ordering == tt_mode_order::ilr)
    {
      return sizes(core_dim);
    }
    else
    {
      return 1;
    }
  }

  template <size_t dimension>
  __boba_host_device__ void contract_core_teams(
    const data_t* scratch_in, // todo: make this a view of some kind
    data_t* scratch_out,      // todo: make this a view of some kind
    ::boba::Array<index_t, dimension>& subextents,
    ::boba::Multiindexer<dimension + 1>& subtensor_sizes_helper,
    ::boba::Array<index_t, dimension>& indices_minima,
    index_t threads_actual,
    size_t core_index) const
  {
    checkpoint();
    using const_accessor = ::boba::DefaultAccessor<data_t const>;
    using const_view_type = TensorView<const_accessor, 2>;
    using const_core_view_type = TensorView<const_accessor, 3>;

    // Core properties
    index_t ranks_left = ranks(core_index);
    index_t ranks_right = ranks(core_index + 1);

    auto core_sizes = get_core_sizes(core_index);

    const_core_view_type core_view(core_data[core_index], core_sizes);

    //
    index_t subextent = subextents[core_index];
    index_t index_offset = indices_minima[core_index];
    index_t current_subtensor_size = subtensor_sizes_helper.strides(core_index);

    ::boba::Multiindexer<2> begin_indexer({ranks_left, current_subtensor_size});
    ::boba::Multiindexer<4> end_indexer({ranks_right, current_subtensor_size, subextent, ranks_left});

    const_view_type scratch_in_view(scratch_in, begin_indexer.sizes());

    index_t tid = detail::thread_id();

    size_t id_offset = 0;
    while (id_offset < end_indexer.size())
    {
      scratch_out[id_offset + tid] = 0.0;
      id_offset += threads_actual;
    }

    checkpoint();
    ::boba::detail::thread_synchronize();
    index_t c = 0;
    while (c < end_indexer.size())
    {
      index_t id = c + tid;
      if (id < end_indexer.size())
      {
        auto [rr, u, k, rl] = end_indexer.multiindex(id);
        index_t index = index_offset + k;
        index_t begin_offset = begin_indexer.index({0, u});
        index_t out_offset = end_indexer.index({0, u, k});
        index_t offset = get_contract_core_teams_offset(core_view, rr, index);
        index_t stride = get_contract_core_teams_stride(core_index);
        auto traverse_value = scratch_in_view(begin_offset + rl);
        auto core_value = core_view(offset + stride * rl);
        auto result = traverse_value * core_value;
        atomics::atomic_add(&scratch_out[out_offset + rr], result);
      }
      c += threads_actual;
    }

    detail::thread_synchronize();
    checkpoint();
  }

  template <size_t dimension, size_t scratch_memory_size>
  __boba_host_device__ void contract_core_teams(
    size_t offset_in,  // todo: make this a view of some kind
    size_t offset_out, // todo: make this a view of some kind
    data_t (&scratch_memory)[scratch_memory_size],
    ::boba::Array<index_t, dimension>& subextents,
    ::boba::Multiindexer<dimension + 1>& subtensor_sizes_helper,
    index_t index_offset,
    index_t threads_actual,
    size_t core_index) const
  {
    checkpoint();
    // Core properties
    index_t ranks_left = ranks(core_index);
    index_t ranks_right = ranks(core_index + 1);

    auto core_sizes = get_core_sizes(core_index);
    auto core_data_ptr = core_data[core_index];

    index_t subextent = subextents[core_index];
    index_t current_subtensor_size = subtensor_sizes_helper.strides(core_index);

    ::boba::Multiindexer<2> begin_indexer({ranks_left, current_subtensor_size});
    ::boba::Multiindexer<3> end_indexer({ranks_right, current_subtensor_size, subextent});
    ::boba::Multiindexer<4> contraction_indexer({ranks_right, current_subtensor_size, subextent, ranks_left});

    index_t tid = detail::thread_id();
    auto extent = sizes(core_index);

    size_t id_offset = 0;
    while (id_offset < end_indexer.size())
    {
      if (id_offset + tid < end_indexer.size())
      {
        scratch_memory[offset_out + id_offset + tid] = 0.0;
      }
      id_offset += threads_actual;
    }

    checkpoint();
    detail::thread_synchronize();

    index_t c = 0;

    while (c < contraction_indexer.size())
    {
      index_t id = c + tid;
      if (id < contraction_indexer.size())
      {
        auto [rr, u, k, rl] = contraction_indexer.multiindex(id);

        // Read id from traversal
        index_t begin_id = begin_indexer.index({rl, u});

        auto traverse_value = scratch_memory[offset_in + begin_id];

        // Get value from core
        index_t index = index_offset + k;
        auto core_value = get_core_value(core_index, rl, index, rr);
        auto result = traverse_value * core_value;

        // Write value to output
        index_t out_id = end_indexer.index({rr, u, k});
        atomics::atomic_add(&scratch_memory[offset_out + out_id], result);
      }
      c += threads_actual;
    }
    detail::thread_synchronize();
    checkpoint();
  }

  template <size_t recursion_index, index_t interpolation_points>
    requires(recursion_index == 0)
  __boba_host_device__
  StaticTensor<StaticArray<std::size_t, ranks(recursion_index)>, data_t>
  interpolation_helper(
    const ::boba::Array<::boba::Array<data_t, interpolation_points>, dimension>& weights,
    const ::boba::Array<::boba::Array<index_t, interpolation_points>, dimension>& indices) const
  {
    checkpoint();
    detail::ignore(weights);
    detail::ignore(indices);
    StaticTensor<StaticArray<std::size_t, 1>, data_t> seed{{1.0}};
    return seed;
  }

  template <size_t recursion_index, index_t interpolation_points>
    requires(recursion_index != 0)
  __boba_host_device__
  StaticTensor<StaticArray<std::size_t, ranks(recursion_index)>, data_t>
  interpolation_helper(
    const ::boba::Array<::boba::Array<data_t, interpolation_points>, dimension>& weights,
    const ::boba::Array<::boba::Array<index_t, interpolation_points>, dimension>& indices) const
  {
    checkpoint();
    constexpr size_t core_index = recursion_index - 1;
    constexpr index_t ranks_right = ranks(recursion_index);
    static_assert(core_index >= 0, "Compile time recursion failure.");
    static_assert(core_index < dimension, "Compile time recursion failure.");
    using traversal_t = StaticTensor<StaticArray<std::size_t, ranks(recursion_index - 1)>, data_t>;
    using output_t = StaticTensor<StaticArray<std::size_t, ranks(recursion_index)>, data_t>;
    traversal_t traversal = interpolation_helper<core_index, interpolation_points>(weights, indices);

    output_t output;
    auto output_view = output.view();
    auto core_weights = weights[core_index];
    auto core_indices = indices[core_index];
    for (index_t r = 0; r < ranks_right; r++)
    {
      output_view(r) = 0.0;
    }
    for (index_t p = 0; p < interpolation_points; p++)
    {
      auto traversed_core = contract_core<core_index>(traversal, core_indices[p]);
      auto traversed_core_view = traversed_core.view();
      auto weight = core_weights[p];
      for (index_t r = 0; r < ranks_right; r++)
      {
        output_view(r) += traversed_core_view(r) * weight;
      }
    }
    return output;
  }

  template <typename core_sizes, size_t core_index>
  __boba_host_device__ void do_contraction(
    StaticTensorView<const data_t, StaticArray<std::size_t, ranks(core_index)>> traversal_view,
    StaticTensorView<data_t, StaticArray<std::size_t, ranks(core_index + 1)>> output_view,
    index_t ranks_left,
    index_t index,
    index_t ranks_right) const
  {
    StaticTensorView<data_t, core_sizes> core_view(core_data[core_index]);
    auto stride = get_contract_core_teams_stride(core_index);
    for (index_t rr = 0; rr < ranks_right; rr++)
    {
      auto result = data_t(0.0);
      auto offset = get_contract_core_teams_offset(core_view, rr, index);
      for (index_t rl = 0; rl < ranks_left; rl++)
      {
        auto traverse_value = traversal_view(rl);
        auto core_value = core_view(offset + stride * rl);
        result += traverse_value * core_value;
      }
      output_view(rr) = result;
    }
  }

  /**
   * \brief
   * Given a vector and an extent index, contracts a slice of a core against a vector.
   * x_r = \sum\limits_l v_l C_{lir}
   */

  template <size_t core_index>
  __boba_host_device__
  StaticTensor<StaticArray<std::size_t, ranks(core_index + 1)>, data_t>
  contract_core(
    StaticTensor<StaticArray<std::size_t, ranks(core_index)>, data_t> traversal,
    const index_t& index) const
  {
    checkpoint();
    constexpr index_t size = sizes(core_index);
    constexpr index_t ranks_left = ranks(core_index);
    constexpr index_t ranks_right = ranks(core_index + 1);
    StaticTensor<StaticArray<std::size_t, ranks_right>, data_t> output;

    //
    // Views
    //
    auto traversal_view = traversal.const_view();
    auto output_view = output.view();

    if constexpr (mode_ordering == boba::tt_mode_order::ilr)
    {
      using core_sizes = StaticArray<std::size_t, size, ranks_left, ranks_right>;
      do_contraction<core_sizes, core_index>(traversal_view, output_view, ranks_left, index, ranks_right);
    }
    if constexpr (mode_ordering == boba::tt_mode_order::lir)
    {
      using core_sizes = StaticArray<std::size_t, ranks_left, size, ranks_right>;
      do_contraction<core_sizes, core_index>(traversal_view, output_view, ranks_left, index, ranks_right);
    }
    if constexpr (mode_ordering == boba::tt_mode_order::lri)
    {
      using core_sizes = StaticArray<std::size_t, ranks_left, ranks_right, size>;
      do_contraction<core_sizes, core_index>(traversal_view, output_view, ranks_left, index, ranks_right);
    }

    return output;
  }

  /**
   * \brief
   * Binary search of viewed tt.
   * Given f, Find i and j such that f(x_i) < f < f(x_j)
   */

  __boba_host_device__
  Array<index_t, 2>
  binary_search(const data_t search_value) const
  {
    index_t id_left = 0;
    index_t id_right = size() - 1;
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

template <::boba::execution_space space, typename _data_t>
QuantizedTensorTrain<space, _data_t> operator*(_data_t scalar, QuantizedTensorTrain<space, _data_t> const& input)
{
  BOBA_CALI_MARK
  QuantizedTensorTrain<space, _data_t> temp{input};
  temp *= scalar;
  return temp;
}

} // namespace boba

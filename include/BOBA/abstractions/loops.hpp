// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include "BOBA/boba.hpp"

namespace boba
{

namespace detail
{

// -----------------------------------------------------
// Thread ids
// -----------------------------------------------------
/**
 * \brief
 * Get the device thread id. Assumes only 1d blocks for now.
 * TODO - generalize to 2D and 3D blocks
 * @return Flattened thread index within the current block.
 */

__boba_host_device__ inline size_t thread_id()
{
#if (defined(BOBA_CUDA) or defined(BOBA_HIP)) && defined(BOBA_DEVICE_CODE)
  using block_mider_t = Multiindexer<3>;
  return block_mider_t::index(block_mider_t::precompute_strides({blockDim.x, blockDim.y, blockDim.z}), {threadIdx.x, threadIdx.y, threadIdx.z});
#else
  return 0;
#endif
}

/**
 * \brief
 * Get the actual number of threads launched in this block
 * Assumes, Uses shared memory.
 * @return Number of participating threads in the current block.
 */

__boba_host_device__ inline size_t threads_actual()
{
  size_t threads_actual = 1;
#ifdef BOBA_DEVICE_CODE
  size_t tid = thread_id();
  __boba_shared__ int threads_actual_shared;
  if (tid == 0)
  {
    threads_actual_shared = 0;
  }
  ::boba::detail::thread_synchronize();
  atomics::atomic_max(&threads_actual_shared, static_cast<int>(tid + 1));
  ::boba::detail::thread_synchronize();
  threads_actual = static_cast<size_t>(threads_actual_shared);
#endif
  return threads_actual;
}

/**
 * \brief
 * Minimizes the entries of an array over all current threads in a block
 * @tparam T Array value type.
 * @tparam N Array length.
 * @param input Per-thread input values.
 * @param this_thread_participates Whether this thread contributes.
 * @return Elementwise minimum across participating threads.
 */

template <typename T, size_t N>
__boba_device__
inline Array<T, N> threads_min_reduce(
  const Array<T, N>& input,
  bool this_thread_participates = true)
{
  Array<T, N> output;
#ifdef BOBA_DEVICE_CODE
  size_t tid = thread_id();
  __boba_shared__ int shared_output[N];
  detail::thread_synchronize();
  if (tid == 0)
  {
    for (size_t d = 0; d < N; d++)
    {
      shared_output[d] = boba::highest_value<int>();
    }
  }
  ::boba::detail::thread_synchronize();
  if (this_thread_participates)
  {
    for (size_t d = 0; d < N; d++)
    {
      atomics::atomic_min(&shared_output[d], static_cast<int>(input[d]));
    }
  }
  ::boba::detail::thread_synchronize();
  for (size_t d = 0; d < N; d++)
  {
    output[d] = static_cast<T>(shared_output[d]);
  }
#else
  detail::ignore(input);
  detail::ignore(this_thread_participates);
#endif
  return output;
}

/**
 * \brief
 * Maximizes the entries of an array over all current threads in a block
 * @tparam T Array value type.
 * @tparam N Array length.
 * @param input Per-thread input values.
 * @param this_thread_participates Whether this thread contributes.
 * @return Elementwise maximum across participating threads.
 */

template <typename T, size_t N>
__boba_device__
inline Array<T, N> threads_max_reduce(
  const Array<T, N>& input,
  bool this_thread_participates = true)
{
  Array<T, N> output;
#ifdef BOBA_DEVICE_CODE
  size_t tid = thread_id();
  __boba_shared__ int shared_output[N];
  detail::thread_synchronize();
  if (tid == 0)
  {
    for (size_t d = 0; d < N; d++)
    {
      shared_output[d] = boba::lowest_value<int>();
    }
  }
  ::boba::detail::thread_synchronize();
  if (this_thread_participates)
  {
    for (size_t d = 0; d < N; d++)
    {
      atomics::atomic_max(&shared_output[d], static_cast<int>(input[d]));
    }
  }
  ::boba::detail::thread_synchronize();
  for (size_t d = 0; d < N; d++)
  {
    output[d] = static_cast<T>(shared_output[d]);
  }
#else
  detail::ignore(input);
  detail::ignore(this_thread_participates);
#endif
  return output;
}

/**
 * @brief Computes elementwise block-local minima and maxima.
 * @tparam T Array value type.
 * @tparam N Array length.
 * @param input Per-thread input values.
 * @param out_min Output block minimum.
 * @param out_max Output block maximum.
 * @param this_thread_participates Whether this thread contributes.
 */
template <typename T, size_t N>
__boba_device__
inline void threads_min_max_reduce(
  const Array<T, N>& input,
  Array<T, N>& out_min,
  Array<T, N>& out_max,
  bool this_thread_participates = true)
{
#ifdef BOBA_DEVICE_CODE
  size_t tid = thread_id();
  __boba_shared__ int shared_output_max[N];
  __boba_shared__ int shared_output_min[N];
  detail::thread_synchronize();
  if (tid == 0)
  {
    for (size_t d = 0; d < N; d++)
    {
      shared_output_max[d] = boba::lowest_value<int>();
      shared_output_min[d] = boba::highest_value<int>();
    }
  }
  ::boba::detail::thread_synchronize();
  if (this_thread_participates)
  {
    for (size_t d = 0; d < N; d++)
    {
      atomics::atomic_min(&shared_output_min[d], static_cast<int>(input[d]));
      atomics::atomic_max(&shared_output_max[d], static_cast<int>(input[d]));
    }
  }
  ::boba::detail::thread_synchronize();
  for (size_t d = 0; d < N; d++)
  {
    out_min[d] = static_cast<T>(shared_output_min[d]);
    out_max[d] = static_cast<T>(shared_output_max[d]);
  }
#else
  detail::ignore(input);
  detail::ignore(out_min);
  detail::ignore(out_max);
  detail::ignore(this_thread_participates);
#endif
}

// ---------------------------------------------------------------------------
// Loop Abstractions
// ---------------------------------------------------------------------------

/**
 * \brief
 * Backend-specific 1D loop dispatch.
 * @tparam space Execution space.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param begin Inclusive lower bound.
 * @param end Exclusive upper bound.
 * @param lambda Loop body.
 */

template <boba::execution_space space, typename Lambda>
inline void loop(index_t begin, index_t end, Lambda&& lambda)
{
  if (end <= begin)
  {
    return;
  }
  if constexpr (space == boba::execution_space::CPU)
  {
    boba::detail::seq_loop(begin, end, std::forward<Lambda>(lambda));
  }
#if defined(BOBA_CUDA)
  else if constexpr (space == boba::execution_space::CUDA)
  {
    boba::detail::cuda_launch(begin, end, std::forward<Lambda>(lambda));
  }
#endif
#if defined(BOBA_HIP)
  else if constexpr (space == boba::execution_space::HIP)
  {
    boba::detail::hip_launch(begin, end, std::forward<Lambda>(lambda));
  }
#endif
  else
  {
    boba_error("Invalid space");
  }
}

/**
 * \brief
 * Backend-specific 2D loop dispatch.
 * @tparam space Execution space.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param begin0 Inclusive lower bound for dimension 0.
 * @param end0 Exclusive upper bound for dimension 0.
 * @param begin1 Inclusive lower bound for dimension 1.
 * @param end1 Exclusive upper bound for dimension 1.
 * @param lambda Loop body.
 */

template <boba::execution_space space, typename Lambda>
inline void loop_2d(index_t begin0, index_t end0, index_t begin1, index_t end1, Lambda&& lambda)
{
  if ((end0 <= begin0) || (end1 <= begin1))
  {
    return;
  }
  if constexpr (space == boba::execution_space::CPU)
  {
    boba::detail::seq_loop_2d(begin0, end0, begin1, end1, std::forward<Lambda>(lambda));
  }
#if defined(BOBA_CUDA)
  else if constexpr (space == boba::execution_space::CUDA)
  {
    boba::detail::cuda_launch_2d(begin0, end0, begin1, end1, std::forward<Lambda>(lambda));
  }
#endif
#if defined(BOBA_HIP)
  else if constexpr (space == boba::execution_space::HIP)
  {
    boba::detail::hip_launch_2d(begin0, end0, begin1, end1, std::forward<Lambda>(lambda));
  }
#endif
  else
  {
    boba_error("Invalid space");
  }
}

/**
 * \brief
 * Backend-specific 3D loop dispatch.
 * @tparam space Execution space.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param begin0 Inclusive lower bound for dimension 0.
 * @param end0 Exclusive upper bound for dimension 0.
 * @param begin1 Inclusive lower bound for dimension 1.
 * @param end1 Exclusive upper bound for dimension 1.
 * @param begin2 Inclusive lower bound for dimension 2.
 * @param end2 Exclusive upper bound for dimension 2.
 * @param lambda Loop body.
 */

template <boba::execution_space space, typename Lambda>
inline void loop_3d(index_t begin0, index_t end0, index_t begin1, index_t end1, index_t begin2, index_t end2, Lambda&& lambda)
{
  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2))
  {
    return;
  }
  if constexpr (space == boba::execution_space::CPU)
  {
    boba::detail::seq_loop_3d(begin0, end0, begin1, end1, begin2, end2, std::forward<Lambda>(lambda));
  }
#if defined(BOBA_CUDA)
  else if constexpr (space == boba::execution_space::CUDA)
  {
    boba::detail::cuda_launch_3d(begin0, end0, begin1, end1, begin2, end2, std::forward<Lambda>(lambda));
  }
#endif
#if defined(BOBA_HIP)
  else if constexpr (space == boba::execution_space::HIP)
  {
    boba::detail::hip_launch_3d(begin0, end0, begin1, end1, begin2, end2, std::forward<Lambda>(lambda));
  }
#endif
  else
  {
    boba_error("Invalid space");
  }
}

template <size_t N, typename Lambda>
struct LoopAdapter;

template <typename Lambda>
struct LoopAdapter<0, Lambda>
{
  /**
   * @brief Executes a degenerate zero-dimensional loop.
   * @tparam space Execution space.
   * @tparam L Callable type.
   * @param begin Ignored begin bound.
   * @param end Ignored end bound.
   * @param lambda Ignored callable.
   */
  template <boba::execution_space space, typename L>
  static void loop(boba::Array<index_t, 0> const&,
                   boba::Array<index_t, 0> const&,
                   L&&)
  {
  }
};

template <typename Lambda>
struct LoopAdapter<1, Lambda>
{
  Lambda m_lambda;

  /**
   * @brief Adapts a scalar loop index to a 1D array index.
   * @param i Loop index.
   */
  __boba_host_device__ void operator()(index_t i) const
  {
    m_lambda(boba::Array<index_t, 1>{i});
  }

  /**
   * @brief Executes a 1D array-index loop.
   * @tparam space Execution space.
   * @tparam L Callable type.
   * @param begin Inclusive lower bound.
   * @param end Exclusive upper bound.
   * @param lambda Loop body.
   */
  template <boba::execution_space space, typename L>
  static void loop(boba::Array<index_t, 1> const& begin,
                   boba::Array<index_t, 1> const& end,
                   L&& lambda)
  {
    boba::detail::loop<space>(
      begin[0], end[0], LoopAdapter{std::forward<L>(lambda)});
  }
};

template <typename Lambda>
struct LoopAdapter1d
{
  Lambda m_lambda;

  /**
   * @brief Forwards a scalar loop index.
   * @param i Loop index.
   */
  __boba_host_device__ void operator()(index_t i) const
  {
    m_lambda(i);
  }

  /**
   * @brief Executes a scalar 1D loop.
   * @tparam space Execution space.
   * @tparam L Callable type.
   * @param begin Inclusive lower bound.
   * @param end Exclusive upper bound.
   * @param lambda Loop body.
   */
  template <boba::execution_space space, typename L>
  static void loop(index_t const& begin,
                   index_t const& end,
                   L&& lambda)
  {
    boba::detail::loop<space>(
      begin, end, std::forward<L>(lambda));
  }
};

template <typename Lambda>
struct LoopAdapter<2, Lambda>
{
  Lambda m_lambda;

  /**
   * @brief Adapts scalar loop indices to a 2D array index.
   * @param i Dimension-0 index.
   * @param j Dimension-1 index.
   */
  __boba_host_device__ void operator()(index_t i, index_t j) const
  {
    m_lambda(boba::Array<index_t, 2>{i, j});
  }

  /**
   * @brief Executes a 2D array-index loop.
   * @tparam space Execution space.
   * @tparam L Callable type.
   * @param begin Inclusive lower bounds.
   * @param end Exclusive upper bounds.
   * @param lambda Loop body.
   */
  template <boba::execution_space space, typename L>
  static void loop(boba::Array<index_t, 2> const& begin,
                   boba::Array<index_t, 2> const& end,
                   L&& lambda)
  {
    boba::detail::loop_2d<space>(
      begin[0], end[0], begin[1], end[1], LoopAdapter{std::forward<L>(lambda)});
  }
};

template <typename Lambda>
struct LoopAdapter<3, Lambda>
{
  Lambda m_lambda;

  /**
   * @brief Adapts scalar loop indices to a 3D array index.
   * @param i Dimension-0 index.
   * @param j Dimension-1 index.
   * @param k Dimension-2 index.
   */
  __boba_host_device__ void operator()(index_t i, index_t j, index_t k) const
  {
    m_lambda(boba::Array<index_t, 3>{i, j, k});
  }

  /**
   * @brief Executes a 3D array-index loop.
   * @tparam space Execution space.
   * @tparam L Callable type.
   * @param begin Inclusive lower bounds.
   * @param end Exclusive upper bounds.
   * @param lambda Loop body.
   */
  template <boba::execution_space space, typename L>
  static void loop(boba::Array<index_t, 3> const& begin,
                   boba::Array<index_t, 3> const& end,
                   L&& lambda)
  {
    boba::detail::loop_3d<space>(
      begin[0], end[0], begin[1], end[1], begin[2], end[2], LoopAdapter{std::forward<L>(lambda)});
  }
};

/**
 * @brief Dispatches an N-dimensional loop through the matching adapter.
 * @tparam space Execution space.
 * @tparam N Number of dimensions.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param begin Inclusive lower bounds.
 * @param end Exclusive upper bounds.
 * @param lambda Loop body.
 */
template <boba::execution_space space, size_t N, typename Lambda>
inline void loop_Nd(boba::Array<index_t, N> const& begin,
                    boba::Array<index_t, N> const& end,
                    Lambda&& lambda)
{
  static_assert(N <= 3, "Unsupported N");

  using adapter_type = LoopAdapter<N, std::decay_t<Lambda>>;

  adapter_type::template loop<space>(begin, end, std::forward<Lambda>(lambda));
}

/**
 * @brief Dispatches a scalar 1D loop through the 1D adapter.
 * @tparam space Execution space.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param begin Inclusive lower bound.
 * @param end Exclusive upper bound.
 * @param lambda Loop body.
 */
template <boba::execution_space space, typename Lambda>
inline void loop_1d(index_t const& begin,
                    index_t const& end,
                    Lambda&& lambda)
{
  using adapter_type = LoopAdapter1d<std::decay_t<Lambda>>;

  adapter_type::template loop<space>(begin, end, std::forward<Lambda>(lambda));
}

/**
 * @brief Synchronizes work in the selected execution space.
 * @tparam space Execution space to synchronize.
 */
template <boba::execution_space space>
inline void synchronize()
{
  if constexpr (space == boba::execution_space::CPU)
  {
    // do nothing
  }
#if defined(BOBA_CUDA)
  else if constexpr (space == boba::execution_space::CUDA)
  {
    boba::detail::cuda_syncronize();
  }
#endif
#if defined(BOBA_HIP)
  else if constexpr (space == boba::execution_space::HIP)
  {
    boba::detail::hip_syncronize();
  }
#endif
  else
  {
    boba_error("Invalid space");
  }
}

} // namespace detail

/**
 * \brief
 * Infer dimension from size of arrays
 * @tparam space Execution space.
 * @tparam dimension Number of dimensions.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param begin Inclusive lower bounds.
 * @param end Exclusive upper bounds.
 * @param lambda Loop body.
 */

template <boba::execution_space space, size_t dimension, typename Lambda>
inline void loop(::boba::Array<index_t, dimension> begin,
                 ::boba::Array<index_t, dimension> end,
                 Lambda&& lambda)
{
  ::boba::detail::loop_Nd<space>(
    begin, end, std::forward<Lambda>(lambda));
}

/**
 * \brief
 * Infer dimension from size of arrays, assume begin = 0, 0, ....
 * @tparam space Execution space.
 * @tparam dimension Number of dimensions.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param end Exclusive upper bounds.
 * @param lambda Loop body.
 */

template <boba::execution_space space, size_t dimension, typename Lambda>
inline void loop(::boba::Array<index_t, dimension> end,
                 Lambda&& lambda)
{
  ::boba::detail::loop_Nd<space, dimension>(
    filled_array<dimension>(0_z), end, std::forward<Lambda>(lambda));
}

/**
 * \brief
 * 1D loops can optionally take in integers instead of arrays.
 * @tparam space Execution space.
 * @tparam dimension Expected dimension, must be `1`.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param begin Inclusive lower bound.
 * @param end Exclusive upper bound.
 * @param lambda Loop body.
 */

template <boba::execution_space space, size_t dimension, typename Lambda>
inline void loop(index_t begin, index_t end, Lambda&& lambda)
{
  static_assert(dimension == 1, "begin and end are scalars so this should be a 1d loop.");
  ::boba::detail::loop_1d<space>(begin, end, std::forward<Lambda>(lambda));
}

/**
 * \brief
 * 1D loops can optionally take in integers instead of arrays. Assume begin = 0.
 * @tparam space Execution space.
 * @tparam dimension Expected dimension, must be `1`.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param end Exclusive upper bound.
 * @param lambda Loop body.
 */

template <boba::execution_space space, size_t dimension, typename Lambda>
inline void loop(index_t end,
                 Lambda&& lambda)
{
  static_assert(dimension == 1, "end is a scalar so this should be a 1d loop.");
  ::boba::detail::loop_1d<space>(0, end, std::forward<Lambda>(lambda));
}

} // namespace boba

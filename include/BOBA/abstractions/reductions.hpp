// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

using reducer_index_t = std::conditional_t<boba_raja_enabled(), std::ptrdiff_t, index_t>;

namespace detail
{

enum struct reducer_t : std::size_t
{
  sum,
  max,
  min,
  max_loc,
  min_loc
};

template <reducer_t reducer_type>
concept IsValueReducer = (reducer_type == reducer_t::sum) or
                         (reducer_type == reducer_t::max) or
                         (reducer_type == reducer_t::min);

template <reducer_t reducer_type>
concept IsLocReducer = (reducer_type == reducer_t::max_loc) or
                       (reducer_type == reducer_t::min_loc);

template <typename data_t, reducer_t reducer_type>
struct ReducerValue
{
  /**
   * @brief Stores an initial value for value-only reducers.
   * @param val Initial reduction value.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires IsValueReducer<_reducer_type>
  __boba_host_device__ ReducerValue(data_t val)
      : m_value(val)
  {
  }

  /**
   * @brief Stores an initial value and location for location reducers.
   * @param val Initial reduction value.
   * @param loc Initial reduction location.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires IsLocReducer<_reducer_type>
  __boba_host_device__ ReducerValue(data_t val, index_t loc)
      : m_value(val),
        m_loc(loc)
  {
  }

  /**
   * @brief Returns the stored reduction value.
   * @return Stored value reference.
   */
  __boba_host_device__ data_t& getVal()
  {
    return m_value;
  }

  /**
   * @brief Returns the stored reduction location.
   * @return Stored location reference.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires IsLocReducer<_reducer_type>
  __boba_host_device__ index_t& getLoc()
  {
    return m_loc;
  }

  /**
   * @brief Returns the stored value/location pair.
   * @return Copy of the stored value and location.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires IsLocReducer<_reducer_type>
  __boba_host_device__ std::pair<data_t, index_t> getPair() const
  {
    return std::make_pair(data_t(m_value), index_t(m_loc));
  }

private:
  data_t m_value;
  index_t m_loc;
};

template <typename data_t, reducer_t reducer_type>
struct ReducerOperator
{
  /**
   * @brief Binds a value-only reducer wrapper.
   * @param value_ptr Reduction storage.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires IsValueReducer<_reducer_type>
  __boba_host_device__ ReducerOperator(ReducerValue<data_t, reducer_type>* value_ptr)
      : m_value_ptr(&value_ptr->getVal())
  {
  }

  /**
   * @brief Binds a value/location reducer wrapper.
   * @param value_ptr Reduction storage.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires IsLocReducer<_reducer_type>
  __boba_host_device__ ReducerOperator(ReducerValue<data_t, reducer_type>* value_ptr)
      : m_value_ptr(&value_ptr->getVal()),
        m_loc_ptr(&value_ptr->getLoc())
  {
  }

  /**
   * @brief Binds value-only reduction storage.
   * @param value_ptr Value storage.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires IsValueReducer<_reducer_type>
  __boba_host_device__ ReducerOperator(data_t* value_ptr)
      : m_value_ptr(value_ptr)
  {
  }

  /**
   * @brief Binds value/location reduction storage.
   * @param value_ptr Value storage.
   * @param loc_ptr Location storage.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires IsLocReducer<_reducer_type>
  __boba_host_device__ ReducerOperator(data_t* value_ptr, index_t* loc_ptr)
      : m_value_ptr(value_ptr),
        m_loc_ptr(loc_ptr)
  {
  }

  /**
   * @brief Applies a minimum update.
   * @param val Candidate value.
   * @return `*this`.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires(_reducer_type == reducer_t::min)
  __boba_host_device__ ReducerOperator& min(data_t val)
  {
    if (val < *m_value_ptr)
    {
      *m_value_ptr = val;
    }
    return *this;
  }

  /**
   * @brief Applies a maximum update.
   * @param val Candidate value.
   * @return `*this`.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires(_reducer_type == reducer_t::max)
  __boba_host_device__ ReducerOperator& max(data_t val)
  {
    if (*m_value_ptr < val)
    {
      *m_value_ptr = val;
    }
    return *this;
  }

  /**
   * @brief Applies a minimum-with-location update.
   * @param val Candidate value.
   * @param loc Candidate location.
   * @return `*this`.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires(_reducer_type == reducer_t::min_loc)
  __boba_host_device__ ReducerOperator& minloc(data_t val, index_t loc)
  {
    if (val < *m_value_ptr)
    {
      *m_value_ptr = val;
      *m_loc_ptr = loc;
    }
    return *this;
  }

  /**
   * @brief Applies a maximum-with-location update.
   * @param val Candidate value.
   * @param loc Candidate location.
   * @return `*this`.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires(_reducer_type == reducer_t::max_loc)
  __boba_host_device__ ReducerOperator& maxloc(data_t val, index_t loc)
  {
    if (*m_value_ptr < val)
    {
      *m_value_ptr = val;
      *m_loc_ptr = loc;
    }
    return *this;
  }

  /**
   * @brief Adds into a sum reduction.
   * @param value Value to add.
   * @return `*this`.
   */
  template <reducer_t _reducer_type = reducer_type>
    requires(_reducer_type == reducer_t::sum)
  __boba_host_device__ ReducerOperator& operator+=(data_t value)
  {
    *m_value_ptr += value;
    return *this;
  }

private:
  data_t* m_value_ptr;
  index_t* m_loc_ptr;
};

} // namespace detail

} // namespace boba

#include "BOBA/backends/boba_raja.hpp"
#include "BOBA/backends/boba_kokkos.hpp"

namespace boba
{

namespace detail
{

} // namespace detail

#if defined(BOBA_RAJA)
//
// sum
//
template <typename data_t>
using sum_reducer_value = data_t;
template <typename data_t>
using sum_reducer_operator = RAJA::expt::ValOp<data_t, RAJA::operators::plus>;
//
// min
//
template <typename data_t>
using min_reducer_value = data_t;
template <typename data_t>
using min_reducer_operator = RAJA::expt::ValOp<data_t, RAJA::operators::minimum>;
//
// max
//
template <typename data_t>
using max_reducer_value = data_t;
template <typename data_t>
using max_reducer_operator = RAJA::expt::ValOp<data_t, RAJA::operators::maximum>;
//
// min loc
//
template <typename data_t>
using min_loc_reducer_value = RAJA::expt::ValLocOp<data_t, reducer_index_t, RAJA::operators::minimum>;
template <typename data_t>
using min_loc_reducer_operator = RAJA::expt::ValLocOp<data_t, reducer_index_t, RAJA::operators::minimum>;
//
// max loc
//
template <typename data_t>
using max_loc_reducer_value = RAJA::expt::ValLocOp<data_t, reducer_index_t, RAJA::operators::maximum>;
template <typename data_t>
using max_loc_reducer_operator = RAJA::expt::ValLocOp<data_t, reducer_index_t, RAJA::operators::maximum>;
#elif defined(BOBA_KOKKOS)
//
// sum
//
template <typename data_t>
using sum_reducer_value = data_t;
template <typename data_t>
using sum_reducer_operator = detail::ReducerOperator<data_t, detail::reducer_t::sum>;
//
// min
//
template <typename data_t>
using min_reducer_value = data_t;
template <typename data_t>
using min_reducer_operator = detail::ReducerOperator<data_t, detail::reducer_t::min>;
//
// max
//
template <typename data_t>
using max_reducer_value = data_t;
template <typename data_t>
using max_reducer_operator = detail::ReducerOperator<data_t, detail::reducer_t::max>;
//
// min loc
//
template <typename data_t>
using min_loc_reducer_value = detail::ReducerValue<data_t, detail::reducer_t::min_loc>;
template <typename data_t>
using min_loc_reducer_operator = detail::ReducerOperator<data_t, detail::reducer_t::min_loc>;
//
// max loc
//
template <typename data_t>
using max_loc_reducer_value = detail::ReducerValue<data_t, detail::reducer_t::max_loc>;
template <typename data_t>
using max_loc_reducer_operator = detail::ReducerOperator<data_t, detail::reducer_t::max_loc>;
#else
//
// sum
//
template <typename data_t>
using sum_reducer_value = detail::ReducerValue<data_t, detail::reducer_t::sum>;
template <typename data_t>
using sum_reducer_operator = detail::ReducerOperator<data_t, detail::reducer_t::sum>;
//
// min
//
template <typename data_t>
using min_reducer_value = detail::ReducerValue<data_t, detail::reducer_t::min>;
template <typename data_t>
using min_reducer_operator = detail::ReducerOperator<data_t, detail::reducer_t::min>;
//
// max
//
template <typename data_t>
using max_reducer_value = detail::ReducerValue<data_t, detail::reducer_t::max>;
template <typename data_t>
using max_reducer_operator = detail::ReducerOperator<data_t, detail::reducer_t::max>;
//
// min loc
//
template <typename data_t>
using min_loc_reducer_value = detail::ReducerValue<data_t, detail::reducer_t::min_loc>;
template <typename data_t>
using min_loc_reducer_operator = detail::ReducerOperator<data_t, detail::reducer_t::min_loc>;
//
// max loc
//
template <typename data_t>
using max_loc_reducer_value = detail::ReducerValue<data_t, detail::reducer_t::max_loc>;
template <typename data_t>
using max_loc_reducer_operator = detail::ReducerOperator<data_t, detail::reducer_t::max_loc>;
#endif

// ---------------------------------------------------------------------------
// Reduction Loop
// ---------------------------------------------------------------------------

namespace detail
{

/**
 * \brief
 * Loop with reducer.
 * @tparam space Execution space.
 * @tparam index_t Index type.
 * @tparam Reducer Reducer type.
 * @tparam Lambda Loop body type.
 * @param begin Inclusive lower bound.
 * @param end Exclusive upper bound.
 * @param reducer Reducer object.
 * @param lambda Loop body.
 */

template <boba::execution_space space, typename Reducer, typename Lambda>
inline void loop(
  index_t begin,
  index_t end,
  Reducer&& reducer,
  Lambda&& lambda)
{
  if (end <= begin)
  {
    return;
  }

  if constexpr (boba_raja_enabled())
  {
    backend::raja::loop<space>(begin, end, std::forward<Reducer>(reducer), std::forward<Lambda>(lambda));
  }
  else if constexpr (boba_kokkos_enabled())
  {
    backend::kokkos::loop<space>(begin, end, std::forward<Reducer>(reducer), std::forward<Lambda>(lambda));
  }
  else
  {
    for (index_t i = begin; i < end; ++i)
    {
      lambda(i, reducer);
    }
  }
}

} // namespace detail

// ---------------------------------------------------------------------------
// Loop wrapper
// ---------------------------------------------------------------------------

/**
 * \brief
 * Loop with reducer.
 * @tparam space Execution space.
 * @tparam index_t Index type.
 * @tparam Reducer Reducer type.
 * @tparam Lambda Loop body type.
 * @param begin Inclusive lower bound.
 * @param end Exclusive upper bound.
 * @param reducer Reducer object.
 * @param lambda Loop body.
 */
template <boba::execution_space space, typename Reducer, typename Lambda>
inline void loop(
  index_t begin,
  index_t end,
  Reducer&& reducer,
  Lambda&& lambda)
{
  detail::loop<space>(begin, end, std::forward<Reducer>(reducer), std::forward<Lambda>(lambda));
}

/**
 * @brief Performs a sum reduction over a 1D index range.
 * @tparam space Execution space.
 * @tparam data_t Reduced value type.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param initial_value Input/output reduction value.
 * @param begin Inclusive lower bound.
 * @param end Exclusive upper bound.
 * @param lambda Loop body.
 */
template <boba::execution_space space, typename data_t, typename Lambda>
inline void sum_reduce(
  data_t& initial_value,
  index_t begin,
  index_t end,
  Lambda&& lambda)
{
  if constexpr (boba_raja_enabled())
  {
    backend::raja::sum_reduce<space>(initial_value, begin, end, std::forward<Lambda>(lambda));
  }
  else if constexpr (boba_kokkos_enabled())
  {
    backend::kokkos::sum_reduce<space>(initial_value, begin, end, std::forward<Lambda>(lambda));
  }
  else
  {
    detail::loop<space>(begin, end, detail::ReducerOperator<data_t, detail::reducer_t::sum>(&initial_value), std::forward<Lambda>(lambda));
  }
}

/**
 * @brief Sum-reductions using manual backends for verification purposes
 * @return Summation of all values of tensorlike
 */

template <typename tensorlike_t>
auto manual_sum_reduce(const tensorlike_t& tensorlike)
{
  constexpr execution_space space = tensorlike_t::get_space();

  if constexpr(boba_cuda_enabled() and space == execution_space::CUDA)
  {
    return detail::sum_reduction(tensorlike.data(), tensorlike.size());
  }
  else
  {
    using data_t = typename tensorlike_t::data_t;
    data_t sum = 0;
    const auto* tensorlike_data = tensorlike.data();
    for(index_t i = 0; i < tensorlike.size(); i++)
    {
      sum += tensorlike_data[i];
    }
    return sum;
  }
}

/**
 * @brief Performs a max reduction over a 1D index range.
 * @tparam space Execution space.
 * @tparam data_t Reduced value type.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param initial_value Input/output reduction value.
 * @param begin Inclusive lower bound.
 * @param end Exclusive upper bound.
 * @param lambda Loop body.
 */
template <boba::execution_space space, typename data_t, typename Lambda>
inline void max_reduce(
  data_t& initial_value,
  index_t begin,
  index_t end,
  Lambda&& lambda)
{
  if constexpr (boba_raja_enabled())
  {
    backend::raja::max_reduce<space>(initial_value, begin, end, std::forward<Lambda>(lambda));
  }
  else if constexpr (boba_kokkos_enabled())
  {
    backend::kokkos::max_reduce<space>(initial_value, begin, end, std::forward<Lambda>(lambda));
  }
  else
  {
    detail::loop<space>(begin, end, detail::ReducerOperator<data_t, detail::reducer_t::max>(&initial_value), std::forward<Lambda>(lambda));
  }
}

/**
 * @brief Performs a min reduction over a 1D index range.
 * @tparam space Execution space.
 * @tparam data_t Reduced value type.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param initial_value Input/output reduction value.
 * @param begin Inclusive lower bound.
 * @param end Exclusive upper bound.
 * @param lambda Loop body.
 */
template <boba::execution_space space, typename data_t, typename Lambda>
inline void min_reduce(
  data_t& initial_value,
  index_t begin,
  index_t end,
  Lambda&& lambda)
{
  if constexpr (boba_raja_enabled())
  {
    backend::raja::min_reduce<space>(initial_value, begin, end, std::forward<Lambda>(lambda));
  }
  else if constexpr (boba_kokkos_enabled())
  {
    backend::kokkos::min_reduce<space>(initial_value, begin, end, std::forward<Lambda>(lambda));
  }
  else
  {
    detail::loop<space>(begin, end, detail::ReducerOperator<data_t, detail::reducer_t::min>(&initial_value), std::forward<Lambda>(lambda));
  }
}

/**
 * @brief Performs a min-location reduction over a 1D index range.
 * @tparam space Execution space.
 * @tparam data_t Reduced value type.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param initial_value Input/output reduction value.
 * @param initial_loc Input/output reduction location.
 * @param begin Inclusive lower bound.
 * @param end Exclusive upper bound.
 * @param lambda Loop body.
 */
template <boba::execution_space space, typename data_t, typename Lambda>
inline void min_loc_reduce(
  data_t& initial_value,
  index_t& initial_loc,
  index_t begin,
  index_t end,
  Lambda&& lambda)
{
  if constexpr (boba_raja_enabled())
  {
    backend::raja::min_loc_reduce<space>(initial_value, initial_loc, begin, end, std::forward<Lambda>(lambda));
  }
  else if constexpr (boba_kokkos_enabled())
  {
    backend::kokkos::min_loc_reduce<space>(initial_value, initial_loc, begin, end, std::forward<Lambda>(lambda));
  }
  else
  {
    detail::loop<space>(begin, end, detail::ReducerOperator<data_t, detail::reducer_t::min_loc>(&initial_value, &initial_loc), std::forward<Lambda>(lambda));
  }
}

/**
 * @brief Performs a max-location reduction over a 1D index range.
 * @tparam space Execution space.
 * @tparam data_t Reduced value type.
 * @tparam index_t Index type.
 * @tparam Lambda Loop body type.
 * @param initial_value Input/output reduction value.
 * @param initial_loc Input/output reduction location.
 * @param begin Inclusive lower bound.
 * @param end Exclusive upper bound.
 * @param lambda Loop body.
 */
template <boba::execution_space space, typename data_t, typename Lambda>
inline void max_loc_reduce(
  data_t& initial_value,
  index_t& initial_loc,
  index_t begin,
  index_t end,
  Lambda&& lambda)
{
  if constexpr (boba_raja_enabled())
  {
    backend::raja::max_loc_reduce<space>(initial_value, initial_loc, begin, end, std::forward<Lambda>(lambda));
  }
  else if constexpr (boba_kokkos_enabled())
  {
    backend::kokkos::max_loc_reduce<space>(initial_value, initial_loc, begin, end, std::forward<Lambda>(lambda));
  }
  else
  {
    detail::loop<space>(begin, end, detail::ReducerOperator<data_t, detail::reducer_t::max_loc>(&initial_value, &initial_loc), std::forward<Lambda>(lambda));
  }
}

namespace reductions
{
/**
 * Sum all elements
 * @tparam space Execution space.
 * @tparam data_t Value type.
 * @tparam index_t Index type.
 * @param data Input data.
 * @param size Number of elements.
 * @return Sum of all entries.
 */

template <execution_space space, typename data_t>
data_t sum_reduce(const data_t* data, index_t size)
{
  data_t value{};

  ::boba::sum_reduce<space>(value, index_t(0), size, [=] __boba_host_device__(index_t i, sum_reducer_operator<data_t> & local_value)
  {
    local_value += data[i];
  });

  return value;
}

/**
 * Calculates the maximum element
 * @tparam space Execution space.
 * @tparam data_t Value type.
 * @tparam index_t Index type.
 * @param data Input data.
 * @param size Number of elements.
 * @return Maximum entry.
 */

template <execution_space space, typename data_t>
data_t max_reduce(const data_t* data, index_t size)
{
  static_assert(std::is_same_v<data_t, real_type_t<data_t>>, "max_reduce is disabled for complex data type");

  data_t value = ::boba::lowest_value<data_t>();

  ::boba::max_reduce<space>(value, index_t(0), size, [=] __boba_host_device__(index_t i, max_reducer_operator<data_t> & local_value)
  {
    local_value.max(data[i]);
  });

  if (size > 0 and not(std::is_same<data_t, size_t>::value))
  {
    boba_assert_gt(value, ::boba::lowest_value<data_t>(), "Reduction failure");
  }

  return value;
}

/**
 * Calculates the maximum magnitude element
 * @tparam space Execution space.
 * @tparam data_t Value type.
 * @tparam index_t Index type.
 * @param data Input data.
 * @param size Number of elements.
 * @return Maximum absolute value.
 */

template <execution_space space, typename data_t>
real_type_t<data_t> max_abs_reduce(const data_t* data, index_t size)
{
  using real_data_t = real_type_t<data_t>;

  auto value = ::boba::lowest_value<real_data_t>();
  ::boba::max_reduce<space>(value, index_t(0), size, [=] __boba_host_device__(index_t i, max_reducer_operator<real_data_t> & local_max)
  {
    real_data_t local_value = ::boba::abs(data[i]);
    local_max.max(local_value);
  });

  if (size > 0 and not(std::is_same<data_t, size_t>::value))
  {
    boba_assert_gt(value, ::boba::lowest_value<real_data_t>(), "Reduction failure");
  }

  return value;
}

/**
 * Calculates the maximum element and the corresponding long index.
 * @tparam space Execution space.
 * @tparam data_t Value type.
 * @tparam index_t Index type.
 * @param data Input data.
 * @param size Number of elements.
 * @return Pair of maximum value and its index.
 */

template <execution_space space, typename data_t>
std::pair<data_t, index_t> max_loc_reduce(const data_t* data, index_t size)
{
  static_assert(std::is_same_v<data_t, real_type_t<data_t>>, "max_loc_reduce is disabled for complex data type");

  auto maximum = ::boba::lowest_value<data_t>();
  auto maximum_index = index_t(0);

  ::boba::max_loc_reduce<space>(maximum, maximum_index, index_t(0), size, [=] __boba_host_device__(::boba::reducer_index_t i, ::boba::max_loc_reducer_operator<data_t> & local_maxloc)
  {
    local_maxloc.maxloc(data[i], i);
  });

  if (size > 0 and not(std::is_same<data_t, size_t>::value))
  {
    boba_assert_gt(maximum, ::boba::lowest_value<data_t>(), "Reduction failure");
  }

  return std::make_pair(maximum, maximum_index);
}

/**
 * Calculates the maximum absolute value element and the corresponding long index.
 * @tparam space Execution space.
 * @tparam data_t Value type.
 * @tparam index_t Index type.
 * @param data Input data.
 * @param size Number of elements.
 * @return Pair of maximum absolute value and its index.
 */

template <execution_space space, typename data_t>
std::pair<real_type_t<data_t>, index_t> max_abs_loc_reduce(const data_t* data, index_t size)
{
  using real_data_t = real_type_t<data_t>;

  real_data_t maximum = ::boba::lowest_value<real_data_t>();
  index_t maximum_index = 0;

  ::boba::max_loc_reduce<space>(maximum, maximum_index, index_t(0), size, [=] __boba_host_device__(::boba::reducer_index_t i, max_loc_reducer_operator<real_data_t> & local_maxloc)
  {
    local_maxloc.maxloc(::boba::abs(data[i]), i);
  });

  if (size > 0 and not(std::is_same<data_t, size_t>::value))
  {
    boba_assert_gt(maximum, ::boba::lowest_value<real_data_t>(), "Reduction failure");
  }

  return std::make_pair(maximum, maximum_index);
}

/**
 * Calculates the minimum element of this tensor.
 * @tparam space Execution space.
 * @tparam data_t Value type.
 * @tparam index_t Index type.
 * @param data Input data.
 * @param size Number of elements.
 * @return Minimum entry.
 */

template <execution_space space, typename data_t>
data_t min_reduce(const data_t* data, index_t size)
{
  static_assert(std::is_same_v<data_t, real_type_t<data_t>>, "min_reduce is disabled for complex data type");

  auto value = ::boba::highest_value<data_t>();
  ::boba::min_reduce<space>(value, index_t(0), size, [=] __boba_host_device__(index_t i, min_reducer_operator<data_t> & local_value)
  {
    local_value.min(data[i]);
  });

  if (size > 0)
  {
    boba_assert_lt(value, ::boba::highest_value<data_t>(), "Reduction failure");
  }

  return value;
}

/**
 * Calculates the minimum of the absolute values of the elements.
 * @tparam space Execution space.
 * @tparam data_t Value type.
 * @tparam index_t Index type.
 * @param data Input data.
 * @param size Number of elements.
 * @return Minimum absolute value.
 */

template <execution_space space, typename data_t>
real_type_t<data_t> min_abs_reduce(const data_t* data, index_t size)
{
  using real_data_t = real_type_t<data_t>;

  auto value = ::boba::highest_value<real_data_t>();

  ::boba::min_reduce<space>(value, index_t(0), size, [=] __boba_host_device__(index_t i, min_reducer_operator<real_data_t> & local_min)
  {
    real_data_t local_value = ::boba::abs(data[i]);
    local_min.min(local_value);
  });

  if (size > 0)
  {
    boba_assert_lt(value, ::boba::highest_value<real_data_t>(), "Reduction failure");
  }

  return value;
}

/**
 * Calculates the minimum element and the corresponding long index.
 * @tparam space Execution space.
 * @tparam data_t Value type.
 * @tparam index_t Index type.
 * @param data Input data.
 * @param size Number of elements.
 * @return Pair of minimum value and its index.
 */

template <execution_space space, typename data_t>
std::pair<data_t, index_t> min_loc_reduce(const data_t* data, index_t size)
{
  static_assert(std::is_same_v<data_t, real_type_t<data_t>>, "min_loc_reduce is disabled for complex type");

  data_t minimum = ::boba::highest_value<data_t>();
  index_t minimum_index = 0;

  ::boba::min_loc_reduce<space>(minimum, minimum_index, 0, size, [=] __boba_host_device__(::boba::reducer_index_t i, ::boba::min_loc_reducer_operator<data_t> & local_minloc)
  {
    local_minloc.minloc(data[i], i);
  });

  if (size > 0)
  {
    boba_assert_lt(minimum, ::boba::highest_value<data_t>(), "Reduction failure");
  }

  return std::make_pair(minimum, minimum_index);
}
} // namespace reductions
} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <type_traits>

#ifdef BOBA_RAJA
#include <RAJA/RAJA.hpp>
#endif

namespace boba::backend::raja
{

#ifdef BOBA_RAJA
template <execution_space space>
struct execution_policy;
template <>
struct execution_policy<execution_space::CPU> { using type = RAJA::seq_exec; };
#ifdef BOBA_CUDA
template <>
struct execution_policy<execution_space::CUDA> { using type = RAJA::cuda_exec_async<256>; };
#endif
#ifdef BOBA_HIP
template <>
struct execution_policy<execution_space::HIP> { using type = RAJA::hip_exec_async<256>; };
#endif

inline constexpr bool enabled = true;
#else
inline constexpr bool enabled = false;
#endif

template <typename data_t, detail::reducer_t reducer_type>
using fallback_value = detail::ReducerValue<data_t, reducer_type>;

template <typename data_t, detail::reducer_t reducer_type>
using fallback_operator = detail::ReducerOperator<data_t, reducer_type>;

#ifdef BOBA_RAJA
struct types
{
  template <typename data_t>
  using sum_value = data_t;
  template <typename data_t>
  using sum_operator = RAJA::expt::ValOp<data_t, RAJA::operators::plus>;
  template <typename data_t>
  using min_value = data_t;
  template <typename data_t>
  using min_operator = RAJA::expt::ValOp<data_t, RAJA::operators::minimum>;
  template <typename data_t>
  using max_value = data_t;
  template <typename data_t>
  using max_operator = RAJA::expt::ValOp<data_t, RAJA::operators::maximum>;
  template <typename data_t>
  using min_loc_value = RAJA::expt::ValLocOp<data_t, reducer_index_t, RAJA::operators::minimum>;
  template <typename data_t>
  using min_loc_operator = RAJA::expt::ValLocOp<data_t, reducer_index_t, RAJA::operators::minimum>;
  template <typename data_t>
  using max_loc_value = RAJA::expt::ValLocOp<data_t, reducer_index_t, RAJA::operators::maximum>;
  template <typename data_t>
  using max_loc_operator = RAJA::expt::ValLocOp<data_t, reducer_index_t, RAJA::operators::maximum>;
};
#else
struct types
{
  template <typename data_t>
  using sum_value = fallback_value<data_t, detail::reducer_t::sum>;
  template <typename data_t>
  using sum_operator = fallback_operator<data_t, detail::reducer_t::sum>;
  template <typename data_t>
  using min_value = fallback_value<data_t, detail::reducer_t::min>;
  template <typename data_t>
  using min_operator = fallback_operator<data_t, detail::reducer_t::min>;
  template <typename data_t>
  using max_value = fallback_value<data_t, detail::reducer_t::max>;
  template <typename data_t>
  using max_operator = fallback_operator<data_t, detail::reducer_t::max>;
  template <typename data_t>
  using min_loc_value = fallback_value<data_t, detail::reducer_t::min_loc>;
  template <typename data_t>
  using min_loc_operator = fallback_operator<data_t, detail::reducer_t::min_loc>;
  template <typename data_t>
  using max_loc_value = fallback_value<data_t, detail::reducer_t::max_loc>;
  template <typename data_t>
  using max_loc_operator = fallback_operator<data_t, detail::reducer_t::max_loc>;
};
#endif

template <execution_space space, typename Reducer, typename Lambda>
inline void loop(index_t begin, index_t end, Reducer&& reducer, Lambda&& lambda)
{
#ifdef BOBA_RAJA
  using policy = typename execution_policy<space>::type;
  RAJA::TypedRangeSegment<index_t> range(begin, end);
  RAJA::forall<policy>(range, std::forward<Reducer>(reducer), std::forward<Lambda>(lambda));
#else
  (void)begin;
  (void)end;
  (void)sizeof(reducer);
  (void)sizeof(lambda);
#endif
}

template <execution_space space, typename data_t, typename Lambda>
inline void sum_reduce(data_t& value, index_t begin, index_t end, Lambda&& lambda)
{
#ifdef BOBA_RAJA
  ::boba::backend::raja::loop<space>(begin, end, RAJA::expt::Reduce<RAJA::operators::plus, data_t>(&value), std::forward<Lambda>(lambda));
#else
  (void)value; (void)begin; (void)end; (void)sizeof(lambda);
#endif
}

template <execution_space space, typename data_t, typename Lambda>
inline void max_reduce(data_t& value, index_t begin, index_t end, Lambda&& lambda)
{
#ifdef BOBA_RAJA
  ::boba::backend::raja::loop<space>(begin, end, RAJA::expt::Reduce<RAJA::operators::maximum, data_t>(&value), std::forward<Lambda>(lambda));
#else
  (void)value; (void)begin; (void)end; (void)sizeof(lambda);
#endif
}

template <execution_space space, typename data_t, typename Lambda>
inline void min_reduce(data_t& value, index_t begin, index_t end, Lambda&& lambda)
{
#ifdef BOBA_RAJA
  ::boba::backend::raja::loop<space>(begin, end, RAJA::expt::Reduce<RAJA::operators::minimum, data_t>(&value), std::forward<Lambda>(lambda));
#else
  (void)value; (void)begin; (void)end; (void)sizeof(lambda);
#endif
}

template <execution_space space, typename data_t, typename Lambda>
inline void min_loc_reduce(data_t& value, index_t& loc, index_t begin, index_t end, Lambda&& lambda)
{
#ifdef BOBA_RAJA
  RAJA::expt::ValLoc<data_t> target(value, static_cast<reducer_index_t>(loc));
  ::boba::backend::raja::loop<space>(begin, end, RAJA::expt::Reduce<RAJA::operators::minimum>(&target), std::forward<Lambda>(lambda));
  value = static_cast<data_t>(target.getVal());
  loc = static_cast<index_t>(target.getLoc());
#else
  (void)value; (void)loc; (void)begin; (void)end; (void)sizeof(lambda);
#endif
}

template <execution_space space, typename data_t, typename Lambda>
inline void max_loc_reduce(data_t& value, index_t& loc, index_t begin, index_t end, Lambda&& lambda)
{
#ifdef BOBA_RAJA
  RAJA::expt::ValLoc<data_t> target(value, static_cast<reducer_index_t>(loc));
  ::boba::backend::raja::loop<space>(begin, end, RAJA::expt::Reduce<RAJA::operators::maximum>(&target), std::forward<Lambda>(lambda));
  value = static_cast<data_t>(target.getVal());
  loc = static_cast<index_t>(target.getLoc());
#else
  (void)value; (void)loc; (void)begin; (void)end; (void)sizeof(lambda);
#endif
}

} // namespace boba::backend::raja

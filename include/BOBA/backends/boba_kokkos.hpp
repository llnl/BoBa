// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_KOKKOS
#include <Kokkos_Core.hpp>
#endif

namespace boba::backend::kokkos
{

#ifdef BOBA_KOKKOS
inline constexpr bool enabled = true;
#else
inline constexpr bool enabled = false;
#endif

template <typename data_t, detail::reducer_t reducer_type>
using fallback_value = detail::ReducerValue<data_t, reducer_type>;

template <typename data_t, detail::reducer_t reducer_type>
using fallback_operator = detail::ReducerOperator<data_t, reducer_type>;

#ifdef BOBA_KOKKOS
template <execution_space space>
struct execution_space_type;
template <>
struct execution_space_type<execution_space::CPU> { using type = Kokkos::Serial; };
#ifdef BOBA_CUDA
template <>
struct execution_space_type<execution_space::CUDA> { using type = Kokkos::Cuda; };
#endif
#ifdef BOBA_HIP
template <>
struct execution_space_type<execution_space::HIP> { using type = Kokkos::HIP; };
#endif

struct types
{
  template <typename data_t> using sum_value = data_t;
  template <typename data_t> using sum_operator = detail::ReducerOperator<data_t, detail::reducer_t::sum>;
  template <typename data_t> using min_value = data_t;
  template <typename data_t> using min_operator = detail::ReducerOperator<data_t, detail::reducer_t::min>;
  template <typename data_t> using max_value = data_t;
  template <typename data_t> using max_operator = detail::ReducerOperator<data_t, detail::reducer_t::max>;
  template <typename data_t> using min_loc_value = detail::ReducerValue<data_t, detail::reducer_t::min_loc>;
  template <typename data_t> using min_loc_operator = detail::ReducerOperator<data_t, detail::reducer_t::min_loc>;
  template <typename data_t> using max_loc_value = detail::ReducerValue<data_t, detail::reducer_t::max_loc>;
  template <typename data_t> using max_loc_operator = detail::ReducerOperator<data_t, detail::reducer_t::max_loc>;
};
#else
struct types
{
  template <typename data_t> using sum_value = fallback_value<data_t, detail::reducer_t::sum>;
  template <typename data_t> using sum_operator = fallback_operator<data_t, detail::reducer_t::sum>;
  template <typename data_t> using min_value = fallback_value<data_t, detail::reducer_t::min>;
  template <typename data_t> using min_operator = fallback_operator<data_t, detail::reducer_t::min>;
  template <typename data_t> using max_value = fallback_value<data_t, detail::reducer_t::max>;
  template <typename data_t> using max_operator = fallback_operator<data_t, detail::reducer_t::max>;
  template <typename data_t> using min_loc_value = fallback_value<data_t, detail::reducer_t::min_loc>;
  template <typename data_t> using min_loc_operator = fallback_operator<data_t, detail::reducer_t::min_loc>;
  template <typename data_t> using max_loc_value = fallback_value<data_t, detail::reducer_t::max_loc>;
  template <typename data_t> using max_loc_operator = fallback_operator<data_t, detail::reducer_t::max_loc>;
};
#endif

template <execution_space space, typename Reducer, typename Lambda>
inline void loop(index_t begin, index_t end, Reducer&& reducer, Lambda&& lambda)
{
#ifdef BOBA_KOKKOS
  using execution_space_t = typename execution_space_type<space>::type;
  Kokkos::parallel_for("boba::loop", Kokkos::RangePolicy<execution_space_t>(begin, end),
    [=] __boba_host_device__ (index_t i) mutable { lambda(i, reducer); });
  execution_space_t().fence();
#else
  (void)begin; (void)end; (void)sizeof(reducer); (void)sizeof(lambda);
#endif
}

template <execution_space space, typename data_t, typename Lambda>
inline void sum_reduce(data_t& value, index_t begin, index_t end, Lambda&& lambda)
{
#ifdef BOBA_KOKKOS
  using execution_space_t = typename execution_space_type<space>::type;
  Kokkos::parallel_reduce("boba::sum_reduce", Kokkos::RangePolicy<execution_space_t>(begin, end),
    [=] __boba_host_device__ (index_t i, data_t& local) { typename types::template sum_operator<data_t> reducer(&local); lambda(i, reducer); }, value);
  execution_space_t().fence();
#else
  (void)value; (void)begin; (void)end; (void)sizeof(lambda);
#endif
}

template <execution_space space, typename data_t, typename Lambda>
inline void max_reduce(data_t& value, index_t begin, index_t end, Lambda&& lambda)
{
#ifdef BOBA_KOKKOS
  using execution_space_t = typename execution_space_type<space>::type;
  Kokkos::parallel_reduce("boba::max_reduce", Kokkos::RangePolicy<execution_space_t>(begin, end),
    [=] __boba_host_device__ (index_t i, data_t& local) { typename types::template max_operator<data_t> reducer(&local); lambda(i, reducer); }, Kokkos::Max<data_t>(value));
  execution_space_t().fence();
#else
  (void)value; (void)begin; (void)end; (void)sizeof(lambda);
#endif
}

template <execution_space space, typename data_t, typename Lambda>
inline void min_reduce(data_t& value, index_t begin, index_t end, Lambda&& lambda)
{
#ifdef BOBA_KOKKOS
  using execution_space_t = typename execution_space_type<space>::type;
  Kokkos::parallel_reduce("boba::min_reduce", Kokkos::RangePolicy<execution_space_t>(begin, end),
    [=] __boba_host_device__ (index_t i, data_t& local) { typename types::template min_operator<data_t> reducer(&local); lambda(i, reducer); }, Kokkos::Min<data_t>(value));
  execution_space_t().fence();
#else
  (void)value; (void)begin; (void)end; (void)sizeof(lambda);
#endif
}

template <execution_space space, typename data_t, typename Lambda>
inline void min_loc_reduce(data_t& value, index_t& loc, index_t begin, index_t end, Lambda&& lambda)
{
#ifdef BOBA_KOKKOS
  using execution_space_t = typename execution_space_type<space>::type;
  Kokkos::ValLocScalar<data_t, index_t> target{value, loc};
  Kokkos::parallel_reduce("boba::min_loc_reduce", Kokkos::RangePolicy<execution_space_t>(begin, end),
    [=] __boba_host_device__ (index_t i, Kokkos::ValLocScalar<data_t, index_t>& local) { typename types::template min_loc_operator<data_t> reducer(&local.val, &local.loc); lambda(i, reducer); }, Kokkos::MinLoc<data_t, index_t>(target));
  execution_space_t().fence();
  value = target.val; loc = target.loc;
#else
  (void)value; (void)loc; (void)begin; (void)end; (void)sizeof(lambda);
#endif
}

template <execution_space space, typename data_t, typename Lambda>
inline void max_loc_reduce(data_t& value, index_t& loc, index_t begin, index_t end, Lambda&& lambda)
{
#ifdef BOBA_KOKKOS
  using execution_space_t = typename execution_space_type<space>::type;
  Kokkos::ValLocScalar<data_t, index_t> target{value, loc};
  Kokkos::parallel_reduce("boba::max_loc_reduce", Kokkos::RangePolicy<execution_space_t>(begin, end),
    [=] __boba_host_device__ (index_t i, Kokkos::ValLocScalar<data_t, index_t>& local) { typename types::template max_loc_operator<data_t> reducer(&local.val, &local.loc); lambda(i, reducer); }, Kokkos::MaxLoc<data_t, index_t>(target));
  execution_space_t().fence();
  value = target.val; loc = target.loc;
#else
  (void)value; (void)loc; (void)begin; (void)end; (void)sizeof(lambda);
#endif
}

} // namespace boba::backend::kokkos

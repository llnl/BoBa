// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief Statically sized vector backed by StaticTensor storage.
 */
template <typename _sizes, typename _data_t>
struct StaticVector : StaticTensor<_sizes, _data_t>
{
  using base = StaticTensor<_sizes, _data_t>;

  template <std::size_t dim, std::size_t new_size>
  using resize_dimension_type = StaticVector<
    typename _sizes::template replace_value_at_index_t<new_size, dim>,
    _data_t>;

  using base::base;
  using base::operator=;

  using typename base::data_t;
  using typename base::index_array;
};

} // namespace boba

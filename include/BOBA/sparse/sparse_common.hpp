// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/tensors/Matrix.hpp"
#include "BOBA/tensors/Vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace boba
{

namespace sparse_detail
{

template <typename data_t>
using host_vector = Vector<host_space, data_t>;

template <typename data_t>
using host_matrix = Matrix<host_space, data_t>;

template <typename data_t>
using host_tensor3 = Tensor<3, host_space, data_t>;

template <typename data_t>
void check_tolerance(data_t tolerance)
{
  boba_always_assert(tolerance >= data_t{0}, "Sparse tolerance must be nonnegative");
  boba_always_assert(std::isfinite(static_cast<double>(tolerance)), "Sparse tolerance must be finite");
}

template <typename data_t>
bool keep(data_t value, data_t tolerance)
{
  return abs(value) > tolerance;
}

template <typename data_t>
void initialize_output(host_vector<data_t>& y, data_t beta)
{
  if (abs(beta) <= 0)
  {
    y.fill_with_zeros();
  }
  else
  {
    y *= beta;
  }
}

} // namespace sparse_detail

} // namespace boba

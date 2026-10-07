// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/tensors/Matrix.hpp"
#include "BOBA/tensors/Vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace boba
{

/**
 * \file SparseMatrixCommon.hpp
 * \brief Shared storage helpers and read-only sparse-matrix views.
 */

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
void check_vector_sizes(index_t rows, index_t cols, host_vector<data_t> const& x, host_vector<data_t> const& y)
{
  boba_always_assert_equal(x.size(), cols, "Sparse matvec input has the wrong size");
  boba_always_assert_equal(y.size(), rows, "Sparse matvec output has the wrong size");
}

template <typename data_t>
void initialize_output(host_vector<data_t>& y, data_t beta)
{
  auto y_view = y.view();
  if (abs(beta) <= 0)
  {
    y.fill_with_zeros();
  }
  else
  {
    for (index_t i = 0; i < y.size(); ++i)
      y_view(i) *= beta;
  }
}

template <typename data_t>
host_matrix<data_t> copy_dense(host_matrix<data_t> const& dense)
{
  return dense;
}

} // namespace sparse_detail

/**
 * \brief Non-owning read-only view of a sparse matrix owner.
 *
 * Structural changes to the owner can invalidate the view.
 */
template <typename owner_t>
struct SparseMatrixConstView
{
  owner_t const* owner = nullptr;

  index_t nrows() const noexcept
  {
    return owner->nrows();
  }
  index_t ncols() const noexcept
  {
    return owner->ncols();
  }
  auto shape() const noexcept
  {
    return owner->shape();
  }
  auto get_element(index_t i, index_t j) const
  {
    return owner->get_element(i, j);
  }

  void matvec(Vector<host_space, typename owner_t::data_type> const& x,
              Vector<host_space, typename owner_t::data_type>& y,
              typename owner_t::data_type alpha = typename owner_t::data_type{1},
              typename owner_t::data_type beta = typename owner_t::data_type{0}) const
  {
    owner->matvec(x, y, alpha, beta);
  }
};

} // namespace boba

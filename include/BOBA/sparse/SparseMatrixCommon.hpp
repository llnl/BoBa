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
    ::boba::detail::loop<host_space>(0, y.size(), [=](index_t i)
    {
      y_view(i) *= beta;
    });
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
  owner_t const* m_owner = nullptr;

  /**
   * \brief Returns the number of rows in the viewed matrix.
   */
  index_t rows() const noexcept
  {
    return m_owner->rows();
  }

  /**
   * \brief Returns the number of columns in the viewed matrix.
   */
  index_t cols() const noexcept
  {
    return m_owner->cols();
  }

  /**
   * \brief Returns the logical shape of the viewed matrix.
   */
  auto shape() const noexcept
  {
    return m_owner->shape();
  }

  /**
   * \brief Returns a stored value or zero when the coordinate is absent.
   */
  auto get_element(index_t i, index_t j) const
  {
    return m_owner->get_element(i, j);
  }

  /**
   * \brief Computes a dense-vector product through the viewed matrix.
   */
  void matvec(Vector<host_space, typename owner_t::data_type> const& x,
              Vector<host_space, typename owner_t::data_type>& y,
              typename owner_t::data_type alpha = typename owner_t::data_type{1},
              typename owner_t::data_type beta = typename owner_t::data_type{0}) const
  {
    m_owner->matvec(x, y, alpha, beta);
  }
};

} // namespace boba

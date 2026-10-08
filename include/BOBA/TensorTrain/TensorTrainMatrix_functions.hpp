// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

// -------------------------------------------------------------------------------------
// Section: NaN checks
// -------------------------------------------------------------------------------------

template <size_t dimension, execution_space space, typename data_t>
void nan_check(TensorTrainMatrix<dimension, space, data_t> const& ttm)
{
  for (size_t d = 0; d < dimension; d++)
  {
    ::boba::nan_check(ttm.cores[d]);
  }
}

// -------------------------------------------------------------------------------------
// Section: Norms
// -------------------------------------------------------------------------------------

/**
 * \brief Computes the Frobenius norm of a tensor train matrix.
 */

template <size_t dimension, execution_space space, typename data_t>
typename TensorTrainMatrix<dimension, space, data_t>::real_data_t
norm_frobenius(TensorTrainMatrix<dimension, space, data_t> const& ttm)
{
  BOBA_CALI_MARK
  TensorTrainMatrix<dimension, space, data_t> orthogonalized_ttm(ttm);
  orthogonalized_ttm.orthogonalize();
  return ::boba::norm_frobenius(orthogonalized_ttm.cores[dimension - 1]);
}

// -------------------------------------------------------------------------------------
// Section: Norm differences
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Frobenius norm difference of two tensor train matrices
 */

template <size_t dimension, execution_space space, typename data_t>
data_t norm_difference_frobenius(
  TensorTrainMatrix<dimension, space, data_t> const& ttm_A,
  TensorTrainMatrix<dimension, space, data_t> const& ttm_B)
{
  BOBA_CALI_MARK
  checkpoint();
  TensorTrainMatrix<dimension, space, data_t> temp = ttm_A - ttm_B;
  temp.rename("norm_difference_temp");
  auto diff_frobenius = ::boba::norm_frobenius(temp);
  return diff_frobenius;
}

} // namespace boba

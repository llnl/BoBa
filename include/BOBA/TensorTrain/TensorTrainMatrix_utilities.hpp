// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{
/**
 * \brief
 * Automatically constructs a ttm to compress the input matrix
 */

template <size_t dimension, execution_space space, typename data_t>
TensorTrainMatrix<dimension, space, data_t> compress_to_TensorTrainMatrix(
  const Matrix<space, data_t>& input_matrix,
  Array<size_t, dimension> rows,
  Array<size_t, dimension> cols,
  data_t svd_tolerance_relative = -1.0,
  data_t svd_tolerance_absolute = -1.0)
{
  BOBA_CALI_MARK
  TensorTrainMatrix<dimension, space, data_t> new_ttm(rows, cols);
  if (svd_tolerance_relative > 0.0)
  {
    new_ttm.svd_tolerance_relative = svd_tolerance_relative;
  }
  if (svd_tolerance_absolute > 0.0)
  {
    new_ttm.svd_tolerance_absolute = svd_tolerance_absolute;
  }
  new_ttm.compress(input_matrix);
  return new_ttm;
}

/**
 * \brief
 * Constructs a rank-one ttm from a sequence of matrices
 */

template <size_t dimension, execution_space space, typename data_t>
TensorTrainMatrix<dimension, space, data_t> make_ttm_from_matrices(Array<Matrix<space, data_t>, dimension> matrices)
{
  Array<size_t, dimension> rows;
  Array<size_t, dimension> cols;
  for (size_t d = 0; d < dimension; d++)
  {
    rows[d] = matrices[d].rows();
    cols[d] = matrices[d].cols();
  }

  TensorTrainMatrix<dimension, space, data_t> new_ttm(rows, cols);
  new_ttm.fill_with_zeros();

  for (size_t d = 0; d < dimension; d++)
  {
    new_ttm.cores[d].reshape(matrices[d]);
  }
  return new_ttm;
}

} // namespace boba

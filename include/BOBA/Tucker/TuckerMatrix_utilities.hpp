// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

template <size_t dimension, execution_space space, typename data_t>
void nan_check(TuckerMatrix<dimension, space, data_t> const& tucker_matrix)
{
  for (size_t d = 0; d < dimension; d++)
  {
    ::boba::nan_check(tucker_matrix.cores[d]);
  }
  ::boba::nan_check(tucker_matrix.R_core);
}

/**
 * \brief Frobenius norm of the Tucker matrix in the sense of vectors.
 */

template <size_t dimension, execution_space space, typename data_t>
auto norm_frobenius(TuckerMatrix<dimension, space, data_t> const& tucker_matrix)
{
  BOBA_CALI_MARK
  const auto product = tucker_matrix.inner_product(tucker_matrix);
  const auto abs_product = boba::abs(product);
  return boba::sqrt(abs_product);
}

/**
 * \brief
 * Constructs a rank-one ttm from a sequence of matrices
 */

template <size_t dimension, execution_space space, typename data_t>
TuckerMatrix<dimension, space, data_t> make_TuckerMatrix_from_matrices(Array<Matrix<space, data_t>, dimension> matrices)
{
  Array<size_t, dimension> rows;
  Array<size_t, dimension> cols;
  for (size_t d = 0; d < dimension; d++)
  {
    rows[d] = matrices[d].rows();
    cols[d] = matrices[d].cols();
  }

  TuckerMatrix<dimension, space, data_t> new_matrix(rows, cols);
  new_matrix.fill_with(1.0);

  for (size_t d = 0; d < dimension; d++)
  {
    new_matrix.cores[d].reshape(matrices[d]);
  }
  return new_matrix;
}

/**
 * @brief Perform the elementwise_product (aka Hadamard product) of inputs
 *
 * @param[in] TuckerMat_A TuckerMatrix
 * @param[in] TuckerMat_B TuckerMatrix
 * @return TuckerMatrix which is the Hadamard product TuckerMat_A * TuckerMat_B
 */

template <size_t dimension, execution_space space, typename data_t>
TuckerMatrix<dimension, space, data_t> elementwise_product(
  TuckerMatrix<dimension, space, data_t> const& TuckerMat_A,
  TuckerMatrix<dimension, space, data_t> const& TuckerMat_B)
{
  BOBA_CALI_MARK
  checkpoint();
  auto this_rows = TuckerMat_A.core_rows();
  auto this_cols = TuckerMat_A.core_cols();
  auto Tucker_A = TuckerMat_A.write_to_tucker();
  auto Tucker_B = TuckerMat_B.write_to_tucker();
  checkpoint();
  auto output_Tucker = elementwise_product(Tucker_A, Tucker_B);
  checkpoint();
  TuckerMatrix<dimension, space, data_t> output;
  output.read_from_Tucker(output_Tucker, this_rows, this_cols);
  return output;
}

} // namespace boba

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{
/**
 * \brief
 * Automatically constructs a Tucker decomposition to compress the input tensor
 */

template <size_t dimension, execution_space space, typename data_t>
Tucker<dimension, space, data_t> compress_to_tucker(
  const Tensor<dimension, space, data_t>& tensor,
  data_t svd_tolerance_relative = -1.0,
  data_t svd_tolerance_absolute = -1.0)
{
  BOBA_CALI_MARK
  Tucker<dimension, space, data_t> new_tucker(tensor.sizes());
  if (svd_tolerance_relative > 0.0)
  {
    new_tucker.svd_tolerance_relative = svd_tolerance_relative;
  }
  if (svd_tolerance_absolute > 0.0)
  {
    new_tucker.svd_tolerance_absolute = svd_tolerance_absolute;
  }
  new_tucker.compress(tensor);
  return new_tucker;
}

/**
 * \brief
 * Constructs a rank-one Tucker from a sequence of vectors
 */

template <size_t dimension, execution_space space, typename data_t>
Tucker<dimension, space, data_t> make_tucker_from_vectors(Array<Vector<space, data_t>, dimension> vectors)
{
  BOBA_CALI_MARK
  Array<size_t, dimension> sizes;
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] = vectors[d].size();
  }

  Tucker<dimension, space, data_t> new_tucker(sizes);
  new_tucker.fill_with(1.0);

  for (size_t d = 0; d < dimension; d++)
  {
    new_tucker.cores[d].reshape(vectors[d]);
  }
  return new_tucker;
}

/**
 * \brief
 * Converts a tensor train to a Tucker
 */

template <size_t dimension, execution_space space, typename data_t>
Tucker<dimension, space, data_t> tt_to_tucker(TensorTrain<dimension, space, data_t> input)
{
  BOBA_CALI_MARK
  Tucker<dimension, space, data_t> output(input.sizes());
  output.svd_tolerance_relative = input.svd_tolerance_relative;
  output.svd_tolerance_absolute = input.svd_tolerance_absolute;
  output.max_ranks = input.max_ranks;

  auto unroll = input.decompress();
  output.compress(unroll);

  return output;
}

/**
 * \brief
 * Converts a tensor train to a Tucker
 */

template <size_t dimension, execution_space space, typename data_t>
TensorTrain<dimension, space, data_t> tucker_to_TensorTrain(Tucker<dimension, space, data_t> input)
{
  BOBA_CALI_MARK
  TensorTrain<dimension, space, data_t> output(input.sizes());
  output.svd_tolerance = input.svd_tolerance;
  output.max_ranks = input.max_ranks;

  auto unroll = input.decompress();
  output.compress(unroll);

  return output;
}

/**
 * \brief
 * Converts a tensor train matrix to a Tucker matrix
 */

template <size_t dimension, execution_space space, typename data_t>
TuckerMatrix<dimension, space, data_t> TensorTrainMatrix_to_TuckerMatrix(TensorTrainMatrix<dimension, space, data_t> input)
{
  BOBA_CALI_MARK
  TuckerMatrix<dimension, space, data_t> output(input.core_rows(), input.core_cols());
  output.svd_tolerance_absolute = input.svd_tolerance_absolute;
  output.svd_tolerance_relative = input.svd_tolerance_relative;
  output.max_ranks = input.max_ranks;

  auto unroll = input.decompress();
  output.compress(unroll);

  return output;
}

/**
 * \brief
 * Converts a Tucker matrix to a tensor train matrix
 */

template <size_t dimension, execution_space space, typename data_t>
TensorTrainMatrix<dimension, space, data_t> TuckerMatrix_to_TensorTrainMatrix(TuckerMatrix<dimension, space, data_t> input)
{
  BOBA_CALI_MARK
  TensorTrainMatrix<dimension, space, data_t> output(input.rows(), input.cols());
  output.svd_tolerance_relative = input.svd_tolerance_relative;
  output.svd_tolerance_absolute = input.svd_tolerance_absolute;
  output.max_ranks = input.max_ranks;

  auto unroll = input.decompress();
  output.compress(unroll);

  return output;
}

/**
 * \brief
 * Iterates through the tolerance parameter on the log scale and prints off the compression rate
 */

template <size_t dimension, execution_space space, typename data_t>
void scan_tucker_compression_versus_tolerance(
  Tensor<dimension, space, data_t>& tensor)
{
  BOBA_CALI_MARK
  std::cout
    << "Tolerance CR Inf_Error Frob_error Rel_Inf_Error Rel_Frob_Error" << std::endl;
  for (size_t ei = 1; ei < 14 * 3; ei++)
  {
    double svd_tolerance_guess = ::boba::pow(10.0, -double(ei / 3.0));
    auto compressed_tensor = compress_to_tucker(tensor, svd_tolerance_guess);
    auto cr = compressed_tensor.compression_rate();
    auto norm_inf = ::boba::norm_inf(tensor);
    auto norm_frob = ::boba::norm_frobenius(tensor);
    auto compressed_tensor_decompress = compressed_tensor.decompress();
    auto error_inf = ::boba::norm_difference_inf(compressed_tensor_decompress, tensor);
    auto error_frob = ::boba::norm_difference_frobenius(compressed_tensor_decompress, tensor);
    auto rel_error_frob = error_frob / norm_frob;
    auto rel_error_inf = error_inf / norm_inf;
    std::cout
      << svd_tolerance_guess << " " << cr << " "
      << error_inf << " " << error_frob << " "
      << rel_error_inf << " " << rel_error_frob << std::endl;
  }
}

/**
 * \brief
 * Compute the various singular value spectra associated with this Tucker decomposition
 */

template <size_t dimension, execution_space space, typename data_t>
::boba::Matrix<space, data_t> compute_tucker_spectra(const boba::Tensor<dimension, space, data_t>& input)
{
  BOBA_CALI_MARK
  checkpoint();
  ::boba::Matrix<space, data_t> spectra;

  Array<::boba::Vector<space, data_t>, dimension> spectrum;

  Tensor<dimension, space, data_t> R_core = input;

  for (size_t d = 0; d < dimension; d++)
  {
    auto initial_sizes = R_core.sizes();
    // Unfold over mode d
    auto unfold_right = unfold(R_core, d);

    // Compute SVD
    SVD<space, data_t> svd;
    svd.tolerance_relative = 1.0e-15;
    svd.tolerance_absolute = 0.0;

    svd(unfold_right);
    // Copy S vector
    spectrum[d] = svd.S;
    apply_as_diagonal_right_in_place(svd.S, svd.V);
    // this->cores[d] = svd.U;

    // Write back to R
    initial_sizes[d] = svd.V.cols();
    R_core = fold(svd.V.transpose(), initial_sizes, d);
  }

  size_t largest_size = 1;
  for (size_t d = 0; d < dimension; d++)
  {
    largest_size = max(largest_size, spectrum[d].size());
  }
  spectra.resize({largest_size, dimension});
  spectra.fill_with_zeros();
  auto spectra_view = spectra.view();

  for (size_t d = 0; d < dimension; d++)
  {
    auto spectrum_view = spectrum[d].view();
    ::boba::loop<space, 1>({spectrum[d].size()},
                           [=] __boba_host_device__(size_t i)
    {
      spectra_view({i, d}) = spectrum_view(i);
    });
  }

  return spectra;
}

} // namespace boba

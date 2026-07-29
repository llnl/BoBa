// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \param[in] tensor Tensor to compress.
 * \param[in] rank Rank to which the CPD will approximate `tensor`.
 * \param[in] ALS_tolerance_relative Optional relative tolerance override.
 * \param[in] ALS_tolerance_absolute Optional absolute tolerance override.
 * \return The CPD approximating `tensor` to the given rank.
 */

template <size_t dimension, execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t> compress_to_cpd(
  const Tensor<dimension, space, data_t>& tensor,
  size_t rank,
  data_t ALS_tolerance_relative = -1.0,
  data_t ALS_tolerance_absolute = -1.0)
{
  BOBA_CALI_MARK
  CanonicalPolyadicDecomposition<dimension, space, data_t> new_cpd(tensor.sizes());
  if (ALS_tolerance_relative > 0.0)
  {
    new_cpd.ALS_tolerance_relative = ALS_tolerance_relative;
  }
  if (ALS_tolerance_absolute > 0.0)
  {
    new_cpd.ALS_tolerance_absolute = ALS_tolerance_absolute;
  }
  new_cpd.compress(tensor, rank);
  return new_cpd;
}

/**
 * \param[in] vectors Basis vectors used to create the CPD.
 * \return A rank-1 CPD using the vectors as basis functions.
 */

template <size_t dimension, execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t> make_cpd_from_vectors(Array<Vector<space, data_t>, dimension> vectors)
{
  BOBA_CALI_MARK
  Array<size_t, dimension> sizes;
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] = vectors[d].size();
  }

  CanonicalPolyadicDecomposition<dimension, space, data_t> new_cpd(sizes);
  new_cpd.fill_with(1.0);

  for (size_t d = 0; d < dimension; d++)
  {
    new_cpd.cores[d].reshape(vectors[d]);
  }
  return new_cpd;
}

/**
 * \brief Iterates through ranks and prints metrics resulting from CPD compression.
 * Useful for exploring the tradeoff between compression rate and error.
 * \param[in,out] tensor Tensor to compress.
 * \param[in] max_rank Maximum rank to test.
 */

template <size_t dimension, execution_space space, typename data_t>
void scan_cpd_compression_versus_rank(
  Tensor<dimension, space, data_t>& tensor,
  size_t max_rank = 20)
{
  BOBA_CALI_MARK
  std::cout
    << "Rank CR Inf_Error Frob_error Rel_Inf_Error Rel_Frob_Error" << std::endl;
  for (size_t rank = 1; rank < max_rank; rank++)
  {
    auto compressed_tensor = compress_to_cpd(tensor, rank);
    auto cr = compressed_tensor.compression_rate();
    auto norm_inf = ::boba::norm_inf(tensor);
    auto norm_frob = ::boba::norm_frobenius(tensor);
    auto error_inf = ::boba::norm_difference_inf(compressed_tensor.decompress(), tensor);
    auto error_frob = ::boba::norm_difference_frobenius(compressed_tensor.decompress(), tensor);
    auto rel_error_frob = error_frob / norm_frob;
    auto rel_error_inf = error_inf / norm_inf;
    std::cout
      << rank << " " << cr << " "
      << error_inf << " " << error_frob << " "
      << rel_error_inf << " " << rel_error_frob << std::endl;
  }
}

} // namespace boba

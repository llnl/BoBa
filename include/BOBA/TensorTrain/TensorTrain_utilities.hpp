// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{
/**
 * \brief
 * Automatically constructs a tt to compress the input tensor
 */

template <size_t dimension, execution_space space, typename data_t>
TensorTrain<dimension, space, data_t> compress_to_TensorTrain(
  const Tensor<dimension, space, data_t>& tensor,
  data_t svd_tolerance_relative = -1.0,
  data_t svd_tolerance_absolute = -1.0)
{
  BOBA_CALI_MARK
  TensorTrain<dimension, space, data_t> new_tt(tensor.sizes());
  if (svd_tolerance_relative > 0.0)
  {
    new_tt.svd_tolerance_relative = svd_tolerance_relative;
  }
  if (svd_tolerance_absolute > 0.0)
  {
    new_tt.svd_tolerance_absolute = svd_tolerance_absolute;
  }
  new_tt.compress(tensor);
  return new_tt;
}

/**
 * \brief
 * Constructs a rank-one tt from a sequence of vectors
 */

template <size_t dimension, execution_space space, typename data_t>
TensorTrain<dimension, space, data_t> make_tt_from_vectors(Array<Vector<space, data_t>, dimension> vectors)
{
  BOBA_CALI_MARK
  Array<size_t, dimension> sizes;
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] = vectors[d].size();
  }

  TensorTrain<dimension, space, data_t> new_tt(sizes);
  new_tt.fill_with_zeros();

  for (size_t d = 0; d < dimension; d++)
  {
    new_tt.cores[d].reshape(vectors[d]);
  }
  return new_tt;
}

/**
 * \brief
 * Takes in a tensor of total size base^E and makes a QTT of length E.
 */

template <size_t dimension, execution_space space, typename data_t>
QuantizedTensorTrain<space, data_t> compress_to_QuantizedTensorTrain(
  Tensor<dimension, space, data_t> tensor,
  size_t base = 2,
  data_t svd_tolerance_relative = -1.0,
  data_t svd_tolerance_absolute = -1.0)
{
  BOBA_CALI_MARK
  checkpoint();
  for (size_t d = 0; d < dimension; d++)
  {
    size_t size = tensor.sizes(d);
    size_t exponent = ::boba::logb(size, base);
    boba_always_assert_equal(::boba::pow(base, exponent), size, " Sizes each must be of the form base^exponent.");
  }
  size_t exponent = ::boba::logb(tensor.size(), base);

  Vector<space, data_t> temp({pow(base, exponent)});
  temp.reshape(tensor);

  checkpoint();
  QuantizedTensorTrain<space, data_t> qtt(base, exponent);
  if (svd_tolerance_relative > 0.0)
  {
    qtt.svd_tolerance_relative = svd_tolerance_relative;
  }
  if (svd_tolerance_absolute > 0.0)
  {
    qtt.svd_tolerance_absolute = svd_tolerance_absolute;
  }
  qtt.compress(temp);

  return qtt;
}

/**
 * \brief
 * Cycles through all permutations of a tensor to find the highest tensor train compression rate for a fixed tolerance.
 * Returns the optimal permutation and the corresponding tt
 */

template <size_t dimension, execution_space space, typename data_t>
std::pair<::boba::Array<size_t, dimension>, TensorTrain<dimension, space, data_t>> find_optimal_tensor_permutation(
  const Tensor<dimension, space, data_t>& tensor,
  const double svd_tolerance_relative,
  const double svd_tolerance_absolute,
  bool verbose = false,
  Array<size_t, 2> begin_end_each_ids = {0, PermutationMultiindexer<dimension>::size()})
{
  BOBA_CALI_MARK

  PermutationMultiindexer<dimension> all_permutations;

  size_t optimal_tt_cr = 0;
  TensorTrain<dimension, space, data_t> optimal_tt(tensor.sizes());
  optimal_tt.fill_with_zeros();
  auto optimal_tt_mid = range<size_t, dimension>();

  auto begin_id = begin_end_each_ids[0];
  auto end_id = begin_end_each_ids[1];

  checkpoint();
  for (size_t I = begin_id; I < end_id; I++)
  {
    auto permutation_mid = all_permutations.multiindex(I);
    if (not(is_valid_permutation(permutation_mid)))
    {
      continue;
    }

    boba_print("copying");
    boba::Tensor<dimension, space, data_t> tensor_copy{tensor};

    boba_print("permuting");
    permute(tensor_copy, permutation_mid);

    {
      boba_print("compressing");
      auto tt = compress_to_TensorTrain(tensor_copy, svd_tolerance_relative, svd_tolerance_absolute);
      auto cr = tt.compression_rate();
      if (cr > optimal_tt_cr)
      {
        optimal_tt_cr = cr;
        optimal_tt_mid = permutation_mid;
        optimal_tt = tt;
        if (verbose)
        {
          std::cout
            << "optimal TT CR: " << optimal_tt_cr
            << " at " << optimal_tt_mid
            << " ranks: " << tt.ranks_string()
            << std::endl;
        }
      }
    }

    size_t percent_compete = 100.0 * static_cast<double>(I - begin_id) / static_cast<double>(end_id - begin_id);
    boba_print(percent_compete);
  }

  return std::make_pair(optimal_tt_mid, optimal_tt);
}

/**
 * \brief
 * Searches through the tolerance parameter to find the highest tolerance that generates a tt
 * estimation of tensor such that the error is below tt_error_tolerance
 */

template <size_t dimension, execution_space space, typename data_t>
std::pair<data_t, TensorTrain<dimension, space, data_t>> find_optimal_tt_tolerance(
  Tensor<dimension, space, data_t>& tensor,
  data_t svd_tolerance_guess,
  data_t tt_error_tolerance,
  bool verbose = false)
{
  BOBA_CALI_MARK
  boba_always_assert_gt(svd_tolerance_guess, 0.0, "Guess must be positive");

  auto optimal_tt = compress_to_TensorTrain(tensor, svd_tolerance_guess);
  auto error_lb = ::boba::norm_difference_inf(optimal_tt.decompress(), tensor);

  // If initial tolerance is too high, exit
  if (svd_tolerance_guess >= 1.0)
  {
    return std::make_pair(error_lb, optimal_tt);
  }

  data_t tol_upper_bnd = 1.0;
  data_t tol_lower_bnd = svd_tolerance_guess;

  auto up_tt = compress_to_TensorTrain(tensor, tol_upper_bnd);
  auto error_ub = ::boba::norm_difference_inf(up_tt.decompress(), tensor);

  if (error_ub <= tt_error_tolerance)
  {
    return std::make_pair(tol_upper_bnd, up_tt);
  }

  auto verbose_output = [verbose, tt_error_tolerance](data_t tl, data_t tu, data_t el, data_t eu)
  {
    if (verbose)
    {
      std::cout << "Tolerance in (" << tl << "," << tu
                << "), error in (" << el << ", " << eu
                << ") vs. " << tt_error_tolerance << std::endl;
    }
  };

  verbose_output(tol_lower_bnd, tol_upper_bnd, error_lb, error_ub);

  // Ensure sensible lower bound error
  while (error_lb > tt_error_tolerance)
  {
    tol_lower_bnd = pow(tol_lower_bnd, 2.0);
    optimal_tt = compress_to_TensorTrain(tensor, tol_lower_bnd);
    error_lb = ::boba::norm_difference_inf(optimal_tt.decompress(), tensor);
    verbose_output(tol_lower_bnd, tol_upper_bnd, error_lb, error_ub);
    if (tol_lower_bnd < 2.0 * epsilon<data_t>())
    {
      return std::make_pair(tol_lower_bnd, optimal_tt);
    }
  }

  bool searching = true;
  size_t iterations = 0;

  // Bisection search with geometric average
  while (searching)
  {
    auto middle = sqrt(tol_upper_bnd * tol_lower_bnd);
    optimal_tt = compress_to_TensorTrain(tensor, middle);
    auto tt_error = ::boba::norm_difference_inf(optimal_tt.decompress(), tensor);

    if (tt_error > tt_error_tolerance)
    {
      tol_upper_bnd = middle;
      error_ub = tt_error;
    }
    else if (tt_error <= tt_error_tolerance)
    {
      tol_lower_bnd = middle;
      error_lb = tt_error;
    }

    double error_range = error_ub - error_lb;
    double error_range_relative = error_range / tt_error_tolerance;

    double tol_range = tol_upper_bnd - tol_lower_bnd;
    double tol_range_relative = tol_range / middle;

    bool large_error_range = error_range_relative > 0.50;
    bool large_tolerance_range = tol_range_relative > 0.01;

    bool search_criteria_0 = large_error_range and large_tolerance_range;
    bool search_criteria_1 = tt_error > tt_error_tolerance;

    searching = (search_criteria_0) or (search_criteria_1);

    verbose_output(tol_lower_bnd, tol_upper_bnd, error_lb, error_ub);

    iterations++;
    if (iterations > 20)
    {
      boba_error("Tolerance solver stalled.");
    }
  }

  return std::make_pair(tol_lower_bnd, optimal_tt);
}

/**
 * \brief
 * Iterates through the tolerance parameter on the log scale and prints off the compression rate
 */

template <size_t dimension, execution_space space, typename data_t>
void scan_tt_compression_versus_tolerance(
  Tensor<dimension, space, data_t>& tensor)
{
  BOBA_CALI_MARK
  std::cout
    << "Tolerance CR Inf_Error Frob_error Rel_Inf_Error Rel_Frob_Error" << std::endl;
  for (size_t ei = 1; ei < 14 * 3; ei++)
  {
    double svd_tolerance_guess = ::boba::pow(10.0, -double(ei / 3.0));
    auto compressed_tensor = compress_to_TensorTrain(tensor, svd_tolerance_guess);
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
 * Compute the various singular value spectra associated with this tensor train decomposition
 */

template <size_t dimension, execution_space space, typename data_t>
::boba::Matrix<space, data_t> compute_tt_spectra(const boba::Tensor<dimension, space, data_t>& input)
{
  BOBA_CALI_MARK
  checkpoint();
  static_assert(dimension > 1, "compute_tt_spectra not defined for dimension 1");
  ::boba::Matrix<space, data_t> spectra;

  Array<::boba::Vector<space, data_t>, dimension - 1> spectrum;

  ::boba::Array<boba::SVD<space, data_t>, dimension - 1> svd;

  for (size_t d = 0; d < dimension - 1; d++)
  {
    svd[d].tolerance_relative = 1.0e-15;
    svd[d].tolerance_absolute = 0.0;
  }

  boba::Matrix<space, data_t> temp_fold({1, product(input.sizes())});
  temp_fold.rename("folding_matrix");
  temp_fold.reshape(input);

  index_t ranks = 1;
  for (size_t d = 0; d < dimension - 1; d++)
  {
    checkpoint();
    size_t this_size = input.sizes(d);
    auto rows = this_size * ranks;
    auto cols = temp_fold.size() / rows;
    checkpoint();
    temp_fold.reshape({rows, cols});
    svd[d](temp_fold);
    // copy S vector
    spectrum[d] = svd[d].S;
    apply_as_diagonal_right_in_place(svd[d].S, svd[d].U);
    ranks = svd[d].significant_singular_values;
    checkpoint();
    temp_fold = svd[d].V.transpose();
    checkpoint();
  }

  checkpoint();
  index_t largest_size = 1;
  for (size_t d = 0; d < dimension - 1; d++)
  {
    largest_size = max(largest_size, spectrum[d].size());
  }

  checkpoint();
  spectra.resize({largest_size, dimension - 1});
  spectra.fill_with_zeros();
  auto spectra_view = spectra.view();

  checkpoint();
  for (size_t d = 0; d < dimension - 1; d++)
  {
    auto spectrum_view = spectrum[d].view();
    ::boba::loop<space, 1>({spectrum[d].size()},
                           [=] __boba_host_device__(index_t i)
    {
      spectra_view({i, d}) = spectrum_view(i);
    });
  }

  checkpoint();
  return spectra;
}

} // namespace boba

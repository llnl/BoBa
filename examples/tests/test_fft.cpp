// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

//
// Fast Fourier Transform Test Suite
//
// This file validates FFT operations for several BOBA tensor formats:
//   - boba::Tensor
//   - boba::TensorTrain
//   - boba::Tucker
//   - boba::CanonicalPolyadicDecomposition
//   - boba::HierarchicalTucker
//
// For each format, the test applies:
//
//     FFT      : forward transform along a specified mode
//     IFFT     : backward transform along the same mode
//
// and checks that:
//
//     IFFT( FFT(x) )   \approx   x
//
// Additionally, structured formats (e.g., TT, Tucker, CP, and HierarchicalTucker) are verified
// against their full tensor “decompressed” representations.

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  boba_print("Tests for Fast Fourier Transform");

  BOBA_CALI_EXTERNAL_MARK

  checkpoint();

  // parse runtime options
  size_t resolution = 6;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(resolution,
                             "-r",
                             "--resolution",
                             "Tensor sizes.");

  args.parse_check();

  // Common setup for the experiments
  constexpr size_t dimension = 3;
  bool check = true;

  auto sizes = boba::filled_array<dimension>(resolution);
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] += d;
  }

  boba_print("Testing the transform of a boba::Tensor<1>.");
  {
    // Create a random Tensor<1, ...>
    boba::Tensor<1, space, boba::complex<double>> input({resolution});
    input.fill_with_random();

    auto fft_of_input = boba::fft_along_dimension<1, space, double>(input, 0, boba::fft_operation::forward);
    auto ifft_of_fft_of_input = boba::fft_along_dimension<1, space, double>(fft_of_input, 0, boba::fft_operation::backward);

    pass_or_fail(check, boba::norm_difference_inf(ifft_of_fft_of_input, input), 1.0e-03);
  }

  boba_print("Testing the transform of a boba::Vector.");
  {
    // Create a random vector
    boba::Vector<space, boba::complex<double>> input({resolution});
    input.fill_with_random();

    auto fft_of_input = boba::fft_along_dimension<1, space, double>(input, 0, boba::fft_operation::forward);
    auto ifft_of_fft_of_input = boba::fft_along_dimension<1, space, double>(fft_of_input, 0, boba::fft_operation::backward);

    pass_or_fail(check, boba::norm_difference_inf(ifft_of_fft_of_input, input), 1.0e-03);
  }

  boba_print("Testing the transform of a boba::Tensor<2>.");
  {
    // Create a random Tensor<2, ...>
    boba::Tensor<2, space, boba::complex<double>> input({resolution, resolution + 1});
    input.fill_with_random();

    for (size_t d = 0; d < 2; ++d)
    {
      auto fft_of_input = boba::fft_along_dimension<2, space, double>(input, d, boba::fft_operation::forward);
      auto ifft_of_fft_of_input = boba::fft_along_dimension<2, space, double>(fft_of_input, d, boba::fft_operation::backward);

      pass_or_fail(check, boba::norm_difference_inf(ifft_of_fft_of_input, input), 1.0e-03);
    }
  }

  boba_print("Testing the transform of a boba::Matrix.");
  {
    // Create a random matrix
    boba::Matrix<space, boba::complex<double>> input({resolution, resolution + 1});
    input.fill_with_random();

    for (size_t d = 0; d < 2; ++d)
    {
      auto fft_of_input = boba::fft_along_dimension<2, space, double>(input, d, boba::fft_operation::forward);
      auto ifft_of_fft_of_input = boba::fft_along_dimension<2, space, double>(fft_of_input, d, boba::fft_operation::backward);

      pass_or_fail(check, boba::norm_difference_inf(ifft_of_fft_of_input, input), 1.0e-03);
    }
  }

  boba_print("Testing the transform of a boba::Matrix with single column.");
  {
    // Create a random matrix
    boba::Matrix<space, boba::complex<double>> input({resolution, 1});
    input.fill_with_random();

    auto fft_of_input = boba::fft_along_dimension<2, space, double>(input, 0, boba::fft_operation::forward);
    auto ifft_of_fft_of_input = boba::fft_along_dimension<2, space, double>(fft_of_input, 0, boba::fft_operation::backward);

    pass_or_fail(check, boba::norm_difference_inf(ifft_of_fft_of_input, input), 1.0e-03);
  }

  boba_print("Testing the transform of a boba::Matrix with single row.");
  {
    // Create a random matrix
    boba::Matrix<space, boba::complex<double>> input({1, resolution});
    input.fill_with_random();

    auto fft_of_input = boba::fft_along_dimension<2, space, double>(input, 1, boba::fft_operation::forward);
    auto ifft_of_fft_of_input = boba::fft_along_dimension<2, space, double>(fft_of_input, 1, boba::fft_operation::backward);

    pass_or_fail(check, boba::norm_difference_inf(ifft_of_fft_of_input, input), 1.0e-03);
  }

  boba_print("Testing the transform of a boba::Tensor<3>.");
  {
    // Create a random Tensor<3,...>
    boba::Tensor<3, space, boba::complex<double>> input(sizes);
    input.fill_with_random();

    for (size_t d = 0; d < dimension; d++)
    {
      // Having to write out the template is not ideal, but it seems we aren't able to deduce
      // data_t when we have complex<data_t>
      auto fft_of_input = boba::fft_along_dimension<dimension, space, double>(input, d, boba::fft_operation::forward);

      auto ifft_of_fft_of_input = boba::fft_along_dimension<dimension, space, double>(fft_of_input, d, boba::fft_operation::backward);

      pass_or_fail(check, boba::norm_difference_inf(ifft_of_fft_of_input, input), 1.0e-3);
    }
  }

  boba_print("Testing the transform of a boba::TensorTrain.");
  {
    // Create a random TensorTrain corresponding to a Tensor<3,...>
    boba::TensorTrain<3, space, boba::complex<double>> input_tt(sizes);

    auto ranks = boba::filled_array<dimension + 1>(3_z);
    ranks[0] = 1;
    ranks[dimension] = 1;

    for (size_t d = 0; d < dimension; d++)
    {
      ranks[d] += d;
    }

    for (size_t d = 0; d < dimension; d++)
    {
      input_tt.cores[d].resize({ranks[d], sizes[d], ranks[d + 1]});
      input_tt.cores[d].fill_with_random();
    }

    for (size_t d = 0; d < dimension; d++)
    {
      auto input = input_tt.decompress();

      // Forward
      auto fft_of_input_tt = boba::fft_along_dimension<dimension, space, double>(input_tt, d, boba::fft_operation::forward);
      auto fft_of_input = boba::fft_along_dimension<dimension, space, double>(input, d, boba::fft_operation::forward);

      auto error_fft = boba::abs(boba::norm_difference_inf(fft_of_input_tt.decompress(), fft_of_input));
      pass_or_fail(check, error_fft, 1.0e-3);

      // Backward
      auto ifft_of_fft_of_input_tt = boba::fft_along_dimension<dimension, space, double>(fft_of_input_tt, d, boba::fft_operation::backward);
      auto ifft_of_fft_of_input = boba::fft_along_dimension<dimension, space, double>(fft_of_input, d, boba::fft_operation::backward);

      auto error_tt = boba::abs(boba::norm_difference_inf(ifft_of_fft_of_input_tt.decompress(), input_tt.decompress()));
      auto error_tt_v_full = boba::abs(boba::norm_difference_inf(ifft_of_fft_of_input, ifft_of_fft_of_input_tt.decompress()));

      pass_or_fail(check, error_tt, 1.0e-3);
      pass_or_fail(check, error_tt_v_full, 1.0e-3);
    }
  }

  boba_print("Testing the transform of a boba::Tucker.");
  {
    boba::Tucker<3, space, boba::complex<double>> input_tucker(sizes);

    auto ranks = boba::filled_array<dimension>(3_z);

    for (size_t d = 0; d < dimension; d++)
    {
      ranks[d] += d;
    }
    input_tucker.R_core.resize(ranks);
    input_tucker.R_core.fill_with_random();
    for (size_t d = 0; d < dimension; d++)
    {
      input_tucker.cores[d].resize({sizes[d], ranks[d]});
      input_tucker.cores[d].fill_with_random();
    }

    // Test transforms one dimension at a time
    for (size_t d = 0; d < dimension; d++)
    {
      auto input = input_tucker.decompress();

      // Forward
      auto fft_of_input_tucker = boba::fft_along_dimension<dimension, space, double>(input_tucker, d, boba::fft_operation::forward);
      auto fft_of_input = boba::fft_along_dimension<dimension, space, double>(input, d, boba::fft_operation::forward);

      auto error_fft = boba::abs(boba::norm_difference_inf(fft_of_input_tucker.decompress(), fft_of_input));
      pass_or_fail(check, error_fft, 1.0e-3);

      // Backward
      auto ifft_of_fft_of_input_tucker = boba::fft_along_dimension<dimension, space, double>(fft_of_input_tucker, d, boba::fft_operation::backward);
      auto ifft_of_fft_of_input = boba::fft_along_dimension<dimension, space, double>(fft_of_input, d, boba::fft_operation::backward);

      auto error_tucker = boba::abs(boba::norm_difference_inf(ifft_of_fft_of_input_tucker.decompress(), input_tucker.decompress()));
      auto error_tucker_v_full = boba::abs(boba::norm_difference_inf(ifft_of_fft_of_input, ifft_of_fft_of_input_tucker.decompress()));

      pass_or_fail(check, error_tucker, 1.0e-3);
      pass_or_fail(check, error_tucker_v_full, 1.0e-3);
    }
  }

  boba_print("Testing the transform of a boba::CanonicalPolyadicDecomposition.");
  {
    boba::CanonicalPolyadicDecomposition<3, space, boba::complex<double>> input_cpd(sizes);

    auto rank = 3_z;
    input_cpd.m_weights.resize(rank);
    input_cpd.m_weights.fill_with_random();
    for (size_t d = 0; d < dimension; d++)
    {
      input_cpd.m_cores[d].resize({sizes[d], rank});
      input_cpd.m_cores[d].fill_with_random();
    }

    for (size_t d = 0; d < dimension; d++)
    {
      auto input = input_cpd.decompress();

      // Forward
      auto fft_of_input_cpd = boba::fft_along_dimension<dimension, space, double>(input_cpd, d, boba::fft_operation::forward);
      auto fft_of_input = boba::fft_along_dimension<dimension, space, double>(input, d, boba::fft_operation::forward);

      auto error_fft = boba::abs(boba::norm_difference_inf(fft_of_input_cpd.decompress(), fft_of_input));
      pass_or_fail(check, error_fft, 1.0e-3);

      // Backward
      auto ifft_of_fft_of_input_cpd = boba::fft_along_dimension<dimension, space, double>(fft_of_input_cpd, d, boba::fft_operation::backward);
      auto ifft_of_fft_of_input = boba::fft_along_dimension<dimension, space, double>(fft_of_input, d, boba::fft_operation::backward);

      auto error_cpd = boba::abs(boba::norm_difference_inf(ifft_of_fft_of_input_cpd.decompress(), input_cpd.decompress()));
      auto error_cpd_v_full = boba::abs(boba::norm_difference_inf(ifft_of_fft_of_input, ifft_of_fft_of_input_cpd.decompress()));

      pass_or_fail(check, error_cpd, 1.0e-3);
      pass_or_fail(check, error_cpd_v_full, 1.0e-3);
    }
  }

  boba_print("Testing the transform of a boba::HierarchicalTucker.");
  {
    // Create a random HierarchicalTucker corresponding to a Tensor<3,...>
    auto dim_tree = boba::DimensionTree(boba::BalancedTreeBuilder(3));
    boba::HierarchicalTucker<3, space, boba::complex<double>> input_htucker(sizes, dim_tree);
    input_htucker.fill_with_random();
    auto input = input_htucker.decompress();

    // Test transforms one dimension at a time
    for (size_t d = 0; d < dimension; ++d)
    {
      // Forward
      auto fft_of_input_htucker = boba::fft_along_dimension<dimension, space, double>(input_htucker, d, boba::fft_operation::forward);
      auto fft_of_input = boba::fft_along_dimension<dimension, space, double>(input, d, boba::fft_operation::forward);
      auto error_fft = boba::abs(boba::norm_difference_inf(fft_of_input_htucker.decompress(), fft_of_input));
      pass_or_fail(check, error_fft, 1.0e-3);

      // Backward
      auto ifft_of_fft_of_input_htucker = boba::fft_along_dimension<dimension, space, double>(fft_of_input_htucker, d, boba::fft_operation::backward);
      auto ifft_of_fft_of_input = boba::fft_along_dimension<dimension, space, double>(fft_of_input, d, boba::fft_operation::backward);

      auto error_htucker = boba::abs(boba::norm_difference_inf(ifft_of_fft_of_input_htucker.decompress(), input));
      auto error_htucker_v_full = boba::abs(boba::norm_difference_inf(ifft_of_fft_of_input, ifft_of_fft_of_input_htucker.decompress()));
      pass_or_fail(check, error_htucker, 1.0e-3);
      pass_or_fail(check, error_htucker_v_full, 1.0e-3);
    }
  }

  boba::finalize();
  return final_check(check);
}

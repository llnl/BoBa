// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <vector>

/*
  Tests the CanonicalPolyadicDecomposition class
*/

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

template <size_t dimension>
void run_test(size_t size_base, size_t rank, bool& check)
{
  checkpoint();
  boba_print("");
  boba_print("-------------------------------------------");
  boba_print(dimension);
  boba_print(rank);
  auto sizes = boba::filled_array<dimension>(size_base);

  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] += 2 * d;
  }

  boba_print(sizes);
  checkpoint();
  boba::Tensor<dimension, space, double> tensor(sizes);

  auto tensor_view = tensor.view();

  ::boba::TicToc<tictoc_units> timer;

  timer.tic();
  // Fill tensor with values
  checkpoint();
  ::boba::loop<space, 1>(tensor_view.size(),
                         [=] __boba_host_device__(size_t I)
  {
    auto mid = tensor_view.multiindex(I);
    auto N = tensor_view.size();
    double value = 1.0;
    for (size_t d1 = 0; d1 < dimension; d1++)
    {
      for (size_t d2 = 0; d2 < dimension; d2++)
      {
        value += mid[d1] * mid[d2] / static_cast<double>(N * N);
      }
    }

    tensor_view(I) = value;
  });
  timer.end_and_print("tensor generation");

  timer.tic();
  {
    checkpoint();
    boba::CanonicalPolyadicDecomposition<dimension, space, double> cpd(tensor.sizes());

    cpd.ALS_tolerance_absolute = 1.0e-07;
    cpd.ALS_tolerance_relative = 1.0e-07;

    checkpoint();
    auto iterations_taken = cpd.compress(tensor, rank);

    checkpoint();
    auto cpd_before_normalization = cpd.decompress();

    checkpoint();
    cpd.normalize_factors_into_weights();

    checkpoint();
    auto cpd_decompress = cpd.decompress();

    checkpoint();
    auto normalization_error = boba::norm_difference_frobenius(cpd_before_normalization, cpd_decompress);
    auto normalization_error_relative = normalization_error / ::boba::norm_frobenius(tensor);

    checkpoint();
    auto error_l2 = boba::norm_difference_frobenius(cpd_decompress, tensor);
    auto error_l2_relative = error_l2 / ::boba::norm_frobenius(tensor);

    boba_print(normalization_error);
    boba_print(normalization_error_relative);
    boba_print(error_l2);
    boba_print(error_l2_relative);
    boba_print(iterations_taken);

    auto tolerance_modifier = pow(10.0, static_cast<double>(dimension));
    pass_or_fail(check, normalization_error_relative, 1.0e-12 * tolerance_modifier);
    pass_or_fail(check, error_l2_relative, 1.0e-7 * tolerance_modifier);
  }
  timer.end_and_print("compression and error checking");
}

/**
 * \brief
 * Driver of the Canonical Polyadic (CP) decomposition
 */

int main(int argc, char* argv[])
{
  boba::splash();
  boba::init();

  size_t rank = 7;
  size_t size_base = 7;
  bool check = true;
  size_t dimension = 0;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(rank,
                             "-r",
                             "--rank",
                             "CP decomposition Rank.");

  args.add_optional_argument(size_base,
                             "-s",
                             "--size_base",
                             "Relative size of tensor extents for test.");

  args.add_optional_argument(dimension,
                             "-D",
                             "--dimension",
                             "Dimensionality of CPD example.");

  args.parse_check();

  if ((dimension == 1) or (dimension == 0))
  {
    run_test<1>(size_base, rank, check);
  }
  if ((dimension == 2) or (dimension == 0))
  {
    run_test<2>(size_base, rank, check);
  }
  if ((dimension == 3) or (dimension == 0))
  {
    run_test<3>(size_base, rank, check);
  }
  if ((dimension == 4) or (dimension == 0))
  {
    run_test<4>(size_base, rank, check);
  }
  if ((dimension == 5) or (dimension == 0))
  {
    run_test<5>(size_base, rank, check);
  }
  if ((dimension == 6) or (dimension == 0))
  {
    run_test<6>(size_base, rank, check);
  }

  boba::finalize();
  return final_check(check);
}

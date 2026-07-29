// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "common.hpp"

/*
  Test BoBa's sum_and_round container interfaces
*/

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

template <typename format_t>
void check_sum_and_round_interfaces(
  bool& check,
  format_t const& A,
  format_t const& reference_result,
  double tol = 1.0e-4)
{
  //
  // (1) Using a std::initializer_list
  //
  {
    boba_print("Checking sum_and_round(): std::initializer_list.");

    std::initializer_list<format_t> inputs{A, 2.0 * A, 3.0 * A};
    auto result_from_il = boba::sum_and_round(inputs);

    pass_or_fail(
      check,
      norm_difference_frobenius(
        result_from_il.decompress(),
        reference_result.decompress()),
      tol);
  }

  //
  // (2) Using a std::vector
  //
  {
    boba_print("Checking sum_and_round(): std::vector.");

    std::vector<format_t> inputs{A, 2.0 * A, 3.0 * A};
    auto result_from_vector = boba::sum_and_round(inputs);

    pass_or_fail(
      check,
      norm_difference_frobenius(
        result_from_vector.decompress(),
        reference_result.decompress()),
      tol);
  }

  //
  // (3) Using a std::span of const entries
  //
  {
    boba_print("Checking sum_and_round(): std::span of const entries.");

    std::vector<format_t> inputs{A, 2.0 * A, 3.0 * A};
    std::span<const format_t> inputs_as_span{inputs};

    auto result_from_span = boba::sum_and_round(inputs_as_span);

    pass_or_fail(
      check,
      norm_difference_frobenius(
        result_from_span.decompress(),
        reference_result.decompress()),
      tol);
  }
}

/*
  Tests sum_and_round procedure
*/

int main(int argc, char* argv[])
{
  boba::splash();
  boba::init();
  boba_print("Tests sum_and_round procedure");

  BOBA_CALI_EXTERNAL_MARK

  size_t resolution = 8_z;

  checkpoint();
  ::boba::argparser args(argc, argv);

  args.add_optional_argument(
    resolution,
    "-r",
    "--resolution",
    "Tensor sizes.");

  args.parse_check();

  constexpr size_t dimension = 3_z;
  bool check = true;

  auto sizes = boba::filled_array<dimension>(resolution);
  for (size_t d = 0_z; d < dimension; d++)
  {
    sizes[d] += d;
  }

  //
  // Tensor train sum_and_round test
  //
  {
    checkpoint();
    boba_print("Checking TensorTrain sum_and_round().");

    boba::TensorTrain<dimension, space, double> tt(sizes);

    auto R1 = 2_z;
    auto R2 = 3_z;

    tt.cores[0].resize({1, sizes[0], R1});
    tt.cores[1].resize({R1, sizes[1], R2});
    tt.cores[2].resize({R2, sizes[2], 1});

    for (size_t d = 0; d < dimension; d++)
    {
      tt.cores[d].fill_with_random();
    }

    auto reference_result = 6.0 * tt;

    check_sum_and_round_interfaces(check, tt, reference_result);
  }

  //
  // Quantized tensor train sum_and_round test
  //
  {
    checkpoint();
    boba_print("Checking QuantizedTensorTrain sum_and_round().");

    size_t base = 2;
    size_t exponent = 4;

    boba::QuantizedTensorTrain<space, double> qtt(base, exponent);

    auto R1 = 2_z;
    auto R2 = 3_z;
    auto R3 = 2_z;

    qtt.cores[0].resize({1, base, R1});
    qtt.cores[1].resize({R1, base, R2});
    qtt.cores[2].resize({R2, base, R3});
    qtt.cores[3].resize({R3, base, 1});

    for (size_t d = 0; d < exponent; d++)
    {
      qtt.cores[d].fill_with_random();
    }

    auto reference_result = 6.0 * qtt;

    check_sum_and_round_interfaces(check, qtt, reference_result);
  }

  //
  // Tucker sum_and_round test
  //
  {
    checkpoint();
    boba_print("Checking Tucker sum_and_round().");

    boba::Tucker<dimension, space, double> tuck(sizes);

    auto R1 = 2_z;
    auto R2 = 3_z;
    auto R3 = 4_z;

    tuck.cores[0].resize({sizes[0], R1});
    tuck.cores[1].resize({sizes[1], R2});
    tuck.cores[2].resize({sizes[2], R3});
    tuck.R_core.resize({R1, R2, R3});

    for (size_t d = 0; d < dimension; d++)
    {
      tuck.cores[d].fill_with_random();
    }
    tuck.R_core.fill_with_random();

    auto reference_result = 6.0 * tuck;

    check_sum_and_round_interfaces(check, tuck, reference_result);
  }

  //
  // Hierarchical Tucker sum_and_round test
  //
  {
    checkpoint();
    boba_print("Checking HierarchicalTucker sum_and_round().");

    constexpr size_t ht_dimension = 4;

    auto dim_tree = boba::DimensionTree(
      boba::BalancedTreeBuilder(ht_dimension));

    auto ht_sizes = boba::filled_array<ht_dimension>(12_z);

    boba::HierarchicalTucker<ht_dimension, space, double> ht(
      ht_sizes,
      dim_tree);

    ht.fill_with_random();

    auto reference_result = 6.0 * ht;

    check_sum_and_round_interfaces(check, ht, reference_result);
  }

  boba::finalize();
  return final_check(check);
}
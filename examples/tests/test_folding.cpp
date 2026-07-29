// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "common.hpp"

#include <BOBA/boba.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>

/*
  Tests BoBa's tensor fold/unfold operations
  https://www.alexejgossmann.com/tensor_decomposition_tucker/
*/

constexpr boba::execution_space space = boba::default_execution_space;

int main(int argc, char* argv[])
{

  boba::detail::ignore(argc);
  boba::detail::ignore(argv);

  boba::splash();
  boba_print("Tests for tensor folding/unfolding.");
  boba::init();

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;
  checkpoint();

  // Allocate space for unfoldings
  ::boba::Tensor<3, space, double> original({2, 3, 4});
  ::boba::Matrix<space, double> known_right_unfold_0({2, 12});
  ::boba::Matrix<space, double> known_right_unfold_1({3, 8});
  ::boba::Matrix<space, double> known_right_unfold_2({4, 6});

  ::boba::Matrix<space, double> known_left_unfold_0({12, 2});
  ::boba::Matrix<space, double> known_left_unfold_1({8, 3});
  ::boba::Matrix<space, double> known_left_unfold_2({6, 4});

  // Create views
  auto original_view = original.view();

  auto known_right_unfold_0_view = known_right_unfold_0.view();
  auto known_right_unfold_1_view = known_right_unfold_1.view();
  auto known_right_unfold_2_view = known_right_unfold_2.view();

  auto known_left_unfold_0_view = known_left_unfold_0.view();
  auto known_left_unfold_1_view = known_left_unfold_1.view();
  auto known_left_unfold_2_view = known_left_unfold_2.view();

  // Fill the unfoldings manually
  ::boba::loop<space, 3>(original.sizes(),
                         [=] __boba_host_device__(::boba::Array<size_t, 3> ijk)
  {
    auto [i, j, k] = ijk;

    double value = static_cast<double>(original_view.index(ijk));

    auto [I, J, K] = original_view.sizes();
    boba::detail::ignore(K);

    original_view({i, j, k}) = value;

    known_right_unfold_0_view({i, j + k * J}) = value;
    known_right_unfold_1_view({j, i + k * I}) = value;
    known_right_unfold_2_view({k, i + j * I}) = value;

    known_left_unfold_0_view({j + k * J, i}) = value;
    known_left_unfold_1_view({i + k * I, j}) = value;
    known_left_unfold_2_view({i + j * I, k}) = value;
  });

  // Right unfolding and folding operations
  {
    checkpoint();
    auto right_unfold_0 = ::boba::unfold(original, 0_z);
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(right_unfold_0, known_right_unfold_0), 1.0e-13);

    checkpoint();
    auto right_unfold_1 = ::boba::unfold(original, 1_z);
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(right_unfold_1, known_right_unfold_1), 1.0e-13);

    checkpoint();
    auto right_unfold_2 = ::boba::unfold(original, 2_z);
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(right_unfold_2, known_right_unfold_2), 1.0e-13);

    checkpoint();
    auto right_fold_0 = ::boba::fold(right_unfold_0, original.sizes(), 0_z);
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(right_fold_0, original), 1.0e-13);

    checkpoint();
    auto right_fold_1 = ::boba::fold(right_unfold_1, original.sizes(), 1_z);
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(right_fold_1, original), 1.0e-13);

    checkpoint();
    auto right_fold_2 = ::boba::fold(right_unfold_2, original.sizes(), 2_z);
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(right_fold_2, original), 1.0e-13);
  }

  // Left unfolding and folding operations
  {
    checkpoint();
    auto left_unfold_0 = ::boba::unfold(original, std::vector<size_t>{1, 2}, std::vector<size_t>{0});
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(left_unfold_0, known_left_unfold_0), 1.0e-13);

    checkpoint();
    auto left_unfold_1 = ::boba::unfold(original, std::vector<size_t>{0, 2}, std::vector<size_t>{1});
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(left_unfold_1, known_left_unfold_1), 1.0e-13);

    checkpoint();
    auto left_unfold_2 = ::boba::unfold(original, std::vector<size_t>{0, 1}, std::vector<size_t>{2});
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(left_unfold_2, known_left_unfold_2), 1.0e-13);

    checkpoint();
    auto left_fold_0 = ::boba::fold(left_unfold_0, original.sizes(), std::vector<size_t>{1, 2});
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(left_fold_0, original), 1.0e-13);

    checkpoint();
    auto left_fold_1 = ::boba::fold(left_unfold_1, original.sizes(), std::vector<size_t>{0, 2});
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(left_fold_1, original), 1.0e-13);

    checkpoint();
    auto left_fold_2 = ::boba::fold(left_unfold_2, original.sizes(), std::vector<size_t>{0, 1});
    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(left_fold_2, original), 1.0e-13);
  }

  // TT core foldings
  {
    checkpoint();
    auto unfold_left = ::boba::compute_unfold_left(original);
    checkpoint();
    auto unfold_right = ::boba::compute_unfold_right(original);

    checkpoint();
    auto unfolded_left = ::boba::write_to_core_from_left_fold(unfold_left, original.sizes(1));
    checkpoint();
    auto unfolded_right = ::boba::write_to_core_from_right_fold(unfold_right, original.sizes(1));

    pass_or_fail(check, ::boba::norm_difference_inf(unfolded_left, original), 1.0e-13);
    pass_or_fail(check, ::boba::norm_difference_inf(unfolded_right, original), 1.0e-13);

    checkpoint();
    auto unfold_left_from_right = ::boba::compute_unfold_left_from_unfold_right(unfold_right, original.sizes(1));
    checkpoint();
    auto unfold_right_from_left = ::boba::compute_unfold_right_from_unfold_left(unfold_left, original.sizes(1));

    checkpoint();
    pass_or_fail(check, ::boba::norm_difference_inf(unfold_left, unfold_left_from_right), 1.0e-13);
    pass_or_fail(check, ::boba::norm_difference_inf(unfold_right, unfold_right_from_left), 1.0e-13);
  }

  boba::finalize();
  return final_check(check);
}

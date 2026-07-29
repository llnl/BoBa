// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "common.hpp"

#include <BOBA/boba.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>

/*
  Test BoBa's subtensorview capability
*/

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::microseconds;

int main(int argc, char* argv[])
{

  boba::splash();
  std::cout << "Tests for tensor operations." << std::endl;
  boba::init();

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;
  checkpoint();

  size_t max_power = 2;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(max_power,
                             "-p",
                             "--power",
                             "Parameters will scaled by 10^power.");

  args.parse_check();

  size_t N = 7;

  using data_t = double;

  ::boba::Array<size_t, 3> sizes{N, N, N};
  ::boba::Tensor<3, space, data_t> tensor_A(sizes);
  ::boba::Tensor<3, space, data_t> tensor_B(sizes);

  ::boba::Array<size_t, 4> new_sizes{sizes[0], sizes[2], sizes[0], sizes[2]};
  ::boba::Tensor<4, space, data_t> tensor_C(new_sizes);

  {
    auto tensor_A_view = tensor_A.view();
    auto tensor_B_view = tensor_B.view();

    ::boba::detail::loop<space>(
      0, tensor_A_view.size(), [=] __boba_host_device__(size_t index)
    {
      tensor_A_view(index) = index;
      tensor_B_view(index) = ::boba::mod(index, 3) + 1;
    });
  }

  ::boba::TicToc<tictoc_units> timer;

  //
  // Compute full contract
  //
  boba_print("Compute full contract");

  timer.tic();
#ifdef BOBA_CUTENSOR
  ::boba::detail::cutensor_contract(tensor_A.const_view(), tensor_B.const_view(), tensor_C.view(), 1, 1);
#endif
#ifdef BOBA_HIPTENSOR
  ::boba::detail::hiptensor_contract(tensor_A.const_view(), tensor_B.const_view(), tensor_C.view(), 1, 1);
#endif
  if constexpr (boba::is_host(space))
  {
    ::boba::detail::eigen_tensor_contract(tensor_A.const_view(), tensor_B.const_view(), tensor_C.view(), 1, 1);
  }
  auto time_full = timer.toc();

  auto tensor_C_view = tensor_C.const_view();

  //
  // Compute naive subtensor view contract
  //
  boba_print("Compute naive subtensorview contract");
  boba_always_assert_nonnegative(int(N / 2) - 1, "Slice would be out of bounds");
  boba_always_assert_lt(N / 2 + 1, N, "Slice would be out of bounds");
  boba_always_assert_nonnegative(int(3 * N / 4) - 1, "Slice would be out of bounds");
  boba_always_assert_lt(3 * N / 4 + 1, N, "Slice would be out of bounds");

  ::boba::Array<size_t, 3> lower_bound{N / 2 - 1, 0, 3 * N / 4 - 1};
  ::boba::Array<size_t, 3> upper_bound{N / 2 + 1, N, 3 * N / 4 + 1};
  auto sv_sizes = upper_bound - lower_bound;
  ::boba::Array<size_t, 4> new_sv_sizes{sv_sizes[0], sv_sizes[2], sv_sizes[0], sv_sizes[2]};
  ::boba::Array<size_t, 4> new_sv_lower_bound{lower_bound[0], lower_bound[2], lower_bound[0], lower_bound[2]};

  auto tensor_A_sv = boba::make_subtensor_const_view(tensor_A, lower_bound, upper_bound);
  auto tensor_B_sv = boba::make_subtensor_const_view(tensor_B, lower_bound, upper_bound);

  ::boba::Tensor<4, space, data_t> tensor_C_naive(new_sv_sizes);
  timer.tic();
  {
    auto tensor_C_naive_view = tensor_C_naive.view();
    auto contraction_length = tensor_A.sizes(1);

    ::boba::detail::loop<space>(
      0, tensor_C_naive_view.size(), [=] __boba_host_device__(size_t index)
    {
      auto mid = tensor_C_naive_view.multiindex(index);
      double value_C = 0.0;
      for (size_t k = 0; k < contraction_length; k++)
      {
        auto value_A = tensor_A_sv({mid[0], k, mid[1]});
        auto value_B = tensor_B_sv({mid[2], k, mid[3]});
        value_C += value_A * value_B;
      }
      tensor_C_naive_view(index) = value_C;
    });
  }
  auto time_naive = timer.toc();

  //
  // Compute subtensor view contract
  //
  boba_print("Computing subtensor view contract");

  ::boba::Tensor<4, space, data_t> tensor_C_sv(new_sv_sizes);

  timer.tic();
#ifdef BOBA_CUTENSOR
  ::boba::detail::cutensor_contract(tensor_A_sv, tensor_B_sv, tensor_C_sv.view(), 1, 1);
#endif
#ifdef BOBA_HIPTENSOR
  ::boba::detail::hiptensor_contract(tensor_A_sv, tensor_B_sv, tensor_C_sv.view(), 1, 1);
#endif
  if constexpr (boba::is_host(space))
  {
    auto subtensor_C_sv = boba::make_subtensor_view(tensor_C_sv, boba::filled_array<4>(0_z), new_sv_sizes);
    ::boba::detail::eigen_tensor_contract(tensor_A_sv, tensor_B_sv, subtensor_C_sv, 1, 1);
  }
  auto time_subtensorview = timer.toc();

  //
  // Validate subtensor view contract
  //
  {
    auto tensor_C_sv_view = tensor_C_sv.const_view();

    data_t error_inf_norm = ::boba::lowest_value<data_t>();
    size_t max_error_index = 0;

    ::boba::max_loc_reduce<space>(
      error_inf_norm, max_error_index, 0_z, tensor_C_sv.size(), [=] __boba_host_device__(::boba::reducer_index_t flat_index, ::boba::max_loc_reducer_operator<data_t> & local_error)
    {
      const size_t flat_index_size = static_cast<size_t>(flat_index);
      auto value_sv = tensor_C_sv_view(flat_index_size);

      auto sv_mid = tensor_C_sv_view.multiindex(flat_index_size);
      auto value_full = tensor_C_view(sv_mid + new_sv_lower_bound);

      auto error_loc = ::boba::abs(value_sv - value_full);

      local_error.maxloc(error_loc, flat_index);
    });

    pass_or_fail(check, error_inf_norm, 1.0e-13);
  }

  //
  // Validate naive implementation
  //
  {
    auto tensor_C_sv_naive_view = tensor_C_naive.const_view();

    data_t error_inf_norm_naive = ::boba::lowest_value<data_t>();
    size_t max_error_index_naive = 0;

    ::boba::max_loc_reduce<space>(
      error_inf_norm_naive, max_error_index_naive, 0_z, tensor_C_sv.size(), [=] __boba_host_device__(::boba::reducer_index_t flat_index, ::boba::max_loc_reducer_operator<data_t> & local_error)
    {
      const size_t flat_index_size = static_cast<size_t>(flat_index);
      auto value_sv = tensor_C_sv_naive_view(flat_index_size);

      auto sv_mid = tensor_C_sv_naive_view.multiindex(flat_index_size);
      auto value_full = tensor_C_view(sv_mid + new_sv_lower_bound);

      auto error_loc = ::boba::abs(value_sv - value_full);

      local_error.maxloc(error_loc, flat_index);
    });

    pass_or_fail(check, error_inf_norm_naive, 1.0e-13);
  }

  std::cout << "Subtensor view contraction;   "
            << ::boba::detail::execution_space_name(space) << "; "
            << tensor_A.sizes() << ","
            << tensor_B.sizes() << "->"
            << tensor_C.sizes() << "; "
            << std::endl;

  std::cout << "time_full    : " << time_full << " " << timer.units_string << "\n"
            << "time_naive   : " << time_naive << " " << timer.units_string << "\n"
            << "time_subtensorview : " << time_subtensorview << " " << timer.units_string << "\n";

  boba::finalize();
  return final_check(check);
}

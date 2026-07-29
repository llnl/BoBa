// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Tests BoBa's implementation of the TT-Cross algorithm
*/

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::execution_space host_space = ::boba::host_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

boba::Vector<space, double> make_one(size_t N)
{
  ::boba::Vector<space, double> one({N});
  one.fill_with(1.0);
  return one;
}

boba::Vector<space, double> make_i_vec(size_t N)
{
  ::boba::Vector<space, double> i_vec({N});
  auto i_vec_view = i_vec.view();
  ::boba::detail::loop<boba::default_execution_space>(0, N, [=] __boba_host_device__(size_t i)
  {
    i_vec_view(i) = static_cast<double>(i);
  });
  return i_vec;
}

int main(int argc, char* argv[])
{

  boba::detail::ignore(argc);
  boba::detail::ignore(argv);
  boba::splash();
  boba::init();
  boba_print("Tests for BoBa's DMRGCross implementations");

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;

  size_t N = 4;

  boba::Array<size_t, 3> sizes{N, N + 7, N + 3};

  // Baseline
  // Make a function f = f(i, j, k) = i + j + k
  // Fill tensor with function
  auto function = [=](::boba::Array<size_t, 3> ijk)
  {
    return ::boba::sum(ijk);
  };
  ::boba::Tensor<3, host_space, double> function_tensor(sizes);
  {
    auto function_tensor_view = function_tensor.view();

    ::boba::detail::loop<boba::default_execution_space>(0, function_tensor_view.size(), [=] __boba_host_device__(size_t I)
    {
      auto mid = function_tensor_view.multiindex(I);
      function_tensor_view(I) = function(mid);
    });
  }
  auto function_norm = ::boba::norm_inf(function_tensor);

  // Example 1 - MAXVOL (default)
  // Make a function f = f(i, j, k) = i + j + k
  // Use cross to make tt approximating f
  {
    std::cout << "\n=== Example 1: DMRGCross with MAXVOL (default) ===" << std::endl;

    ::boba::TensorTrain<3, host_space, double> tt_initial_guess(sizes);
    tt_initial_guess.fill_with_zeros();

    checkpoint();
    tt_initial_guess.cores[0].resize({1, sizes[0], 1});
    tt_initial_guess.cores[1].resize({1, sizes[1], 3});
    tt_initial_guess.cores[2].resize({3, sizes[2], 1});
    tt_initial_guess.cores[0].fill_with_random();
    tt_initial_guess.cores[1].fill_with_random();
    tt_initial_guess.cores[2].fill_with_random();

    boba::DMRGCross<double> make_tt_from_arbitrary_function;
    // Set to MAXVOL (if unset, this is currently the default)
    make_tt_from_arbitrary_function.submatrix_selection_type =
      boba::DMRGCross<double>::SubmatrixSelectionType::MAXVOL;

    auto ytt = make_tt_from_arbitrary_function.apply(tt_initial_guess, function);
    ytt.rename("ytt");
    auto ytt_decompress = ytt.decompress();
    auto ytt_decompress_view = ytt_decompress.const_view();

    // Validate ytt
    double error = ::boba::lowest_value<double>();
    size_t index = 0;

    ::boba::max_loc_reduce<boba::default_execution_space>(
      error, index, 0_z, ytt_decompress_view.size(), [=] __boba_host_device__(::boba::reducer_index_t index_A, ::boba::max_loc_reducer_operator<double> & local_error)
    {
      const size_t index_A_size = static_cast<size_t>(index_A);
      auto mid_A = ytt_decompress_view.multiindex(index_A_size);

      double expected = function(mid_A);
      double computed = ytt_decompress_view(index_A_size);
      double error_loc = ::boba::abs(computed - expected);
      local_error.maxloc(error_loc, index_A);
    });

    double error_inf_norm = error / function_norm;

    std::cout << "MAXVOL Error (inf norm): " << error_inf_norm << std::endl;
    pass_or_fail(check, error_inf_norm, 1.0e-9);
  }

  // Example 2 - MAXVOL (default)
  // Make a function f = f(x) = x
  // Make a tt xtt = i + j + k
  // Use cross to make a tensor train approximating f(xtt)
  {
    std::cout << "\n=== Example 2: DMRGCross with MAXVOL on tensor train input ===" << std::endl;

    ::boba::TensorTrain<3, host_space, double> tt_initial_guess(sizes);
    tt_initial_guess.fill_with_zeros();

    tt_initial_guess.cores[0].resize({1, sizes[0], 2});
    tt_initial_guess.cores[1].resize({2, sizes[1], 3});
    tt_initial_guess.cores[2].resize({3, sizes[2], 1});

    tt_initial_guess.cores[0].fill_with_random();
    tt_initial_guess.cores[1].fill_with_random();
    tt_initial_guess.cores[2].fill_with_random();

    auto input_tt =
      ::boba::make_tt_from_vectors<3, space, double>({make_i_vec(sizes[0]), make_one(sizes[1]), make_one(sizes[2])}) + ::boba::make_tt_from_vectors<3, space, double>({make_one(sizes[0]), make_i_vec(sizes[1]), make_one(sizes[2])}) + ::boba::make_tt_from_vectors<3, space, double>({make_one(sizes[0]), make_one(sizes[1]), make_i_vec(sizes[2])});

    boba::DynamicTensorTrainView<3, double> input_tt_view(input_tt);

    auto cross_func = [=](::boba::Array<size_t, 3> ijk)
    {
      return input_tt_view.unroll_value(ijk);
    };

    boba::DMRGCross<double> apply_tt_to_arbitrary_function;
    apply_tt_to_arbitrary_function.submatrix_selection_type =
      boba::DMRGCross<double>::SubmatrixSelectionType::MAXVOL;

    auto ytt = apply_tt_to_arbitrary_function.apply(tt_initial_guess, cross_func);
    ytt.rename("ytt");
    auto ytt_decompress = ytt.decompress();
    auto ytt_decompress_view = ytt_decompress.const_view();

    // Validate ytt
    double error_inf_norm = ::boba::lowest_value<double>();
    size_t index = 0;

    ::boba::max_loc_reduce<boba::default_execution_space>(
      error_inf_norm, index, 0_z, ytt_decompress_view.size(), [=] __boba_host_device__(::boba::reducer_index_t index_A, ::boba::max_loc_reducer_operator<double> & local_error)
    {
      const size_t index_A_size = static_cast<size_t>(index_A);
      auto mid_A = ytt_decompress_view.multiindex(index_A_size);
      double expected = boba::sum(mid_A);
      double computed = ytt_decompress_view(index_A_size);
      double error_loc = ::boba::abs(computed - expected);
      local_error.maxloc(error_loc, index_A);
    });

    error_inf_norm /= function_norm;

    std::cout << "MAXVOL Error (inf norm): " << error_inf_norm << std::endl;
    pass_or_fail(check, error_inf_norm, 1.0e-9);
  }

  // Example 3 - DEIM
  // Make a function f = f(i, j, k) = i + j + k
  // Use cross with DEIM to make tt approximating f
  {
    std::cout << "\n=== Example 3: DMRGCross with DEIM ===" << std::endl;

    ::boba::TensorTrain<3, host_space, double> tt_initial_guess(sizes);
    tt_initial_guess.fill_with_zeros();

    checkpoint();
    tt_initial_guess.cores[0].resize({1, sizes[0], 1});
    tt_initial_guess.cores[1].resize({1, sizes[1], 3});
    tt_initial_guess.cores[2].resize({3, sizes[2], 1});
    tt_initial_guess.cores[0].fill_with_random();
    tt_initial_guess.cores[1].fill_with_random();
    tt_initial_guess.cores[2].fill_with_random();

    boba::DMRGCross<double> make_tt_from_arbitrary_function;
    // Set to DEIM
    make_tt_from_arbitrary_function.submatrix_selection_type =
      boba::DMRGCross<double>::SubmatrixSelectionType::DEIM;

    auto ytt = make_tt_from_arbitrary_function.apply(tt_initial_guess, function);
    ytt.rename("ytt_deim");
    auto ytt_decompress = ytt.decompress();
    auto ytt_decompress_view = ytt_decompress.const_view();

    // Validate ytt
    double error = ::boba::lowest_value<double>();
    size_t index = 0;

    ::boba::max_loc_reduce<boba::default_execution_space>(
      error, index, 0_z, ytt_decompress_view.size(), [=] __boba_host_device__(::boba::reducer_index_t index_A, ::boba::max_loc_reducer_operator<double> & local_error)
    {
      const size_t index_A_size = static_cast<size_t>(index_A);
      auto mid_A = ytt_decompress_view.multiindex(index_A_size);

      double expected = function(mid_A);
      double computed = ytt_decompress_view(index_A_size);
      double error_loc = ::boba::abs(computed - expected);
      local_error.maxloc(error_loc, index_A);
    });

    double error_inf_norm = error / function_norm;

    std::cout << "DEIM Error (inf norm): " << error_inf_norm << std::endl;
    pass_or_fail(check, error_inf_norm, 1.0e-9);
  }

  // Example 4 - DEIM
  // Make a function f = f(x) = x
  // Make a tt xtt = i + j + k
  // Use cross with DEIM to make a tensor train approximating f(xtt)
  {
    std::cout << "\n=== Example 4: DMRGCross with DEIM on tensor train input ===" << std::endl;

    ::boba::TensorTrain<3, host_space, double> tt_initial_guess(sizes);
    tt_initial_guess.fill_with_zeros();

    tt_initial_guess.cores[0].resize({1, sizes[0], 2});
    tt_initial_guess.cores[1].resize({2, sizes[1], 3});
    tt_initial_guess.cores[2].resize({3, sizes[2], 1});

    tt_initial_guess.cores[0].fill_with_random();
    tt_initial_guess.cores[1].fill_with_random();
    tt_initial_guess.cores[2].fill_with_random();

    auto input_tt =
      ::boba::make_tt_from_vectors<3, space, double>({make_i_vec(sizes[0]), make_one(sizes[1]), make_one(sizes[2])}) + ::boba::make_tt_from_vectors<3, space, double>({make_one(sizes[0]), make_i_vec(sizes[1]), make_one(sizes[2])}) + ::boba::make_tt_from_vectors<3, space, double>({make_one(sizes[0]), make_one(sizes[1]), make_i_vec(sizes[2])});

    boba::DynamicTensorTrainView<3, double> input_tt_view(input_tt);

    auto cross_func = [=](::boba::Array<size_t, 3> ijk)
    {
      return input_tt_view.unroll_value(ijk);
    };

    boba::DMRGCross<double> apply_tt_to_arbitrary_function;
    // Set to DEIM
    apply_tt_to_arbitrary_function.submatrix_selection_type =
      boba::DMRGCross<double>::SubmatrixSelectionType::DEIM;

    auto ytt = apply_tt_to_arbitrary_function.apply(tt_initial_guess, cross_func);
    ytt.rename("ytt_deim");
    auto ytt_decompress = ytt.decompress();
    auto ytt_decompress_view = ytt_decompress.const_view();

    // Validate ytt
    double error_inf_norm = ::boba::lowest_value<double>();
    size_t index = 0;

    ::boba::max_loc_reduce<boba::default_execution_space>(
      error_inf_norm, index, 0_z, ytt_decompress_view.size(), [=] __boba_host_device__(::boba::reducer_index_t index_A, ::boba::max_loc_reducer_operator<double> & local_error)
    {
      const size_t index_A_size = static_cast<size_t>(index_A);
      auto mid_A = ytt_decompress_view.multiindex(index_A_size);
      double expected = boba::sum(mid_A);
      double computed = ytt_decompress_view(index_A_size);
      double error_loc = ::boba::abs(computed - expected);
      local_error.maxloc(error_loc, index_A);
    });

    error_inf_norm /= function_norm;

    std::cout << "DEIM Error (inf norm): " << error_inf_norm << std::endl;
    pass_or_fail(check, error_inf_norm, 1.0e-9);
  }

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Both MAXVOL and DEIM submatrix selection algorithms tested successfully" << std::endl;

  boba::finalize();
  return final_check(check);
}

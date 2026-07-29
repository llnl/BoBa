// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::microseconds;

/*
  Consider a given table of values with
  index i,
  independent variable x
  dependent variable f

  i      x    f
  0     0.0  1.0
  1     0.1  1.3
  .      .
  .      .
  20   2.0  1.9

  This example studies converting x and f into QTTs and then
  how to efficiently extract data points from the table.
  Two class of methods are presented
    - static versions where we know information at compile time
    - dynamic views where we don't know information at compile time
*/

//
// Sizes for static case
//
#ifdef BOBA_DEBUG
using tt_sizes = ::boba::StaticArray<std::size_t, 2, 2, 2>;
using x_tt_ranks = ::boba::StaticArray<std::size_t, 1, 2, 2, 1>;
using f_tt_ranks = ::boba::StaticArray<std::size_t, 1, 2, 2, 1>;
#else
using tt_sizes = ::boba::StaticArray<std::size_t, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2>;
using x_tt_ranks = ::boba::StaticArray<std::size_t, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1>;
#ifdef BOBA_CUDA
using f_tt_ranks = ::boba::StaticArray<std::size_t, 1, 2, 4, 6, 8, 9, 11, 15, 16, 8, 4, 2, 1>;
#else
using f_tt_ranks = ::boba::StaticArray<std::size_t, 1, 2, 4, 6, 8, 9, 11, 16, 16, 8, 4, 2, 1>;
#endif
#endif

template <boba::tt_mode_order core_mode_ordering, bool is_teams, typename samples_view_t, typename x_qtt_view_t, typename f_qtt_view_t, typename output_view_t>
void run_test(
  x_qtt_view_t x_qtt_view,
  f_qtt_view_t f_qtt_view,
  samples_view_t samples_view,
  output_view_t output_view)
{
  size_t num_samples = samples_view.size();

  checkpoint();
  if constexpr (is_teams and (space == ::boba::host_space))
  {
    boba_warn("CPU batching not yet implemented");
    // CPU batching
    /*
    constexpr size_t batch_size = 128_z;

    auto num_loops = ::boba::ceil(double(num_samples)/double(batch_size));

    ::boba::loop<space, 1>(num_loops,
      [=](size_t batch_id)
    {
      ::boba::Array<double, batch_size> search_value_batch;

      auto id_offset = batch_id*batch_size;

      auto search_value_minima = samples_view(id_offset);
      auto search_value_maxima = search_value_minima;

      for(size_t b = 0; b < batch_size; b++)
      {
        auto search_id = id_offset + b;

        if(search_id >= num_samples)
        {
          search_id = id_offset;
        }

        double search_value = samples_view(search_id);

        search_value_batch[b] = search_value;

        search_value_minima = ::boba::min(search_value_minima, search_value);
        search_value_maxima = ::boba::max(search_value_maxima, search_value);
      }

      auto brackets_minima = x_qtt_view.binary_search(search_value_minima);
      auto brackets_maxima = x_qtt_view.binary_search(search_value_maxima);

      auto batch_id_bracket_left = brackets_minima[0];
      auto batch_id_bracket_right = brackets_maxima[1];

      boba_always_assert_lt(batch_id_bracket_left, batch_id_bracket_right, "Brackets need to describe a valid range.");

      // Prepare scratch memory
      static constexpr size_t shared_memory_limit = 48 * 1024 * 10;
      static constexpr size_t memory_scratch_doubles = shared_memory_limit/sizeof(double);

      double x_scratch_memory[memory_scratch_doubles];
      double f_scratch_memory[memory_scratch_doubles];

      auto indices_minima = x_qtt_view.multiindex(batch_id_bracket_left);
      auto indices_maxima = x_qtt_view.multiindex(batch_id_bracket_right);

      for(size_t d = 0; d < x_qtt_view.get_dimension(); ++d)
      {
        auto temp = indices_maxima[d];
        indices_maxima[d] = ::boba::max(indices_minima[d], temp);
        indices_minima[d] = ::boba::min(indices_minima[d], temp);
      }

      size_t x_tensor_offset = 0;
      size_t f_tensor_offset = 0;

      auto x_subtensor_mider = x_qtt_view.fetch_subtensor(
        indices_minima,
        indices_maxima,
        x_scratch_memory,
        1, // serial
        x_tensor_offset);

      auto f_subtensor_mider = f_qtt_view.fetch_subtensor(
        indices_minima,
        indices_maxima,
        f_scratch_memory,
        1, // serial
        f_tensor_offset);

      boba_always_assert_positive(x_subtensor_mider.size(), "nonsense decompressed size");
      boba_always_assert_positive(f_subtensor_mider.size(), "nonsense decompressed size");

      for(size_t b = 0; b < batch_size; b++)
      {
        auto id_bracket_left = 0_z;
        auto id_bracket_right = f_subtensor_mider.size();
        auto search_value = search_value_batch[b];

        while(id_bracket_right > id_bracket_left + 1)
        {
          auto id_guess = (id_bracket_left + id_bracket_right)/2;
          auto value = x_scratch_memory[x_tensor_offset + id_guess];
          if(value < search_value)
          {
            id_bracket_left = id_guess;
          }
          else
          {
            id_bracket_right = id_guess;
          }
        }

        double x_bracket_left = x_scratch_memory[x_tensor_offset + id_bracket_left];
        double x_bracket_right = x_scratch_memory[x_tensor_offset + id_bracket_right];
        ::boba::Array<double, 2> x_brackets{x_bracket_left, x_bracket_right};

        double f_bracket_left = f_scratch_memory[f_tensor_offset + id_bracket_left];
        double f_bracket_right = f_scratch_memory[f_tensor_offset + id_bracket_right];
        ::boba::Array<double, 2> f_brackets{f_bracket_left, f_bracket_right};

        double f_value = ::boba::lagrange_interpolation(x_brackets, f_brackets, search_value);

        auto search_id = id_offset + b;
        if(search_id < num_samples)
        {
          output_view(search_id) = f_value;
        }
      }
    });
    */
  }
  else if (is_teams)
  {
    boba_warn("Teams not yet implemented");
    /*
    ::boba::loop<space, 1>(num_samples,
      [=]__boba_host_device__(size_t search_id)
    {
      double search_value = samples_view(search_id);

      //
      // Step 1: QTT binary search with static_view
      //
      auto brackets = x_qtt_view.binary_search_teams(search_value);
      auto id_bracket_left = brackets[0];
      auto id_bracket_right = brackets[1];

      //
      // Step 2: Use brackets to evaluate table values from QTT
      //
      double x_bracket_left = x_qtt_view.unroll_value(id_bracket_left);
      double x_bracket_right = x_qtt_view.unroll_value(id_bracket_right);
      ::boba::Array<double, 2> x_brackets{x_bracket_left, x_bracket_right};

      double f_bracket_left = f_qtt_view.unroll_value(id_bracket_left);
      double f_bracket_right = f_qtt_view.unroll_value(id_bracket_right);
      ::boba::Array<double, 2> f_brackets{f_bracket_left, f_bracket_right};

      //
      // Step 3: Interpolate values to get result
      //
      double f_value = lagrange_interpolation(x_brackets, f_brackets, search_value);

      //
      // Step 4: Output result
      //
      output_view(search_id) = f_value;
    });
    */
  }
  else
  {
    ::boba::loop<space, 1>(num_samples,
                           [=] __boba_host_device__(size_t search_id)
    {
      double search_value = samples_view(search_id);

      //
      // Step 1: QTT binary search with static_view
      //
      auto brackets = x_qtt_view.binary_search(search_value);
      auto id_bracket_left = brackets[0];
      auto id_bracket_right = brackets[1];

      //
      // Step 2: Use brackets to evaluate table values from QTT
      //
      double x_bracket_left = x_qtt_view.unroll_value(id_bracket_left);
      double x_bracket_right = x_qtt_view.unroll_value(id_bracket_right);
      ::boba::Array<double, 2> x_brackets{x_bracket_left, x_bracket_right};

      double f_bracket_left = f_qtt_view.unroll_value(id_bracket_left);
      double f_bracket_right = f_qtt_view.unroll_value(id_bracket_right);
      ::boba::Array<double, 2> f_brackets{f_bracket_left, f_bracket_right};

      //
      // Step 3: Interpolate values to get result
      //
      double f_value = lagrange_interpolation(x_brackets, f_brackets, search_value);

      //
      // Step 4: Output result
      //
      output_view(search_id) = f_value;
    });
  }
}

//
// test_tt_view
//
template <boba::tt_mode_order core_mode_ordering, bool is_teams, bool is_dynamic, typename qtt_t>
void test_tt_view(
  qtt_t x_qtt,
  qtt_t f_qtt,
  const ::boba::Vector<space, double>& independent_variable_samples,
  const ::boba::Vector<space, double>& dependent_variable_samples,
  bool& check)
{
  size_t num_samples = independent_variable_samples.size();
  auto samples_view = independent_variable_samples.const_view();

  ::boba::Vector<space, double> output({num_samples});
  auto output_view = output.view();

  std::string timing_str = "Bisection search - ";
  if (is_dynamic)
  {
    timing_str += " dynamic qtt view";
  }
  else
  {
    timing_str += " static qtt view";
  }
  if (core_mode_ordering == boba::tt_mode_order::lir)
  {
    timing_str += " lir";
  }
  if (core_mode_ordering == boba::tt_mode_order::ilr)
  {
    timing_str += " ilr";
  }
  if (core_mode_ordering == boba::tt_mode_order::lri)
  {
    timing_str += " lri";
  }
  if (is_teams)
  {
    timing_str += " teams";
  }
  output.rename(timing_str);

  ::boba::TicToc<tictoc_units> timer;

  std::cout << "Running " << timing_str << std::endl;

  //
  // QTT Algorithm: Given x, find tabulated f(x)
  //
  if constexpr (is_dynamic)
  {
    if constexpr (boba::is_host(space))
    {
      // TODO<cuda, hip> port dynamic_qtt_views to gpus

      // TODO<feature> implement tt_mode_order for DynamicQuantizedTensorTrainView
      constexpr bool is_permuted = core_mode_ordering != boba::tt_mode_order::lir;

      ::boba::DynamicQuantizedTensorTrainView<double, is_permuted> x_qtt_view(x_qtt);
      ::boba::DynamicQuantizedTensorTrainView<double, is_permuted> f_qtt_view(f_qtt);

      timer.tic();
      run_test<core_mode_ordering, is_teams>(x_qtt_view, f_qtt_view, samples_view, output_view);
      timer.end_and_print(timing_str);
    }
    else
    {
      boba_error("Dynamic QTT views are not yet implemented on GPUs");
    }
  }
  else
  {
    ::boba::StaticTensorTrainView<tt_sizes, x_tt_ranks, double, core_mode_ordering> x_qtt_view(x_qtt);
    ::boba::StaticTensorTrainView<tt_sizes, f_tt_ranks, double, core_mode_ordering> f_qtt_view(f_qtt);

    timer.tic();
    run_test<core_mode_ordering, is_teams>(x_qtt_view, f_qtt_view, samples_view, output_view);
    timer.end_and_print(timing_str);
  }

  checkpoint();

  if (num_samples <= 8)
  {
    output.print();
    dependent_variable_samples.print();
  }

  pass_or_fail(check, ::boba::norm_difference_inf(output, dependent_variable_samples), 0.01); // 100.0*svd_tolerance);
}

//
//
//

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  std::cout << "Tests for quantized tensor train implementation" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  bool check = 1;

  checkpoint();

  constexpr size_t exponent = boba::is_boba_debug_mode() ? 3 : 12;
  constexpr size_t base = 2;

  size_t resolution = ::pow(base, exponent);
  size_t num_samples = resolution;
  size_t qtt_base = 2;
  double svd_tolerance = 1.0e-07;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(resolution,
                             "-n",
                             "--resolution",
                             "Number of elements in each dimension.");

  args.add_optional_argument(qtt_base,
                             "-b",
                             "--base",
                             "Base for quantized tensor train.");

  args.add_optional_argument(svd_tolerance,
                             "-t",
                             "--tolerance",
                             "Tolerance for SVD truncation.");

  args.add_optional_argument(num_samples,
                             "-s",
                             "--samples",
                             "Number of samples to test.");

  args.parse_check();

  //
  // Create data
  //
  checkpoint();
  ::boba::Vector<space, double> independent_variable({resolution});
  independent_variable.rename("independent_variable");
  auto x_view = independent_variable.view();

  ::boba::Vector<space, double> dependent_variable({resolution});
  dependent_variable.rename("dependent_variable");
  auto f_view = dependent_variable.view();

  checkpoint();
  ::boba::loop<space, 1>(resolution,
                         [=] __boba_host_device__(size_t i)
  {
    x_view(i) = i;
    f_view(i) = 1.0 / (1.0 + i);
  });

  auto x_min = independent_variable.min_reduce();
  auto x_max = independent_variable.max_reduce();

  ::boba::Vector<space, double> independent_variable_samples({num_samples});
  ::boba::Vector<space, double> dependent_variable_samples({num_samples});
  independent_variable_samples.rename("independent_variable_samples");
  dependent_variable_samples.rename("dependent_variable_samples");
  auto independent_variable_samples_view = independent_variable_samples.view();
  auto dependent_variable_samples_view = dependent_variable_samples.view();

  ::boba::loop<space, 1>(num_samples,
                         [=] __boba_host_device__(size_t id)
  {
    double d_init = x_min + 0.1 * (x_max - x_min);
    double d_fini = x_min + 0.9 * (x_max - x_min);
    double dx = num_samples - 1_z;
    double d_slope = (d_fini - d_init) / dx;
    double d_index = d_slope * double(id) + d_init;

    size_t i_low = ::boba::floor(d_index);
    size_t i_hi = ::boba::ceil(d_index);

    auto x_low = x_view(i_low);
    auto x_hi = x_view(i_hi);
    auto f_low = f_view(i_low);
    auto f_hi = f_view(i_hi);

    auto x = d_index;
    auto f = ::boba::lagrange_interpolation({x_low, x_hi}, {f_low, f_hi}, x);

    independent_variable_samples_view(id) = x;
    dependent_variable_samples_view(id) = f;
  });

  if (resolution <= 8)
  {
    independent_variable.print();
    dependent_variable.print();

    independent_variable_samples.print();
    dependent_variable_samples.print();
  }

  checkpoint();
  ::boba::Vector<space, double> output({num_samples});
  output.rename("output");
  auto output_view = output.view();

  ::boba::TicToc<tictoc_units> timer;
  timer.tic();
  //
  // Conventional algorithm: Given x, find tabulated f(x)
  //
  checkpoint();
  ::boba::loop<space, 1>(num_samples,
                         [=] __boba_host_device__(size_t search_id)
  {
    double search_value = independent_variable_samples_view(search_id);
    //
    // Step 1: Binary search for brackets of i
    //
    size_t id_bracket_left = 0;
    size_t id_bracket_right = resolution - 1;

    while (id_bracket_right > id_bracket_left + 1)
    {
      size_t id_guess = (id_bracket_left + id_bracket_right) / 2;
      auto value = x_view(id_guess);
      if (value < search_value)
      {
        id_bracket_left = id_guess;
      }
      else
      {
        id_bracket_right = id_guess;
      }
    }

    boba_always_assert_lt(id_bracket_left, id_bracket_right, "Unexpected values");

    //
    // Step 2: Use brackets to evaluate table values from QTT
    //
    double x_bracket_left = x_view(id_bracket_left);
    double x_bracket_right = x_view(id_bracket_right);
    ::boba::Array<double, 2> x_brackets{x_bracket_left, x_bracket_right};

    double f_bracket_left = f_view(id_bracket_left);
    double f_bracket_right = f_view(id_bracket_right);
    ::boba::Array<double, 2> f_brackets{f_bracket_left, f_bracket_right};

    //
    // Step 3: Interpolate values to get result
    //
    double f_value = lagrange_interpolation(x_brackets, f_brackets, search_value);

    //
    // Step 4: Output result
    //
    output_view(search_id) = f_value;
  });

  timer.end_and_print("Bisection search - conventional");
  pass_or_fail(check, ::boba::norm_difference_inf(output, dependent_variable_samples), 1.0e-9);

  //
  // Create QTTs
  //
  timer.tic();
  auto x_qtt = ::boba::compress_to_QuantizedTensorTrain(independent_variable, qtt_base, svd_tolerance);
  auto f_qtt = ::boba::compress_to_QuantizedTensorTrain(dependent_variable, qtt_base, svd_tolerance);
  timer.end_and_print("Compress QTT");

  //
  // Report diagnostics
  //
  std::cout << "Ranks, x : " << x_qtt.ranks_string() << std::endl;
  std::cout << "Ranks, f : " << f_qtt.ranks_string() << std::endl;
  if (resolution <= 8)
  {
    x_qtt.rename("x_qtt");
    x_qtt.print();
    auto x_qtt_decompress = x_qtt.decompress();
    x_qtt_decompress.rename("x_qtt_decompress");
    x_qtt_decompress.print();

    f_qtt.rename("f_qtt");
    f_qtt.print();
    auto f_qtt_decompress = f_qtt.decompress();
    f_qtt_decompress.rename("f_qtt_decompress");
    f_qtt_decompress.print();
  }

  size_t string_length = 25;
  auto x_cr = x_qtt.compression_rate();
  auto f_cr = f_qtt.compression_rate();

  std::cout
    << fill_up_string_end("Compression Rate, x QTT : ", string_length)
    << fill_up_string_end(boba::to_string(x_cr), string_length) << std::endl;

  std::cout
    << fill_up_string_end("Compression Rate, f QTT : ", string_length)
    << fill_up_string_end(boba::to_string(f_cr), string_length) << std::endl;

  checkpoint();

  //
  // QTT Algorithm via static views
  //
  if (resolution == ::boba::pow(base, exponent))
  {
    test_tt_view<boba::tt_mode_order::lir, false, false>(x_qtt, f_qtt, independent_variable_samples, dependent_variable_samples, check);

    test_tt_view<boba::tt_mode_order::ilr, false, false>(x_qtt, f_qtt, independent_variable_samples, dependent_variable_samples, check);

    test_tt_view<boba::tt_mode_order::lri, false, false>(x_qtt, f_qtt, independent_variable_samples, dependent_variable_samples, check);

    /*
    // not yet implemented

    // teams
    test_tt_view<false, true, false>(x_qtt, f_qtt, independent_variable_samples, dependent_variable_samples, check);

    // permuted teams
    test_tt_view<true, true, false>(x_qtt, f_qtt, independent_variable_samples, dependent_variable_samples, check);
*/
  }

  //
  // QTT Search algorithm via dynamic views
  //
  if constexpr (boba::is_host(space))
  {
    // TODO<cuda, hip> port dynamic views to gpus
    if (resolution == ::boba::pow(base, exponent))
    {
      test_tt_view<boba::tt_mode_order::lir, false, true>(x_qtt, f_qtt, independent_variable_samples, dependent_variable_samples, check);

      test_tt_view<boba::tt_mode_order::ilr, false, true>(x_qtt, f_qtt, independent_variable_samples, dependent_variable_samples, check);

      test_tt_view<boba::tt_mode_order::lri, false, true>(x_qtt, f_qtt, independent_variable_samples, dependent_variable_samples, check);

      /*
      // not yet implemented

      // teams
      test_tt_view<false, true, true>(x_qtt, f_qtt, independent_variable_samples, dependent_variable_samples, check);

      // permuted teams
      test_tt_view<true, true, true>(x_qtt, f_qtt, independent_variable_samples, dependent_variable_samples, check);
      */
    }
  }
  else
  {
    boba_error("Dynamic QTT views are not yet implemented on GPUs");
  }

  boba::finalize();
  return final_check(check);
}

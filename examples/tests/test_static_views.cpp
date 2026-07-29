// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Test BoBa's static views of decompositions, which assume offline decomposition so that
  one can exploit compile time information for optimally performing information extraction
*/

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::microseconds;
static size_t end_and_print_length = 70;

#include "test_static_views.hpp"

namespace tt_static_views
{

template <typename tensor_sizes, typename tt_ranks>
void initialize_test(::boba::TensorTrain<tensor_sizes{}.size(), space, double>& test)
{
  constexpr auto dimension = tensor_sizes{}.size();
  static_assert(dimension > 2, "Must be 3D");
  static_assert(dimension <= 5, "Must be at most 5D");

  auto sizes = test.sizes();
  constexpr auto R1 = tt_ranks{}[1];
  constexpr auto R2 = tt_ranks{}[2];
  constexpr auto R3 = (dimension >= 4) ? tt_ranks{}[3] : 1;
  constexpr auto R4 = (dimension >= 5) ? tt_ranks{}[4] : 1;

  test.cores[0].resize({1, sizes[0], R1});
  test.cores[1].resize({R1, sizes[1], R2});
  test.cores[2].resize({R2, sizes[2], (dimension <= 3) ? 1 : R3});
  if constexpr (dimension >= 4)
  {
    test.cores[3].resize({R3, sizes[3], (dimension <= 4) ? 1 : R4});
  }
  if constexpr (dimension >= 5)
  {
    test.cores[4].resize({R4, sizes[4], 1});
  }

  for (size_t d = 0; d < dimension; d++)
  {
    test.cores[d].fill_with_random();
  }
}

template <size_t interpolation_points, typename tensor_sizes, typename tt_ranks, boba::tt_mode_order core_mode_ordering, bool teams>
void evaluate_samples_dimension(
  ::boba::TensorTrain<tensor_sizes{}.size(), space, double>& test,
  const ::boba::Tensor<2, space, double>& samples,
  ::boba::Tensor<1, space, double>& output,
  const std::string& timer_name)
{
  checkpoint();
  constexpr auto dimension = tensor_sizes{}.size();

  ::boba::StaticTensorTrainView<tensor_sizes, tt_ranks, double, core_mode_ordering> test_view(test);

  size_t num_samples = samples.sizes(0);
  auto samples_const_view = samples.const_view();
  auto output_view = output.view();

  output.fill_with_zeros();

  boba_always_assert_equal(num_samples, output_view.size(), "Size alignment issue.");

  boba::TicToc<tictoc_units> timer;
  timer.end_and_print_length = end_and_print_length;

  checkpoint();
  if constexpr (teams)
  {
    if constexpr (space == ::boba::execution_space::CPU)
    {
      constexpr size_t batch_size = 256_z;

      auto kernel = [=] __boba_host_device__(size_t batch_id)
      {
        ::boba::Array<::boba::Array<::boba::Array<double, interpolation_points>, dimension>, batch_size> weights_batch;
        ::boba::Array<::boba::Array<::boba::Array<size_t, interpolation_points>, dimension>, batch_size> index_batch;

        auto id_offset = batch_id * batch_size;

        for (size_t b = 0; b < batch_size; b++)
        {
          auto id = id_offset + b;

          if (id >= num_samples)
          {
            id = id_offset;
          }

          ::boba::Array<double, dimension> sample;
          for (size_t d = 0; d < dimension; d++)
          {
            sample[d] = samples_const_view({id, d});
          }

          index_batch[b] = indices_low_high<interpolation_points, size_t, dimension>(sample);
          weights_batch[b] = make_weights<interpolation_points, double, dimension>(sample);
        }

        auto value_batch = test_view.template interpolation_batched<interpolation_points, batch_size>(weights_batch, index_batch);

        for (size_t b = 0; b < batch_size; b++)
        {
          auto id = id_offset + b;
          if (id < num_samples)
          {
            output_view(id) = value_batch[b];
          }
        }
      };

      timer.tic();
      ::boba::loop<space, 1>(num_samples, kernel);
      timer.end();

      boba::detail::device_sync();
      output.fill_with_zeros();
      boba::detail::device_sync();

      checkpoint();
      timer.tic();
      ::boba::loop<space, 1>(num_samples, kernel);
      timer.end_and_print(timer_name);
      boba::detail::device_sync();
    }
    else
    {
      if (interpolation_points == 1)
      {
        auto kernel = [=] __boba_host_device__(size_t id)
        {
          ::boba::Array<double, dimension> sample;
          for (size_t d = 0; d < dimension; d++)
          {
            sample[d] = samples_const_view({id, d});
          }

          output_view(id) = test_view.unroll_value_teams(indices_low_high<1, size_t, dimension>(sample));
        };

        timer.tic();
        ::boba::loop<space, 1>(num_samples, kernel);
        timer.end();

        boba::detail::device_sync();
        output.fill_with_zeros();
        boba::detail::device_sync();

        checkpoint();
        timer.tic();
        ::boba::loop<space, 1>(num_samples, kernel);
        timer.end_and_print(timer_name);
        boba::detail::device_sync();
      }
      else
      {
        auto kernel = [=] __boba_host_device__(size_t id)
        {
          ::boba::Array<double, dimension> sample;
          for (size_t d = 0; d < dimension; d++)
          {
            sample[d] = samples_const_view({id, d});
          }

          ::boba::Array<::boba::Array<size_t, 2>, dimension> index_bounds = indices_low_high<2, size_t, dimension>(sample);
          ::boba::Array<::boba::Array<double, 2>, dimension> weights = make_weights<2, double, dimension>(sample);
          output_view(id) = test_view.template interpolation_teams<2>(weights, index_bounds);
        };

        timer.tic();
        ::boba::loop<space, 1>(num_samples, kernel);
        timer.end();

        boba::detail::device_sync();
        output.fill_with_zeros();
        boba::detail::device_sync();

        checkpoint();
        timer.tic();
        ::boba::loop<space, 1>(num_samples, kernel);
        timer.end_and_print(timer_name);
        boba::detail::device_sync();
      }
    }
  }
  else
  {
    auto kernel = [=] __boba_host_device__(size_t id)
    {
      ::boba::Array<double, dimension> sample;
      for (size_t d = 0; d < dimension; d++)
      {
        sample[d] = samples_const_view({id, d});
      }

      ::boba::Array<::boba::Array<size_t, interpolation_points>, dimension> index_bounds = indices_low_high<interpolation_points, size_t, dimension>(sample);
      ::boba::Array<::boba::Array<double, interpolation_points>, dimension> weights = make_weights<interpolation_points, double, dimension>(sample);

      output_view(id) = test_view.template interpolation<interpolation_points>(weights, index_bounds);
    };

    timer.tic();
    ::boba::loop<space, 1>(num_samples, kernel);
    timer.end();

    boba::detail::device_sync();
    output.fill_with_zeros();
    boba::detail::device_sync();

    checkpoint();
    timer.tic();
    ::boba::loop<space, 1>(num_samples, kernel);
    timer.end_and_print(timer_name);
    boba::detail::device_sync();
  }
}

template <size_t interpolation_points, typename tensor_sizes, typename tt_ranks, boba::tt_mode_order core_mode_ordering, bool teams>
void loop_over_dimensions(
  boba::TensorTrain<tensor_sizes{}.size(), space, double> test_input,
  const boba::Tensor<1, space, double>& verification_values,
  boba::Tensor<2, space, double>& samples,
  const std::string& comment,
  bool& check,
  double sampling_tolerance)
{
  if constexpr ((interpolation_points == 2) and teams)
  {
    std::cout << "Skipping " << comment << std::endl;
    return;
  }
  if constexpr ((space == ::boba::execution_space::CPU) and teams)
  {
    std::cout << "Skipping " << comment << std::endl;
    return;
  }

  ::boba::Tensor<1, space, double> test_output({verification_values.size()});
  evaluate_samples_dimension<interpolation_points, tensor_sizes, tt_ranks, core_mode_ordering, teams>(test_input, samples, test_output, comment);
  pass_or_fail(check, ::boba::norm_difference_inf(test_output, verification_values), sampling_tolerance);
}

template <size_t interpolation_points, typename tensor_sizes, typename tt_ranks>
void run_single_case(bool& check, size_t num_samples, size_t sample_path)
{
  constexpr size_t dimension = tensor_sizes{}.size();
  double sampling_tolerance = 1.0e-09;

  checkpoint();
  boba_print(dimension);
  boba_print(interpolation_points);

  auto sizes = ::boba::typed_array<size_t>(::boba::to_array(tensor_sizes{}));
  boba::TensorTrain<dimension, space, double> test(sizes);

  initialize_test<tensor_sizes, tt_ranks>(test);
  auto full_tensor = test.decompress();

  checkpoint();
  auto samples = make_samples<space>(num_samples, sample_path, sizes);

  checkpoint();
  auto output_tensor = test_tensor_full<interpolation_points, tensor_sizes>(full_tensor, samples);

  checkpoint();
  loop_over_dimensions<interpolation_points, tensor_sizes, tt_ranks, boba::tt_mode_order::lir, false>(test, output_tensor, samples, "tt sampling - lir, path: " + samples.name(), check, sampling_tolerance);

  checkpoint();
  loop_over_dimensions<interpolation_points, tensor_sizes, tt_ranks, boba::tt_mode_order::ilr, false>(test, output_tensor, samples, "tt sampling - ilr, path: " + samples.name(), check, sampling_tolerance);

  checkpoint();
  loop_over_dimensions<interpolation_points, tensor_sizes, tt_ranks, boba::tt_mode_order::lri, false>(test, output_tensor, samples, "tt sampling - lri, path: " + samples.name(), check, sampling_tolerance);

  checkpoint();
  loop_over_dimensions<interpolation_points, tensor_sizes, tt_ranks, boba::tt_mode_order::lir, true>(test, output_tensor, samples, "tt sampling teams - lir, path: " + samples.name(), check, sampling_tolerance);

  checkpoint();
  loop_over_dimensions<interpolation_points, tensor_sizes, tt_ranks, boba::tt_mode_order::ilr, true>(test, output_tensor, samples, "tt sampling teams - ilr, path: " + samples.name(), check, sampling_tolerance);

  checkpoint();
  loop_over_dimensions<interpolation_points, tensor_sizes, tt_ranks, boba::tt_mode_order::lri, true>(test, output_tensor, samples, "tt sampling teams - lri, path: " + samples.name(), check, sampling_tolerance);
}

template <typename tensor_sizes_3d, typename tt_ranks_3d, typename tensor_sizes_4d, typename tt_ranks_4d, typename tensor_sizes_5d, typename tt_ranks_5d>
void run_dimensions(bool& check, size_t num_samples, size_t run_dimension, size_t sample_path, size_t interpolation_points)
{
  if ((run_dimension == 3_z) or (run_dimension == 0_z))
  {
    if ((interpolation_points == 1_z) or (interpolation_points == 0_z))
    {
      run_single_case<1, tensor_sizes_3d, tt_ranks_3d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------");
    }
    if ((interpolation_points == 2_z) or (interpolation_points == 0_z))
    {
      run_single_case<2, tensor_sizes_3d, tt_ranks_3d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------\n\n");
    }
  }
  if ((run_dimension == 4_z) or (run_dimension == 0_z))
  {
    if ((interpolation_points == 1_z) or (interpolation_points == 0_z))
    {
      run_single_case<1, tensor_sizes_4d, tt_ranks_4d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------");
    }
    if ((interpolation_points == 2_z) or (interpolation_points == 0_z))
    {
      run_single_case<2, tensor_sizes_4d, tt_ranks_4d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------\n\n");
    }
  }
  if ((run_dimension == 5_z) or (run_dimension == 0_z))
  {
    if ((interpolation_points == 1_z) or (interpolation_points == 0_z))
    {
      run_single_case<1, tensor_sizes_5d, tt_ranks_5d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------");
    }
    if ((interpolation_points == 2_z) or (interpolation_points == 0_z))
    {
      run_single_case<2, tensor_sizes_5d, tt_ranks_5d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------\n\n");
    }
  }
}

void run_selected(bool& check, size_t num_samples, size_t run_dimension, int sample_path_parm, size_t interpolation_points)
{
#ifdef BOBA_DEBUG
  using tensor_sizes_5d = boba::StaticArray<size_t, 101, 101, 2, 5, 16>;
  using tt_ranks_5d = boba::StaticArray<size_t, 1, 3, 3, 2, 4, 1>;
  using tensor_sizes_4d = boba::StaticArray<size_t, tensor_sizes_5d{}[0], tensor_sizes_5d{}[1], tensor_sizes_5d{}[2], tensor_sizes_5d{}[3]>;
  using tt_ranks_4d = boba::StaticArray<size_t, 1, tt_ranks_5d{}[0], tt_ranks_5d{}[1], tt_ranks_5d{}[2], 1>;
  using tensor_sizes_3d = boba::StaticArray<size_t, tensor_sizes_5d{}[0], tensor_sizes_5d{}[1], tensor_sizes_5d{}[2]>;
  using tt_ranks_3d = boba::StaticArray<size_t, 1, tt_ranks_5d{}[0], tt_ranks_5d{}[1], 1>;
#elif defined(BOBA_CI)
  using tensor_sizes_5d = boba::StaticArray<size_t, 101, 101, 2, 5, 16>;
  using tt_ranks_5d = boba::StaticArray<size_t, 1, 5, 25, 36, 5, 1>;
  using tensor_sizes_4d = boba::StaticArray<size_t, tensor_sizes_5d{}[0], tensor_sizes_5d{}[1], tensor_sizes_5d{}[2], tensor_sizes_5d{}[3]>;
  using tt_ranks_4d = boba::StaticArray<size_t, 1, tt_ranks_5d{}[0], tt_ranks_5d{}[1], tt_ranks_5d{}[2], 1>;
  using tensor_sizes_3d = boba::StaticArray<size_t, tensor_sizes_5d{}[0], tensor_sizes_5d{}[1], tensor_sizes_5d{}[2]>;
  using tt_ranks_3d = boba::StaticArray<size_t, 1, tt_ranks_5d{}[0], tt_ranks_5d{}[1], 1>;
#else
  constexpr size_t ranks_factor = 3;
  constexpr size_t R1 = 5 * ranks_factor;
  constexpr size_t R2 = 4 * ranks_factor;
  constexpr size_t R3 = 5 * ranks_factor;
  constexpr size_t R4 = 7 * ranks_factor;
  using tensor_sizes_5d = boba::StaticArray<size_t, 101, 101, 2, 5, 16>;
  using tt_ranks_5d = boba::StaticArray<size_t, 1, R1, R2, R3, R4, 1>;
  using tensor_sizes_4d = boba::StaticArray<size_t, tensor_sizes_5d{}[0], tensor_sizes_5d{}[1], tensor_sizes_5d{}[2], tensor_sizes_5d{}[3]>;
  using tt_ranks_4d = boba::StaticArray<size_t, 1, tt_ranks_5d{}[0], tt_ranks_5d{}[1], tt_ranks_5d{}[2], 1>;
  using tensor_sizes_3d = boba::StaticArray<size_t, tensor_sizes_5d{}[0], tensor_sizes_5d{}[1], tensor_sizes_5d{}[2]>;
  using tt_ranks_3d = boba::StaticArray<size_t, 1, tt_ranks_5d{}[0], tt_ranks_5d{}[1], 1>;
#endif

  for (size_t sample_path = 0_z; sample_path < num_possible_paths; sample_path++)
  {
    if (sample_path_parm >= 0)
    {
      sample_path = static_cast<size_t>(sample_path_parm);
    }

    boba_print(sample_path);
    run_dimensions<tensor_sizes_3d, tt_ranks_3d, tensor_sizes_4d, tt_ranks_4d, tensor_sizes_5d, tt_ranks_5d>(
      check, num_samples, run_dimension, sample_path, interpolation_points);
    boba_print("-----------------------------------------------");

    if (sample_path_parm >= 0)
    {
      break;
    }
  }
}

} // namespace tt_static_views

#if !defined(BOBA_CUDA) && !defined(BOBA_HIP)
namespace tucker_static_views
{

template <typename tensor_sizes, typename Tucker_ranks>
void initialize_test(::boba::Tucker<tensor_sizes{}.size(), space, double>& test)
{
  constexpr auto dimension = tensor_sizes{}.size();
  static_assert(dimension > 2, "Must be 3D");
  static_assert(dimension <= 5, "Must be at most 5D");

  auto sizes = test.sizes();
  constexpr auto R1 = Tucker_ranks{}[0];
  constexpr auto R2 = Tucker_ranks{}[1];
  constexpr auto R3 = Tucker_ranks{}[2];
  constexpr auto R4 = (dimension >= 4) ? Tucker_ranks{}[3] : 1;
  constexpr auto R5 = (dimension >= 5) ? Tucker_ranks{}[4] : 1;

  test.cores[0].resize({sizes[0], R1});
  test.cores[1].resize({sizes[1], R2});
  test.cores[2].resize({sizes[2], R3});
  if constexpr (dimension >= 4)
  {
    test.cores[3].resize({sizes[3], R4});
  }
  if constexpr (dimension >= 5)
  {
    test.cores[4].resize({sizes[4], R5});
  }

  if constexpr (dimension == 3)
  {
    test.R_core.resize({R1, R2, R3});
  }
  if constexpr (dimension == 4)
  {
    test.R_core.resize({R1, R2, R3, R4});
  }
  if constexpr (dimension == 5)
  {
    test.R_core.resize({R1, R2, R3, R4, R5});
  }

  for (size_t d = 0; d < dimension; d++)
  {
    test.cores[d].fill_with_random();
  }
  test.R_core.fill_with_random();
}

template <size_t interpolation_points, typename tensor_sizes, typename Tucker_ranks, boba::tucker_mode_order core_mode_ordering, bool teams>
void evaluate_samples_dimension(
  ::boba::Tucker<tensor_sizes{}.size(), space, double>& test,
  const ::boba::Tensor<2, space, double>& samples,
  ::boba::Tensor<1, space, double>& output,
  const std::string& timer_name)
{
  checkpoint();

  ::boba::StaticTuckerView<tensor_sizes, Tucker_ranks, double, core_mode_ordering> test_view(test);

  size_t num_samples = samples.sizes(0);
  auto samples_const_view = samples.const_view();
  auto output_view = output.view();

  output.fill_with_zeros();

  boba_always_assert_equal(num_samples, output_view.size(), "Size alignment issue.");

  boba::TicToc<tictoc_units> timer;
  timer.end_and_print_length = end_and_print_length;

  checkpoint();
  auto kernel = [=] __boba_host_device__(size_t id)
  {
    constexpr auto dimension = tensor_sizes{}.size();
    ::boba::Array<double, dimension> sample;
    for (size_t d = 0; d < dimension; d++)
    {
      sample[d] = samples_const_view({id, d});
    }

    ::boba::Array<::boba::Array<size_t, interpolation_points>, dimension> index_bounds = indices_low_high<interpolation_points, size_t, dimension>(sample);
    ::boba::Array<::boba::Array<double, interpolation_points>, dimension> weights = make_weights<interpolation_points, double, dimension>(sample);

    output_view(id) = test_view.template interpolation<interpolation_points>(weights, index_bounds);
  };

  timer.tic();
  ::boba::loop<space, 1>(num_samples, kernel);
  timer.end();

  boba::detail::device_sync();
  output.fill_with_zeros();
  boba::detail::device_sync();

  checkpoint();
  timer.tic();
  ::boba::loop<space, 1>(num_samples, kernel);
  timer.end_and_print(timer_name);
  boba::detail::device_sync();
}

template <size_t interpolation_points, typename tensor_sizes, typename Tucker_ranks, boba::tucker_mode_order core_mode_ordering, bool teams>
void loop_over_dimensions(
  boba::Tucker<tensor_sizes{}.size(), space, double> test_input,
  const boba::Tensor<1, space, double>& verification_values,
  boba::Tensor<2, space, double>& samples,
  const std::string& comment,
  bool& check,
  double sampling_tolerance)
{
  if constexpr (teams)
  {
    std::cout << "Skipping " << comment << " - not yet implemented " << std::endl;
    return;
  }

  ::boba::Tensor<1, space, double> test_output({verification_values.size()});
  evaluate_samples_dimension<interpolation_points, tensor_sizes, Tucker_ranks, core_mode_ordering, teams>(test_input, samples, test_output, comment);
  pass_or_fail(check, ::boba::norm_difference_inf(test_output, verification_values), sampling_tolerance);
}

template <size_t interpolation_points, typename tensor_sizes, typename Tucker_ranks>
void run_single_case(bool& check, size_t num_samples, size_t sample_path)
{
  constexpr size_t dimension = tensor_sizes{}.size();
  double sampling_tolerance = 1.0e-09;

  checkpoint();
  boba_print(dimension);
  boba_print(interpolation_points);

  auto sizes = ::boba::typed_array<size_t>(::boba::to_array(tensor_sizes{}));
  boba::Tucker<dimension, space, double> test(sizes);

  initialize_test<tensor_sizes, Tucker_ranks>(test);
  auto full_tensor = test.decompress();

  checkpoint();
  auto samples = make_samples<space>(num_samples, sample_path, sizes);

  checkpoint();
  auto output_tensor = test_tensor_full<interpolation_points, tensor_sizes>(full_tensor, samples);

  checkpoint();
  loop_over_dimensions<interpolation_points, tensor_sizes, Tucker_ranks, boba::tucker_mode_order::ir, false>(test, output_tensor, samples, "tucker sampling - ir, path: " + samples.name(), check, sampling_tolerance);

  checkpoint();
  loop_over_dimensions<interpolation_points, tensor_sizes, Tucker_ranks, boba::tucker_mode_order::ri, false>(test, output_tensor, samples, "tucker sampling - ri, path: " + samples.name(), check, sampling_tolerance);

  checkpoint();
  loop_over_dimensions<interpolation_points, tensor_sizes, Tucker_ranks, boba::tucker_mode_order::ir, true>(test, output_tensor, samples, "tucker sampling teams - ir, path: " + samples.name(), check, sampling_tolerance);

  checkpoint();
  loop_over_dimensions<interpolation_points, tensor_sizes, Tucker_ranks, boba::tucker_mode_order::ri, true>(test, output_tensor, samples, "tucker sampling teams - ri, path: " + samples.name(), check, sampling_tolerance);
}

template <typename tensor_sizes_3d, typename Tucker_ranks_3d, typename tensor_sizes_4d, typename Tucker_ranks_4d, typename tensor_sizes_5d, typename Tucker_ranks_5d>
void run_dimensions(bool& check, size_t num_samples, size_t run_dimension, size_t sample_path, size_t interpolation_points)
{
  if ((run_dimension == 3_z) or (run_dimension == 0_z))
  {
    if ((interpolation_points == 1_z) or (interpolation_points == 0_z))
    {
      run_single_case<1, tensor_sizes_3d, Tucker_ranks_3d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------");
    }
    if ((interpolation_points == 2_z) or (interpolation_points == 0_z))
    {
      run_single_case<2, tensor_sizes_3d, Tucker_ranks_3d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------\n\n");
    }
  }
  if ((run_dimension == 4_z) or (run_dimension == 0_z))
  {
    if ((interpolation_points == 1_z) or (interpolation_points == 0_z))
    {
      run_single_case<1, tensor_sizes_4d, Tucker_ranks_4d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------");
    }
    if ((interpolation_points == 2_z) or (interpolation_points == 0_z))
    {
      run_single_case<2, tensor_sizes_4d, Tucker_ranks_4d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------\n\n");
    }
  }
  if ((run_dimension == 5_z) or (run_dimension == 0_z))
  {
    if ((interpolation_points == 1_z) or (interpolation_points == 0_z))
    {
      run_single_case<1, tensor_sizes_5d, Tucker_ranks_5d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------");
    }
    if ((interpolation_points == 2_z) or (interpolation_points == 0_z))
    {
      run_single_case<2, tensor_sizes_5d, Tucker_ranks_5d>(check, num_samples, sample_path);
      boba_print("-----------------------------------------------\n\n");
    }
  }
}

void run_selected(bool& check, size_t num_samples, size_t run_dimension, int sample_path_parm, size_t interpolation_points)
{
#ifdef BOBA_CI
  using tensor_sizes_5d = boba::StaticArray<size_t, 101, 101, 2, 5, 16>;
  using Tucker_ranks_5d = boba::StaticArray<size_t, 2, 3, 3, 2, 4>;
  using tensor_sizes_4d = boba::StaticArray<size_t, tensor_sizes_5d{}[0], tensor_sizes_5d{}[1], tensor_sizes_5d{}[2], tensor_sizes_5d{}[3]>;
  using Tucker_ranks_4d = boba::StaticArray<size_t, Tucker_ranks_5d{}[0], Tucker_ranks_5d{}[1], Tucker_ranks_5d{}[2], Tucker_ranks_5d{}[3]>;
  using tensor_sizes_3d = boba::StaticArray<size_t, tensor_sizes_5d{}[0], tensor_sizes_5d{}[1], tensor_sizes_5d{}[2]>;
  using Tucker_ranks_3d = boba::StaticArray<size_t, Tucker_ranks_5d{}[0], Tucker_ranks_5d{}[1], Tucker_ranks_5d{}[2]>;
#else
  constexpr size_t ranks_factor = 1;
  constexpr size_t R0 = 5 * ranks_factor;
  constexpr size_t R1 = 5 * ranks_factor;
  constexpr size_t R2 = 4 * ranks_factor;
  constexpr size_t R3 = 5 * ranks_factor;
  constexpr size_t R4 = 7 * ranks_factor;
  using tensor_sizes_5d = boba::StaticArray<size_t, 101, 101, 2, 5, 16>;
  using Tucker_ranks_5d = boba::StaticArray<size_t, R0, R1, R2, R3, R4>;
  using tensor_sizes_4d = boba::StaticArray<size_t, tensor_sizes_5d{}[0], tensor_sizes_5d{}[1], tensor_sizes_5d{}[2], tensor_sizes_5d{}[3]>;
  using Tucker_ranks_4d = boba::StaticArray<size_t, Tucker_ranks_5d{}[0], Tucker_ranks_5d{}[1], Tucker_ranks_5d{}[2], Tucker_ranks_5d{}[3]>;
  using tensor_sizes_3d = boba::StaticArray<size_t, tensor_sizes_5d{}[0], tensor_sizes_5d{}[1], tensor_sizes_5d{}[2]>;
  using Tucker_ranks_3d = boba::StaticArray<size_t, Tucker_ranks_5d{}[0], Tucker_ranks_5d{}[1], Tucker_ranks_5d{}[2]>;
#endif

  for (size_t sample_path = 0_z; sample_path < num_possible_paths; sample_path++)
  {
    if (sample_path_parm >= 0)
    {
      sample_path = static_cast<size_t>(sample_path_parm);
    }

    boba_print(sample_path);
    run_dimensions<tensor_sizes_3d, Tucker_ranks_3d, tensor_sizes_4d, Tucker_ranks_4d, tensor_sizes_5d, Tucker_ranks_5d>(
      check, num_samples, run_dimension, sample_path, interpolation_points);
    boba_print("-----------------------------------------------");

    if (sample_path_parm >= 0)
    {
      break;
    }
  }
}

} // namespace tucker_static_views
#endif

int main(int argc, char* argv[])
{
  boba::splash();
  boba::init();
  std::cout << "Tests for static views and random access" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;
  checkpoint();

  using tensor_sizes_5d_for_samples = boba::StaticArray<size_t, 101, 101, 2, 5, 16>;

  size_t num_samples = 4 * boba::max(::boba::typed_array<size_t>(::boba::to_array(tensor_sizes_5d_for_samples{})));
  size_t run_dimension = 0;
  int sample_path_parm = -1;
  size_t interpolation_points = 0;
  size_t which_view = 0;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(num_samples,
                             "-s",
                             "--samples",
                             "Number of samples to test.");

  args.add_optional_argument(run_dimension,
                             "-d",
                             "--dimension",
                             "Table dimensionality.");

  args.add_optional_argument(sample_path_parm,
                             "-p",
                             "--path",
                             "Select which path used to iterate through the data. See code for more details.");

  args.add_optional_argument(interpolation_points,
                             "-i",
                             "--interpolation",
                             "How many interpolation points to use, eg: 1 - nearest neighbor, 2 - multilinear interpolation.");

  args.add_optional_argument(which_view,
                             "-w",
                             "--which",
                             "Static view backend to test (0 = all, 1 = tensor train, 2 = Tucker).");

  args.parse_check();

  if ((which_view == 0) or (which_view == 1))
  {
    std::cout << "Running tensor train static-view tests" << std::endl;
    tt_static_views::run_selected(check, num_samples, run_dimension, sample_path_parm, interpolation_points);
  }

  if ((which_view == 0) or (which_view == 2))
  {
#if !defined(BOBA_CUDA) && !defined(BOBA_HIP)
    std::cout << "Running Tucker static-view tests" << std::endl;
    tucker_static_views::run_selected(check, num_samples, run_dimension, sample_path_parm, interpolation_points);
#else
    std::cout << "Skipping Tucker static-view tests on this backend" << std::endl;
#endif
  }

  return final_check(check);
}

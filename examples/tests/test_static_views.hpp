// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
  Map elements of sample to the bounding indices
*/

template <typename types, size_t dimension, typename in_type>
__boba_host_device__ void indices_low_high_helper(::boba::Array<in_type, dimension>& sample, ::boba::Array<::boba::Array<types, 2>, dimension>& low_high)
{
  for (size_t d = 0; d < dimension; d++)
  {
    low_high[d][0] = static_cast<types>(::boba::floor(sample[d]));
    low_high[d][1] = low_high[d][0] + 1;
  }
}

/*
  Map elements of sample to a truncated matching of the index
*/

template <typename types, size_t dimension, typename in_type>
__boba_host_device__ void indices_low_high_helper(::boba::Array<in_type, dimension>& sample, ::boba::Array<::boba::Array<types, 1>, dimension>& low_high)
{
  for (size_t d = 0; d < dimension; d++)
  {
    low_high[d][0] = static_cast<types>(::boba::floor(sample[d]));
  }
}

/*
  Map elements of sample to a truncated matching of the index
*/

template <typename types, size_t dimension, typename in_type>
__boba_host_device__ void indices_low_high_helper(::boba::Array<in_type, dimension>& sample, ::boba::Array<types, dimension>& low_high)
{
  for (size_t d = 0; d < dimension; d++)
  {
    low_high[d] = static_cast<types>(::boba::floor(sample[d]));
  }
}

/*
   Given a 'continuous' set of indices J_i < j_i < J_i+1, finds the set of bounding indices J_i and J_i+1
*/
template <size_t interpolation_points, typename types, size_t dimension, typename in_type>
__boba_host_device__ ::boba::Array<::boba::Array<types, interpolation_points>, dimension> indices_low_high(::boba::Array<in_type, dimension>& sample)
{
  ::boba::Array<::boba::Array<types, interpolation_points>, dimension> low_high;
  indices_low_high_helper(sample, low_high);
  return low_high;
}

template <typename types, size_t dimension, typename in_type>
__boba_host_device__ ::boba::Array<types, dimension> indices_low_high_flat(::boba::Array<in_type, dimension>& sample)
{
  ::boba::Array<types, dimension> low_high;
  indices_low_high_helper(sample, low_high);
  return low_high;
}

/*
  Lagrange interpolation weights
*/

template <typename value_t, typename id_t, size_t dimension>
__boba_host_device__ void make_weights_helper(::boba::Array<double, dimension>& sample, ::boba::Array<::boba::Array<id_t, 2>, dimension>& d_index_bounds, ::boba::Array<::boba::Array<value_t, 2>, dimension>& weights)
{
  for (size_t d = 0; d < dimension; d++)
  {
    weights[d] = ::boba::lagrange_weights(d_index_bounds[d], sample[d]);
  }
}

/*
  Uniform weighting for a single sample
*/

template <typename value_t, typename id_t, size_t dimension>
__boba_host_device__ void make_weights_helper(::boba::Array<double, dimension>& sample, ::boba::Array<::boba::Array<id_t, 1>, dimension>& d_index_bounds, ::boba::Array<::boba::Array<value_t, 1>, dimension>& weights)
{
  boba::detail::ignore(sample);
  boba::detail::ignore(d_index_bounds);
  for (size_t d = 0; d < dimension; d++)
  {
    weights[d] = 1.0;
  }
}

/*
  Interpolation weights
*/

template <size_t interpolation_points, typename types, size_t dimension>
__boba_host_device__ ::boba::Array<::boba::Array<types, interpolation_points>, dimension> make_weights(::boba::Array<double, dimension>& sample)
{
  auto d_index_bounds = indices_low_high<interpolation_points, types, dimension>(sample);
  ::boba::Array<::boba::Array<types, interpolation_points>, dimension> weights;
  make_weights_helper(sample, d_index_bounds, weights);
  return weights;
}

//
// Define sampling logic for random access sampling tests
// different paths move along different strides of the full table
//

static constexpr size_t num_possible_paths = 8;

template <boba::execution_space space, size_t dimension>
boba::Tensor<2, space, double> make_samples(
  size_t num_samples,
  size_t sample_path,
  boba::Array<size_t, dimension> sizes)
{
  checkpoint();
  boba::Tensor<2, space, double> samples({num_samples, dimension});
  auto samples_view = samples.view();

  checkpoint();
  switch (sample_path)
  {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
    // Stride along the i'th dimension
    {
      samples.rename("stride dimension " + std::to_string(sample_path));

      ::boba::loop<space, 1>(num_samples,
                             [=] __boba_host_device__(size_t id)
      {
        for (size_t d = 0; d < dimension; d++)
        {
          auto N = sizes[d];
          double d_init = 0.0;
          double d_fini = double(N - 1);
          double d_slope = double(d_fini - d_init) / double(num_samples);
          double i = (0.5) * double(N - 1);
          if (sample_path == d)
          {
            i = d_slope * double(id) + d_init;
          }
          samples_view({id, d}) = i;
        }
      });
      break;
    }
  case 5:
    // Randomly distributed points
    {
      samples.fill_with_random();
      samples.rename("randomly distributed");

      ::boba::loop<space, 1>(num_samples,
                             [=] __boba_host_device__(size_t id)
      {
        for (size_t d = 0; d < dimension; d++)
        {
          auto N = sizes[d];
          auto i = samples_view({id, d}) * double(N - 1) * 0.999;
          samples_view({id, d}) = i;
        }
      });
      break;
    }
  case 6:
    // Linearly path along hyperdiagonal
    {
      samples.rename("stride hyperdiagonal");

      ::boba::loop<space, 1>(num_samples,
                             [=] __boba_host_device__(size_t id)
      {
        for (size_t d = 0; d < dimension; d++)
        {
          auto N = sizes[d];
          double d_init = 0.0;
          double d_fini = double(N - 1);
          double d_slope = double(d_fini - d_init) / double(num_samples);
          double i = d_slope * double(id) + d_init;
          samples_view({id, d}) = i;
        }
      });
      break;
    }
  case 7:
    // All samples are the same value, around the halfway value
    {
      samples.rename("sample single index");

      ::boba::loop<space, 1>(num_samples,
                             [=] __boba_host_device__(size_t id)
      {
        for (size_t d = 0; d < dimension; d++)
        {
          auto i = static_cast<size_t>(double(sizes[d] - 1) * 0.5);
          samples_view({id, d}) = i;
        }
      });
      break;
    }
  default:
    boba_always_assert_le(sample_path, 6, "Undefined path ");
  }

  return samples;
}

//
// Test full tensor random access
//
template <size_t interpolation_points, typename tensor_sizes, size_t dimension>
boba::Tensor<1, space, double> test_tensor_full(
  boba::Tensor<dimension, space, double>& full_tensor,
  boba::Tensor<2, space, double>& samples)
{
  auto num_samples = samples.sizes(0);
  checkpoint();
  boba::Tensor<1, space, double> output_tensor({num_samples});
  {
    auto samples_const_view = samples.const_view();
    auto output_tensor_view = output_tensor.view();

    auto full_tensor_view = full_tensor.view();

    ::boba::StaticTensorView<double, tensor_sizes> static_view;
    static_view.reset(full_tensor_view);

    boba::TicToc<tictoc_units> timer;
    timer.end_and_print_length = end_and_print_length;
    timer.tic();

    //
    // Run warmup kernel first
    //
    auto kernel = [=] __boba_host_device__(size_t id)
    {
      ::boba::Array<double, dimension> sample;
      for (size_t d = 0; d < dimension; d++)
      {
        sample[d] = samples_const_view({id, d});
      }
      double value;

      if (interpolation_points == 1)
      {
        value = static_view(indices_low_high_flat<size_t, dimension>(sample));
      }
      else
      {
        ::boba::Array<::boba::Array<size_t, 2>, dimension> index_bounds = indices_low_high<2, size_t, dimension>(sample);
        ::boba::Array<::boba::Array<double, 2>, dimension> weights = make_weights<2, double, dimension>(sample);
        value = static_view.template interpolation<2>(weights, index_bounds);
      }

      output_tensor_view(id) = value;
    };

    ::boba::loop<space, 1>(num_samples, kernel);
    timer.end();

    boba::detail::device_sync();
    output_tensor.fill_with_zeros();
    boba::detail::device_sync();

    //
    // Now time it for real
    //
    checkpoint();
    timer.tic();
    ::boba::loop<space, 1>(num_samples, kernel);
    timer.end_and_print("tensor_sampling, path: " + samples.name());
    boba::detail::device_sync();
  }
  return output_tensor;
}

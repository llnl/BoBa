// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <stdexcept>

namespace boba
{

/**
 * @brief Selects the direction of an FFT.
 */
enum fft_operation : size_t
{
  forward, // Forward
  backward // Inverse
};

/**
 * \brief Performs a batched FFT along one tensor dimension.
 *
 * Leaves all other dimensions untransformed. The input tensor must store complex values.
 *
 * @tparam dimension Tensor dimension.
 * @tparam space Tensor execution space.
 * @tparam data_t Real scalar type underlying the complex values.
 * @tparam index_t Index type.
 * @param input Input tensor.
 * @param transform_dimension Dimension to transform.
 * @param operation Transform direction.
 * @return Tensor containing the transformed values.
 */
template <size_t dimension, execution_space space, typename data_t>
[[nodiscard]]
Tensor<dimension, space, complex<data_t>> fft_along_dimension(
  const Tensor<dimension, space, complex<data_t>>& input,
  index_t transform_dimension,
  fft_operation operation)
{
  auto input_copy = input;

  size_t batch_size = 0;

  if constexpr (dimension > 1)
  {
    auto untransformed_dimensions_sizes = boba::delete_element(input.sizes(), transform_dimension);
    batch_size = product(untransformed_dimensions_sizes);

    //
    // Want transformed dimension to be contiguous in memory
    //
    boba::Array<size_t, dimension> permute_dimension_to_front;
    permute_dimension_to_front[0] = transform_dimension;
    for (size_t i = 1; i < dimension; i++)
    {
      permute_dimension_to_front[i] = (i <= transform_dimension) ? i - 1 : i;
    }

    boba::permute(input_copy, permute_dimension_to_front);
  }
  else
  {
    batch_size = 1;
  }

  Tensor<dimension, space, complex<data_t>> output(input_copy.sizes());

  bool is_inverse = operation == fft_operation::backward;

  if constexpr (space == execution_space::CPU)
  {
    eigen_fft<dimension, data_t>(input_copy, output, is_inverse);
  }
  else if constexpr (space == execution_space::CUDA)
  {
    cuda_fft(input_copy, output, batch_size, is_inverse);
  }
  else if constexpr (space == execution_space::HIP)
  {
    hip_fft(input_copy, output, batch_size, is_inverse);
  }
  else
  {
    boba_error("Unknown space!");
  }

  if constexpr (dimension > 1)
  {
    //
    // Put transformed dimension back to original axis
    //
    boba::Array<size_t, dimension> permute_front_to_dimension;
    for (size_t i = 0; i < dimension; i++)
    {
      permute_front_to_dimension[i] = (i < transform_dimension) ? i + 1 : i;
    }
    permute_front_to_dimension[transform_dimension] = 0;

    boba::permute(output, permute_front_to_dimension);
  }

  return output;
}

} // namespace boba

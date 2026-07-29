// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <span>

namespace boba
{

// -------------------------------------------------------------------------------------
// Section: NaN checks
// -------------------------------------------------------------------------------------

template <size_t dimension, execution_space space, typename data_t>
void nan_check(TensorTrain<dimension, space, data_t> const& train)
{
  for (size_t d = 0; d < dimension; d++)
  {
    ::boba::nan_check(train.cores[d]);
  }
}

// -------------------------------------------------------------------------------------
// Section: Norms
// -------------------------------------------------------------------------------------

/**
 * \brief Frobenius norm of the tensor train in the sense of vectors.
 */

template <size_t dimension, execution_space space, typename data_t>
auto norm_frobenius(TensorTrain<dimension, space, data_t> const& train)
{
  BOBA_CALI_MARK
  const auto product = train.inner_product(train);
  const auto abs_product = boba::abs(product);
  return boba::sqrt(abs_product);
}

// -------------------------------------------------------------------------------------
// Section: Multiply
// -------------------------------------------------------------------------------------

/**
 * @brief Perform the elementwise_product (aka Hadamard product) of inputs
 *
 * @param[in] train_A TensorTrain
 * @param[in] train_B TensorTrain
 * @return TensorTrain which is the Hadamard product train_A * train_B
 */

template <size_t dimension, execution_space space, typename data_t>
TensorTrain<dimension, space, data_t> elementwise_product(
  TensorTrain<dimension, space, data_t> const& train_A,
  TensorTrain<dimension, space, data_t> const& train_B)
{
  BOBA_CALI_MARK

  checkpoint();
  TensorTrain<dimension, space, data_t> output(train_A.sizes());

  ::boba::Array<size_t, dimension> new_ranks_left = train_A.get_ranks_left() * train_B.get_ranks_left();
  ::boba::Array<size_t, dimension> new_ranks_right = train_A.get_ranks_right() * train_B.get_ranks_right();
  boba_always_assert_equal(train_A.sizes(), train_B.sizes(), "sizes must match");
  if (product(new_ranks_left * new_ranks_right) == 0)
  {
    output.fill_with_zeros();
    return output;
  }

  checkpoint();
  for (size_t d = 0; d < dimension; d++)
  {
    auto rank_left_indexer = ::boba::Multiindexer<2>({train_A.get_ranks_left(d), train_B.get_ranks_left(d)});
    auto rank_right_indexer = ::boba::Multiindexer<2>({train_A.get_ranks_right(d), train_B.get_ranks_right(d)});
    auto A_core_view = train_A.cores[d].const_view();
    auto B_core_view = train_B.cores[d].const_view();
    checkpoint();
    output.cores[d].resize({rank_left_indexer.size(), output.sizes(d), rank_right_indexer.size()});
    auto temp_view = output.cores[d].view();
    checkpoint();
    ::boba::loop<space, 3>(temp_view.sizes(),
                           [=] __boba_host_device__(::boba::Array<size_t, 3> ijk)
    {
      auto [rank_left, index, rank_right] = ijk;
      auto [rank_A_left, rank_B_left] = rank_left_indexer.multiindex(rank_left);
      auto [rank_A_right, rank_B_right] = rank_right_indexer.multiindex(rank_right);
      const auto value_A = A_core_view({rank_A_left, index, rank_A_right});
      const auto value_B = B_core_view({rank_B_left, index, rank_B_right});
      temp_view({rank_left, index, rank_right}) = value_A * value_B;
    });
    checkpoint();
  }
  checkpoint();
  return output;
}

/**
 * \brief Compute the elementwise square of a tensor train.
 *
 * \param[in] train input tensor train
 * \return tensor train containing the elementwise square
 */

template <size_t dimension, execution_space space, typename data_t>
TensorTrain<dimension, space, data_t> elementwise_square(
  TensorTrain<dimension, space, data_t> const& train)
{
  BOBA_CALI_MARK

  TensorTrain<dimension, space, data_t> output(train.sizes());

  constexpr size_t power = 2;
  checkpoint();

  ::boba::Array<size_t, dimension> new_ranks_left;
  ::boba::Array<size_t, dimension> new_ranks_right;
  for (size_t d = 0; d < dimension; d++)
  {
    new_ranks_left[d] = number_nonincreasing_multiindices<power>(train.get_ranks_left(d));
    new_ranks_right[d] = number_nonincreasing_multiindices<power>(train.get_ranks_right(d));
    if ((new_ranks_left[d] == 0) or (new_ranks_right[d] == 0))
    {
      output.fill_with_zeros();
      return output;
    }
  }
  checkpoint();
  for (size_t d = 0; d < dimension; d++)
  {
    auto rank_left_indexer = ::boba::SimplicialMultiindexer<power>(train.get_ranks_left(d));
    auto rank_right_indexer = ::boba::SimplicialMultiindexer<power>(train.get_ranks_right(d));
    auto input_core_view = train.cores[d].const_view();
    checkpoint();
    Tensor<3, space, data_t> temp_core({rank_left_indexer.total_size(), train.sizes(d), rank_right_indexer.total_size()});
    auto temp_view = temp_core.view();
    checkpoint();
    ::boba::loop<space, 3>(temp_core.sizes(),
                           [=] __boba_host_device__(::boba::Array<size_t, 3> ijk)
    {
      auto [rank_left, index, rank_right] = ijk;
      auto [rank_A_left, rank_B_left] = rank_left_indexer.multiindex(rank_left);
      auto [rank_A_right, rank_B_right] = rank_right_indexer.multiindex(rank_right);
      const auto value_A = input_core_view({rank_A_left, index, rank_A_right});
      const auto value_B = input_core_view({rank_B_left, index, rank_B_right});

      data_t temp_value = value_A * value_B;

      if (rank_A_left == rank_B_left and rank_A_right != rank_B_right)
      {
        temp_value *= static_cast<data_t>(2.0);
      }
      else if (rank_A_left != rank_B_left and rank_A_right != rank_B_right)
      {
        const auto value_Aprime = input_core_view({rank_A_left, index, rank_B_right});
        const auto value_Bprime = input_core_view({rank_B_left, index, rank_A_right});
        temp_value += value_Aprime * value_Bprime;
      }

      temp_view({rank_left, index, rank_right}) = temp_value;
    });

    checkpoint();
    output.cores[d] = temp_core;
  }
  checkpoint();
  return output;
}

/**
 * \brief Iterative computation of the elementwise power of a tensor train.
 *
 * Repeated squaring is used, with rounding between powers.
 *
 * \param[in] train input tensor train
 * \tparam power exponent to raise the tensor train to
 * \return tensor train raised to the requested power
 */
template <size_t power, size_t dimension, execution_space space, typename data_t>
::boba::TensorTrain<dimension, space, data_t> elementwise_power_iterative(
  ::boba::TensorTrain<dimension, space, data_t>& train)
{
  TensorTrain<dimension, space, data_t> output(train.sizes());

  if (power == 0)
  {
    output.fill_with(1.0);
    return output;
  }
  if (power == 1)
  {
    output = train;
    return output;
  }

  constexpr bool is_power_even = is_even(power);
  constexpr size_t half_power = is_power_even ? power / 2 : (power - 1) / 2;
  ::boba::TensorTrain<dimension, space, data_t> tt_to_half_power(train.sizes());
  if (half_power > 1)
  {
    tt_to_half_power = elementwise_power_iterative<half_power>(train);
  }
  else
  {
    tt_to_half_power = train;
  }
  output = boba::elementwise_square(tt_to_half_power);
  if (!is_power_even)
  {
    output.round();
    auto temp = boba::elementwise_product(output, train);
    output = temp;
  }
  output.round();
  return output;
}

/**
 * \brief Direct computation of elementwise power, that is a Hadamard product of a tensor train with itself 'power' times.  Direct method computes the power directly without
 * intermediate power w/ rounding between them. Relies on looping over permutations of multiindices.
 *
 * This computes the power directly, without intermediate rounding.
 *
 * \param[in] train input tensor train
 * \tparam power exponent to raise the tensor train to
 * \return tensor train raised to the requested power
 */

template <size_t power, size_t dimension, execution_space space, typename data_t>
TensorTrain<dimension, space, data_t> elementwise_power_direct(
  TensorTrain<dimension, space, data_t> const& train)
{
  BOBA_CALI_MARK

  TensorTrain<dimension, space, data_t> output(train.sizes());
  if (power == 0)
  {
    output.fill_with(1.0);
    return output;
  }
  if (power == 1)
  {
    output = train;
    return output;
  }

  checkpoint();

  ::boba::Array<size_t, dimension> new_ranks_left;
  ::boba::Array<size_t, dimension> new_ranks_right;
  for (size_t d = 0; d < dimension; d++)
  {
    new_ranks_left[d] = number_nonincreasing_multiindices<power>(train.get_ranks_left(d));
    new_ranks_right[d] = number_nonincreasing_multiindices<power>(train.get_ranks_right(d));
    if ((new_ranks_left[d] == 0) or (new_ranks_right[d] == 0))
    {
      output.fill_with_zeros();
      return output;
    }
  }
  checkpoint();
  for (size_t d = 0; d < dimension; d++)
  {
    auto rank_left_indexer = ::boba::SimplicialMultiindexer<power>(train.get_ranks_left(d));
    auto rank_right_indexer = ::boba::SimplicialMultiindexer<power>(train.get_ranks_right(d));
    auto input_core_view = train.cores[d].const_view();
    checkpoint();
    Tensor<3, space, data_t> temp_core({rank_left_indexer.total_size(), train.sizes(d), rank_right_indexer.total_size()});
    auto temp_view = temp_core.view();
    checkpoint();
    ::boba::loop<space, 3>(temp_core.sizes(),
                           [=] __boba_host_device__(::boba::Array<size_t, 3> ijk)
    {
      auto [rank_left, index, rank_right] = ijk;
      auto left_ranks = rank_left_indexer.multiindex(rank_left);
      auto right_ranks = rank_right_indexer.multiindex(rank_right);

      size_t number_permutations_of_right_ranks = rank_right_indexer.number_permutations(right_ranks);

      data_t temp_value = 0.;

      for (size_t p = 0; p < number_permutations_of_right_ranks; p++)
      {
        data_t temp_temp_value = 1.0;
        for (size_t term = 0; term < power; term++)
        {
          temp_temp_value *= input_core_view({left_ranks[term], index, right_ranks[term]});
        }
        temp_value += temp_temp_value;
        rank_right_indexer.get_next_permutation(right_ranks);
      }

      temp_view({rank_left, index, rank_right}) = temp_value;
    });

    checkpoint();
    output.cores[d] = temp_core;
  }
  checkpoint();
  return output;
}

/**
 * @brief Elementwise power of a TT with option to use low-memory method ( default; iterative squaring, w/ product for odd powers ) or direct method
 * ( looping over permutations of multiindices )
 *
 * @param[in] train Input tensor train
 * @tparam power Exponent to raise the tensor train to
 * @param low_memory If true, use low-memory iterative method; if false, use direct method
 * @return a TT which is the elementwise power of the input tensor train
 */
template <size_t power, size_t dimension, execution_space space, typename data_t>
::boba::TensorTrain<dimension, space, data_t> elementwise_power(
  ::boba::TensorTrain<dimension, space, data_t>& train,
  bool low_memory = true)
{
  if (low_memory)
  {
    return elementwise_power_iterative<power>(train);
  }
  else
  {
    return elementwise_power_direct<power>(train);
  }
}

// -------------------------------------------------------------------------
// sum and round
// -------------------------------------------------------------------------

/**
 * @brief Computes the sum of a sequence of tensor trains,
 * rounding after each addition.
 *
 * Each tensor in @p sequence is added to an accumulator, followed by a rounding
 * operation to maintain a low-rank representation.
 *
 * @tparam dimension  Number of tensor dimensions.
 * @tparam space      Execution space (e.g., host or device).
 * @tparam data_t  Value type for tensor entries.
 * @param sequence std::span of TensorTrain objects to be summed and rounded.
 * @return A new TensorTrain representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
TensorTrain<dimension, space, data_t>
sum_and_round(std::span<const TensorTrain<dimension, space, data_t>> sequence)
{
  using tt_t = TensorTrain<dimension, space, data_t>;

  boba_always_assert(!sequence.empty(), "Cannot perform sum_and_round on an empty sequence.");

  // Use the first tensor to infer shape
  const auto& first = sequence.front();
  tt_t output(first.sizes());

  for (const auto& item : sequence)
  {
    output += item;
    output.round();
  }

  return output;
}

/**
 * @brief Lightweight adapter for converting std::vector into std::span for
 * addition and rounding.
 *
 * @tparam dimension  Number of tensor dimensions.
 * @tparam space      Execution space.
 * @tparam data_t  Value type for tensor entries.
 * @param sequence std::vector of TensorTrain objects to be summed and rounded.
 * @return A new TensorTrain representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
TensorTrain<dimension, space, data_t>
sum_and_round(const std::vector<TensorTrain<dimension, space, data_t>>& sequence)
{
  using tt_t = TensorTrain<dimension, space, data_t>;
  return sum_and_round(std::span<const tt_t>{sequence});
}

/**
 * @brief Lightweight adapter for converting std::initializer_list into
 * std::span for addition and rounding.
 *
 * @note This overload is convenient, but it may copy the input tensors into the
 * initializer-list backing array. Prefer the std::span or std::vector overloads
 * for performance-sensitive code.
 *
 * @tparam dimension  Number of tensor dimensions.
 * @tparam space      Execution space.
 * @tparam data_t  Value type for tensor entries.
 * @param sequence Initializer list of TensorTrain objects to be summed and rounded.
 * @return A new TensorTrain representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
TensorTrain<dimension, space, data_t>
sum_and_round(const std::initializer_list<TensorTrain<dimension, space, data_t>> sequence)
{
  using tt_t = TensorTrain<dimension, space, data_t>;
  return sum_and_round(std::span<const tt_t>{sequence.begin(), sequence.size()});
}

// -------------------------------------------------------------------------------------
// Section: Inner product
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Inner product <A, B> of two tensor trains.
 */

template <size_t dimension, execution_space space, typename data_t>
auto inner_product(
  TensorTrain<dimension, space, data_t> const& tt_A,
  TensorTrain<dimension, space, data_t> const& tt_B)
{
  checkpoint();
  return tt_A.inner_product(tt_B);
}

// -------------------------------------------------------------------------------------
// Section: Norm differences
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Frobenius norm of a tt
 */

template <size_t dimension, execution_space space, typename data_t>
auto norm_difference_frobenius(
  TensorTrain<dimension, space, data_t> const& tt_A,
  TensorTrain<dimension, space, data_t> const& tt_B)
{
  BOBA_CALI_MARK
  checkpoint();
  TensorTrain<dimension, space, data_t> temp = tt_A - tt_B;
  checkpoint();
  temp.rename("norm_difference_temp");
  temp.round();
  data_t diff_frobenius = ::boba::norm_frobenius(temp);
  return diff_frobenius;
}

// -------------------------------------------------------------------------------------
// Section: FFT
// -------------------------------------------------------------------------------------

/**
 * \brief
 * fft along one dimension of the decomposition, consistent with fft_along_dimension
 */

template <size_t dimension, execution_space space, typename data_t>
[[nodiscard]]
TensorTrain<dimension, space, complex<data_t>> fft_along_dimension(
  const TensorTrain<dimension, space, complex<data_t>>& input,
  index_t transform_dimension,
  fft_operation operation)
{
  auto output = input;
  auto extent_dimension = 1_z;
  output.cores[transform_dimension] = fft_along_dimension<3, space, double>(input.cores[transform_dimension], extent_dimension, operation);
  return output;
}

} // namespace boba

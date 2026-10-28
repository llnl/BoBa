// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <vector>

namespace boba
{

/**
 * \brief One-site TT cross (OneSiteCross) for tensor trains whose number of cores is chosen at run time.
 *
 * Same algorithm and options as OneSiteCross; the function is called with the full multi-index as a
 * `std::vector<size_t> const&` and the result is a RuntimeTensorTrain.
 *
 * @tparam data_t Scalar value type.
 */
template <typename data_t>
struct RuntimeOneSiteCross : public OneSiteCross<data_t>
{
  using entry_function_t = typename OneSiteCross<data_t>::entry_function_t;

  /**
   * @brief Approximates the tensor `function(i_0, ..., i_{d-1})`, `0 <= i_k < sizes[k]`, by one-site cross.
   * @param sizes Mode sizes (the number of cores is sizes.size()).
   * @param function Entry-evaluation callable `data_t(std::vector<size_t> const&)`.
   * @param random_generator UniformRandomBitGenerator used for the random fibers and the validation set.
   * @return A host-space tensor train approximation.
   */
  template <typename lambda_t, typename random_generator_t>
  [[nodiscard]]
  ::boba::RuntimeTensorTrain<host_space, data_t> apply(
    std::vector<size_t> const& sizes,
    lambda_t&& function,
    random_generator_t& random_generator)
  {
    entry_function_t entry_function(std::forward<lambda_t>(function));
    return this->apply_erased(sizes, entry_function, random_generator);
  }

  /**
   * @brief Applies runtime one-site cross using the process-wide default generator.
   */
  template <typename lambda_t>
  [[nodiscard]]
  ::boba::RuntimeTensorTrain<host_space, data_t> apply(
    std::vector<size_t> const& sizes,
    lambda_t&& function)
  {
    auto& random_generator = ::boba::random::default_context();
    return apply(sizes, std::forward<lambda_t>(function), random_generator);
  }
};

}  // namespace boba

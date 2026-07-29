// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

// -------------------------------------------------------------------------
// NaN checks
// -------------------------------------------------------------------------

template <size_t dimension, ::boba::execution_space space, typename data_t>
void nan_check(Tucker<dimension, space, data_t> const& tucker)
{
  for (size_t d = 0; d < dimension; d++)
  {
    ::boba::nan_check(tucker.cores[d]);
  }
  ::boba::nan_check(tucker.R_core);
}

// -------------------------------------------------------------------------
// norms
// -------------------------------------------------------------------------

/**
 * \brief Frobenius norm of the Tucker tensor in the sense of vectors.
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
auto norm_frobenius(Tucker<dimension, space, data_t> const& tucker)
{
  BOBA_CALI_MARK
  const auto product = tucker.inner_product(tucker);
  const auto abs_product = boba::abs(product);
  return boba::sqrt(abs_product);
}

// -------------------------------------------------------------------------
// sum and round
// -------------------------------------------------------------------------

/**
 * @brief Computes the sum of a sequence of Tucker tensors,
 * rounding after each addition.
 *
 * Each tensor in @p sequence is added to an accumulator, followed by a rounding
 * operation to maintain a low-rank representation.
 *
 * @tparam dimension  Number of tensor dimensions.
 * @tparam space      Execution space (e.g., host or device).
 * @tparam data_t  Value type for tensor entries.
 * @param sequence std::span of HierarchicalTucker tensors to be summed and rounded.
 * @return A new Tucker tensor representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
Tucker<dimension, space, data_t>
sum_and_round(std::span<const Tucker<dimension, space, data_t>> sequence)
{
  using tucker_t = Tucker<dimension, space, data_t>;

  boba_always_assert(!sequence.empty(), "Cannot perform sum_and_round on an empty sequence.");

  // Use the first tensor to infer shape
  const auto& first = sequence.front();
  tucker_t output(first.sizes());

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
 * @param sequence std::vector of Tucker tensors to be summed and rounded.
 * @return A new Tucker tensor representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
Tucker<dimension, space, data_t>
sum_and_round(const std::vector<Tucker<dimension, space, data_t>>& sequence)
{
  using tucker_t = Tucker<dimension, space, data_t>;
  return sum_and_round(std::span<const tucker_t>{sequence});
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
 * @param sequence Initializer list of Tucker tensors to be summed and rounded.
 * @return A new Tucker tensor representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
Tucker<dimension, space, data_t>
sum_and_round(const std::initializer_list<Tucker<dimension, space, data_t>> sequence)
{
  using tucker_t = Tucker<dimension, space, data_t>;
  return sum_and_round(std::span<const tucker_t>{sequence.begin(), sequence.size()});
}

// -------------------------------------------------------------------------------------
// Section: Norm differences
// -------------------------------------------------------------------------------------

/**
 * \brief Compute the Frobenius norm difference between two Tucker tensors.
 *
 * \param[in] tucker_A first Tucker tensor
 * \param[in] tucker_B second Tucker tensor
 * \return Frobenius norm of the difference
 */

template <size_t dimension, execution_space space, typename data_t>
auto norm_difference_frobenius(
  Tucker<dimension, space, data_t> const& tucker_A,
  Tucker<dimension, space, data_t> const& tucker_B)
{
  BOBA_CALI_MARK
  checkpoint();
  Tucker<dimension, space, data_t> temp = tucker_A - tucker_B;
  checkpoint();
  temp.rename("norm_difference_temp");
  temp.round();
  data_t diff_frobenius = ::boba::norm_frobenius(temp);
  return diff_frobenius;
}

// -------------------------------------------------------------------------------------
// Section: Multiply
// -------------------------------------------------------------------------------------

/**
 * \brief Perform the elementwise product of two Tucker tensors.
 *
 * \param[in] tucker_A first Tucker tensor
 * \param[in] tucker_B second Tucker tensor
 * \return Tucker tensor containing the Hadamard product
 */

template <size_t dimension, execution_space space, typename data_t>
Tucker<dimension, space, data_t> elementwise_product(
  Tucker<dimension, space, data_t> const& tucker_A,
  Tucker<dimension, space, data_t> const& tucker_B)
{
  BOBA_CALI_MARK

  Tucker<dimension, space, data_t> output;

  for (size_t d = 0; d < dimension; d++)
  {
    auto new_core = mode_n_tensor_product(tucker_A.cores[d], tucker_B.cores[d], 1);
    output.cores[d] = new_core;
  }

  auto new_R_core = tensor_product(tucker_A.R_core, tucker_B.R_core);
  output.R_core = new_R_core;
  return output;
}

// -------------------------------------------------------------------------------------
// Section: Inner product
// -------------------------------------------------------------------------------------

/**
 * \brief Compute the inner product of two Tucker tensors.
 *
 * \param[in] tuck_A first Tucker tensor
 * \param[in] tuck_B second Tucker tensor
 * \return inner product of the two inputs
 */

template <size_t dimension, execution_space space, typename data_t>
data_t inner_product(
  Tucker<dimension, space, data_t> const& tuck_A,
  Tucker<dimension, space, data_t> const& tuck_B)
{
  checkpoint();
  return tuck_A.inner_product(tuck_B);
}

/**
 * \brief
 * Write the tt to a file in a way consistent with Tensor::write_to_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_file(const Tucker<dimension, space, data_t>& Tucker, std::string_view filename = "")
{
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = Tucker.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  for (size_t d = 0; d < dimension; d++)
  {
    write_to_file(Tucker.cores[d], print_filename + "_core_" + std::to_string(d));
  }
  write_to_file(Tucker.R_core, print_filename + "_R_core");
}

/**
 * \brief
 * Read from a file generated from write_to_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_file(Tucker<dimension, space, data_t>& Tucker, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = Tucker.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  for (size_t d = 0; d < dimension; d++)
  {
    read_from_file(Tucker.cores[d], print_filename + "_core_" + std::to_string(d));
  }
  read_from_file(Tucker.R_core, print_filename + "_R_core");
}

/**
 * \brief
 * Write the tt to a MATLAB mat-file in a way consistent with Tensor::write_to_mat_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_mat_file(const Tucker<dimension, space, data_t>& Tucker, std::string_view filename = "")
{
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = Tucker.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  detail::MatFile matlab_file(print_filename, "w");

  for (size_t d = 0; d < dimension; d++)
  {
    std::string print_core = print_filename + "_core_" + std::to_string(d);
    matlab_file.write_array(print_core, Tucker.cores[d]);
  }
  std::string print_core = print_filename + "_R_core";
  matlab_file.write_array(print_core, Tucker.R_core);
}

/**
 * \brief
 * Read from a file generated from write_to_mat_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_mat_file(Tucker<dimension, space, data_t>& Tucker, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = Tucker.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  detail::MatFile matlab_file(print_filename, "r");

  for (size_t d = 0; d < dimension; d++)
  {
    std::string print_core = print_filename + "_core_" + std::to_string(d);
    matlab_file.read_array(print_core, Tucker.cores[d]);
  }
  std::string print_core = print_filename + "_R_core";
  matlab_file.read_array(print_core, Tucker.R_core);
}

/**
 * \brief
 * Write the tt to a hdf5 mat-file in a way consistent with Tensor::write_to_hdf5_file
 * mode is used to determine if file should be truncated on open "w", or untruncated "rw"
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_hdf5_file(const Tucker<dimension, space, data_t>& Tucker, std::string_view filename, std::string object_name = "", std::string mode = "w")
{
  if (object_name.empty())
  {
    object_name = Tucker.name();
  }

  detail::HDF5File h5_file(filename, mode);

  for (size_t d = 0; d < dimension; d++)
  {
    std::string print_core = object_name + "_core_" + std::to_string(d);
    h5_file.write_array(print_core, Tucker.cores[d]);
  }
  std::string print_core = object_name + "_R_core";
  h5_file.write_array(print_core, Tucker.R_core);
}

/**
 * \brief
 * Read from a file generated from write_to_hdf5_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_hdf5_file(Tucker<dimension, space, data_t>& Tucker, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = Tucker.name();
  }

  detail::HDF5File h5_file(filename, "r");

  for (size_t d = 0; d < dimension; d++)
  {
    std::string print_core = object_name + "_core_" + std::to_string(d);
    h5_file.read_array(print_core, Tucker.cores[d]);
  }
  std::string print_core = object_name + "_R_core";
  h5_file.read_array(print_core, Tucker.R_core);
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
Tucker<dimension, space, complex<data_t>> fft_along_dimension(
  const Tucker<dimension, space, complex<data_t>>& input,
  index_t transform_dimension,
  fft_operation operation)
{
  auto output = input;
  auto extent_dimension = 0_z;
  output.cores[transform_dimension].reshape(fft_along_dimension<2, space, data_t>(input.cores[transform_dimension], extent_dimension, operation));
  return output;
}

} // namespace boba

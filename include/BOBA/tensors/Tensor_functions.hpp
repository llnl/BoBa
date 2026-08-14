// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef BOBA_EIGEN_TENSOR
#include "unsupported/Eigen/CXX11/Tensor"
#endif

namespace boba
{
template <size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, data_t> nonnegative_part(
  Tensor<dimension, space, data_t> const& input)
{
  return input.nonnegative_part();
}

template <execution_space space, typename data_t>
Matrix<space, data_t> nonnegative_part(
  Matrix<space, data_t> const& input)
{
  return input.nonnegative_part();
}

template <execution_space space, typename data_t>
Vector<space, data_t> nonnegative_part(
  Vector<space, data_t> const& input)
{
  return input.nonnegative_part();
}

template <size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, data_t> nonpositive_part(
  Tensor<dimension, space, data_t> const& input)
{
  return input.nonpositive_part();
}

template <execution_space space, typename data_t>
Matrix<space, data_t> nonpositive_part(
  Matrix<space, data_t> const& input)
{
  return input.nonpositive_part();
}

template <execution_space space, typename data_t>
Vector<space, data_t> nonpositive_part(
  Vector<space, data_t> const& input)
{
  return input.nonpositive_part();
}

/**
 * \brief
 * Tensor inner product, aka dot product
 */

template <size_t dimension, execution_space space, typename data_t>
data_t inner_product(
  const Tensor<dimension, space, data_t>& tensor_A,
  const Tensor<dimension, space, data_t>& tensor_B)
{
  BOBA_CALI_OBJECT_MARK

  auto tensor_A_view = tensor_A.const_view();
  auto tensor_B_view = tensor_B.const_view();

  boba_always_assert_equal(tensor_A_view.sizes(), tensor_B_view.sizes(), "Inconsistent tensor sizes.");

  if (tensor_A.size() == 0)
  {
    return data_t{0};
  }
  if (tensor_B.size() == 0)
  {
    return data_t{0};
  }

  if constexpr (
    (space == ::boba::execution_space::CUDA) and
    ::boba::boba_cuda_enabled() and
    ::boba::boba_cuda_libs_enabled() and
    (std::is_same_v<data_t, float> or
     std::is_same_v<data_t, double> or
     std::is_same_v<data_t, complex<float>> or
     std::is_same_v<data_t, complex<double>>))
  {
    return ::boba::detail::cublas_dot(tensor_A_view, tensor_B_view);
  }
  else if constexpr (
    (space == ::boba::execution_space::HIP) and
    ::boba::boba_hip_enabled() and
    ::boba::boba_hip_libs_enabled() and
    (std::is_same_v<data_t, float> or
     std::is_same_v<data_t, double> or
     std::is_same_v<data_t, complex<float>> or
     std::is_same_v<data_t, complex<double>>))
  {
    return ::boba::detail::hipblas_dot(tensor_A_view, tensor_B_view);
  }
  else if constexpr (
    (space == ::boba::execution_space::CPU) and
    ::boba::is_cpu_enabled() and
    ::boba::boba_eigen_tensor_enabled())
  {
    return ::boba::detail::eigen_inner_product(tensor_A_view, tensor_B_view);
  }

  data_t value = data_t{0};
  ::boba::sum_reduce<space>(value, index_t(0), tensor_A_view.size(), [=] __boba_host_device__(index_t i, sum_reducer_operator<data_t> & local_value)
  {
    local_value += tensor_A_view(i) * tensor_B_view(i);
  });

  return value;
}

/**
 * \brief
 * Apply a function to every element of a tensor
 */

template <size_t dimension, execution_space space, typename data_t, typename func_t>
[[nodiscard]]
Tensor<dimension, space, data_t> apply_function(
  const Tensor<dimension, space, data_t>& tensor_A,
  const func_t& function)
{
  BOBA_CALI_OBJECT_MARK

  Tensor<dimension, space, data_t> output(tensor_A.sizes());

  auto in_view = tensor_A.const_view();
  auto out_view = output.view();

  ::boba::loop<space, 1>(tensor_A.size(),
                         [=] __boba_host_device__(size_t i)
  {
    out_view(i) = function(in_view(i));
  });
  return output;
}

/**
 * \brief
 * Apply a function to every element of a matrix
 */

template <execution_space space, typename data_t, typename func_t>
[[nodiscard]]
Matrix<space, data_t> apply_function(
  const Matrix<space, data_t>& matrix_A,
  const func_t& function)
{
  BOBA_CALI_OBJECT_MARK

  Matrix<space, data_t> output(matrix_A.sizes());

  auto in_view = matrix_A.const_view();
  auto out_view = output.view();

  ::boba::loop<space, 1>(matrix_A.size(),
                         [=] __boba_host_device__(size_t i)
  {
    out_view(i) = function(in_view(i));
  });
  return output;
}

/**
 * \brief
 * Apply a function to every element of a vector
 */

template <execution_space space, typename data_t, typename func_t>
[[nodiscard]]
Vector<space, data_t> apply_function(
  const Vector<space, data_t>& vector_A,
  const func_t& function)
{
  BOBA_CALI_OBJECT_MARK

  Vector<space, data_t> output(vector_A.sizes());

  auto in_view = vector_A.const_view();
  auto out_view = output.view();

  ::boba::loop<space, 1>(vector_A.size(),
                         [=] __boba_host_device__(size_t i)
  {
    out_view(i) = function(in_view(i));
  });
  return output;
}

/**
 * @brief Raise each element of a tensor to a specified power
 *
 * @param[in] tensor_A tensor
 * @return tensor such that output(i) = pow(tensor_A(i), power)
 */

template <size_t dimension, execution_space space, typename data_t>
[[nodiscard]]
Tensor<dimension, space, data_t> power(
  const Tensor<dimension, space, data_t>& tensor_A,
  const data_t _power)
{
  BOBA_CALI_OBJECT_MARK
  return apply_function(tensor_A, [=] __boba_host_device__(data_t x)
  {
    return ::boba::pow(x, _power);
  });
}

/**
 * @brief Raise each element of a matrix to a specified power
 *
 * @param[in] matrix_A matrix
 * @return matrix such that output(i) = pow(matrix_A(i), power)
 */

template <execution_space space, typename data_t>
[[nodiscard]]
Matrix<space, data_t> power(
  const Matrix<space, data_t>& matrix_A,
  const data_t _power)
{
  BOBA_CALI_OBJECT_MARK
  using base = typename Matrix<space, data_t>::base;
  return Matrix<space, data_t>(power(static_cast<const base&>(matrix_A), _power));
}

/**
 * @brief Raise each element of a vector to a specified power
 *
 * @param[in] vector_A vector
 * @return tensor such that output(i) = pow(vector(i), power)
 */

template <execution_space space, typename data_t>
[[nodiscard]]
Vector<space, data_t> power(
  const Vector<space, data_t>& vector_A,
  const data_t _power)
{
  BOBA_CALI_OBJECT_MARK
  using base = typename Vector<space, data_t>::base;
  return Vector<space, data_t>(power(static_cast<const base&>(vector_A), _power));
}

/**
 * @brief Perform the elementwise_product (aka Hadamard product) of inputs
 *
 * @param[in] tensor_A tensor
 * @param[in] tensor_B tensor
 * @return tensor which is the Hadamard product tensor_A * tensor_B
 */

template <size_t dimension, execution_space space, typename data_t>
[[nodiscard]]
Tensor<dimension, space, data_t> elementwise_product(
  const Tensor<dimension, space, data_t>& tensor_A,
  const Tensor<dimension, space, data_t>& tensor_B)
{
  BOBA_CALI_OBJECT_MARK

  auto A_view = tensor_A.const_view();
  auto B_view = tensor_B.const_view();
  boba_assert_equal(tensor_A.sizes(), tensor_B.sizes(), "Incompatible tensors.");

  Tensor<dimension, space, data_t> output(tensor_A.sizes());
  auto output_view = output.view();

  ::boba::loop<space, 1>(output.size(),
                         [=] __boba_host_device__(size_t index)
  {
    output_view(index) = A_view(index) * B_view(index);
  });

  return output;
}

/**
 * @brief Perform the elementwise_product (aka Hadamard product) of inputs
 *
 * @param[in] matrix_A matrix
 * @param[in] matrix_B matrix
 * @return matrix which is the Hadamard product matrix_A * matrix_B
 */

template <execution_space space, typename data_t>
[[nodiscard]]
Matrix<space, data_t> elementwise_product(
  const Matrix<space, data_t>& matrix_A,
  const Matrix<space, data_t>& matrix_B)
{
  BOBA_CALI_OBJECT_MARK
  using base = typename Matrix<space, data_t>::base;
  return Matrix<space, data_t>(elementwise_product(static_cast<const base&>(matrix_A), static_cast<const base&>(matrix_B)));
}

/**
 * @brief Perform the elementwise_product (aka Hadamard product) of inputs
 *
 * @param[in] vector_A vector
 * @param[in] vector_B vector
 * @return vector which is the Hadamard product vector_A * vector_B
 */

template <execution_space space, typename data_t>
[[nodiscard]]
Vector<space, data_t> elementwise_product(
  const Vector<space, data_t>& vector_A,
  const Vector<space, data_t>& vector_B)
{
  BOBA_CALI_OBJECT_MARK
  using base = typename Vector<space, data_t>::base;
  return Vector<space, data_t>(elementwise_product(static_cast<const base&>(vector_A), static_cast<const base&>(vector_B)));
}

/**
 * @brief permutes a given tensor
 *
 * @param [in] original_indices The descriptions of the input indices, e.g. {"ax", "k", "c2"}
 * @param [in, out] tensor_permutation The tensor to be permuted in-place
 * @param [in] permuted_indices A valid permutation of original_indices, e.g. {"c2", "ax", "k"}
 *
 */

template <execution_space space, size_t dimension, typename data_t, bool force_naive = false, typename DimLabel_t = std::string>
void permute(
  const Array<DimLabel_t, dimension>& original_indices,
  Tensor<dimension, space, data_t>& tensor_permutation,
  const Array<DimLabel_t, dimension>& permuted_indices)
{
  Array<index_t, dimension> permutations;
  for (size_t d = 0; d < dimension; d++)
  {
    auto this_index = permuted_indices[d];
    index_t id = 0;
    for (size_t di = 0; di < dimension; di++)
    {
      if (this_index == original_indices[di])
      {
        id = di;
      }
    }
    permutations[d] = id;
  }
  permute<space, dimension, data_t, force_naive>(tensor_permutation, permutations);
}

/**
 * @brief permutes a given Tensor in place
 * This is analogous to MATLAB's permute command
 *
 * @param [in, out] tensor_permutation Tensor to be permuted, in-place
 * @param [in] permutations A set of indices indicating how to permute the tensor
 */

template <execution_space space, size_t dimension, typename data_t, bool force_naive = false>
void permute(
  Tensor<dimension, space, data_t>& tensor_permutation,
  const Array<index_t, dimension>& permutations)
{
  auto new_view = tensor_permutation.view();
  permute<space, dimension, data_t, force_naive>(new_view, permutations);
  tensor_permutation.reshape(new_view.sizes());
}

/**
 * @brief Creates a permutation of the tensorial data in TensorView
 * This is analogous to MATLAB's permute command
 *
 * @param [in, out] tensor_permutation Tensorial data to be permuted, in-place
 * @param [in] permutations A set of indices indicating how to permute the tensor
 */

template <execution_space space, size_t dimension, typename data_t, bool force_naive = false>
void permute(
  TensorView<DefaultAccessor<data_t>, dimension>& tensor_permutation,
  const Array<index_t, dimension>& permutations)
{
  boba_always_assert(is_valid_permutation(permutations), "Invalid permutation.");
  if (permutations == range<index_t, dimension>())
  {
    // permutation is the identity (0, 1, ..., dimension)
    return;
  }

  Array<index_t, dimension> local_permutations = permutations;
  Array<index_t, dimension> local_sizes_old = tensor_permutation.sizes();
  Array<index_t, dimension> local_sizes_new = permute(local_sizes_old, local_permutations);

  Tensor<dimension, space, data_t> old_tensor(local_sizes_old);

  checkpoint();
  ::boba::detail::memcpy<space, space>(
    old_tensor.data(), tensor_permutation.const_data(), static_cast<size_t>(old_tensor.size()));

  // Resize output tensor
  checkpoint();
  tensor_permutation.reshape(local_sizes_new);

  if (tensor_permutation.size() == 0)
  {
    return;
  }

  //
  // Perform permutation
  //
  if constexpr (
    (space == ::boba::execution_space::CUDA) and
    ::boba::boba_cuda_enabled() and
    ::boba::boba_cutensor_enabled() and
    not(force_naive))
  {
    ::boba::detail::cutensor_permute(old_tensor.const_view(), tensor_permutation, local_permutations);
    return;
  }
  else if constexpr (
    (space == ::boba::execution_space::HIP) and
    ::boba::boba_hip_enabled() and
    ::boba::boba_hiptensor_enabled() and
    not(force_naive))
  {
    if constexpr (std::is_same<data_t, float>::value)
    {
      ::boba::detail::hiptensor_permute(old_tensor.const_view(), tensor_permutation, local_permutations);
      return;
    }
  }
  else if constexpr (
    (space == ::boba::execution_space::CPU) and
    ::boba::is_cpu_enabled() and
    ::boba::boba_metal_enabled() and
    not(force_naive))
  {
    ::boba::detail::metal_permute(old_tensor.const_view(), tensor_permutation, local_permutations);
    return;
  }
  else if constexpr (
    (space == ::boba::execution_space::CPU) and
    ::boba::is_cpu_enabled() and
    ::boba::boba_eigen_tensor_enabled() and
    not(force_naive))
  {
    ::boba::detail::eigen_tensor_permute(old_tensor.const_view(), tensor_permutation, local_permutations);
    return;
  }

  //
  // Fallback scheme
  //
  auto old_view = old_tensor.const_view();
  ::boba::loop<space, 1>(tensor_permutation.size(),
                         [=] __boba_host_device__(index_t i)
  {
    auto multiindex_old = old_view.multiindex(i);
    auto multiindex_new = permute(multiindex_old, local_permutations);
    auto value = old_view(multiindex_old);
    tensor_permutation(multiindex_new) = value;
  });
}

/**
 * \brief
 * In-place complex conjugate of tensor
 */

template <size_t dimension, execution_space space, typename data_t>
void complex_conjugate_in_place(Tensor<dimension, space, data_t>& input)
{
  auto input_view = input.view();
  complex_conjugate_in_place<dimension, space, data_t>(input_view);
}

template <size_t dimension, execution_space space, typename data_t>
void complex_conjugate_in_place(TensorView<DefaultAccessor<data_t>, dimension>& input_view)
{
  using real_data_t = real_type_t<data_t>;
  if constexpr (std::is_same_v<data_t, real_data_t>)
  {
    return;
  }

  ::boba::loop<space, 1>(input_view.size(), [=] __boba_host_device__(index_t i)
  {
    input_view(i) = conj(input_view(i));
  });
}

/**
 * @return Complex conjugate of tensor (out-of-place).
 */

template <size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, data_t> get_conj(Tensor<dimension, space, data_t> const& input)
{
  // If the tensor is real valued, simply return the input without a copy (no cost)
  using real_data_t = real_type_t<data_t>;
  if constexpr (std::is_same_v<data_t, real_data_t>)
  {
    return input;
  }

  // Make a copy of the input tensor and conjugate it in-place
  auto output = input;
  complex_conjugate_in_place<dimension, space, data_t>(output);
  return output;
}

/**
 * @return Complex conjugate of matrix (out-of-place).
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> get_conj(Matrix<space, data_t> const& input)
{
  Matrix<space, data_t> output;
  using base = Matrix<space, data_t>::base;
  static_cast<base&>(output) = get_conj(static_cast<base const&>(input));
  return output;
}

/**
 * @return Complex conjugate of vector (out-of-place).
 */

template <execution_space space, typename data_t>
Vector<space, data_t> get_conj(Vector<space, data_t> const& input)
{
  Vector<space, data_t> output;
  using base = Vector<space, data_t>::base;
  static_cast<base&>(output) = get_conj(static_cast<base const&>(input));
  return output;
}

/**
 * \brief
 * Extracts the real part of a tensor.
 */

template <size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, real_type_t<data_t>> get_real(Tensor<dimension, space, data_t> const& input)
{
  using real_data_t = real_type_t<data_t>;

  // If the tensor is real, simply return the input (no cost)
  if constexpr (std::is_same_v<data_t, real_data_t>)
  {
    return input;
  }

  // Otherwise, the data is not real, so we create a real tensor and extract the real part from the input
  Tensor<dimension, space, real_data_t> output(input.sizes());

  auto input_view = input.const_view();
  auto output_view = output.view();

  ::boba::loop<space, 1>(input_view.size(), [=] __boba_host_device__(index_t i)
  {
    output_view(i) = real(input_view(i));
  });

  return output;
}

/**
 * \brief
 * Extracts the imaginary part of a tensor.
 */

template <size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, real_type_t<data_t>> get_imag(Tensor<dimension, space, data_t> const& input)
{
  using real_data_t = real_type_t<data_t>;

  // If the tensor is real, simply return the zero tensor
  if constexpr (std::is_same_v<data_t, real_data_t>)
  {
    Tensor<dimension, space, real_data_t> output(input.sizes());
    output.fill_with(static_cast<real_data_t>(0));
    return output;
  }

  // Otherwise, the data is not real, so we create a real tensor and extract the imaginary part from the input
  Tensor<dimension, space, real_data_t> output(input.sizes());

  auto input_view = input.const_view();
  auto output_view = output.view();

  ::boba::loop<space, 1>(input_view.size(), [=] __boba_host_device__(index_t i)
  {
    output_view(i) = imag(input_view(i));
  });

  return output;
}

/**
 * @brief Given two Arrays of einsum labels (e.g. strings or ints), finds num_common_labels common labels (must be set via template).
 * @tparam num_common_labels How many common labels to search for
 * @param [in] indices_A Array of einsum labels
 * @param [in] indices_B Array of einsum labels
 *
 * @returns Array of common einsum labels
 */

template <size_t num_common_labels, size_t dimension_A, size_t dimension_B, typename DimLabel_t>
Array<DimLabel_t, num_common_labels> get_common_labels(
  const Array<DimLabel_t, dimension_A>& indices_A,
  const Array<DimLabel_t, dimension_B>& indices_B)
{
  Array<DimLabel_t, num_common_labels> common_labels;
  size_t commmon_label = 0;
  for (size_t ia = 0; ia < dimension_A; ia++)
  {
    auto label_A = indices_A[ia];
    for (size_t ib = 0; ib < dimension_B; ib++)
    {
      auto label_B = indices_B[ib];
      if (label_A == label_B)
      {
        common_labels[commmon_label] = label_A;
        commmon_label++;
        if (commmon_label == num_common_labels)
        {
          return common_labels;
        }
      }
    }
  }
  boba_error("Inconsistent labels for indices.");
  return common_labels;
}

/**
 * @brief Finds the indices of einsum labels in a given einsum label Array
 *
 * @param [in] labels Array of einsum labels
 * @param [in] search_labels Array of einsum labels for which to fetch the corresponding indices in labels
 *
 * @returns Array of indices of where each label in search_labels can be found in labels, maintaining the same order as found in search_labels
 */

template <size_t num_search_labels, size_t dimension_A, typename DimLabel_t>
Array<size_t, num_search_labels> get_label_indices(
  const Array<DimLabel_t, dimension_A>& labels,
  const Array<DimLabel_t, num_search_labels>& search_labels)
{
  Array<size_t, num_search_labels> label_indices = filled_array<num_search_labels>(dimension_A);
  size_t current_label_id = 0;
  for (auto label : search_labels)
  {
    for (size_t ia = 0; ia < dimension_A; ia++)
    {
      if (label == labels[ia])
      {
        label_indices[current_label_id] = ia;
        current_label_id++;
      }
    }
  }
  return label_indices;
}

/**
 * @brief Given Arrays of einsum labels contraction indices, finds the labels corresponding to uncontracted dimensions
 *
 * @param [in] labels_A Array of einsum labels for the left tensor
 * @param [in] contraction_indices_A Array of contraction indices for the left tensor
 * @param [in] labels_B Array of einsum labels for the the right tensor
 * @param [in] contraction_indices_B Array of contraction indices for the right tensor
 *
 * @returns Array of einsum labels where we remove contraction_indices_A from labels_A,
 * and contraction_indices_B from labels_B, and then concatenate the results
 */

template <size_t num_contractions, size_t dimension_A, size_t dimension_B, typename DimLabel_t>
Array<DimLabel_t, (dimension_A + dimension_B) - 2 * num_contractions> get_uncontracted_labels(
  const Array<DimLabel_t, dimension_A>& labels_A,
  const Array<size_t, num_contractions>& contraction_indices_A,
  const Array<DimLabel_t, dimension_B>& labels_B,
  const Array<size_t, num_contractions>& contraction_indices_B)
{
  Array<DimLabel_t, (dimension_A + dimension_B) - 2 * num_contractions> uncontracted_labels{labels_A[0]};
  size_t uncontracted_label_id = 0;
  for (size_t ia = 0; ia < dimension_A; ia++)
  {
    bool is_contracted_label = false;
    for (auto contracted_id : contraction_indices_A)
    {
      if (ia == contracted_id)
      {
        is_contracted_label = true;
      }
    }
    if (not(is_contracted_label))
    {
      uncontracted_labels[uncontracted_label_id] = labels_A[ia];
      uncontracted_label_id++;
    }
  }
  for (size_t ib = 0; ib < dimension_B; ib++)
  {
    bool is_contracted_label = false;
    for (auto contracted_id : contraction_indices_B)
    {
      if (ib == contracted_id)
      {
        is_contracted_label = true;
      }
    }
    if (not(is_contracted_label))
    {
      uncontracted_labels[uncontracted_label_id] = labels_B[ib];
      uncontracted_label_id++;
    }
  }

  return uncontracted_labels;
}

/**
 * @brief Contracts two tensors using einsum notation
 * This version supports one or more contracted index pairs.
 *
 * @param [in] labels_A Array of einsum labels for the left tensor
 * @param [in] tensor_A the left tensor
 * @param [in] labels_B Array of einsum labels for the right tensor
 * @param [in] tensor_B the right tensor
 * @param [in] labels_C Array of einsum labels expressing the desired order of the uncontracted dimensions
 *
 * @returns the output of the contraction, permuted according to labels_C
 */

template <size_t contractions, execution_space space, size_t dimension_A, size_t dimension_B, typename data_t, bool force_naive = false, typename DimLabel_t = std::string>
  requires((dimension_A > 1) or (dimension_B > 1))
Tensor<(dimension_A + dimension_B) - 2*contractions, space, data_t> tensor_contraction(
  const Array<DimLabel_t, dimension_A>& labels_A,
  const Tensor<dimension_A, space, data_t>& tensor_A,
  const Array<DimLabel_t, dimension_B>& labels_B,
  const Tensor<dimension_B, space, data_t>& tensor_B,
  const Array<DimLabel_t, (dimension_A + dimension_B) - 2*contractions>& labels_C)
{
  BOBA_CALI_MARK
  checkpoint();
  auto common_labels = get_common_labels<contractions>(labels_A, labels_B);
  auto contraction_indices_A = get_label_indices(labels_A, common_labels);
  auto contraction_indices_B = get_label_indices(labels_B, common_labels);
  auto uncontracted_labels = get_uncontracted_labels(labels_A, contraction_indices_A, labels_B, contraction_indices_B);
  auto output = tensor_contraction<contractions, space, dimension_A, dimension_B, data_t, force_naive>(
    tensor_A,
    tensor_B,
    contraction_indices_A,
    contraction_indices_B);
  checkpoint();
  permute(uncontracted_labels, output, labels_C);
  return output;
}

/**
 * \brief
 * Given a set of tensor sizes 'precontracted_sizes', and 'contraction_dimensions', this returns the
 * sizes of the uncontracted dimensions, preserving the precontracted order
 */

template <size_t contracted_dimensions, size_t dimension>
::boba::Multiindexer<dimension - contracted_dimensions> make_contracted_dimensions(
  ::boba::Array<index_t, dimension> precontracted_sizes,
  ::boba::Array<index_t, contracted_dimensions> contraction_dimensions)
{
  auto contraction_dimensions_vec = ::boba::make_std_vector(contraction_dimensions);
  std::sort(contraction_dimensions_vec.begin(), contraction_dimensions_vec.end());

  constexpr size_t final_dimension = dimension - contracted_dimensions;

  Array<index_t, final_dimension> new_sizes = ::boba::filled_array<final_dimension>(0_z);
  index_t fin_id = 0;
  index_t check_id = 0;
  if constexpr (final_dimension > 0)
  {
    for (index_t pre_id = 0; pre_id < dimension; pre_id++)
    {
      bool is_skipped = false;
      if (check_id < contracted_dimensions)
      {
        is_skipped = contraction_dimensions_vec.at(check_id) == pre_id;
      }

      if (is_skipped)
      {
        check_id++;
      }
      else
      {
        new_sizes[fin_id] = precontracted_sizes[pre_id];
        fin_id++;
      }
    }
  }

  ::boba::Multiindexer<final_dimension> output_mider(new_sizes);
  return output_mider;
}

/**
 * \brief
 * Helper function for contraction schemes
 */
template <size_t contracted_dimensions, size_t contractions>
__boba_host_device__ ::boba::Array<index_t, contracted_dimensions + contractions> make_uncontracted_mid(
  ::boba::Array<index_t, contracted_dimensions> contracted_mid,
  ::boba::Array<index_t, contractions> contraction_mid,
  ::boba::Array<index_t, contractions> contraction_dimensions)
{
  constexpr size_t uncontracted_dimensions = contracted_dimensions + contractions;
  ::boba::Array<index_t, uncontracted_dimensions> uncontracted_mid;

  for (size_t i = 0; i < contractions; i++)
  {
    uncontracted_mid[contraction_dimensions[i]] = contraction_mid[i];
  }
  size_t j = 0;
  if constexpr (contracted_dimensions > 0)
  {
    for (size_t i = 0; i < uncontracted_dimensions; i++)
    {
      bool is_contraction_dimension = false;
      for (size_t k = 0; k < contractions; k++)
      {
        if (contraction_dimensions[k] == i)
        {
          is_contraction_dimension = true;
        }
      }
      if (not(is_contraction_dimension))
      {
        uncontracted_mid[i] = contracted_mid[j];
        j++;
      }
    }
  }
  return uncontracted_mid;
}

/**
 * \brief
 * Contracts A and B along given dimensions
 * Given tensors A({i, j, ..., k, ...}) and  B({l, p, ..., k, ...}),
 * computes C({i, j, ..., l, p, ...}) = \sum_k A({..., k, ...}) * B({..., k, ...})
 */

template <size_t contractions, execution_space space, size_t dimension_A, size_t dimension_B, typename data_t, bool force_naive = false>
  requires((contractions == 1) and (dimension_A == 1) and (dimension_B == 1))
Tensor<1, space, data_t> tensor_contraction(
  const Tensor<dimension_A, space, data_t>& tensor_A,
  const Tensor<dimension_B, space, data_t>& tensor_B,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  ::boba::detail::ignore(contraction_dimensions_A);
  ::boba::detail::ignore(contraction_dimensions_B);
  Tensor<1, space, data_t> out({1});
  out.fill_with(inner_product(tensor_A, tensor_B));
  return out;
}

template <size_t contractions, execution_space space, size_t dimension_A, size_t dimension_B, typename data_t, bool force_naive = false>
  requires((dimension_A > 1) or (dimension_B > 1))
Tensor<(dimension_A + dimension_B) - 2*contractions, space, data_t> tensor_contraction(
  const Tensor<dimension_A, space, data_t>& tensor_A,
  const Tensor<dimension_B, space, data_t>& tensor_B,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  return tensor_contraction<contractions, space, dimension_A, dimension_B, data_t, force_naive>(tensor_A.const_view(), tensor_B.const_view(), contraction_dimensions_A, contraction_dimensions_B);
}

template <size_t contractions, execution_space space, size_t dimension_A, size_t dimension_B, typename data_t, bool force_naive = false>
Tensor<(dimension_A + dimension_B) - 2*contractions, space, data_t> tensor_contraction(
  const TensorView<DefaultAccessor<data_t const>, dimension_A>& tensor_A_view,
  const TensorView<DefaultAccessor<data_t const>, dimension_B>& tensor_B_view,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  BOBA_CALI_MARK
  checkpoint();
  constexpr size_t new_dimension = (dimension_A + dimension_B) - 2*contractions;
  auto sizes_A = tensor_A_view.sizes();
  auto sizes_B = tensor_B_view.sizes();
  for(size_t c = 0; c < contractions; c++)
  {
    boba_always_assert_equal(sizes_A[contraction_dimensions_A[c]], sizes_B[contraction_dimensions_B[c]], "Invalid contraction.");
  }

  //
  // Sizes of A/B less the contracted dimension
  //
  auto contracted_A_mider = make_contracted_dimensions<contractions>(sizes_A, contraction_dimensions_A);
  auto contracted_B_mider = make_contracted_dimensions<contractions>(sizes_B, contraction_dimensions_B);

  //
  // Sizes of new tensor
  //
  Array<index_t, new_dimension> new_sizes = concatenate(contracted_A_mider.sizes(), contracted_B_mider.sizes());
  Tensor<new_dimension, space, data_t> tensor_C(new_sizes);

  if (tensor_C.size() == 0)
  {
    return tensor_C;
  }

  //
  // Perform Contraction
  //
  if constexpr (
    (space == ::boba::execution_space::HIP) and
    ::boba::boba_hip_enabled() and
    ::boba::boba_hiptensor_enabled() and
    not(force_naive))
  {
    tensor_C.fill_with_zeros();
    ::boba::detail::hiptensor_contract(tensor_A_view, tensor_B_view, tensor_C.view(), contraction_dimensions_A, contraction_dimensions_B);
    return tensor_C;
  }
  else if constexpr (
    (space == ::boba::execution_space::CUDA) and
    ::boba::boba_cuda_enabled() and
    ::boba::boba_cutensor_enabled() and
    not(force_naive))
  {
    tensor_C.fill_with_zeros();
    ::boba::detail::cutensor_contract(tensor_A_view, tensor_B_view, tensor_C.view(), contraction_dimensions_A, contraction_dimensions_B);
    return tensor_C;
  }
  else if constexpr (
    (space == ::boba::execution_space::CPU) and
    ::boba::is_cpu_enabled() and
    ::boba::boba_metal_enabled() and
    not(force_naive))
  {
    tensor_C.fill_with_zeros();
    ::boba::detail::metal_contract(tensor_A_view, tensor_B_view, tensor_C.view(), contraction_dimensions_A, contraction_dimensions_B);
    return tensor_C;
  }
  else if constexpr (
    (space == ::boba::execution_space::CPU) and
    ::boba::is_cpu_enabled() and
    ::boba::boba_eigen_tensor_enabled() and
    not(force_naive))
  {
    ::boba::detail::eigen_tensor_contract(tensor_A_view, tensor_B_view, tensor_C.view(), contraction_dimensions_A, contraction_dimensions_B);
    return tensor_C;
  }

  //
  // Fallback scheme
  //
  BOBA_CALI_BEGIN("reference_implementation");

  tensor_C.fill_with_zeros();
  auto tensor_C_view = tensor_C.atomic_view();
  auto cA_size = contracted_A_mider.empty() ? 1_z : contracted_A_mider.size();
  auto cB_size = contracted_B_mider.empty() ? 1_z : contracted_B_mider.size();
  auto multiindex_contracted_AB = ::boba::Multiindexer<2>({cA_size, cB_size});

  ::boba::Array<size_t, contractions> contractions_lengths;
  for(size_t c = 0; c < contractions; c++)
  {
    contractions_lengths[c] = sizes_A[contraction_dimensions_A[c]];
  }

  auto multiindex_contraction = ::boba::Multiindexer<contractions>(contractions_lengths);

  ::boba::loop<space, 2>(
    {multiindex_contracted_AB.size(), multiindex_contraction.size()},
    [=] __boba_host_device__(::boba::Array<size_t, 2> ijk)
  {
    auto [index_C, contract_mid] = ijk;
    auto contraction_ids = multiindex_contraction.multiindex(contract_mid);

    //
    // Indexing logic
    //
    auto cmid_AB = multiindex_contracted_AB.multiindex(index_C);
    auto [cindex_A, cindex_B] = cmid_AB;

    auto cmid_A = contracted_A_mider.multiindex(cindex_A);
    auto cmid_B = contracted_B_mider.multiindex(cindex_B);

    auto mid_A = make_uncontracted_mid(cmid_A, contraction_ids, contraction_dimensions_A);
    auto mid_B = make_uncontracted_mid(cmid_B, contraction_ids, contraction_dimensions_B);

    auto value_A = tensor_A_view(mid_A);
    auto value_B = tensor_B_view(mid_B);

    //
    // Contract
    //
    tensor_C_view(index_C) += value_A * value_B;
  });
  BOBA_CALI_END("reference_implementation");


  checkpoint();
  return tensor_C;
}

/**
 * @brief Finds the labels from labels that are not in labels_subset
 *
 * @tparam num_search_labels to search for
 * @param [in] labels Array of einsum labels
 * @param [in] search_labels Subset of labels
 *
 * @returns Array of labels that are in labels but missing from search_labels
 */

template <size_t num_search_labels, size_t dimension_A, typename DimLabel_t>
Array<DimLabel_t, dimension_A - num_search_labels> get_missing_labels(
  const Array<DimLabel_t, dimension_A>& labels,
  const Array<DimLabel_t, num_search_labels>& labels_subset)
{
  constexpr auto num_output_labels = dimension_A - num_search_labels;
  Array<DimLabel_t, num_output_labels> missing_labels;

  size_t current_label_id = 0;
  for (auto label : labels)
  {

    bool is_in_lables_subset = false;
    for (auto sublabel : labels_subset)
    {
      if (sublabel == label)
      {
        is_in_lables_subset = true;
      }
    }

    if (not(is_in_lables_subset))
    {
      missing_labels[current_label_id] = label;
      current_label_id++;
    }

    if (current_label_id == num_output_labels)
    {
      break;
    }
  }

  return missing_labels;
}

/**
 * @brief Reduces a tensor using einsum notation.
 *
 * @param [in] labels Array of einsum labels for tensor A
 * @param [in] tensor_A the input tensor
 * @param [in] final_labels Array of einsum labels expressing the desired order of the unreduced dimensions
 *
 * @returns the output of the reduction, permuted according to final_labels
 */

template <size_t reductions, execution_space space, size_t dimension_A, typename data_t, bool force_naive = false, typename DimLabel_t = std::string>
Tensor<dimension_A - reductions, space, data_t> tensor_reduction(
  const Array<DimLabel_t, dimension_A>& labels,
  const Tensor<dimension_A, space, data_t>& tensor_A,
  const Array<DimLabel_t, dimension_A - reductions>& final_labels)
{
  BOBA_CALI_MARK
  auto missing_labels = get_missing_labels(labels, final_labels);
  auto reduced_labels = get_missing_labels(labels, missing_labels);
  auto missing_indices = get_label_indices(labels, missing_labels);
  auto output = tensor_reduction<reductions, space, dimension_A, data_t, force_naive>(
    tensor_A,
    missing_indices);
  checkpoint();
  permute(reduced_labels, output, final_labels);
  return output;
}

/**
 * \brief
 * Reduces A along one or more of its own indices.
 * Given tensor A({i, j, ..., k, ...}), computes the sum over the selected indices.
 */

template <size_t reductions, execution_space space, size_t dimension_A, typename data_t, bool force_naive = false>
Tensor<dimension_A - reductions, space, data_t> tensor_reduction(
  const Tensor<dimension_A, space, data_t>& tensor_A,
  const boba::Array<size_t, reductions> contraction_dimensions)
{
  return tensor_reduction<reductions, space, dimension_A, data_t, force_naive>(tensor_A.const_view(), contraction_dimensions);
}

template <size_t reductions, execution_space space, size_t dimension_A, typename data_t, bool force_naive = false>
Tensor<dimension_A - reductions, space, data_t> tensor_reduction(
  const TensorView<DefaultAccessor<data_t const>, dimension_A>& tensor_A_view,
  const boba::Array<size_t, reductions> contraction_dimensions)
{
  BOBA_CALI_MARK
  checkpoint();
  constexpr size_t new_dimension = dimension_A - reductions;
  static_assert(new_dimension > 0_z, "Invalid choice of tensors and/or reduction");
  auto sizes_A = tensor_A_view.sizes();

  //
  // Sizes of A less the contracted dimensions
  //
  auto contracted_A_mider = make_contracted_dimensions<reductions>(sizes_A, contraction_dimensions);

  Tensor<new_dimension, space, data_t> tensor_C(contracted_A_mider.sizes());
  tensor_C.fill_with_zeros();

  if (tensor_C.size() == 0)
  {
    return tensor_C;
  }

  //
  // Perform Reduction
  //
  if constexpr (
    (space == ::boba::execution_space::CUDA) and
    ::boba::boba_cuda_enabled() and
    ::boba::boba_cutensor_enabled() and
    not(force_naive))
  {
    ::boba::detail::cutensor_reduce(tensor_A_view, tensor_C.view(), contraction_dimensions);
    return tensor_C;
  }
  else if constexpr (
    (space == ::boba::execution_space::HIP) and
    ::boba::boba_hip_enabled() and
    ::boba::boba_hiptensor_enabled() and
    not(force_naive))
  {
    ::boba::detail::hiptensor_reduce(tensor_A_view, tensor_C.view(), contraction_dimensions);
    return tensor_C;
  }
  else if constexpr (
    (space == ::boba::execution_space::CPU) and
    ::boba::is_cpu_enabled() and
    ::boba::boba_metal_enabled() and
    not(force_naive))
  {
    ::boba::detail::metal_reduce(tensor_A_view, tensor_C.view(), contraction_dimensions);
    return tensor_C;
  }
  else if constexpr (
    (space == ::boba::execution_space::CPU) and
    ::boba::is_cpu_enabled() and
    ::boba::boba_eigen_tensor_enabled() and
    not(force_naive))
  {
    ::boba::detail::eigen_tensor_reduce(tensor_A_view, tensor_C.view(), contraction_dimensions);
    return tensor_C;
  }

  //
  // Fallback scheme
  //
  auto tensor_C_view = tensor_C.atomic_view();

  ::boba::Array<size_t, reductions> contraction_lengths;
  for (size_t r = 0; r < reductions; r++)
  {
    contraction_lengths[r] = sizes_A[contraction_dimensions[r]];
  }
  auto multiindex_contraction = ::boba::Multiindexer<reductions>(contraction_lengths);

  checkpoint();
  ::boba::loop<space, 2>({contracted_A_mider.size(), multiindex_contraction.size()},
                         [=] __boba_host_device__(Array<size_t, 2> indices)
  {
    size_t index_C = indices[0];
    size_t contraction_index = indices[1];

    //
    // Indexing logic
    //
    auto cmid_A = contracted_A_mider.multiindex(index_C);
    auto contraction_ids = multiindex_contraction.multiindex(contraction_index);
    auto mid_A = make_uncontracted_mid(cmid_A, contraction_ids, contraction_dimensions);

    //
    // Operation
    //
    auto value_A = tensor_A_view(mid_A);
    tensor_C_view(index_C) += value_A;
  });

  checkpoint();
  return tensor_C;
}

/**
 * @brief Returns the tensor (aka kronecker) product of two tensors
 *
 * @param[in] left tensor
 * @param[in] right tensor
 * @return tensor product of left x right
 */

template <size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, data_t> tensor_product(
  Tensor<dimension, space, data_t> const& left,
  Tensor<dimension, space, data_t> const& right)
{
  auto left_sizes = left.sizes();
  auto right_sizes = right.sizes();
  auto new_sizes = left_sizes;
  for (size_t d = 0; d < dimension; d++)
  {
    new_sizes[d] *= right_sizes[d];
  }
  Tensor<dimension, space, data_t> new_tensor(new_sizes);

  auto left_view = left.const_view();
  auto right_view = right.const_view();
  auto new_view = new_tensor.view();

  loop<space, 2>({left.size(), right.size()},
                 [=] __boba_host_device__(Array<size_t, 2> mid)
  {
    auto left_value = left_view(mid[0]);
    auto right_value = right_view(mid[1]);
    auto new_value = left_value * right_value;

    auto left_mid = left_view.multiindex(mid[0]);
    auto right_mid = right_view.multiindex(mid[1]);
    auto new_mid = left_mid;
    for (size_t d = 0; d < dimension; d++)
    {
      new_mid[d] = left_mid[d] + left_view.sizes(d) * right_mid[d];
    }
    new_view(new_mid) = new_value;
  });

  return new_tensor;
}

/**
 * @brief Returns the tensor (aka kronecker) product of two tensors
 *
 * @param[in] left matrix
 * @param[in] right matrix
 * @return tensor product of left x right
 */

template <execution_space space, typename data_t>
[[nodiscard]]
Matrix<space, data_t> tensor_product(
  const Matrix<space, data_t>& matrix_A,
  const Matrix<space, data_t>& matrix_B)
{
  BOBA_CALI_OBJECT_MARK
  using base = typename Matrix<space, data_t>::base;
  return Matrix<space, data_t>(tensor_product(static_cast<const base&>(matrix_A), static_cast<const base&>(matrix_B)));
}

/**
 * @brief Returns the tensor (aka kronecker) product of two tensors
 *
 * @param[in] left vector
 * @param[in] right vector
 * @return tensor product of left x right
 */

template <execution_space space, typename data_t>
[[nodiscard]]
Vector<space, data_t> tensor_product(
  const Vector<space, data_t>& vector_A,
  const Vector<space, data_t>& vector_B)
{
  BOBA_CALI_OBJECT_MARK
  using base = typename Vector<space, data_t>::base;
  return Vector<space, data_t>(tensor_product(static_cast<const base&>(vector_A), static_cast<const base&>(vector_B)));
}

/**
 * @brief Returns the tensor (aka kronecker) product along mode n of two tensors
 *
 * @param[in] left tensor
 * @param[in] right tensor
 * @param[in] mode mode along which the tensor product will be taken
 * @return tensor product of left x right along mode n, all other modes left the same
 */

template <size_t dimension, execution_space space, typename data_t>
[[nodiscard]]
Tensor<dimension, space, data_t> mode_n_tensor_product(
  Tensor<dimension, space, data_t> const& left,
  Tensor<dimension, space, data_t> const& right,
  size_t mode)
{
  boba_assert_positive(left.sizes(), "Invalid tensor");
  boba_assert_positive(right.sizes(), "Invalid tensor");

  auto left_sizes = left.sizes();
  auto right_sizes = right.sizes();
  auto new_sizes = left_sizes;

  for (size_t d = 0; d < dimension; d++)
  {
    if (d == mode)
    {
      new_sizes[d] *= right_sizes[d];
    }
    else
    {
      boba_always_assert_equal(new_sizes[d], right_sizes[d], "Sizes must be equal.");
    }
  }

  Tensor<dimension, space, data_t> new_tensor(new_sizes);

  auto left_view = left.const_view();
  auto right_view = right.const_view();
  auto new_view = new_tensor.view();

  Multiindexer<2> mode_n_mider({left_sizes[mode], right_sizes[mode]});

  loop<space, 1>({new_view.size()},
                 [=] __boba_host_device__(size_t id)
  {
    auto new_mid = new_view.multiindex(id);
    // left_mid/right_mid is new_mid except for id mode
    auto mode_mid = mode_n_mider.multiindex(new_mid[mode]);
    auto left_mid = new_mid;
    auto right_mid = new_mid;
    // new_mid[mode] = left_mid[mode] + left.sizes(mode) * right_mid[mode]
    left_mid[mode] = mode_mid[0];
    right_mid[mode] = mode_mid[1];

    new_view(new_mid) = left_view(left_mid) * right_view(right_mid);
  });

  return new_tensor;
}

/**
 * @brief Returns the tensor (aka kronecker) product of two tensors
 *
 * @param[in] left matrix
 * @param[in] right matrix
 * @return tensor product of left x right
 */

template <execution_space space, typename data_t>
[[nodiscard]]
Matrix<space, data_t> mode_n_tensor_product(
  const Matrix<space, data_t>& matrix_A,
  const Matrix<space, data_t>& matrix_B,
  size_t mode)
{
  BOBA_CALI_OBJECT_MARK
  using base = typename Matrix<space, data_t>::base;
  return Matrix<space, data_t>(mode_n_tensor_product(static_cast<const base&>(matrix_A), static_cast<const base&>(matrix_B), mode));
}

/**
 * @brief Returns the tensor (aka kronecker) product of two vectors (degenerate case of mode_n_tensor_product where there is only one mode)
 *
 * @param[in] left vector
 * @param[in] right vector
 * @return tensor product of left x right
 */

template <execution_space space, typename data_t>
[[nodiscard]]
Vector<space, data_t> mode_n_tensor_product(
  const Vector<space, data_t>& vector_A,
  const Vector<space, data_t>& vector_B,
  size_t mode)
{
  BOBA_CALI_OBJECT_MARK
  ::boba::detail::ignore(mode);
  return tensor_product(vector_A, vector_B);
}

/**
 * \brief
 * Returns a global estimate of the interpolation error of interpolation_tensor, based on multilinear interpolation.
 */

template <size_t dimension, execution_space space, typename data_t>
std::pair<data_t, size_t> tensor_interpolation_error_estimator(
  ::boba::Tensor<dimension, space, data_t>& interpolation_tensor,
  size_t estimation_order = 1)
{
  BOBA_CALI_OBJECT_MARK

  if (estimation_order != 1)
  {
    boba_error("Error estimation  orders other than 1 not yet implemented.");
  }

  auto interp_error_sizes = interpolation_tensor.sizes();
  auto vertices = interp_error_sizes;
  for (size_t d = 0; d < dimension; d++)
  {
    interp_error_sizes[d] -= 1;
    vertices[d] = 2;
  }

  ::boba::Multiindexer<dimension> vertices_mid(vertices);
  ::boba::Multiindexer<dimension> interpolation_mid(interp_error_sizes);

  auto tensor_view = interpolation_tensor.const_view();

  data_t value = ::boba::lowest_value<data_t>();
  index_t index = 0;

  ::boba::max_loc_reduce<space>(
    value, index, index_t(0), interpolation_mid.size(), [=] __boba_host_device__(::boba::reducer_index_t id, ::boba::max_loc_reducer_operator<data_t> & local_maxloc)
  {
    const size_t id_size = static_cast<size_t>(id);
    auto mid = interpolation_mid.multiindex(id_size);
    auto min_value = tensor_view(mid);
    auto max_value = tensor_view(mid);

    for (size_t i = 0; i < vertices_mid.size(); i++)
    {
      auto delta_mid = vertices_mid.multiindex(i);
      auto vertex_mid = vertices_mid.multiindex(i);

      for (size_t d = 0; d < dimension; d++)
      {
        vertex_mid[d] = mid[d] + delta_mid[d];
      }

      auto this_value = tensor_view(vertex_mid);
      min_value = ::boba::min(min_value, this_value);
      max_value = ::boba::max(max_value, this_value);
    }
    auto estimate = ::boba::abs(max_value - min_value);
    local_maxloc.maxloc(estimate, id);
  });

  return std::make_pair<data_t>(data_t(value), index_t(index));
}

// -------------------------------------------------------------------------------------
// Section: NaN checks
// -------------------------------------------------------------------------------------

template <size_t dimension, execution_space space, typename data_t>
void nan_check(Tensor<dimension, space, data_t> const& tensor)
{
  BOBA_CALI_MARK

  if constexpr (space == host_space)
  {
    using real_data_t = typename Tensor<dimension, space, data_t>::real_data_t;
    auto data = tensor.const_data();

    size_t nans_count = 0;

    ::boba::sum_reduce<space>(nans_count, index_t(0), tensor.size(), [=](index_t i, sum_reducer_operator<index_t>& local_nans_count)
    {
      auto x = data[i];
      bool trap = false;
      if constexpr (std::is_same_v<data_t, real_data_t>)
      {
        trap = boba::isfinite(x);
      }
      else
      {
        trap = boba::isfinite(boba::real(x));
        trap = trap && boba::isfinite(boba::imag(x));
      }
      if (not(trap))
      {
        local_nans_count += 1;
      }
    });

    boba_always_assert_equal(nans_count, index_t(0), "nantrap sprung!");
  }
  else
  {
    Tensor<dimension, host_space, data_t> tensor_cpu{tensor};
    nan_check(tensor_cpu);
  }
}

template <execution_space space, typename data_t>
void nan_check(Matrix<space, data_t> const& matrix)
{
  nan_check(static_cast<Tensor<2, space, data_t> const&>(matrix));
}

template <execution_space space, typename data_t>
void nan_check(Vector<space, data_t> const& vector)
{
  nan_check(static_cast<Tensor<1, space, data_t> const&>(vector));
}

// -------------------------------------------------------------------------------------
// Section: Norm differences
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Largest element magnitude of a tensor.
 */

template <size_t dimension, execution_space space, typename data_t>
auto norm_inf(Tensor<dimension, space, data_t> const& tensor)
{
  checkpoint();
  return reductions::max_abs_reduce<space>(tensor.const_data(), tensor.size());
}

template <execution_space space, typename data_t>
auto norm_inf(Matrix<space, data_t> const& matrix)
{
  checkpoint();
  return norm_inf(static_cast<Tensor<2, space, data_t> const&>(matrix));
}

template <execution_space space, typename data_t>
auto norm_inf(Vector<space, data_t> const& vector)
{
  checkpoint();
  return norm_inf(static_cast<Tensor<1, space, data_t> const&>(vector));
}

/**
 * \brief
 * Frobenius norm of a tensor.
 */

template <size_t dimension, execution_space space, typename data_t>
typename Tensor<dimension, space, data_t>::real_data_t
norm_frobenius(Tensor<dimension, space, data_t> const& tensor)
{
  checkpoint();
  using real_data_t = typename Tensor<dimension, space, data_t>::real_data_t;

  auto tensor_data = tensor.const_data();

  real_data_t value = 0.0;

  ::boba::sum_reduce<space>(value, index_t(0), tensor.size(), [=] __boba_host_device__(index_t i, sum_reducer_operator<real_data_t> & local_value)
  {
    auto abs_local_entry = boba::abs(tensor_data[i]);
    local_value += abs_local_entry * abs_local_entry;
  });

  return boba::sqrt(value);
}

template <execution_space space, typename data_t>
auto norm_frobenius(Matrix<space, data_t> const& matrix)
{
  checkpoint();
  return norm_frobenius(static_cast<Tensor<2, space, data_t> const&>(matrix));
}

template <execution_space space, typename data_t>
auto norm_frobenius(Vector<space, data_t> const& vector)
{
  checkpoint();
  return norm_frobenius(static_cast<Tensor<1, space, data_t> const&>(vector));
}

/**
 * \brief
 * Infinity norm difference between two tensors.
 */

template <size_t dimension, execution_space space, typename data_t>
typename Tensor<dimension, space, data_t>::real_data_t
norm_difference_inf(
  Tensor<dimension, space, data_t> const& tensor_A,
  Tensor<dimension, space, data_t> const& tensor_B)
{
  checkpoint();
  using real_data_t = typename Tensor<dimension, space, data_t>::real_data_t;

  if (not(tensor_A.sizes() == tensor_B.sizes()))
  {
    throw std::length_error("Tensor shapes must match for norm_difference_inf.");
  }

  auto tensor_A_data = tensor_A.const_data();
  auto tensor_B_data = tensor_B.const_data();

  real_data_t difference_norm = 0.0;

  ::boba::max_reduce<space>(difference_norm, 0_z, tensor_A.size(), [=] __boba_host_device__(index_t i, max_reducer_operator<real_data_t> & local_difference_norm)
  {
    auto local_value = boba::abs(tensor_B_data[i] - tensor_A_data[i]);
    local_difference_norm.max(local_value);
  });

  return difference_norm;
}

template <size_t dimension, execution_space lhs_space, execution_space rhs_space, typename data_t>
  requires(lhs_space != rhs_space)
auto norm_difference_inf(
  Tensor<dimension, lhs_space, data_t> const& tensor_A,
  Tensor<dimension, rhs_space, data_t> const& tensor_B)
{
  checkpoint();
  Tensor<dimension, lhs_space, data_t> tensor_B_in_lhs_space{tensor_B};
  return norm_difference_inf(tensor_A, tensor_B_in_lhs_space);
}

/**
 * \brief
 * Infinity norm difference between two matrices.
 */

template <execution_space space, typename data_t>
auto norm_difference_inf(
  Matrix<space, data_t> const& tensor_A,
  Matrix<space, data_t> const& tensor_B)
{
  checkpoint();
  return norm_difference_inf(
    static_cast<Tensor<2, space, data_t> const&>(tensor_A),
    static_cast<Tensor<2, space, data_t> const&>(tensor_B));
}

template <execution_space lhs_space, execution_space rhs_space, typename data_t>
  requires(lhs_space != rhs_space)
auto norm_difference_inf(
  Matrix<lhs_space, data_t> const& tensor_A,
  Matrix<rhs_space, data_t> const& tensor_B)
{
  checkpoint();
  return norm_difference_inf(
    static_cast<Tensor<2, lhs_space, data_t> const&>(tensor_A),
    static_cast<Tensor<2, rhs_space, data_t> const&>(tensor_B));
}

/**
 * \brief
 * Infinity norm difference between two vectors.
 */

template <execution_space space, typename data_t>
auto norm_difference_inf(
  Vector<space, data_t> const& tensor_A,
  Vector<space, data_t> const& tensor_B)
{
  checkpoint();
  return norm_difference_inf(
    static_cast<Tensor<1, space, data_t> const&>(tensor_A),
    static_cast<Tensor<1, space, data_t> const&>(tensor_B));
}

template <execution_space lhs_space, execution_space rhs_space, typename data_t>
  requires(lhs_space != rhs_space)
auto norm_difference_inf(
  Vector<lhs_space, data_t> const& tensor_A,
  Vector<rhs_space, data_t> const& tensor_B)
{
  checkpoint();
  return norm_difference_inf(
    static_cast<Tensor<1, lhs_space, data_t> const&>(tensor_A),
    static_cast<Tensor<1, rhs_space, data_t> const&>(tensor_B));
}

/**
 * \brief
 * Frobenius norm difference between two tensors.
 */

template <size_t dimension, execution_space space, typename data_t>
typename Tensor<dimension, space, data_t>::real_data_t
norm_difference_frobenius(
  Tensor<dimension, space, data_t> const& tensor_A,
  Tensor<dimension, space, data_t> const& tensor_B)
{
  checkpoint();
  using real_data_t = typename Tensor<dimension, space, data_t>::real_data_t;

  if (not(tensor_A.sizes() == tensor_B.sizes()))
    throw std::length_error("Tensor shapes must match for norm_difference_frobenius.");

  auto tensor_A_data = tensor_A.const_data();
  auto tensor_B_data = tensor_B.const_data();

  auto value = real_data_t(0.0);

  ::boba::sum_reduce<space>(value, index_t(0), tensor_A.size(), [=] __boba_host_device__(index_t i, sum_reducer_operator<real_data_t> & local_value)
  {
    auto difference = boba::abs(tensor_B_data[i] - tensor_A_data[i]);
    local_value += difference * difference;
  });

  return boba::sqrt(value);
}

template <size_t dimension, execution_space lhs_space, execution_space rhs_space, typename data_t>
  requires(lhs_space != rhs_space)
auto norm_difference_frobenius(
  Tensor<dimension, lhs_space, data_t> const& tensor_A,
  Tensor<dimension, rhs_space, data_t> const& tensor_B)
{
  checkpoint();
  Tensor<dimension, lhs_space, data_t> tensor_B_in_lhs_space{tensor_B};
  return norm_difference_frobenius(tensor_A, tensor_B_in_lhs_space);
}

/**
 * \brief
 * Frobenius norm difference between two matrices.
 */

template <execution_space space, typename data_t>
auto norm_difference_frobenius(
  Matrix<space, data_t> const& tensor_A,
  Matrix<space, data_t> const& tensor_B)
{
  checkpoint();
  return norm_difference_frobenius(
    static_cast<Tensor<2, space, data_t> const&>(tensor_A),
    static_cast<Tensor<2, space, data_t> const&>(tensor_B));
}

template <execution_space lhs_space, execution_space rhs_space, typename data_t>
  requires(lhs_space != rhs_space)
auto norm_difference_frobenius(
  Matrix<lhs_space, data_t> const& tensor_A,
  Matrix<rhs_space, data_t> const& tensor_B)
{
  checkpoint();
  return norm_difference_frobenius(
    static_cast<Tensor<2, lhs_space, data_t> const&>(tensor_A),
    static_cast<Tensor<2, rhs_space, data_t> const&>(tensor_B));
}

/**
 * \brief
 * Frobenius norm difference between two matrices.
 */

template <execution_space space, typename data_t>
auto norm_difference_frobenius(
  Vector<space, data_t> const& tensor_A,
  Vector<space, data_t> const& tensor_B)
{
  checkpoint();
  return norm_difference_frobenius(
    static_cast<Tensor<1, space, data_t> const&>(tensor_A),
    static_cast<Tensor<1, space, data_t> const&>(tensor_B));
}

template <execution_space lhs_space, execution_space rhs_space, typename data_t>
  requires(lhs_space != rhs_space)
auto norm_difference_frobenius(
  Vector<lhs_space, data_t> const& tensor_A,
  Vector<rhs_space, data_t> const& tensor_B)
{
  checkpoint();
  return norm_difference_frobenius(
    static_cast<Tensor<1, lhs_space, data_t> const&>(tensor_A),
    static_cast<Tensor<1, rhs_space, data_t> const&>(tensor_B));
}

/**
 * \brief
 * Maximum pointwise relative difference
 */

template <size_t dimension, execution_space space, typename data_t>
data_t max_pointwise_relative_difference(
  Tensor<dimension, space, data_t> const& tensor_A,
  Tensor<dimension, space, data_t> const& tensor_B)
{
  checkpoint();
  return tensor_B.max_pointwise_relative_difference(tensor_A);
}

/**
 * \brief
 * Maximum pointwise relative difference
 */

template <execution_space space, typename data_t>
data_t max_pointwise_relative_difference(
  Matrix<space, data_t> const& tensor_A,
  Matrix<space, data_t> const& tensor_B)
{
  checkpoint();
  return tensor_B.max_pointwise_relative_difference(tensor_A);
}

/**
 * \brief
 * Maximum pointwise relative difference
 */

template <execution_space space, typename data_t>
data_t max_pointwise_relative_difference(
  Vector<space, data_t> const& tensor_A,
  Vector<space, data_t> const& tensor_B)
{
  checkpoint();
  return tensor_B.max_pointwise_relative_difference(tensor_A);
}

// -------------------------------------------------------------------------------------
// Section: Matrix-matrix
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Apply a vector as if it were a diagonal matrix, C = A*diagonalize(v).
 */

template <execution_space space, typename vectordata_t, typename matrixdata_t>
void apply_as_diagonal_right_in_place(
  const ::boba::Vector<space, vectordata_t>& vector,
  ::boba::Matrix<space, matrixdata_t>& output)
{
  BOBA_CALI_MARK
  checkpoint();

  auto output_view = output.view();
  auto vector_view = vector.const_view();
  boba_always_assert_equal(output.cols(), vector.size(), "incompatible dimensions");
  checkpoint();
  ::boba::loop<space, 2>(output.sizes(),
                         [=] __boba_host_device__(Array<size_t, 2> rc)
  {
    size_t col = rc[1];
    output_view(rc) = output_view(rc) * vector_view({col});
  });
  checkpoint();
}

/**
 * \brief
 * Apply a vector as if it were a diagonal matrix, C = diag(v)*A.
 */

template <execution_space space, typename vectordata_t, typename matrixdata_t>
void apply_as_diagonal_left_in_place(
  const ::boba::Vector<space, vectordata_t>& vector,
  ::boba::Matrix<space, matrixdata_t>& output)
{
  BOBA_CALI_MARK
  checkpoint();

  auto output_view = output.view();
  auto vector_view = vector.const_view();
  boba_always_assert_equal(output.rows(), vector.size(), "incompatible dimensions");
  checkpoint();
  ::boba::loop<space, 2>(output.sizes(),
                         [=] __boba_host_device__(Array<size_t, 2> rc)
  {
    size_t row = rc[0];
    output_view(rc) = vector_view({row}) * output_view(rc);
  });
  checkpoint();
}

// ---------------------------------------------------------------------------
// Flattening tensor into vector
// ---------------------------------------------------------------------------

template <size_t dimension, execution_space space, typename data_t>
Vector<space, data_t>
flatten(const Tensor<dimension, space, data_t>& input_tensor)
{
  Vector<space, data_t> output_vector({input_tensor.size()});

  output_vector.reshape(input_tensor);
  return output_vector;
}

// ---------------------------------------------------------------------------
// Unfolding and folding operations
// ---------------------------------------------------------------------------

/**
 * \brief Unfold a tensor into a matrix whose rows correspond to a single mode.
 * A({i, j, k, l, p}) -> A({k}, {i, j, l, p})
 *
 * \param input Tensor to unfold.
 * \param mode_k Tensor mode to place on the matrix rows.
 * \return Matrix unfolding of \p input.
 *
 * \note The remaining modes are inferred and placed on the matrix columns.
 */
template <size_t dimension, execution_space space, typename data_t>
Matrix<space, data_t> unfold(
  const Tensor<dimension, space, data_t>& input,
  const size_t mode_k)
{
  BOBA_CALI_MARK
  checkpoint();

  return unfold(input, std::vector<size_t>{mode_k});
}

/**
 * \brief Unfold a tensor into a matrix whose rows correspond to a grouping of selected modes.
 *
 * \param input Tensor to unfold.
 * \param row_dims Tensor modes to group on the matrix rows.
 * \return Matrix unfolding of \p input.
 *
 * \note The entries of \p row_dims are sorted internally. All remaining
 *       modes are placed on the matrix columns.
 */
template <size_t dimension, execution_space space, typename data_t>
Matrix<space, data_t> unfold(
  const Tensor<dimension, space, data_t>& input,
  std::vector<size_t> row_dims)
{
  BOBA_CALI_MARK
  checkpoint();

  std::vector<size_t> all_dims(dimension);
  std::iota(all_dims.begin(), all_dims.end(), 0_z);

  std::sort(row_dims.begin(), row_dims.end());

  std::vector<size_t> col_dims;
  std::set_difference(all_dims.begin(), all_dims.end(), row_dims.begin(), row_dims.end(), std::back_inserter(col_dims));

  return unfold(input, row_dims, col_dims);
}

/**
 * \brief Unfold a tensor into a matrix using explicit row and column groups.
 *
 * A({i, j, k, l, p}) -> A({row_dims}, {col_dims})
 *
 * \param input Tensor to unfold.
 * \param row_dims Tensor modes to group on the matrix rows.
 * \param col_dims Tensor modes to group on the matrix columns.
 * \return Matrix unfolding of \p input.
 *
 * \note The order of \p row_dims and \p col_dims is preserved.
 * \note The entries of \p row_dims and \p col_dims must form a valid
 *       permutation of the tensor modes.
 */
template <size_t dimension, execution_space space, typename data_t>
Matrix<space, data_t> unfold(
  const Tensor<dimension, space, data_t>& input,
  std::vector<size_t> row_dims,
  std::vector<size_t> col_dims)
{
  BOBA_CALI_MARK
  checkpoint();

  boba_always_assert(row_dims.size() + col_dims.size() == input.get_dimension(),
                     "Inconsistent tensor and matrix dimensions.");

  auto input_copy = input;
  auto input_sizes = input.sizes();

  Array<size_t, dimension> ordering;
  size_t idx = 0;
  for (size_t i : row_dims)
  {
    ordering[idx++] = i;
  }
  for (size_t i : col_dims)
  {
    ordering[idx++] = i;
  }

  boba_always_assert(is_valid_permutation(ordering),
                     "Permutation violates one or more conditions.");

  auto ordering_as_index = typed_array<index_t>(ordering);
  permute(input_copy, ordering_as_index);

  size_t row_prod = 1;
  for (size_t dim_idx : row_dims)
  {
    row_prod *= input_sizes[dim_idx];
  }

  size_t col_prod = 1;
  for (size_t dim_idx : col_dims)
  {
    col_prod *= input_sizes[dim_idx];
  }

  Matrix<space, data_t> output({row_prod, col_prod});
  output.reshape(input_copy);
  return output;
}

/**
 * \brief Fold a matrix into a tensor, where the matrix rows represent one
 * tensor mode.
 * A({k}, {i, j, l, p}) -> A({i, j, k, l, p})
 *
 * \param input Matrix to fold.
 * \param sizes Sizes of the output tensor modes.
 * \param mode_k Original tensor mode represented by the matrix rows.
 * \return Tensor obtained by folding \p input.
 *
 * \note This is the inverse operation for unfold(input, mode_k).
 * \note The matrix columns are interpreted as the remaining tensor modes.
 */
template <size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, data_t> fold(
  const Matrix<space, data_t>& input,
  const Array<size_t, dimension>& sizes,
  const size_t mode_k)
{
  BOBA_CALI_MARK
  checkpoint();

  return fold(input, sizes, std::vector<size_t>{mode_k});
}

/**
 * \brief Fold a matrix into a tensor, where the matrix rows represent a group
 * of tensor modes.
 *
 * \param input Matrix to fold.
 * \param sizes Sizes of the output tensor modes.
 * \param row_dims Original tensor modes that were grouped into the matrix rows
 *        during unfolding.
 * \return Tensor obtained by folding \p input.
 *
 * \note This is the inverse operation for unfold(input, row_dims).
 * \note The entries of \p row_dims are sorted internally. All remaining
 *       modes are interpreted as the matrix columns.
 */
template <size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, data_t> fold(
  const Matrix<space, data_t>& input,
  const Array<size_t, dimension>& sizes,
  std::vector<size_t> row_dims)
{
  BOBA_CALI_MARK
  checkpoint();

  std::vector<size_t> all_dims(dimension);
  std::iota(all_dims.begin(), all_dims.end(), 0_z);

  std::sort(row_dims.begin(), row_dims.end());

  std::vector<size_t> col_dims;
  std::set_difference(all_dims.begin(), all_dims.end(), row_dims.begin(), row_dims.end(), std::back_inserter(col_dims));

  return fold(input, sizes, row_dims, col_dims);
}

/**
 * \brief Fold a matrix into a tensor using explicit tensor modes for the
 * matrix rows and columns.
 *
 * A({row_dims}, {col_dims}) -> A({i, j, k, l, p})
 *
 * \param input Matrix to fold.
 * \param sizes Sizes of the output tensor modes.
 * \param row_dims Original tensor modes that were grouped into the matrix rows
 *        during unfolding.
 * \param col_dims Original tensor modes that were grouped into the matrix columns
 *        during unfolding.
 * \return Tensor obtained by folding \p input.
 *
 * \note This is the inverse operation for unfold(input, row_dims, col_dims).
 * \note The order of \p row_dims and \p col_dims is preserved.
 * \note The entries of \p row_dims and \p col_dims must form a valid
 *       permutation of the tensor modes.
 */
template <size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, data_t> fold(
  const Matrix<space, data_t>& input,
  const Array<size_t, dimension>& sizes,
  std::vector<size_t> row_dims,
  std::vector<size_t> col_dims)
{
  BOBA_CALI_MARK
  checkpoint();

  size_t prod_sizes = 1;
  for (size_t d = 0; d < dimension; d++)
  {
    prod_sizes *= sizes[d];
  }
  boba_always_assert_equal(prod_sizes, input.size(), "Inconsistent tensor and matrix sizes.");

  Array<size_t, dimension> ordering;
  size_t idx = 0;
  for (size_t i : row_dims)
  {
    ordering[idx++] = i;
  }
  for (size_t i : col_dims)
  {
    ordering[idx++] = i;
  }
  boba_always_assert(is_valid_permutation(ordering),
                     "Permutation violates one or more conditions.");

  Array<size_t, dimension> permuted_sizes;
  for (size_t d = 0; d < dimension; d++)
  {
    permuted_sizes[d] = sizes[ordering[d]];
  }

  Tensor<dimension, space, data_t> output(typed_array<index_t>(permuted_sizes));
  output.reshape(input);

  Array<size_t, dimension> inverse_ordering;
  for (size_t i = 0; i < dimension; ++i)
  {
    inverse_ordering[ordering[i]] = i;
  }
  permute(output, typed_array<index_t>(inverse_ordering));
  return output;
}

// ---------------------------------------------------------------------------
// reshape
// ---------------------------------------------------------------------------

template <size_t dimension_out, size_t dimension_in, execution_space space, typename data_t>
::boba::Tensor<dimension_out, space, data_t> reshape(
  const ::boba::Tensor<dimension_in, space, data_t>& input,
  const ::boba::Array<size_t, dimension_out>& new_sizes)
{
  ::boba::Tensor<dimension_out, space, data_t> out(new_sizes);
  out.reshape(input);
  return out;
}

template <size_t dimension_in, execution_space space, typename data_t>
::boba::Matrix<space, data_t> reshape_to_matrix(
  const ::boba::Tensor<dimension_in, space, data_t>& input,
  const ::boba::Array<size_t, 2>& new_sizes)
{
  ::boba::Matrix<space, data_t> out(new_sizes);
  out.reshape(input);
  return out;
}

template <size_t dimension_out, execution_space space, typename data_t>
::boba::Tensor<dimension_out, space, data_t> reshape_from_matrix(
  const ::boba::Matrix<space, data_t>& input,
  const ::boba::Array<size_t, dimension_out>& new_sizes)
{
  ::boba::Tensor<dimension_out, space, data_t> out(new_sizes);
  out.reshape(input);
  return out;
}

template <execution_space space, typename data_t>
::boba::Matrix<space, data_t> reshape(
  const ::boba::Matrix<space, data_t>& input,
  const ::boba::Array<size_t, 2>& new_sizes)
{
  ::boba::Matrix<space, data_t> out(new_sizes);
  out.reshape(input);
  return out;
}

} // namespace boba

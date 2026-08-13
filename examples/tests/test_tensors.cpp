// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "common.hpp"

#include <BOBA/boba.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::microseconds;

/*
  Tests the tensor contractions abstraction layer
*/

//
// test_tensor_permutation
//
template <typename data_t, size_t dimension_A, bool force_naive>
void test_tensor_permutation(
  ::boba::Array<size_t, dimension_A> sizes_A,
  ::boba::Array<size_t, dimension_A> permutation,
  bool& check)
{
  auto sizes_permuted = ::boba::permute(sizes_A, permutation);
  ::boba::Multiindexer<dimension_A> tensor_permuted(sizes_permuted);

  ::boba::Tensor<dimension_A, space, data_t> tensor_A(sizes_A);
  auto tensor_A_view = tensor_A.view();

  // A = sum of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_A_view.size(), [=] __boba_host_device__(size_t index_A)
  {
    auto mid_A = tensor_A_view.multiindex(index_A);
    tensor_A_view(mid_A) = ::boba::sum(mid_A);
  });

  ::boba::TicToc<tictoc_units> timer;
  ::boba::permute<space, dimension_A, data_t, force_naive>(tensor_A, permutation);
  checkpoint();
  timer.end();

  // Validate A
  data_t error_inf_norm = ::boba::lowest_value<data_t>();
  size_t loc = 0;

  ::boba::max_loc_reduce<boba::default_execution_space>(
    error_inf_norm, loc, 0_z, tensor_A.size(), [=] __boba_host_device__(::boba::reducer_index_t index_A, boba::max_loc_reducer_operator<data_t> & local_error)
  {
    const size_t index_A_size = static_cast<size_t>(index_A);
    auto mid_A = tensor_permuted.multiindex(index_A_size);

    data_t sum = ::boba::sum(mid_A);
    data_t a = tensor_A_view(index_A_size);
    data_t error_loc = ::boba::abs(a - sum);
    local_error.maxloc(error_loc, index_A);
  });

  std::cout << "Permutation; "
            << ::boba::detail::execution_space_name(space) << "; "
            << sizes_A << "->"
            << tensor_A.sizes() << "; "
            << timer.duration << "; "
            << timer.units_string << std::endl;

  pass_or_fail(check, error_inf_norm, 1.0e-13);
}

//
// test_tensor_permutation_einsum
//
template <typename data_t, size_t dimension_A, bool force_naive>
void test_tensor_permutation_einsum(
  ::boba::Array<std::string, dimension_A> indices,
  ::boba::Array<size_t, dimension_A> sizes_A,
  ::boba::Array<std::string, dimension_A> permutation,
  bool& check)
{
  auto sizes_permuted = ::boba::permute(indices, sizes_A, permutation);
  ::boba::Multiindexer<dimension_A> tensor_permuted(sizes_permuted);

  ::boba::Tensor<dimension_A, space, data_t> tensor_A(sizes_A);
  auto tensor_A_view = tensor_A.view();

  // A = sum of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_A_view.size(), [=] __boba_host_device__(size_t index_A)
  {
    auto mid_A = tensor_A_view.multiindex(index_A);
    tensor_A_view(mid_A) = ::boba::sum(mid_A);
  });

  ::boba::TicToc<tictoc_units> timer;
  ::boba::permute<space, dimension_A, data_t, force_naive>(indices, tensor_A, permutation);
  checkpoint();
  timer.end();

  // Validate A
  data_t error_inf_norm = ::boba::lowest_value<data_t>();
  size_t loc = 0;

  ::boba::max_loc_reduce<boba::default_execution_space>(
    error_inf_norm, loc, 0_z, tensor_A.size(), [=] __boba_host_device__(::boba::reducer_index_t index_A, boba::max_loc_reducer_operator<data_t> & local_error)
  {
    const size_t index_A_size = static_cast<size_t>(index_A);
    auto mid_A = tensor_permuted.multiindex(index_A_size);

    data_t sum = ::boba::sum(mid_A);
    data_t a = tensor_A_view(index_A_size);
    data_t error_loc = ::boba::abs(a - sum);
    local_error.maxloc(error_loc, index_A);
  });

  std::cout << "Permutation; "
            << ::boba::detail::execution_space_name(space) << "; "
            << sizes_A << "->"
            << tensor_A.sizes() << "; "
            << timer.duration << "; "
            << timer.units_string << std::endl;

  pass_or_fail(check, error_inf_norm, 1.0e-13);
}

//
// test_tensor_reduction_einsum
//
template <typename data_t, size_t dimension_A, bool force_naive>
void test_tensor_reduction_einsum(
  ::boba::Array<std::string, dimension_A> indices,
  ::boba::Array<size_t, dimension_A> sizes_A,
  ::boba::Array<std::string, dimension_A - 2> permutation,
  bool& check)
{
  ::boba::Tensor<dimension_A, space, data_t> tensor_A(sizes_A);
  auto tensor_A_view = tensor_A.view();

  // A = sum of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_A_view.size(), [=] __boba_host_device__(size_t index_A)
  {
    auto mid_A = tensor_A_view.multiindex(index_A);
    tensor_A_view(mid_A) = ::boba::sum(mid_A);
  });

  ::boba::TicToc<tictoc_units> timer;
  auto tensor_C = ::boba::tensor_reduction_double_index<space, dimension_A, data_t, force_naive>(indices, tensor_A, permutation);
  checkpoint();
  timer.end();

  auto tensor_C_view = tensor_C.view();
  auto contraction_labels = ::boba::get_missing_labels(indices, permutation);
  auto contraction_indices = ::boba::get_label_indices(indices, contraction_labels);
  size_t contraction_length_A = sizes_A[contraction_indices[0]];
  size_t contraction_length_B = sizes_A[contraction_indices[1]];

  // Validate A
  data_t error_inf_norm = ::boba::lowest_value<data_t>();
  size_t loc = 0;

  ::boba::max_loc_reduce<boba::default_execution_space>(
    error_inf_norm, loc, 0_z, tensor_C.size(), [=] __boba_host_device__(::boba::reducer_index_t index_C, ::boba::max_loc_reducer_operator<data_t> & local_error)
  {
    const size_t index_C_size = static_cast<size_t>(index_C);
    auto mid_C = tensor_C_view.multiindex(index_C_size);

    data_t sum_1 = ::boba::sum(mid_C) * contraction_length_A * contraction_length_B;
    data_t sum_2 = contraction_length_B * ::boba::sum_of_i(contraction_length_A - 1);
    data_t sum_3 = contraction_length_A * ::boba::sum_of_i(contraction_length_B - 1);
    data_t sum = sum_1 + sum_2 + sum_3;

    data_t c = tensor_C_view(index_C_size);
    data_t error_loc = ::boba::abs(c - sum);

    local_error.maxloc(error_loc, index_C);
  });

  std::cout << "Reduction; "
            << ::boba::detail::execution_space_name(space) << "; "
            << sizes_A << "->"
            << tensor_A.sizes() << "; "
            << timer.duration << "; "
            << timer.units_string << std::endl;

  pass_or_fail(check, error_inf_norm, 1.0e-13);
}

//
// test_tensor_contraction_einsum
//
template <typename data_t, size_t dimension_A, size_t dimension_B, bool force_naive>
void test_tensor_contraction_einsum(
  ::boba::Array<std::string, dimension_A> indices_A,
  ::boba::Array<size_t, dimension_A> sizes_A,
  ::boba::Array<std::string, dimension_B> indices_B,
  ::boba::Array<size_t, dimension_B> sizes_B,
  ::boba::Array<std::string, dimension_A + dimension_B - 4> permutation,
  bool& check)
{
  ::boba::Tensor<dimension_A, space, data_t> tensor_A(sizes_A);
  auto tensor_A_view = tensor_A.view();

  // A = sum of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_A_view.size(), [=] __boba_host_device__(size_t index_A)
  {
    auto mid_A = tensor_A_view.multiindex(index_A);
    tensor_A_view(mid_A) = ::boba::sum(mid_A);
  });

  ::boba::Tensor<dimension_B, space, data_t> tensor_B(sizes_B);
  auto tensor_B_view = tensor_B.view();

  // A = sum of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_B_view.size(), [=] __boba_host_device__(size_t index_B)
  {
    auto mid_B = tensor_B_view.multiindex(index_B);
    tensor_B_view(mid_B) = ::boba::sum(mid_B);
  });

  ::boba::TicToc<tictoc_units> timer;
  auto tensor_C = ::boba::tensor_contraction<2, space, dimension_A, dimension_B, data_t, force_naive>(indices_A, tensor_A, indices_B, tensor_B, permutation);
  checkpoint();
  timer.end();

  auto tensor_C_view = tensor_C.view();
  auto common_labels = ::boba::get_common_labels<2>(indices_A, indices_B);
  auto contraction_indices_A = ::boba::get_label_indices(indices_A, common_labels);
  auto contraction_indices_B = ::boba::get_label_indices(indices_B, common_labels);
  auto uncontracted_labels_A = ::boba::get_missing_labels(indices_A, common_labels);
  auto uncontracted_labels_B = ::boba::get_missing_labels(indices_B, common_labels);
  auto output_indices_A = ::boba::get_label_indices(permutation, uncontracted_labels_A);
  auto output_indices_B = ::boba::get_label_indices(permutation, uncontracted_labels_B);
  size_t contraction_length_A = sizes_A[contraction_indices_A[0]];
  size_t contraction_length_B = sizes_A[contraction_indices_A[1]];
  data_t contraction_sum = contraction_length_B * ::boba::sum_of_i(contraction_length_A - 1) + contraction_length_A * ::boba::sum_of_i(contraction_length_B - 1);
  data_t contraction_sum_of_squares = contraction_length_B * ::boba::sum_of_i2(contraction_length_A - 1) + 2 * ::boba::sum_of_i(contraction_length_A - 1) * ::boba::sum_of_i(contraction_length_B - 1) + contraction_length_A * ::boba::sum_of_i2(contraction_length_B - 1);

  // Validate A
  data_t error_inf_norm = ::boba::lowest_value<data_t>();
  size_t loc = 0;

  ::boba::max_loc_reduce<boba::default_execution_space>(
    error_inf_norm, loc, 0_z, tensor_C.size(), [=] __boba_host_device__(::boba::reducer_index_t index_C, ::boba::max_loc_reducer_operator<data_t> & local_error)
  {
    const size_t index_C_size = static_cast<size_t>(index_C);
    auto mid_C = tensor_C_view.multiindex(index_C_size);

    data_t sum_A = 0;
    for (size_t i = 0; i < dimension_A - 2; i++)
    {
      sum_A += mid_C[output_indices_A[i]];
    }

    data_t sum_B = 0;
    for (size_t i = 0; i < dimension_B - 2; i++)
    {
      sum_B += mid_C[output_indices_B[i]];
    }

    data_t sum = contraction_length_A * contraction_length_B * sum_A * sum_B + (sum_A + sum_B) * contraction_sum + contraction_sum_of_squares;

    data_t c = tensor_C_view(index_C_size);
    data_t error_loc = ::boba::abs(c - sum);

    local_error.maxloc(error_loc, index_C);
  });

  std::cout << "Contraction; "
            << ::boba::detail::execution_space_name(space) << "; "
            << sizes_A << ","
            << sizes_B << "->"
            << tensor_C.sizes() << "; "
            << timer.duration << "; "
            << timer.units_string << std::endl;

  pass_or_fail(check, error_inf_norm, 1.0e-13);
}

//
// test_tensor_reduction_single
//

template <typename data_t, size_t dimension_A, bool force_naive>
void test_tensor_reduction_single(
  ::boba::Array<size_t, dimension_A> sizes_A,
  size_t contraction_index_A,
  bool& check)
{
  ::boba::Tensor<dimension_A, space, data_t> tensor_A(sizes_A);
  auto tensor_A_view = tensor_A.view();

  // A = sum of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_A_view.size(), [=] __boba_host_device__(size_t index_A)
  {
    auto mid_A = tensor_A_view.multiindex(index_A);
    tensor_A_view(mid_A) = ::boba::sum(mid_A);
  });

  ::boba::TicToc<tictoc_units> timer;
  auto tensor_C = ::boba::tensor_reduction<space, dimension_A, data_t, force_naive>(
    tensor_A,
    contraction_index_A);
  checkpoint();
  timer.end();

  auto tensor_C_view = tensor_C.view();
  size_t contraction_length = sizes_A[contraction_index_A];

  // Validate C
  data_t error_inf_norm = ::boba::lowest_value<data_t>();
  size_t loc = 0;

  ::boba::max_loc_reduce<boba::default_execution_space>(
    error_inf_norm, loc, 0_z, tensor_C.size(), [=] __boba_host_device__(::boba::reducer_index_t index_C, ::boba::max_loc_reducer_operator<data_t> & local_error)
  {
    const size_t index_C_size = static_cast<size_t>(index_C);
    auto mid_C = tensor_C_view.multiindex(index_C_size);
    data_t sum_1 = 0;
    for (size_t i = 0; i < dimension_A - 1; i++)
    {
      sum_1 += mid_C[i];
    }
    sum_1 *= contraction_length;
    data_t sum_2 = ::boba::sum_of_i(contraction_length - 1);

    data_t sum = sum_1 + sum_2;

    data_t c = tensor_C_view(index_C_size);
    data_t error_loc = ::boba::abs(c - sum);

    local_error.maxloc(error_loc, index_C);
  });

  std::cout << "Reduce single;   "
            << ::boba::detail::execution_space_name(space) << "; "
            << sizes_A << "->"
            << tensor_C.sizes() << "; "
            << timer.duration << "; "
            << timer.units_string << std::endl;
  pass_or_fail(check, error_inf_norm, 1.0e-13);
}

//
// test_tensor_reduction_double
//

template <typename data_t, size_t dimension_A, bool force_naive>
void test_tensor_reduction_double(
  ::boba::Array<size_t, dimension_A> sizes_A,
  size_t contraction_index_A,
  size_t contraction_index_B,
  bool& check)
{
  ::boba::Tensor<dimension_A, space, data_t> tensor_A(sizes_A);
  auto tensor_A_view = tensor_A.view();

  // A = sum of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_A_view.size(), [=] __boba_host_device__(size_t index_A)
  {
    auto mid_A = tensor_A_view.multiindex(index_A);
    tensor_A_view(mid_A) = ::boba::sum(mid_A);
  });

  ::boba::TicToc<tictoc_units> timer;
  auto tensor_C = ::boba::tensor_reduction<space, dimension_A, data_t, force_naive>(
    tensor_A,
    contraction_index_A,
    contraction_index_B);
  checkpoint();
  timer.end();

  auto tensor_C_view = tensor_C.view();
  size_t contraction_length_A = sizes_A[contraction_index_A];
  size_t contraction_length_B = sizes_A[contraction_index_B];

  // Validate C
  data_t error_inf_norm = ::boba::lowest_value<data_t>();
  size_t loc = 0;

  ::boba::max_loc_reduce<boba::default_execution_space>(
    error_inf_norm, loc, 0_z, tensor_C.size(), [=] __boba_host_device__(::boba::reducer_index_t index_C, ::boba::max_loc_reducer_operator<data_t> & local_error)
  {
    const size_t index_C_size = static_cast<size_t>(index_C);
    auto mid_C = tensor_C_view.multiindex(index_C_size);

    data_t sum_1 = 0;
    for (size_t i = 0; i < dimension_A - 2; i++)
    {
      sum_1 += mid_C[i];
    }
    sum_1 *= contraction_length_A * contraction_length_B;

    data_t sum_2 = contraction_length_B * ::boba::sum_of_i(contraction_length_A - 1);

    data_t sum_3 = contraction_length_A * ::boba::sum_of_i(contraction_length_B - 1);

    data_t sum = sum_1 + sum_2 + sum_3;

    data_t c = tensor_C_view(index_C_size);
    data_t error_loc = ::boba::abs(c - sum);

    local_error.maxloc(error_loc, index_C);
  });

  std::cout << "Reduce double;   "
            << ::boba::detail::execution_space_name(space) << "; "
            << sizes_A << "->"
            << tensor_C.sizes() << "; "
            << timer.duration << "; "
            << timer.units_string << std::endl;
  pass_or_fail(check, error_inf_norm, 1.0e-13);
}

//
// test_tensor_contraction_single_index_two_tensors
//

template <typename data_t, size_t dimension_A, size_t dimension_B, bool force_naive>
void test_tensor_contraction_single_index_two_tensors(
  ::boba::Array<size_t, dimension_A> sizes_A,
  ::boba::Array<size_t, dimension_B> sizes_B,
  size_t contraction_index_A,
  size_t contraction_index_B,
  bool& check)
{
  ::boba::Tensor<dimension_A, space, data_t> tensor_A(sizes_A);
  ::boba::Tensor<dimension_B, space, data_t> tensor_B(sizes_B);
  auto tensor_A_view = tensor_A.view();
  auto tensor_B_view = tensor_B.view();

  // A = sum of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_A_view.size(), [=] __boba_host_device__(size_t index_A)
  {
    auto mid_A = tensor_A_view.multiindex(index_A);
    tensor_A_view(mid_A) = ::boba::sum(mid_A);
  });

  // B = alternating sum/differences of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_B_view.size(), [=] __boba_host_device__(size_t index_B)
  {
    auto mid_B = tensor_B_view.multiindex(index_B);
    data_t sum = 0.0;
    for (size_t d = 0; d < dimension_B; d++)
    {
      data_t plus_or_minus = static_cast<data_t>(2 * ::boba::mod(d, 2)) - static_cast<data_t>(1.0);
      sum += plus_or_minus * static_cast<data_t>(mid_B[d]);
    }
    tensor_B_view(mid_B) = sum;
  });

  ::boba::TicToc<tictoc_units> timer;
  auto tensor_C = ::boba::tensor_contraction<1, space, dimension_A, dimension_B, data_t, force_naive>(
    tensor_A,
    tensor_B,
    {contraction_index_A},
    {contraction_index_B});
  checkpoint();
  timer.end();

  auto tensor_C_view = tensor_C.view();
  size_t contraction_length = sizes_A[contraction_index_A];

  // Validate C
  data_t error_inf_norm = ::boba::lowest_value<data_t>();
  size_t loc = 0;

  ::boba::max_loc_reduce<boba::default_execution_space>(
    error_inf_norm, loc, 0_z, tensor_C.size(), [=] __boba_host_device__(::boba::reducer_index_t index_C, ::boba::max_loc_reducer_operator<data_t> & local_error)
  {
    const size_t index_C_size = static_cast<size_t>(index_C);
    auto mid_C = tensor_C_view.multiindex(index_C_size);

    ::boba::Array<size_t, dimension_A - 1> cmid_A;
    for (size_t i = 0; i < dimension_A - 1; i++)
    {
      cmid_A[i] = mid_C[i];
    }

    ::boba::Array<size_t, dimension_B - 1> cmid_B;
    for (size_t i = 0; i < dimension_B - 1; i++)
    {
      cmid_B[i] = mid_C[(i + dimension_A) - 1];
    }

    data_t sum_1 = 0;
    for (size_t i = 0; i < dimension_A - 1; i++)
    {
      data_t alpha = cmid_A[i];
      for (size_t j = 0; j < dimension_B - 1; j++)
      {
        data_t beta = cmid_B[j];
        size_t ell = (j < contraction_index_B) ? j : j + 1;
        data_t plus_or_minus = static_cast<data_t>(2 * ::boba::mod(ell, 2)) - static_cast<data_t>(1.0);
        sum_1 += plus_or_minus * alpha * beta;
      }
    }
    sum_1 *= contraction_length;

    data_t sum_2 = 0.0;
    for (size_t i = 0; i < dimension_A - 1; i++)
    {
      data_t plus_or_minus = 2.0 * ::boba::mod(contraction_index_B, 2) - 1.0;
      sum_2 += plus_or_minus * cmid_A[i];
    }
    sum_2 *= ::boba::sum_of_i(contraction_length - 1);

    data_t sum_3 = 0;
    for (size_t j = 0; j < dimension_B - 1; j++)
    {
      data_t ell = (j < contraction_index_B) ? j : j + 1;
      data_t plus_or_minus = 2.0 * ::boba::mod(ell, 2) - 1.0;
      sum_3 += plus_or_minus * cmid_B[j];
    }
    sum_3 *= ::boba::sum_of_i(contraction_length - 1);

    data_t plus_or_minus = 2.0 * ::boba::mod(contraction_index_B, 2) - 1.0;
    data_t sum_4 = plus_or_minus * ::boba::sum_of_i2(contraction_length - 1);

    data_t sum = sum_1 + sum_2 + sum_3 + sum_4;

    data_t c = tensor_C_view(index_C_size);
    data_t error_loc = ::boba::abs(c - sum);

    local_error.maxloc(error_loc, index_C);
  });

  std::cout << "Contraction; "
            << ::boba::detail::execution_space_name(space) << "; "
            << sizes_A << ","
            << sizes_B << "->"
            << tensor_C.sizes() << "; "
            << timer.duration << "; "
            << timer.units_string << std::endl;

  pass_or_fail(check, error_inf_norm, 1.0e-13);
}

//
// test_tensor_contraction_double_index_two_tensors
//

template <typename data_t, size_t dimension_A, size_t dimension_B, bool force_naive>
void test_tensor_contraction_double_index_two_tensors(
  ::boba::Array<size_t, dimension_A> sizes_A,
  ::boba::Array<size_t, dimension_B> sizes_B,
  size_t contraction_1_index_A,
  size_t contraction_2_index_A,
  size_t contraction_1_index_B,
  size_t contraction_2_index_B,
  bool& check)
{
  ::boba::Tensor<dimension_A, space, data_t> tensor_A(sizes_A);
  ::boba::Tensor<dimension_B, space, data_t> tensor_B(sizes_B);
  auto tensor_A_view = tensor_A.view();
  auto tensor_B_view = tensor_B.view();

  // A = sum of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_A_view.size(), [=] __boba_host_device__(size_t index_A)
  {
    auto mid_A = tensor_A_view.multiindex(index_A);
    tensor_A_view(mid_A) = ::boba::sum(mid_A);
  });

  // B = alternating sum/differences of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_B_view.size(), [=] __boba_host_device__(size_t index_B)
  {
    auto mid_B = tensor_B_view.multiindex(index_B);
    data_t sum = 0;
    for (size_t d = 0; d < dimension_B; d++)
    {
      sum += static_cast<data_t>(2 * ::boba::mod(mid_B[d], 2)) - static_cast<data_t>(1.0);
    }
    tensor_B_view(mid_B) = sum;
  });

  ::boba::TicToc<tictoc_units> timer;
  auto tensor_C = ::boba::tensor_contraction<2, space, dimension_A, dimension_B, data_t, force_naive>(
    tensor_A,
    tensor_B,
    {contraction_1_index_A, contraction_2_index_A},
    {contraction_1_index_B, contraction_2_index_B});
  checkpoint();
  timer.end();

  auto tensor_C_view = tensor_C.view();
  size_t contraction_length_1 = sizes_A[contraction_1_index_A];
  size_t contraction_length_2 = sizes_A[contraction_2_index_A];

  data_t length_even_1 = ::boba::floor((contraction_length_1 + 1) / 2.0);
  data_t length_even_2 = ::boba::floor((contraction_length_2 + 1) / 2.0);
  data_t length_odd_1 = ::boba::floor(contraction_length_1 / 2.0);
  data_t length_odd_2 = ::boba::floor(contraction_length_2 / 2.0);

  data_t sum_even_1 = ::boba::sum_of_i(length_even_1 - 1);
  data_t sum_even_2 = ::boba::sum_of_i(length_even_2 - 1);
  data_t sum_odd_1 = (contraction_length_1 == 1) ? 0.0 : ::boba::sum_of_i(length_odd_1 - 1);
  data_t sum_odd_2 = (contraction_length_2 == 1) ? 0.0 : ::boba::sum_of_i(length_odd_2 - 1);

  // Validate C
  data_t error_inf_norm = ::boba::lowest_value<data_t>();
  size_t loc = 0;

  ::boba::max_loc_reduce<boba::default_execution_space>(
    error_inf_norm, loc, 0_z, tensor_C.size(), [=] __boba_host_device__(::boba::reducer_index_t index_C, ::boba::max_loc_reducer_operator<data_t> & local_error)
  {
    const size_t index_C_size = static_cast<size_t>(index_C);
    auto mid_C = tensor_C_view.multiindex(index_C_size);

    ::boba::Array<size_t, dimension_A - 2> cmid_A;
    for (size_t i = 0; i < dimension_A - 2; i++)
    {
      cmid_A[i] = mid_C[i];
    }

    ::boba::Array<size_t, dimension_B - 2> cmid_B;
    for (size_t i = 0; i < dimension_B - 2; i++)
    {
      cmid_B[i] = mid_C[(i + dimension_A) - 2];
    }

    data_t sum_A = ::boba::sum(cmid_A);
    data_t sum_B = 0;
    for (size_t d = 0; d < dimension_B - 2; d++)
    {
      sum_B += static_cast<data_t>(2.0 * ::boba::mod(cmid_B[d], 2)) - static_cast<data_t>(1.0);
    }

    data_t sum_1 = (length_even_1 * length_even_2 * (sum_A) + 2 * (length_even_1 * sum_even_2 + length_even_2 * sum_even_1)) * (sum_B - 2);
    data_t sum_2 = (length_odd_1 * length_even_2 * (sum_A + 1) + 2 * (length_odd_1 * sum_even_2 + length_even_2 * sum_odd_1)) * (sum_B);
    data_t sum_3 = (length_even_1 * length_odd_2 * (sum_A + 1) + 2 * (length_even_1 * sum_odd_2 + length_odd_2 * sum_even_1)) * (sum_B);
    data_t sum_4 = (length_odd_1 * length_odd_2 * (sum_A + 2) + 2 * (length_odd_1 * sum_odd_2 + length_odd_2 * sum_odd_1)) * (sum_B + 2);

    data_t sum = sum_1 + sum_2 + sum_3 + sum_4;

    data_t c = tensor_C_view(index_C_size);
    data_t error_loc = ::boba::abs(c - sum);

    local_error.maxloc(error_loc, index_C);
  });

  std::cout << "Double_contraction; "
            << ::boba::detail::execution_space_name(space) << "; "
            << sizes_A << ","
            << sizes_B << "->"
            << tensor_C.sizes() << "; "
            << timer.duration << "; "
            << timer.units_string << std::endl;

  pass_or_fail(check, error_inf_norm, 1.0e-13);
}

//
// test_tensor_contraction_triple_index_two_tensors
//

template <typename data_t, size_t dimension_A, size_t dimension_B, bool force_naive>
void test_tensor_contraction_triple_index_two_tensors(
  ::boba::Array<size_t, dimension_A> sizes_A,
  ::boba::Array<size_t, dimension_B> sizes_B,
  size_t contraction_1_index_A,
  size_t contraction_2_index_A,
  size_t contraction_3_index_A,
  size_t contraction_1_index_B,
  size_t contraction_2_index_B,
  size_t contraction_3_index_B,
  bool& check)
{
  ::boba::Tensor<dimension_A, space, data_t> tensor_A(sizes_A);
  ::boba::Tensor<dimension_B, space, data_t> tensor_B(sizes_B);
  auto tensor_A_view = tensor_A.view();
  auto tensor_B_view = tensor_B.view();

  // A = sum of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_A_view.size(), [=] __boba_host_device__(size_t index_A)
  {
    auto mid_A = tensor_A_view.multiindex(index_A);
    tensor_A_view(mid_A) = ::boba::sum(mid_A);
  });

  // B = alternating sum/differences of the indices
  checkpoint();
  ::boba::detail::loop<space>(
    0, tensor_B_view.size(), [=] __boba_host_device__(size_t index_B)
  {
    auto mid_B = tensor_B_view.multiindex(index_B);
    data_t sum = 0;
    for (size_t d = 0; d < dimension_B; d++)
    {
      sum += static_cast<data_t>(2 * ::boba::mod(mid_B[d], 2)) - static_cast<data_t>(1.0);
    }
    tensor_B_view(mid_B) = sum;
  });

  ::boba::TicToc<tictoc_units> timer;
  auto tensor_C = ::boba::tensor_contraction<3, space, dimension_A, dimension_B, data_t, force_naive>(
    tensor_A,
    tensor_B,
    {contraction_1_index_A, contraction_2_index_A, contraction_3_index_A},
    {contraction_1_index_B, contraction_2_index_B, contraction_3_index_B});
  checkpoint();
  timer.end();

  auto tensor_C_view = tensor_C.view();
  size_t contraction_length_1 = sizes_A[contraction_1_index_A];
  size_t contraction_length_2 = sizes_A[contraction_2_index_A];
  size_t contraction_length_3 = sizes_A[contraction_3_index_A];

  auto alternating_sum = [](size_t length)
  {
    return -static_cast<data_t>(length % 2);
  };

  auto index_sum = [](size_t length)
  {
    return static_cast<data_t>(::boba::sum_of_i(length - 1));
  };

  auto alternating_weighted_index_sum = [](size_t length)
  {
    data_t half_length = static_cast<data_t>(length / 2);
    return (length % 2 == 0) ? half_length : -half_length;
  };

  data_t length_1 = static_cast<data_t>(contraction_length_1);
  data_t length_2 = static_cast<data_t>(contraction_length_2);
  data_t length_3 = static_cast<data_t>(contraction_length_3);
  data_t contraction_size = length_1 * length_2 * length_3;

  data_t alternating_1 = alternating_sum(contraction_length_1);
  data_t alternating_2 = alternating_sum(contraction_length_2);
  data_t alternating_3 = alternating_sum(contraction_length_3);

  data_t index_sum_1 = index_sum(contraction_length_1);
  data_t index_sum_2 = index_sum(contraction_length_2);
  data_t index_sum_3 = index_sum(contraction_length_3);

  data_t alternating_weighted_1 = alternating_weighted_index_sum(contraction_length_1);
  data_t alternating_weighted_2 = alternating_weighted_index_sum(contraction_length_2);
  data_t alternating_weighted_3 = alternating_weighted_index_sum(contraction_length_3);

  // Validate C
  data_t error_inf_norm = ::boba::lowest_value<data_t>();
  size_t loc = 0;

  ::boba::max_loc_reduce<boba::default_execution_space>(
    error_inf_norm, loc, 0_z, tensor_C.size(), [=] __boba_host_device__(::boba::reducer_index_t index_C, ::boba::max_loc_reducer_operator<data_t> & local_error)
  {
    const size_t index_C_size = static_cast<size_t>(index_C);
    auto mid_C = tensor_C_view.multiindex(index_C_size);

    ::boba::Array<size_t, dimension_A - 3> cmid_A;
    for (size_t i = 0; i < dimension_A - 3; i++)
    {
      cmid_A[i] = mid_C[i];
    }

    ::boba::Array<size_t, dimension_B - 3> cmid_B;
    for (size_t i = 0; i < dimension_B - 3; i++)
    {
      cmid_B[i] = mid_C[(i + dimension_A) - 3];
    }

    data_t sum_A = ::boba::sum(cmid_A);
    data_t sum_B = 0;
    for (size_t d = 0; d < dimension_B - 3; d++)
    {
      sum_B += static_cast<data_t>(2.0 * ::boba::mod(cmid_B[d], 2)) - static_cast<data_t>(1.0);
    }

    data_t contracted_index_sum =
      index_sum_1 * length_2 * length_3 + length_1 * index_sum_2 * length_3 + length_1 * length_2 * index_sum_3;

    data_t contracted_alternating_sum =
      alternating_1 * length_2 * length_3 + length_1 * alternating_2 * length_3 + length_1 * length_2 * alternating_3;

    data_t contracted_cross_sum =
      alternating_weighted_1 * length_2 * length_3 + index_sum_1 * alternating_2 * length_3 + index_sum_1 * length_2 * alternating_3 + alternating_1 * index_sum_2 * length_3 + length_1 * alternating_weighted_2 * length_3 + length_1 * index_sum_2 * alternating_3 + alternating_1 * length_2 * index_sum_3 + length_1 * alternating_2 * index_sum_3 + length_1 * length_2 * alternating_weighted_3;

    data_t sum =
      sum_A * sum_B * contraction_size + sum_A * contracted_alternating_sum + sum_B * contracted_index_sum + contracted_cross_sum;

    data_t c = tensor_C_view(index_C_size);
    data_t error_loc = ::boba::abs(c - sum);

    local_error.maxloc(error_loc, index_C);
  });

  std::cout << "Triple_contraction; "
            << ::boba::detail::execution_space_name(space) << "; "
            << sizes_A << ","
            << sizes_B << "->"
            << tensor_C.sizes() << "; "
            << timer.duration << "; "
            << timer.units_string << std::endl;

  pass_or_fail(check, error_inf_norm, 1.0e-13);
}

template <typename data_t, bool force_naive>
void test_cases(bool& check, size_t size_factor, size_t which)
{
  // TODO<external> hiptensor fails at certain sizes (set this to 8)
  size_t max_size_factor = 6;

  //
  // Test permutation
  //
  checkpoint();
  if ((which == 0) or (which == 1))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (4 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (5 * size_factor) : (5 * j);
        test_tensor_permutation<data_t, 2, force_naive>({N, M}, {1, 0}, check);
      }
    }
  }

  //
  // Test permutation
  //
  checkpoint();
  if ((which == 0) or (which == 2))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (3 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (5 * size_factor) : (5 * j);
        test_tensor_permutation<data_t, 4, force_naive>({N, N, M, M}, {2, 1, 3, 0}, check);
      }
    }
  }

  //
  // Test permutation einsum
  //
  checkpoint();
  if ((which == 0) or (which == 2))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (3 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (5 * size_factor) : (5 * j);
        test_tensor_permutation_einsum<data_t, 4, force_naive>(
          {"a", "b", "c", "d"},
          {N, N, M, M},
          {"c", "b", "d", "a"},
          check);
      }
    }
  }

  //
  // Test reduction einsum
  //
  checkpoint();
  if ((which == 0) or (which == 3))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (3 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (5 * size_factor) : (5 * j);
        test_tensor_reduction_einsum<data_t, 4, force_naive>(
          {"a", "b", "c", "d"},
          {N, N, M, M},
          {"c", "a"},
          check);
      }
    }
  }

  //
  // Test contraction einsum
  //
  checkpoint();
  if ((which == 0) or (which == 4))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (3 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (5 * size_factor) : (5 * j);
        test_tensor_contraction_einsum<data_t, 4, 4, force_naive>(
          {"a", "b", "c", "d"}, {N, N, M, M}, {"e", "b", "f", "d"}, {N, N, M, M}, {"f", "a", "c", "e"}, check);
      }
    }
  }

  //
  // Test tensor single reductions
  //
  checkpoint();
  if ((which == 0) or (which == 5))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (4 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (5 * size_factor) : (5 * j);
        test_tensor_reduction_single<data_t, 4, force_naive>({N, N, M, M}, 2, check);
      }
    }
  }
  if ((which == 0) or (which == 6))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (3 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (2 * size_factor) : (5 * j);
        test_tensor_reduction_single<data_t, 6, force_naive>({N, M, M, N, M, M}, 0, check);
      }
    }
  }
  //
  // Test tensor double reductions
  //
  checkpoint();
  if ((which == 0) or (which == 7))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (4 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (5 * size_factor) : (5 * j);
        test_tensor_reduction_double<data_t, 4, force_naive>({N, N, M, M}, 2, 3, check);
      }
    }
  }
  if ((which == 0) or (which == 8))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {

      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (3 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (2 * size_factor) : (5 * j);
        test_tensor_reduction_double<data_t, 6, force_naive>({N, M, M, N, M, M}, 0, 3, check);
      }
    }
  }
  //
  // Test tensor_contraction_single_index_two_tensors
  //
  checkpoint();
  if ((which == 0) or (which == 9))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (2 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (2 * size_factor) : (5 * j);
        test_tensor_contraction_single_index_two_tensors<data_t, 2, 2, force_naive>({N, M}, {M, M}, 1, 0, check);
      }
    }
  }
  checkpoint();
  if ((which == 0) or (which == 10))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (2 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (3 * size_factor) : (5 * j);
        test_tensor_contraction_single_index_two_tensors<data_t, 4, 3, force_naive>({N, N, M, M}, {N, M, M}, 1, 0, check);
      }
    }
  }

  //
  // Test tensor_contraction_double_index_two_tensors
  //
  checkpoint();
  if ((which == 0) or (which == 11))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (3 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (4 * size_factor) : (5 * j);
        test_tensor_contraction_double_index_two_tensors<data_t, 3, 2, force_naive>({N, M, N}, {N, M}, 0, 1, 0, 1, check);
      }
    }
  }
  checkpoint();
  if ((which == 0) or (which == 12))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      for (size_t j = 1; j < max_size_factor; j++)
      {
        size_t N = (size_factor > 0) ? (2 * size_factor) : (4 * i);
        size_t M = (size_factor > 0) ? (3 * size_factor) : (5 * j);
        test_tensor_contraction_double_index_two_tensors<data_t, 4, 3, force_naive>({N, M, N, M}, {N, M, N}, 0, 2, 0, 2, check);
      }
    }
  }

  checkpoint();
  if ((which == 0) or (which == 13))
  {
    auto i = 2_z;
    auto j = 3_z;
    size_t N = (size_factor > 0) ? (2 * size_factor) : (2 * i);
    size_t M = (size_factor > 0) ? (3 * size_factor) : (3 * j);

    test_tensor_contraction_triple_index_two_tensors<data_t, 4, 3, force_naive>({N, N, N, M}, {N, N, N}, 0, 1, 2, 0, 1, 2, check);
  }

  checkpoint();
  if ((which == 0) or (which == 14))
  {
    auto i = 2_z;
    auto j = 3_z;
    size_t N = (size_factor > 0) ? (2 * size_factor) : (2 * i);
    size_t M = (size_factor > 0) ? (3 * size_factor) : (3 * j);

    test_tensor_contraction_triple_index_two_tensors<data_t, 5, 4, force_naive>({N, M, N, M, N}, {N, M, N, N}, 0, 2, 4, 0, 2, 3, check);
  }

  //
  // Test test_tensor_contraction_single_index_two_tensors, degenerate case
  //
  checkpoint();
  if ((which == 0) or (which == 15))
  {
    for (size_t i = 1; i < max_size_factor; i++)
    {
      size_t N = (size_factor > 0) ? (3 * size_factor) : (4 * i);
      test_tensor_contraction_single_index_two_tensors<data_t, 1, 1, force_naive>({N}, {N}, 0, 0, check);
    }
  }
}

int main(int argc, char* argv[])
{

  boba::splash();
  std::cout << "Tests for tensor operations." << std::endl;
  boba::init();

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;
  checkpoint();

  size_t size_factor = 0;
  size_t which = 0;
  size_t precision = 0;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(size_factor,
                             "-s",
                             "--size_factor",
                             "Size factor that will increase the size of the tensors. By default a variety of sizes will be run. Otherwise, a fixed sized will be run.");

  args.add_optional_argument(which,
                             "-w",
                             "--which",
                             "Choose to run a specific kernel.");

  args.add_optional_argument(precision,
                             "-p",
                             "--precision",
                             "Floating type precision: 0 - both, 1 - double 2, - single.");

  args.parse_check();
  size_t repetitions = 1;

  bool run_single_precision = (precision == 0) or (precision == 1);

  boba_print("-----------------------------------------------------------------");
  boba_print(" Testing BoBa Fallback schemes ");
  boba_print("-----------------------------------------------------------------");
  for (size_t repetition = 0; repetition < repetitions; repetition++)
  {
    if (run_single_precision)
    {
      test_cases<float, true>(check, size_factor, which);
    }
    else
    {
      test_cases<double, true>(check, size_factor, which);
    }
  }

  boba_print("-----------------------------------------------------------------");
  boba_print(" Testing third party schemes ");
  boba_print("-----------------------------------------------------------------");
  for (size_t repetition = 0; repetition < repetitions; repetition++)
  {
    if (run_single_precision)
    {
      test_cases<float, false>(check, size_factor, which);
    }
    else
    {
      test_cases<double, false>(check, size_factor, which);
    }
  }

  boba::finalize();
  return final_check(check);
}

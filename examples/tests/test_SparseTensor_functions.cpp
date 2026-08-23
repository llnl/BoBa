// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>
#include <iostream>
#include <string>

constexpr boba::execution_space space = boba::default_execution_space;

template <std::size_t dimension>
::boba::Array<::boba::index_t, dimension> make_sizes()
{
  ::boba::Array<::boba::index_t, dimension> sizes;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    sizes[d] = static_cast<::boba::index_t>(d + 2);
  }
  return sizes;
}

template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
bool contraction_indices_match(
  ::boba::Array<::boba::index_t, dimension_A> indices_A,
  ::boba::Array<::boba::index_t, dimension_B> indices_B,
  ::boba::Array<size_t, contractions> contraction_dimensions_A,
  ::boba::Array<size_t, contractions> contraction_dimensions_B)
{
  for (std::size_t c = 0; c < contractions; ++c)
  {
    if (indices_A[contraction_dimensions_A[c]] != indices_B[contraction_dimensions_B[c]])
    {
      return false;
    }
  }
  return true;
}

template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
::boba::Array<::boba::index_t, dimension_A + dimension_B - 2 * contractions> contraction_output_indices(
  ::boba::Array<::boba::index_t, dimension_A> indices_A,
  ::boba::Array<::boba::index_t, dimension_B> indices_B,
  ::boba::Array<size_t, contractions> contraction_dimensions_A,
  ::boba::Array<size_t, contractions> contraction_dimensions_B)
{
  constexpr std::size_t output_dimension = dimension_A + dimension_B - 2 * contractions;
  ::boba::Array<::boba::index_t, output_dimension> output_indices;
  std::size_t output_dimension_id = 0;
  for (std::size_t d = 0; d < dimension_A; ++d)
  {
    bool is_contracted = false;
    for (std::size_t c = 0; c < contractions; ++c)
    {
      is_contracted = is_contracted || (contraction_dimensions_A[c] == d);
    }
    if (!is_contracted)
    {
      output_indices[output_dimension_id] = indices_A[d];
      ++output_dimension_id;
    }
  }
  for (std::size_t d = 0; d < dimension_B; ++d)
  {
    bool is_contracted = false;
    for (std::size_t c = 0; c < contractions; ++c)
    {
      is_contracted = is_contracted || (contraction_dimensions_B[c] == d);
    }
    if (!is_contracted)
    {
      output_indices[output_dimension_id] = indices_B[d];
      ++output_dimension_id;
    }
  }
  return output_indices;
}

template <std::size_t dimension>
::boba::Tensor<dimension, ::boba::host_space, double> make_sparse_test_tensor(
  ::boba::Array<::boba::index_t, dimension> sizes,
  ::boba::index_t seed)
{
  ::boba::Tensor<dimension, ::boba::host_space, double> tensor(sizes);
  auto tensor_view = tensor.view();
  for (::boba::index_t linear_index = 0; linear_index < tensor.size(); ++linear_index)
  {
    const bool is_nonzero = ((linear_index + seed) % 3 == 0) || ((linear_index + seed) % 7 == 0);
    const double sign = (linear_index % 2 == 0) ? 1.0 : -1.0;
    tensor_view(linear_index) = is_nonzero ? sign * static_cast<double>((linear_index % 11) + 1) : 0.0;
  }
  return tensor;
}

template <std::size_t dimension, ::boba::execution_space sparse_space>
void compare_sparse_to_dense(
  ::boba::SparseTensor<dimension, sparse_space, double> const& sparse,
  ::boba::Tensor<dimension, ::boba::host_space, double> const& expected,
  bool& check)
{
  ::boba::SparseTensor<dimension, ::boba::host_space, double> sparse_host(sparse);
  auto sparse_view = sparse_host.const_view();
  auto expected_view = expected.const_view();
  double error_inf_norm = 0.0;
  for (::boba::index_t linear_index = 0; linear_index < expected.size(); ++linear_index)
  {
    error_inf_norm = std::max(error_inf_norm, ::boba::abs(sparse_view(linear_index) - expected_view(linear_index)));
  }
  pass_or_fail(check, error_inf_norm, 1.0e-14);
}

template <std::size_t dimension>
void test_sparse_permutation(
  ::boba::Array<::boba::index_t, dimension> permutation,
  bool& check)
{
  std::cout << "Testing sparse permutation dimension " << dimension << std::endl;

  auto dense = make_sparse_test_tensor(make_sizes<dimension>(), 1);
  auto sparse_host = ::boba::convert_to_SparseTensor(dense, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  ::boba::SparseTensor<dimension, space, double> sparse(sparse_host);

  ::boba::Tensor<dimension, ::boba::host_space, double> expected(::boba::permute(dense.sizes(), permutation));
  expected.fill_with_zeros();
  auto dense_view = dense.const_view();
  auto expected_view = expected.view();
  for (::boba::index_t linear_index = 0; linear_index < dense.size(); ++linear_index)
  {
    const auto indices = dense.multiindex(linear_index);
    expected_view(::boba::permute(indices, permutation)) = dense_view(linear_index);
  }

  ::boba::permute<space, dimension, double, true>(sparse, permutation);
  pass_or_fail_bool(check, sparse.sizes() == expected.sizes());
  compare_sparse_to_dense(sparse, expected, check);

  ::boba::SparseTensor<dimension, ::boba::host_space, double> sparse_host_result(sparse);
  pass_or_fail(check, sparse_host_result.const_view()(::boba::filled_array<dimension>(static_cast<::boba::index_t>(0))) - expected.const_view()(0), 1.0e-14);
}

template <std::size_t reductions, std::size_t dimension>
void test_sparse_reduction(
  ::boba::Array<size_t, reductions> reduction_dimensions,
  bool& check)
{
  std::cout << "Testing sparse reduction dimension " << dimension << " reductions " << reductions << std::endl;
  constexpr std::size_t output_dimension = dimension - reductions;

  auto dense = make_sparse_test_tensor(make_sizes<dimension>(), 2);
  auto sparse_host = ::boba::convert_to_SparseTensor(dense, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  ::boba::SparseTensor<dimension, space, double> sparse(sparse_host);

  auto output_mider = ::boba::make_contracted_dimensions<reductions>(dense.sizes(), reduction_dimensions);
  ::boba::Tensor<output_dimension, ::boba::host_space, double> expected(output_mider.sizes());
  expected.fill_with_zeros();
  auto dense_view = dense.const_view();
  auto expected_view = expected.view();
  for (::boba::index_t linear_index = 0; linear_index < dense.size(); ++linear_index)
  {
    const auto indices = dense.multiindex(linear_index);
    expected_view(::boba::delete_elements(indices, reduction_dimensions)) += dense_view(linear_index);
  }

  auto reduced = ::boba::tensor_reduction<reductions, space, dimension, double, true>(sparse, reduction_dimensions);
  pass_or_fail_bool(check, reduced.sizes() == expected.sizes());
  compare_sparse_to_dense(reduced, expected, check);
}

template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
void test_sparse_contraction(
  ::boba::Array<::boba::index_t, dimension_A> sizes_A,
  ::boba::Array<::boba::index_t, dimension_B> sizes_B,
  ::boba::Array<size_t, contractions> contraction_dimensions_A,
  ::boba::Array<size_t, contractions> contraction_dimensions_B,
  bool& check)
{
  std::cout << "Testing sparse contraction dimensions " << dimension_A << " and " << dimension_B << " contractions " << contractions << std::endl;
  constexpr std::size_t output_dimension = dimension_A + dimension_B - 2 * contractions;

  auto dense_A = make_sparse_test_tensor(sizes_A, 0);
  auto dense_B = make_sparse_test_tensor(sizes_B, 4);
  auto sparse_A_host = ::boba::convert_to_SparseTensor(dense_A, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  auto sparse_B_host = ::boba::convert_to_SparseTensor(dense_B, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  ::boba::SparseTensor<dimension_A, space, double> sparse_A(sparse_A_host);
  ::boba::SparseTensor<dimension_B, space, double> sparse_B(sparse_B_host);

  auto contracted_A_mider = ::boba::make_contracted_dimensions<contractions>(sizes_A, contraction_dimensions_A);
  auto contracted_B_mider = ::boba::make_contracted_dimensions<contractions>(sizes_B, contraction_dimensions_B);
  ::boba::Tensor<output_dimension, ::boba::host_space, double> expected(::boba::concatenate(contracted_A_mider.sizes(), contracted_B_mider.sizes()));
  expected.fill_with_zeros();

  auto dense_A_view = dense_A.const_view();
  auto dense_B_view = dense_B.const_view();
  auto expected_view = expected.view();
  for (::boba::index_t linear_index_A = 0; linear_index_A < dense_A.size(); ++linear_index_A)
  {
    const auto indices_A = dense_A.multiindex(linear_index_A);
    for (::boba::index_t linear_index_B = 0; linear_index_B < dense_B.size(); ++linear_index_B)
    {
      const auto indices_B = dense_B.multiindex(linear_index_B);
      if (contraction_indices_match(indices_A, indices_B, contraction_dimensions_A, contraction_dimensions_B))
      {
        const auto indices_C = contraction_output_indices(indices_A, indices_B, contraction_dimensions_A, contraction_dimensions_B);
        expected_view(indices_C) += dense_A_view(linear_index_A) * dense_B_view(linear_index_B);
      }
    }
  }

  auto contracted = ::boba::tensor_contraction<contractions, space, dimension_A, dimension_B, double, true>(
    sparse_A,
    sparse_B,
    contraction_dimensions_A,
    contraction_dimensions_B);
  pass_or_fail_bool(check, contracted.sizes() == expected.sizes());
  compare_sparse_to_dense(contracted, expected, check);
}

void test_sparse_label_apis(bool& check)
{
  std::cout << "Testing sparse label APIs" << std::endl;
  auto dense = make_sparse_test_tensor<3>({2, 3, 4}, 3);
  auto sparse = ::boba::convert_to_SparseTensor(dense, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });

  auto reduced = ::boba::tensor_reduction<1, ::boba::host_space, 3, double, true>(
    ::boba::Array<std::string, 3>{"i", "j", "k"},
    sparse,
    ::boba::Array<std::string, 2>{"k", "i"});
  auto direct = ::boba::tensor_reduction<1, ::boba::host_space, 3, double, true>(
    sparse,
    ::boba::Array<size_t, 1>{1});
  ::boba::permute<::boba::host_space, 2, double, true>(direct, ::boba::Array<::boba::index_t, 2>{1, 0});

  ::boba::Tensor<2, ::boba::host_space, double> expected(direct.sizes());
  expected.fill_with_zeros();
  auto direct_view = direct.const_view();
  auto expected_view = expected.view();
  for (::boba::index_t linear_index = 0; linear_index < expected.size(); ++linear_index)
  {
    expected_view(linear_index) = direct_view(linear_index);
  }
  compare_sparse_to_dense(reduced, expected, check);

  auto dense_A = make_sparse_test_tensor<2>({2, 3}, 5);
  auto dense_B = make_sparse_test_tensor<2>({4, 2}, 6);
  auto sparse_A = ::boba::convert_to_SparseTensor(dense_A, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  auto sparse_B = ::boba::convert_to_SparseTensor(dense_B, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  auto contracted = ::boba::tensor_contraction<1, ::boba::host_space, 2, 2, double, true>(
    ::boba::Array<std::string, 2>{"i", "j"},
    sparse_A,
    ::boba::Array<std::string, 2>{"k", "i"},
    sparse_B,
    ::boba::Array<std::string, 2>{"k", "j"});
  auto direct_contracted = ::boba::tensor_contraction<1, ::boba::host_space, 2, 2, double, true>(
    sparse_A,
    sparse_B,
    ::boba::Array<size_t, 1>{0},
    ::boba::Array<size_t, 1>{1});
  ::boba::permute<::boba::host_space, 2, double, true>(direct_contracted, ::boba::Array<::boba::index_t, 2>{1, 0});

  ::boba::Tensor<2, ::boba::host_space, double> contraction_expected(direct_contracted.sizes());
  contraction_expected.fill_with_zeros();
  auto direct_contracted_view = direct_contracted.const_view();
  auto contraction_expected_view = contraction_expected.view();
  for (::boba::index_t linear_index = 0; linear_index < contraction_expected.size(); ++linear_index)
  {
    contraction_expected_view(linear_index) = direct_contracted_view(linear_index);
  }
  compare_sparse_to_dense(contracted, contraction_expected, check);
}

void test_sparse_filter_small_entries(bool& check)
{
  std::cout << "Testing sparse small-entry filtering" << std::endl;
  ::boba::SparseTensor<1, ::boba::host_space, double> sparse({4});
  sparse.set({0}, 1.0);
  sparse.set({1}, 1.0e-18);
  sparse.set({2}, 0.0);
  sparse.set({3}, -2.0);

  pass_or_fail(check, ::boba::norm_inf(sparse) - 2.0, 1.0e-14);
  auto filtered = ::boba::filter_small_entries(sparse);
  pass_or_fail_bool(check, filtered.number_nonzeros() == 2);
  pass_or_fail(check, filtered({0}) - 1.0, 1.0e-14);
  pass_or_fail(check, filtered({1}), 1.0e-14);
  pass_or_fail(check, filtered({2}), 1.0e-14);
  pass_or_fail(check, filtered({3}) + 2.0, 1.0e-14);
}

int main()
{
  boba::init();

  bool check = true;

  test_sparse_permutation<2>({1, 0}, check);
  test_sparse_permutation<3>({2, 0, 1}, check);
  test_sparse_permutation<4>({3, 1, 0, 2}, check);
  test_sparse_permutation<5>({4, 2, 0, 3, 1}, check);
  test_sparse_permutation<6>({5, 0, 4, 1, 3, 2}, check);
  test_sparse_reduction<1, 2>({0}, check);
  test_sparse_reduction<1, 3>({1}, check);
  test_sparse_reduction<2, 4>({0, 3}, check);
  test_sparse_reduction<2, 5>({1, 4}, check);
  test_sparse_reduction<3, 6>({0, 2, 5}, check);
  test_sparse_contraction<1, 2, 2>({2, 3}, {3, 4}, {1}, {0}, check);
  test_sparse_contraction<2, 3, 3>({2, 3, 4}, {4, 5, 3}, {1, 2}, {2, 0}, check);
  test_sparse_contraction<1, 4, 3>({2, 2, 2, 2}, {2, 2, 2}, {1}, {0}, check);
  test_sparse_contraction<3, 6, 4>({2, 2, 2, 2, 2, 2}, {2, 2, 2, 2}, {0, 2, 5}, {1, 2, 3}, check);
  test_sparse_label_apis(check);
  test_sparse_filter_small_entries(check);

  const int result = final_check(check);
  boba::finalize();
  return result;
}

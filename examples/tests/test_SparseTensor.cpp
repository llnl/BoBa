// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>
#include <iostream>

constexpr boba::execution_space space = boba::default_execution_space;

template <std::size_t dimension>
::boba::Array<::boba::index_t, dimension> make_sizes()
{
  ::boba::Array<::boba::index_t, dimension> sizes;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    sizes[d] = static_cast<::boba::index_t>(d + 4);
  }
  return sizes;
}

template <std::size_t dimension>
::boba::Array<::boba::index_t, dimension> make_middle_index(
  ::boba::Array<::boba::index_t, dimension> sizes)
{
  ::boba::Array<::boba::index_t, dimension> indices;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    indices[d] = sizes[d] / 2;
  }
  return indices;
}

template <std::size_t dimension>
::boba::Array<::boba::index_t, dimension> make_last_index(
  ::boba::Array<::boba::index_t, dimension> sizes)
{
  ::boba::Array<::boba::index_t, dimension> indices;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    indices[d] = sizes[d] - 1;
  }
  return indices;
}

template <std::size_t dimension, ::boba::execution_space sparse_space>
void validate_view(const ::boba::SparseTensor<dimension, sparse_space, double>& sparse,
                   const ::boba::SparseTensor<dimension, ::boba::host_space, double>& expected,
                   bool& check)
{
  ::boba::Tensor<dimension, sparse_space, double> dense_from_view(sparse.sizes());
  auto sparse_view = sparse.const_view();
  auto dense_view = dense_from_view.view();

  ::boba::detail::loop<sparse_space>(0, dense_from_view.size(), [=] __boba_host_device__(::boba::index_t linear_index)
  {
    dense_view(linear_index) = sparse_view(linear_index);
  });

  ::boba::Tensor<dimension, ::boba::host_space, double> dense_host(dense_from_view);
  auto dense_host_view = dense_host.const_view();
  double error_inf_norm = 0.0;
  for (::boba::index_t linear_index = 0; linear_index < expected.size(); ++linear_index)
  {
    const auto indices = expected.multiindex(linear_index);
    error_inf_norm = std::max(error_inf_norm, ::boba::abs(dense_host_view(linear_index) - expected(indices)));
  }
  pass_or_fail(check, error_inf_norm, 1.0e-14);
}

template <std::size_t dimension>
void validate_host_index_lists(const ::boba::SparseTensor<dimension, ::boba::host_space, double>& sparse,
                               bool& check)
{
  for (std::size_t d = 0; d < dimension; ++d)
  {
    pass_or_fail_bool(check, sparse.index_list(d).size() == sparse.number_nonzeros());
    pass_or_fail_bool(check, sparse.const_view().index_list(d) == sparse.index_list(d).const_view().data());
  }
}

template <std::size_t dimension>
void test_manual_sparse_tensor(bool& check)
{
  const auto sizes = make_sizes<dimension>();
  const auto zero_index = ::boba::filled_array<dimension>(static_cast<::boba::index_t>(0));
  const auto middle_index = make_middle_index(sizes);
  const auto last_index = make_last_index(sizes);
  auto missing_index = zero_index;
  missing_index[0] = 1;

  ::boba::SparseTensor<dimension, ::boba::host_space, double> sparse(sizes);

  pass_or_fail_bool(check, sparse.sizes() == sizes);
  pass_or_fail_bool(check, sparse.size() == ::boba::product(sizes));
  pass_or_fail_bool(check, sparse.number_nonzeros() == 0);
  pass_or_fail(check, sparse(middle_index), 1.0e-14);
  pass_or_fail(check, sparse.const_view()(middle_index), 1.0e-14);

  sparse.set(zero_index, 1.5);
  sparse.set(middle_index, -2.0);
  sparse.set(last_index, 3.25);

  pass_or_fail_bool(check, sparse.number_nonzeros() == 3);
  pass_or_fail_bool(check, sparse.contains(zero_index));
  pass_or_fail_bool(check, sparse.contains(middle_index));
  pass_or_fail_bool(check, sparse.contains(last_index));
  pass_or_fail(check, sparse(zero_index) - 1.5, 1.0e-14);
  pass_or_fail(check, sparse.const_view()(middle_index) + 2.0, 1.0e-14);
  pass_or_fail(check, sparse.const_view()(last_index) - 3.25, 1.0e-14);
  pass_or_fail(check, sparse.const_view()(missing_index), 1.0e-14);
  pass_or_fail(check, ::boba::norm_l1(sparse) - 6.75, 1.0e-14);
  validate_host_index_lists(sparse, check);

  sparse.set(middle_index, 4.5);
  pass_or_fail_bool(check, sparse.number_nonzeros() == 3);
  pass_or_fail(check, sparse(middle_index) - 4.5, 1.0e-14);
  pass_or_fail(check, ::boba::norm_l1(sparse) - 9.25, 1.0e-14);

  sparse.set(zero_index, 0.0);
  pass_or_fail_bool(check, sparse.number_nonzeros() == 3);
  pass_or_fail_bool(check, sparse.contains(zero_index));
  pass_or_fail(check, sparse(zero_index), 1.0e-14);
  pass_or_fail(check, ::boba::norm_l1(sparse) - 7.75, 1.0e-14);

  sparse.set(zero_index, 0.0);
  pass_or_fail_bool(check, sparse.number_nonzeros() == 3);
  pass_or_fail_bool(check, sparse.contains(zero_index));
  pass_or_fail(check, sparse(zero_index), 1.0e-14);

  sparse.erase(zero_index);
  pass_or_fail_bool(check, sparse.number_nonzeros() == 2);
  pass_or_fail_bool(check, !sparse.contains(zero_index));
  pass_or_fail(check, sparse(zero_index), 1.0e-14);
  pass_or_fail(check, ::boba::norm_l1(sparse) - 7.75, 1.0e-14);
  validate_host_index_lists(sparse, check);
  for (std::size_t d = 0; d < dimension; ++d)
  {
    pass_or_fail_bool(check, sparse.index_list(d).const_view()(0) == middle_index[d]);
    pass_or_fail_bool(check, sparse.index_list(d).const_view()(1) == last_index[d]);
  }

  ::boba::SparseTensor<dimension, space, double> sparse_space(sparse);
  validate_view(sparse_space, sparse, check);
}

template <std::size_t dimension>
void test_convert_to_sparse_tensor(bool& check)
{
  const auto sizes = make_sizes<dimension>();
  ::boba::Tensor<dimension, ::boba::host_space, double> dense(sizes);
  auto dense_view = dense.view();
  for (::boba::index_t linear_index = 0; linear_index < dense.size(); ++linear_index)
  {
    dense_view(linear_index) = static_cast<double>(linear_index + 1);
  }

  auto value_filtered = ::boba::convert_to_SparseTensor(dense, [](double value, auto)
  {
    const auto integer_value = static_cast<::boba::index_t>(value);
    return integer_value % 2 == 0;
  });
  auto dense_const_view = dense.const_view();

  ::boba::index_t expected_value_filtered_nonzeros = 0;
  double value_filter_error = 0.0;
  for (::boba::index_t linear_index = 0; linear_index < dense.size(); ++linear_index)
  {
    const bool should_store = (linear_index + 1) % 2 == 0;
    expected_value_filtered_nonzeros += should_store ? 1 : 0;
    const auto indices = dense.multiindex(linear_index);
    const double expected_value = should_store ? dense_const_view(linear_index) : 0.0;
    value_filter_error = std::max(value_filter_error, ::boba::abs(value_filtered(indices) - expected_value));
  }
  pass_or_fail_bool(check, value_filtered.number_nonzeros() == expected_value_filtered_nonzeros);
  pass_or_fail(check, value_filter_error, 1.0e-14);
  validate_host_index_lists(value_filtered, check);
  validate_view(value_filtered, value_filtered, check);

  dense.view()(0) = 0.0;
  auto index_filtered = ::boba::convert_to_SparseTensor(dense, [](double, auto indices)
  {
    return indices[0] == 0;
  });

  ::boba::index_t expected_index_filtered_nonzeros = 0;
  double index_filter_error = 0.0;
  for (::boba::index_t linear_index = 0; linear_index < dense.size(); ++linear_index)
  {
    const auto indices = dense.multiindex(linear_index);
    const bool should_store = indices[0] == 0;
    expected_index_filtered_nonzeros += should_store ? 1 : 0;
    const double expected_value = should_store ? dense_const_view(linear_index) : 0.0;
    index_filter_error = std::max(index_filter_error, ::boba::abs(index_filtered(indices) - expected_value));
  }
  pass_or_fail_bool(check, index_filtered.number_nonzeros() == expected_index_filtered_nonzeros);
  pass_or_fail(check, index_filtered(::boba::filled_array<dimension>(static_cast<::boba::index_t>(0))), 1.0e-14);
  pass_or_fail(check, index_filter_error, 1.0e-14);
  validate_host_index_lists(index_filtered, check);
  for (::boba::index_t sparse_index = 0; sparse_index < index_filtered.number_nonzeros(); ++sparse_index)
  {
    pass_or_fail_bool(check, index_filtered.index_list(0).const_view()(sparse_index) == 0);
  }

  ::boba::Tensor<dimension, space, double> dense_space(dense);
  auto sparse_space = ::boba::convert_to_SparseTensor(dense_space, [](double value, auto)
  {
    return value > 0.0 && static_cast<::boba::index_t>(value) % 3 == 0;
  });
  auto expected_host = ::boba::convert_to_SparseTensor(dense, [](double value, auto)
  {
    return value > 0.0 && static_cast<::boba::index_t>(value) % 3 == 0;
  });

  pass_or_fail_bool(check, sparse_space.number_nonzeros() == expected_host.number_nonzeros());
  validate_view(sparse_space, expected_host, check);
}

template <std::size_t dimension>
void test_sparse_tensor_dimension(bool& check)
{
  std::cout << "Testing SparseTensor dimension " << dimension << std::endl;
  test_manual_sparse_tensor<dimension>(check);
  test_convert_to_sparse_tensor<dimension>(check);
}

int main()
{
  boba::init();

  bool check = true;

  test_sparse_tensor_dimension<1>(check);
  test_sparse_tensor_dimension<2>(check);
  test_sparse_tensor_dimension<3>(check);
  test_sparse_tensor_dimension<4>(check);

  const int result = final_check(check);
  boba::finalize();
  return result;
}

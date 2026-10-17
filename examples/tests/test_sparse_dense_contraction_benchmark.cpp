// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <utility>

namespace
{

using data_t = double;

template <std::size_t dimension>
::boba::Tensor<dimension, ::boba::host_space, data_t> make_mixed_test_tensor(
  ::boba::Array<::boba::index_t, dimension> sizes,
  ::boba::index_t seed)
{
  ::boba::Tensor<dimension, ::boba::host_space, data_t> tensor(sizes);
  auto view = tensor.view();
  for (::boba::index_t linear_index = 0; linear_index < tensor.size(); ++linear_index)
  {
    const bool is_nonzero = ((linear_index + seed) % 3 == 0) || ((linear_index + seed) % 7 == 0);
    const data_t sign = (linear_index % 2 == 0) ? 1.0 : -1.0;
    view(linear_index) = is_nonzero ? sign * static_cast<data_t>((linear_index % 11) + 1) : 0.0;
  }
  return tensor;
}

template <std::size_t dimension>
::boba::Tensor<dimension, ::boba::host_space, data_t> make_dense_test_tensor(
  ::boba::Array<::boba::index_t, dimension> sizes,
  ::boba::index_t seed)
{
  ::boba::Tensor<dimension, ::boba::host_space, data_t> tensor(sizes);
  auto view = tensor.view();
  for (::boba::index_t linear_index = 0; linear_index < tensor.size(); ++linear_index)
  {
    const data_t sign = ((linear_index + seed) % 2 == 0) ? 1.0 : -1.0;
    view(linear_index) = sign * static_cast<data_t>((linear_index % 17) + 1);
  }
  return tensor;
}

template <std::size_t dimension>
::boba::Tensor<dimension, ::boba::host_space, data_t> make_sparsified_test_tensor(
  ::boba::Array<::boba::index_t, dimension> sizes,
  ::boba::index_t seed,
  ::boba::index_t period)
{
  ::boba::Tensor<dimension, ::boba::host_space, data_t> tensor(sizes);
  auto view = tensor.view();
  for (::boba::index_t linear_index = 0; linear_index < tensor.size(); ++linear_index)
  {
    const bool is_nonzero = (linear_index + seed) % period == 0;
    const data_t sign = ((linear_index + seed) % 2 == 0) ? 1.0 : -1.0;
    view(linear_index) = is_nonzero ? sign * static_cast<data_t>((linear_index % 17) + 1) : 0.0;
  }
  return tensor;
}

template <typename output_t>
struct benchmark_result
{
  output_t output;
  double milliseconds = 0.0;
};

template <typename contraction_t>
auto benchmark(contraction_t contraction, std::size_t repetitions)
{
  using output_t = std::remove_cvref_t<decltype(contraction())>;
  output_t output;
  volatile data_t checksum = 0.0;

  const auto begin = std::chrono::steady_clock::now();
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition)
  {
    output = contraction();
    const auto output_view = output.const_view();
    checksum = checksum + output_view(0) + output_view(output.size() / 2) + output_view(output.size() - 1);
  }
  const auto end = std::chrono::steady_clock::now();

  std::cout << "  checksum: " << checksum << std::endl;
  return benchmark_result<output_t>{
    std::move(output),
    std::chrono::duration<double, std::milli>(end - begin).count() / static_cast<double>(repetitions)};
}

void test_small_mixed_contractions(bool& check)
{
  std::cout << "Testing small sparse/dense contractions" << std::endl;

  const ::boba::Array<::boba::index_t, 2> sizes_A{2, 3};
  const ::boba::Array<::boba::index_t, 2> sizes_B{3, 4};
  const ::boba::Array<size_t, 1> contraction_dimensions_A{1};
  const ::boba::Array<size_t, 1> contraction_dimensions_B{0};
  auto dense_A = make_mixed_test_tensor(sizes_A, 0);
  auto dense_B = make_mixed_test_tensor(sizes_B, 4);
  auto sparse_A = ::boba::convert_to_SparseTensor(dense_A, [](data_t value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  auto sparse_B = ::boba::convert_to_SparseTensor(dense_B, [](data_t value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });

  const auto dense_dense = ::boba::tensor_contraction<1, ::boba::host_space, 2, 2, data_t>(
    dense_A, dense_B, contraction_dimensions_A, contraction_dimensions_B);
  const auto sparse_dense = ::boba::tensor_contraction<1, ::boba::host_space, 2, 2, data_t, true>(
    sparse_A, dense_B, contraction_dimensions_A, contraction_dimensions_B);
  const auto dense_sparse = ::boba::tensor_contraction<1, ::boba::host_space, 2, 2, data_t, true>(
    dense_A, sparse_B, contraction_dimensions_A, contraction_dimensions_B);

  pass_or_fail(check, ::boba::norm_difference_inf(sparse_dense, dense_dense), 1.0e-14);
  pass_or_fail(check, ::boba::norm_difference_inf(dense_sparse, dense_dense), 1.0e-14);

  const auto labeled_sparse_dense = ::boba::tensor_contraction<1, ::boba::host_space, 2, 2, data_t, true>(
    ::boba::Array<std::string, 2>{"i", "j"},
    sparse_A,
    ::boba::Array<std::string, 2>{"j", "k"},
    dense_B,
    ::boba::Array<std::string, 2>{"i", "k"});
  const auto labeled_dense_sparse = ::boba::tensor_contraction<1, ::boba::host_space, 2, 2, data_t, true>(
    ::boba::Array<std::string, 2>{"i", "j"},
    dense_A,
    ::boba::Array<std::string, 2>{"j", "k"},
    sparse_B,
    ::boba::Array<std::string, 2>{"i", "k"});
  pass_or_fail(check, ::boba::norm_difference_inf(labeled_sparse_dense, dense_dense), 1.0e-14);
  pass_or_fail(check, ::boba::norm_difference_inf(labeled_dense_sparse, dense_dense), 1.0e-14);
}

void benchmark_large_mixed_contractions()
{
  std::cout << "Benchmarking large mixed contractions across sparsities" << std::endl;

  const ::boba::Array<::boba::index_t, 4> sizes_A{24, 24, 24, 24};
  const ::boba::Array<::boba::index_t, 3> sizes_B{24, 24, 24};
  const ::boba::Array<size_t, 1> contraction_dimensions_A{3};
  const ::boba::Array<size_t, 1> contraction_dimensions_B{0};
  const auto dense_A = make_dense_test_tensor(sizes_A, 0);
  const auto dense_B = make_dense_test_tensor(sizes_B, 1);

  const auto dense_dense = [&]()
  {
    return ::boba::tensor_contraction<1, ::boba::host_space, 4, 3, data_t>(
      dense_A, dense_B, contraction_dimensions_A, contraction_dimensions_B);
  };

  constexpr std::size_t repetitions = 5;
  const auto dense_dense_result = benchmark(dense_dense, repetitions);
  constexpr ::boba::index_t sparsity_periods[] = {10, 100, 1000, 10000};

  for (const auto sparsity_period : sparsity_periods)
  {
    const auto sparse_source_A = make_sparsified_test_tensor(sizes_A, 0, sparsity_period);
    const auto sparse_source_B = make_sparsified_test_tensor(sizes_B, 1, sparsity_period);
    const auto sparse_A = ::boba::convert_to_SparseTensor(sparse_source_A, [](data_t value, auto)
    {
      return ::boba::abs(value) > 0.0;
    });
    const auto sparse_B = ::boba::convert_to_SparseTensor(sparse_source_B, [](data_t value, auto)
    {
      return ::boba::abs(value) > 0.0;
    });

    const auto sparse_dense = [&]()
    {
      return ::boba::tensor_contraction<1, ::boba::host_space, 4, 3, data_t, true>(
        sparse_A, dense_B, contraction_dimensions_A, contraction_dimensions_B);
    };
    const auto dense_sparse = [&]()
    {
      return ::boba::tensor_contraction<1, ::boba::host_space, 4, 3, data_t, true>(
        dense_A, sparse_B, contraction_dimensions_A, contraction_dimensions_B);
    };

    const auto sparse_dense_result = benchmark(sparse_dense, repetitions);
    const auto dense_sparse_result = benchmark(dense_sparse, repetitions);
    const double sparsity = 1.0 / static_cast<double>(sparsity_period);
    std::cout << std::setprecision(10)
              << "MIXED_TIMING sparsity=" << sparsity
              << " dense_dense_ms=" << dense_dense_result.milliseconds
              << " dense_sparse_ms=" << dense_sparse_result.milliseconds
              << " sparse_dense_ms=" << sparse_dense_result.milliseconds
              << " sparse_A_nnz=" << sparse_A.number_nonzeros()
              << " sparse_B_nnz=" << sparse_B.number_nonzeros() << std::endl;
  }
}

} // namespace

int main()
{
  ::boba::init();

  bool check = true;
#ifdef BOBA_CI
  test_small_mixed_contractions(check);
#else
  benchmark_large_mixed_contractions();
#endif

  const int result = final_check(check);
  ::boba::finalize();
  return result;
}

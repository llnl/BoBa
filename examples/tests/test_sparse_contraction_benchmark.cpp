// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

namespace
{

using data_t = double;
constexpr ::boba::execution_space device_space = ::boba::default_execution_space;

/**
 * @brief Builds a host tensor with signed nonzeros where `(index + seed)` is a multiple of 3 or 7, zeros elsewhere.
 */
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

/**
 * @brief Builds a fully dense host tensor with deterministic signed values in `[-17, 17]`.
 */
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

/**
 * @brief Builds a host tensor with one nonzero every \p period entries, offset by \p seed.
 */
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

/**
 * @brief Last output of a timed contraction and its mean time per repetition.
 */
template <typename output_t>
struct benchmark_result
{
  output_t output;
  double milliseconds = 0.0;
};

/**
 * @brief Runs \p contraction \p repetitions times and reports the mean wall time in milliseconds.
 *
 * Each output feeds a printed checksum so the compiler cannot skip repetitions.
 * Device outputs contribute their size because entries are not host-readable.
 */
template <typename contraction_t>
auto benchmark(contraction_t contraction, std::size_t repetitions)
{
  using output_t = std::remove_cvref_t<decltype(contraction())>;
  output_t output;
  volatile data_t checksum = 0.0;

  using timer_t = ::boba::TicToc<::boba::tictoc_units::microseconds>;
  timer_t timer;
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition)
  {
    output = contraction();
    if constexpr (output_t::get_space() == ::boba::host_space)
    {
      const auto output_view = output.const_view();
      checksum = checksum + output_view(0) + output_view(output.size() / 2) + output_view(output.size() - 1);
    }
    else
    {
      // Device entries are not readable here; the size still keeps the result observable.
      checksum = checksum + static_cast<data_t>(output.size());
    }
  }
  const auto elapsed_microseconds = static_cast<double>(timer.timing());

  std::cout << "  checksum: " << checksum << std::endl;
  return benchmark_result<output_t>{
    std::move(output),
    timer_t::convert<::boba::tictoc_units::microseconds, ::boba::tictoc_units::milliseconds>(elapsed_microseconds) / static_cast<double>(repetitions)};
}

/**
 * @brief Checks small 2 x 2 naive sparse x dense and dense x sparse contractions, positional and labeled, against dense.
 */
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

/**
 * @brief Times naive sparse x dense and dense x sparse against dense x dense on 24^4 x 24^3 tensors at several sparsities.
 *
 * Enabled with `--sweep`; prints `MIXED_TIMING` lines and performs no checks.
 */
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

/**
 * @brief Builds a sparse tensor with about `density * size` entries at uniformly random positions.
 *
 * A fixed seed keeps the structure reproducible. Duplicate keys are coalesced by
 * the SparseTensor constructor, so the result can hold slightly fewer entries.
 */
template <std::size_t dimension>
::boba::SparseTensor<dimension, ::boba::host_space, data_t> make_random_sparse_tensor(
  ::boba::Array<::boba::index_t, dimension> sizes,
  double density,
  unsigned seed)
{
  const ::boba::index_t size = ::boba::product(sizes);
  const auto target_nnz = static_cast<::boba::index_t>(std::ceil(density * static_cast<double>(size)));
  ::boba::Vector<::boba::host_space, ::boba::index_t> keys({target_nnz});
  ::boba::Vector<::boba::host_space, data_t> values({target_nnz});
  auto keys_view = keys.view();
  auto values_view = values.view();
  std::mt19937 generator(seed);
  std::uniform_int_distribution<::boba::index_t> key_distribution(0, size - 1);
  std::uniform_real_distribution<data_t> value_distribution(-1.0, 1.0);
  for (::boba::index_t entry = 0; entry < target_nnz; ++entry)
  {
    keys_view(entry) = key_distribution(generator);
    values_view(entry) = value_distribution(generator);
  }
  return ::boba::SparseTensor<dimension, ::boba::host_space, data_t>(sizes, keys, values);
}

/**
 * @brief Checks that a random sparse tensor stays near the requested density and below dense storage.
 *
 * Duplicate random keys coalesce, so the lower bound on nnz is loose.
 */
template <std::size_t dimension>
void check_sparse_storage(::boba::SparseTensor<dimension, ::boba::host_space, data_t> const& sparse,
                          double density,
                          bool& check)
{
  const auto target_nnz = static_cast<::boba::index_t>(std::ceil(density * static_cast<double>(sparse.size())));
  const auto sparse_bytes = static_cast<std::size_t>(sparse.nnz()) * (sizeof(::boba::index_t) + sizeof(data_t));
  const auto dense_bytes = static_cast<std::size_t>(sparse.size()) * sizeof(data_t);
  pass_or_fail_bool(check, sparse.nnz() <= target_nnz);
  pass_or_fail_bool(check, 2 * sparse.nnz() >= target_nnz);
  pass_or_fail_bool(check, density >= 0.5 or sparse_bytes < dense_bytes);
}

/**
 * @brief Times unfolded SpMM, naive, and dense contractions of a random sparse tensor with a dense tensor.
 *
 * The sparse result is checked against the dense result, and the sparse storage is checked.
 * When the default execution space is not the host, the unfolded path is also
 * timed there with inputs copied beforehand, and checked against the host result.
 */
template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
void benchmark_random_sparse_dense_contraction(::boba::Array<::boba::index_t, dimension_A> sizes_A,
                                  ::boba::Array<::boba::index_t, dimension_B> sizes_B,
                                  ::boba::Array<size_t, contractions> contraction_dimensions_A,
                                  ::boba::Array<size_t, contractions> contraction_dimensions_B,
                                  double density,
                                  std::size_t repetitions,
                                  bool& check)
{
  std::cout << "Benchmarking random sparse x dense, contractions " << contractions << ", density " << density << std::endl;
  const auto sparse_A = make_random_sparse_tensor(sizes_A, density, 2024u);
  check_sparse_storage(sparse_A, density, check);
  const auto dense_A = ::boba::to_dense(sparse_A);
  const auto dense_B = make_dense_test_tensor(sizes_B, 1);

  const auto dense_dense_result = benchmark([&]()
  {
    return ::boba::tensor_contraction<contractions, ::boba::host_space, dimension_A, dimension_B, data_t>(
      dense_A, dense_B, contraction_dimensions_A, contraction_dimensions_B);
  }, repetitions);
  const auto unfolded_result = benchmark([&]()
  {
    return ::boba::tensor_contraction<contractions, ::boba::host_space, dimension_A, dimension_B, data_t>(
      sparse_A, dense_B, contraction_dimensions_A, contraction_dimensions_B);
  }, repetitions);
  const auto naive_result = benchmark([&]()
  {
    return ::boba::tensor_contraction<contractions, ::boba::host_space, dimension_A, dimension_B, data_t, true>(
      sparse_A, dense_B, contraction_dimensions_A, contraction_dimensions_B);
  }, repetitions);

  const auto scale = std::max(1.0, ::boba::norm_inf(dense_dense_result.output));
  pass_or_fail(check, ::boba::norm_difference_inf(unfolded_result.output, dense_dense_result.output) / scale, 1.0e-12);
  pass_or_fail(check, ::boba::norm_difference_inf(naive_result.output, dense_dense_result.output) / scale, 1.0e-12);

  std::cout << std::setprecision(10)
            << "RANDOM_TIMING contractions=" << contractions
            << " density=" << density
            << " nnz=" << sparse_A.nnz()
            << " dense_dense_ms=" << dense_dense_result.milliseconds
            << " unfolded_ms=" << unfolded_result.milliseconds
            << " naive_ms=" << naive_result.milliseconds << std::endl;

  if constexpr (device_space != ::boba::host_space)
  {
    const ::boba::SparseTensor<dimension_A, device_space, data_t> sparse_A_device(sparse_A);
    const ::boba::Tensor<dimension_B, device_space, data_t> dense_B_device(dense_B);
    const auto device_result = benchmark([&]()
    {
      return ::boba::tensor_contraction<contractions, device_space, dimension_A, dimension_B, data_t>(
        sparse_A_device, dense_B_device, contraction_dimensions_A, contraction_dimensions_B);
    }, repetitions);
    const ::boba::Tensor<dimension_A + dimension_B - 2 * contractions, ::boba::host_space, data_t> device_output(device_result.output);
    pass_or_fail(check, ::boba::norm_difference_inf(device_output, dense_dense_result.output) / scale, 1.0e-12);

    std::cout << "DEVICE_TIMING contractions=" << contractions
              << " density=" << density
              << " space=" << ::boba::detail::execution_space_name(device_space)
              << " unfolded_ms=" << device_result.milliseconds << std::endl;
  }
}

/**
 * @brief Times SpGEMM and naive sparse x sparse against dense x dense contraction of two random sparse tensors.
 *
 * The densified sparse result is checked against the dense result, and the
 * result storage must not exceed the dense output size. When the default
 * execution space is not the host, SpGEMM is also timed there and checked.
 */
template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
void benchmark_random_sparse_sparse_contraction(::boba::Array<::boba::index_t, dimension_A> sizes_A,
                                                ::boba::Array<::boba::index_t, dimension_B> sizes_B,
                                                ::boba::Array<size_t, contractions> contraction_dimensions_A,
                                                ::boba::Array<size_t, contractions> contraction_dimensions_B,
                                                double density,
                                                std::size_t repetitions,
                                                bool& check)
{
  std::cout << "Benchmarking random sparse x sparse, contractions " << contractions << ", density " << density << std::endl;
  const auto sparse_A = make_random_sparse_tensor(sizes_A, density, 2024u);
  const auto sparse_B = make_random_sparse_tensor(sizes_B, density, 2025u);
  check_sparse_storage(sparse_A, density, check);
  check_sparse_storage(sparse_B, density, check);
  const auto dense_A = ::boba::to_dense(sparse_A);
  const auto dense_B = ::boba::to_dense(sparse_B);

  const auto dense_dense_result = benchmark([&]()
  {
    return ::boba::tensor_contraction<contractions, ::boba::host_space, dimension_A, dimension_B, data_t>(
      dense_A, dense_B, contraction_dimensions_A, contraction_dimensions_B);
  }, repetitions);
  const auto sparse_sparse_result = benchmark([&]()
  {
    return ::boba::tensor_contraction<contractions, ::boba::host_space, dimension_A, dimension_B, data_t>(
      sparse_A, sparse_B, contraction_dimensions_A, contraction_dimensions_B);
  }, repetitions);
  const auto naive_result = benchmark([&]()
  {
    return ::boba::tensor_contraction<contractions, ::boba::host_space, dimension_A, dimension_B, data_t, true>(
      sparse_A, sparse_B, contraction_dimensions_A, contraction_dimensions_B);
  }, repetitions);

  const auto& sparse_output = sparse_sparse_result.output;
  pass_or_fail_bool(check, sparse_output.nnz() <= sparse_output.size());
  const auto scale = std::max(1.0, ::boba::norm_inf(dense_dense_result.output));
  pass_or_fail(check, ::boba::norm_difference_inf(::boba::to_dense(sparse_output), dense_dense_result.output) / scale, 1.0e-12);
  pass_or_fail(check, ::boba::norm_difference_inf(::boba::to_dense(naive_result.output), dense_dense_result.output) / scale, 1.0e-12);

  std::cout << std::setprecision(10)
            << "RANDOM_SPARSE_TIMING contractions=" << contractions
            << " density=" << density
            << " nnz_A=" << sparse_A.nnz()
            << " nnz_B=" << sparse_B.nnz()
            << " nnz_C=" << sparse_output.nnz()
            << " dense_dense_ms=" << dense_dense_result.milliseconds
            << " sparse_sparse_ms=" << sparse_sparse_result.milliseconds
            << " naive_ms=" << naive_result.milliseconds << std::endl;

  if constexpr (device_space != ::boba::host_space)
  {
    const ::boba::SparseTensor<dimension_A, device_space, data_t> sparse_A_device(sparse_A);
    const ::boba::SparseTensor<dimension_B, device_space, data_t> sparse_B_device(sparse_B);
    const auto device_result = benchmark([&]()
    {
      return ::boba::tensor_contraction<contractions, device_space, dimension_A, dimension_B, data_t>(
        sparse_A_device, sparse_B_device, contraction_dimensions_A, contraction_dimensions_B);
    }, repetitions);
    const ::boba::SparseTensor<dimension_A + dimension_B - 2 * contractions, ::boba::host_space, data_t> device_output(device_result.output);
    pass_or_fail(check, ::boba::norm_difference_inf(::boba::to_dense(device_output), dense_dense_result.output) / scale, 1.0e-12);

    std::cout << "DEVICE_SPARSE_TIMING contractions=" << contractions
              << " density=" << density
              << " space=" << ::boba::detail::execution_space_name(device_space)
              << " sparse_sparse_ms=" << device_result.milliseconds << std::endl;
  }
}

/**
 * @brief Runs one- and two-index random sparse x dense and sparse x sparse contractions of n^4 x n^3 tensors.
 */
void benchmark_random_contractions(::boba::index_t n, double density, std::size_t repetitions, bool& check)
{
  benchmark_random_sparse_dense_contraction<1, 4, 3>({n, n, n, n}, {n, n, n}, {3}, {0}, density, repetitions, check);
  benchmark_random_sparse_dense_contraction<2, 4, 3>({n, n, n, n}, {n, n, n}, {1, 3}, {2, 0}, density, repetitions, check);
  benchmark_random_sparse_sparse_contraction<1, 4, 3>({n, n, n, n}, {n, n, n}, {3}, {0}, density, repetitions, check);
  benchmark_random_sparse_sparse_contraction<2, 4, 3>({n, n, n, n}, {n, n, n}, {1, 3}, {2, 0}, density, repetitions, check);
}

} // namespace

/**
 * \brief
 * Driver for sparse x dense and sparse x sparse contraction tests and benchmarks.
 */

int main(int argc, char* argv[])
{
  boba::splash();
  boba::init();

  double density = 0.01;
  size_t mode_size = 8;
  size_t repetitions = 1;
  bool sweep = false;
  bool check = true;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(density,
                             "-d",
                             "--density",
                             "Fraction of stored entries in the random sparse tensors.");

  args.add_optional_argument(mode_size,
                             "-n",
                             "--mode_size",
                             "Extent of every mode of the random tensors.");

  args.add_optional_argument(repetitions,
                             "-r",
                             "--repetitions",
                             "Number of timed repetitions per contraction.");

  args.add_optional_argument(sweep,
                             "-s",
                             "--sweep",
                             "Also run the large periodic sparsity sweep.");

  args.parse_check();

  test_small_mixed_contractions(check);
  benchmark_random_contractions(static_cast<::boba::index_t>(mode_size), density, repetitions, check);
  if (sweep)
  {
    benchmark_large_mixed_contractions();
  }

  boba::finalize();
  return final_check(check);
}

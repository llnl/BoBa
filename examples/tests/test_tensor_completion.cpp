// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file
 * \brief Demonstrates CP-APR tensor completion on a histogram of sampled points.
 *
 * Let `P` be the number of samples, `D` the tensor order, `R` the CP rank,
 * `K` the number of outer sweeps, `L` the maximum inner updates per mode,
 * `N_nz` the number of occupied bins, and `G` the total number of bins.
 * Binning the points costs `O(P * D)`; the dense staging histogram used by this
 * miniapp also costs `O(G)` time and memory.
 *
 * The dominant sparse CP-APR work is `O(K * L * N_nz * R * D^2)`, while the
 * dense path costs `O(K * L * G * R * D^2)`. Thus, for fixed rank, order, and
 * iteration limits, sparse fitting scales linearly with occupied bins and at
 * worst linearly with samples because `N_nz <= P`. Repeated samples in an
 * occupied bin increase its count without increasing the sparse fitting cost.
 */

#include "BOBA/boba.hpp"
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr boba::execution_space space = boba::execution_space::CPU;

template <std::size_t dimension>
using tensor_shape = boba::Array<boba::index_t, dimension>;

template <std::size_t dimension>
using dense_tensor = boba::Tensor<dimension, space, double>;

template <std::size_t dimension>
using sparse_tensor = boba::SparseTensor<dimension, space, double>;

template <std::size_t dimension>
using CPAPRModel = boba::CanonicalPolyadicDecomposition<dimension, space, double>;

struct GaussianHistogramConfig
{
  std::vector<int> dims;
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
  unsigned seed = 12345;
};

constexpr double kEstimateTolerance = 0.5;

struct CommandLineOptions
{
  std::size_t sample_count = 2500;
  int rank = 5;
  double bin_width = 1.0;
  std::vector<int> dims = {20, 20, 20, 20, 20, 20};
  bool dense_apr = false;
  bool verbose_bins = false;
};

constexpr auto clamp_into_range = [](double value, double low, double high)
{
  if (value < low)
  {
    return low;
  }

  const double high_open = std::nextafter(high, low);
  if (value >= high)
  {
    return high_open;
  }
  return value;
};

const auto format_int_values = [](const std::vector<int>& values)
{
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    if (i > 0)
    {
      out << ", ";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
};

const auto print_usage = [](const char* program)
{
  boba_print(
    std::string("Usage: ") + program +
    " [--samples N] [--rank R] [--bin-width W] [--dims D1 D2 ... Dk] [--dense-apr] [--verbose-bins]");
  boba_print("Defaults: --samples 2500 --rank 5 --bin-width 1.0 --dims 20 20 20 20 20 20");
  boba_print("Supported tensor order: 1 through 8");
};

CommandLineOptions parse_args(int argc, char** argv)
{
  CommandLineOptions options;
  ::boba::argparser args(argc, argv);
  args.add_optional_argument(options.sample_count, "", "--samples", "Number of synthetic samples.");
  args.add_optional_argument(options.rank, "", "--rank", "CP decomposition rank.");
  args.add_optional_argument(options.bin_width, "", "--bin-width", "Histogram bin width.");
  args.add_optional_argument(options.dims, "", "--dims", "Tensor extents per mode.");
  args.add_optional_argument(options.dense_apr, "", "--dense-apr", "Use dense APR input path.");
  args.add_optional_argument(options.verbose_bins, "", "--verbose-bins", "Print bin details.");

  args.parse();
  if (options.sample_count == 0)
  {
    throw std::invalid_argument("invalid value for --samples: 0");
  }
  if (options.rank <= 0)
  {
    throw std::invalid_argument("invalid value for --rank: " + std::to_string(options.rank));
  }
  if (options.bin_width <= 0.0)
  {
    throw std::invalid_argument("invalid value for --bin-width: " + std::to_string(options.bin_width));
  }
  if (options.dims.empty())
  {
    throw std::invalid_argument("invalid value for --dims: []");
  }
  for (int dim : options.dims)
  {
    if (dim <= 0)
    {
      throw std::invalid_argument("invalid value for --dims: " + format_int_values(options.dims));
    }
  }
  return options;
}

template <typename index_t, std::size_t dimension>
std::string format_index(const boba::Array<index_t, dimension>& idx)
{
  std::ostringstream out;
  out << "(";
  for (std::size_t mode = 0; mode < dimension; ++mode)
  {
    if (mode > 0)
    {
      out << ", ";
    }
    out << idx[mode];
  }
  out << ")";
  return out.str();
}

std::string format_values(const std::vector<double>& values)
{
  std::ostringstream out;
  out << "[";
  for (std::size_t mode = 0; mode < values.size(); ++mode)
  {
    if (mode > 0)
    {
      out << ", ";
    }
    out << values[mode];
  }
  out << "]";
  return out.str();
}

template <typename index_t, std::size_t dimension>
std::string format_dims(const boba::Array<index_t, dimension>& dims)
{
  std::ostringstream out;
  out << "[";
  for (std::size_t mode = 0; mode < dimension; ++mode)
  {
    if (mode > 0)
    {
      out << ", ";
    }
    out << dims[mode];
  }
  out << "]";
  return out.str();
}

template <std::size_t dimension>
tensor_shape<dimension> to_tensor_shape(const std::vector<int>& dims)
{
  if (dims.size() != dimension)
  {
    throw std::invalid_argument("tensor order does not match CP model dimension");
  }

  auto sizes = boba::filled_array<dimension>(boba::index_t(0));
  for (std::size_t mode = 0; mode < dimension; ++mode)
  {
    sizes[mode] = static_cast<boba::index_t>(dims[mode]);
  }
  return sizes;
}

template <std::size_t dimension>
void print_lambda(const CPAPRModel<dimension>& model)
{
  std::ostringstream out;
  out << "Learned lambda:";
  auto weights_view = model.weights().const_view();
  for (boba::index_t r = 0; r < model.weights().size(); ++r)
  {
    out << ' ' << std::fixed << std::setprecision(4) << weights_view(r);
  }
  boba_print(out.str());
}

template <std::size_t dimension, typename index_t>
double raw_tensor_entry_estimate(const CPAPRModel<dimension>& model, const boba::Array<index_t, dimension>& idx)
{
  const auto weights_view = model.weights().const_view();
  const auto cores = model.get_core_const_views();

  double estimate = 0.0;
  for (boba::index_t r = 0; r < model.weights().size(); ++r)
  {
    double component = weights_view(r);
    for (std::size_t mode = 0; mode < dimension; ++mode)
    {
      component *= cores[mode]({static_cast<boba::index_t>(idx[mode]), r});
    }
    estimate += component;
  }
  return estimate;
}

template <std::size_t dimension, typename index_t>
double tensor_entry_estimate(const CPAPRModel<dimension>& model, const boba::Array<index_t, dimension>& idx)
{
  const double estimate = raw_tensor_entry_estimate(model, idx);
  return (estimate < kEstimateTolerance) ? 0.0 : estimate;
}

template <std::size_t dimension>
void print_observed_counts(const sparse_tensor<dimension>& tensor, const CPAPRModel<dimension>& model)
{
  std::cout << "\nNon-zero count bin data (CP-APR)\n";
  auto tensor_view = tensor.const_view();
  auto values_view = tensor.values_tensor().const_view();
  const auto nnz = tensor.number_nonzeros();
  for (boba::index_t entry = 0; entry < nnz; ++entry)
  {
    const auto idx = tensor_view.entry_multiindex(entry);
    const long long original_count = static_cast<long long>(values_view(entry));
    std::ostringstream out;
    out << format_index(idx) << " "
        << std::fixed << std::setprecision(3)
        << tensor_entry_estimate(model, idx)
        << "(" << original_count << ")";
    boba_print(out.str());
  }
}

template <std::size_t dimension>
void print_empty_estimated_counts(const sparse_tensor<dimension>& tensor, const CPAPRModel<dimension>& model)
{
  const auto bins = static_cast<std::size_t>(tensor.size());
  auto tensor_view = tensor.const_view();
  std::vector<bool> observed_bins(bins, false);

  const auto nnz = tensor.number_nonzeros();
  for (boba::index_t entry = 0; entry < nnz; ++entry)
  {
    const auto idx = tensor_view.entry_multiindex(entry);
    observed_bins[static_cast<std::size_t>(tensor_view.index(idx))] = true;
  }

  boba_print("");
  boba_print("Originally empty bins with non-zero Poisson estimate (CP-APR)");

  bool found = false;
  for (std::size_t flat = 0; flat < bins; ++flat)
  {
    if (!observed_bins[flat])
    {
      const auto idx = tensor.multiindex(static_cast<boba::index_t>(flat));
      const double estimate = tensor_entry_estimate(model, idx);
      if (estimate > 0.0)
      {
        found = true;
        std::ostringstream out;
        out << format_index(idx) << " "
            << std::defaultfloat << std::setprecision(6)
            << estimate << "(0)";
        boba_print(out.str());
      }
    }
  }

  if (!found)
  {
    boba_print("none");
  }
}

template <std::size_t dimension>
bool model_is_valid_probability_model(const CPAPRModel<dimension>& model)
{
  auto weights_view = model.weights().const_view();
  for (boba::index_t r = 0; r < model.weights().size(); ++r)
  {
    if (!std::isfinite(weights_view(r)) || weights_view(r) < 0.0)
    {
      return false;
    }
  }

  for (std::size_t mode = 0; mode < dimension; ++mode)
  {
    auto factor_view = model.m_cores[mode].const_view();
    for (boba::index_t r = 0; r < model.m_cores[mode].cols(); ++r)
    {
      double column_sum = 0.0;
      for (boba::index_t i = 0; i < model.m_cores[mode].rows(); ++i)
      {
        const double value = factor_view({i, r});
        if (!std::isfinite(value) || value < 0.0)
        {
          return false;
        }
        column_sum += value;
      }
      if (std::abs(column_sum - 1.0) > 1.0e-10)
      {
        return false;
      }
    }
  }

  return true;
}

} // namespace

template <std::size_t dimension>
sparse_tensor<dimension> generate_gaussian_binned_tensor(
  std::size_t sample_count,
  const GaussianHistogramConfig& config)
{
  if (config.dims.size() != dimension)
  {
    throw std::invalid_argument("histogram dimensions must match tensor order");
  }
  if (config.lower_bounds.size() != config.dims.size() ||
      config.upper_bounds.size() != config.dims.size())
  {
    throw std::invalid_argument("histogram bounds must match tensor order");
  }

  const auto dims = to_tensor_shape<dimension>(config.dims);
  const auto indexer = boba::Multiindexer<dimension>(dims);
  const std::size_t order = dimension;
  for (std::size_t mode = 0; mode < dimension; ++mode)
  {
    const auto dim = dims[mode];
    if (dim <= 0)
    {
      throw std::invalid_argument("all tensor dimensions must be positive");
    }
  }

  std::vector<double> dense_counts(static_cast<std::size_t>(indexer.size()), 0.0);
  std::mt19937 generator(config.seed);

  std::vector<double> means(order, 0.0);
  for (std::size_t mode = 0; mode < order; ++mode)
  {
    const double low = config.lower_bounds[mode];
    const double high = config.upper_bounds[mode];
    if (high <= low)
    {
      throw std::invalid_argument("upper bound must exceed lower bound");
    }
    means[mode] = 0.5 * (low + high);
  }

  for (std::size_t sample = 0; sample < sample_count; ++sample)
  {
    auto idx = boba::filled_array<dimension>(boba::index_t(0));
    for (std::size_t mode = 0; mode < order; ++mode)
    {
      const double low = config.lower_bounds[mode];
      const double high = config.upper_bounds[mode];
      std::normal_distribution<double> distribution(means[mode], 1.0);
      const double coord = clamp_into_range(distribution(generator), low, high);
      const double width = (high - low) / static_cast<double>(dims[mode]);
      int bin = static_cast<int>(std::floor((coord - low) / width));
      bin = std::max(0, std::min(bin, static_cast<int>(dims[mode] - 1)));
      idx[mode] = static_cast<boba::index_t>(bin);
    }
    dense_counts[static_cast<std::size_t>(indexer.index(idx))] += 1.0;
  }

  const boba::index_t nnz = static_cast<boba::index_t>(
    std::count_if(dense_counts.begin(), dense_counts.end(), [](double value)
  {
    return value > 0.0;
  }));

  typename sparse_tensor<dimension>::index_list_array_t index_lists;
  for (std::size_t mode = 0; mode < dimension; ++mode)
  {
    index_lists[mode].resize({nnz});
  }

  typename sparse_tensor<dimension>::values_list_t values({nnz});
  auto values_view = values.view();

  boba::index_t entry = 0;
  for (std::size_t flat = 0; flat < dense_counts.size(); ++flat)
  {
    if (dense_counts[flat] > 0.0)
    {
      const auto idx = indexer.multiindex(static_cast<boba::index_t>(flat));
      for (std::size_t mode = 0; mode < dimension; ++mode)
      {
        index_lists[mode].view()(entry) = static_cast<boba::index_t>(idx[mode]);
      }
      values_view(entry) = dense_counts[flat];
      ++entry;
    }
  }
  return sparse_tensor<dimension>(dims, std::move(index_lists), std::move(values));
}

template <std::size_t dimension>
dense_tensor<dimension> dense_tensor_from_sparse(const sparse_tensor<dimension>& tensor)
{
  dense_tensor<dimension> dense(tensor.sizes());
  dense.fill_with_zeros();
  auto tensor_view = tensor.const_view();
  auto values_view = tensor.values_tensor().const_view();
  auto dense_view = dense.view();

  ::boba::detail::loop<space>(
    0_z,
    static_cast<std::size_t>(tensor.number_nonzeros()),
    [=] __boba_host_device__(std::size_t entry)
  {
    const auto boba_entry = static_cast<boba::index_t>(entry);
    const auto idx = tensor_view.entry_multiindex(boba_entry);
    dense_view(idx) = values_view(boba_entry);
  });

  return dense;
}

template <std::size_t dimension>
double sparse_tensor_entropy(const sparse_tensor<dimension>& tensor)
{
  auto values_view = tensor.values_tensor().const_view();
  const auto nnz = static_cast<std::size_t>(tensor.number_nonzeros());
  double mass = 0.0;
  ::boba::sum_reduce<space>(mass, 0_z, nnz, [=] __boba_host_device__(std::size_t entry, boba::sum_reducer_operator<double>& local_mass)
  {
    local_mass += values_view(static_cast<boba::index_t>(entry));
  });
  if (mass <= 0.0)
  {
    return 0.0;
  }

  double entropy = 0.0;
  ::boba::sum_reduce<space>(entropy, 0_z, nnz, [=] __boba_host_device__(std::size_t entry, boba::sum_reducer_operator<double>& local_entropy)
  {
    const double probability = values_view(static_cast<boba::index_t>(entry)) / mass;
    if (probability > 0.0)
    {
      local_entropy += -probability * std::log(probability);
    }
  });
  return entropy;
}

template <std::size_t dimension>
double tensor_entropy(const CPAPRModel<dimension>& model, const tensor_shape<dimension>& dims)
{
  const auto indexer = boba::Multiindexer<dimension>(dims);
  const auto bins = static_cast<std::size_t>(indexer.size());
  std::vector<double> estimates(bins, 0.0);
  double max_estimate = 0.0;
  for (std::size_t flat = 0; flat < bins; ++flat)
  {
    const auto idx = indexer.multiindex(static_cast<boba::index_t>(flat));
    const double estimate = raw_tensor_entry_estimate(model, idx);
    if (std::isfinite(estimate) && estimate > 0.0)
    {
      estimates[flat] = estimate;
      max_estimate = std::max(max_estimate, estimate);
    }
  }

  if (!(max_estimate > 0.0) || !std::isfinite(max_estimate))
  {
    return 0.0;
  }

  double scaled_mass = 0.0;
  for (double estimate : estimates)
  {
    scaled_mass += estimate / max_estimate;
  }

  if (!(scaled_mass > 0.0) || !std::isfinite(scaled_mass))
  {
    return 0.0;
  }

  double entropy = 0.0;
  for (double estimate : estimates)
  {
    if (estimate > 0.0)
    {
      const double probability = (estimate / max_estimate) / scaled_mass;
      if (probability > 0.0)
      {
        entropy -= probability * std::log(probability);
      }
    }
  }

  return entropy;
}

void test_cp_apr_solver_reuse(bool& check)
{
  constexpr std::size_t dimension = 2;
  const tensor_shape<dimension> sizes{2, 3};
  dense_tensor<dimension> counts(sizes);
  counts.fill_with(1.0);

  CPAPRModel<dimension> initial_model(sizes);
  initial_model.m_weights.resize(2);
  initial_model.m_weights.fill_with(1.0);
  for (std::size_t mode = 0; mode < dimension; ++mode)
  {
    initial_model.m_cores[mode].resize({sizes[mode], 2});
    initial_model.m_cores[mode].fill_with(
      1.0 / static_cast<double>(sizes[mode]));
  }

  boba::CPAPRParameters<double> parameters;
  parameters.max_outer_iterations = 2;
  parameters.max_inner_iterations = 2;
  parameters.tolerance = 1.0e-8;

  const auto expected = boba::cp_apr_fit(counts, initial_model, parameters);
  boba::CPAPRSolver<dimension, space, double> solver(parameters);
  const auto actual = solver.fit(counts, initial_model);
  const auto reused = solver.fit(counts, initial_model);

  pass_or_fail(
    check,
    boba::norm_difference_frobenius(expected.decompress(), actual.decompress()),
    1.0e-12);
  pass_or_fail(
    check,
    boba::norm_difference_frobenius(actual.decompress(), reused.decompress()),
    1.0e-12);
}

template <std::size_t dimension>
int run_tensor_completion(const CommandLineOptions& options)
{
  bool check = true;
  test_cp_apr_solver_reuse(check);

  GaussianHistogramConfig generator_config;
  generator_config.dims = options.dims;
  generator_config.lower_bounds = std::vector<double>(generator_config.dims.size(), 0.0);
  generator_config.upper_bounds.reserve(generator_config.dims.size());
  for (int dim : generator_config.dims)
  {
    generator_config.upper_bounds.push_back(options.bin_width * static_cast<double>(dim));
  }
  generator_config.seed = 20260323;

  const sparse_tensor<dimension> observed_tensor =
    generate_gaussian_binned_tensor<dimension>(options.sample_count, generator_config);

  boba::CPAPRParameters<double> parameters;
  parameters.max_outer_iterations = 150;
  parameters.max_inner_iterations = 15;
  parameters.tolerance = 1.0e-5;

  std::mt19937 random_generator(20260323);
  boba::CPAPRSolver<dimension, space, double> solver(parameters);
  CPAPRModel<dimension> model;
  if (options.dense_apr)
  {
    const auto dense_observed_tensor = dense_tensor_from_sparse<dimension>(observed_tensor);
    model = solver.fit(
      dense_observed_tensor,
      static_cast<std::size_t>(options.rank),
      random_generator);
  }
  else
  {
    model = solver.fit(
      observed_tensor,
      static_cast<std::size_t>(options.rank),
      random_generator);
  }

  const auto observed_dims = observed_tensor.sizes();
  const double observed_mass = observed_tensor.values_tensor().sum_reduce();
  const double observed_tensor_entropy = sparse_tensor_entropy(observed_tensor);
  const double completed_tensor_entropy = tensor_entropy(model, observed_dims);

  boba_print(std::string("Observed tensor entropy (nats): ") + std::to_string(observed_tensor_entropy));
  boba_print(std::string("Completed tensor entropy (nats): ") + std::to_string(completed_tensor_entropy));

  pass_or_fail_bool(check, observed_dims.size() == dimension);
  pass_or_fail(check, observed_mass - static_cast<double>(options.sample_count), 1.0e-12);
  pass_or_fail_bool(check, std::isfinite(observed_tensor_entropy));
  pass_or_fail_bool(check, observed_tensor_entropy >= 0.0);
  pass_or_fail_bool(check, std::isfinite(completed_tensor_entropy));
  pass_or_fail_bool(check, completed_tensor_entropy >= 0.0);
  pass_or_fail_bool(check, model_is_valid_probability_model(model));
  pass_or_fail(check, model.weights().sum_reduce() - observed_mass, 1.0e-10);

  if (options.verbose_bins)
  {
    boba_print(std::string("Observed tensor dims: ") + format_dims(observed_dims));
    boba_print(
      std::string("Bin widths: ") +
      format_values(std::vector<double>(observed_dims.size(), options.bin_width)));
    boba_print(std::string("Samples binned: ") + std::to_string(options.sample_count));
    boba_print(std::string("CP rank: ") + std::to_string(options.rank));
    boba_print(std::string("APR input path: ") + (options.dense_apr ? "dense" : "sparse"));
    boba_print(std::string("Total bins: ") + std::to_string(static_cast<long long>(observed_tensor.size())));
    boba_print(std::string("Nonzero bins: ") + std::to_string(static_cast<long long>(observed_tensor.number_nonzeros())));
    boba_print(std::string("Observed total mass: ") + std::to_string(observed_mass));
    print_lambda(model);
    print_observed_counts(observed_tensor, model);
    print_empty_estimated_counts(observed_tensor, model);
  }

  return final_check(check);
}

int main(int argc, char** argv)
{
  boba::splash();
  boba::init();

  int return_code = 1;
  try
  {
    const CommandLineOptions options = parse_args(argc, argv);
    switch (options.dims.size())
    {
    case 1:
      return_code = run_tensor_completion<1>(options);
      break;
    case 2:
      return_code = run_tensor_completion<2>(options);
      break;
    case 3:
      return_code = run_tensor_completion<3>(options);
      break;
    case 4:
      return_code = run_tensor_completion<4>(options);
      break;
    case 5:
      return_code = run_tensor_completion<5>(options);
      break;
    case 6:
      return_code = run_tensor_completion<6>(options);
      break;
    case 7:
      return_code = run_tensor_completion<7>(options);
      break;
    case 8:
      return_code = run_tensor_completion<8>(options);
      break;
    default:
      throw std::invalid_argument("tensor completion currently supports tensor order 1 through 8");
    }
  }
  catch (const std::exception& ex)
  {
    boba_print(ex.what());
    print_usage(argv[0]);
    return_code = 1;
  }

  boba::finalize();
  return return_code;
}

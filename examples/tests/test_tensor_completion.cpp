#include "common.hpp"
#include "test_tensor_completion.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

using host_matrix = boba::Matrix<tensor_completion_space, double>;
using host_vector = boba::Vector<tensor_completion_space, double>;

template<std::size_t dimension>
using dense_tensor = boba::Tensor<dimension, tensor_completion_space, double>;

template<std::size_t dimension>
using sparse_tensor = TensorCOO<dimension>;

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
  if(value < low)
  {
    return low;
  }

  const double high_open = std::nextafter(high, low);
  if(value >= high)
  {
    return high_open;
  }
  return value;
};

constexpr auto clamp_min = [](double x, double eps)
{
  return (x < eps) ? eps : x;
};

const auto format_parse_error = [](const ::boba::argparser &args)
{
  std::ostringstream out;
  args.print_error(out);
  std::string message = out.str();
  while(!message.empty() && (message.back() == '\n' || message.back() == '\r'))
  {
    message.pop_back();
  }
  return message;
};

const auto format_int_values = [](const std::vector<int> &values)
{
  std::ostringstream out;
  out << "[";
  for(std::size_t i = 0; i < values.size(); ++i)
  {
    if(i > 0)
    {
      out << ", ";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
};

const auto print_usage = [](const char *program)
{
  boba_print(
    std::string("Usage: ") + program +
    " [--samples N] [--rank R] [--bin-width W] [--dims D1 D2 ... Dk] [--dense-apr] [--verbose-bins]");
  boba_print("Defaults: --samples 2500 --rank 5 --bin-width 1.0 --dims 20 20 20 20 20 20");
  boba_print("Supported tensor order for this BoBa CPD driver: 1 through 8");
};

CommandLineOptions parse_args(int argc, char **argv)
{
  CommandLineOptions options;
  ::boba::argparser args(argc, argv);
  args.add_optional_argument(options.sample_count, "", "--samples", "Number of synthetic samples.");
  args.add_optional_argument(options.rank, "", "--rank", "CP decomposition rank.");
  args.add_optional_argument(options.bin_width, "", "--bin-width", "Histogram bin width.");
  args.add_optional_argument(options.dims, "", "--dims", "Tensor extents per mode.");
  args.add_optional_argument(options.dense_apr, "", "--dense-apr", "Use dense APR input path.");
  args.add_optional_argument(options.verbose_bins, "", "--verbose-bins", "Print bin details.");

  const auto result = args.parse();
  if(options.sample_count == 0)
  {
    throw std::invalid_argument("invalid value for --samples: 0");
  }
  if(options.rank <= 0)
  {
    throw std::invalid_argument("invalid value for --rank: " + std::to_string(options.rank));
  }
  if(options.bin_width <= 0.0)
  {
    throw std::invalid_argument("invalid value for --bin-width: " + std::to_string(options.bin_width));
  }
  if(options.dims.empty())
  {
    throw std::invalid_argument("invalid value for --dims: []");
  }
  for(int dim : options.dims)
  {
    if(dim <= 0)
    {
      throw std::invalid_argument("invalid value for --dims: " + format_int_values(options.dims));
    }
  }
  return options;
}

template<std::size_t dimension>
double total_mass(const sparse_tensor<dimension> &tensor)
{
  auto values_view = tensor.values.const_view();
  double total = 0.0;
  ::boba::sum_reduce<tensor_completion_space>(total, 0_z, static_cast<std::size_t>(tensor.values.size()),
      [=]__boba_host_device__(std::size_t entry, boba::sum_reducer_operator<double> &local_total)
  {
    local_total += values_view(static_cast<boba::index_t>(entry));
  });
  return total;
}

template<std::size_t dimension>
std::size_t total_bins(const coo_index<dimension> &dims)
{
  std::size_t total = 1;
  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    total *= static_cast<std::size_t>(dims[mode]);
  }
  return total;
}

template<typename index_t, std::size_t dimension>
std::string format_index(const boba::Array<index_t, dimension> &idx)
{
  std::ostringstream out;
  out << "(";
  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    if(mode > 0)
    {
      out << ", ";
    }
    out << idx[mode];
  }
  out << ")";
  return out.str();
}

std::string format_values(const std::vector<double> &values)
{
  std::ostringstream out;
  out << "[";
  for(std::size_t mode = 0; mode < values.size(); ++mode)
  {
    if(mode > 0)
    {
      out << ", ";
    }
    out << values[mode];
  }
  out << "]";
  return out.str();
}

template<typename index_t, std::size_t dimension>
std::string format_dims(const boba::Array<index_t, dimension> &dims)
{
  std::ostringstream out;
  out << "[";
  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    if(mode > 0)
    {
      out << ", ";
    }
    out << dims[mode];
  }
  out << "]";
  return out.str();
}

template<std::size_t dimension>
coo_index<dimension> to_coo_dims(const std::vector<int> &dims)
{
  if(dims.size() != dimension)
  {
    throw std::invalid_argument("tensor order does not match CP model dimension");
  }

  auto sizes = boba::filled_array<dimension>(boba::index_t(0));
  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    sizes[mode] = static_cast<boba::index_t>(dims[mode]);
  }
  return sizes;
}

template<std::size_t dimension>
boba::Array<boba::index_t, dimension> to_boba_dims(const coo_index<dimension> &dims)
{
  auto sizes = boba::filled_array<dimension>(0_z);
  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    sizes[mode] = static_cast<boba::index_t>(dims[mode]);
  }
  return sizes;
}

template<std::size_t dimension>
auto make_multiindexer(const coo_index<dimension> &dims)
{
  return boba::Multiindexer<dimension>(to_boba_dims(dims));
}

template<std::size_t dimension>
boba::index_t sparse_nnz(const sparse_tensor<dimension> &tensor)
{
  return tensor.values.size();
}

template<std::size_t dimension, typename indices_view_t>
__boba_host_device__
coo_index<dimension> coo_index_at(const indices_view_t &indices_view, boba::index_t entry)
{
  auto idx = boba::filled_array<dimension>(boba::index_t(0));
  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    idx[mode] = indices_view({entry, static_cast<boba::index_t>(mode)});
  }
  return idx;
}

template<std::size_t dimension>
coo_index<dimension> coo_index_at(const sparse_tensor<dimension> &tensor, boba::index_t entry)
{
  return coo_index_at<dimension>(tensor.get_indices_const_view(), entry);
}

void column_sum_normalize(host_matrix &A)
{
  auto A_view = A.view();
  for(boba::index_t r = 0; r < A.cols(); ++r)
  {
    double sum = 0.0;
    ::boba::sum_reduce<tensor_completion_space>(sum, 0_z, static_cast<std::size_t>(A.rows()),
        [=]__boba_host_device__(std::size_t i, boba::sum_reducer_operator<double> &local_sum)
    {
      local_sum += A_view({static_cast<boba::index_t>(i), r});
    });

    if(sum <= 0.0)
    {
      const double uniform = 1.0 / static_cast<double>(A.rows());
      ::boba::detail::loop<tensor_completion_space>(0_z, static_cast<std::size_t>(A.rows()),
          [=]__boba_host_device__(std::size_t i)
      {
        A_view({static_cast<boba::index_t>(i), r}) = uniform;
      });
      continue;
    }

    const double inverse = 1.0 / sum;
    ::boba::detail::loop<tensor_completion_space>(0_z, static_cast<std::size_t>(A.rows()),
        [=]__boba_host_device__(std::size_t i)
    {
      A_view({static_cast<boba::index_t>(i), r}) *= inverse;
    });
  }
}

host_vector col_sums(const host_matrix &B)
{
  host_vector lambda({B.cols()});
  lambda.fill_with_zeros();

  auto lambda_view = lambda.view();
  auto B_view = B.const_view();
  for(boba::index_t r = 0; r < B.cols(); ++r)
  {
    double sum = 0.0;
    ::boba::sum_reduce<tensor_completion_space>(sum, 0_z, static_cast<std::size_t>(B.rows()),
        [=]__boba_host_device__(std::size_t i, boba::sum_reducer_operator<double> &local_sum)
    {
      local_sum += B_view({static_cast<boba::index_t>(i), r});
    });
    lambda_view(r) = sum;
  }

  return lambda;
}

void unabsorb_and_normalize(const host_matrix &B, const host_vector &lambda, host_matrix &A_out)
{
  A_out.resize({B.rows(), B.cols()});
  A_out.fill_with_zeros();

  auto A_view = A_out.view();
  auto B_view = B.const_view();
  auto lambda_view = lambda.const_view();
  for(boba::index_t r = 0; r < B.cols(); ++r)
  {
    const double lam = lambda_view(r);
    if(lam > 0.0)
    {
      const double inverse = 1.0 / lam;
      ::boba::detail::loop<tensor_completion_space>(0_z, static_cast<std::size_t>(B.rows()),
          [=]__boba_host_device__(std::size_t i)
      {
        const auto row = static_cast<boba::index_t>(i);
        A_view({row, r}) = B_view({row, r}) * inverse;
      });
    }
    else
    {
      const double uniform = 1.0 / static_cast<double>(B.rows());
      ::boba::detail::loop<tensor_completion_space>(0_z, static_cast<std::size_t>(B.rows()),
          [=]__boba_host_device__(std::size_t i)
      {
        A_view({static_cast<boba::index_t>(i), r}) = uniform;
      });
    }
  }

  column_sum_normalize(A_out);
}

double inf_norm_mat_min_B_1minusPhi(const host_matrix &B, const host_matrix &Phi)
{
  double norm = 0.0;
  auto B_view = B.const_view();
  auto Phi_view = Phi.const_view();
  ::boba::max_reduce<tensor_completion_space>(norm, 0_z, static_cast<std::size_t>(B.rows() * B.cols()),
      [=]__boba_host_device__(std::size_t flat, boba::max_reducer_operator<double> &local_norm)
  {
    const auto [i, r] = B_view.multiindex(flat);
    const double b = B_view({i, r});
    const double g = 1.0 - Phi_view({i, r});
    local_norm.max(std::abs(std::min(b, g)));
  });

  return norm;
}

template<std::size_t dimension, typename index_t>
void build_w_for_index(
  const boba::Array<index_t, dimension> &idx,
  std::size_t mode_to_skip,
  const CPAPRModel<dimension> &model,
  std::vector<double> &w_out)
{
  const auto cores = model.get_core_const_views();
  w_out.assign(model.rank(), 1.0);

  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    if(mode == mode_to_skip)
    {
      continue;
    }

    const auto row = static_cast<boba::index_t>(idx[mode]);
    for(boba::index_t r = 0; r < model.m_cores[mode].cols(); ++r)
    {
      w_out[static_cast<std::size_t>(r)] *= cores[mode]({row, r});
    }
  }
}

template<std::size_t dimension>
void compute_Phi_mode_n(
  const sparse_tensor<dimension> &tensor,
  std::size_t mode,
  const CPAPRModel<dimension> &model,
  const host_matrix &B,
  double epsilon,
  host_matrix &Phi_out)
{
  Phi_out.resize({B.rows(), B.cols()});
  Phi_out.fill_with_zeros();

  auto Phi_view = Phi_out.view();
  auto B_view = B.const_view();
  auto indices_view = tensor.get_indices_const_view();
  auto values_view = tensor.values.const_view();
  std::vector<double> w;

  for(boba::index_t entry = 0; entry < sparse_nnz(tensor); ++entry)
  {
    const auto idx = coo_index_at<dimension>(indices_view, entry);
    const auto i_n = idx[mode];
    build_w_for_index(idx, mode, model, w);

    double model_prediction = 0.0;
    for(boba::index_t r = 0; r < B.cols(); ++r)
    {
      model_prediction += w[static_cast<std::size_t>(r)] * B_view({i_n, r});
    }

    model_prediction = clamp_min(model_prediction, epsilon);
    const double hatv = values_view(entry) / model_prediction;
    for(boba::index_t r = 0; r < B.cols(); ++r)
    {
      Phi_view({i_n, r}) += hatv * w[static_cast<std::size_t>(r)];
    }
  }
}

template<std::size_t dimension>
void compute_Phi_mode_n(
  const dense_tensor<dimension> &tensor,
  std::size_t mode,
  const CPAPRModel<dimension> &model,
  const host_matrix &B,
  double epsilon,
  host_matrix &Phi_out)
{
  Phi_out.resize({B.rows(), B.cols()});
  Phi_out.fill_with_zeros();

  auto Phi_view = Phi_out.view();
  auto B_view = B.const_view();
  std::vector<double> w;

  for(boba::index_t flat = 0; flat < tensor.size(); ++flat)
  {
    const double value = tensor.const_data()[flat];
    if(value <= 0.0)
    {
      continue;
    }

    const auto multiindex = tensor.multiindex(flat);
    const auto i_n = multiindex[mode];
    build_w_for_index(multiindex, mode, model, w);

    double model_prediction = 0.0;
    for(boba::index_t r = 0; r < B.cols(); ++r)
    {
      model_prediction += w[static_cast<std::size_t>(r)] * B_view({i_n, r});
    }

    model_prediction = clamp_min(model_prediction, epsilon);
    const double hatv = value / model_prediction;

    for(boba::index_t r = 0; r < B.cols(); ++r)
    {
      Phi_view({i_n, r}) += hatv * w[static_cast<std::size_t>(r)];
    }
  }
}

template<std::size_t dimension>
void print_lambda(const CPAPRModel<dimension> &model)
{
  std::ostringstream out;
  out << "Learned lambda:";
  auto weights_view = model.weights().const_view();
  for(boba::index_t r = 0; r < model.weights().size(); ++r)
  {
    out << ' ' << std::fixed << std::setprecision(4) << weights_view(r);
  }
  boba_print(out.str());
}

template<std::size_t dimension, typename index_t>
double raw_tensor_entry_estimate(const CPAPRModel<dimension> &model, const boba::Array<index_t, dimension> &idx)
{
  const auto weights_view = model.weights().const_view();
  const auto cores = model.get_core_const_views();

  double estimate = 0.0;
  for(boba::index_t r = 0; r < model.weights().size(); ++r)
  {
    double component = weights_view(r);
    for(std::size_t mode = 0; mode < dimension; ++mode)
    {
      component *= cores[mode]({static_cast<boba::index_t>(idx[mode]), r});
    }
    estimate += component;
  }
  return estimate;
}

template<std::size_t dimension, typename index_t>
double tensor_entry_estimate(const CPAPRModel<dimension> &model, const boba::Array<index_t, dimension> &idx)
{
  const double estimate = raw_tensor_entry_estimate(model, idx);
  return (estimate < kEstimateTolerance) ? 0.0 : estimate;
}

template<std::size_t dimension>
void print_observed_counts(const sparse_tensor<dimension> &tensor, const CPAPRModel<dimension> &model)
{
  std::cout << "\nNon-zero count bin data (CP-APR)\n";
  auto indices_view = tensor.get_indices_const_view();
  auto values_view = tensor.values.const_view();
  for(boba::index_t entry = 0; entry < sparse_nnz(tensor); ++entry)
  {
    const auto idx = coo_index_at<dimension>(indices_view, entry);
    const long long original_count = static_cast<long long>(values_view(entry));
    std::ostringstream out;
    out << format_index(idx) << " "
        << std::fixed << std::setprecision(3)
        << tensor_entry_estimate(model, idx)
        << "(" << original_count << ")";
    boba_print(out.str());
  }
}

template<std::size_t dimension>
void print_empty_estimated_counts(const sparse_tensor<dimension> &tensor, const CPAPRModel<dimension> &model)
{
  const auto indexer = make_multiindexer(tensor.dims);
  auto indices_view = tensor.get_indices_const_view();
  const auto bins = static_cast<std::size_t>(indexer.size());
  std::vector<bool> observed_bins(bins, false);
  for(boba::index_t entry = 0; entry < sparse_nnz(tensor); ++entry)
  {
    observed_bins[static_cast<std::size_t>(indexer.index(coo_index_at<dimension>(indices_view, entry)))] = true;
  }

  boba_print("");
  boba_print("Originally empty bins with non-zero Poisson estimate (CP-APR)");

  bool found = false;
  for(std::size_t flat = 0; flat < bins; ++flat)
  {
    const auto idx = indexer.multiindex(static_cast<boba::index_t>(flat));
    if(!observed_bins[flat])
    {
      const double estimate = tensor_entry_estimate(model, idx);
      if(estimate > 0.0)
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

  if(!found)
  {
    boba_print("none");
  }
}

template<std::size_t dimension>
bool model_is_finite(const CPAPRModel<dimension> &model)
{
  auto weights_view = model.weights().const_view();
  for(boba::index_t r = 0; r < model.weights().size(); ++r)
  {
    if(!std::isfinite(weights_view(r)))
    {
      return false;
    }
  }

  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    auto factor_view = model.m_cores[mode].const_view();
    for(boba::index_t i = 0; i < model.m_cores[mode].rows(); ++i)
    {
      for(boba::index_t r = 0; r < model.m_cores[mode].cols(); ++r)
      {
        if(!std::isfinite(factor_view({i, r})))
        {
          return false;
        }
      }
    }
  }

  return true;
}

} // namespace

template<std::size_t dimension>
sparse_tensor<dimension> generate_gaussian_binned_tensor(
  std::size_t sample_count,
  const GaussianHistogramConfig &config)
{
  if(config.dims.size() != dimension)
  {
    throw std::invalid_argument("histogram dimensions must match tensor order");
  }
  if(config.lower_bounds.size() != config.dims.size() ||
     config.upper_bounds.size() != config.dims.size())
  {
    throw std::invalid_argument("histogram bounds must match tensor order");
  }

  const auto dims = to_coo_dims<dimension>(config.dims);
  const auto indexer = make_multiindexer(dims);
  const std::size_t order = dimension;
  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    const auto dim = dims[mode];
    if(dim <= 0)
    {
      throw std::invalid_argument("all tensor dimensions must be positive");
    }
  }

  std::vector<double> dense_counts(static_cast<std::size_t>(indexer.size()), 0.0);
  std::mt19937 generator(config.seed);

  std::vector<double> means(order, 0.0);
  std::vector<double> sigmas(order, 0.0);
  for(std::size_t mode = 0; mode < order; ++mode)
  {
    const double low = config.lower_bounds[mode];
    const double high = config.upper_bounds[mode];
    if(high <= low)
    {
      throw std::invalid_argument("upper bound must exceed lower bound");
    }
    means[mode] = 0.5 * (low + high);
    sigmas[mode] = 1.0;
  }

  for(std::size_t sample = 0; sample < sample_count; ++sample)
  {
    auto idx = boba::filled_array<dimension>(boba::index_t(0));
    for(std::size_t mode = 0; mode < order; ++mode)
    {
      std::normal_distribution<double> dist(means[mode], sigmas[mode]);
      const double low = config.lower_bounds[mode];
      const double high = config.upper_bounds[mode];
      const double coord = clamp_into_range(dist(generator), low, high);
      const double width = (high - low) / static_cast<double>(dims[mode]);
      int bin = static_cast<int>(std::floor((coord - low) / width));
      bin = std::max(0, std::min(bin, static_cast<int>(dims[mode] - 1)));
      idx[mode] = static_cast<boba::index_t>(bin);
    }
    dense_counts[static_cast<std::size_t>(indexer.index(idx))] += 1.0;
  }

  sparse_tensor<dimension> tensor;
  tensor.dims = dims;
  const boba::index_t nnz = static_cast<boba::index_t>(
    std::count_if(dense_counts.begin(), dense_counts.end(), [](double value) { return value > 0.0; }));
  tensor.indices.resize({nnz, static_cast<boba::index_t>(dimension)});
  tensor.values.resize(nnz);
  auto indices_view = tensor.indices.view();
  auto values_view = tensor.values.view();
  boba::index_t entry = 0;
  for(std::size_t flat = 0; flat < dense_counts.size(); ++flat)
  {
    if(dense_counts[flat] > 0.0)
    {
      const auto idx = indexer.multiindex(static_cast<boba::index_t>(flat));
      for(std::size_t mode = 0; mode < dimension; ++mode)
      {
        indices_view({entry, static_cast<boba::index_t>(mode)}) = static_cast<boba::index_t>(idx[mode]);
      }
      values_view(entry) = dense_counts[flat];
      ++entry;
    }
  }
  return tensor;
}

template<std::size_t dimension>
dense_tensor<dimension> dense_tensor_from_sparse(const sparse_tensor<dimension> &tensor)
{
  dense_tensor<dimension> dense(to_boba_dims(tensor.dims));
  dense.fill_with_zeros();
  auto indices_view = tensor.get_indices_const_view();
  auto values_view = tensor.values.const_view();
  auto dense_view = dense.view();

  ::boba::detail::loop<tensor_completion_space>(0_z, static_cast<std::size_t>(sparse_nnz(tensor)),
      [=]__boba_host_device__(std::size_t entry)
  {
    const auto boba_entry = static_cast<boba::index_t>(entry);
    const auto idx = coo_index_at<dimension>(indices_view, boba_entry);
    dense_view(to_boba_dims(idx)) = values_view(boba_entry);
  });

  return dense;
}

template<std::size_t dimension>
double sparse_tensor_entropy(const sparse_tensor<dimension> &tensor)
{
  auto values_view = tensor.values.const_view();
  double mass = 0.0;
  ::boba::sum_reduce<tensor_completion_space>(mass, 0_z, static_cast<std::size_t>(sparse_nnz(tensor)),
      [=]__boba_host_device__(std::size_t entry, boba::sum_reducer_operator<double> &local_mass)
  {
    local_mass += values_view(static_cast<boba::index_t>(entry));
  });
  if(mass <= 0.0)
  {
    return 0.0;
  }

  double entropy = 0.0;
  ::boba::sum_reduce<tensor_completion_space>(entropy, 0_z, static_cast<std::size_t>(sparse_nnz(tensor)),
      [=]__boba_host_device__(std::size_t entry, boba::sum_reducer_operator<double> &local_entropy)
  {
    const double probability = values_view(static_cast<boba::index_t>(entry)) / mass;
    if(probability > 0.0)
    {
      local_entropy += -probability * std::log(probability);
    }
  });
  return entropy;
}

template<std::size_t dimension>
double tensor_entropy(const CPAPRModel<dimension> &model, const coo_index<dimension> &dims)
{
  const auto indexer = make_multiindexer(dims);
  const auto bins = static_cast<std::size_t>(indexer.size());
  double max_estimate = 0.0;
  for(std::size_t flat = 0; flat < bins; ++flat)
  {
    const auto idx = indexer.multiindex(static_cast<boba::index_t>(flat));
    const double estimate = raw_tensor_entry_estimate(model, idx);
    if(std::isfinite(estimate) && estimate > 0.0)
    {
      max_estimate = std::max(max_estimate, estimate);
    }
  }

  if(!(max_estimate > 0.0) || !std::isfinite(max_estimate))
  {
    return 0.0;
  }

  double scaled_mass = 0.0;
  for(std::size_t flat = 0; flat < bins; ++flat)
  {
    const auto idx = indexer.multiindex(static_cast<boba::index_t>(flat));
    const double estimate = raw_tensor_entry_estimate(model, idx);
    if(std::isfinite(estimate) && estimate > 0.0)
    {
      scaled_mass += estimate / max_estimate;
    }
  }

  if(!(scaled_mass > 0.0) || !std::isfinite(scaled_mass))
  {
    return 0.0;
  }

  double entropy = 0.0;
  for(std::size_t flat = 0; flat < bins; ++flat)
  {
    const auto idx = indexer.multiindex(static_cast<boba::index_t>(flat));
    const double estimate = raw_tensor_entry_estimate(model, idx);
    if(std::isfinite(estimate) && estimate > 0.0)
    {
      const double probability = (estimate / max_estimate) / scaled_mass;
      if(probability > 0.0 && std::isfinite(probability))
      {
        entropy -= probability * std::log(probability);
      }
    }
  }

  return entropy;
}

template<std::size_t dimension>
CPAPRModel<dimension> initialize_random_stochastic_from_dims(
  const coo_index<dimension> &dims,
  int rank,
  unsigned seed)
{
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  CPAPRModel<dimension> model(to_boba_dims<dimension>(dims));
  model.rename("tensor_completion_cpd");
  model.m_weights.resize({static_cast<boba::index_t>(rank)});
  model.m_weights.fill_with(1.0);

  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    model.m_cores[mode].resize(
      {static_cast<boba::index_t>(dims[mode]), static_cast<boba::index_t>(rank)});

    auto factor_view = model.m_cores[mode].view();
    for(boba::index_t i = 0; i < model.m_cores[mode].rows(); ++i)
    {
      for(boba::index_t r = 0; r < model.m_cores[mode].cols(); ++r)
      {
        factor_view({i, r}) = dist(generator) + 1.0e-6;
      }
    }
    column_sum_normalize(model.m_cores[mode]);
  }

  return model;
}

template<std::size_t dimension>
CPAPRModel<dimension> initialize_random_stochastic(const sparse_tensor<dimension> &tensor, int rank, unsigned seed)
{
  return initialize_random_stochastic_from_dims<dimension>(tensor.dims, rank, seed);
}

template<std::size_t dimension>
CPAPRModel<dimension> initialize_random_stochastic(const dense_tensor<dimension> &tensor, int rank, unsigned seed)
{
  auto dims = boba::filled_array<dimension>(boba::index_t(0));
  for(std::size_t mode = 0; mode < dimension; ++mode)
  {
    dims[mode] = tensor.sizes(static_cast<boba::index_t>(mode));
  }
  return initialize_random_stochastic_from_dims<dimension>(dims, rank, seed);
}

template<std::size_t dimension>
CPAPRModel<dimension> cp_apr_fit(
  const sparse_tensor<dimension> &tensor,
  CPAPRParams params,
  CPAPRModel<dimension> model)
{
  host_matrix Phi;

  for(int k = 1; k <= params.kmax; ++k)
  {
    bool is_converged = true;

    for(std::size_t mode = 0; mode < dimension; ++mode)
    {
      const auto extent = static_cast<boba::index_t>(tensor.dims[mode]);
      host_matrix S({extent, static_cast<boba::index_t>(model.rank())});
      S.fill_with_zeros();

      host_matrix B({extent, static_cast<boba::index_t>(model.rank())});
      auto B_view = B.view();
      auto factor_view = model.m_cores[mode].const_view();
      auto lambda_view = model.m_weights.const_view();
      ::boba::loop<tensor_completion_space, 2>(
          {static_cast<std::size_t>(extent), static_cast<std::size_t>(B.cols())},
          [=]__boba_host_device__(::boba::Array<std::size_t, 2> ij)
      {
        const auto i = static_cast<boba::index_t>(ij[0]);
        const auto r = static_cast<boba::index_t>(ij[1]);
        B_view({i, r}) = factor_view({i, r}) * lambda_view(r);
      });

      compute_Phi_mode_n(tensor, mode, model, B, params.epsilon, Phi);

      if(k > 1)
      {
        auto S_view = S.view();
        auto Phi_view = Phi.const_view();
        ::boba::loop<tensor_completion_space, 2>(
            {static_cast<std::size_t>(extent), static_cast<std::size_t>(B.cols())},
            [=]__boba_host_device__(::boba::Array<std::size_t, 2> ij)
        {
          const auto i = static_cast<boba::index_t>(ij[0]);
          const auto r = static_cast<boba::index_t>(ij[1]);
          if(factor_view({i, r}) < params.kappa_tol && Phi_view({i, r}) > 1.0)
          {
            S_view({i, r}) = params.kappa;
          }
        });
      }

      auto S_view = S.const_view();
      ::boba::loop<tensor_completion_space, 2>(
          {static_cast<std::size_t>(extent), static_cast<std::size_t>(B.cols())},
          [=]__boba_host_device__(::boba::Array<std::size_t, 2> ij)
      {
        const auto i = static_cast<boba::index_t>(ij[0]);
        const auto r = static_cast<boba::index_t>(ij[1]);
        B_view({i, r}) = (factor_view({i, r}) + S_view({i, r})) * lambda_view(r);
      });

      for(int ell = 1; ell <= params.ellmax; ++ell)
      {
        compute_Phi_mode_n(tensor, mode, model, B, params.epsilon, Phi);

        const double kkt = inf_norm_mat_min_B_1minusPhi(B, Phi);
        if(kkt < params.tau)
        {
          break;
        }

        is_converged = false;
        auto Phi_view = Phi.const_view();
        ::boba::loop<tensor_completion_space, 2>(
            {static_cast<std::size_t>(extent), static_cast<std::size_t>(B.cols())},
            [=]__boba_host_device__(::boba::Array<std::size_t, 2> ij)
        {
          const auto i = static_cast<boba::index_t>(ij[0]);
          const auto r = static_cast<boba::index_t>(ij[1]);
          B_view({i, r}) *= Phi_view({i, r});
        });
      }

      model.m_weights = col_sums(B);
      unabsorb_and_normalize(B, model.m_weights, model.m_cores[mode]);
    }

    if(is_converged)
    {
      break;
    }
  }

  return model;
}

template<std::size_t dimension>
CPAPRModel<dimension> cp_apr_fit(
  const dense_tensor<dimension> &tensor,
  CPAPRParams params,
  CPAPRModel<dimension> model)
{
  host_matrix Phi;

  for(int k = 1; k <= params.kmax; ++k)
  {
    bool is_converged = true;

    for(std::size_t mode = 0; mode < dimension; ++mode)
    {
      const auto extent = tensor.sizes(static_cast<boba::index_t>(mode));
      host_matrix S({extent, static_cast<boba::index_t>(model.rank())});
      S.fill_with_zeros();

      host_matrix B({extent, static_cast<boba::index_t>(model.rank())});
      auto B_view = B.view();
      auto factor_view = model.m_cores[mode].const_view();
      auto lambda_view = model.m_weights.const_view();
      ::boba::loop<tensor_completion_space, 2>(
          {static_cast<std::size_t>(extent), static_cast<std::size_t>(B.cols())},
          [=]__boba_host_device__(::boba::Array<std::size_t, 2> ij)
      {
        const auto i = static_cast<boba::index_t>(ij[0]);
        const auto r = static_cast<boba::index_t>(ij[1]);
        B_view({i, r}) = factor_view({i, r}) * lambda_view(r);
      });

      compute_Phi_mode_n(tensor, mode, model, B, params.epsilon, Phi);

      if(k > 1)
      {
        auto S_view = S.view();
        auto Phi_view = Phi.const_view();
        ::boba::loop<tensor_completion_space, 2>(
            {static_cast<std::size_t>(extent), static_cast<std::size_t>(B.cols())},
            [=]__boba_host_device__(::boba::Array<std::size_t, 2> ij)
        {
          const auto i = static_cast<boba::index_t>(ij[0]);
          const auto r = static_cast<boba::index_t>(ij[1]);
          if(factor_view({i, r}) < params.kappa_tol && Phi_view({i, r}) > 1.0)
          {
            S_view({i, r}) = params.kappa;
          }
        });
      }

      auto S_view = S.const_view();
      ::boba::loop<tensor_completion_space, 2>(
          {static_cast<std::size_t>(extent), static_cast<std::size_t>(B.cols())},
          [=]__boba_host_device__(::boba::Array<std::size_t, 2> ij)
      {
        const auto i = static_cast<boba::index_t>(ij[0]);
        const auto r = static_cast<boba::index_t>(ij[1]);
        B_view({i, r}) = (factor_view({i, r}) + S_view({i, r})) * lambda_view(r);
      });

      for(int ell = 1; ell <= params.ellmax; ++ell)
      {
        compute_Phi_mode_n(tensor, mode, model, B, params.epsilon, Phi);

        const double kkt = inf_norm_mat_min_B_1minusPhi(B, Phi);
        if(kkt < params.tau)
        {
          break;
        }

        is_converged = false;
        auto Phi_view = Phi.const_view();
        ::boba::loop<tensor_completion_space, 2>(
            {static_cast<std::size_t>(extent), static_cast<std::size_t>(B.cols())},
            [=]__boba_host_device__(::boba::Array<std::size_t, 2> ij)
        {
          const auto i = static_cast<boba::index_t>(ij[0]);
          const auto r = static_cast<boba::index_t>(ij[1]);
          B_view({i, r}) *= Phi_view({i, r});
        });
      }

      model.m_weights = col_sums(B);
      unabsorb_and_normalize(B, model.m_weights, model.m_cores[mode]);
    }

    if(is_converged)
    {
      break;
    }
  }

  return model;
}

template<std::size_t dimension>
int run_tensor_completion(const CommandLineOptions &options)
{
  bool check = true;

  GaussianHistogramConfig generator_config;
  generator_config.dims = options.dims;
  generator_config.lower_bounds = std::vector<double>(generator_config.dims.size(), 0.0);
  generator_config.upper_bounds.reserve(generator_config.dims.size());
  for(int dim : generator_config.dims)
  {
    generator_config.upper_bounds.push_back(options.bin_width * static_cast<double>(dim));
  }
  generator_config.seed = 20260323;

  const sparse_tensor<dimension> observed_tensor =
    generate_gaussian_binned_tensor<dimension>(options.sample_count, generator_config);

  CPAPRParams params;
  params.kmax = 150;
  params.ellmax = 15;
  params.tau = 1.0e-5;
  params.kappa = 1.0e-2;
  params.kappa_tol = 1.0e-10;
  params.epsilon = 1.0e-10;
  params.seed = 20260323;

  auto model = initialize_random_stochastic<dimension>(observed_tensor, options.rank, params.seed);
  if(options.dense_apr)
  {
    const auto dense_observed_tensor = dense_tensor_from_sparse<dimension>(observed_tensor);
    model = cp_apr_fit(dense_observed_tensor, params, std::move(model));
  }
  else
  {
    model = cp_apr_fit(observed_tensor, params, std::move(model));
  }

  const double observed_tensor_entropy = sparse_tensor_entropy(observed_tensor);
  const double completed_tensor_entropy = tensor_entropy(model, observed_tensor.dims);

  boba_print(std::string("Observed tensor entropy (nats): ") + std::to_string(observed_tensor_entropy));
  boba_print(std::string("Completed tensor entropy (nats): ") + std::to_string(completed_tensor_entropy));

  pass_or_fail_bool(check, observed_tensor.dims.size() == dimension);
  pass_or_fail(check, total_mass(observed_tensor) - static_cast<double>(options.sample_count), 1.0e-12);
  pass_or_fail_bool(check, std::isfinite(observed_tensor_entropy));
  pass_or_fail_bool(check, observed_tensor_entropy >= 0.0);
  pass_or_fail_bool(check, std::isfinite(completed_tensor_entropy));
  pass_or_fail_bool(check, completed_tensor_entropy >= 0.0);
  pass_or_fail_bool(check, model_is_finite(model));

  if(options.verbose_bins)
  {
    boba_print(std::string("Observed tensor dims: ") + format_dims(observed_tensor.dims));
    boba_print(
      std::string("Bin widths: ") +
      format_values(std::vector<double>(observed_tensor.dims.size(), options.bin_width)));
    boba_print(std::string("Samples binned: ") + std::to_string(options.sample_count));
    boba_print(std::string("CP rank: ") + std::to_string(options.rank));
    boba_print(std::string("APR input path: ") + (options.dense_apr ? "dense" : "sparse COO"));
    boba_print(std::string("Total bins: ") + std::to_string(total_bins(observed_tensor.dims)));
    boba_print(std::string("Nonzero bins: ") + std::to_string(observed_tensor.values.size()));
    boba_print(std::string("Observed total mass: ") + std::to_string(total_mass(observed_tensor)));
    print_lambda(model);
    print_observed_counts(observed_tensor, model);
    print_empty_estimated_counts(observed_tensor, model);
  }

  return final_check(check);
}

int main(int argc, char **argv)
{
  boba::splash();
  boba::init();

  int return_code = 1;
  try
  {
    const CommandLineOptions options = parse_args(argc, argv);
    switch(options.dims.size())
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
  catch(const std::exception &ex)
  {
    boba_print(ex.what());
    print_usage(argv[0]);
    return_code = 1;
  }

  boba::finalize();
  return return_code;
}

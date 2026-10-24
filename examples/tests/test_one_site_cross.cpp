// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <set>

/*
  Tests OneSiteCross (compile-time number of cores) and RuntimeOneSiteCross: the two give the same train from the same
  seed; exact recovery of a low-rank tensor, accuracy on a smooth function that is not low rank
  (checked on the full tensor for d = 3, 4, 6 and on sampled entries for d = 10), and the evaluation cache (no
  multi-index evaluated twice, same result as without the cache).
*/

constexpr boba::execution_space host_space = boba::host_space;

// f(i) = 1 / (1 + sum_d (d + 1) x_d), x_d = i_d / (n_d - 1).
double smooth_function(std::vector<size_t> const& index, std::vector<size_t> const& sizes)
{
  double s = 1.0;
  for (size_t d = 0; d < sizes.size(); d++)
  {
    s += static_cast<double>(d + 1) * static_cast<double>(index[d]) / static_cast<double>(sizes[d] - 1);
  }
  return 1.0 / s;
}

// f(i) = sum_d i_d: TT rank 2.
double sum_function(std::vector<size_t> const& index, std::vector<size_t> const&)
{
  double s = 0.0;
  for (auto i : index)
  {
    s += static_cast<double>(i);
  }
  return s;
}

double train_entry(boba::RuntimeTensorTrain<host_space, double> const& train, std::vector<size_t> const& index)
{
  std::vector<std::vector<double>> unit(index.size());
  auto sizes = train.sizes();
  for (size_t d = 0; d < index.size(); d++)
  {
    unit[d].assign(static_cast<size_t>(sizes[d]), 0.0);
    unit[d][index[d]] = 1.0;
  }
  return boba::contract_with_vectors(train, unit);
}

// Max error over the whole grid (first mode fastest).
template <typename function_t>
double full_grid_error(boba::RuntimeTensorTrain<host_space, double> const& train, std::vector<size_t> const& sizes,
                       function_t const& function)
{
  std::vector<size_t> index(sizes.size(), 0);
  double error = 0.0;
  while (true)
  {
    error = std::max(error, std::abs(train_entry(train, index) - function(index, sizes)));
    size_t k = 0;
    while (k < sizes.size() and ++index[k] == sizes[k])
    {
      index[k] = 0;
      k++;
    }
    if (k == sizes.size())
    {
      return error;
    }
  }
}

template <typename function_t>
void check_full(bool& check, char const* name, size_t dimension, function_t const& function, double tolerance)
{
  std::vector<size_t> sizes(dimension);
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] = 5 + d % 3;
  }
  std::mt19937 generator(1234);
  boba::RuntimeOneSiteCross<double> cross;
  cross.tolerance = 1.0e-10;
  auto train = cross.apply(sizes, [&](std::vector<size_t> const& index) { return function(index, sizes); }, generator);
  const double error = full_grid_error(train, sizes, function);
  std::cout << name << " d = " << dimension << ": " << cross.info.iterations << " iterations, "
            << cross.info.evaluations << " evaluations, max rank " << train.max_rank() << ", validation error "
            << cross.info.validation_error << ", max error " << error << std::endl;
  pass_or_fail_bool(check, cross.info.converged);
  pass_or_fail(check, error, tolerance);
}

int main()
{
  boba::init();
  bool check = true;

  for (size_t dimension : {3_z, 4_z, 6_z})
  {
    check_full(check, "sum", dimension, sum_function, 1.0e-12);
    check_full(check, "smooth", dimension, smooth_function, 1.0e-8);
  }

  // Dimension 10 (6^10 entries): sampled check.
  {
    const size_t dimension = 10;
    std::vector<size_t> sizes(dimension, 6);
    std::mt19937 generator(5);
    boba::RuntimeOneSiteCross<double> cross;
    cross.tolerance = 1.0e-8;
    auto train = cross.apply(
      sizes, [&](std::vector<size_t> const& index) { return smooth_function(index, sizes); }, generator);
    std::mt19937 sampler(17);
    double error = 0.0;
    for (int sample = 0; sample < 500; sample++)
    {
      std::vector<size_t> index(dimension);
      for (auto& i : index)
      {
        i = sampler() % 6;
      }
      error = std::max(error, std::abs(train_entry(train, index) - smooth_function(index, sizes)));
    }
    std::cout << "smooth d = 10: " << cross.info.evaluations << " evaluations of 6^10 = 60466176, max rank "
              << train.max_rank() << ", max sampled error " << error << std::endl;
    pass_or_fail_bool(check, cross.info.converged);
    pass_or_fail(check, error, 1.0e-6);
  }

  // OneSiteCross<d> and RuntimeOneSiteCross: same seed, same train.
  {
    constexpr size_t dimension = 4;
    const std::vector<size_t> sizes{6, 7, 5, 6};
    boba::Array<boba::index_t, dimension> static_sizes{6, 7, 5, 6};
    std::mt19937 generator_static(21), generator_runtime(21);
    boba::OneSiteCross<double> static_cross;
    boba::RuntimeOneSiteCross<double> runtime_cross;
    auto static_train = static_cross.apply(
      static_sizes,
      [&](boba::Array<size_t, dimension> const& index)
      {
        return smooth_function(std::vector<size_t>(index.begin(), index.end()), sizes);
      },
      generator_static);
    auto runtime_train = runtime_cross.apply(
      sizes, [&](std::vector<size_t> const& index) { return smooth_function(index, sizes); }, generator_runtime);
    const double difference =
      boba::norm_difference_inf(runtime_train.to_static<dimension>().decompress(), static_train.decompress());
    std::cout << "static vs runtime: " << static_cross.info.evaluations << " and " << runtime_cross.info.evaluations
              << " evaluations, max difference " << difference << std::endl;
    pass_or_fail_bool(check, static_cross.info.evaluations == runtime_cross.info.evaluations);
    pass_or_fail(check, difference, 0.0 + std::numeric_limits<double>::min());
  }

  // Evaluation cache: each multi-index evaluated at most once, and the same train as without the cache.
  {
    const std::vector<size_t> sizes{6, 7, 5, 6};
    std::set<std::vector<size_t>> seen;
    bool repeated = false;
    auto counting = [&](std::vector<size_t> const& index)
    {
      repeated = repeated or not seen.insert(index).second;
      return smooth_function(index, sizes);
    };
    auto plain = [&](std::vector<size_t> const& index) { return smooth_function(index, sizes); };

    std::mt19937 generator_cached(99), generator_plain(99);
    boba::RuntimeOneSiteCross<double> cached, uncached;
    cached.cache_evaluations = true;
    auto train_cached = cached.apply(sizes, counting, generator_cached);
    auto train_plain = uncached.apply(sizes, plain, generator_plain);
    std::cout << "cache: " << cached.info.evaluations << " evaluations with cache, " << uncached.info.evaluations
              << " without" << std::endl;
    pass_or_fail_bool(check, not repeated);
    pass_or_fail_bool(check, cached.info.evaluations == seen.size());
    pass_or_fail_bool(check, cached.info.evaluations < uncached.info.evaluations);
    pass_or_fail(check, full_grid_error(train_cached, sizes, [&](auto const& i, auto const&) { return train_entry(train_plain, i); }),
                 1.0e-14);
  }

  boba::finalize();
  return final_check(check);
}

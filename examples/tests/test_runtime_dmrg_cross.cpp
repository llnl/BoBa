// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Tests RuntimeDMRGCross (number of cores chosen at run time) against DMRGCross<dimension>: from the same initial
  guess, function and random seed both must return the same tensor train, and both must approximate the function.
  A train with more cores than any compiled-in dimension is then checked on sampled entries.
*/

constexpr boba::execution_space host_space = boba::host_space;

static double error_tolerance = 1.0e-10;

// f(i) = 1 / (1 + sum_d (d + 1) x_d), x_d = i_d / (n_d - 1): smooth, not low rank, positive.
template <typename index_container_t>
double test_function(index_container_t const& index, std::vector<boba::index_t> const& sizes)
{
  double s = 1.0;
  for (size_t d = 0; d < sizes.size(); d++)
  {
    s += static_cast<double>(d + 1) * static_cast<double>(index[d]) / static_cast<double>(sizes[d] - 1);
  }
  return 1.0 / s;
}

// DMRGCross<dimension>::apply cannot take a function-local lambda under nvcc (it is templated on the callable and
// contains extended __host__ __device__ lambdas), so the reference uses a namespace-scope functor.
template <size_t dimension>
struct StaticTestFunction
{
  std::vector<boba::index_t> sizes;
  double operator()(boba::Array<size_t, dimension> const& index) const
  {
    return test_function(index, sizes);
  }
};

template <size_t dimension>
void compare(bool& check)
{
  boba::Array<boba::index_t, dimension> sizes;
  std::vector<boba::index_t> runtime_sizes(dimension);
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] = 5 + d;
    runtime_sizes[d] = sizes[d];
  }

  std::mt19937 guess_generator(3);
  boba::TensorTrain<dimension, host_space, double> guess(sizes);
  for (size_t d = 0; d < dimension; d++)
  {
    const size_t rl = (d == 0) ? 1 : 2;
    const size_t rr = (d == dimension - 1) ? 1 : 2;
    guess.cores[d].resize({rl, sizes[d], rr});
    guess.cores[d].fill_with_random(-1.0, 1.0, guess_generator);
  }
  boba::RuntimeTensorTrain<host_space, double> runtime_guess(guess);

  StaticTestFunction<dimension> static_function{runtime_sizes};
  auto runtime_function = [=](std::vector<size_t> const& index)
  {
    return test_function(index, runtime_sizes);
  };

  std::mt19937 static_generator(11);
  std::mt19937 runtime_generator(11);
  boba::DMRGCross<double> static_cross;
  boba::RuntimeDMRGCross<double> runtime_cross;
  auto static_result = static_cross.apply(guess, static_function, static_generator);
  auto runtime_result = runtime_cross.apply(runtime_guess, runtime_function, runtime_generator);

  // Same algorithm, same inputs, same random stream: the same train.
  pass_or_fail_bool(check, runtime_result.get_dimension() == dimension);
  bool same_ranks = true;
  for (size_t d = 0; d <= dimension; d++)
  {
    same_ranks = same_ranks and (runtime_result.ranks()[d] == static_result.ranks(d));
  }
  pass_or_fail_bool(check, same_ranks);
  auto static_full = static_result.decompress();
  auto runtime_full = runtime_result.template to_static<dimension>().decompress();
  pass_or_fail(check, boba::norm_difference_inf(runtime_full, static_full), error_tolerance);

  // And it approximates the function.
  auto view = runtime_full.const_view();
  double error = 0.0;
  for (size_t k = 0; k < view.size(); k++)
  {
    error = std::max(error, std::abs(view(k) - static_function(view.multiindex(k))));
  }
  std::cout << "dimension " << dimension << ": max rank " << runtime_result.max_rank() << ", max error " << error
            << std::endl;
  pass_or_fail(check, error, 1.0e-6);
}

int main(int argc, char** argv)
{
  boba::init();
  bool check = true;

  const size_t only = (argc > 1) ? static_cast<size_t>(std::atoi(argv[1])) : 0_z;
  if (only == 0 or only == 2)
  {
    compare<2>(check);
  }
  if (only == 0 or only == 3)
  {
    compare<3>(check);
  }
  if (only == 0 or only == 5)
  {
    compare<5>(check);
  }

  // Dimension known only at run time: check sampled entries against the function.
  {
    const size_t dimension = (argc > 2) ? static_cast<size_t>(std::atoi(argv[2])) : 10_z;
    std::vector<boba::index_t> sizes(dimension, 6_z);
    boba::RuntimeTensorTrain<host_space, double> guess;
    std::mt19937 generator(5);
    for (size_t d = 0; d < dimension; d++)
    {
      const size_t rl = (d == 0) ? 1 : 2;
      const size_t rr = (d == dimension - 1) ? 1 : 2;
      guess.cores.push_back(boba::Tensor<3, host_space, double>({rl, sizes[d], rr}));
      guess.cores[d].fill_with_random(-1.0, 1.0, generator);
    }
    size_t evaluations = 0;
    auto function = [&](std::vector<size_t> const& index)
    {
      evaluations++;
      return test_function(index, sizes);
    };
    boba::RuntimeDMRGCross<double> cross;
    auto result = cross.apply(guess, function, generator);

    // Evaluate the train at random multi-indices by contracting with unit vectors.
    std::uniform_int_distribution<size_t> pick(0, 5);
    double error = 0.0;
    for (int sample = 0; sample < 200; sample++)
    {
      std::vector<size_t> index(dimension);
      std::vector<std::vector<double>> unit(dimension, std::vector<double>(6, 0.0));
      for (size_t d = 0; d < dimension; d++)
      {
        index[d] = pick(generator);
        unit[d][index[d]] = 1.0;
      }
      error = std::max(error, std::abs(boba::contract_with_vectors(result, unit) - test_function(index, sizes)));
    }
    std::cout << "dimension " << dimension << " (runtime only): max rank " << result.max_rank() << ", "
              << evaluations << " evaluations of " << std::pow(6.0, static_cast<double>(dimension))
              << " entries, max sampled error " << error << std::endl;
    pass_or_fail(check, error, 1.0e-6);
  }

  boba::finalize();
  return final_check(check);
}

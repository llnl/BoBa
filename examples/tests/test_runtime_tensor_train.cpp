// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Tests RuntimeTensorTrain (number of cores chosen at run time) against TensorTrain<dimension> on the same cores:
  addition, scaling, subtraction, elementwise product, rounding, weighted contraction, and conversion.
*/

constexpr boba::execution_space space = boba::default_execution_space;

static double error_tolerance = 1.0e-10;

template <size_t dimension>
void compare(bool& check)
{
  std::mt19937 generator(7);
  boba::Array<boba::index_t, dimension> sizes;
  std::vector<boba::index_t> runtime_sizes(dimension);
  std::vector<size_t> ranks(dimension + 1, 3_z);
  ranks[0] = 1;
  ranks[dimension] = 1;
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] = 4 + d;
    runtime_sizes[d] = sizes[d];
  }

  // Two random rank-3 trains with identical cores in both representations.
  auto make_static = [&]()
  {
    boba::TensorTrain<dimension, space, double> train(sizes);
    for (size_t d = 0; d < dimension; d++)
    {
      train.cores[d].resize({ranks[d], sizes[d], ranks[d + 1]});
      train.cores[d].fill_with_random(-1.0, 1.0, generator);
    }
    return train;
  };
  auto A = make_static();
  auto B = make_static();
  boba::RuntimeTensorTrain<space, double> rA(A);
  boba::RuntimeTensorTrain<space, double> rB(B);

  pass_or_fail_bool(check, rA.get_dimension() == dimension);
  pass_or_fail_bool(check, rA.sizes() == runtime_sizes);

  // Exact arithmetic: compare the decompressed tensors.
  auto sum = A + B;
  auto r_sum = rA + rB;
  pass_or_fail(check, boba::norm_difference_inf(r_sum.template to_static<dimension>().decompress(), sum.decompress()),
               error_tolerance);

  auto difference = A - B * 2.0;
  auto r_difference = rA - rB * 2.0;
  pass_or_fail(check,
               boba::norm_difference_inf(r_difference.template to_static<dimension>().decompress(), difference.decompress()),
               error_tolerance);

  auto product = boba::elementwise_product(A, B);
  auto r_product = boba::elementwise_product(rA, rB);
  pass_or_fail(check,
               boba::norm_difference_inf(r_product.template to_static<dimension>().decompress(), product.decompress()),
               error_tolerance);

  // Rounding: same algorithm and tolerances, so the same ranks and the same tensor.
  auto rounded = sum + sum;
  rounded.round();
  auto r_rounded = r_sum + r_sum;
  r_rounded.round();
  bool same_ranks = true;
  for (size_t d = 0; d <= dimension; d++)
  {
    same_ranks = same_ranks and (r_rounded.ranks()[d] == rounded.ranks(d));
  }
  pass_or_fail_bool(check, same_ranks);
  pass_or_fail(check,
               boba::norm_difference_inf(r_rounded.template to_static<dimension>().decompress(), rounded.decompress()),
               error_tolerance);

  // Weighted contraction: all-ones weights sum every entry of the tensor.
  std::vector<std::vector<double>> ones(dimension);
  for (size_t d = 0; d < dimension; d++)
  {
    ones[d].assign(sizes[d], 1.0);
  }
  auto full = sum.decompress();
  pass_or_fail(check, std::abs(boba::contract_with_vectors(r_sum, ones) - full.sum_reduce()), error_tolerance * full.size());
}

int main(int argc, char** argv)
{
  boba::init();
  bool check = true;

  // The dimension is chosen at run time (default: all of 2..6).
  const size_t only = (argc > 1) ? static_cast<size_t>(std::atoi(argv[1])) : 0_z;
  if (only == 0 or only == 2)
  {
    compare<2>(check);
  }
  if (only == 0 or only == 3)
  {
    compare<3>(check);
  }
  if (only == 0 or only == 4)
  {
    compare<4>(check);
  }
  if (only == 0 or only == 6)
  {
    compare<6>(check);
  }

  // A train built purely at run time: dimension from the command line, rank grows under sums, round recovers it.
  {
    const size_t dimension = (argc > 2) ? static_cast<size_t>(std::atoi(argv[2])) : 9_z;
    std::vector<boba::index_t> sizes(dimension, 5_z);
    boba::RuntimeTensorTrain<space, double> ones(sizes);
    ones.fill_with(1.0);
    auto many = ones;
    for (int k = 0; k < 7; k++)
    {
      many += ones;
    }
    pass_or_fail_bool(check, many.max_rank() == 8_z);
    many.round();
    pass_or_fail_bool(check, many.max_rank() == 1_z);
    std::vector<std::vector<double>> w(dimension, std::vector<double>(5, 0.2));
    pass_or_fail(check, std::abs(boba::contract_with_vectors(many, w) - 8.0), error_tolerance);
  }

  // round_frobenius: ||A - round(A)||_F <= eps ||A||_F, with the minimal ranks.
  // A = sum_k s_k a_k (x) b_k (x) c_k with orthonormal factors has TT-SVD spectrum s at both splits.
  {
    const size_t n = 6;
    const std::vector<double> s{1.0, 1.0e-2, 1.0e-4, 1.0e-6};
    const size_t rank = s.size();
    std::mt19937 generator(13);
    // Orthonormal columns: Q of a random n x n matrix.
    auto orthonormal = [&]()
    {
      boba::Matrix<space, double> m({n, n});
      m.fill_with_random(-1.0, 1.0, generator);
      boba::QR<space, double> qr;
      qr(m);
      return boba::Matrix<boba::host_space, double>(qr.Q);
    };
    const auto a = orthonormal();
    const auto b = orthonormal();
    const auto c = orthonormal();
    auto av = a.const_view();
    auto bv = b.const_view();
    auto cv = c.const_view();
    boba::Tensor<3, boba::host_space, double> c0({1, n, rank}), c1({rank, n, rank}), c2({rank, n, 1});
    c0.fill_with_zeros();
    c1.fill_with_zeros();
    c2.fill_with_zeros();
    auto v0 = c0.view();
    auto v1 = c1.view();
    auto v2 = c2.view();
    for (size_t k = 0; k < rank; k++)
    {
      for (size_t i = 0; i < n; i++)
      {
        v0({0, i, k}) = s[k] * av({i, k});
        v1({k, i, k}) = bv({i, k});
        v2({k, i, 0}) = cv({i, k});
      }
    }
    boba::RuntimeTensorTrain<space, double> A;
    A.cores.push_back(boba::Tensor<3, space, double>(c0));
    A.cores.push_back(boba::Tensor<3, space, double>(c1));
    A.cores.push_back(boba::Tensor<3, space, double>(c2));
    const auto full = A.to_static<3>().decompress();
    const double norm = boba::norm_frobenius(full);

    // delta = eps ||A||_F / sqrt(2); keep the fewest terms whose dropped tail energy is <= delta^2.
    // eps = 1e-3: delta^2 ~ 5e-7 >= 1e-8 + 1e-12 -> rank 2. eps = 1e-5: delta^2 ~ 5e-11, drops only 1e-6 -> 3.
    for (auto [eps, expected_rank] : {std::pair{1.0e-3, 2_z}, std::pair{1.0e-5, 3_z}, std::pair{1.0e-9, 4_z}})
    {
      auto R = A;
      R.round_frobenius(eps);
      const double error = boba::norm_difference_frobenius(R.to_static<3>().decompress(), full) / norm;
      std::cout << "round_frobenius eps " << eps << ": rank " << R.max_rank() << ", relative error " << error
                << std::endl;
      pass_or_fail_bool(check, R.max_rank() == expected_rank);
      pass_or_fail(check, error, eps);
    }
  }

  boba::finalize();
  return final_check(check);
}

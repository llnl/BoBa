// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

constexpr boba::execution_space space = boba::default_execution_space;

template <size_t dimension>
void test_linear_prefix_sum(bool& check)
{
  using cpd_t = boba::CanonicalPolyadicDecomposition<dimension, space, double>;

  constexpr boba::index_t rank = 3;
  boba::Array<boba::index_t, dimension> sizes;
  for (size_t d = 0; d < dimension; ++d)
  {
    sizes[d] = d + 2;
  }

  cpd_t input(sizes);
  input.rename("linear_prefix_sum_input");
  input.m_weights.resize(rank);
  input.m_weights.fill_with_random();

  for (size_t d = 0; d < dimension; ++d)
  {
    input.m_cores[d].resize({sizes[d], rank});
    input.m_cores[d].fill_with_random();
  }

  const auto tensor_result = boba::linear_prefix_sum(input.decompress());
  const auto cpd_result = boba::linear_prefix_sum(input);
  const auto decompressed_cpd_result = cpd_result.decompress();

  const auto error = boba::norm_difference_frobenius(tensor_result, decompressed_cpd_result);
  const auto relative_error = error / boba::norm_frobenius(tensor_result);

  pass_or_fail_bool(check, cpd_result.rank() == dimension * rank);
  pass_or_fail(check, relative_error, 1.0e-12);
}

int main()
{
  boba::splash();
  boba::init();

  bool check = true;

  test_linear_prefix_sum<1>(check);
  test_linear_prefix_sum<2>(check);
  test_linear_prefix_sum<3>(check);
  test_linear_prefix_sum<4>(check);
  test_linear_prefix_sum<5>(check);

  boba::finalize();
  return final_check(check);
}

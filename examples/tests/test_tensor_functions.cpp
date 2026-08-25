// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Test various tensor utility functions
*/

#include <array>

constexpr boba::execution_space space = boba::default_execution_space;

template <typename object_type>
void compare_square_and_product(bool& check)
{
  auto sizes = boba::filled_array<object_type::get_dimension()>(10_z);
  object_type A(sizes);
  A.fill_with_random();

  auto Apow2 = boba::power(A, 2.0);
  auto Aprod2 = boba::elementwise_product(A, A);

  auto error = boba::norm_difference_inf(Apow2, Aprod2);

  pass_or_fail(check, error, 1.0e-13);
}

template <typename object_type>
void test_tensor_product_identity(bool& check)
{
  auto sizes = boba::filled_array<object_type::get_dimension()>(10_z);
  object_type A(sizes);
  A.fill_with_random();
  object_type B(sizes);
  B.fill_with_random();
  object_type C(sizes);
  C.fill_with_random();
  object_type D(sizes);
  D.fill_with_random();

  // (A x B) dot (C x D) = (A dot C) x (B dot D)

  // Method 1
  auto AxB = boba::tensor_product(A, B);
  auto CxD = boba::tensor_product(C, D);
  auto AxB_dot_CxD = boba::inner_product(AxB, CxD);

  // Method 2
  auto A_dot_C = boba::inner_product(A, C);
  auto B_dot_D = boba::inner_product(B, D);
  auto A_dot_C_x_B_dot_D = A_dot_C * B_dot_D;

  auto error = boba::abs(A_dot_C_x_B_dot_D - AxB_dot_CxD);

  pass_or_fail(check, error, 1.0e-8);
}

template <typename object_type>
void test_positive_negative_parts(bool& check)
{
  auto sizes = boba::filled_array<object_type::get_dimension()>(4_z);
  object_type input(sizes);

  auto input_view = input.view();
  ::boba::detail::loop<space>(0_z, input.size(), [=] __boba_host_device__(size_t i)
  {
    input_view(i) = static_cast<typename object_type::data_t>(i) - 2.5;
  });

  auto member_positive = input.nonnegative_part();
  auto member_negative = input.nonpositive_part();
  auto free_positive = boba::nonnegative_part(input);
  auto free_negative = boba::nonpositive_part(input);

  pass_or_fail(check, boba::norm_difference_inf(member_positive, free_positive), 1.0e-13);
  pass_or_fail(check, boba::norm_difference_inf(member_negative, free_negative), 1.0e-13);

  auto input_host = ::boba::Tensor<object_type::get_dimension(), boba::execution_space::CPU, typename object_type::data_t>(input);
  auto positive_host = ::boba::Tensor<object_type::get_dimension(), boba::execution_space::CPU, typename object_type::data_t>(member_positive);
  auto negative_host = ::boba::Tensor<object_type::get_dimension(), boba::execution_space::CPU, typename object_type::data_t>(member_negative);

  auto input_const_view = input_host.const_view();
  auto positive_view = positive_host.const_view();
  auto negative_view = negative_host.const_view();

  bool local_check = true;
  for (size_t i = 0; i < input_host.size(); ++i)
  {
    auto x = input_const_view(i);
    auto expected_positive = ::boba::positive_part(x);
    auto expected_negative = ::boba::positive_part(-x);
    local_check = local_check && (::boba::abs(positive_view(i) - expected_positive) < 1.0e-13);
    local_check = local_check && (::boba::abs(negative_view(i) - expected_negative) < 1.0e-13);
  }

  pass_or_fail_bool(check, local_check);
}

void test_tensor_cumulative_sum(bool& check)
{
  using tensor_t = boba::Tensor<2, space, double>;
  using host_tensor_t = boba::Tensor<2, boba::execution_space::CPU, double>;

  host_tensor_t pdf_host({3_z, 4_z});
  pdf_host.fill_with_zeros();
  pdf_host({0_z, 0_z}) = 1.0;
  pdf_host({1_z, 0_z}) = 2.0;
  pdf_host({2_z, 0_z}) = 3.0;
  pdf_host({0_z, 1_z}) = 4.0;
  pdf_host({1_z, 1_z}) = 5.0;
  pdf_host({2_z, 1_z}) = 6.0;
  pdf_host({0_z, 2_z}) = 7.0;
  pdf_host({1_z, 2_z}) = 8.0;
  pdf_host({2_z, 2_z}) = 9.0;
  pdf_host({0_z, 3_z}) = 10.0;
  pdf_host({1_z, 3_z}) = 11.0;
  pdf_host({2_z, 3_z}) = 12.0;

  tensor_t pdf(pdf_host);
  const tensor_t cdf = boba::cumulative_sum(pdf);
  const host_tensor_t cdf_host(cdf);

  bool local_check = true;
  for (boba::index_t i = 0; i < pdf_host.sizes(0); ++i)
  {
    for (boba::index_t j = 0; j < pdf_host.sizes(1); ++j)
    {
      double expected = 0.0;
      for (boba::index_t ii = 0; ii <= i; ++ii)
      {
        for (boba::index_t jj = 0; jj <= j; ++jj)
        {
          expected += pdf_host({ii, jj});
        }
      }
      local_check = local_check && (boba::abs(cdf_host({i, j}) - expected) < 1.0e-13);
    }
  }

  pass_or_fail_bool(check, local_check);
}

void test_cpd_cumulative_sum_matches_tensor(bool& check)
{
  using tensor_t = boba::Tensor<2, space, double>;
  using host_tensor_t = boba::Tensor<2, boba::execution_space::CPU, double>;
  using cpd_t = boba::CanonicalPolyadicDecomposition<2, space, double>;
  using host_cpd_t = boba::CanonicalPolyadicDecomposition<2, boba::execution_space::CPU, double>;

  constexpr boba::index_t m = 5;
  constexpr boba::index_t n = 4;

  const auto sizes = boba::Array<boba::index_t, 2>{m, n};

  host_tensor_t pdf_host({m, n});
  pdf_host.fill_with_zeros();

  std::array<double, static_cast<std::size_t>(m)> a = {0.1, 0.2, 0.3, 0.4, 0.5};
  std::array<double, static_cast<std::size_t>(n)> b = {0.25, 0.5, 0.75, 1.0};

  for (boba::index_t i = 0; i < m; ++i)
  {
    for (boba::index_t j = 0; j < n; ++j)
    {
      pdf_host({i, j}) = a[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(j)];
    }
  }

  const tensor_t pdf_tensor(pdf_host);
  const tensor_t cdf_tensor = boba::cumulative_sum(pdf_tensor);
  const host_tensor_t cdf_tensor_host(cdf_tensor);

  cpd_t pdf_cpd(sizes);
  pdf_cpd.rename("pdf_cpd_rank1");
  pdf_cpd.m_weights.resize(static_cast<boba::index_t>(1));
  pdf_cpd.m_weights.fill_with(1.0);
  pdf_cpd.m_cores[0].resize({m, 1});
  pdf_cpd.m_cores[1].resize({n, 1});

  auto core0_view = pdf_cpd.m_cores[0].view();
  auto core1_view = pdf_cpd.m_cores[1].view();

  ::boba::detail::loop<space>(0_z, static_cast<size_t>(m), [=] __boba_host_device__(size_t i)
  {
    core0_view({static_cast<boba::index_t>(i), 0}) = a[i];
  });
  ::boba::detail::loop<space>(0_z, static_cast<size_t>(n), [=] __boba_host_device__(size_t j)
  {
    core1_view({static_cast<boba::index_t>(j), 0}) = b[j];
  });

  const cpd_t cdf_cpd = boba::cumulative_sum(pdf_cpd);
  const host_cpd_t cdf_cpd_host(cdf_cpd);

  const auto weights_view = cdf_cpd_host.weights().const_view();
  const auto cores = cdf_cpd_host.get_core_const_views();

  bool local_check = true;
  for (boba::index_t i = 0; i < m; ++i)
  {
    for (boba::index_t j = 0; j < n; ++j)
    {
      double estimate = 0.0;
      for (boba::index_t r = 0; r < cdf_cpd_host.weights().size(); ++r)
      {
        estimate += weights_view(r) * cores[0]({i, r}) * cores[1]({j, r});
      }

      local_check = local_check && (boba::abs(estimate - cdf_tensor_host({i, j})) < 1.0e-13);
    }
  }

  pass_or_fail_bool(check, local_check);
}

void test_tucker_cumulative_sum_matches_tensor(bool& check)
{
  using tensor_t = boba::Tensor<3, space, double>;
  using tucker_t = boba::Tucker<3, space, double>;

  constexpr boba::index_t n0 = 4;
  constexpr boba::index_t n1 = 3;
  constexpr boba::index_t n2 = 5;

  constexpr boba::index_t r0 = 2;
  constexpr boba::index_t r1 = 3;
  constexpr boba::index_t r2 = 2;

  const auto sizes = boba::Array<boba::index_t, 3>{n0, n1, n2};

  tucker_t pdf_tucker(sizes);
  pdf_tucker.rename("pdf_tucker");
  pdf_tucker.cores[0].resize({n0, r0});
  pdf_tucker.cores[1].resize({n1, r1});
  pdf_tucker.cores[2].resize({n2, r2});
  pdf_tucker.R_core.resize({r0, r1, r2});

  for (size_t d = 0; d < 3; ++d)
  {
    auto U_view = pdf_tucker.cores[d].view();
    const auto rows = pdf_tucker.cores[d].rows();
    const auto cols = pdf_tucker.cores[d].cols();
    ::boba::detail::loop<space>(0_z, static_cast<size_t>(rows * cols), [=] __boba_host_device__(size_t flat)
    {
      const auto i = static_cast<boba::index_t>(flat % static_cast<size_t>(rows));
      const auto j = static_cast<boba::index_t>(flat / static_cast<size_t>(rows));
      U_view({i, j}) = 0.1 + 0.01 * static_cast<double>(i) + 0.02 * static_cast<double>(j) + 0.001 * static_cast<double>(d);
    });
  }

  {
    auto G_view = pdf_tucker.R_core.view();
    ::boba::detail::loop<space>(0_z, static_cast<size_t>(pdf_tucker.R_core.size()), [=] __boba_host_device__(size_t lin)
    {
      G_view(static_cast<boba::index_t>(lin)) = 0.05 + 0.002 * static_cast<double>(lin);
    });
  }

  const tensor_t pdf_dense = pdf_tucker.decompress();
  const tensor_t cdf_dense = boba::cumulative_sum(pdf_dense);
  const tensor_t cdf_tucker_dense = boba::cumulative_sum(pdf_tucker).decompress();

  pass_or_fail(check, boba::norm_difference_inf(cdf_dense, cdf_tucker_dense), 1.0e-12);
}

void test_tensortrain_cumulative_sum_matches_tensor(bool& check)
{
  using tensor_t = boba::Tensor<3, space, double>;
  using tt_t = boba::TensorTrain<3, space, double>;

  constexpr boba::index_t n0 = 4;
  constexpr boba::index_t n1 = 3;
  constexpr boba::index_t n2 = 5;

  constexpr boba::index_t r1 = 2;
  constexpr boba::index_t r2 = 3;

  const auto sizes = boba::Array<boba::index_t, 3>{n0, n1, n2};

  tt_t pdf_tt(sizes);
  pdf_tt.rename("pdf_tt");
  pdf_tt.cores[0].resize({1, n0, r1});
  pdf_tt.cores[1].resize({r1, n1, r2});
  pdf_tt.cores[2].resize({r2, n2, 1});

  for (size_t d = 0; d < 3; ++d)
  {
    auto core_view = pdf_tt.cores[d].view();
    ::boba::detail::loop<space>(0_z, static_cast<size_t>(pdf_tt.cores[d].size()), [=] __boba_host_device__(size_t lin)
    {
      core_view(static_cast<boba::index_t>(lin)) = 0.1 + 0.001 * static_cast<double>(lin) + 0.01 * static_cast<double>(d);
    });
  }

  const tensor_t pdf_dense = pdf_tt.decompress();
  const tensor_t cdf_dense = boba::cumulative_sum(pdf_dense);
  const tensor_t cdf_tt_dense = boba::cumulative_sum(pdf_tt).decompress();

  pass_or_fail(check, boba::norm_difference_inf(cdf_dense, cdf_tt_dense), 1.0e-12);
}

void test_htucker_cumulative_sum_matches_tensor(bool& check)
{
  constexpr size_t ht_dimension = 4;

  using tensor_t = boba::Tensor<ht_dimension, space, double>;
  using ht_t = boba::HierarchicalTucker<ht_dimension, space, double>;

  auto dim_tree = boba::DimensionTree(boba::BalancedTreeBuilder(ht_dimension));
  const auto sizes = boba::Array<boba::index_t, ht_dimension>{3, 4, 2, 3};

  ht_t pdf_ht(sizes, dim_tree);
  pdf_ht.set_name("pdf_ht_rank1");
  pdf_ht.fill_with(1.0);

  {
    const auto set_leaf = [&](size_t d)
    {
      const size_t node = dim_tree.get_dim2idx_of_dim(d);
      auto U = pdf_ht.get_basis_matrix(node);
      auto U_view = U.view();
      const auto rows = U.rows();
      ::boba::detail::loop<space>(0_z, static_cast<size_t>(rows), [=] __boba_host_device__(size_t i)
      {
        const double ii = static_cast<double>(i);
        U_view({static_cast<boba::index_t>(i), 0}) = 0.1 + 0.03 * static_cast<double>(d) + 0.02 * ii;
      });
      pdf_ht.set_basis_matrix(node, std::move(U));
    };

    set_leaf(0);
    set_leaf(1);
    set_leaf(2);
    set_leaf(3);
  }

  const tensor_t pdf_dense = pdf_ht.decompress();
  const tensor_t cdf_dense = boba::cumulative_sum(pdf_dense);
  const tensor_t cdf_ht_dense = boba::cumulative_sum(pdf_ht).decompress();

  pass_or_fail(check, boba::norm_difference_inf(cdf_dense, cdf_ht_dense), 1.0e-12);
}

int main(int argc, char* argv[])
{
  boba::detail::ignore(argc);
  boba::detail::ignore(argv);

  boba::splash();
  boba::init();

  bool check = true;

  checkpoint();
  compare_square_and_product<boba::Matrix<space, double>>(check);
  checkpoint();
  compare_square_and_product<boba::Vector<space, double>>(check);
  checkpoint();
  compare_square_and_product<boba::Tensor<3, space, double>>(check);
  checkpoint();
  test_positive_negative_parts<boba::Matrix<space, double>>(check);
  checkpoint();
  test_positive_negative_parts<boba::Vector<space, double>>(check);
  checkpoint();
  test_positive_negative_parts<boba::Tensor<3, space, double>>(check);
  checkpoint();
  test_tensor_product_identity<boba::Matrix<space, double>>(check);
  checkpoint();
  test_tensor_product_identity<boba::Vector<space, double>>(check);
  checkpoint();
  test_tensor_product_identity<boba::Tensor<3, space, double>>(check);
  checkpoint();
  test_tensor_cumulative_sum(check);
  checkpoint();
  test_cpd_cumulative_sum_matches_tensor(check);
  checkpoint();
  test_tucker_cumulative_sum_matches_tensor(check);
  checkpoint();
  test_tensortrain_cumulative_sum_matches_tensor(check);
  checkpoint();
  test_htucker_cumulative_sum_matches_tensor(check);
  checkpoint();
  boba::finalize();

  return final_check(check);
}

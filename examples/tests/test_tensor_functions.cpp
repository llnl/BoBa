// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Test various tensor utility functions
*/

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
  boba::finalize();

  return final_check(check);
}

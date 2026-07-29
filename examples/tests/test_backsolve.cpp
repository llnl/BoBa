// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "common.hpp"

/*
  Tests solving a linear system of equations in a portable way.
*/

constexpr boba::execution_space space = boba::default_execution_space;

namespace
{
template <typename data_t>
boba::Matrix<boba::host_space, data_t> make_host_matrix(
  std::initializer_list<std::initializer_list<data_t>> rows)
{
  auto row_count = rows.size();
  auto col_count = rows.begin()->size();
  boba::Matrix<boba::host_space, data_t> output({row_count, col_count});
  output.fill_with_zeros();

  size_t r = 0;
  for (auto const& row : rows)
  {
    size_t c = 0;
    for (auto const& value : row)
    {
      output({r, c}) = value;
      c++;
    }
    r++;
  }

  return output;
}

template <typename data_t>
boba::Vector<boba::host_space, data_t> make_host_vector(
  std::initializer_list<data_t> values)
{
  boba::Vector<boba::host_space, data_t> output({values.size()});
  auto output_view = output.view();

  size_t i = 0;
  for (auto const& value : values)
  {
    output_view(i) = value;
    i++;
  }

  return output;
}

template <typename data_t>
void test_square_backsolve_matrix(bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;
  constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

  std::cout << "square backsolve, matrix rhs" << std::endl;

  auto A_host = make_host_matrix<data_t>({
    {3, 1, 0},
    {0, 2, 1},
    {0, 0, 4},
  });
  auto X_expected_host = make_host_matrix<data_t>({
    {1, 2},
    {3, 4},
    {5, 6},
  });
  auto rhs_host = A_host * X_expected_host;

  boba::Matrix<space, data_t> A = A_host;
  boba::Matrix<space, data_t> rhs = rhs_host;
  boba::Matrix<space, data_t> X_expected = X_expected_host;

  auto X = boba::backsolve(A, rhs);

  pass_or_fail(check, boba::norm_difference_inf(X, X_expected), tolerance);
  pass_or_fail(check, boba::norm_difference_inf(A * X, rhs), tolerance);
}

template <typename data_t>
void test_square_backsolve_vector(bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;
  constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

  std::cout << "square backsolve, vector rhs" << std::endl;

  auto A_host = make_host_matrix<data_t>({
    {3, 1, 0},
    {0, 2, 1},
    {0, 0, 4},
  });
  auto x_expected_host = make_host_vector<data_t>({1, 2, 3});
  auto rhs_host = flatten(A_host * reshape_to_matrix(x_expected_host, {x_expected_host.size(), 1}));

  const boba::Matrix<space, data_t> A = A_host;
  const boba::Vector<space, data_t> rhs = rhs_host;
  const boba::Vector<space, data_t> x_expected = x_expected_host;

  auto x = boba::backsolve(A, rhs);

  pass_or_fail(check, boba::norm_difference_inf(x, x_expected), tolerance);
  pass_or_fail(check, boba::norm_difference_inf(A * x, rhs), tolerance);
}

template <typename data_t>
void test_tall_backsolve_matrix(bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;
  constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

  std::cout << "tall backsolve, matrix rhs" << std::endl;

  auto A_host = make_host_matrix<data_t>({
    {1, 0},
    {0, 1},
    {1, 1},
  });
  auto X_expected_host = make_host_matrix<data_t>({
    {2, 3},
    {4, 5},
  });
  auto rhs_host = A_host * X_expected_host;

  const boba::Matrix<space, data_t> A = A_host;
  const boba::Matrix<space, data_t> rhs = rhs_host;
  const boba::Matrix<space, data_t> X_expected = X_expected_host;

  auto X = boba::backsolve(A, rhs);

  pass_or_fail(check, boba::norm_difference_inf(X, X_expected), tolerance);
  pass_or_fail(check, boba::norm_difference_inf(A * X, rhs), tolerance);
}

template <typename data_t>
void test_wide_backsolve_matrix(bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;
  constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

  std::cout << "wide backsolve, matrix rhs" << std::endl;

  auto A_host = make_host_matrix<data_t>({
    {1, 1, 1},
  });
  auto third = data_t(1) / data_t(3);
  auto two_thirds = data_t(2) / data_t(3);
  auto X_expected_host = make_host_matrix<data_t>({
    {third, two_thirds},
    {third, two_thirds},
    {third, two_thirds},
  });
  auto rhs_host = A_host * X_expected_host;

  const boba::Matrix<space, data_t> A = A_host;
  const boba::Matrix<space, data_t> rhs = rhs_host;
  const boba::Matrix<space, data_t> X_expected = X_expected_host;

  auto X = boba::backsolve(A, rhs);

  pass_or_fail(check, boba::norm_difference_inf(X, X_expected), tolerance);
  pass_or_fail(check, boba::norm_difference_inf(A * X, rhs), tolerance);
}

template <typename data_t>
void test_right_backsolve_matrix(bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;
  constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

  std::cout << "right backsolve, matrix rhs" << std::endl;

  auto A_host = make_host_matrix<data_t>({
    {1, 0},
    {0, 1},
    {0, 0},
  });
  auto X_expected_host = make_host_matrix<data_t>({
    {1, 2, 0},
    {4, 5, 0},
  });
  auto rhs_host = X_expected_host * A_host;

  const boba::Matrix<space, data_t> A = A_host;
  const boba::Matrix<space, data_t> rhs = rhs_host;
  const boba::Matrix<space, data_t> X_expected = X_expected_host;

  auto X = boba::right_backsolve(A, rhs);

  pass_or_fail(check, boba::norm_difference_inf(X, X_expected), tolerance);
  pass_or_fail(check, boba::norm_difference_inf(X * A, rhs), tolerance);
}
} // namespace

int main(int argc, char* argv[])
{
  boba::splash();
  boba::init();
  std::cout << "Tests for backsolve and right_backsolve" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  checkpoint();

  ::boba::detail::ignore(argc);
  ::boba::detail::ignore(argv);

  bool check = true;

  test_square_backsolve_matrix<double>(check);
  test_square_backsolve_vector<double>(check);
  test_tall_backsolve_matrix<double>(check);
  test_wide_backsolve_matrix<double>(check);
  test_right_backsolve_matrix<double>(check);

  boba::finalize();
  return final_check(check);
}

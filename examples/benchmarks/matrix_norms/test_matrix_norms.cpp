// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../../tests/common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

namespace
{

constexpr boba::execution_space space = boba::default_execution_space;

void check_norm(bool& check, const char* name, double actual, double expected)
{
  std::cout << name << ": actual=" << actual << " expected=" << expected << std::endl;
  pass_or_fail(check, actual - expected, 1.0e-12 * std::max(1.0, std::abs(expected)));
}

template <typename norm_t>
void measure_norm(const char* name, norm_t norm, size_t rows, size_t cols, size_t repetitions)
{
  volatile double checksum = 0.0;
  boba::detail::device_sync();
  const auto start = std::chrono::steady_clock::now();
  for (size_t repetition = 0; repetition < repetitions; ++repetition)
  {
    checksum = checksum + norm();
  }
  boba::detail::device_sync();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double mean_us = std::chrono::duration<double, std::micro>(elapsed).count() /
                         static_cast<double>(repetitions);
  std::cout << "timing backend=" << boba::default_execution_space_name()
            << " rows=" << rows << " cols=" << cols << " norm=" << name
            << " mean_us=" << std::fixed << std::setprecision(3) << mean_us
            << " repetitions=" << repetitions << " checksum=" << checksum << std::endl;
}

} // namespace

int main(int argc, char* argv[])
{
  boba::init();

  size_t rows = 128;
  size_t cols = 128;
  size_t repetitions = 100;
  bool timing = false;

  boba::argparser args(argc, argv);
  args.add_optional_argument(rows, "-r", "--rows", "Number of matrix rows.");
  args.add_optional_argument(cols, "-c", "--cols", "Number of matrix columns.");
  args.add_optional_argument(timing, "-t", "--timing", "Measure mean wall time per norm call.");
  args.add_optional_argument(repetitions, "-n", "--repetitions", "Number of timed calls per norm.");
  args.parse_check();

  if (rows == 0 || cols == 0 || rows > std::numeric_limits<boba::index_t>::max() / cols ||
      (timing && repetitions == 0))
  {
    std::cerr << "Rows and columns must be positive, their product must fit in index_t, "
              << "and timed repetitions must be positive." << std::endl;
    boba::finalize();
    return 1;
  }

  boba::Matrix<boba::host_space, double> host_matrix({rows, cols});
  auto host_view = host_matrix.view();
  double expected_sum_squares = 0.0;
  std::vector<double> row_sums(rows, 0.0);
  std::vector<double> column_sums(cols, 0.0);
  for (boba::index_t row = 0; row < rows; ++row)
  {
    for (boba::index_t col = 0; col < cols; ++col)
    {
      const boba::index_t i = row * cols + col;
      const double magnitude = static_cast<double>((i % 17) + 1);
      host_view({row, col}) = (i % 2 == 0) ? magnitude : -magnitude;
      expected_sum_squares += magnitude * magnitude;
      row_sums[row] += magnitude;
      column_sums[col] += magnitude;
    }
  }
  const double expected_frobenius = std::sqrt(expected_sum_squares);
  const double expected_matrix_one = *std::max_element(column_sums.begin(), column_sums.end());
  const double expected_matrix_inf = *std::max_element(row_sums.begin(), row_sums.end());

  boba::Matrix<space, double> matrix(host_matrix);
  std::cout << "Matrix norms: rows=" << rows << " cols=" << cols << std::endl;
  bool check = true;
  const double host_frobenius = boba::norm_frobenius(host_matrix);
  const double host_frobenius_naive = boba::norm_frobenius<true>(host_matrix);
  const double device_frobenius = boba::norm_frobenius(matrix);
  const double device_frobenius_naive = boba::norm_frobenius<true>(matrix);
  check_norm(check, "host norm_frobenius", host_frobenius, expected_frobenius);
  check_norm(check, "host norm_frobenius force_naive", host_frobenius_naive, expected_frobenius);
  check_norm(check, "norm_frobenius host default vs force_naive", host_frobenius, host_frobenius_naive);
  check_norm(check, "device norm_frobenius", device_frobenius, expected_frobenius);
  check_norm(check, "device norm_frobenius force_naive", device_frobenius_naive, expected_frobenius);
  check_norm(check, "norm_frobenius default vs force_naive", device_frobenius, device_frobenius_naive);
  check_norm(check, "norm_frobenius device vs host", device_frobenius, host_frobenius);

  const double host_matrix_one = host_matrix.matrix_one_norm();
  const double host_matrix_one_naive = host_matrix.matrix_one_norm<true>();
  const double device_matrix_one = matrix.matrix_one_norm();
  const double device_matrix_one_naive = matrix.matrix_one_norm<true>();
  check_norm(check, "host matrix_one_norm", host_matrix_one, expected_matrix_one);
  check_norm(check, "host matrix_one_norm force_naive", host_matrix_one_naive, expected_matrix_one);
  check_norm(check, "matrix_one_norm host default vs force_naive", host_matrix_one, host_matrix_one_naive);
  check_norm(check, "device matrix_one_norm", device_matrix_one, expected_matrix_one);
  check_norm(check, "device matrix_one_norm force_naive", device_matrix_one_naive, expected_matrix_one);
  check_norm(check, "matrix_one_norm default vs force_naive", device_matrix_one, device_matrix_one_naive);
  check_norm(check, "matrix_one_norm device vs host", device_matrix_one, host_matrix_one);

  const double host_matrix_inf = host_matrix.matrix_inf_norm();
  const double host_matrix_inf_naive = host_matrix.matrix_inf_norm<true>();
  const double device_matrix_inf = matrix.matrix_inf_norm();
  const double device_matrix_inf_naive = matrix.matrix_inf_norm<true>();
  check_norm(check, "host matrix_inf_norm", host_matrix_inf, expected_matrix_inf);
  check_norm(check, "host matrix_inf_norm force_naive", host_matrix_inf_naive, expected_matrix_inf);
  check_norm(check, "matrix_inf_norm host default vs force_naive", host_matrix_inf, host_matrix_inf_naive);
  check_norm(check, "device matrix_inf_norm", device_matrix_inf, expected_matrix_inf);
  check_norm(check, "device matrix_inf_norm force_naive", device_matrix_inf_naive, expected_matrix_inf);
  check_norm(check, "matrix_inf_norm default vs force_naive", device_matrix_inf, device_matrix_inf_naive);
  check_norm(check, "matrix_inf_norm device vs host", device_matrix_inf, host_matrix_inf);

  if (timing && check)
  {
    measure_norm("frobenius", [&matrix]()
    {
      return boba::norm_frobenius(matrix);
    },
                 rows,
                 cols,
                 repetitions);
    measure_norm("frobenius_force_naive", [&matrix]()
    {
      return boba::norm_frobenius<true>(matrix);
    },
                 rows,
                 cols,
                 repetitions);
    measure_norm("matrix_one", [&matrix]()
    {
      return matrix.matrix_one_norm();
    },
                 rows,
                 cols,
                 repetitions);
    measure_norm("matrix_one_force_naive", [&matrix]()
    {
      return matrix.matrix_one_norm<true>();
    },
                 rows,
                 cols,
                 repetitions);
    measure_norm("matrix_inf", [&matrix]()
    {
      return matrix.matrix_inf_norm();
    },
                 rows,
                 cols,
                 repetitions);
    measure_norm("matrix_inf_force_naive", [&matrix]()
    {
      return matrix.matrix_inf_norm<true>();
    },
                 rows,
                 cols,
                 repetitions);
  }

  boba::finalize();
  return final_check(check);
}

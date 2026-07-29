// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"
#include "common_ttm.hpp"

/*
  Test BoBa's Krylov solvers
*/

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::execution_space host_space = ::boba::host_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::seconds;

int main()
{
  boba::splash();
  boba::init();
  boba_print("Test for NNLS");

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;

  size_t dimension = 0;

  // Test 1: simple exact case
  // A = [1 0; 0 1; 1 1], b = [1; 2; 3]
  // Unconstrained LS solution: x = [1; 2], which is >= 0 and fits exactly.
  boba::Matrix<host_space, double> A({3, 2});
  A.fill_with_zeros();
  A({0, 0}) = 1.0;
  A({1, 1}) = 1.0;
  A({2, 0}) = 1.0;
  A({2, 1}) = 1.0;

  boba::Vector<host_space, double> b({3});
  b({0}) = 1.0;
  b({1}) = 2.0;
  b({2}) = 3.0;

  auto x = boba::nnls(A, b);
  auto r = A * x - b;

  std::cout << "Test 1: simple exact case" << std::endl;
  std::cout << "x =\n"
            << x << std::endl;
  std::cout << "Residual norm ||A x - b|| = " << ::boba::norm_frobenius(r) << std::endl;

  // Expected: x ~= [1, 2], residual ~= 0

  // Basic KKT-style checks
  auto w = A.transpose() * (b - A * x);

  std::cout << "w = A^T (b - A x) =\n"
            << w << std::endl;

  double tol = 1e-8;

  bool nonneg_ok = x.min_reduce() >= -tol;
  bool comp_slack_ok = true;
  for (size_t i = 0; i < x.size(); ++i)
  {
    if (x({i}) <= tol && w({i}) > tol)
    {
      comp_slack_ok = false;
    }
  }

  std::cout << "Nonnegativity satisfied: "
            << (nonneg_ok ? "yes" : "no") << std::endl;
  std::cout << "Complementarity / dual feasibility approx satisfied: "
            << (comp_slack_ok ? "yes" : "no") << std::endl;

  // Test 2: random overdetermined system with nonnegative ground truth
  {
    std::cout << "\nTest 2: random system with known nonnegative solution"
              << std::endl;
    size_t rows = 50;
    size_t cols = 10;
    boba::Matrix<host_space, double> A2({rows, cols});
    A2.fill_with_random();

    boba::Vector<host_space, double> x_true({cols});
    x_true.fill_with_random();

    // Enforce nonnegativity on true solution
    for (size_t i = 0; i < cols; ++i)
    {
      if (x_true({i}) < 0)
      {
        x_true({i}) = -x_true({i});
      }
    }

    auto b2 = A2 * x_true;

    auto x_hat = boba::nnls(A2, b2);
    auto r2 = A2 * x_hat - b2;

    std::cout << "||x_hat - x_true|| = "
              << ::boba::norm_frobenius(x_hat - x_true) << std::endl;
    std::cout << "||A x_hat - b|| = " << ::boba::norm_frobenius(r2) << std::endl;
    std::cout << "All x_hat >= 0: "
              << ((x_hat.min_reduce() >= -tol) ? "yes" : "no")
              << std::endl;
  }

  boba::finalize();
  return final_check(check);
}

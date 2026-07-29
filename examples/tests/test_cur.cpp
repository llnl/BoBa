// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "common.hpp"

/*
  Tests BoBa's implementation of the CUR algorithm
*/

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::execution_space host_space = ::boba::host_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

template <typename data_t>
void setup_test_matrix(
  size_t which,
  boba::Matrix<space, data_t>& test_matrix)
{
  auto test_matrix_view = test_matrix.view();

  switch (which)
  {
  case 0: // Hilbert
  {
    ::boba::loop<space, 2>(test_matrix_view.sizes(),
                           [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
    {
      auto [r, c] = ij;
      test_matrix_view({r, c}) = data_t(1) / (data_t(1) + data_t(r) + data_t(c));
    });
    test_matrix.rename("hilbert");
    return;
  }
  case 1: // Low-rank
  {
    ::boba::loop<space, 2>(test_matrix_view.sizes(),
                           [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
    {
      auto [r, c] = ij;
      size_t M = test_matrix_view.sizes(0);
      size_t N = test_matrix_view.sizes(1);
      data_t pi = ::boba::pi;
      data_t u = ::boba::sin(pi * data_t(r) / data_t(M));
      data_t v = ::boba::cos(pi * data_t(c) / data_t(N));
      test_matrix_view({r, c}) = u * v;
    });
    test_matrix.rename("low_rank");
    return;
  }
  case 2: // Sum matrix (i + j)
  {
    ::boba::loop<space, 2>(test_matrix_view.sizes(),
                           [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
    {
      auto [r, c] = ij;
      test_matrix_view({r, c}) = data_t(r + c);
    });
    test_matrix.rename("sum");
    return;
  }
  default:
    boba_error("Invalid matrix choice");
  }
}

int main(int argc, char* argv[])
{

  boba::detail::ignore(argc);
  boba::detail::ignore(argv);
  boba::splash();
  boba::init();
  std::cout << "Testing CUR-DEIM decomposition" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;

  size_t M = 20;
  size_t N = 15;
  size_t kickrank = 1;

  boba::Array<size_t, 2> sizes{M, N};

  // Test 3 different matrix types
  for (size_t matrix_type = 0; matrix_type < 3; matrix_type++)
  {
    std::cout << "\n=== Test " << (matrix_type + 1) << " ===" << std::endl;

    // Create test matrix
    boba::Matrix<space, double> original_matrix(sizes);
    setup_test_matrix(matrix_type, original_matrix);

    std::cout << "Matrix: " << original_matrix.name()
              << " (" << M << "x" << N << ")" << std::endl;
    std::cout << "Kickrank: " << kickrank << std::endl;

    auto original_norm = ::boba::norm_frobenius(original_matrix);

    // Apply CUR decomposition
    std::cout << "\n--- CUR-DEIM Decomposition ---" << std::endl;
    {
      boba::Matrix<space, double> test_matrix = original_matrix;

      ::boba::CUR<space, double> cur;
      cur.kick_rank = kickrank;
      cur(test_matrix);

      auto CUR_matrix = cur.reform_matrix();

      auto cur_error = boba::norm_difference_frobenius(CUR_matrix, original_matrix) / original_norm;
      pass_or_fail(check, cur_error, 1.0e-6);
    }
  }

  boba::finalize();
  return final_check(check);
}

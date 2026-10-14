// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Tests for atomics and other abstractions.

  Atomics are operations where only one parallel thread can write to the data
  at a given time, which prevents race conditions.
*/

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  std::cout << "Tests for atomics and other abstractions " << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  checkpoint();
  bool check = true;

  ::boba::argparser args(argc, argv);

  size_t N = 20;

  args.add_optional_argument(N,
                             "-n",
                             "--resolution",
                             "Number of elements.");

  args.parse_check();

  // Repeat these tests with atomics

  {
    std::cout << " sum_{i=0}^{N} i " << std::endl;

    ::boba::Vector<space, size_t> summation_exec({1_z});
    summation_exec.fill_with_zeros();
    auto summation_view = summation_exec.view();

    ::boba::loop<space, 1>(N + 1,
                           [=] __boba_host_device__(size_t i)
    {
      ::boba::atomics::reference<space>(summation_view(0)) += i;
    });

    size_t summation = summation_exec.sum_reduce();
    size_t exact = (N * (N + 1)) / 2;
    boba_print(summation);
    boba_print(exact);
    pass_or_fail(check, exact - summation, 1.0e-13);
    std::cout << std::endl;
  }

  {
    std::cout << " cancel sum_{i=0}^{N} i " << std::endl;

    ::boba::Vector<space, size_t> cancellation_exec({1_z});
    cancellation_exec.fill_with((N * (N + 1)) / 2);
    auto cancellation_view = cancellation_exec.view();

    ::boba::loop<space, 1>(N + 1,
                           [=] __boba_host_device__(size_t i)
    {
      ::boba::atomics::reference<space>(cancellation_view(0)) -= i;
    });

    size_t cancellation = cancellation_exec.sum_reduce();
    const size_t exact = 0;
    boba_print(cancellation);
    boba_print(exact);
    pass_or_fail(check, exact - cancellation, 1.0e-13);
    std::cout << std::endl;
  }

  {
    std::cout << " assign N " << std::endl;

    const size_t exact = N;
    ::boba::Vector<space, size_t> assigned_exec({1_z});
    assigned_exec.fill_with_zeros();
    auto assigned_view = assigned_exec.view();

    ::boba::loop<space, 1>(N + 1,
                           [=] __boba_host_device__(size_t i)
    {
      ::boba::detail::ignore(i);
      ::boba::atomics::reference<space>(assigned_view(0)) = exact;
    });

    size_t assigned = assigned_exec.sum_reduce();
    boba_print(assigned);
    boba_print(exact);
    pass_or_fail(check, exact - assigned, 1.0e-13);
    std::cout << std::endl;
  }

  {
    std::cout << " sum_{i=0}^{N} 2*i^2  " << std::endl;

    ::boba::Vector<space, size_t> summation_exec({1});
    summation_exec.fill_with_zeros();
    auto summation_atomic_view = summation_exec.atomic_view();

    ::boba::loop<space, 1>(N + 1,
                           [=] __boba_host_device__(size_t i)
    {
      summation_atomic_view(0) += i * i;
    });

    summation_exec *= 2.0;

    size_t summation = summation_exec.sum_reduce();
    size_t exact = 2 * (N * (N + 1) * (2 * N + 1)) / 6;
    boba_print(summation);
    boba_print(exact);
    pass_or_fail(check, exact - summation, 1.0e-13);
    std::cout << std::endl;
  }

  {
    std::cout << " cancel sum_{i=0}^{N} i^2  " << std::endl;

    ::boba::Vector<space, size_t> cancellation_exec({1});
    cancellation_exec.fill_with((N * (N + 1) * (2 * N + 1)) / 6);
    auto cancellation_atomic_view = cancellation_exec.atomic_view();

    ::boba::loop<space, 1>(N + 1,
                           [=] __boba_host_device__(size_t i)
    {
      cancellation_atomic_view(0) -= i * i;
    });

    size_t cancellation = cancellation_exec.sum_reduce();
    const size_t exact = 0;
    boba_print(cancellation);
    boba_print(exact);
    pass_or_fail(check, exact - cancellation, 1.0e-13);
    std::cout << std::endl;
  }

  {
    std::cout << " assign N^2  " << std::endl;

    const size_t exact = N * N;
    ::boba::Vector<space, size_t> assigned_exec({1});
    assigned_exec.fill_with_zeros();
    auto assigned_atomic_view = assigned_exec.atomic_view();

    ::boba::loop<space, 1>(N + 1,
                           [=] __boba_host_device__(size_t i)
    {
      ::boba::detail::ignore(i);
      assigned_atomic_view(0) = exact;
    });

    size_t assigned = assigned_exec.sum_reduce();
    boba_print(assigned);
    boba_print(exact);
    pass_or_fail(check, exact - assigned, 1.0e-13);
    std::cout << std::endl;
  }

  {
    std::cout << " y = Ax, with and without atomics " << std::endl;

    ::boba::Matrix<space, double> A({N, N});
    A.fill_with(1.0);
    auto A_view = A.const_view();

    ::boba::Vector<space, double> x({N});
    ::boba::Vector<space, double> y({N});
    ::boba::Vector<space, double> y_atomic({N});
    x.fill_with(1.0);
    y.fill_with(-1.0);
    y_atomic.fill_with_zeros();
    auto x_view = x.const_view();
    auto y_view = y.view();
    auto y_atomic_view = y_atomic.atomic_view();

    ::boba::TicToc<tictoc_units> timer;

    ::boba::loop<space, 1>(N,
                           [=] __boba_host_device__(size_t row)
    {
      double row_sum = 0.;
      for (size_t col = 0; col < N; col++)
      {
        row_sum += A_view({row, col}) * x_view(col);
      }
      y_view(row) = row_sum;
    });

    timer.end_and_print("non-atomic matvec");
    timer.tic();

    ::boba::loop<space, 2>({N, N},
                           [=] __boba_host_device__(::boba::Array<size_t, 2> rc)
    {
      size_t row = rc[0];
      size_t col = rc[1];
      auto value = A_view({row, col}) * x_view(col);
      y_atomic_view(row) += value;
    });

    timer.end_and_print("atomic matvec");

    pass_or_fail(check, boba::norm_difference_inf(y, y_atomic), 1e-12);

    std::cout << std::endl;
  }

  {
    std::cout << " " << std::endl;

    ::boba::Tensor<2, space, double> test_error({N, N});
    auto test_view = test_error.view();

    ::boba::loop<space, 2>(test_error.sizes(),
                           [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
    {
      test_view(ij) = ij[0] + ij[1];
    });

    auto estimate = ::boba::tensor_interpolation_error_estimator(test_error);

    double value = std::get<0>(estimate);
    size_t loc = std::get<1>(estimate);

    boba_print(value);
    boba_print(loc);
    pass_or_fail(check, value - 2, 1.0e-13);

    std::cout << std::endl;
  }

  // Apply function
  {
    // You can apply functions to tensors using this nifty helper function
    ::boba::Tensor<3, space, double> example({9, 9, 9});
    // Fill the tensor with random values
    example.fill_with_random();

    auto cos_example = boba::apply_function(example, [] __boba_host_device__(double x)
    {
      return boba::cos(x);
    });
    auto acos_cos_example = boba::apply_function(cos_example, [] __boba_host_device__(double x)
    {
      return boba::acos(x);
    });

    // In this example, we should get back the original example tensor
    auto error = ::boba::norm_difference_frobenius(example, acos_cos_example);

    pass_or_fail(check, error, 1e-09);
  }

  // Math functions
  {
    pass_or_fail(check, ::boba::factorial(0) - 1, 1e-09);
    pass_or_fail(check, ::boba::factorial(1) - 1, 1e-09);
    pass_or_fail(check, ::boba::factorial(2) - 2, 1e-09);
    pass_or_fail(check, ::boba::factorial(3) - 6, 1e-09);
    pass_or_fail(check, ::boba::factorial(4) - 24, 1e-09);
    pass_or_fail(check, ::boba::factorial(5) - 120, 1e-09);

    pass_or_fail(check, ::boba::double_factorial(0) - 1, 1e-09);
    pass_or_fail(check, ::boba::double_factorial(1) - 1, 1e-09);
    pass_or_fail(check, ::boba::double_factorial(2) - 2, 1e-09);
    pass_or_fail(check, ::boba::double_factorial(3) - 3, 1e-09);
    pass_or_fail(check, ::boba::double_factorial(4) - 8, 1e-09);
    pass_or_fail(check, ::boba::double_factorial(5) - 15, 1e-09);
  }

  boba::finalize();
  return final_check(check);
}

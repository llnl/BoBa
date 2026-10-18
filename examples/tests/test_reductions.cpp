// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <fstream>
#include <iomanip>

/*
  Tests for reductions.

  Reductions are operations where you aggregate a sequence of numbers into a single number.
  For example, summations are a form of reductions.
*/

constexpr boba::execution_space space = boba::default_execution_space;

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  std::cout << "Tests for reductions " << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  checkpoint();
  bool check = true;

  ::boba::argparser args(argc, argv);

  size_t N = 20;
  std::string timing_filename = "reduction_timings.csv";

  args.add_optional_argument(N,
                             "-n",
                             "--resolution",
                             "Number of elements.");
  args.add_optional_argument(timing_filename,
                             "-t",
                             "--timing-file",
                             "CSV file to which manual reduction timings are appended.");

  args.parse_check();

  std::ofstream timing_file(timing_filename, std::ios::app);
  if (!timing_file)
  {
    std::cerr << "Unable to open timing file: " << timing_filename << std::endl;
    boba::finalize();
    return 1;
  }
  if (timing_file.tellp() == std::streampos(0))
  {
    timing_file << "N,device_ms,host_ms,raja_ms\n";
  }

  {
    std::cout << " sum_{i=0}^{N} i " << std::endl;
    size_t summation = 0;
    ::boba::sum_reduce<space>(summation, 0_z, N + 1, [=] __boba_host_device__(size_t i, boba::sum_reducer_operator<size_t>& _summation)
    {
      _summation += i;
    });

    size_t exact = (N * (N + 1)) / 2;
    boba_print(summation);
    boba_print(exact);
    pass_or_fail(check, exact - summation, 1.0e-13);
    std::cout << std::endl;
  }

  {
    std::cout << " sum_{i=0}^{N} i^2  " << std::endl;
    size_t summation = 0;
    ::boba::sum_reduce<space>(summation, 0_z, N + 1, [=] __boba_host_device__(size_t i, boba::sum_reducer_operator<size_t>& _summation)
    {
      _summation += i * i;
    });

    size_t exact = (N * (N + 1) * (2 * N + 1)) / 6;
    boba_print(summation);
    boba_print(exact);
    pass_or_fail(check, exact - summation, 1.0e-13);
    std::cout << std::endl;
  }

  {
    std::cout << " min_{i=0}^{N} 7 + (3-i)^2  " << std::endl;
    double minimum = ::boba::highest_value<double>();
    size_t index = 0;

    ::boba::min_loc_reduce<space>(
      minimum, index, 0_z, 20_z, [=] __boba_host_device__(::boba::reducer_index_t i, ::boba::min_loc_reducer_operator<double> & local_minloc)
    {
      local_minloc.minloc(7.0 + boba::pow(3.0 - i, 2.0), i);
    });

    boba_print(minimum);
    boba_print(index);
    pass_or_fail(check, minimum - 7.0, 1.0e-13);
    pass_or_fail(check, index - 3, 1.0e-13);
    std::cout << std::endl;
  }

  {
    std::cout << " Check log2(2^p) = p " << std::endl;

    constexpr size_t largest_r = 1000;
    constexpr size_t largest_i = ::boba::log2(largest_r);

    double maximum = ::boba::lowest_value<double>();
    size_t index = 0;

    ::boba::max_loc_reduce<space>(
      maximum, index, 0_z, largest_i, [=] __boba_host_device__(::boba::reducer_index_t i, ::boba::max_loc_reducer_operator<double> & local_maxloc)
    {
      double di = static_cast<double>(i);
      double r = ::boba::pow(2.0, di);
      double ii = ::boba::log2(r);
      double error_loc = ::boba::abs(di - ii);
      local_maxloc.maxloc(error_loc, i);
    });

    boba_print(maximum);
    boba_print(index);
    pass_or_fail(check, maximum, 1.0e-13);

    std::cout << std::endl;
  }

  {
    std::cout << " sum_{i=0}^{N} i using manual device reductions" << std::endl;
    boba::Vector<boba::host_space, double> reduction_vector({N});
    reduction_vector.fill_with_random();
    boba::Vector<boba::default_execution_space, double> device_reduction_vector(reduction_vector);
    auto vector_view = device_reduction_vector.const_view();

    boba::TicToc<boba::tictoc_units::microseconds> timer;
    timer.tic();
    auto device_result = boba::manual_sum_reduce(device_reduction_vector);
    const double device_time_ms = static_cast<double>(timer.toc()) / 1000.0;

    timer.tic();
    auto host_result = boba::manual_sum_reduce(reduction_vector);
    const double host_time_ms = static_cast<double>(timer.toc()) / 1000.0;

    double raja_result = 0.0;
    timer.tic();
    ::boba::sum_reduce<space>(raja_result, 0_z, vector_view.size(), [=] __boba_host_device__(size_t i, boba::sum_reducer_operator<double>& _summation)
    {
      _summation += vector_view(i);
    });
    const double raja_time_ms = static_cast<double>(timer.toc()) / 1000.0;

    timing_file << N << ','
                << std::setprecision(17) << device_time_ms << ','
                << host_time_ms << ','
                << raja_time_ms << '\n';

    pass_or_fail(check, device_result - host_result, 1.0e-10);
    pass_or_fail(check, device_result - raja_result, 1.0e-10);
    pass_or_fail(check, raja_result - host_result, 1.0e-10);
    std::cout << std::endl;
  }


  boba::finalize();
  return final_check(check);
}

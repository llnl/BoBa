// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::microseconds;

/*
  Tests the general capabilities of Quantized tensor trains
*/

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  std::cout << "Tests for quantized tensor train implementation" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  bool check = 1;

  checkpoint();

  constexpr size_t exponent = boba::is_boba_debug_mode() ? 3 : 12;
  constexpr size_t base = 2;

  size_t N = ::pow(base, exponent);
  size_t qtt_base = 2;
  double svd_tolerance = 1.0e-07;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(N,
                             "-n",
                             "--resolution",
                             "Number of elements in each dimension.");

  args.add_optional_argument(qtt_base,
                             "-b",
                             "--base",
                             "Base for quantized tensor train.");

  args.add_optional_argument(svd_tolerance,
                             "-t",
                             "--tolerance",
                             "Tolerance for SVD truncation.");

  args.parse_check();

  //
  // Create data
  //
  checkpoint();
  ::boba::Vector<space, double> independent_variable({N});
  independent_variable.rename("independent_variable");
  auto x_view = independent_variable.view();

  ::boba::Vector<space, double> dependent_variable({N});
  dependent_variable.rename("dependent_variable");
  auto f_view = dependent_variable.view();

  checkpoint();
  ::boba::loop<space, 1>(N,
                         [=] __boba_host_device__(size_t i)
  {
    x_view(i) = i;
    f_view(i) = 1.0 / (1.0 + i);
  });

  if (N <= 8)
  {
    independent_variable.print();
    dependent_variable.print();
  }

  ::boba::TicToc<tictoc_units> timer;

  //
  // Create QTTs
  //
  auto x_qtt = ::boba::compress_to_QuantizedTensorTrain(independent_variable, qtt_base, svd_tolerance);
  auto f_qtt = ::boba::compress_to_QuantizedTensorTrain(dependent_variable, qtt_base, svd_tolerance);

  timer.end_and_print("Compress QTT");

  //
  // Report diagnostics
  //
  std::cout << "Ranks, x : " << x_qtt.ranks_string() << std::endl;
  std::cout << "Ranks, f : " << f_qtt.ranks_string() << std::endl;
  if (N <= 8)
  {
    x_qtt.print();
    x_qtt.decompress().print();

    f_qtt.print();
    f_qtt.decompress().print();
  }

  size_t string_length = 25;
  auto x_cr = x_qtt.compression_rate();
  auto f_cr = f_qtt.compression_rate();

  std::cout
    << fill_up_string_end("Compression Rate, x QTT : ", string_length)
    << fill_up_string_end(boba::to_string(x_cr), string_length) << std::endl;

  std::cout
    << fill_up_string_end("Compression Rate, f QTT : ", string_length)
    << fill_up_string_end(boba::to_string(f_cr), string_length) << std::endl;

  //
  // Extract Rank One test
  //
  checkpoint();
  {
    std::cout << "\n Running Extract Rank One Tests \n"
              << std::endl;

    checkpoint();
    auto x_qtt_rank1 = x_qtt.extract_rank_one_TensorTrain();
    auto f_qtt_rank1 = f_qtt.extract_rank_one_TensorTrain();
    auto c_qtt = x_qtt_rank1;
    c_qtt.add_subtrain(x_qtt);
    auto C = c_qtt.extract_rank_one_TensorTrain();

    auto d_qtt = f_qtt_rank1;
    d_qtt.add_subtrain(f_qtt);
    auto D = d_qtt.extract_rank_one_TensorTrain();

    pass_or_fail(check, ::boba::norm_difference_inf(C.decompress(), x_qtt_rank1.decompress()), 1.0e-12);
    pass_or_fail(check, ::boba::norm_difference_inf(D.decompress(), f_qtt_rank1.decompress()), 1.0e-12);
  }

  boba::finalize();
  return final_check(check);
}

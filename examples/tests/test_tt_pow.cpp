// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Tests powers of tensor trains and compares against naive Hadamard product of tt with itself
*/

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::microseconds;

static double error_tolerance = 4.0e-10;

template <size_t power, size_t dimension>
void test_tensor_power(
  ::boba::TensorTrain<dimension, space, double>& tt_to_raise,
  bool& check,
  bool round_btw_slow_powers = false)
{
  ::boba::detail::device_sync();

  auto slow_power = tt_to_raise;

  checkpoint();
  boba::TicToc<tictoc_units> timer;

  timer.tic();
  // Raise the TT to the desired power the slow way, by repeatedly doing elementwise products
  for (size_t p = 1; p < power; p++)
  {
    auto temp = elementwise_product(tt_to_raise, slow_power);
    slow_power = temp;
    if (round_btw_slow_powers)
    {
      slow_power.round();
    }
  }
  if (round_btw_slow_powers)
  {
    timer.end_and_print("slow power total time");
  }
  else
  {
    timer.end_and_print("slow power ");
  }

  boba_print("Size data for slow power of TT");
  boba_print(slow_power.sizes());
  boba_print(slow_power.get_number_elements());
  boba_print(slow_power.ranks_string());

  timer.tic();
  auto fast_power = elementwise_power<power>(tt_to_raise, false);
  timer.end_and_print("fast power ");
  boba_print("Size data for fast power of TT");
  boba_print(fast_power.sizes());
  boba_print(fast_power.get_number_elements());
  boba_print(fast_power.ranks_string());

  checkpoint();

  pass_or_fail(check, ::boba::norm_difference_inf(slow_power.decompress(), fast_power.decompress()), error_tolerance);

  checkpoint();

  if (!round_btw_slow_powers)
  {
    timer.tic();
    slow_power.round();
    timer.end_and_print("compression of slow power");
  }

  checkpoint();

  timer.tic();
  fast_power.round();
  timer.end_and_print("compression of fast power");

  boba_print("Post-compression size data: ");
  boba_print(slow_power.ranks_string());
  boba_print(fast_power.ranks_string());

  pass_or_fail(check, ::boba::norm_difference_inf(slow_power.decompress(), fast_power.decompress()), error_tolerance);

  // Do higher powers the iterative way
  timer.tic();
  auto fast_power_it = elementwise_power<power>(tt_to_raise, true);
  timer.end_and_print("Iterative fast power");
  pass_or_fail(check, ::boba::norm_difference_inf(slow_power.decompress(), fast_power_it.decompress()), error_tolerance);
}

/*
  Constructs a tensor train representing the function f(x) = \sum_{k=1}^{ranks} ( \prod_{d=0}^{last_participating_dimension-1} sin(2*pi*k*x_d) )
  The size of the tensor in dimension d will be N_min + d*N_increment
*/
template <size_t dimension>
::boba::TensorTrain<dimension, space, double> make_sinusoidal_tt(const size_t N_min, const size_t N_increment, const size_t ranks, const size_t last_participating_dimension = dimension)
{
  ::boba::Array<size_t, dimension> sizes;
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] = N_min + d * N_increment; // Make sure tensor sizes are different in every dimension
  }
  ::boba::Tensor<dimension, space, double> my_tensor(sizes);

  double twopi = 2.0 * boba::pi;

  auto my_tensor_view = my_tensor.view();
  ::boba::loop<space, 1>(my_tensor_view.size(),
                         [=] __boba_host_device__(size_t I)
  {
    auto mid = my_tensor_view.multiindex(I);
    double val = 0.;
    double var = 0.;
    for (size_t r = 1; r <= ranks; r++)
    {
      double val_tmp = 1.;
      for (size_t d = 0; d < last_participating_dimension; d++)
      {
        var = static_cast<double>(mid[d]) / static_cast<double>(sizes[d]);
        val_tmp *= sin(twopi * r * var);
      }
      val += val_tmp;
    }
    my_tensor_view(mid) = val;
  });

  ::boba::TensorTrain<dimension, space, double> my_tt(my_tensor.sizes());

  my_tt.compress(my_tensor);

  return my_tt;
}

/***********************************************************************************/

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  std::cout << "Tests for boba computation of tensor train powers" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  checkpoint();

  ::boba::argparser args(argc, argv);

  size_t N = 0;
  size_t r = 5;
  size_t compress_btw_pwrs = 0;
  size_t do_asym_rank_test = 1;

  args.add_optional_argument(N,
                             "-N",
                             "--resolution",
                             "Number of elements in first dimension.");
  args.add_optional_argument(r,
                             "-r",
                             "--ranks",
                             "Number of ranks in sinusoidal TT testing.");
  args.add_optional_argument(compress_btw_pwrs,
                             "-ca",
                             "--compress-always",
                             "When taking powers larger than 2 the slow way, whether to compress every time you take the product");
  args.add_optional_argument(do_asym_rank_test,
                             "-at",
                             "--asym-test",
                             "Whether to do the test of TTs with asymmetric ranks... I only turn this off when doing scaling studies");

  args.parse_check();

  bool check = 1;
  bool compress_between_slow_powers = (compress_btw_pwrs != 0);
  bool do_asymmetric_ranks_test = (do_asym_rank_test == 1);
  boba_print(compress_between_slow_powers);

  std::vector<size_t> v = {12, 14, 15};

  checkpoint();

  if (N > 0)
  {
    v.resize(0);
    v.push_back(N);
  }

  for (size_t N_test : v)
  {
    auto sizes = ::boba::filled_array<3>(10_z);
    sizes[0] = N_test;
    ::boba::TensorTrain<3, space, double> tt_ones(sizes);
    tt_ones.fill_with(1.0);

    test_tensor_power<2>(tt_ones, check);

    boba_print(" ");
    boba_print("Starting with sinusoidal TT test");
    boba_print(" ");

    if (do_asymmetric_ranks_test)
    {
      boba_print("Asymmetric rank TTs:");
      // Constructs a TT with in which all ranks are distinct
      auto tt_asym_3 = make_sinusoidal_tt<3>(N_test, 1, r, 1);
      auto ttB = make_sinusoidal_tt<3>(N_test, 1, r, 2);
      auto ttC = make_sinusoidal_tt<3>(N_test, 1, r, 3);
      tt_asym_3.TensorTrain_add(ttB);
      tt_asym_3.TensorTrain_add(ttC);
      tt_asym_3.round();
      // For default parameters, the ranks string is (1, 5, 6, 1)
      boba_print(tt_asym_3.ranks_string());

      // Tests powers of that TT
      boba_print("Square:");
      test_tensor_power<2>(tt_asym_3, check, compress_between_slow_powers);
      boba_print(" ");
      boba_print("Cube:");
      test_tensor_power<3>(tt_asym_3, check, compress_between_slow_powers);
      boba_print(" ");
      boba_print("Fourth Power:");
      test_tensor_power<4>(tt_asym_3, check, compress_between_slow_powers);
    }

    boba_print(" ");
    boba_print("Three-dimensional tests");
    boba_print(" ");
    ::boba::TensorTrain<3, space, double> tt_sine3 = make_sinusoidal_tt<3>(N_test, 1, r);

    boba_print("Square:");
    test_tensor_power<2>(tt_sine3, check);
    boba_print(" ");
    boba_print("Cube:");
    test_tensor_power<3>(tt_sine3, check, compress_between_slow_powers);
    boba_print(" ");
    boba_print("Fourth power:");
    test_tensor_power<4>(tt_sine3, check, compress_between_slow_powers);

    boba_print(" ");
    boba_print("Four-dimensional test");
    ::boba::TensorTrain<4, space, double> tt_sine4 = make_sinusoidal_tt<4>(N_test, 1, r);
    boba_print("Square:");
    test_tensor_power<2>(tt_sine4, check);
    boba_print(" ");
    boba_print("Cube:");
    test_tensor_power<3>(tt_sine4, check, compress_between_slow_powers);
    boba_print(" ");
    boba_print("Fourth power:");
    test_tensor_power<4>(tt_sine4, check, compress_between_slow_powers);
  }

  boba::finalize();

  return final_check(check);
}

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Tests some basic tensor train operations and demonstrates a few important concepts, including:

  - how to create a tensor and compress it into a tensor train.
  - how to create tensor cores and then use those cores to form tensor trains.
  - how to create an empty tensor train and resize it to directly define the cores.
*/

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

static double error_tolerance = 1.0e-10;

//
// Individual compression tests
//
template <size_t dimension>
::boba::TensorTrain<dimension, space, double> test_tensor_compression(
  ::boba::Tensor<dimension, space, double>& tensor,
  bool& check)
{
  checkpoint();
  ::boba::detail::device_sync();

  checkpoint();
  boba::TicToc<tictoc_units> timer;

  boba_print(tensor.name());

  timer.tic();
  auto test_train = ::boba::compress_to_TensorTrain(tensor);
  timer.end_and_print("compress " + tensor.name());

  bool old_check = check;
  pass_or_fail(check, ::boba::norm_difference_inf(test_train.decompress(), tensor), error_tolerance);
  bool new_fail = not(check) and old_check;

  test_train.rename("test_train");
  if (new_fail and (test_train.get_full_size() < 100))
  {
    test_train.print();
  }

  timer.tic();
  auto A_decompressed = test_train.decompress();
  timer.end_and_print("unroll " + tensor.name());

  if (new_fail and (A_decompressed.size() < 30))
  {
    A_decompressed.print();
  }

  pass_or_fail(check, ::boba::norm_difference_inf(A_decompressed, tensor), error_tolerance);

  timer.tic();
  auto start = ::boba::filled_array<dimension>(0_z);
  auto end = tensor.sizes();
  auto A_decompressed_subtensor = test_train.unroll_subtensor(start, end);
  timer.end_and_print("unroll subtensor " + tensor.name());

  pass_or_fail(check, ::boba::norm_difference_inf(A_decompressed_subtensor, tensor), error_tolerance);

  //
  // Perform rounding and check error again
  //
  timer.tic();
  test_train.round();
  timer.end_and_print("round " + tensor.name());

  pass_or_fail(check, ::boba::norm_difference_inf(test_train.decompress(), tensor), error_tolerance);

  pass_or_fail(check, ::boba::norm_difference_inf(A_decompressed, tensor), error_tolerance);

  pass_or_fail(check, ::boba::norm_difference_inf(A_decompressed_subtensor, tensor), error_tolerance);

  return test_train;
}

//
// Series of tt tests
//
template <size_t dimension>
void tt_tests(size_t N, bool& check)
{
  std::cout << "N = " << N << ", dimension = " << dimension << std::endl;

  auto sizes = ::boba::filled_array<dimension>(N);

  // Make sizes all different to catch bugs hidden w/ identical sizes
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] += d;
  }

  //
  // A = ones
  //
  checkpoint();
  boba::Tensor<dimension, space, double> A(sizes);
  A.rename("ones");
  A.fill_with(1.0);

  checkpoint();
  auto tt_A = test_tensor_compression(A, check);

  //
  // f(x, y) = x
  //
  checkpoint();
  boba::Tensor<dimension, space, double> B(sizes);
  B.rename("x");
  auto B_view = B.view();

  for (size_t dim = 0_z; dim < dimension; dim++)
  {
    ::boba::Multiindexer<dimension> mider(A.sizes());

    ::boba::loop<space, 1>(A.size(),
                           [=] __boba_host_device__(size_t i)
    {
      auto ijk = mider.multiindex(i);
      B_view(ijk) = ijk[dim];
    });

    auto tt_B = test_tensor_compression(B, check);
  }

  auto tt_B = test_tensor_compression(B, check);

  //
  // Zeros
  //
  checkpoint();
  boba::Tensor<dimension, space, double> C(sizes);
  C.rename("zeros");
  C.fill_with_zeros();

  auto tt_C = test_tensor_compression(C, check);

  //
  // f(x, y) = x
  //
  checkpoint();
  boba::TensorTrain<dimension, space, double> tt_E(sizes);
  size_t R = 2;

  for (size_t dim = 0_z; dim < dimension; dim++)
  {
    auto rleft = (dim == 0) ? 1 : R;
    auto rright = (dim == dimension - 1) ? 1 : R;

    tt_E.cores[dim].resize({rleft, N, rright});
    auto E_view = tt_E.cores[dim].view();

    ::boba::loop<space, 3>(tt_E.cores[dim].sizes(),
                           [=] __boba_host_device__(::boba::Array<size_t, 3> ijk)
    {
      double value = ijk[1] + ::boba::mod(dim, 2) * ijk[0] + ijk[2];
      E_view(ijk) = value / double(1 + 2 * R + N);
    });
  }

  auto E = tt_E.decompress();
  E.rename("hilbert-esque");

  auto tt_E_2 = test_tensor_compression(E, check);

  pass_or_fail(check, ::boba::norm_difference_inf(tt_E.decompress(), E), error_tolerance);

  pass_or_fail(check, ::boba::norm_difference_inf(tt_E.decompress(), tt_E_2.decompress()), error_tolerance);

  //
  // C = A + C
  //
  checkpoint();
  boba::TicToc<tictoc_units> timer;

  tt_C += tt_A;
  timer.end_and_print("C += A");

  pass_or_fail(check, ::boba::norm_difference_inf(tt_A.decompress(), tt_C.decompress()), error_tolerance);

  //
  // D = A + B
  //
  checkpoint();
  boba::Tensor<dimension, space, double> D(sizes);
  D.rename("1+x");
  D = A + B;

  auto tt_D = test_tensor_compression(D, check);

  timer.tic();
  boba::TensorTrain<dimension, space, double> tt_D_2(sizes);
  tt_D_2 = tt_A + tt_B;
  timer.end_and_print("D = A + B");

  pass_or_fail(check, ::boba::norm_difference_inf(tt_D.decompress(), tt_D_2.decompress()), error_tolerance);

  //
  // F = B * D
  //
  checkpoint();
  // Set F equal to elementwise product of F and D
  auto F = boba::elementwise_product(B, D);
  F.rename("x*(1+x)");

  auto tt_F = test_tensor_compression(F, check);

  timer.tic();
  boba::TensorTrain<dimension, space, double> tt_F_2(sizes);
  tt_F_2 = boba::elementwise_product(tt_B, tt_D);
  timer.end_and_print("F = B * D");
  pass_or_fail(check, ::boba::norm_difference_inf(tt_F.decompress(), tt_F_2.decompress()), error_tolerance);

  //
  // test inject_dimension and partial decompress
  //
  checkpoint();
  {
    auto tt_D_plus_1 = tt_D.insert_dimension(1);
    auto tt_D_plus_1_minus_1 = tt_D_plus_1.partial_decompress(1);

    pass_or_fail(check, ::boba::norm_difference_inf(tt_D_plus_1_minus_1.decompress(), tt_D.decompress()), error_tolerance);
  }

  //
  // Test tolerance finder
  //
  // Tight error
  auto [tolerance_1, optimal_1] = ::boba::find_optimal_tt_tolerance(E, 0.01, 0.1);

  // Loose error
  auto [tolerance_2, optimal_2] = ::boba::find_optimal_tt_tolerance(E, 0.0001, 0.001);

  boba_always_assert_ge(
    tolerance_1,
    tolerance_2,
    "Unexpected optimal tolerances.");
  boba_always_assert_ge(
    optimal_1.compression_rate(),
    optimal_2.compression_rate(),
    "Unexpected results - higher tolerances should lead to higher error.");
}

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  std::cout << "Tests for boba tensor train implementation" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  checkpoint();

  ::boba::argparser args(argc, argv);

  size_t N = 0;

  args.add_optional_argument(N,
                             "-n",
                             "--resolution",
                             "Number of elements in each dimension.");

  args.parse_check();

  bool check = 1;

  checkpoint();

  std::vector<size_t> v = {5, 10, 15};

  if (N > 0)
  {
    v.resize(1);
    v[0] = N;
  }

  for (auto N_test : v)
  {
    boba_print(N_test);
    tt_tests<1>(N_test, check);
    tt_tests<2>(N_test, check);
    tt_tests<3>(N_test, check);
    tt_tests<4>(N_test, check);
    tt_tests<5>(N_test, check);
  }

  std::cout << "Testing tensor products " << std::endl;
  {
    ::boba::Vector<space, double> vector_A({7});
    ::boba::Vector<space, double> vector_B({10});

    vector_B.set_to_basis({1});

    vector_A.fill_with(2.0);

    auto full_vector = ::boba::tensor_product(vector_B, vector_A);

    ::boba::Tensor<2, space, double> outer_product({vector_B.size(), vector_A.size()});
    outer_product.reshape(full_vector);

    ::boba::TensorTrain<2, space, double> tt({vector_B.size(), vector_A.size()});
    tt.compress(outer_product);

    auto tt_from_vectors = ::boba::make_tt_from_vectors<2, ::boba::default_execution_space, double>({vector_B, vector_A});
    pass_or_fail(check, ::boba::norm_difference_frobenius(tt, tt_from_vectors), 1.0e-09);

    auto tt_decompress = tt.decompress();
    boba_print(tt.ranks_string());
    boba_print(tt.compression_rate());
    pass_or_fail(check, ::boba::norm_difference_frobenius(tt_decompress, outer_product), 1.0e-09);

    tt.compress(tt_decompress);
    pass_or_fail(check, ::boba::norm_difference_frobenius(tt.decompress(), outer_product), 1.0e-09);

    auto tt_decompress_2 = tt.decompress();
    pass_or_fail(check, ::boba::norm_difference_frobenius(tt_decompress_2, outer_product), 1.0e-09);
  }

  std::cout << "Testing static tt view " << std::endl;
  checkpoint();
  {
    constexpr size_t static_N = 10;
    constexpr size_t R1 = 2;
    constexpr size_t R2 = 3;

    auto _sizes = ::boba::filled_array<3>(static_N);
    boba::TensorTrain<3, space, double> statically_sized_tt(_sizes);
    statically_sized_tt.rename("statically_sized_tt");

    statically_sized_tt.cores[0].resize({1, static_N, R1});
    statically_sized_tt.cores[1].resize({R1, static_N, R2});
    statically_sized_tt.cores[2].resize({R2, static_N, 1});

    auto A3_a_view = statically_sized_tt.cores[0].view();
    auto A3_b_view = statically_sized_tt.cores[1].view();
    auto A3_c_view = statically_sized_tt.cores[2].view();

    ::boba::loop<space, 2>({static_N, R1},
                           [=] __boba_host_device__(::boba::Array<size_t, 2> ijk)
    {
      auto [i, j] = ijk;
      A3_a_view({0, i, j}) = double(i + j) / double(1 + R1 + static_N);
    });

    ::boba::loop<space, 3>({R1, static_N, R2},
                           [=] __boba_host_device__(::boba::Array<size_t, 3> ijk)
    {
      auto [i, j, k] = ijk;
      A3_b_view({i, j, k}) = (double(j) + double(2 * i) - double(k)) / double(1 + R1 + R2 + static_N);
    });

    ::boba::loop<space, 2>({R2, static_N},
                           [=] __boba_host_device__(::boba::Array<size_t, 2> ijk)
    {
      auto [i, j] = ijk;
      A3_c_view({i, j, 0}) = (double(j) - double(i)) / double(1 + R2 + static_N);
    });

    using static_sizes = ::boba::StaticArray<std::size_t, static_N, static_N, static_N>;
    using static_ranks = ::boba::StaticArray<std::size_t, 1, R1, R2, 1>;
    ::boba::StaticTensorTrainView<static_sizes, static_ranks, double> tt_table_view(statically_sized_tt);

    auto statically_sized_tt_decompressed = statically_sized_tt.decompress();

    auto decompressed_view = statically_sized_tt_decompressed.const_view();

    size_t scale = 3;
    size_t test_size = scale * (static_N - 1);

    ::boba::Tensor<3, space, double> error({test_size, test_size, test_size});
    auto error_view = error.view();

    checkpoint();
    ::boba::loop<space, 3>(error.sizes(),
                           [=] __boba_host_device__(::boba::Array<size_t, 3> ijk)
    {
      size_t j = ijk[0];
      size_t i = ijk[1];
      size_t k = ijk[2];

      //
      // Test coordinate
      //
      double I = double(i) / double(scale);
      double J = double(j) / double(scale);
      double K = double(k) / double(scale);

      // Brackets
      ::boba::Array<size_t, 2> I_indices{size_t(I), size_t(I) + 1};
      ::boba::Array<size_t, 2> J_indices{size_t(J), size_t(J) + 1};
      ::boba::Array<size_t, 2> K_indices{size_t(K), size_t(K) + 1};

      ::boba::Array<double, 2> Id_indices{I, I + 1.0};
      ::boba::Array<double, 2> Jd_indices{J, J + 1.0};
      ::boba::Array<double, 2> Kd_indices{K, K + 1.0};

      // Interpolation weights
      ::boba::Array<double, 2> I_weights = ::boba::lagrange_weights(Id_indices, I);
      ::boba::Array<double, 2> J_weights = ::boba::lagrange_weights(Jd_indices, J);
      ::boba::Array<double, 2> K_weights = ::boba::lagrange_weights(Kd_indices, K);

      //
      // TT multilinear interpolation
      //
      auto tt_value = tt_table_view.interpolation<2>({I_weights, J_weights, K_weights}, {I_indices, J_indices, K_indices});

      //
      // Conventional multilinear interpolation
      //
      double conventional_value = 0.0;
      for (size_t I_i = 0; I_i < 2; I_i++)
      {
        for (size_t J_i = 0; J_i < 2; J_i++)
        {
          for (size_t K_i = 0; K_i < 2; K_i++)
          {
            double wi = I_weights[I_i];
            double wj = J_weights[J_i];
            double wk = K_weights[K_i];
            double wgts = wi * wj * wk;
            double value = decompressed_view({size_t(I) + I_i, size_t(J) + J_i, size_t(K) + K_i});
            conventional_value += wgts * value;
          }
        }
      }

      error_view(ijk) = ::boba::abs(conventional_value - tt_value);
    });

    pass_or_fail(check, error.max_abs_reduce(), 1.0e-10);
  }

  boba::finalize();
  return final_check(check);
}

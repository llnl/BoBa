// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

/*
  Tests writing various tensor objects to a file and reading them back in.
*/

template <typename tensor_sizes, typename data_t>
void initialize(::boba::Tensor<tensor_sizes{}.size(), space, data_t>& test)
{
  constexpr auto dimension = tensor_sizes{}.size();
  auto sizes = ::boba::to_array(tensor_sizes{});
  test.resize(sizes);
  test.fill_with_random();
}

template <typename tensor_sizes, typename data_t>
void initialize(::boba::TensorTrain<tensor_sizes{}.size(), space, data_t>& test_tt)
{
  constexpr auto dimension = tensor_sizes{}.size();
  static_assert(dimension > 2, "Must be 3D");
  static_assert(dimension <= 5, "Must be at most 5D");

  auto sizes = ::boba::to_array(tensor_sizes{});
  constexpr auto R1 = 5;
  constexpr auto R2 = 3;
  constexpr auto R3 = (dimension >= 4) ? 3 : 1;
  constexpr auto R4 = (dimension >= 5) ? 2 : 1;

  test_tt.cores[0].resize({1, sizes[0], R1});
  test_tt.cores[1].resize({R1, sizes[1], R2});
  test_tt.cores[2].resize({R2, sizes[2], (dimension <= 3) ? 1 : R3});
  if constexpr (dimension >= 4)
  {
    test_tt.cores[3].resize({R3, sizes[3], (dimension <= 4) ? 1 : R4});
  }
  if constexpr (dimension >= 5)
  {
    test_tt.cores[4].resize({R4, sizes[4], 1});
  }

  //
  // Fill cores
  //
  for (size_t d = 0; d < dimension; d++)
  {
    test_tt.cores[d].fill_with_random();
  }
}

template <typename tensor_sizes, typename data_t>
void initialize(::boba::TensorTrainMatrix<tensor_sizes{}.size(), space, data_t>& test)
{
  checkpoint();
  constexpr auto dimension = tensor_sizes{}.size();
  static_assert(dimension > 2, "Must be 3D");
  static_assert(dimension <= 5, "Must be at most 5D");

  auto rows = ::boba::to_array(tensor_sizes{});
  auto cols = rows + 1_z;
  constexpr auto R1 = 5;
  constexpr auto R2 = 3;
  constexpr auto R3 = (dimension >= 4) ? 3 : 1;
  constexpr auto R4 = (dimension >= 5) ? 2 : 1;

  test.cores[0].resize({1, rows[0], cols[0], R1});
  test.cores[1].resize({R1, rows[1], cols[1], R2});
  test.cores[2].resize({R2, rows[2], cols[2], (dimension <= 3) ? 1 : R3});
  if constexpr (dimension >= 4)
  {
    test.cores[3].resize({R3, rows[3], cols[3], (dimension <= 4) ? 1 : R4});
  }
  if constexpr (dimension >= 5)
  {
    test.cores[4].resize({R4, rows[4], cols[4], 1});
  }

  //
  // Fill cores
  //
  checkpoint();
  for (size_t d = 0; d < dimension; d++)
  {
    test.cores[d].fill_with_random();
  }
}

template <typename tensor_sizes, typename data_t>
void initialize(::boba::Tucker<tensor_sizes{}.size(), space, data_t>& test_Tucker)
{
  constexpr auto dimension = tensor_sizes{}.size();
  static_assert(dimension > 2, "Must be 3D");
  static_assert(dimension <= 5, "Must be at most 5D");

  auto sizes = ::boba::to_array(tensor_sizes{});
  auto R_core_resize = sizes;
  constexpr auto R0 = 3;
  constexpr auto R1 = 5;
  constexpr auto R2 = 3;
  constexpr auto R3 = (dimension >= 4) ? 3 : 1;
  constexpr auto R4 = (dimension >= 5) ? 2 : 1;

  R_core_resize[0] = R0;
  R_core_resize[1] = R1;
  R_core_resize[2] = R2;
  test_Tucker.cores[0].resize({sizes[0], R0});
  test_Tucker.cores[1].resize({sizes[1], R1});
  test_Tucker.cores[2].resize({sizes[2], R2});
  if constexpr (dimension >= 4)
  {
    test_Tucker.cores[3].resize({sizes[3], R3});
    R_core_resize[3] = R3;
  }
  if constexpr (dimension >= 5)
  {
    test_Tucker.cores[4].resize({sizes[4], R4});
    R_core_resize[4] = R4;
  }

  test_Tucker.R_core.resize(R_core_resize);
  test_Tucker.R_core.fill_with_random();

  //
  // Fill cores
  //
  for (size_t d = 0; d < dimension; d++)
  {
    test_Tucker.cores[d].fill_with_random();
  }
}

template <typename tensor_sizes, typename data_t>
void initialize(::boba::CanonicalPolyadicDecomposition<tensor_sizes{}.size(), space, data_t>& test_CPD)
{
  constexpr auto dimension = tensor_sizes{}.size();
  static_assert(dimension > 2, "Must be 3D");
  static_assert(dimension <= 5, "Must be at most 5D");

  auto sizes = ::boba::to_array(tensor_sizes{});
  constexpr auto R0 = 3;
  test_CPD.m_cores[0].resize({sizes[0], R0});
  test_CPD.m_cores[1].resize({sizes[1], R0});
  test_CPD.m_cores[2].resize({sizes[2], R0});
  if constexpr (dimension >= 4)
  {
    test_CPD.m_cores[3].resize({sizes[3], R0});
  }
  if constexpr (dimension >= 5)
  {
    test_CPD.m_cores[4].resize({sizes[4], R0});
  }

  test_CPD.m_weights.resize({R0});
  test_CPD.m_weights.fill_with_random();

  //
  // Fill cores
  //
  for (size_t d = 0; d < dimension; d++)
  {
    test_CPD.m_cores[d].fill_with_random();
  }
}

template <typename tensor_sizes, typename data_t>
void initialize(::boba::BlockVector<boba::TensorTrain<tensor_sizes{}.size(), space, data_t>>& test)
{
  checkpoint();
  for (size_t row = 0; row < test.block_size; row++)
  {
    initialize<tensor_sizes>(test({row}));
  }
}

template <typename tensor_sizes, typename data_t>
void initialize(::boba::BlockOperator<boba::TensorTrainMatrix<tensor_sizes{}.size(), space, data_t>>& test)
{
  checkpoint();
  for (size_t row = 0; row < test.block_rows; row++)
  {
    for (size_t col = 0; col < test.block_cols; col++)
    {
      initialize<tensor_sizes>(test({row, col}));
    }
  }
}

template <typename tensor_sizes, typename data_t>
void initialize(::boba::BlockVector<boba::Tensor<tensor_sizes{}.size(), space, data_t>>& test)
{
  checkpoint();
  for (size_t row = 0; row < test.block_size; row++)
  {
    initialize<tensor_sizes>(test({row}));
  }
}

template <typename tensor_sizes, typename data_t>
void initialize(::boba::QuantizedTensorTrain<space, data_t>& test_qtt)
{
  constexpr auto dimension = tensor_sizes{}.size();
  static_assert(dimension > 2, "Must be 3D");
  static_assert(dimension <= 5, "Must be at most 5D");

  auto sizes = ::boba::to_array(tensor_sizes{});
  auto base = sizes[0];
  constexpr auto R1 = 5;
  constexpr auto R2 = 3;
  constexpr auto R3 = (dimension >= 4) ? 3 : 1;
  constexpr auto R4 = (dimension >= 5) ? 2 : 1;
  test_qtt.base = base;
  test_qtt.exponent = dimension;
  test_qtt.cores[0].resize({1, base, R1});
  test_qtt.cores[1].resize({R1, base, R2});
  test_qtt.cores[2].resize({R2, base, (dimension <= 3) ? 1 : R3});
  if constexpr (dimension >= 4)
  {
    test_qtt.cores[3].resize({R3, base, (dimension <= 4) ? 1 : R4});
  }
  if constexpr (dimension >= 5)
  {
    test_qtt.cores[4].resize({R4, base, 1});
  }

  //
  // Fill cores
  //
  for (size_t d = 0; d < dimension; d++)
  {
    test_qtt.cores[d].fill_with_random();
  }
}

template <typename tensor_sizes, typename data_t>
void initialize(::boba::HierarchicalTucker<tensor_sizes{}.size(), space, data_t>& test_HTucker)
{
  constexpr auto dimension = tensor_sizes{}.size();
  static_assert(dimension > 2, "Must be 3D");
  static_assert(dimension <= 5, "Must be at most 5D");

  test_HTucker.fill_with_random();
}

//
// Test runner
//

template <typename tensor_sizes, typename data_t>
struct run
{
  using real_data_t = boba::real_type_t<data_t>;

  run(bool& check, const std::string& type_id, real_data_t sampling_tolerance, size_t which_test)
  {
    go(check, type_id, sampling_tolerance, which_test);
  }

  void go(bool& check, const std::string& type_id, real_data_t sampling_tolerance, size_t which_test)
  {
    //
    // Create data and write to file
    //
    static constexpr size_t dimension = tensor_sizes{}.size();

    checkpoint();
    std::string exe_string = ::boba::name_flag();
    std::string prefix = "file_";
    std::string suffix = type_id + "_" + std::to_string(dimension) + "d" + exe_string;

    std::cout << "dimension = " << dimension << " data_t = " << type_id << std::endl;

    if ((which_test == 0) or (which_test == 1))
    {
      boba_print("\n tensor i/o tests");
      auto sizes = ::boba::to_array(tensor_sizes{});
      boba::Tensor<dimension, space, data_t> test(sizes);
      test.rename(prefix + "tensor_" + suffix);
      initialize<tensor_sizes>(test);

      auto error_measurement = [](const auto& A, const auto& B)
      {
        return ::boba::norm_inf(A - B);
      };

      run_tests(test, error_measurement, check, sampling_tolerance);
    }

    if ((which_test == 0) or (which_test == 2))
    {
      boba_print("\n TT i/o tests");
      auto sizes = ::boba::to_array(tensor_sizes{});
      boba::TensorTrain<dimension, space, data_t> test(sizes);
      test.rename(prefix + "tt_" + suffix);
      initialize<tensor_sizes>(test);

      auto error_measurement = [](const auto& A, const auto& B)
      {
        return ::boba::norm_inf((A - B).decompress());
      };

      run_tests(test, error_measurement, check, sampling_tolerance);
    }

    if ((which_test == 0) or (which_test == 3))
    {
      boba_print("\n TTM i/o tests");
      auto rows = ::boba::to_array(tensor_sizes{});
      auto cols = rows + 1_z;
      boba::TensorTrainMatrix<dimension, space, data_t> test(rows, cols);
      test.rename(prefix + "ttm_" + suffix);
      initialize<tensor_sizes>(test);

      auto error_measurement = [](const auto& A, const auto& B)
      {
        return ::boba::norm_inf((A - B).decompress());
      };

      run_tests(test, error_measurement, check, sampling_tolerance);
    }

    if ((which_test == 0) or (which_test == 4))
    {
      boba_print("\n Tucker i/o tests");
      auto sizes = ::boba::to_array(tensor_sizes{});
      boba::Tucker<dimension, space, data_t> test(sizes);
      test.rename(prefix + "tucker_" + suffix);
      initialize<tensor_sizes>(test);
      auto error_measurement = [](const auto& A, const auto& B)
      {
        return ::boba::norm_inf((A - B).decompress());
      };

      run_tests(test, error_measurement, check, sampling_tolerance);
    }

    if ((which_test == 0) or (which_test == 5))
    {
      boba_print("\n CPD i/o tests");
      auto sizes = ::boba::to_array(tensor_sizes{});
      boba::CanonicalPolyadicDecomposition<dimension, space, double> test(sizes);
      test.rename(prefix + "cpd_" + suffix);
      initialize<tensor_sizes>(test);
      auto error_measurement = [](const auto& A, const auto& B)
      {
        return ::boba::norm_inf(A.decompress() - B.decompress());
      };

      run_tests(test, error_measurement, check, sampling_tolerance);
    }

    if ((which_test == 0) or (which_test == 6))
    {
      boba_print("\n QTT i/o tests");
      auto sizes = ::boba::to_array(tensor_sizes{});
      auto base = sizes[0];

      boba::QuantizedTensorTrain<space, data_t> test(base, dimension);
      test.rename(prefix + "qtt_" + suffix);
      initialize<tensor_sizes>(test);
      auto error_measurement = [](const auto& A, const auto& B)
      {
        return ::boba::norm_inf(A.decompress() - B.decompress());
      };

      run_tests(test, error_measurement, check, sampling_tolerance);
    }

    if ((which_test == 0) or (which_test == 7))
    {
      boba_print("\n Block vector i/o tests");
      auto sizes = ::boba::to_array(tensor_sizes{});
      constexpr size_t block_size = 2;
      ::boba::BlockVector<boba::TensorTrain<dimension, space, data_t>> test(block_size);
      test.rename(prefix + "block_vector_" + suffix);
      initialize<tensor_sizes>(test);
      auto error_measurement = [block_size](const auto& A, const auto& B)
      {
        real_data_t total_error = 0.0;
        for (size_t row = 0; row < block_size; row++)
        {
          auto error_element = ::boba::norm_inf((A({row}) - B({row})).decompress());
          total_error += error_element;
        }
        return total_error;
      };

      run_tests(test, error_measurement, check, sampling_tolerance);
    }

    if ((which_test == 0) or (which_test == 8))
    {
      boba_print("\n Block operator i/o tests");
      auto sizes = ::boba::to_array(tensor_sizes{});
      constexpr size_t block_rows = 2;
      constexpr size_t block_cols = 2;
      ::boba::BlockOperator<boba::TensorTrainMatrix<tensor_sizes{}.size(), space, data_t>> test(block_rows, block_cols);
      test.rename(prefix + "block_operator_" + suffix);
      initialize<tensor_sizes>(test);
      auto error_measurement = [block_rows, block_cols](const auto& A, const auto& B)
      {
        real_data_t total_error = 0.0;
        for (size_t row = 0; row < block_rows; row++)
        {
          for (size_t col = 0; col < block_cols; col++)
          {
            auto error_element = ::boba::norm_inf((A({row, col}) - B({row, col})).decompress());
            total_error += error_element;
          }
        }
        return total_error;
      };

      run_tests(test, error_measurement, check, sampling_tolerance);
    }

    if ((which_test == 0) or (which_test == 9))
    {
      boba_print("\n Block vector i/o tests");
      auto sizes = ::boba::to_array(tensor_sizes{});
      constexpr size_t block_size = 2;
      ::boba::BlockVector<boba::Tensor<dimension, space, data_t>> test(block_size);
      test.rename(prefix + "block_vector_" + suffix);
      initialize<tensor_sizes>(test);
      auto error_measurement = [block_size](const auto& A, const auto& B)
      {
        real_data_t total_error = 0.0;
        for (size_t row = 0; row < block_size; row++)
        {
          auto error_element = ::boba::norm_inf(A({row}) - B({row}));
          total_error += error_element;
        }
        return total_error;
      };

      run_tests(test, error_measurement, check, sampling_tolerance);
    }

    if ((which_test == 0) or (which_test == 10))
    {
      boba_print("\n HTucker i/o tests");
      auto sizes = ::boba::to_array(tensor_sizes{});
      auto dim_tree = ::boba::DimensionTree(boba::BalancedTreeBuilder(dimension));
      ::boba::HierarchicalTucker<dimension, space, data_t> test(sizes, dim_tree);
      test.set_name(prefix + "htucker_" + suffix);
      initialize<tensor_sizes>(test);
      auto error_measurement = [](const auto& A, const auto& B)
      {
        auto error = A - B;

        if constexpr (std::is_floating_point_v<std::remove_cv_t<data_t>>)
        {
          error.orthogonalize();
        }

        return ::boba::norm_inf(error.decompress());
      };

      auto htucker_tolerance = sampling_tolerance;
      // TODO<htucker> This test requires a very high tolerance to pass consistently (otherwise sometimes randomly fails)
      if constexpr ((dimension == 5) && std::is_same_v<std::remove_cv_t<data_t>, boba::complex<float>>)
      {
        htucker_tolerance = real_data_t{3.0e-2};
      }
      run_tests(test, error_measurement, check, htucker_tolerance);
    }
  }

  template <typename test_object_t>
  void run_tests(const test_object_t& test_object, auto error_measurement_function, bool& check, real_data_t sampling_tolerance)
  {
    ::boba::TicToc<tictoc_units> timer;

    auto name = test_object.name();
    checkpoint();
    {
      {
        boba_print("Testing writer...");
        timer.tic();
        boba::write_to_file(test_object);
        timer.end_and_print("write_to_file");
      }
      if constexpr (boba::is_boba_MATLAB_enabled())
      {
        boba_print("Testing MATLAB mat file writer");
        timer.tic();
        boba::write_to_mat_file(test_object);
        timer.end_and_print("write_to_file matlab");
      }
      if constexpr (boba::is_boba_hdf5_enabled())
      {
        std::cout << "Testing hdf5 writer" << std::endl;
        timer.tic();
        std::string filename = name + ".hdf5";
        boba::write_to_hdf5_file(test_object, filename);
        boba::detail::HDF5File test_file(filename, "rw");
        test_file.write_int("test_int", 4);
        timer.end_and_print("write_to_file hdf5");
      }
    }

    //
    // Read from file
    //
    {
      boba_print("Testing boba reader...");
      test_object_t test_io;
      timer.tic();
      boba::read_from_file(test_io, name);
      timer.end_and_print("read_from_file");
      timer.tic();
      auto discrepancy = error_measurement_function(test_io, test_object);
      timer.end_and_print("verify io");

      pass_or_fail(check, discrepancy, sampling_tolerance);
    }
    if constexpr (boba::is_boba_MATLAB_enabled())
    {
      boba_print("Testing MATLAB mat file read/write.");
      test_object_t test_matlab;
      timer.tic();
      boba::read_from_mat_file(test_matlab, name);
      timer.end_and_print("read_from_file matlab");
      timer.tic();
      auto discrepancy = error_measurement_function(test_matlab, test_object);
      timer.end_and_print("verify read_from_file matlab");
      pass_or_fail(check, discrepancy, sampling_tolerance);
    }
    if constexpr (boba::is_boba_hdf5_enabled())
    {
      boba_print("Testing hdf5 file read/write.");
      test_object_t test_h5;
      timer.tic();
      std::string filename = name + ".hdf5";
      boba::read_from_hdf5_file(test_h5, filename, name);
      timer.end_and_print("read_from_file hdf5");
      timer.tic();
      auto discrepancy = error_measurement_function(test_h5, test_object);
      timer.end_and_print("verify read_from_file hdf5");
      pass_or_fail(check, discrepancy, sampling_tolerance);
      boba::detail::HDF5File test_file(filename, "r");
      auto test_int = test_file.read_int("test_int");
      pass_or_fail(check, static_cast<double>(test_int - 4), 1.0e-13);
    }
  }
};

//
//
//
int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  std::cout << "Tests for tensor train io" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;

  checkpoint();

  using tensor_sizes_5d = boba::StaticArray<size_t, 3, 2, 2, 1, 2>;
  using tensor_sizes_4d = boba::StaticArray<size_t, 3, 3, 1, 2>;
  using tensor_sizes_3d = boba::StaticArray<size_t, 5, 4, 6>;

  size_t which_data_type = 0;
  size_t run_dimension = 0;
  size_t which_test = 0;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(which_data_type,
                             "-t",
                             "--datatype",
                             "Data type (1 = float, 2 = complex_float, 3 = double, 4 = complex_double, 0 = all.");

  args.add_optional_argument(run_dimension,
                             "-d",
                             "--dimension",
                             "Table dimensionality.");

  args.add_optional_argument(which_test,
                             "-w",
                             "--which",
                             "Pick which test.");

  args.parse_check();

  if ((run_dimension == 3_z) or (run_dimension == 0_z))
  {
    if ((which_data_type == 1_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_3d, float>(check, "float", 1.0e-03f, which_test);
    }

    if ((which_data_type == 2_z) or (which_data_type == 0_z))
    {
#ifdef BOBA_CPU
      run<tensor_sizes_3d, boba::complex<float>>(check, "complex_float", 1.0e-03f, which_test);
#else
      boba_error("Complex IO is not yet supported on GPU");
#endif
    }

    if ((which_data_type == 3_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_3d, double>(check, "double", 1.0e-10, which_test);
    }

    if ((which_data_type == 4_z) or (which_data_type == 0_z))
    {
#ifdef BOBA_CPU
      run<tensor_sizes_3d, boba::complex<double>>(check, "complex_double", 1.0e-10, which_test);
#else
      boba_error("Complex IO is not yet supported on GPU");
#endif
    }

    if ((which_data_type == 5_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_3d, size_t>(check, "size_t", 1, which_test);
    }

    if ((which_data_type == 6_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_3d, int>(check, "int", 1, which_test);
    }
  }

  if ((run_dimension == 4_z) or (run_dimension == 0_z))
  {
    if ((which_data_type == 1_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_4d, float>(check, "float", 1.0e-03f, which_test);
    }

    if ((which_data_type == 2_z) or (which_data_type == 0_z))
    {
#ifdef BOBA_CPU
      run<tensor_sizes_4d, boba::complex<float>>(check, "complex_float", 1.0e-03f, which_test);
#else
      boba_error("Complex IO is not yet supported on GPU");
#endif
    }

    if ((which_data_type == 3_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_4d, double>(check, "double", 1.0e-10, which_test);
    }

    if ((which_data_type == 4_z) or (which_data_type == 0_z))
    {
#ifdef BOBA_CPU
      run<tensor_sizes_4d, boba::complex<double>>(check, "complex_double", 1.0e-10, which_test);
#else
      boba_error("Complex IO is not yet supported on GPU");
#endif
    }

    if ((which_data_type == 5_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_4d, size_t>(check, "size_t", 1, which_test);
    }

    if ((which_data_type == 6_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_4d, int>(check, "int", 1, which_test);
    }
  }

  if ((run_dimension == 5_z) or (run_dimension == 0_z))
  {
    if ((which_data_type == 1_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_5d, float>(check, "float", 1.0e-03f, which_test);
    }

    if ((which_data_type == 2_z) or (which_data_type == 0_z))
    {
#ifdef BOBA_CPU
      run<tensor_sizes_5d, boba::complex<float>>(check, "complex_float", 1.0e-03f, which_test);
#else
      boba_error("Complex IO is not yet supported on GPU");
#endif
    }

    if ((which_data_type == 3_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_5d, double>(check, "double", 1.0e-10, which_test);
    }

    if ((which_data_type == 4_z) or (which_data_type == 0_z))
    {
#ifdef BOBA_CPU
      run<tensor_sizes_5d, boba::complex<double>>(check, "complex_double", 1.0e-10, which_test);
#else
      boba_error("Complex IO is not yet supported on GPU");
#endif
    }

    if ((which_data_type == 5_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_5d, size_t>(check, "size_t", 1, which_test);
    }

    if ((which_data_type == 6_z) or (which_data_type == 0_z))
    {
      run<tensor_sizes_5d, int>(check, "int", 1, which_test);
    }
  }

  return final_check(check);
}
